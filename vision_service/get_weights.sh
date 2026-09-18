#!/usr/bin/env bash
# 把画面向量模型（SigLIP）的权重下到本地目录，几个国内源挨个试，下好自动写进 .env。
#
#   ./vision_service/get_weights.sh          下（已经有了就跳过）
#   ./vision_service/get_weights.sh --force  重下
#
# ./up.sh deploy 会自动调它：权重还没在本地时先下，下好了直接跳过。
#
# 为什么要有它：机器直连 huggingface.co 卡死不报错，走 hf-mirror 有时也不通；
# 一个个源手敲很烦。这里按顺序试：
#   1. ModelScope（阿里）——实测最稳，10MB/s 级
#   2. hf-mirror.com（HF 的国内镜像；2026-09 实测它对这个模型只是 308 跳回 huggingface.co，等于没镜像）
#   3. huggingface.co 官方
# 哪个通用哪个，边下边打进度。全不通就明说，告诉人怎么从别的电脑拷过来。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_ID="${SIGLIP_MODEL_ID:-google/siglip-base-patch16-224}"
# 下到仓库里的 models/vision/ 下（跟别的模型放一起好管理；这个子目录不进 git）
DEST="${SIGLIP_LOCAL_DIR:-models/vision/$(basename "$MODEL_ID")}"
ENV_FILE="vision_service/.env"
PY_BIN="${PY_BIN:-python}"
# pip 源：几个国内镜像 + 官方测速选最快的（结果缓存一天）。手动指定就不测：
#   PIP_INDEX_URL=https://pypi.org/simple ./up.sh deploy
source "$(dirname "${BASH_SOURCE[0]}")/pick_pip_mirror.sh"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# 权重齐不齐：config + 至少一个权重文件
have_weights() {
    [ -f "$DEST/config.json" ] || return 1
    local f
    for f in "$DEST"/*.safetensors "$DEST"/*.bin; do [ -f "$f" ] && return 0; done
    return 1
}

write_env() {
    local abs
    abs="$(cd "$DEST" && pwd)"
    touch "$ENV_FILE"
    if grep -q '^EMBED_MODEL=' "$ENV_FILE"; then
        sed -i "s|^EMBED_MODEL=.*|EMBED_MODEL=$abs|" "$ENV_FILE"
    else
        echo "EMBED_MODEL=$abs" >> "$ENV_FILE"
    fi
    echo "已写入 $ENV_FILE：EMBED_MODEL=$abs"
}

if [ "$FORCE" = "0" ] && have_weights; then
    echo "权重已在本地：$DEST（要重下加 --force）"
    grep -q "^EMBED_MODEL=" "$ENV_FILE" 2>/dev/null || write_env
    exit 0
fi

[ "${DRY_RUN:-0}" = "1" ] && { echo "（DRY_RUN）会把 $MODEL_ID 下到 $DEST"; exit 0; }

mkdir -p "$DEST"
try_hf() {   # $1 endpoint 名字 $2 endpoint
    echo "▶ 试 $1（$2）"
    HF_ENDPOINT="$2" HF_HUB_ENABLE_HF_TRANSFER=0 "$PY_BIN" - "$MODEL_ID" "$DEST" <<'EOF'
import sys, time
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

model, dest = sys.argv[1], sys.argv[2]
state = {"done": 0, "total": 0, "t0": time.monotonic()}

class T(tqdm):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        state["total"] += int(self.total or 0)
    def update(self, n=1):
        state["done"] += int(n or 0)
        el = max(1e-3, time.monotonic() - state["t0"])
        sp = state["done"] / el
        eta = (state["total"] - state["done"]) / sp if sp > 0 else 0
        pct = state["done"] / state["total"] * 100 if state["total"] else 0
        bar = int(pct // 5)
        sys.stdout.write("\r  [%s%s] %5.1f%%  %.0f/%.0f MB  %.1f MB/s  剩 %d 分 %02d 秒   "
                         % ("#" * bar, "." * (20 - bar), pct, state["done"] / 1e6, state["total"] / 1e6, sp / 1e6, eta // 60, eta % 60))
        sys.stdout.flush()
        return super().update(n)

try:
    # 先只拉 config，通不通几秒内就知道，别在一个不通的源上等半天
    from huggingface_hub import hf_hub_download
    hf_hub_download(model, "config.json", local_dir=dest, etag_timeout=15)
    snapshot_download(model, local_dir=dest, tqdm_class=T, etag_timeout=15)
    print()
except Exception as e:
    print("\n  ✗ %s: %s" % (type(e).__name__, str(e)[:200]))
    sys.exit(1)
EOF
}

try_modelscope() {
    echo "▶ 试 ModelScope（阿里）"
    "$PY_BIN" -c "import modelscope" 2>/dev/null || "$PY_BIN" -m pip install -q modelscope >/dev/null 2>&1 || { echo "  ✗ 装不上 modelscope"; return 1; }
    "$PY_BIN" - "$MODEL_ID" "$DEST" <<'EOF'
import sys
from modelscope import snapshot_download
model, dest = sys.argv[1], sys.argv[2]
try:
    # 只要 safetensors 那一份：仓库里 .bin 和 .safetensors 各 800MB，内容一样，下一份就够
    snapshot_download(model, local_dir=dest, ignore_file_pattern=[r".*\.bin$", r".*\.h5$", r".*\.msgpack$"])
except Exception as e:
    print("  ✗ %s: %s" % (type(e).__name__, str(e)[:200]))
    sys.exit(1)
EOF
}

"$PY_BIN" -c "import huggingface_hub, tqdm" 2>/dev/null || "$PY_BIN" -m pip install -q huggingface_hub tqdm >/dev/null 2>&1

for attempt in "modelscope" "hf-mirror.com|https://hf-mirror.com" "huggingface.co|https://huggingface.co"; do
    if [ "$attempt" = "modelscope" ]; then
        try_modelscope && have_weights && break
    else
        try_hf "${attempt%%|*}" "${attempt##*|}" && have_weights && break
    fi
done

if have_weights; then
    echo "✓ 权重下好了：$DEST"
    write_env
    exit 0
fi

echo
echo "✗ 三个源都不通。这台机器出不了网的话，在能上网的电脑上下好再拷过来："
echo "    pip install -U huggingface_hub"
echo "    hf download $MODEL_ID --local-dir $(basename "$MODEL_ID")     # 老版本是 huggingface-cli download"
echo "    scp -r $(basename "$MODEL_ID") $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):$(pwd)/$DEST"
echo "  拷完再跑一次这个脚本（或 ./up.sh deploy），它会发现已经有了、自动写 .env"
exit 1
