#!/bin/bash
# GR5513 小模型：训完 → 导入 label_service 训练任务 → 平台接口导出到端侧（train<号>）→ 平台训练记录（停用）
set -o pipefail
LOG=~/claude_exp/export_w8/small_pipeline.log
exec >> "$LOG" 2>&1
say() { echo "$(date '+%F %T') $*"; }
fail() { say "停：$1"; exit 1; }
ST=~/claude_exp/export_w8/small_status.txt
while [ "$(grep -c '训练结束' $ST 2>/dev/null)" -lt 2 ]; do sleep 30; done
grep -q "rc=[1-9]" $ST && fail "训练有失败：$(cat $ST)"
say "两个小模型训完"
WT=~/claude_exp/imu_pages; IT=~/imu_train
mkdir -p $IT/tmp/import_logs
python3 - <<'EOF' || exit 1
import json, os, shutil
WT = os.path.expanduser("~/claude_exp/imu_pages"); IT = os.path.expanduser("~/imu_train")
LOGS = os.path.expanduser("~/claude_exp/export_w8")
EXTRA = [{"date": "vehkaoja2022", "export_json": "public_datasets/vehkaoja2022/export.json", "source_hz": 100.0, "exclude_classes": []},
         {"date": "marcato2023", "export_json": "public_datasets/marcato2023/export.json", "source_hz": 100.0, "exclude_classes": []}]
items = []
for ax in (6, 3):
    tag = f"cnn_w8s_{ax}ax_pub"
    run = f"pool_20261116_all__{tag}"
    for top in ("results", "data"):
        src, dst = os.path.join(WT, top, f"processed_{run}"), os.path.join(IT, top, f"processed_{run}")
        if not os.path.exists(dst):
            shutil.copytree(src, dst, symlinks=True)
    spec = json.load(open(os.path.join(WT, "experiments/claude_2026_11/specs", f"train_v2_{tag}.json"), encoding="utf-8"))
    log = os.path.join(IT, "tmp", "import_logs", f"{tag}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"▶ 宿主机 GPU 上训练（同一份 src/ml/train_v2.py，spec 见 experiments/claude_2026_11/specs/train_v2_{tag}.json）\n"
                "  GR5513 小模型：通道减半 [32,64,64]，其余跟 train36 / train37 一样\n"
                "  数据：数据池全部 + 公开数据集 Vehkaoja 2022、Marcato 2023\n\n")
        f.write(open(os.path.join(LOGS, f"train_{tag}.log"), encoding="utf-8").read())
    items.append({"tag": tag, "run": run, "model_type": "cnn", "secs": 900,
                  "model_path": f"/app/results/processed_{run}/20hz_v2/cnn/dl_cnn_best.pt",
                  "log": f"/app/tmp/import_logs/{tag}.log",
                  "command": f"python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_{tag}.json",
                  "spec": {"axes": ax, "window_s": 8.0, "stride_s": 1.0, "edge_size": True,
                           "model_params": spec["model_params"], "extra_datasets": EXTRA}})
json.dump(items, open(os.path.join(IT, "tmp", "import_items_small.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("items", [i["tag"] for i in items])
EOF
cp $WT/experiments/claude_2026_11/specs/train_v2_cnn_w8s_*ax_pub.json $IT/experiments/claude_2026_11/specs/ 2>/dev/null
LS=label_service-label-service-1
BEFORE=$(ls $IT/label_service/jobs/*.json | sed 's/.*\///;s/\.json//' | sort -n | tail -1)
docker exec -w /app $LS python /app/experiments/claude_2026_11/import_jobs.py /app/tmp/import_items_small.json || fail "import_jobs"
NEW_IDS=$(ls $IT/label_service/jobs/*.json | sed 's/.*\///;s/\.json//' | sort -n | awk -v b=$BEFORE '$1>b')
say "新任务号：$NEW_IDS"
[ -n "$NEW_IDS" ] || fail "没看到新任务"
for id in $NEW_IDS; do
  say "导出到端侧 #$id"
  curl -s -m 1900 -X POST http://127.0.0.1:8383/api/v1/label/train/$id/export_edge > ~/claude_exp/export_w8/export_job$id.json
  python3 -c "
import json; r=json.load(open('$HOME/claude_exp/export_w8/export_job$id.json'))
print('  ', r.get('tag'), r.get('spec'), 'reloaded', r.get('reloaded'), r.get('reload_error'), (r.get('footprint') or {}).get('summary') if isinstance(r.get('footprint'), dict) else '')
" || { cat ~/claude_exp/export_w8/export_job$id.json; fail "导出 #$id"; }
done
python3 - $NEW_IDS <<'EOF' || exit 1
import json, os, sys
IT = os.path.expanduser("~/imu_train")
items = json.load(open(os.path.join(IT, "tmp", "import_items_small.json"), encoding="utf-8"))
out = []
for jid in sys.argv[1:]:
    job = json.load(open(os.path.join(IT, "label_service", "jobs", f"{jid}.json"), encoding="utf-8"))
    out.append({"job_id": int(jid), "model_type": job["model_type"], "spec": job["dataset_spec"]})
json.dump(out, open("/tmp/small_items_done.json", "w", encoding="utf-8"), ensure_ascii=False)
print("records to create:", [o["job_id"] for o in out])
EOF
docker cp $IT/experiments/claude_2026_11/mv_import.py deploy-api-1:/tmp/mv_import.py
docker cp /tmp/small_items_done.json deploy-api-1:/tmp/small_items_done.json
docker exec -w /app deploy-api-1 python /tmp/mv_import.py /tmp/small_items_done.json || fail "mv_import"
say "平台训练记录建好（停用）"
# label_service 重启后才会把导入的任务挂成服务端模型 train<号>（跟 #36~#43 一样）
Q=$(curl -s -m 5 http://127.0.0.1:8383/api/v1/label/queue)
say "队列 $Q"
for i in $(seq 1 60); do
  echo "$Q" | grep -q '"running":0,' && echo "$Q" | grep -q '"waiting":0,' && break
  sleep 30; Q=$(curl -s -m 5 http://127.0.0.1:8383/api/v1/label/queue)
done
cd $IT && bash label_service/up.sh -d > /tmp/ls_up_small.log 2>&1 || fail "label_service 重启"
for i in $(seq 1 60); do curl -fsS -m 2 -o /dev/null http://127.0.0.1:8383/health && break; sleep 2; done
curl -s -m 10 http://127.0.0.1:8383/api/v1/label/models | python3 -c "import json,sys; d=json.load(sys.stdin); ms=d if isinstance(d,list) else d.get('models',d); print('服务端模型：', [m.get('tag') for m in ms][-6:])"
curl -s -m 10 http://127.0.0.1:8900/models 2>/dev/null | head -c 600; echo
say "完成"
