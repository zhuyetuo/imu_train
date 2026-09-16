"""多模型注册表：同时挂几个模型，推理时按标签选。

    python -m pytest label_service/test_registry.py -q

盯的全是**错了不报错**的事：
  · 不认识的标签悄悄退回默认模型 —— 结果标着 acc3、内容是默认模型跑的，
    模型对比里混进一个冒名顶替的，而这件事没有任何迹象
  · 默认模型被 LABEL_MODELS 里的同名标签顶掉 —— 线上标注在用那个
  · 后处理用错模型的几何 —— 片段时间戳整体偏掉，而片段看起来完全正常
  · 通配符匹配到多个时替人挑一个 —— 挑错了不报错
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from label_service import registry as R  # noqa: E402


# ── LABEL_MODELS 的解析 ───────────────────────────────────────────────────


def test_parses_tag_equals_path():
    assert R.parse_spec("a=x.pkl,b=y.pkl") == [("a", "x.pkl"), ("b", "y.pkl")]


def test_skips_malformed_entries_instead_of_dying():
    """一条写错不该让整个服务起不来——它是线上标注在用的。"""
    got = R.parse_spec("a=x.pkl, 没有等号, =空标签, b=y.pkl, c=")
    assert got == [("a", "x.pkl"), ("b", "y.pkl")]


def test_default_tag_cannot_be_hijacked():
    """`default=...` 会把默认模型顶掉，而默认模型是线上在用的那个。"""
    assert R.parse_spec(f"{R.DEFAULT_TAG}=evil.pkl,a=x.pkl") == [("a", "x.pkl")]


def test_empty_spec_is_no_extra_models():
    assert R.parse_spec("") == []
    assert R.parse_spec(None) == []


# ── 路径解析 ──────────────────────────────────────────────────────────────


def test_glob_matching_multiple_is_refused(tmp_path, caplog):
    """匹配到多个**不替人挑**。

    挑错了不报错，只会让平台上的结果对应到另一份模型——跟 LABEL_MODEL
    那边一个规矩。
    """
    (tmp_path / "a.pkl").write_bytes(b"x")
    (tmp_path / "b.pkl").write_bytes(b"x")
    assert R._resolve(str(tmp_path / "*.pkl")) is None


def test_glob_matching_one_resolves(tmp_path):
    f = tmp_path / "ml_rf.pkl"
    f.write_bytes(b"x")
    assert R._resolve(str(tmp_path / "*.pkl")) == str(f)


def test_missing_file_resolves_to_none(tmp_path):
    assert R._resolve(str(tmp_path / "nope.pkl")) is None


# ── 注册表本身 ────────────────────────────────────────────────────────────


@pytest.fixture
def reg():
    return R.Registry()


#: 测试里拿来做身份比对的哨兵——`is` 比的必须是**传进去的那个**对象
SENTINEL_POOL = object()
SENTINEL_BUNDLE = {"classes": ["活动"], "hz": 16, "window_s": 1.0,
                   "stride_s": 0.5, "label_mode": "majority",
                   "model_path": "/m/default.pkl"}


def _fake_default(reg):
    reg.set_default("/m/default.pkl", SENTINEL_BUNDLE, SENTINEL_POOL)


def test_no_tag_means_default(reg):
    _fake_default(reg)
    assert reg.resolve_tag(None) == R.DEFAULT_TAG
    assert reg.resolve_tag("") == R.DEFAULT_TAG


def test_unknown_tag_raises_instead_of_falling_back(reg):
    """**不能退回默认**。

    退回的话，平台存的结果标着 acc3、内容却是默认模型跑的——
    模型对比里混进一个冒名顶替的，而这件事没有任何迹象。
    """
    _fake_default(reg)
    with pytest.raises(KeyError) as e:
        reg.resolve_tag("acc3")
    assert "acc3" in str(e.value)


def test_known_tag_resolves(reg, tmp_path):
    _fake_default(reg)
    f = tmp_path / "ml_rf.pkl"
    f.write_bytes(b"x")
    reg.load_extra(f"acc3={f}")
    assert reg.resolve_tag("acc3") == "acc3"
    assert reg.path_of("acc3") == str(f)


def test_missing_extra_model_is_skipped_not_fatal(reg, tmp_path):
    """实验模型没训出来时，服务照起，默认模型不受影响。"""
    _fake_default(reg)
    reg.load_extra(f"acc3={tmp_path / 'nope.pkl'}")
    assert reg.tags() == [R.DEFAULT_TAG]
    assert reg.resolve_tag(None) == R.DEFAULT_TAG


def test_extra_models_are_lazy(reg, tmp_path):
    """登记了不等于加载了。一个 pkl 50MB，还要起进程池——
    为了列个下拉就全加载一遍太贵。"""
    _fake_default(reg)
    f = tmp_path / "ml_rf.pkl"
    f.write_bytes(b"x")
    reg.load_extra(f"acc3={f}")
    by = {m["tag"]: m for m in reg.describe()}
    assert by["acc3"]["loaded"] is False
    assert by[R.DEFAULT_TAG]["loaded"] is True


def test_describe_marks_the_default(reg, tmp_path):
    _fake_default(reg)
    f = tmp_path / "ml_rf.pkl"
    f.write_bytes(b"x")
    reg.load_extra(f"acc3={f}")
    by = {m["tag"]: m for m in reg.describe()}
    assert by[R.DEFAULT_TAG]["is_default"] is True
    assert by["acc3"]["is_default"] is False


def test_default_pool_is_the_same_object_not_a_copy(reg):
    """默认模型在注册表里必须是**同一个** pool/bundle。

    另建一份的话默认模型会有两个进程池（内存翻倍），而且
    /model/switch 只换得掉其中一个——换完一半请求还跑着老模型。
    """
    _fake_default(reg)
    b, p = reg.get(R.DEFAULT_TAG)
    # 比的是**传进 set_default 的那个对象**，不是"注册表里存的那个"——
    # 后者跟自己比永远成立，第一版就是这么写的，变异测试里活下来了
    assert p is SENTINEL_POOL, "默认模型的进程池被换成了另一个"
    assert b is SENTINEL_BUNDLE, "默认模型的 bundle 被复制了一份"


# ── app.py 那边的接线 ─────────────────────────────────────────────────────


def _src(name):
    with open(os.path.join(_HERE, name), encoding="utf-8") as f:
        return f.read()


def test_postprocess_uses_the_selected_models_geometry():
    """后处理的几何必须从**被选中的那个模型**的 bundle 读。

    用默认模型的 window_s/stride_s 去给另一个模型的窗口重建片段
    **不会报错**，只会让时间戳整体偏掉——而偏掉的片段看起来完全正常。
    这里用 AST 查：_infer_in_pool 函数体里不能再直接碰模块级的 _bundle。
    """
    import ast

    tree = ast.parse(_src("app.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_infer_in_pool")
    # 允许的一处：b = bundle if bundle is not None else _bundle
    names = [x.id for x in ast.walk(fn) if isinstance(x, ast.Name)]
    assert names.count("_bundle") <= 1, \
        "_infer_in_pool 里还在直接用默认模型的 _bundle，选了别的模型时几何会用错"
    assert names.count("_pool") <= 1, \
        "_infer_in_pool 里还在直接用默认模型的 _pool"


def test_batch_resolves_the_model_once():
    """一批共用一个模型，解析一次。每条解析一次的话，懒加载那把锁
    会被抢几百次，而且日志里"首次使用"会刷屏。"""
    src = _src("app.py")
    assert "b, px = _pick_model(req.model)      # 一批共用一个模型" in src


def test_unknown_tag_is_422_not_a_silent_default():
    src = _src("app.py")
    assert "raise HTTPException(422, str(e)) from e" in src


def test_models_endpoint_exists():
    assert '@app.get("/api/v1/label/models")' in _src("app.py")


# ── 部署层面：环境变量得真的能进到容器里 ──────────────────────────────────


def test_compose_passes_label_models_through():
    """`LABEL_MODELS=... bash up.sh` 要真的生效。

    compose 的 environment 里不列出来的话，那个变量**只到了宿主机的 shell**，
    容器里看不到——服务照常起、日志一切正常，只是平台的下拉里少一组，
    人会以为是平台没刷新。这个坑踩过一次。
    """
    import yaml

    p = os.path.join(_HERE, "docker-compose.yml")
    with open(p, encoding="utf-8") as f:
        env = yaml.safe_load(f)["services"]["label-service"]["environment"]
    assert "LABEL_MODELS" in env, "compose 没透传 LABEL_MODELS，命令行给的值进不去容器"


def test_compose_does_not_pass_the_default_model_through():
    """**LABEL_MODEL（单数）不能透传。**

    它在 config.py 里有非空默认值。按 ${LABEL_MODEL:-} 透传的话，宿主机没设
    这个变量时会往容器里塞一个空字符串，把默认模型路径清掉——
    服务直接起不来，而报错是"通配符没有匹配到任何文件"，
    跟"compose 多列了一行"看不出关系。
    """
    import yaml

    p = os.path.join(_HERE, "docker-compose.yml")
    with open(p, encoding="utf-8") as f:
        env = yaml.safe_load(f)["services"]["label-service"]["environment"]
    assert "LABEL_MODEL" not in env


def test_empty_label_models_is_the_old_behavior():
    """compose 默认透传的是空字符串，那必须等于"没有额外模型"。

    要是空字符串被当成一条配置去解析，启动日志里会多出一堆
    "格式不对，跳过"的告警，而那是每次都会出现的噪声。
    """
    assert R.parse_spec("") == []


def test_up_sh_has_a_recreate_without_rebuild_option():
    """up.sh 要有"不重建镜像、但重新创建容器"这一档。

    少了这一档的话，改 LABEL_MODELS 只剩两个选择：
      · -d（restart）—— 读不到新环境变量，**不生效而且不报错**
      · 默认（--build）—— 冷缓存的机器上要十几分钟装 torch 和一堆轮子

    第一次给部署命令时就是漏了这一档，让人白等了十几分钟，
    而且那次 LABEL_MODELS 还没透传，等完也没生效。
    """
    with open(os.path.join(_HERE, "up.sh"), encoding="utf-8") as f:
        src = f.read()
    assert "-u|--up)" in src, "up.sh 没有 -u（不重建镜像，只重新创建容器）"
    # 那一档**不能**带 --build，带了就跟默认那档一样慢了。
    #
    # **只看真正的命令行，把注释剥掉**——那一段的注释里正好写着"不 --build"，
    # 扫原文会被自己的说明绊住。这类错（源码扫描撞上自己的注释）
    # 在这个项目里犯过四次了。
    body = src[src.index("-u|--up)"):]
    body = body[:body.index("exit 0")]
    code = [ln.split("#", 1)[0] for ln in body.splitlines()]
    assert "--build" not in "\n".join(code), "-u 那档带了 --build，就跟默认一样慢了"
    assert any("up -d" in ln for ln in code), "-u 那档没有真的跑 docker compose up -d"


def test_restart_warns_that_env_vars_do_not_take_effect():
    """-d 那档要提醒一句。

    不提醒的话，改完 LABEL_MODELS 跑个 -d，服务重启了、日志一切正常，
    只是平台下拉里少一组——人会以为是平台没刷新。
    """
    with open(os.path.join(_HERE, "up.sh"), encoding="utf-8") as f:
        src = f.read()
    seg = src[src.index("-d|--restart)"):src.index("-u|--up)")]
    assert "LABEL_MODELS" in seg and "不生效" in seg
