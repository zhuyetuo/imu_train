"""./up.sh deploy：更新到最新的一条命令。真跑要 docker 和网，用 DRY_RUN=1 看它打算跑什么。"""

import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UP = os.path.join(ROOT, "up.sh")


def _dry(*args, **env):
    r = subprocess.run(["bash", UP, "deploy", *args], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "DRY_RUN": "1", **env}, timeout=60)
    return r.returncode, r.stdout + r.stderr


def _cmds(out):
    return [l.strip()[2:] for l in out.splitlines() if l.strip().startswith("$ ")]


def test_语法():
    assert subprocess.run(["bash", "-n", UP]).returncode == 0


def test_两个服务都更新_顺序对_子模块跟着():
    rc, out = _dry()
    assert rc == 0, out
    c = _cmds(out)
    assert c[:2] == ["git pull --ff-only", "git submodule update --init --recursive"]
    assert "bash label_service/up.sh -u" in c and "bash label_service/up.sh -d" in c
    assert "./vision_service/run.sh down" in c and "./vision_service/run.sh -d" in c
    assert c.index("bash label_service/up.sh -d") < c.index("./vision_service/run.sh -d")
    assert "全部更新完成" in out


def test_only_和_gpu():
    rc, out = _dry("--only", "vision")
    assert rc == 0 and "label_service/up.sh" not in out and "./vision_service/run.sh -d" in out
    rc, out = _dry("-g")
    assert "label_service/up.sh -g -u" in out and "label_service/up.sh -g -d" in out
    rc, out = _dry(DEPLOY_GPU="1")
    assert "label_service/up.sh -g -u" in out
