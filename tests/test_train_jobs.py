"""训练任务：每一版各用各的目录、能实时看日志、能删掉某一版。

背景（2026-09-23 第一次在网页上提交训练）：
  - 同一份数据集再训一次，train_custom.sh 的 --clean 先 rm -rf 掉上一版模型
    → 每一版只活到下一次提交为止
  - 提交训练是并发的，两次用不同归并表的训练同时整理同一份数据集，
    后一个会在前一个读到一半时改写 merged_tmp.json
  - 网页上只能看到「排队中」，步骤 0 挂了也看不到
"""

import json
import os

import pytest

jobs = pytest.importorskip("label_service.jobs")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(jobs.config, "JOBS_DIR", str(tmp_path / "jobs"))
    (tmp_path / "jobs").mkdir()
    (tmp_path / "configs").mkdir()
    return tmp_path


def _touch(p, text="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _job(repo, job_id, **kw):
    j = {"job_id": job_id, "status": "done", "dataset_spec": {"date": "ds_a"},
         "model_type": "rf", "tag": None, **kw}
    (repo / "jobs" / f"{job_id}.json").write_text(json.dumps(j), encoding="utf-8")
    return j


# ── 每一版各用各的目录 ──────────────────────────────────────────────────


def test_每个任务自己一个名字():
    assert jobs.run_date("ds_a", 7) == "ds_a__job7"
    assert jobs.run_date("ds_a", None) == "ds_a", "命令行手动跑、老调用方不变"


def test_同一份数据集两次训练_命令里的名字不同(repo):
    spec = {"date": "ds_a", "label_remap": {"活动": "活动"}}
    c1 = jobs.build_command(spec, "rf", None, 1)
    c2 = jobs.build_command(spec, "rf", None, 2)
    assert c1[c1.index("--date") + 1] == "ds_a__job1"
    assert c2[c2.index("--date") + 1] == "ds_a__job2"
    # 归并表也各是各的——训完第二版，第一版用的那张还找得回来
    assert c1[c1.index("--remap") + 1] != c2[c2.index("--remap") + 1]


def test_一起训练的那几份也按任务隔离(repo):
    spec = {"date": "ds_a", "extra_datasets": [{"date": "ds_old", "export_json": "x", "source_hz": 16}]}
    cmd = jobs.build_command(spec, "rf", None, 3)
    assert cmd[cmd.index("--extra_date") + 1] == "ds_old__job3:16"


# ── 实时日志 ────────────────────────────────────────────────────────────


def test_按offset接着读_不重复(repo):
    _job(repo, 1, status="running")
    log = repo / "jobs" / "1.log"
    log.write_text("▶ 步骤0：合并\n第一行\n", encoding="utf-8")
    r1 = jobs.read_log(1, 0)
    assert "第一行" in r1["text"]
    with open(log, "a", encoding="utf-8") as f:
        f.write("▶ 步骤1：生成CSV\n第二行\n")
    r2 = jobs.read_log(1, r1["offset"])
    assert "第一行" not in r2["text"] and "第二行" in r2["text"]
    assert r2["stage"] == "步骤1：生成CSV", "stage 是最后一个 ▶，告诉人跑到哪一步了"


def test_切在汉字中间不出乱码(repo, monkeypatch):
    """一次读一块，可能正好切在一个三字节汉字中间——那几个字节留到下一次。"""
    _job(repo, 2, status="running")
    (repo / "jobs" / "2.log").write_text("抓挠抓挠", encoding="utf-8")   # 12 字节
    monkeypatch.setattr(jobs, "_LOG_CHUNK", 4)                        # 切在第 2 个字中间
    r = jobs.read_log(2, 0)
    assert r["text"] == "抓" and r["offset"] == 3
    r = jobs.read_log(2, r["offset"])
    assert r["text"] == "挠"


def test_还没开跑_日志文件不存在也不报错(repo):
    _job(repo, 3, status="queued")
    r = jobs.read_log(3, 0)
    assert r["text"] == "" and r["status"] == "queued"


def test_任务不存在返回None(repo):
    assert jobs.read_log(999, 0) is None


# ── 删除某一版 ──────────────────────────────────────────────────────────


def test_删一版只删它自己那一摊_别的版本不动(repo):
    """这是 rm -rf，最要紧的就是这一条。"""
    for jid in (1, 2):
        rd = f"ds_a__job{jid}"
        _touch(repo / "data" / "raw_custom" / rd / "merged_tmp.json")
        _touch(repo / "data" / f"processed_{rd}_missing_none" / "x.npz")
        _touch(repo / "results" / f"processed_{rd}_missing_none" / "16hz_r" / "rf" / "ml_rf.pkl")
        _touch(repo / "configs" / f"remap_ui_job{jid}.yaml")
        _touch(repo / "jobs" / f"{jid}.log")
        _job(repo, jid, run_date=rd,
             model_path=str(repo / "results" / f"processed_{rd}_missing_none" / "16hz_r" / "rf" / "ml_rf.pkl"))

    out = jobs.delete_job(1, active_model_path=None)
    assert out["deleted"], "什么都没删"
    assert not (repo / "data" / "raw_custom" / "ds_a__job1").exists()
    assert not (repo / "results" / "processed_ds_a__job1_missing_none").exists()
    assert not (repo / "configs" / "remap_ui_job1.yaml").exists()
    assert not (repo / "jobs" / "1.json").exists()
    # 第 2 版一根毛都不能少
    assert (repo / "data" / "raw_custom" / "ds_a__job2" / "merged_tmp.json").exists()
    assert (repo / "results" / "processed_ds_a__job2_missing_none" / "16hz_r" / "rf" / "ml_rf.pkl").exists()
    assert (repo / "configs" / "remap_ui_job2.yaml").exists()
    assert (repo / "jobs" / "2.json").exists()


def test_job1不能顺带删掉job10(repo):
    """通配符 *__job1 要是写成前缀匹配，会把 job10、job11 一起带走。"""
    for jid in (1, 10):
        rd = f"ds_a__job{jid}"
        _touch(repo / "data" / "raw_custom" / rd / "merged_tmp.json")
        _job(repo, jid, run_date=rd)
    jobs.delete_job(1, active_model_path=None)
    assert (repo / "data" / "raw_custom" / "ds_a__job10" / "merged_tmp.json").exists()


def test_run_date不带任务号时绝不按通配符删(repo):
    """run_date 要是空的，拼出来就是 data/processed_* ——所有训练数据一次清光。"""
    _touch(repo / "data" / "processed_something_else" / "x.npz")
    _touch(repo / "results" / "processed_something_else" / "rf" / "ml_rf.pkl")
    _job(repo, 5, run_date="")                       # 坏数据：空的
    jobs.delete_job(5, active_model_path=None)
    assert (repo / "data" / "processed_something_else" / "x.npz").exists()
    _job(repo, 6, run_date="ds_a")                   # 坏数据：不带 __job6
    jobs.delete_job(6, active_model_path=None)
    assert (repo / "results" / "processed_something_else" / "rf" / "ml_rf.pkl").exists()


def test_老任务只删模型那一层(repo):
    """老任务那时候各版本共用目录，按目录删会连别的版本一起带走。"""
    shared = repo / "results" / "processed_ds_a_missing_none"
    _touch(shared / "16hz_r" / "rf" / "ml_rf.pkl")
    _touch(shared / "16hz_r" / "lgbm" / "ml_lgbm.pkl")          # 同一份数据集的另一版
    _job(repo, 4, model_path=str(shared / "16hz_r" / "rf" / "ml_rf.pkl"))   # 没有 run_date
    jobs.delete_job(4, active_model_path=None)
    assert not (shared / "16hz_r" / "rf").exists()
    assert (shared / "16hz_r" / "lgbm" / "ml_lgbm.pkl").exists()


def test_还在跑的不能删(repo):
    # pid 用测试进程自己的——它肯定活着，模拟"真的还在跑"
    _job(repo, 8, status="running", run_date="ds_a__job8", pid=os.getpid())
    with pytest.raises(jobs.JobBusy):
        jobs.delete_job(8, active_model_path=None)
    _job(repo, 9, status="queued", run_date="ds_a__job9")
    with pytest.raises(jobs.JobBusy):
        jobs.delete_job(9, active_model_path=None)


def test_正在用的模型不能删(repo):
    """删了之后下次重启服务就起不来。"""
    mp = repo / "results" / "processed_ds_a__job11_x" / "rf" / "ml_rf.pkl"
    _touch(mp)
    _job(repo, 11, run_date="ds_a__job11", model_path=str(mp))
    with pytest.raises(jobs.JobBusy):
        jobs.delete_job(11, active_model_path=str(mp))
    assert mp.exists()


def test_不存在的任务_删了也不报错(repo):
    assert jobs.delete_job(404, active_model_path=None) == {"deleted": []}


# ── 孤儿任务 / 停止 ─────────────────────────────────────────────────────
#
# 训练是在服务进程里拉起来的。服务一重启（部署就会重启）它们就跟着没了，可任务
# 文件里还写着 running——永远不会变。2026-09-23：两版失败的调试任务一直
# 「训练中」，删除按钮一直灰着，怎么都删不掉。


def _dead_pid():
    """一个肯定已经不在了的 pid。"""
    import subprocess

    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_启动时把孤儿任务标成失败(repo):
    _job(repo, 20, status="running", pid=_dead_pid())
    _job(repo, 21, status="queued")
    _job(repo, 22, status="done")
    fixed = jobs.reconcile_orphans()
    assert sorted(fixed) == [20, 21]
    assert jobs.get_job(20)["status"] == "failed"
    assert "重启" in jobs.get_job(20)["error"]
    assert jobs.get_job(22)["status"] == "done", "跑完的别碰"


def test_启动时真还活着的不碰(repo):
    """极少见（服务没随容器一起重启），但真活着的标成失败就是错杀。"""
    _job(repo, 23, status="running", pid=os.getpid())
    assert jobs.reconcile_orphans() == []
    assert jobs.get_job(23)["status"] == "running"


def test_标着在跑但进程没了_能直接删(repo):
    """不用等下次重启——删的时候发现进程不在了，就当孤儿处理。"""
    _touch(repo / "data" / "raw_custom" / "ds_a__job24" / "merged_tmp.json")
    _job(repo, 24, status="running", run_date="ds_a__job24", pid=_dead_pid())
    out = jobs.delete_job(24, active_model_path=None)
    assert out["deleted"]
    assert not (repo / "data" / "raw_custom" / "ds_a__job24").exists()


def test_停止要把整个进程组一起停(repo):
    """脚本会再拉起一堆 python，「带合成」那一版还是后台 & 跑的。只停 bash 那一个
    的话，孩子们照样在后台吃 CPU、往目录里写文件。

    这里真的起一个 bash，让它在后台再拉一个 sleep，然后停——两个都得没了。"""
    import subprocess
    import time

    child_file = repo / "child.pid"
    proc = subprocess.Popen(
        ["bash", "-c", f"sleep 60 & echo $! > {child_file}; wait"],
        start_new_session=True,
    )
    for _ in range(50):
        if child_file.exists() and child_file.read_text().strip():
            break
        time.sleep(0.05)
    child = int(child_file.read_text().strip())
    assert jobs._alive(proc.pid) and jobs._alive(child)

    _job(repo, 25, status="running", pid=proc.pid)
    jobs.cancel_job(25)
    proc.wait(timeout=5)
    for _ in range(50):
        if not jobs._alive(child):
            break
        time.sleep(0.05)
    assert not jobs._alive(child), "后台那个孩子还活着——只停了 bash 没停整组"
    assert jobs.get_job(25)["status"] == "failed"
    assert jobs.get_job(25)["error"] == "手动停止"


def test_停已经结束的_原样返回不改(repo):
    _job(repo, 26, status="done")
    assert jobs.cancel_job(26)["status"] == "done"


def test_一开跑就有日志_不是一直等日志(repo):
    """整理数据集要一会儿，以前这段时间日志文件还不存在，网页上一直「等日志…」。"""
    src = open("label_service/jobs.py", encoding="utf-8").read()
    i_first_write = src.index("▶ 整理数据集")
    i_prepare = src.index("await asyncio.to_thread(prepare_export")
    assert i_first_write < i_prepare, "得先往日志里写一句，再去整理数据集"


# ── 任务号只增不减 ──────────────────────────────────────────────────────


def test_删光了也不会从1重来(repo):
    """以前是"现有最大号 + 1"：删光就从 1 重来（2026-09-23：平台第 4 版，这边是 job1）。
    平台记录里存的就是这个号，复用了的话旧记录会指到别人的任务上。"""
    a = jobs._next_job_id()
    b = jobs._next_job_id()
    assert b == a + 1
    for n in os.listdir(repo / "jobs"):
        if n.endswith(".json"):
            os.remove(repo / "jobs" / n)
    assert jobs._next_job_id() == b + 1


def test_删掉最新那个_号也不复用(repo):
    _job(repo, 1)
    _job(repo, 2)
    n = jobs._next_job_id()
    assert n == 3
    os.remove(repo / "jobs" / "2.json")
    assert jobs._next_job_id() == 4


def test_计数器文件丢了_退回按现有最大号(repo):
    """至少不比原来差。"""
    _job(repo, 7)
    counter = repo / "jobs" / ".next_id"
    if counter.exists():
        counter.unlink()
    assert jobs._next_job_id() == 8


def test_脚本里的tail要带pid_不能靠事后kill():
    """$! 拿到的是管道最后一个进程（sed）。kill 它只杀了 sed，前头的 tail -f 永远
    不退，后面 wait 整条管道就卡死——脚本停在「方案 A 完成」之后，模型路径打不
    出来，任务永远「训练中」。"""
    src = open("train_custom.sh", encoding="utf-8").read()
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if s.startswith("tail ") and " -f " in f" {s} ":
            assert "--pid=" in s, f"tail -f 没带 --pid，训练结束后会一直跟着文件不退：{s}"
    assert 'kill "$TAIL_A"' not in src


# ── 模型其实存好了，别扔 ────────────────────────────────────────────────


def _saved_model(repo, jid):
    d = repo / "results" / f"processed_ds_a__job{jid}_missing_none_acc3" / "16hz_remap_ui_job1" / "rf"
    _touch(d / "ml_rf.pkl")
    (d / "ml_rf.json").write_text(json.dumps({"macro_f1": 0.53, "classes": ["抓挠-头颈耳"]}), encoding="utf-8")
    return d


def test_重启时模型已经存好了_按完成算(repo):
    """2026-09-23：方案 A 训完存好，脚本在收尾时卡死，「模型路径:」没打出来。
    重启后一律标成失败，等于把训好的模型扔了。"""
    d = _saved_model(repo, 30)
    (repo / "jobs" / "30.log").write_text(
        f"[A] [ml/train] 结果保存至 {os.path.relpath(d, repo)}/\n  ✅ 方案 A 完成\n", encoding="utf-8")
    _job(repo, 30, status="running", pid=_dead_pid())
    jobs.reconcile_orphans()
    j = jobs.get_job(30)
    assert j["status"] == "done"
    assert j["model_path"].endswith("ml_rf.pkl")
    assert j["metrics"]["macro_f1"] == 0.53, "指标也要带上，网页上才有 F1"


def test_日志说存了但文件不在_照样算失败(repo):
    """只认真存在的文件——日志里写了、文件被删了的，不能当成功。"""
    (repo / "jobs" / "31.log").write_text("[A] [ml/train] 结果保存至 results/nope/rf/\n", encoding="utf-8")
    _job(repo, 31, status="running", pid=_dead_pid())
    jobs.reconcile_orphans()
    assert jobs.get_job(31)["status"] == "failed"


def test_手动停止时模型已经存好了_也按完成算(repo):
    d = _saved_model(repo, 32)
    (repo / "jobs" / "32.log").write_text(f"结果保存至 {os.path.relpath(d, repo)}/\n", encoding="utf-8")
    _job(repo, 32, status="running", pid=_dead_pid())
    j = jobs.cancel_job(32)
    assert j["status"] == "done" and j["model_path"].endswith("ml_rf.pkl")


def test_有带合成的就用带合成的(repo):
    """跟正常结束时的取法一致。"""
    a = repo / "results" / "p" / "16hz_r" / "rf"
    b = repo / "results" / "p" / "16hz_r_syn" / "rf"
    for d in (a, b):
        _touch(d / "ml_rf.pkl")
    (repo / "jobs" / "33.log").write_text(
        f"[B] 结果保存至 {os.path.relpath(b, repo)}/\n[A] 结果保存至 {os.path.relpath(a, repo)}/\n", encoding="utf-8")
    _job(repo, 33, status="running", pid=_dead_pid())
    jobs.reconcile_orphans()
    assert "_syn" in jobs.get_job(33)["model_path"]
