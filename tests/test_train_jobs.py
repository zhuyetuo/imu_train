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
    _job(repo, 8, status="running", run_date="ds_a__job8")
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
