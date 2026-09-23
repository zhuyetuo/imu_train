"""训练时的类别归并：把太少的细类并走、把不训的行为折成负样本。

## 为什么需要它

训练取的是整条链的第 0 个（labels[0]，见 src/data/labelstudio_to_custom.py），
所以「抓挠-肩胸」本来就会变成「抓挠」。但人要做的归并不止这一种：

  1. 细类太少（实测 ds_20260923_1136：抓挠-肩胸 1 段 9 秒）→ 并进兄弟类别
  2. 不训的行为（舔、甩头/抖身）→ 折进「活动」当负样本

第 2 种是必须的：configs/remap_custom_3class.yaml 用的还是旧模板的名字
（舔身体 / 甩身体 / 蹭擦身体），现在模板发的是「舔」「甩头/抖身」「蹭」，
对不上，apply_remap() 就把这些样本**静默丢掉**了（只在训练日志里打一句）。
实测那份数据集 155 段里有 13 段是这么没的。
"""

import pytest

# 只导 jobs，不导 app：重的推理依赖（joblib 等）挂在 app.py 上，
# 而这里要测的是 jobs 里那个纯函数
jobs = pytest.importorskip("label_service.jobs")


def _tasks(*chains):
    return [{"annotations": [{"result": [
        {"value": {"timeserieslabels": list(c), "start": "s", "end": "e"}} for c in chains
    ]}]}]


def _labels(tasks):
    return [seg["value"]["timeserieslabels"]
            for t in tasks for a in t["annotations"] for seg in a["result"]]


def test_按细类归并_只动命中的那一条():
    """抓挠-肩胸只有 1 段，并进抓挠-躯干；头颈耳不受影响。"""
    tasks = _tasks(["抓挠", "抓挠-肩胸"], ["抓挠", "抓挠-头颈耳"])
    n = jobs.apply_label_remap(tasks, {"抓挠-肩胸": "抓挠-躯干"})
    assert n == 1
    assert _labels(tasks) == [["抓挠-躯干"], ["抓挠", "抓挠-头颈耳"]]


def test_按大类归并_整条链一起搬():
    """舔不是要训的目标，整个大类折进「活动」当负样本——不管它有没有二级。"""
    tasks = _tasks(["舔"], ["舔", "舔-躯干"], ["抓挠", "抓挠-躯干"])
    n = jobs.apply_label_remap(tasks, {"舔": "活动"})
    assert n == 2
    assert _labels(tasks) == [["活动"], ["活动"], ["抓挠", "抓挠-躯干"]]


def test_细类优先于大类():
    """两条都配了就听细的那条——不然"把舔折成活动、但舔-躯干单独留着"做不到。"""
    tasks = _tasks(["舔", "舔-躯干"], ["舔", "舔-前肢"])
    jobs.apply_label_remap(tasks, {"舔": "活动", "舔-躯干": "抓挠"})
    assert _labels(tasks) == [["抓挠"], ["活动"]]


def test_命中后整条链换掉_只留新名():
    """训练只看第 0 个，留着旧父级没意义，还会让人以为层级还在。"""
    tasks = _tasks(["抓挠", "抓挠-头颈耳"])
    jobs.apply_label_remap(tasks, {"抓挠-头颈耳": "抓挠"})
    assert _labels(tasks) == [["抓挠"]]


def test_映射成自己等于没改_不计数():
    """界面上「保持原样」会把原名填成同名，别把它算成改过。"""
    tasks = _tasks(["抓挠", "抓挠-躯干"])
    assert jobs.apply_label_remap(tasks, {"抓挠-躯干": "抓挠"}) == 1   # 根变了，算改
    tasks2 = _tasks(["抓挠"])
    assert jobs.apply_label_remap(tasks2, {"抓挠": "抓挠"}) == 0        # 根没变
    assert _labels(tasks2) == [["抓挠"]]


def test_没给映射就一个字节都不动():
    tasks = _tasks(["抓挠", "抓挠-躯干"], ["舔"])
    assert jobs.apply_label_remap(tasks, {}) == 0
    assert _labels(tasks) == [["抓挠", "抓挠-躯干"], ["舔"]]


def test_空链和缺字段不会炸():
    """老数据/脏数据里真有 result 为空、value 没 timeserieslabels 的。"""
    tasks = [{"annotations": [{"result": [
        {"value": {"timeserieslabels": []}}, {"value": {}}, {},
    ]}]}, {"annotations": []}, {}]
    assert jobs.apply_label_remap(tasks, {"舔": "活动"}) == 0


# ── 归并表 → 训练类别 ────────────────────────────────────────────────────
#
# 这一段管的是「能不能识别出是什么抓挠」。训练取链的第 0 个，所以把
# 「抓挠-头颈耳」收成独立一段还不够——train.py 后面那张 remap 表里没有这个
# 名字的话，样本照样被丢掉。训练类别就是归并表里那些目标名。


def _spec(tmp_path, remap, date="ds_x"):
    return {"date": date, "label_remap": remap}


def test_把细类映射成自己就能自成一类(tmp_path, monkeypatch):
    """这就是「除了识别抓挠，还能识别是什么抓挠」的做法。"""
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    rel = jobs.write_runtime_remap(_spec(tmp_path, {
        "抓挠-头颈耳": "抓挠-头颈耳",
        "抓挠-躯干": "抓挠-躯干",
        "舔": "活动",
        "睡觉": "睡觉",
    }))
    text = (tmp_path / rel).read_text(encoding="utf-8")
    body = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert body == ["抓挠-头颈耳: 抓挠-头颈耳", "抓挠-躯干: 抓挠-躯干",
                    "活动: 活动", "睡觉: 睡觉"]


def test_只按用到的目标建类_不留空类(tmp_path, monkeypatch):
    """抓挠全拆成部位之后「抓挠」自己一个样本都没有，不该还占一个类别。"""
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    rel = jobs.write_runtime_remap(_spec(tmp_path, {
        "抓挠-头颈耳": "抓挠-头颈耳", "抓挠-躯干": "抓挠-躯干", "活动": "活动",
    }))
    body = [l for l in (tmp_path / rel).read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")]
    assert "抓挠: 抓挠" not in body


def test_没给归并表就不写文件_走默认那张(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    assert jobs.write_runtime_remap({"date": "ds_x"}) is None
    assert jobs.write_runtime_remap({"date": "ds_x", "label_remap": {}}) is None


def test_文件名里的数据集名要洗干净(tmp_path, monkeypatch):
    """数据集名会进文件名，再进 results/ 的目录名——带斜杠就写到别处去了。"""
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    rel = jobs.write_runtime_remap(_spec(tmp_path, {"活动": "活动"}, date="../../etc/ds x"))
    assert rel.startswith("configs/remap_ui_")
    assert "/" not in rel[len("configs/"):]
    assert (tmp_path / rel).is_file()


def test_命令里带上生成的那张表(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    (tmp_path / "configs").mkdir()
    cmd = jobs.build_command(_spec(tmp_path, {"抓挠-躯干": "抓挠-躯干"}), "rf", None)
    assert "--remap" in cmd
    assert cmd[cmd.index("--remap") + 1].startswith("configs/remap_ui_")


def test_没归并表时命令里不带remap(tmp_path, monkeypatch):
    """默认那张 3 类表是 train_custom.sh 自己的默认值，别多此一举地传一遍。"""
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    assert "--remap" not in jobs.build_command({"date": "ds_x"}, "rf", None)
