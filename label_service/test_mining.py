"""负例挖掘：负例类别的片段挪进「活动」、变成候选；跟抓挠 / 已有候选重叠的不给；上限按类别各算"""
from label_service.mining import REASON, mine_negatives


def seg(a, b, conf=0.8):
    return {"start_ts": f"2026-11-17 10:00:{a:02d}.000", "end_ts": f"2026-11-17 10:00:{b:02d}.000",
            "conf_max": conf, "conf_mean": conf, "n_windows": b - a}


def test_negatives_become_candidates_and_activity():
    segments = {"活动": [seg(0, 5)], "抓挠": [seg(20, 25)], "进食": [seg(6, 10), seg(22, 24)], "饮水": [seg(30, 33)]}
    classes = ["未佩戴", "静止/休息", "活动", "抓挠", "甩头/抖身", "进食", "饮水", "嗅闻"]
    out, cands = mine_negatives(segments, [], classes, ["进食", "饮水", "嗅闻"])
    assert "进食" not in out and "饮水" not in out
    assert [s["start_ts"][-6:-4] for s in out["活动"]] == ["00", "06", "22", "30"]
    # 跟抓挠重叠的那段进食不给候选（22~24 落在抓挠 20~25 里）
    assert sorted((c["label"], c["start_ts"][-6:-4]) for c in cands) == [("进食", "06"), ("饮水", "30")]
    assert all(c["reason"] == REASON for c in cands)


def test_skip_overlap_with_existing_candidates_and_cap():
    segments = {"嗅闻": [seg(i, i + 1, conf=i / 100) for i in range(0, 50)]}
    existing = [{"start_ts": "2026-11-17 10:00:00.000", "end_ts": "2026-11-17 10:00:01.000"}]
    _, cands = mine_negatives(segments, existing, ["活动", "嗅闻"], ["嗅闻"], max_per_label=5)
    assert len(cands) == 5
    assert cands[0]["conf_mean"] == 0.49   # 置信度高的排前面
    assert all(c["start_ts"] != existing[0]["start_ts"] for c in cands)


def test_model_without_negative_classes_unchanged():
    segments = {"活动": [seg(0, 5)], "抓挠": [seg(6, 9)]}
    out, cands = mine_negatives(segments, [], ["活动", "抓挠"], ["进食"])
    assert out == {"活动": [seg(0, 5)], "抓挠": [seg(6, 9)]} and cands == []
