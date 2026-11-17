"""负例挖掘：负例类别的片段挪进「活动」、变成候选；跟抓挠 / 已有候选重叠的不给；上限按类别各算"""
from label_service.mining import REASON, mine_negatives


def seg(a, b, conf=0.8):
    return {"start_ts": f"2026-11-17 10:00:{a:02d}.000", "end_ts": f"2026-11-17 10:00:{b:02d}.000",
            "conf_max": conf, "conf_mean": conf, "n_windows": b - a}


def test_edge_touch_with_scratch_still_given():
    """一段长饮水开头跟短抓挠擦边（盖住不到一半）照样给"""
    segments = {"抓挠": [seg(0, 5)], "饮水": [seg(4, 40)]}
    _, cands = mine_negatives(segments, [], ["活动", "抓挠", "饮水"], ["饮水"])
    assert [c["label"] for c in cands] == ["饮水"]


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
    # 疑似抓挠候选（不带 label）压住的不给
    existing = [{"start_ts": "2026-11-17 10:00:49.000", "end_ts": "2026-11-17 10:00:50.000"}]
    _, cands = mine_negatives(segments, existing, ["活动", "嗅闻"], ["嗅闻"], max_per_label=5)
    assert len(cands) == 5
    assert cands[0]["conf_mean"] == 0.48   # 置信度高的排前面；0.49 那段被疑似抓挠压住了


def test_other_kind_rule_candidates_do_not_block_same_kind_does():
    """规则候选（姿态像舔、振荡像甩头）跟它重叠照样给；同一类的已有候选重叠就不重复给"""
    segments = {"饮水": [seg(10, 20, conf=0.99)], "进食": [seg(30, 40, conf=0.9)]}
    existing = [{**seg(12, 15), "label": "舔身体", "reason": "grooming"},
                {**seg(13, 14), "label": "甩头/抖身", "reason": "shaking"},
                {**seg(31, 33), "label": "进食", "reason": "eating"}]
    _, cands = mine_negatives(segments, existing, ["活动", "进食", "饮水"], ["进食", "饮水"])
    assert [c["label"] for c in cands] == ["饮水"]


def test_model_without_negative_classes_unchanged():
    segments = {"活动": [seg(0, 5)], "抓挠": [seg(6, 9)]}
    out, cands = mine_negatives(segments, [], ["活动", "抓挠"], ["进食"])
    assert out == {"活动": [seg(0, 5)], "抓挠": [seg(6, 9)]} and cands == []
