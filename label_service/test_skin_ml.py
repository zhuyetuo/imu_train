"""皮肤 ML 整批接口里不用训练模型就能测的部分：输入整理、真实标签不够时拒绝训练。"""
from label_service import skin_ml as M


def test_frames_parses_events_and_merges_wear():
    rows = [
        {"pet_id": "IMU1+IMU2", "date": "2026-11-01", "wear_seconds": 36000, "breed": "比熊",
         "events": [["2026-11-01 10:00:00.000", "2026-11-01 10:00:30.000"], ["坏的", "x"],
                    ["2026-11-01 11:00:00", "2026-11-01 10:59:00"]]},
        {"pet_id": "IMU1+IMU2", "date": "2026-11-01", "wear_seconds": 72000, "events": []},
    ]
    ev, wear, breed = M._frames(rows)
    assert len(ev) == 1 and ev.iloc[0]["duration_sec"] == 30
    assert len(wear) == 1 and wear.iloc[0]["valid_wear_hours"] == 20.0
    assert breed == {"IMU1+IMU2": "比熊"}


def test_train_refuses_when_labels_too_few(monkeypatch):
    monkeypatch.setattr(M, "models", lambda: (None, None))
    r = M.train_b_real([], [{"pet_id": "a", "date": "2026-11-01", "s_tier": "S1"}])
    assert r["ok"] is False and "不够" in r["message"]
    assert M.train_b_real([], [])["ok"] is False
