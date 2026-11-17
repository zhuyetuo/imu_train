"""公开数据集 → 我们自己的格式：每段录制一个项圈 CSV（timestamp, acc_x/y/z [g], gyro_x/y/z [°/s]，颈部 / 项圈那个传感器）
+ 一份跟平台导出一样的 JSON（data.csv / sample_code，annotations.result.value 里 start/end/start_ms/end_ms/timeserieslabels）。
train_v2 把它当一份额外的数据集读，重采样到 20 Hz、换方向无关通道，跟我们自己的数据走同一条路。

  Vehkaoja 2022（Mendeley vxhx934tbn，CC BY 4.0）：45 只狗，ActiGraph 颈部 + 背部，100 Hz
  Marcato 2023（Mendeley mpph6bmn7g，CC BY 4.0）：42 只狗，颈 / 背 / 胸，100 Hz

分组：sample_code 写成 pub_<数据集>_<假日期>_imu<狗号>，train_v2 按「天 × 项圈」分组时同一只狗永远在同一折，不会泄漏。
用法：python convert_public.py  （读 ~/ai_data/public_datasets/*/extracted，写 collar_csv/ 和 export.json）
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/ai_data/public_datasets")
HZ = 100.0
MIN_SEG_S = 0.5

# 标签链：最后一个是叶子；train_v2 从叶子往上找归并表 / 训练类别（「嗅闻」不在表里就落到「活动」）
VEHK_MAP = {
    "Lying chest": ["静止/休息"], "Sitting": ["静止/休息"], "Standing": ["静止/休息"],
    "Walking": ["活动"], "Trotting": ["活动"], "Galloping": ["活动"], "Pacing": ["活动"], "Playing": ["活动"],
    "Jumping": ["活动"], "Carrying object": ["活动"], "Tugging": ["活动"], "Panting": ["活动", "喘气"],
    "Sniffing": ["活动", "嗅闻"], "Eating": ["活动", "进食"], "Drinking": ["活动", "饮水"],
    "Shaking": ["甩头/抖身"],
}
VEHK_PRIORITY = ["Shaking", "Drinking", "Eating", "Sniffing"]   # 三列同时标了的话，这几种优先
MARC_MAP = {"standing": ["静止/休息"], "sitting": ["静止/休息"], "lying down": ["静止/休息"],
            "walking": ["活动"], "body shake": ["甩头/抖身"]}


def runs(labels: np.ndarray):
    """连续相同标签 → (起, 止, 标签)，下标是采样点"""
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(labels)]
    return [(int(a), int(b), labels[a]) for a, b in zip(starts, ends)]


def write_rec(ds: str, rec_id: str, dog: int, day: datetime, acc: np.ndarray, gyr: np.ndarray, labels: np.ndarray,
              mapping: dict, out_dir: str, tasks: list, stats: dict):
    n = len(acc)
    t = [day + timedelta(seconds=i / HZ) for i in range(n)]
    csv_rel = os.path.join("public_datasets", ds, "collar_csv", f"{rec_id}.csv")
    path = os.path.join(out_dir, f"{rec_id}.csv")
    df = pd.DataFrame({"timestamp": [x.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for x in t],
                       "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
                       "gyro_x": gyr[:, 0], "gyro_y": gyr[:, 1], "gyro_z": gyr[:, 2]})
    df.to_csv(path, index=False, float_format="%.6f")
    result = []
    for a, b, lab in runs(labels):
        chain = mapping.get(lab)
        if not chain or (b - a) / HZ < MIN_SEG_S:
            continue
        result.append({"from_name": "label", "to_name": "ts", "type": "timeserieslabels", "value": {
            "start": t[a].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "end": t[b - 1].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "start_ms": int(round(a / HZ * 1000)), "end_ms": int(round((b - 1) / HZ * 1000)),
            "timeserieslabels": chain, "source_label": lab}})
        k = chain[-1]
        stats.setdefault(k, [0, 0.0])
        stats[k][0] += 1
        stats[k][1] += (b - a) / HZ
    tasks.append({"id": len(tasks) + 1,
                  "data": {"csv": csv_rel, "sample_code": f"pub_{ds[:4]}_{day:%Y%m%d}_imu{dog}", "sample_rate_hz": HZ,
                           "source": ds, "dog": dog},
                  "annotations": [{"id": len(tasks) + 1, "result": result}]})


def vehkaoja():
    ds = "vehkaoja2022"
    src = os.path.join(BASE, ds, "extracted", "csv", "DogMoveData.csv")
    out = os.path.join(BASE, ds, "collar_csv")
    os.makedirs(out, exist_ok=True)
    cols = ["DogID", "TestNum", "ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z",
            "Behavior_1", "Behavior_2", "Behavior_3"]
    d = pd.read_csv(src, usecols=cols)
    tasks, stats = [], {}
    for (dog, test), g in d.groupby(["DogID", "TestNum"], sort=True):
        b = g[["Behavior_1", "Behavior_2", "Behavior_3"]].to_numpy(str)
        lab = b[:, 0].copy()
        for p in reversed(VEHK_PRIORITY):           # 优先级高的最后写，覆盖前面的
            lab[(b == p).any(1)] = p
        day = datetime(2019, 1, 1) + timedelta(days=int(dog))  # 同一只狗两次测试同一天 → 同一组
        if test == 2:
            day += timedelta(hours=12)
        write_rec(ds, f"vehk_dog{dog}_test{test}", int(dog), day, g[["ANeck_x", "ANeck_y", "ANeck_z"]].to_numpy(),
                  g[["GNeck_x", "GNeck_y", "GNeck_z"]].to_numpy(), lab, VEHK_MAP, out, tasks, stats)
    return ds, tasks, stats


def marcato():
    ds = "marcato2023"
    src = os.path.join(BASE, ds, "extracted", "Inertial sensor dataset for Dog Posture Recognition", "df_raw.csv")
    out = os.path.join(BASE, ds, "collar_csv")
    os.makedirs(out, exist_ok=True)
    cols = ["Timestamp", "Neck.Acc.X", "Neck.Acc.Y", "Neck.Acc.Z", "Neck.Gyr.X", "Neck.Gyr.Y", "Neck.Gyr.Z",
            "Position", "Subject"]
    d = pd.read_csv(src, usecols=cols)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    subjects = {s: i + 100 for i, s in enumerate(sorted(d["Subject"].unique()))}   # 狗号 100+，跟 Vehkaoja 分开
    tasks, stats = [], {}
    for subj, g in d.groupby("Subject", sort=True):
        g = g.sort_values("Timestamp")
        gap = g["Timestamp"].diff().dt.total_seconds().fillna(0).to_numpy() > 1.0   # 断开 1 s 以上算另一段录制
        for k, r in g.groupby(np.cumsum(gap)):
            if len(r) < HZ * 2:
                continue
            day = datetime(2019, 6, 1) + timedelta(days=subjects[subj], seconds=int(k) * 3600)
            write_rec(ds, f"marc_{subj}_{k}", subjects[subj], day.replace(hour=0) + timedelta(minutes=int(k) * 30),
                      r[["Neck.Acc.X", "Neck.Acc.Y", "Neck.Acc.Z"]].to_numpy(),
                      r[["Neck.Gyr.X", "Neck.Gyr.Y", "Neck.Gyr.Z"]].to_numpy(),
                      r["Position"].to_numpy(str), MARC_MAP, out, tasks, stats)
    return ds, tasks, stats


if __name__ == "__main__":
    for fn in (vehkaoja, marcato):
        ds, tasks, stats = fn()
        with open(os.path.join(BASE, ds, "export.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False)
        print(f"== {ds}: {len(tasks)} 段录制")
        for k, (n, s) in sorted(stats.items(), key=lambda x: -x[1][1]):
            print(f"   {k:<8} {n:>5} 段 {s / 60:8.1f} 分钟")
