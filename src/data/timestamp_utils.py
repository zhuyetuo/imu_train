"""
pc_ms（Unix epoch毫秒）转本地时间的统一实现——推理端（infer_csv_scratch.py）和
裁剪端（extract_clips.py）都要用同一份，不能各自实现一遍：之前就是分别在两处
独立解析时间戳，一个地方加了对pc_ms的支持、另一个地方没同步改，导致同一批数据
在两个环节的时间戳解析结果对不上（一个能用、一个失败）。以后要改时间戳解析逻辑，
只改这一个文件。
"""
import time

import pandas as pd


def local_utc_offset_seconds() -> float:
    """当前系统的本地时区相对UTC的偏移秒数（正值=本地时间比UTC晚，比如中国UTC+8是28800）。"""
    return -time.altzone if time.localtime().tm_isdst else -time.timezone


def pc_ms_to_local_datetime(ms_series) -> pd.Series:
    """
    把 pc_ms（Unix epoch毫秒，真实时刻不分时区）转成本地时间的 pandas datetime。

    效果要等价于对每个值调用 datetime.fromtimestamp(ms/1000)（采集端当年生成
    "timestamp"绝对时间字符串列用的就是这个函数，它是按本地时区转换的）——但
    pd.to_datetime(..., unit="ms") 默认按UTC解释毫秒数，不做本地时区转换，两者
    直接混用会导致解析出来的时间差一个时区偏移量（国内UTC+8就是差8小时）。

    做法：转换前先把毫秒数加上本地时区偏移量，再交给 pd.to_datetime 按"UTC"解释——
    这样pandas计算出来的日历字段（年月日时分秒）就跟本地时间的日历字段一致了。
    跨夏令时切换时刻的极端情况没有特殊处理（这类IMU数据一般不会精确卡在切换
    那一刻，不追求处理这种边界情况）。
    """
    ms = pd.to_numeric(ms_series, errors="coerce")
    sec = ms / 1000.0
    return pd.to_datetime(sec + local_utc_offset_seconds(), unit="s")


def pc_ms_value_to_ts_string(ms_value) -> str:
    """单个 pc_ms 值转成 "%Y-%m-%d %H:%M:%S.%f"[:-3] 格式字符串（跟 timestamp 列
    的字符串格式一致），给 extract_clips.py 那些只认字符串格式的老代码复用，
    不用改它们内部的解析逻辑。"""
    dt = pd.to_datetime(float(ms_value) / 1000.0 + local_utc_offset_seconds(), unit="s")
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
