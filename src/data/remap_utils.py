"""
标签重映射工具，ml/train.py 和 dl/train.py 共用——之前这套逻辑只在
ml/train.py里，dl/train.py完全没有重映射能力，只能拿16个原始细分类别
直接训练（跟ml这边训出来的4类模型不是同一个任务，没法比较，也不是
实际想要的行为分类粒度）。抽到这里统一维护，两边import，不用各自
一份、容易改一边漏改另一边。
"""

import numpy as np
import yaml


def load_remap_yaml(path: str) -> dict:
    """加载remap yaml，过滤掉注释行（以#开头的key，用来在yaml里写"故意不映射"
    这类说明，不是真的映射规则）。"""
    with open(path) as f:
        remap_cfg = yaml.safe_load(f)
    return {k: v for k, v in remap_cfg.items() if not str(k).startswith("#")}


def apply_remap(y, classes, remap: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    remap: {"Lying chest": "睡觉", "Walking": "活动", ...}
    返回重映射后的 y、新 classes 列表、keep_mask。

    remap配置里没覆盖到的原始类别（比如自采数据里的"未佩戴"——这个不是
    行为，是设备没戴在狗身上，训练行为分类器不该把这类样本也塞进去，见
    src/data/wear_state.py专门的未佩戴/静置检测）会被过滤掉，不是直接
    KeyError崩溃：mapping字典本来就只收录remap覆盖到的类别(`if c in
    remap`)，但之前这里对y的每个样本都无条件查表，没有同步过滤，遇到
    没覆盖的类别就崩了。keep_mask标记哪些样本被保留，调用方要用同一个
    mask去过滤对应的X，保持X/y行数一致。
    """
    new_class_names = list(dict.fromkeys(remap.values()))  # 保序去重
    new_class2id = {c: i for i, c in enumerate(new_class_names)}
    mapping = {i: new_class2id[remap[c]] for i, c in enumerate(classes) if c in remap}
    keep_mask = np.array([int(label) in mapping for label in y], dtype=bool)
    new_y = np.array([mapping[int(label)] for label in y[keep_mask]], dtype=np.int64)
    return new_y, new_class_names, keep_mask


def apply_remap_seq(y_seq, classes, remap: dict, new_class_names: list) -> np.ndarray:
    """逐帧版本，给many-to-many模型用——y_seq是(N, T)的逐帧原始类别id。
    跟窗口级apply_remap用同一份mapping/new_class_names（调用方先对窗口级
    y跑一遍apply_remap拿到new_class_names，再传进来，保证两边的类别id
    编号一致）。remap没覆盖到的帧不是丢窗口，是标成-1（"忽略"），
    make_loader()/m2m_loss()本来就是按这个惯例设计的（"-1 表示未映射帧，
    训练时忽略"）——一个窗口内大部分帧是目标行为、边界处夹杂了几帧
    别的动作，没必要整个窗口作废，只忽略那几帧就行。"""
    new_class2id = {c: i for i, c in enumerate(new_class_names)}
    mapping = {i: new_class2id[remap[c]] for i, c in enumerate(classes) if c in remap}
    vmapped = np.vectorize(lambda v: mapping.get(int(v), -1))
    return vmapped(y_seq).astype(np.int64)
