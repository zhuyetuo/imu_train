"""
加载模型 + 元数据，逻辑跟 src/infer_csv_scratch.main() 里那段一样（.pt 走 DL、
.pkl 走 joblib，参数从同名 .json 读）。放在 label_service 自己目录里、不去改
src/ 下的原有代码——imu_train 原来的命令行怎么用还怎么用，这个目录纯粹是
新增的。如果以后 main() 那段加载逻辑变了，这里要跟着同步一下。
"""

import json
import os

import joblib

from infer_csv_scratch import _load_dl_model  # src/ 已由 app.py 加进 sys.path


def load_model_bundle(model_path: str) -> dict:
    """返回 dict: model, classes, is_dl, gravity_aligned, hz, window_s, stride_s, label_mode"""
    is_dl = model_path.endswith(".pt")
    classes, gravity_aligned, t_hz, t_window_s, t_stride_s = [], True, 16, 2.0, 1.0
    label_mode = "majority"
    if is_dl:
        model, dl_meta = _load_dl_model(model_path)
        classes         = dl_meta["classes"]
        gravity_aligned = dl_meta["gravity_aligned"]
        t_hz            = int(dl_meta["hz"])
        t_window_s      = dl_meta["window_size"] / t_hz
        t_stride_s      = dl_meta["stride"] / t_hz
        label_mode      = dl_meta.get("label_mode", "majority")
    else:
        model = joblib.load(model_path)
        meta_path = model_path.replace(".pkl", ".json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            classes         = meta.get("classes", [])
            gravity_aligned = meta.get("gravity_aligned", True)
            t_hz            = int(meta.get("hz", 16))
            t_window_s      = float(meta.get("window_s", 2.0))
            t_stride_s      = float(meta.get("stride_s", 1.0))
            label_mode      = meta.get("label_mode", "majority")
        else:
            classes = list(model.classes_) if hasattr(model, "classes_") else []
    print(f"[模型] 采样率={t_hz}Hz  窗口={t_window_s}s  步长={t_stride_s}s  "
          f"重力对齐={gravity_aligned}  label_mode={label_mode}  类别={classes}")
    return {
        "model": model, "classes": classes, "is_dl": is_dl,
        "gravity_aligned": gravity_aligned, "hz": t_hz,
        "window_s": t_window_s, "stride_s": t_stride_s, "label_mode": label_mode,
    }
