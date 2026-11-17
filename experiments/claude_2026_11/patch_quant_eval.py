"""multiscale_exp.py 加 --quant-eval：评估（留出文件连续推理）时，原始输入先按有损上传的量化方案处理
（加速度 1/Q_ACC g、陀螺仪 1/Q_GYR °/s），训练不变。用来看有损压缩上传对识别有没有影响。"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = """            Xc_, st_c, _ = cw
            Xm_c, _ = T.featurize(Xc_, hz)"""
new = """            Xc_, st_c, _ = cw
            if a.quant_eval:
                Xc_ = Xc_.copy()
                Xc_[:, :, :3] = np.round(Xc_[:, :, :3] * a.quant_eval) / a.quant_eval
                if Xc_.shape[2] >= 6:
                    Xc_[:, :, 3:6] = np.round(Xc_[:, :, 3:6] * 2.0) / 2.0
            Xm_c, _ = T.featurize(Xc_, hz)"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
old2 = '    p.add_argument("--fine", default="", help="单独训、再并回去的细类，逗号分隔")'
new2 = old2 + '\n    p.add_argument("--quant-eval", type=float, default=0, help="评估时加速度量化到 1/N g（陀螺仪 0.5 °/s），0 = 不量化")'
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
print("ok")
