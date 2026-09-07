"""
从 pm_skin_scoring/code/questionnaire_app.py 抽出纯规则函数/常量，生成
label_service/skin_rules.py（用 ast/inspect 逐字抽，不是手抄），然后跑一遍一致性
校验：两边对同一组输入必须给同样的输出。PM 规则改了之后跑一次：

    python label_service/tools/sync_skin_rules.py
"""

import ast
import inspect
import itertools
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QA_DIR = os.path.join(REPO, "pm_skin_scoring", "code")
OUT = os.path.join(REPO, "label_service", "skin_rules.py")

CONSTS = ["DOG_NAME_OPTIONS", "IMU_DOG_DEFAULT_MAP", "SKIN_GROUP_WEIGHT", "HAIR_GROUP_WEIGHT", "C_WEIGHT",
          "SKIN_COLOR_OPTIONS", "ODOR_OPTIONS", "LESION_OPTIONS", "HAIR_SPOT_OPTIONS", "HAIR_DIAMETER_OPTIONS",
          "COAT_QUALITY_OPTIONS", "C_BASELINE_DENOM_FLOOR", "C_DELTA_TIERS", "STATS_CSV_NAME",
          "DEFAULT_IMU_STATS_ROOTS", "RECORD_COLUMNS", "WEEKLY_REPORT_COLUMNS", "W_DATE",
          "_WEEKLY_DEFAULT_CHAIN", "WEEKLY_FORM_INDICES", "WEEKLY_AUTOFILL_INDICES"]
FUNCS = ["_round_half_up", "_score_of", "_question_group_scores", "compute_score", "c_tier_of", "s_tier_of",
         "_c_delta_score_one", "_c_score_delta", "_c_score_cluster", "_c_score_persistence",
         "_c_score_interruption", "compute_c_score", "compute_s_total", "_letter_of", "_missing_questions",
         "_parse_bool", "_to_float_or_none", "_to_iso_date", "_dog_breed", "_pm_answers_to_rf_ordinals",
         "_tier_match_note", "recompute_weekly_errors", "_apply_weekly_defaults"]

HEADER = '''"""
PM 皮肤评估规则——从 pm_skin_scoring/code/questionnaire_app.py 逐字抽出来的纯函数和常量
（用 ast/inspect 自动生成，不是手抄），给 label_service 的 /api/v1/skin/* 接口用；Gradio
那边照旧用它自己的。规则有改动时重新生成：python label_service/tools/sync_skin_rules.py
（生成后会自动跑一遍两边输出一致性校验）。
"""
import math
import re

'''


def generate():
    sys.path.insert(0, QA_DIR)
    import questionnaire_app as qa  # noqa: E402

    src = open(os.path.join(QA_DIR, "questionnaire_app.py"), encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    def assign_src(name):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = []
                for t in node.targets:
                    names += [e.id for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
                if name in names:
                    return "".join(lines[node.lineno - 1:node.end_lineno])
        raise KeyError(name)

    out, seen = [HEADER], set()
    for n in CONSTS:
        s = assign_src(n)
        if s not in seen:
            seen.add(s)
            out.append(s + "\n")
    for n in FUNCS:
        out.append("\n" + inspect.getsource(getattr(qa, n)) + "\n")
    open(OUT, "w", encoding="utf-8").write("".join(out))
    return qa


def verify(qa):
    sys.path.insert(0, REPO)
    import importlib
    from label_service import skin_rules as R
    importlib.reload(R)
    rng = random.Random(0)
    n = 0
    # C 值：边界值 + 随机
    grid = [0, 1, 2, 3, 4, 5, 9, 10, 15, 20, 21, 30, 45, 60]
    for bc, tc in itertools.product([0, 3, 4, 10], grid):
        for bd, td in [(0, 0), (3, 9), (10, 30), (10, 31), (10, 40), (5, 7.5)]:
            for cl, pd_, zn, zd, ls, hb in itertools.product([0, 1, 3], [0, 1, 2, 3], [0, 6], [0, 1, 3], [False, True], [True, False]):
                a = qa.compute_c_score(bc, bd, tc, td, cl, pd_, zn, zd, ls, hb)
                b = R.compute_c_score(bc, bd, tc, td, cl, pd_, zn, zd, ls, hb)
                assert a == b, (bc, bd, tc, td, cl, pd_, zn, zd, ls, hb, a[:2], b[:2])
                n += 1
    # 问答 / S
    opts = [qa.SKIN_COLOR_OPTIONS, qa.ODOR_OPTIONS, qa.LESION_OPTIONS, qa.HAIR_SPOT_OPTIONS, qa.HAIR_DIAMETER_OPTIONS, qa.COAT_QUALITY_OPTIONS]
    for _ in range(2000):
        ans = [rng.choice(o)[0] if rng.random() > 0.1 else None for o in opts]
        hl = rng.choice(["是", "否", None])
        assert qa.compute_score(hl, *ans) == R.compute_score(hl, *ans)
        c = rng.choice([None, 0, 12, 29.9, 30, 49.9, 50, 70, 100])
        hint = rng.choice(["", "C0", "C1", "C2"])
        assert qa.compute_s_total(c, hint, hl, *ans) == R.compute_s_total(c, hint, hl, *ans)
        n += 2
    for v in [None, 0, 11.99, 12, 19.99, 20, 74.25]:
        assert qa.s_tier_of(v) == R.s_tier_of(v) and qa.s_tier_of(v, True) == R.s_tier_of(v, True)
        assert qa.c_tier_of(v) == R.c_tier_of(v)
    print(f"一致性校验通过：{n} 组输入两边输出完全一致")


if __name__ == "__main__":
    qa = generate()
    print(f"已生成 {OUT}")
    verify(qa)
