# -*- coding: utf-8 -*-
"""G3 硬门:属性过滤空集的成因分布(CORE-1 覆盖率归因的地基验证)。

问题:σ 返回空集时,能否用属性覆盖率把成因分成
  COVERAGE_GAP(覆盖缺失,该走文本补全) vs OVER_CONSTRAINED(约束过严,该走松弛)?
若两类都有可观占比 → 覆盖率归因有真实分布支撑,CORE-1 成立;
若几乎全一类 → 归因无区分对象,CORE-1 降级(按 v2 §10 退路)。

只读 kg_v3,不改任何数据。输出分布表 + 判定。
"""
import json
import re
import sys
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KGV3 = ROOT / "kg_v3"

THETA_LO = 0.30   # 覆盖率低于此 → 该键为"覆盖缺失"嫌疑(仅用于报告全局覆盖率)
MIN_TRUST = 10    # 两键齐备雷达 < 此 → 数据太少,空集判 COVERAGE_GAP;≥ 此且空 → OVER_CONSTRAINED
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(raw):
    """从 value_raw 抽第一个数(区间取首值);抽不到返回 None。"""
    if raw is None:
        return None
    m = NUM_RE.search(str(raw).replace(",", ""))
    return float(m.group()) if m else None


def main():
    N = json.loads((KGV3 / "entities.json").read_text(encoding="utf-8"))
    rad = [n for n in N if n.get("type") == "Radar"]
    total = len(rad)

    # 每个雷达 -> {key: numeric_value}(只留能解析出数值的属性)
    ent_vals = []
    key_count = defaultdict(int)
    for n in rad:
        vals = {}
        for a in n.get("attributes", []):
            if a.get("unmapped"):          # 自由文本描述键,归文本通道,不进属性索引
                continue
            v = parse_num(a.get("value_raw"))
            if v is not None:
                k = a.get("attr")
                vals[k] = v
                key_count[k] += 1
        ent_vals.append(vals)

    # 取数值属性最密的 top 键
    num_keys = [k for k, c in sorted(key_count.items(), key=lambda x: -x[1]) if c >= 100][:8]
    cover = {k: key_count[k] / total for k in num_keys}

    print(f"=== 雷达实体 {total} | 数值键(cover=有该键的雷达/全体) ===")
    for k in num_keys:
        print(f"  {k:16} n={key_count[k]:4}  cover={cover[k]:.0%}")

    # 每个键的数值分布(分位阈值)
    def quantile(k, q):
        xs = sorted(v[k] for v in ent_vals if k in v)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    # 模拟真实的两键合取过滤:key1 > qA ∧ key2 > qB,阈值取多档分位
    rng = random.Random(42)
    QS = [0.4, 0.6, 0.8, 0.95, 1.05]   # 1.05 = 超过最大值,强制过严
    empties = {"COVERAGE_GAP": 0, "OVER_CONSTRAINED": 0, "AMBIG": 0}
    total_q = 0
    examples = {"COVERAGE_GAP": [], "OVER_CONSTRAINED": []}

    for k1, k2 in combinations(num_keys, 2):
        for qa in QS:
            for qb in QS:
                total_q += 1
                # 阈值(1.05 分位用 max*1.05 近似"超过最大")
                def thr(k, q):
                    xs = sorted(v[k] for v in ent_vals if k in v)
                    return xs[-1] * 1.05 if q > 1.0 else quantile(k, q)
                t1, t2 = thr(k1, qa), thr(k2, qb)
                # 结果集:同时有两键且都满足
                res = [i for i, v in enumerate(ent_vals)
                       if k1 in v and k2 in v and v[k1] >= t1 and v[k2] >= t2]
                if res:
                    continue
                # 空集 → 归因。判据 = 这条查询实际可判的数据量 n_both(两键齐备的雷达数),
                # 不是全局覆盖率分数(所有键覆盖率都<31%,全局判据会恒判 COVERAGE_GAP,失真)。
                n_both = sum(1 for v in ent_vals if k1 in v and k2 in v)
                if n_both < MIN_TRUST:
                    empties["COVERAGE_GAP"] += 1
                    if len(examples["COVERAGE_GAP"]) < 3:
                        examples["COVERAGE_GAP"].append(
                            f"{k1}>{t1:.0f} ∧ {k2}>{t2:.0f} | 两键齐备雷达={n_both}<{MIN_TRUST}(数据太少,空集不可信判无解)")
                else:
                    empties["OVER_CONSTRAINED"] += 1
                    if len(examples["OVER_CONSTRAINED"]) < 3:
                        examples["OVER_CONSTRAINED"].append(
                            f"{k1}>{t1:.0f} ∧ {k2}>{t2:.0f} | 两键齐备雷达={n_both}(有充足数据但无一满足→约束过严)")

    tot_empty = sum(empties.values())
    print(f"\n=== 模拟两键合取过滤 {total_q} 条,其中空集 {tot_empty} 条 ===")
    for cause, c in empties.items():
        pct = c / tot_empty if tot_empty else 0
        print(f"  {cause:16} {c:4}  ({pct:.0%})")
    print("\n--- COVERAGE_GAP 例 ---")
    for e in examples["COVERAGE_GAP"]:
        print("   ", e)
    print("--- OVER_CONSTRAINED 例 ---")
    for e in examples["OVER_CONSTRAINED"]:
        print("   ", e)

    # 预注册判定
    cg = empties["COVERAGE_GAP"] / tot_empty if tot_empty else 0
    oc = empties["OVER_CONSTRAINED"] / tot_empty if tot_empty else 0
    print("\n=== G3 判定(预注册规则) ===")
    minor = min(cg, oc)
    if minor >= 0.15:
        print(f"  ✅ 两类都有可观占比(次多类 {minor:.0%} ≥ 15%)→ 覆盖率归因有真实区分对象,CORE-1 成立")
    else:
        dom = "COVERAGE_GAP" if cg > oc else "OVER_CONSTRAINED"
        print(f"  ⚠️ 几乎全为 {dom}({max(cg,oc):.0%})→ 二分归因区分度弱")
        print(f"     但若主导为 COVERAGE_GAP,'不完整性主导空集'这一结论本身强化了覆盖补全机制;")
        print(f"     二分叙事需按 §10 退路收缩为'覆盖率触发的文本补全',而非'三分归因'。")


if __name__ == "__main__":
    main()
