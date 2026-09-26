"""
Score the filled gold sheet -> KG precision with a confidence interval (Gao VLDB'19).
=====================================================================================
Reads gold_annotation.csv (the `correct` column filled with 1/0/?), computes:
  - per-source precision (1 / (1+0), '?' excluded) + verifiable rate,
  - STRATIFIED overall precision  p = sum_s W_s * p_s   (W_s = source share of the KG,
    from gold_weights.json) with a stratified-variance 95% CI,
  - a pooled Wilson 95% CI as a simple cross-check.
Only filled rows are scored; partial annotation just widens the CI.

Run:  python experiments/kg_eval/score_gold.py
"""
import csv, json, math
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
SHEET = HERE / "gold_annotation.csv"
W = json.load(open(HERE / "gold_weights.json", encoding="utf-8"))
SRC_SIZE = W["src_size"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def read_rows():
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return list(csv.DictReader(open(SHEET, encoding=enc)))
        except Exception:
            continue
    raise RuntimeError("could not read gold_annotation.csv in any known encoding")


def main():
    rows = read_rows()
    # find columns robustly
    def col(r, *keys):
        for k in r:
            if any(x in (k or "") for x in keys):
                return r[k]
        return ""
    per = defaultdict(lambda: {"1": 0, "0": 0, "?": 0})
    done = 0
    for r in rows:
        lab = (col(r, "correct") or "").strip()
        src = (col(r, "来源", "source") or "").strip()
        if lab not in ("1", "0", "?"):
            continue
        done += 1
        per[src][lab] += 1
    if done == 0:
        print("还没有已填写的行(correct 列填 1/0/?)。先填 gold_annotation.csv 再运行。")
        return

    print(f"已标注 {done} 行。\n")
    print(f"{'来源':<20}{'对':>4}{'错':>4}{'?':>4}{'精度':>8}{'可核率':>8}")
    strat_p, strat_var, wsum = 0.0, 0.0, 0.0
    pooled_k = pooled_n = 0
    for src in sorted(per, key=lambda s: -SRC_SIZE.get(s, 0)):
        c = per[src]; ver = c["1"] + c["0"]; tot = ver + c["?"]
        p = c["1"] / ver if ver else float("nan")
        vr = ver / tot if tot else 0
        ps = f"{p:.0%}" if ver else "—"
        print(f"{src:<20}{c['1']:>4}{c['0']:>4}{c['?']:>4}{ps:>8}{vr:>8.0%}")
        if ver:
            Ws = SRC_SIZE.get(src, 0)
            strat_p += Ws * p
            strat_var += (Ws ** 2) * (p * (1 - p) / ver)
            wsum += Ws
            pooled_k += c["1"]; pooled_n += ver
    if wsum:
        sp = strat_p / wsum
        se = math.sqrt(strat_var) / wsum
        lo, hi = max(0, sp - 1.96 * se), min(1, sp + 1.96 * se)
        wlo, whi = wilson(pooled_k, pooled_n)
        print("\n" + "=" * 56)
        print(f"分层加权精度(按来源在全库占比): {sp:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"合并 Wilson 精度(交叉验证):     {pooled_k}/{pooled_n}={pooled_k/pooled_n:.1%}"
              f"  95% CI [{wlo:.1%}, {whi:.1%}]")
        print(f"整体可核实率: {pooled_n}/{done} = {pooled_n/done:.0%}（其余为 '?'）")
        print("\n→ 这就是可写进论文的『人工金标准确率(分层抽样,带 95% CI)』。")


if __name__ == "__main__":
    main()
