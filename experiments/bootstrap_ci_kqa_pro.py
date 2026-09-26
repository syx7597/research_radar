"""
Bootstrap 95% CI for the KQA Pro 500-Q end-to-end comparison.

Reads results/kqa_pro_e2e_500.json (~500 questions with baseline/strategy
correctness). Computes:
  - Per-system accuracy with bootstrap CI (unpaired)
  - Paired Δ(Strategy − Baseline) with bootstrap CI
  - Per-type breakdown

Writes results/kqa_pro_e2e_500_ci_summary.md
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_BOOTSTRAP = 1000
SEED = 42

import sys as _sys
_SUFFIX = _sys.argv[1] if len(_sys.argv) > 1 else ""  # "" → v1, "_sgv2" → v2
IN_PATH  = ROOT / "results" / f"kqa_pro_e2e_500{_SUFFIX}.json"
OUT_MD   = ROOT / "results" / f"kqa_pro_e2e_500{_SUFFIX}_ci_summary.md"


def bootstrap_mean_ci(bools, n_iter=N_BOOTSTRAP, ci=0.95, rng=None):
    if not bools:
        return 0.0, 0.0, 0.0
    rng = rng or random.Random(SEED)
    n = len(bools)
    means = []
    for _ in range(n_iter):
        sample = [bools[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_iter)]
    hi = means[int((1 + ci) / 2 * n_iter) - 1]
    return sum(bools)/n, lo, hi


def bootstrap_paired_delta(a, b, n_iter=N_BOOTSTRAP, ci=0.95, rng=None):
    rng = rng or random.Random(SEED + 1)
    assert len(a) == len(b)
    n = len(a)
    deltas = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idx) / n
        mb = sum(b[i] for i in idx) / n
        deltas.append(ma - mb)
    deltas.sort()
    lo = deltas[int((1 - ci) / 2 * n_iter)]
    hi = deltas[int((1 + ci) / 2 * n_iter) - 1]
    return sum(a)/n - sum(b)/n, lo, hi


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        recs = json.load(f)["per_question"]
    print(f"Loaded {len(recs)} questions")

    by_fn = defaultdict(lambda: {"b": [], "s": []})
    overall = {"b": [], "s": []}
    for r in recs:
        b = bool(r.get("baseline_correct", False))
        s = bool(r.get("strategy_correct", False))
        fn = r.get("kqapro_fn", "?")
        by_fn[fn]["b"].append(b)
        by_fn[fn]["s"].append(s)
        overall["b"].append(b)
        overall["s"].append(s)

    def fmt_ci(d):
        return f"{d[0]*100:.1f} [{d[1]*100:.1f}, {d[2]*100:.1f}]"

    def fmt_d(d):
        return f"{d[0]*100:+.1f} [{d[1]*100:+.1f}, {d[2]*100:+.1f}]"

    lines = ["# KQA Pro 500-Q E2E + 95% Bootstrap CI", ""]
    lines.append(f"N = {len(recs)}; {N_BOOTSTRAP} resamples; seed = {SEED}.")
    lines.append("Δ column is paired bootstrap (same resample indices for Baseline and Strategy).")
    lines.append("")
    lines.append("| KQA Pro fn | n | Baseline (%) | Strategy (%) | delta (pp) |")
    lines.append("|---|---:|---:|---:|---:|")
    for fn in sorted(by_fn, key=lambda k: -len(by_fn[k]["b"])):
        d = by_fn[fn]
        n = len(d["b"])
        b_ci = bootstrap_mean_ci(d["b"])
        s_ci = bootstrap_mean_ci(d["s"])
        dlt  = bootstrap_paired_delta(d["s"], d["b"])
        lines.append(f"| {fn} | {n} | {fmt_ci(b_ci)} | {fmt_ci(s_ci)} | {fmt_d(dlt)} |")
    n_total = len(overall["b"])
    ob = bootstrap_mean_ci(overall["b"])
    os_ = bootstrap_mean_ci(overall["s"])
    dlt = bootstrap_paired_delta(overall["s"], overall["b"])
    lines.append(f"| **OVERALL** | **{n_total}** | **{fmt_ci(ob)}** | **{fmt_ci(os_)}** | **{fmt_d(dlt)}** |")
    lines.append("")
    sig = "**significant**" if dlt[1] > 0 else "*not significant at 95%*"
    lines.append(f"**Headline**: Strategy − Baseline = +{dlt[0]*100:.1f} pp "
                 f"[{dlt[1]*100:+.1f}, {dlt[2]*100:+.1f}] → {sig}.")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    safe = "\n".join(lines).replace("−", "-")
    print(safe)
    print(f"\nSaved: {OUT_MD}")


if __name__ == "__main__":
    main()
