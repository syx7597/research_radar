"""
Bootstrap 95% confidence intervals for Table 1 main results (M5 from REVIEW_NOTES.md).

Reads results/qa500_3way_full.json (499 questions, 3-way results), resamples
1000 times with replacement, reports 95% percentile CIs for OVERALL and per-type
accuracy of Baseline / RoG / Strategy.

Writes:
  results/qa500_3way_full_ci.json     (CI per cell)
  results/qa500_3way_full_ci_summary.md  (markdown table for paper)
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

IN_PATH  = ROOT / "results" / "qa500_3way_full.json"
OUT_JSON = ROOT / "results" / "qa500_3way_full_ci.json"
OUT_MD   = ROOT / "results" / "qa500_3way_full_ci_summary.md"


def correctness_arrays(recs):
    """Return per-system, per-type boolean lists, plus overall."""
    by_t = defaultdict(lambda: {"baseline": [], "rog_style": [], "strategy": []})
    overall = {"baseline": [], "rog_style": [], "strategy": []}
    for r in recs:
        t = r["type"]
        for sys_name in ("baseline", "rog_style", "strategy"):
            c = r[sys_name]["score"].get("correct", False)
            by_t[t][sys_name].append(c)
            overall[sys_name].append(c)
    return by_t, overall


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
    lo_idx = int((1 - ci) / 2 * n_iter)
    hi_idx = int((1 + ci) / 2 * n_iter) - 1
    point = sum(bools) / n
    return point, means[lo_idx], means[hi_idx]


def bootstrap_paired_delta_ci(a, b, n_iter=N_BOOTSTRAP, ci=0.95, rng=None):
    """Paired bootstrap: sample question indices, compute Δ = mean(a) - mean(b)
    on each resample. Correctly accounts for the paired design."""
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
    lo_idx = int((1 - ci) / 2 * n_iter)
    hi_idx = int((1 + ci) / 2 * n_iter) - 1
    point = sum(a) / n - sum(b) / n
    return point, deltas[lo_idx], deltas[hi_idx]


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        recs = json.load(f)["per_question"]
    print(f"Loaded {len(recs)} records")

    by_t, overall = correctness_arrays(recs)

    out = {"per_type": {}, "overall": {}}

    # per-type CIs
    for t in sorted(by_t):
        d = by_t[t]
        n = len(d["baseline"])
        row = {"n": n}
        for sysname in ("baseline", "rog_style", "strategy"):
            p, lo, hi = bootstrap_mean_ci(d[sysname])
            row[sysname] = {"acc": p, "ci_lo": lo, "ci_hi": hi}
        # paired deltas
        p, lo, hi = bootstrap_paired_delta_ci(d["strategy"], d["baseline"])
        row["delta_s_b"] = {"value": p, "ci_lo": lo, "ci_hi": hi}
        p, lo, hi = bootstrap_paired_delta_ci(d["strategy"], d["rog_style"])
        row["delta_s_r"] = {"value": p, "ci_lo": lo, "ci_hi": hi}
        out["per_type"][t] = row

    # overall
    n = len(overall["baseline"])
    out["overall"]["n"] = n
    for sysname in ("baseline", "rog_style", "strategy"):
        p, lo, hi = bootstrap_mean_ci(overall[sysname])
        out["overall"][sysname] = {"acc": p, "ci_lo": lo, "ci_hi": hi}
    p, lo, hi = bootstrap_paired_delta_ci(overall["strategy"], overall["baseline"])
    out["overall"]["delta_s_b"] = {"value": p, "ci_lo": lo, "ci_hi": hi}
    p, lo, hi = bootstrap_paired_delta_ci(overall["strategy"], overall["rog_style"])
    out["overall"]["delta_s_r"] = {"value": p, "ci_lo": lo, "ci_hi": hi}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Markdown table
    def fmt_ci(d):
        return f"{d['acc']*100:.1f} [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"

    def fmt_delta(d):
        return f"{d['value']*100:+.1f} [{d['ci_lo']*100:+.1f}, {d['ci_hi']*100:+.1f}]"

    lines = ["# qa_500 Full 3-way + 95% Bootstrap CI", ""]
    lines.append(f"Bootstrap: {N_BOOTSTRAP} resamples with replacement, seed={SEED}.")
    lines.append("Per-type intervals are unpaired; Δ intervals are *paired* over the same resample indices.")
    lines.append("Values in % with 95% CI in brackets.")
    lines.append("")
    lines.append("| Type | n | Baseline | RoG | Strategy | delta(S-B) | delta(S-R) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in sorted(out["per_type"]):
        row = out["per_type"][t]
        lines.append(
            f"| {t} | {row['n']} | {fmt_ci(row['baseline'])} | "
            f"{fmt_ci(row['rog_style'])} | {fmt_ci(row['strategy'])} | "
            f"{fmt_delta(row['delta_s_b'])} | {fmt_delta(row['delta_s_r'])} |"
        )
    ov = out["overall"]
    lines.append(
        f"| **OVERALL** | **{ov['n']}** | **{fmt_ci(ov['baseline'])}** | "
        f"**{fmt_ci(ov['rog_style'])}** | **{fmt_ci(ov['strategy'])}** | "
        f"**{fmt_delta(ov['delta_s_b'])}** | **{fmt_delta(ov['delta_s_r'])}** |"
    )
    lines.append("")
    sb = ov["delta_s_b"]
    sr = ov["delta_s_r"]
    sig_b = "**significant**" if sb["ci_lo"] > 0 else "not significant"
    sig_r = "**significant**" if sr["ci_lo"] > 0 else "not significant"
    lines.append(f"**Headline**: Strategy vs Baseline = +{sb['value']*100:.1f} pp "
                 f"[{sb['ci_lo']*100:+.1f}, {sb['ci_hi']*100:+.1f}] → {sig_b} (CI excludes 0).")
    lines.append(f"**Headline**: Strategy vs RoG = +{sr['value']*100:.1f} pp "
                 f"[{sr['ci_lo']*100:+.1f}, {sr['ci_hi']*100:+.1f}] → {sig_r} (CI excludes 0).")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
