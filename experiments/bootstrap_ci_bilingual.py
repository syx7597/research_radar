"""
Bootstrap 95% CI for the bilingual stress-set ablation (m1).

Reads results/bilingual_stress_OOD_results.json (26 questions, ON vs OFF).
Computes paired bootstrap CI for the Δ(ON − OFF) on the 26-Q OOD set,
both per-type and overall.

Writes results/bilingual_stress_OOD_ci_summary.md.
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

IN_PATH  = ROOT / "results" / "bilingual_stress_OOD_results.json"
OUT_MD   = ROOT / "results" / "bilingual_stress_OOD_ci_summary.md"


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
    print(f"Loaded {len(recs)} OOD stress questions")

    by_t = defaultdict(lambda: {"on": [], "off": []})
    overall = {"on": [], "off": []}
    for r in recs:
        t = r["type"]
        c_on  = r["on"]["score"].get("correct", False)
        c_off = r["off"]["score"].get("correct", False)
        by_t[t]["on"].append(c_on)
        by_t[t]["off"].append(c_off)
        overall["on"].append(c_on)
        overall["off"].append(c_off)

    def fmt_ci(d):
        return f"{d['acc']*100:.1f} [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"
    def fmt_delta(d):
        return f"{d['value']*100:+.1f} [{d['ci_lo']*100:+.1f}, {d['ci_hi']*100:+.1f}]"

    rows = []
    for t in sorted(by_t):
        d = by_t[t]
        n = len(d["on"])
        p_on,  lo_on,  hi_on  = bootstrap_mean_ci(d["on"])
        p_off, lo_off, hi_off = bootstrap_mean_ci(d["off"])
        delta, lo_d, hi_d     = bootstrap_paired_delta(d["on"], d["off"])
        rows.append({
            "type": t, "n": n,
            "on":  {"acc": p_on,  "ci_lo": lo_on,  "ci_hi": hi_on},
            "off": {"acc": p_off, "ci_lo": lo_off, "ci_hi": hi_off},
            "delta_on_off": {"value": delta, "ci_lo": lo_d, "ci_hi": hi_d},
        })
    n_total = len(overall["on"])
    p_on,  lo_on,  hi_on  = bootstrap_mean_ci(overall["on"])
    p_off, lo_off, hi_off = bootstrap_mean_ci(overall["off"])
    delta, lo_d, hi_d     = bootstrap_paired_delta(overall["on"], overall["off"])
    ov = {
        "n": n_total,
        "on":  {"acc": p_on,  "ci_lo": lo_on,  "ci_hi": hi_on},
        "off": {"acc": p_off, "ci_lo": lo_off, "ci_hi": hi_off},
        "delta_on_off": {"value": delta, "ci_lo": lo_d, "ci_hi": hi_d},
    }

    lines = ["# Bilingual Stress Set — 95% Bootstrap CI", ""]
    lines.append(f"N = {n_total} questions; {N_BOOTSTRAP} resamples; seed = {SEED}.")
    lines.append("Δ column is paired bootstrap (same resample indices for ON and OFF).")
    lines.append("")
    lines.append("| Type | n | Bilingual ON (%) | Bilingual OFF (%) | delta(ON-OFF) (pp) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(f"| {row['type']} | {row['n']} | {fmt_ci(row['on'])} | "
                     f"{fmt_ci(row['off'])} | {fmt_delta(row['delta_on_off'])} |")
    lines.append(f"| **OVERALL** | **{ov['n']}** | **{fmt_ci(ov['on'])}** | "
                 f"**{fmt_ci(ov['off'])}** | **{fmt_delta(ov['delta_on_off'])}** |")
    lines.append("")
    d = ov["delta_on_off"]
    sig = "**significant**" if d["ci_lo"] > 0 else "*not significant at 95%*"
    lines.append(f"**Headline**: Bilingual layer ON vs OFF on OOD stress set = "
                 f"+{d['value']*100:.1f} pp [{d['ci_lo']*100:+.1f}, {d['ci_hi']*100:+.1f}] → {sig}.")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nSaved: {OUT_MD}")


if __name__ == "__main__":
    main()
