"""
Score the manual annotation -> precision with 95% CI, overall and per source.
Fill the `correct(1/0)` column in annotation_sheet.csv (1=correct, 0=wrong; leave
blank to skip), then run this.

Reports precision = (#correct / #annotated) with a Wald 95% confidence interval,
overall and stratified by source (the gold accuracy number for thesis §3.5, per
Gao et al. VLDB'19). Also prints the per-source confidence the KG assigned, so you
can see whether the confidence tiers (§3.4) match measured accuracy.

Run:  python experiments/kg_eval/score_annotation.py
"""
import csv, math
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
SHEET = HERE / "annotation_facts.csv"          # the precision sample (verifiable facts)
COL = "correct(1/0/?)"


def wald_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    half = z * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0, p - half), min(1, p + half))


def load(path):
    lines = [ln for ln in open(path, encoding="utf-8-sig") if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main():
    rows = load(SHEET)
    ann = [r for r in rows if str(r.get(COL, "")).strip() in ("0", "1")]
    unver = sum(1 for r in rows if str(r.get(COL, "")).strip() == "?")
    if not ann:
        print(f"No rows annotated yet. In {SHEET.name} fill the '{COL}' column "
              f"(1=对 / 0=错 / ?=无法核实) and re-run.  ({len(rows)} rows to annotate.)")
        return

    k = sum(1 for r in ann if r[COL].strip() == "1")
    n = len(ann)
    lo, hi = wald_ci(k, n)
    print(f"Annotated: {n} judged (+{unver} marked '?' unverifiable) / {len(rows)}\n")
    print(f"OVERALL precision (extracted facts): {k/n:.1%}  "
          f"(95% CI {lo:.1%}–{hi:.1%}; {k}/{n} correct)")
    if unver:
        print(f"  unverifiable from available evidence: {unver}/{len(rows)} "
              f"= {unver/len(rows):.0%}  (a reportable finding in itself)")
    print()
    # stratify by evidence CATEGORY (TEXT/WIKI/POINTER/ATTR) — more meaningful than raw source
    by = defaultdict(lambda: [0, 0])
    for r in ann:
        c = r.get("category", "?")
        by[c][0] += int(r[COL].strip() == "1"); by[c][1] += 1
    print(f"  {'evidence category':<20}{'precision':>11}{'95% CI':>16}{'n':>5}")
    for ccat in sorted(by, key=lambda x: -by[x][1]):
        c, tot = by[ccat]
        lo, hi = wald_ci(c, tot)
        print(f"  {ccat:<20}{c/tot:>10.0%}{f'{lo:.0%}-{hi:.0%}':>16}{tot:>5}")
    print("\n  TEXT=原文证据  WIKI=维基核对  POINTER=手册页码  ATTR=属性值")
    print("  per-category precision shows which extraction channel is most reliable.")
    inf = HERE / "annotation_inference.csv"
    if inf.exists():
        irows = load(inf)
        ia = [r for r in irows if str(r.get(COL, "")).strip() in ("0", "1")]
        if ia:
            ik = sum(1 for r in ia if r[COL].strip() == "1")
            print(f"\n  [inference sheet, reported SEPARATELY] plausible: {ik}/{len(ia)} "
                  f"= {ik/len(ia):.0%} (NOT counted in extraction precision)")


if __name__ == "__main__":
    main()
