"""Compute per-type median gold-answer cardinality vs Δ (Strategy − Baseline) on
RadarKG-QA-499 and KQA Pro B3, for the §4.4 cardinality-vs-Δ figure / table.

Produces a single TSV that pairs each question-type bucket with:
  - n
  - median answer cardinality
  - baseline / RoG / strategy accuracy %
  - Δ Strategy − Baseline
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "cardinality_vs_delta.tsv"


def gold_card(g):
    """Heuristic cardinality estimate from a gold answer field."""
    if g is None:
        return 1
    if isinstance(g, list):
        return max(1, len(g))
    if isinstance(g, (int, float)):
        # for agg_count, the gold IS the cardinality
        return int(g) if g >= 1 else 1
    if isinstance(g, str):
        s = g.strip()
        # try int parse (count answers)
        try:
            v = int(s)
            return v if v >= 1 else 1
        except ValueError:
            pass
        # multi-element list rendered as text
        for sep in ["; ", "、", ", ", "; "]:
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                if len(parts) > 1:
                    return len(parts)
        return 1
    return 1


def analyze_radar():
    data = json.load(open(RES / "qa500_3way_full.json", encoding="utf-8"))["per_question"]
    buckets = defaultdict(lambda: {"cards": [], "b": [], "r": [], "s": []})

    def _ok(rec):
        s = (rec or {}).get("score") or {}
        return 1 if s.get("correct") else 0

    for q in data:
        t = q.get("type", "unk")
        b = buckets[t]
        # For agg_count, use the gold integer directly as cardinality.
        gold = q.get("gold_answer")
        if t == "agg_count":
            try:
                card = int(str(gold).strip())
            except (ValueError, TypeError):
                card = gold_card(gold)
        else:
            card = gold_card(gold)
        b["cards"].append(card)
        b["b"].append(_ok(q.get("baseline")))
        b["r"].append(_ok(q.get("rog_style")))
        b["s"].append(_ok(q.get("strategy")))

    rows = []
    for t in sorted(buckets, key=lambda k: -statistics.median(buckets[k]["cards"])):
        v = buckets[t]
        n = len(v["cards"])
        rows.append({
            "benchmark": "RadarKG-499",
            "type": t,
            "n": n,
            "median_card": statistics.median(v["cards"]),
            "mean_card": round(statistics.mean(v["cards"]), 1),
            "baseline_pct": round(100 * sum(v["b"]) / n, 1),
            "rog_pct": round(100 * sum(v["r"]) / n, 1),
            "strategy_pct": round(100 * sum(v["s"]) / n, 1),
        })
        rows[-1]["delta_S_B"] = round(rows[-1]["strategy_pct"] - rows[-1]["baseline_pct"], 1)
    return rows


def analyze_kqa_pro():
    """KQA Pro B3 (LLM parser, 2-hop, full ~502 questions)."""
    path = RES / "kqa_pro_e2e_500.json"
    if not path.exists():
        return []
    raw = json.load(open(path, encoding="utf-8"))
    records = raw if isinstance(raw, list) else (raw.get("per_question") or raw.get("results") or raw.get("items") or [])
    if not records:
        return []

    buckets = defaultdict(lambda: {"cards": [], "b": [], "s": []})
    for r in records:
        t = r.get("kqapro_fn") or r.get("function") or r.get("type") or "unknown"
        gold = r.get("gold_answer") or r.get("gold")
        card = gold_card(gold)
        buckets[t]["cards"].append(card)
        buckets[t]["b"].append(1 if r.get("baseline_correct") else 0)
        buckets[t]["s"].append(1 if r.get("strategy_correct") else 0)

    rows = []
    for t in sorted(buckets, key=lambda k: -statistics.median(buckets[k]["cards"])):
        v = buckets[t]
        n = len(v["cards"])
        rows.append({
            "benchmark": "KQA-Pro-502",
            "type": t,
            "n": n,
            "median_card": statistics.median(v["cards"]),
            "mean_card": round(statistics.mean(v["cards"]), 1),
            "baseline_pct": round(100 * sum(v["b"]) / n, 1),
            "rog_pct": None,
            "strategy_pct": round(100 * sum(v["s"]) / n, 1),
        })
        rows[-1]["delta_S_B"] = round(rows[-1]["strategy_pct"] - rows[-1]["baseline_pct"], 1)
    return rows


def main():
    radar = analyze_radar()
    kqa = analyze_kqa_pro()

    cols = ["benchmark", "type", "n", "median_card", "mean_card",
            "baseline_pct", "rog_pct", "strategy_pct", "delta_S_B"]
    lines = ["\t".join(cols)]
    for row in radar + kqa:
        lines.append("\t".join("" if row.get(c) is None else str(row[c]) for c in cols))
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
