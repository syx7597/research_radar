"""Combine all planner conditions into a 5-way per-type table:
Baseline / RoG-zeroshot(DeepSeek) / RoG-zeroshot(Qwen-7B) / RoG-FT(Qwen-7B) / Strategy.

Reads:
  results/qa500_3way_full.json   -> baseline, rog_style(DeepSeek-zs), strategy
  results/rog_qwen_zs_eval.json  -> Qwen-7B zero-shot planner
  results/rog_ft_eval.json       -> Qwen-7B fine-tuned planner
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
full = {r["id"]: r for r in json.load(open(ROOT / "results/qa500_3way_full.json", encoding="utf-8"))["per_question"]}
qzs = {r["id"]: bool(r["score"].get("correct")) for r in json.load(open(ROOT / "results/rog_qwen_zs_eval.json", encoding="utf-8"))}
ft = {r["id"]: bool(r["score"].get("correct")) for r in json.load(open(ROOT / "results/rog_ft_eval.json", encoding="utf-8"))}

by_t = defaultdict(lambda: {"b": [], "dzs": [], "qzs": [], "ft": [], "s": []})
for qid, r in full.items():
    t = r["type"]
    by_t[t]["b"].append(bool(r["baseline"]["score"].get("correct")))
    by_t[t]["dzs"].append(bool(r["rog_style"]["score"].get("correct")))
    by_t[t]["s"].append(bool(r["strategy"]["score"].get("correct")))
    by_t[t]["qzs"].append(qzs.get(qid, False))
    by_t[t]["ft"].append(ft.get(qid, False))

lines = ["# RoG 5-way: isolating fine-tuning vs base-model effect", "",
         "| Type | n | Baseline | RoG-zs(DeepSeek) | RoG-zs(Qwen7B) | RoG-FT(Qwen7B) | Strategy | Δ FT−Qwenzs |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
tot = defaultdict(int)
for t in sorted(by_t):
    d = by_t[t]; n = len(d["b"])
    b, dzs, qzs_, ft_, s = (sum(d[k]) for k in ["b", "dzs", "qzs", "ft", "s"])
    for k, v in [("n", n), ("b", b), ("dzs", dzs), ("qzs", qzs_), ("ft", ft_), ("s", s)]:
        tot[k] += v
    lines.append(f"| {t} | {n} | {b/n*100:.1f} | {dzs/n*100:.1f} | {qzs_/n*100:.1f} | "
                 f"{ft_/n*100:.1f} | {s/n*100:.1f} | {(ft_-qzs_)/n*100:+.1f} |")
n = tot["n"]
lines.append(f"| **OVERALL** | **{n}** | **{tot['b']/n*100:.1f}** | **{tot['dzs']/n*100:.1f}** | "
             f"**{tot['qzs']/n*100:.1f}** | **{tot['ft']/n*100:.1f}** | **{tot['s']/n*100:.1f}** | "
             f"**{(tot['ft']-tot['qzs'])/n*100:+.1f}** |")
out = ROOT / "results/rog_5way_summary.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved: {out}")
