# -*- coding: utf-8 -*-
"""Quantify the auditability of generated reports.

Two metrics over a sample of equipment dossiers:
  (1) citation coverage  : fraction of deterministic facts that carry a source
                           (+ confidence). By construction this should be 100%.
  (2) summary support    : of the entity/number tokens in the LLM executive
                           summary, the fraction that trace to a cited fact
                           (the LLM-narrative faithfulness; 1.0 = no fabrication).
"""
import json, random
from pathlib import Path
from collections import Counter
from report import load_reporter

ROOT = Path(__file__).resolve().parent.parent
random.seed(11)
rp = load_reporter()
triples = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))

# sample radar entities that have a reasonable number of dossier-relevant facts
from report import DOSSIER_RELS
facts_per_head = Counter()
for t in triples:
    if t["relation"] in set(DOSSIER_RELS):
        facts_per_head[t["head"]] += 1
candidates = [h for h, c in facts_per_head.items() if c >= 4]
random.shuffle(candidates)
sample = candidates[:30]

rows = []
cov_total = cov_cited = 0
support_ratios = []
for e in sample:
    d = rp.dossier(e, summary=True)
    if not d.get("found") or not d["facts"]:
        continue
    # citation coverage: each fact tuple is (label, value, source, confidence)
    n = len(d["facts"])
    cited = sum(1 for (_l, _v, s, _c) in d["facts"] if s and s != "?")
    cov_total += n; cov_cited += cited
    f = d.get("faithful") or {}
    if f.get("checked_tokens", 0) > 0:
        support_ratios.append(f["support_ratio"])
    rows.append({"entity": d["entity"], "n_facts": n, "cited": cited,
                 "summary_support": f.get("support_ratio"), "unsupported": f.get("unsupported")})

json.dump(rows, open(ROOT / "results" / "report_faithfulness.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"sampled dossiers: {len(rows)}")
print(f"(1) 确定性事实引证覆盖率: {cov_cited}/{cov_total} = {cov_cited/max(cov_total,1)*100:.1f}%")
if support_ratios:
    avg = sum(support_ratios) / len(support_ratios)
    perfect = sum(1 for r in support_ratios if r >= 0.999)
    print(f"(2) LLM 概述可溯源率: 均值 {avg*100:.1f}% over {len(support_ratios)} 份"
          f"；完全无编造(=100%) 的报告 {perfect}/{len(support_ratios)}")
print("\n样例 unsupported（LLM 概述里未溯源的 token）：")
for r in rows[:10]:
    if r.get("unsupported"):
        print(f"  {r['entity']}: {r['unsupported']}")
