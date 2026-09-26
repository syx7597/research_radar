# -*- coding: utf-8 -*-
"""Scan the KG for numeric attributes and their coverage, to decide which
attributes the aggregate/calculator operator can reliably use."""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
triples = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))

NUM = re.compile(r"-?\d+(?:\.\d+)?")
radar_types = {"Radar", "RadarSystem"}

# per relation: distinct heads, distinct heads with numeric tail, sample tails
rel_heads = defaultdict(set)
rel_num_heads = defaultdict(set)
rel_samples = defaultdict(list)
radar_entities = set()

for t in triples:
    h, r, ta, ht = t["head"], t["relation"], t["tail"], t.get("head_type", "")
    if ht in radar_types:
        radar_entities.add(h)
    rel_heads[r].add(h)
    if NUM.search(str(ta)):
        rel_num_heads[r].add(h)
        if len(rel_samples[r]) < 5:
            rel_samples[r].append(str(ta))

n_radar = len(radar_entities)
print(f"total radar entities: {n_radar}\n")
print(f"{'relation':24s} {'#heads':>7s} {'#numeric':>9s} {'num%':>6s}  samples")
rows = []
for r in rel_heads:
    nh = len(rel_heads[r]); nn = len(rel_num_heads[r])
    if nn >= 10:  # candidate numeric attribute
        rows.append((nn, r, nh, nn, rel_samples[r]))
for nn, r, nh, _, samp in sorted(rows, reverse=True):
    frac = nn / max(nh, 1) * 100
    cov_radar = len(rel_num_heads[r] & radar_entities) / max(n_radar, 1) * 100
    print(f"{r:24s} {nh:>7d} {nn:>9d} {frac:>5.0f}%  radar-cov={cov_radar:.0f}%  e.g. {samp[:3]}")
