# -*- coding: utf-8 -*-
"""Validate the composition executor with hand-written plan trees (no LLM)."""
import json
from composition import load_executor

ex = load_executor()

CASES = [
    ("美国研制的雷达有多少款？（count ∘ constraint）",
     {"op": "count", "arg": {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"}}),

    ("美国研制 且 S 波段的雷达有多少款？（count ∘ intersect）",
     {"op": "count", "arg": {"op": "intersect", "args": [
         {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"},
         {"op": "constraint", "relation": "hasFrequencyBand", "value": "S"}]}}),

    ("列出美国研制 且 S 波段的雷达（enumerate ∘ intersect）",
     {"op": "enumerate", "arg": {"op": "intersect", "args": [
         {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"},
         {"op": "constraint", "relation": "hasFrequencyBand", "value": "S"}]}}),

    ("AN/TPY-2 的研制方的所属国（path 2-hop）",
     {"op": "path", "start": "AN/TPY-2", "chain": ["developedBy", "affiliatedTo"]}),

    ("Raytheon 研制了哪些雷达（path 反向 r^-1）",
     {"op": "enumerate", "arg": {"op": "path", "start": "Raytheon", "chain": ["developedBy^-1"]}}),

    ("美国研制 但 不在 S 波段的雷达（difference）",
     {"op": "count", "arg": {"op": "difference", "args": [
         {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"},
         {"op": "constraint", "relation": "hasFrequencyBand", "value": "S"}]}}),

    ("AN/TPY-2 是否出口到日本（contains）",
     {"op": "contains",
      "set": {"op": "path", "start": "AN/TPY-2", "chain": ["exportedTo"]},
      "member": "日本"}),

    ("AN/TPY-2 与 AN/SPY-1 研制方对比（compare）",
     {"op": "compare", "a": "AN/TPY-2", "b": "AN/SPY-1", "relation": "developedBy"}),

    ("美国雷达的平均探测距离（aggregate avg over range_km）",
     {"op": "aggregate", "func": "avg", "attribute": "range_km",
      "over": {"op": "constraint", "relation": "countryOfOrigin", "value": "美国"}}),
]


def short(v):
    if isinstance(v, set):
        v = sorted(v)
        return f"[{len(v)}] " + ", ".join(v[:6]) + (" ..." if len(v) > 6 else "")
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)


for desc, plan in CASES:
    try:
        val, tr = ex.evaluate(plan)
        print(f"\n■ {desc}")
        print(f"  result: {short(val)}")
    except Exception as e:
        print(f"\n■ {desc}\n  ERROR: {type(e).__name__}: {e}")
