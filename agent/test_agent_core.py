# -*- coding: utf-8 -*-
"""End-to-end test of the inner composition agent (question -> plan -> execute -> answer)."""
import json
from composition import load_executor
from planner import CompositionAgent

agent = CompositionAgent(load_executor())

QS = [
    "美国研制且工作在S波段的雷达有多少款？",          # count ∘ intersect
    "列出中国研制的雷达",                              # enumerate ∘ constraint
    "Raytheon 研制了哪些雷达？",                        # path inverse
    "AN/TPY-2 的研制方属于哪个国家？",                 # path 2-hop
    "AN/TPY-2 是否出口到日本？",                        # contains
    "美国在役雷达的平均探测距离是多少？",              # aggregate
    "AN/SPY-1 和 AN/TPY-2 的研制方相同吗？",           # compare
]

for q in QS:
    try:
        r = agent.run(q)
        print(f"\nQ: {q}")
        print(f"  plan: {json.dumps(r['plan'], ensure_ascii=False)}")
        print(f"  answer: {r['answer'][:220]}")
    except Exception as e:
        print(f"\nQ: {q}\n  ERROR: {type(e).__name__}: {e}")
