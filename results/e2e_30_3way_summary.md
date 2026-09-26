# 3-Way End-to-End Comparison: Strategy-Routed vs RoG-style vs Baseline

**Date**: 2026-04-27
**Setup**: 30-question validation set, all three systems use DeepSeek as LLM (fairness control), single-pass scoring per question

## Headline table

| Type | n | Baseline | RoG-style | Strategy (ours) | Δ vs B | Δ vs RoG |
|---|---|---|---|---|---|---|
| agg_count | 5 | 0.000 | 0.400 | **1.000** | +100.0 pp | +60.0 pp |
| agg_enum | 5 | 0.200 | 0.600 | **1.000** | +80.0 pp | +40.0 pp |
| attr_filter | 2 | 0.000 | 0.000 | **1.000** | +100.0 pp | +100.0 pp |
| negation | 10 | 1.000 | 0.800 | **1.000** | 0.0 pp | +20.0 pp |
| set_compare | 2 | 1.000 | 0.500 | **1.000** | 0.0 pp | +50.0 pp |
| two_hop_bridge | 5 | 0.600 | **0.800** | **0.800** | +20.0 pp | 0.0 pp |
| three_hop_chain | 1 | 1.000 | 0.000 | **1.000** | 0.0 pp | +100.0 pp |
| **OVERALL** | **30** | **0.567** | **0.600** | **0.967** | **+40.0 pp** | **+36.7 pp** |

## Three-system comparison

### Baseline (current GraphRAG: BM25+RRF+graph expansion)
- 17/30 = 56.7%
- Wins on: negation (10/10), set_compare (2/2), three_hop (1/1)
- Catastrophic on: agg_count (0/5), attr_filter (0/2), agg_enum (1/5)
- Why: top-K retrieval fundamentally cannot enumerate >K entities. For "美国一共多少款雷达", top-8 returns 8 of 237.

### RoG-style (uniform LLM-planning + KG-walking, beam=3)
- 18/30 = 60.0% (only +3.3 pp over baseline!)
- Wins on: two_hop_bridge (4/5, +1 over baseline)
- **Loses on**:
  - **negation 10→8**: beam union pollutes the evidence (planner emits 3 paths, all walked, all in evidence; LLM can't isolate "is X in operatedBy?" cleanly)
  - **set_compare 2→1**: planner can only start from one entity; can't naturally produce dual subgraphs
  - **three_hop 1→0**: planner picked wrong chain on the only sample
- Mediocre on agg: 2/5 count, 3/5 enum — beam union over-counts. For "美国共多少款雷达" the planner emits {operatedBy⁻¹, countryOfOrigin⁻¹, exportedTo⁻¹}, walks all three, unions to 356 entities, LLM answers 356 instead of 237.

### Strategy-Routed (ours: 11 question types → 6 strategies)
- 29/30 = 96.7% (+40.0 pp over baseline, +36.7 pp over RoG-style)
- Matches or exceeds the better of the other two on every category
- Only failure: 1/5 on two_hop_bridge (mh_02 — a multi-hop question whose semantics is genuinely ambiguous between "list all radars by Raytheon" and "find Raytheon's other works")

## Why the comparison matters

This isolates the contribution: **the value is in routing, not in planning**.

- If RoG-style had matched our numbers, the contribution would just be "we ported RoG to bilingual radar KG" — incremental.
- If RoG-style had outperformed baseline by a lot (say +20 pp), then path planning would be the headline and our routing would be a marginal addition.
- Actual result: RoG-style barely beats baseline (+3.3 pp) and **loses on three categories** because it applies one strategy uniformly. Our routing gets +36.7 pp on top of RoG-style, demonstrating that **knowing when NOT to plan a path is as important as knowing how to plan one**.

## Cost comparison

LLM calls per question:
- Baseline: 1 (final answer)
- RoG-style: 2 (planner + answer)
- Strategy-Routed: 3 (router + parser + answer)

Latency:
- Baseline: ~3 s
- RoG-style: ~6 s
- Strategy-Routed: ~9 s

Cost per question (DeepSeek pricing):
- Baseline: ~$0.0003
- RoG-style: ~$0.0006
- Strategy-Routed: ~$0.0010

3× cost for +40 pp accuracy is a strongly positive trade.

## Paper-ready data

This is enough material for the experimental section's main results table. Remaining work for full paper:

1. **Scale to 500 questions** — current 30 is proof-of-concept; for statistical significance per type need ~50/type
2. **Per-strategy ablation** — turn off one strategy at a time and replace with `lookup` fallback, see drop
3. **Router error analysis** — when router misclassifies, does the wrong strategy still produce a usable answer? (graceful degradation matters)
4. **KG noise robustness** — measure how much each system is hurt by the known noisy edges (Raytheon → 意大利/法国, etc.)
5. **English-only run** — show the bilingual entity resolution layer matters (ablate the alias index)
