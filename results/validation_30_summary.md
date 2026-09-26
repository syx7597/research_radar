# 30-Question Validation: Strategy-Routed GraphRAG vs Current Pipeline

**Date**: 2026-04-27
**Setup**: Retrieval-only (no LLM), top_k=20 for baseline
**KG**: 5610 triples / 28 relations

## Key results

| Question type | n | Baseline recall | Strategy recall | Δ (abs) |
|---|---|---|---|---|
| agg_count | 5 | 0.169 | 1.000 | **+83.1 pp** |
| agg_enum | 5 | 0.440 | 1.000 | **+56.0 pp** |
| negation | 10 | 0.400 (grounding) | 1.000 (correct) | **+60.0 pp** |
| attr_filter | 2 | 0.500 | 1.000 | +50.0 pp |
| two_hop_bridge | 2 | 0.833 | 1.000 | +16.7 pp |
| three_hop_chain | 1 | 1.000 | 1.000 | 0.0 pp |

Decision gate: aggregation ≥ 15% AND negation ≥ 20% → **PASSED with large margin**.

## Per-question detail (counts)

| ID | Gold | Baseline returned | Strategy |
|---|---|---|---|
| agg_count_01 | 237 | 9 | 237 ✓ |
| agg_count_02 | 36 | 5 | 36 ✓ |
| agg_count_03 | 46 | 7 | 46 ✓ |
| agg_count_04 | 14 | 4 | 14 ✓ |
| agg_count_05 | 48 | 11 | 48 ✓ |

Baseline gets **0/5 count questions correct**. The undercount is structural — top-K cannot return more than K answers regardless of question type.

## Interpretation

1. **Aggregation is broken in current GraphRAG by design**, not by tuning. The fix (exhaustive type-pool retrieval) is trivial mechanically but requires question-type detection upfront.

2. **Negation needs grounding via complement check**. 60% of negation questions returned ZERO supporting facts in baseline retrieval; LLM would hallucinate. Strategy correctly identifies absence by direct (head, relation) lookup.

3. **Multi-hop is NOT a differentiator** in this dataset — baseline already gets 83-100% recall. This means RoG-style path planning is reproducible as a baseline but should not be the headline contribution. The contribution lives in aggregation + negation.

4. **The story is not "more sophisticated retrieval"** but **"top-K retrieval is the wrong primitive for ~30% of natural KGQA questions; identify those and route around top-K"**.

## Decision

Proceed with full implementation:
- Question-type router (LLM zero-shot, 11 types)
- Exhaustive strategy for agg_count / agg_enum / relation_inverse
- Complement strategy for negation / unanswerable
- Constrained-join strategy for attr_filter
- Path-plan strategy for multi-hop (reproduced from RoG, not claimed as novelty)
- Reuse current BM25+Vector+RRF+graph for single_hop / distractor

## Threats to validity (to address before scaling)

- **Question-type router accuracy** is the system-level bottleneck. If router misclassifies single_hop as agg_enum, we waste budget; if it misclassifies agg_enum as single_hop, we lose the recall gain.
- Current evaluator scores retrieval, not LLM output. Final answer correctness depends on LLM faithfully following the structured retrieval. Need full pipeline test on 50 questions before claiming end-to-end gain.
- Validation gold answers depend on KG cleanliness. Some `affiliatedTo` triples have noise (Raytheon → 意大利, 法国). For aggregation queries this can shift counts. Expected to affect ~5% of cases at scale.
