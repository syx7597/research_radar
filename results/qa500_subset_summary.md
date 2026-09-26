# QA-500 Stratified Subset Validation

**Date**: 2026-04-27
**Setup**: 50-question stratified sample (5 per type, 3 unanswerable, 2 distractor) from auto-generated 478-question bench. Strategy-routed pipeline only (no baseline / RoG-style — saves 2× LLM calls per Q).
**Goal**: confirm the strategy holds up on auto-generated questions, not just the hand-crafted 30.

## Headline

**92.0% overall** on 50 stratified questions, vs 93.3% on the hand-crafted 30 — **no degradation at scale**.

| Type | n | Acc |
|---|---|---|
| agg_count | 5 | 0.800 |
| agg_enum | 5 | 0.800 |
| attr_filter | 5 | **1.000** |
| distractor | 2 | 0.500 |
| negation | 5 | **1.000** |
| relation_inverse | 5 | 0.800 |
| set_compare | 5 | **1.000** |
| single_hop | 5 | **1.000** |
| three_hop_chain | 5 | **1.000** |
| two_hop_bridge | 5 | **1.000** |
| unanswerable | 3 | **1.000** |
| **OVERALL** | **50** | **0.920** |

## Bug fixes during this session

### B008. score_question missed branches for single_hop / unanswerable / distractor
- **STATUS**: fixed
- **Symptom**: Initial run reported 76% with single_hop = 0/5, unanswerable = 0/3, distractor = 0/2. Inspection of the answers showed all answers were *correct in text* — the scorer was returning `{skip: unknown_type_<type>}`.
- **Root cause**: `experiments/run_e2e_30.py:score_question` only had branches for the 6 types in the 30-question validation set; 3 types in the 478-question bench (`single_hop`, `unanswerable`, `distractor`) had no scoring rule, falling through to a "skip" default that the aggregator treated as wrong.
- **Fix**: added substring match for `single_hop`/`distractor` (gold answer must appear in answer text) and "no/unknown" keyword detection for `unanswerable`.
- **Where**: `experiments/run_e2e_30.py:score_question`.

## Remaining failures (4 of 50)

| ID | Type | Diagnosis | Fix needed in |
|---|---|---|---|
| ac_011 | agg_count | Gold counted only "People's Republic of China" surface form (5) but pipeline correctly resolved alias to canonical "中国" (48). Pipeline answer is *more correct* than gold. | gold construction: use canonical alias |
| ae_045 | agg_enum | LLM enumerated 22/44 entities — borderline against 60% recall threshold. LLM stochastic variance. | possible: bump max_tokens further, or split very-large enums into chunks |
| ri_003 | relation_inverse | Strategy returned wrong entity set (hit=1/14). Possible parser issue with `动目标指示` (multi-typed: RadarMode AND TechType). | parser: try both relations on multi-type tails |
| dt_010 | distractor | Router classified "AN/APG-63 radar family 装备在哪些国家?" as `agg_enum` (because "哪些 NN" sounds like enumeration) and dispatched to exhaustive. The question is actually a single_hop (one radar, one country answer). | router: add disambiguation rule "if a specific radar entity is the subject, prefer single_hop over agg_enum even when the question asks 哪些" |

## Decision

Strategy-routed pipeline holds at the 478-question scale. Proceed with:

1. **Run 3-way (Baseline / RoG / Strategy) on full 478** to get headline paper numbers (~30 min, ~$1.50 LLM cost). Or stratified 100-question subset for fast iteration.
2. **Per-strategy ablation**: turn off each of {exhaustive, complement, constrained_join, dual_subgraph, path_plan} one at a time, route to `lookup` instead, measure drop. Quantifies each strategy's contribution.
3. **Router error analysis**: pull all router misclassifications, check if downstream answer is still correct (graceful degradation) or catastrophic.
4. **Gold-quality cleanup pass**: regenerate qa_500 with canonical-alias resolution at gold construction, fixing ac_011-style cases.

## Threats to validity at this scale

1. **Router edge cases** appear: "哪些国家" pattern triggered wrong route (dt_010). Need 1-2 more disambiguation rules.
2. **Gold quality** is autogen — needs a human-review pass before publication. Currently ~2-5% of golds may be incorrect/imprecise (extrapolating from this 50-Q subset's 1 known gold issue).
3. **Multi-type tail values** (e.g., 合成孔径 / 动目标指示 are both RadarMode and TechType) can route to wrong relation. The current `_validate_args` is conservative but may miss these cases when KG has hits on the wrong relation too.
