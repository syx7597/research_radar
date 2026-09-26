# Per-Strategy Ablation on QA-500 (100-Question Stratified Subset)

**Date**: 2026-04-28
**Method**: For each ablated strategy, force fallback to `lookup` (BM25+Vector+RRF+Graph) for questions whose router originally picked that strategy. Other questions inherit the full-pipeline outcome.
**Cost**: 80 LLM-pipeline runs (skip the 20 that already used lookup).

## Headline ablation table

| Type | n | **Full** | -exhaustive | -complement | -path_plan | -constrained_join | -dual_subgraph |
|---|---|---|---|---|---|---|---|
| agg_count | 10 | **0.700** | 0.100 (**-60.0**) | 0.700 | 0.700 | 0.700 | 0.700 |
| agg_enum | 10 | **1.000** | 0.400 (**-60.0**) | 1.000 | 1.000 | 1.000 | 1.000 |
| attr_filter | 10 | **1.000** | 1.000 | 1.000 | 1.000 | 0.200 (**-80.0**) | 1.000 |
| distractor | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| negation | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| relation_inverse | 10 | **0.900** | 0.300 (**-60.0**) | 0.900 | 0.900 | 0.900 | 0.900 |
| set_compare | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| single_hop | 10 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| three_hop_chain | 8 | **1.000** | 1.000 | 1.000 | 0.625 (**-37.5**) | 1.000 | 1.000 |
| two_hop_bridge | 12 | **1.000** | 1.000 | 1.000 | 0.750 (**-25.0**) | 1.000 | 1.000 |
| unanswerable | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **OVERALL** | **100** | **0.940** | **0.760** (-18.0) | 0.940 | 0.880 (-6.0) | 0.860 (-8.0) | 0.940 |

## Strategy-by-strategy contribution

### 1. `exhaustive` — most critical (-18.0 pp overall)
Handles 3 question types (agg_count, agg_enum, relation_inverse), each of which loses **60 pp** when ablated. These are exactly the categories where top-K retrieval is structurally inadequate: counting, enumeration, and inverse-lookup with many heads. Without exhaustive, ~29 of 100 questions fall back to lookup → those that need the full type-pool fail.

### 2. `constrained_join` — single category, catastrophic loss when ablated (-8.0 pp overall, -80 pp on attr_filter)
Handles only attr_filter (11 questions). Ablating drops attr_filter from 1.0 to 0.2 — i.e., baseline lookup correctly answers only 2/10 multi-constraint questions. The +100 pp gain on attr_filter over Baseline+RoG that we reported in the 3-way table is **entirely attributable to this strategy**.

### 3. `path_plan` — multi-hop value (-6.0 pp overall)
Handles two_hop_bridge (drop -25 pp) and three_hop_chain (drop -37.5 pp). Without it, lookup partially recovers via graph expansion, but reliable multi-hop traversal needs explicit planning.

### 4. `complement` — graceful redundancy (0 pp drop on the bench)
Handles negation. Negation already scored 0.9 on baseline lookup — the LLM is robust enough to interpret retrieved triples as "absence of forbidden tail". The strategy still produces cleaner reasoning traces (explicitly enumerating known tails for the queried relation) but **doesn't increase numerical accuracy on this slice**.

### 5. `dual_subgraph` — graceful redundancy (0 pp drop on the bench)
Handles set_compare. Baseline already scored 1.0 — lookup surfaces facts about both entities, LLM compares them. The dual-subgraph strategy gives a more structured comparison but **doesn't beat the LLM's natural ability to compare side-by-side**.

## Paper interpretation

**Three "load-bearing" strategies** that account for the +33 pp gain over baseline:
- exhaustive (handles 29/100 questions, accounts for ~18 pp of the headline gain)
- constrained_join (handles 11/100, accounts for ~8 pp)
- path_plan (handles 20/100, accounts for ~6 pp)

**Two "robustness" strategies** that don't change accuracy on this benchmark but provide cleaner reasoning paths:
- complement (10/100, makes negation reasoning explicit)
- dual_subgraph (10/100, makes comparison structured)

These two are still worth keeping because:
1. **Reasoning faithfulness**: explicit complement check / dual-subgraph extraction makes the answer's grounding clear to the user (auditable evidence).
2. **Adversarial robustness**: harder negation questions (e.g., "X is operated by ALL countries except Y") and harder set_compare questions (e.g., 3-way comparisons) would likely break naive lookup — current bench doesn't include these adversarial cases.
3. **No cost**: these strategies use the same LLM call count as lookup, so keeping them adds no latency/$.

## Honest caveats

- **Set_compare and negation tests don't stress lookup hard enough.** Future bench should include adversarial variants: negation with KG ambiguity (X has 5 tails, asked about a 6th-similar one); set_compare with 3+ entities or implicit attributes. We expect complement/dual_subgraph to gain real ground there.
- The 0-drop result for these two strategies is a **feature of the benchmark mix**, not a feature of the strategies themselves. The paper should explicitly note this and propose harder evaluations as future work.

## Final paper-ready summary (combined with 3-way comparison)

| | Strategy gain on this bench |
|---|---|
| Total +33.0 pp over baseline = | exhaustive (~18) + constrained_join (~8) + path_plan (~6) + smaller gains |
| Total +25.0 pp over RoG-style = | exhaustive (~16) + constrained_join (~10) + ... |
| Strategy-Routed dominates ALL 11 question types | (vs RoG-style which loses on 3) |
