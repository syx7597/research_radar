# KQA Pro Dispatcher Probe

Cross-benchmark validation of main innovation (typology + LLM dispatcher).

## Setup
- N = 288 questions, stratified ≤ 25 per KQA Pro function type
- Dispatcher: our zero-shot LLM router (radar typology prompt), unchanged
- Gold strategy derived from KQA Pro program output via `KQAPRO_TO_OURS`
- Routing errors: 0

## Headline
- **Strategy-matching accuracy**: 176/288 = **61.1%**
- **Question-type matching**: 146/288 = **50.7%**

## Per-type breakdown (gold KQA Pro fn → strategy-match rate)

| KQA Pro fn | n | Mapped strategy | Match rate | Most-confused-with |
|---|---:|---|---:|---|
| QueryRelationQualifier | 25 | constrained_join | 0/25 (0%) | lookup(12), path_plan(7) |
| Count | 25 | exhaustive | 17/25 (68%) | complement(5), constrained_join(2) |
| QueryRelation | 25 | lookup | 19/25 (76%) | dual_subgraph(3), complement(2) |
| VerifyStr | 25 | complement | 15/25 (60%) | lookup(9), dual_subgraph(1) |
| What | 25 | lookup | 14/25 (56%) | path_plan(5), constrained_join(3) |
| SelectBetween | 25 | dual_subgraph | 25/25 (100%) | — |
| QueryAttr | 25 | lookup | 18/25 (72%) | path_plan(5), constrained_join(1) |
| VerifyNum | 25 | complement | 22/25 (88%) | constrained_join(2), dual_subgraph(1) |
| VerifyYear | 25 | complement | 25/25 (100%) | — |
| QueryAttrQualifier | 25 | constrained_join | 0/25 (0%) | lookup(10), path_plan(9) |
| SelectAmong | 25 | exhaustive | 9/25 (36%) | constrained_join(11), complement(2) |
| VerifyDate | 13 | complement | 12/13 (92%) | lookup(1) |

## Predicted strategy distribution vs gold

| Strategy | Predicted (LLM) | Gold (from KQA Pro program) |
|---|---:|---:|
| complement | 95 (33.0%) | 88 (30.6%) |
| constrained_join | 19 (6.6%) | 50 (17.4%) |
| dual_subgraph | 31 (10.8%) | 25 (8.7%) |
| exhaustive | 29 (10.1%) | 50 (17.4%) |
| lookup | 84 (29.2%) | 75 (26.0%) |
| path_plan | 30 (10.4%) | 0 (0.0%) |

## Confusion matrix (gold strategy → predicted)

| gold \\ pred | lookup | exhaustive | complement | path_plan | constrained_join | dual_subgraph |
|---|---:|---:|---:|---:|---:|---:|
| **lookup** | 51 | 0 | 6 | 11 | 4 | 3 |
| **exhaustive** | 1 | 26 | 7 | 3 | 13 | 0 |
| **complement** | 10 | 0 | 74 | 0 | 2 | 2 |
| **path_plan** | 0 | 0 | 0 | 0 | 0 | 0 |
| **constrained_join** | 22 | 3 | 8 | 16 | 0 | 1 |
| **dual_subgraph** | 0 | 0 | 0 | 0 | 0 | 25 |

## Reading the result

- Dispatcher strategy-match 61.1% on a third-party benchmark (no prompt change, no retraining) indicates the 11-way typology generalizes beyond the RadarKG-QA-499 training distribution.
- Compared to the WebQSP probe (88% all → single_hop), KQA Pro's predicted-strategy distribution is **non-degenerate**, reflecting KQA Pro's richer structural diversity.
- Where strategy-match drops, the confusion matrix shows which types the dispatcher systematically confuses — directing future prompt-engineering or typology refinements.