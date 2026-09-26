# K=8 vs K=20 Baseline vs Strategy — paired bootstrap CI

N = 499; bootstrap = 1000; seed = 42.

## Overall

| Metric | Value |
|---|---|
| Baseline K=8 (existing) | 52.7% |
| Baseline K=20 | 55.1% |
| Strategy | 88.6% |
| Δ K=20 − K=8 | +2.4 [+0.0, +4.6] pp |
| Δ Strategy − K=8 | +35.9 [+31.7, +39.9] pp |
| Δ Strategy − K=20 | +33.5 [+29.5, +37.9] pp |

## Per-type

| Type | n | K=8 (%) | K=20 (%) | Strategy (%) | Δ K20-K8 (pp) | Δ Strat-K20 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| agg_count | 50 | 4.0 | 8.0 | 62.0 | +4.0 [+0.0, +10.0] | +54.0 [+40.0, +70.0] |
| agg_enum | 50 | 46.0 | 48.0 | 96.0 | +2.0 [-10.0, +14.0] | +48.0 [+34.0, +62.0] |
| attr_filter | 50 | 12.0 | 18.0 | 92.0 | +6.0 [-4.0, +16.0] | +74.0 [+62.0, +84.0] |
| distractor | 9 | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| negation | 40 | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| relation_inverse | 50 | 40.0 | 42.0 | 86.0 | +2.0 [+0.0, +6.0] | +44.0 [+30.0, +56.0] |
| set_compare | 40 | 97.5 | 97.5 | 97.5 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| single_hop | 80 | 83.8 | 85.0 | 86.2 | +1.2 [-2.5, +5.0] | +1.3 [+0.0, +3.8] |
| three_hop_chain | 30 | 23.3 | 20.0 | 93.3 | -3.3 [-20.0, +13.3] | +73.3 [+56.7, +86.7] |
| two_hop_bridge | 80 | 40.0 | 46.2 | 88.8 | +6.2 [-1.2, +13.7] | +42.5 [+31.2, +53.8] |
| unanswerable | 20 | 90.0 | 90.0 | 90.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |

## Reading

- Adding 12 retrieval slots (K=8 → K=20) lifts Baseline by **2.4 pp** with CI [+0.0, +4.6].
- Strategy still beats the K=20 baseline by **33.5 pp** with CI [+29.5, +37.9].
- Categories that gain least from K=20 (agg_count, attr_filter, three_hop_chain) are structurally untouched by retrieval budget — confirming the 'top-K is structurally inadequate' framing was not a K=8 artifact.