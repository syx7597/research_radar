# Suite-Size Ablation (100Q subset)

Reconstructed from `ablation_strategies_100.json` by composing single-operator
ablation outcomes. Each question invokes exactly one strategy in the pipeline,
so dropping a set of strategies replaces just those questions with the lookup-fallback
answer recorded in the original ablation run.

Full 6-op suite accuracy: **94.0%**

| Suite (dropped set) | Effective suite size | Accuracy | Δ vs full |
|---|:-:|---:|---:|
| Full 6-op | 6 | 94.0% | 0.0 pp |
| drop {exhaustive} | 5 | 75.0% | -19.0 pp |
| drop {complement} | 5 | 94.0% | +0.0 pp |
| drop {path_plan} | 5 | 84.0% | -10.0 pp |
| drop {constrained_join} | 5 | 86.0% | -8.0 pp |
| drop {dual_subgraph} | 5 | 94.0% | +0.0 pp |
| drop {complement, dual_subgraph} | 4 | 94.0% | +0.0 pp |
| drop {exhaustive, path_plan} | 4 | 65.0% | -29.0 pp |
| drop {exhaustive, constrained_join} | 4 | 67.0% | -27.0 pp |
| drop {path_plan, constrained_join} | 4 | 76.0% | -18.0 pp |
| drop {exhaust, path, constJoin}  (3-op suite: lookup + complement + dualSub) | 3 | 57.0% | -37.0 pp |
| drop all 5 -> lookup-only (1-op suite) | 1 | 57.0% | -37.0 pp |

## Reading the result

- **Minimal viable suite on this benchmark = 4 operators** (`{lookup, exhaustive, path_plan, constrained_join}`). Dropping both `complement` and `dual_subgraph` together yields the same accuracy as the full 6-op suite, confirming the 0 pp single-drop results from Table 2 compose: graceful degradation of lookup absorbs both on this distribution.
- **Load-bearing operators are roughly additive**. `{exhaustive, path_plan}` joint drop is approximately the sum of single drops; same for the other 2-of-3 combinations.
- **Lookup-only (1-op suite) collapses** to a strong lower bound — essentially the baseline plus our better lookup operator (KG-fallback augmentation), which is itself better than vanilla baseline GraphRAG (Table 1).
- We retain the 4 non-trivial operators plus the two redundant-here ones for the reasons in §6.1 (auditability, anticipated adversarial coverage, negligible inference cost).