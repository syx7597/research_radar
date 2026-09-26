# Router Error Analysis on QA-500 (100-Q Subset)

**Date**: 2026-04-28
**Method**: Compare router-predicted strategy vs gold `expected_strategy` for each of 100 questions. No LLM calls — pure analysis of `qa500_3way_100_v2.json`.

## Headline

| Metric | Value |
|---|---|
| Router strategy-matching accuracy | **93.0%** (93/100) |
| End-to-end answer accuracy | **94.0%** (94/100) |
| Graceful-degradation rate (when misrouted) | **85.7%** (6 of 7 misroutes recover) |

**Notable**: end-to-end accuracy (94%) > router accuracy (93%), because graceful degradation cases (6) outnumber downstream errors on correctly-routed questions (5).

## 4-way classification

| Bucket | Count | % | Interpretation |
|---|---|---|---|
| A. Routed correctly + answer correct | 88 | 88.0% | Best case |
| B. Routed correctly + answer wrong | 5 | 5.0% | Downstream issue (LLM/data) |
| C. Misrouted + answer still correct | **6** | 6.0% | **Graceful degradation** |
| D. Misrouted + answer wrong | **1** | 1.0% | **Catastrophic** |

## Strategy confusion matrix (gold → predicted)

|   | complement | constrained_join | dual_subgraph | exhaustive | lookup | path_plan |
|---|---|---|---|---|---|---|
| complement | 10 | 0 | 0 | 0 | **6** | 0 |
| constrained_join | 0 | 10 | 0 | 0 | 0 | 0 |
| dual_subgraph | 0 | 0 | 10 | 0 | 0 | 0 |
| exhaustive | 0 | **1** | 0 | 29 | 0 | 0 |
| lookup | 0 | 0 | 0 | 0 | 14 | 0 |
| path_plan | 0 | 0 | 0 | 0 | 0 | 20 |

Two misroute patterns:
- **`complement → lookup`** (6 cases, all graceful): unanswerable questions. Lookup retrieves no facts, LLM correctly answers "未知". Same final outcome as complement.
- **`exhaustive → constrained_join`** (1 case, catastrophic): `ri_016` "支持 钻地波束扫描 模式的雷达有哪些？" — router misread "支持X模式" as a multi-constraint pattern. Constrained_join needs 2+ constraints, gets only 1, returns empty.

## Per-question-type router behavior

| Type | n | Router acc | E2E acc | Misroutes | Graceful % |
|---|---|---|---|---|---|
| agg_count | 10 | 1.000 | 0.700 | 0 | n/a |
| agg_enum | 10 | 1.000 | 1.000 | 0 | n/a |
| attr_filter | 10 | 1.000 | 1.000 | 0 | n/a |
| distractor | 4 | 1.000 | 1.000 | 0 | n/a |
| negation | 10 | 1.000 | 1.000 | 0 | n/a |
| relation_inverse | 10 | 0.900 | 0.900 | 1 | 0% (1 catastrophic) |
| set_compare | 10 | 1.000 | 1.000 | 0 | n/a |
| single_hop | 10 | 1.000 | 0.800 | 0 | n/a |
| three_hop_chain | 8 | 1.000 | 1.000 | 0 | n/a |
| two_hop_bridge | 12 | 1.000 | 1.000 | 0 | n/a |
| unanswerable | 6 | **0.000** | **1.000** | 6 | **100%** |

The unanswerable row tells the cleanest story: router gets these 0/6 (always picks lookup instead of complement), but end-to-end accuracy is **6/6 = 100%** because lookup naturally produces "未知" answers when the KG yields no relevant retrieval.

## Why graceful degradation works

The pipeline benefits from a **defensive design**: when uncertain, falling through to `lookup` (BM25+Vector+RRF+Graph) is always available. The LLM then either:

1. **Synthesizes from retrieved facts** (when present) → correct answer
2. **Reports absence** (when no facts retrieved) → correct "未知" for unanswerable questions

Strategies that *cannot* gracefully degrade to lookup:
- **constrained_join**: lookup retrieves a flat list of triples, can't enforce AND-intersection. Without the strategy, attr_filter drops -80 pp (per ablation Table).
- **exhaustive**: lookup is top-K limited; can't enumerate all heads. Without the strategy, agg_count/enum drop -60 pp each.
- **path_plan**: lookup with graph expansion partially recovers, but multi-hop still drops -25 to -37 pp.

So routing failures TO these strategies (false negatives) are dangerous. Routing failures AWAY from these strategies (false positives, e.g., complement-when-not-needed) are safe.

## Paper-ready takeaways

1. **Router accuracy is high enough** (93%) to be effective without fine-tuning a dedicated classifier.
2. **System is robust to router errors**: 85.7% of misroutes still produce correct answers. Even at 70% router accuracy hypothetically, the e2e accuracy might only drop ~9 pp (since most additional misroutes would route to lookup, which usually works).
3. **The 1 catastrophic case (ri_016)** suggests a simple prompt fix: clarify that single-constraint enumeration goes to `exhaustive`, not `constrained_join`. Easy to implement; would push accuracy to ~95%.
4. **For paper Section 6.x ("Robustness"), the key claim**: "Even with imperfect routing, the lookup fallback ensures graceful degradation; only 1 of 100 questions experienced a router-induced catastrophic failure".

## Suggested router prompt fix (for follow-up)

Add to PARSER_SYSTEM:
```
- 单一约束的枚举：如果问题只有一个 (relation, tail) 约束 + 要求枚举/计数（"哪些X"/"多少款"），归 agg_enum 或 agg_count，不要归 attr_filter。attr_filter 必须有 2 个或以上独立的约束。
```

Estimated impact: +1 pp end-to-end (the ri_016 case).
