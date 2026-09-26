# Oracle vs LLM Dispatcher (499Q full)

Oracle uses gold q['type'] → type_to_strategy; LLM uses qa_router.

| Type | n | LLM-routed | Oracle | Δ (Oracle−LLM) |
|---|---:|---:|---:|---:|
| agg_count | 50 | 0.620 | 0.620 | +0.0pp |
| agg_enum | 50 | 0.960 | 0.960 | +0.0pp |
| attr_filter | 50 | 0.920 | 0.940 | +2.0pp |
| distractor | 9 | 1.000 | 1.000 | +0.0pp |
| negation | 40 | 1.000 | 1.000 | +0.0pp |
| relation_inverse | 50 | 0.860 | 0.880 | +2.0pp |
| set_compare | 40 | 0.975 | 0.975 | +0.0pp |
| single_hop | 80 | 0.863 | 0.875 | +1.2pp |
| three_hop_chain | 30 | 0.933 | 0.800 | -13.3pp |
| two_hop_bridge | 80 | 0.887 | 0.850 | -3.8pp |
| unanswerable | 20 | 0.900 | 1.000 | +10.0pp |
| **OVERALL** | **499** | **0.886** | **0.882** | **-0.4pp** |

**Headline**: Oracle dispatcher ceiling = 88.2%, LLM dispatcher = 88.6%, gap = -0.4pp.

This isolates **dispatcher noise** (-0.4pp) from **operator+KG ceiling** (11.8pp room).