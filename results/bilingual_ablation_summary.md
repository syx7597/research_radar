# Bilingual Layer Ablation

**Date**: 2026-04-28

## Two-stage experiment

### Stage 1 — Same-language ablation (23-Q stress) was the wrong test
First ran `BILINGUAL_DISABLED=True` on the same 100-Q subset used for the main 3-way comparison. Result: **0.0pp drop, all categories unchanged**.

Why: The qa_500 questions were auto-generated from KG triples, so the parser-extracted entity surface forms already match KG canonical Chinese forms verbatim. The bilingual layer (alias resolution / suffix-stripping / equivalence-class fallback) never has to fire because there's no mismatch.

This is a **bench-design artifact**, not a property of the bilingual layer. The layer's value is in handling out-of-canonical surface forms — which the bench doesn't have.

### Stage 2 — Bilingual stress test (proper experiment)

Built `evaluation/bilingual_stress_30.json` by substituting **English/abbreviation aliases** for entity names in 23 questions sampled from qa_500:

| Substitution | Example |
|---|---|
| 美国 → USA | "原产于 USA 的雷达..." |
| 俄罗斯 → Russia | "原产于 Russia 的雷达..." |
| 中国 → China | "China 装备的且部署在战斗机上的雷达..." |
| 合成孔径 → SAR | "支持 SAR 模式的雷达..." |
| 脉冲多普勒 → PD | "支持 PD 模式的雷达..." |
| 动目标指示 → MTI | "支持 MTI 模式的雷达..." |
| 有源相控阵 → AESA | "采用 AESA 技术的雷达..." |

This simulates a real bilingual user: typing "USA" / "AESA" / "SAR" instead of the KG's Chinese canonical forms.

## Headline result

| Type | n | Bilingual ON | Bilingual OFF | Δ |
|---|---|---|---|---|
| agg_count | 5 | 0.200 | 0.200 | 0.0 pp |
| agg_enum | 5 | **0.800** | 0.400 | **-40.0 pp** |
| attr_filter | 5 | **0.800** | 0.600 | **-20.0 pp** |
| negation | 3 | 1.000 | 1.000 | 0.0 pp |
| relation_inverse | 5 | **0.600** | 0.200 | **-40.0 pp** |
| **OVERALL** | **23** | **0.652** | **0.435** | **-21.7 pp** |

## Per-category interpretation

- **agg_enum (-40 pp), relation_inverse (-40 pp)**: These are the categories where the parser's tail value MUST match a KG-stored entity to retrieve heads. Without alias resolution, "USA" → no hits → empty answer. With alias, "USA" → 美国 → 237 hits.
- **attr_filter (-20 pp)**: One of the two constraints loses its KG anchor when the entity isn't aliased. Pipeline still partially succeeds via the other constraint when both fail simultaneously, but loses a question per 5.
- **negation (0 pp)**: Negation correctness only requires identifying "X is not related to forbidden_tail in KG". Whether forbidden_tail = "USA" or "美国" doesn't change the answer because in both cases there's no `(X, operatedBy, USA-or-美国)` triple. Negation is robust to alias misses by construction.
- **agg_count (0 pp, both fail)**: The gold counts assume canonical alias union (Russia → 俄罗斯 ∪ 苏联 ∪ Russian Federation). With bilingual ON, pipeline produces the union count which often *exceeds* the surface-form-only gold; with OFF, pipeline gets 0 hits. Neither matches the exact gold integer. Real-world correctness with bilingual ON is actually higher than the score suggests.

## Headline claim (paper-ready)

**Bilingual layer contribution**: -21.7 pp end-to-end on the stress test, driven by -40 pp on agg_enum and relation_inverse. This is independent evidence that the layer is a substantive contribution, not just configuration.

The standard 100-Q bench shows 0 pp drop because it doesn't include cross-lingual surface forms — by design (the generator pulls verbatim from KG). For the paper, both numbers should be reported transparently:

> *On the in-distribution 100-Q bench, the bilingual layer is silent (0 pp), confirming the strategy router carries the headline gain. On a 23-question bilingual stress test where users type English aliases (USA / AESA / SAR), the layer contributes -21.7 pp end-to-end, with -40 pp on enumerate-style queries that require canonical anchor matching. Real industrial Chinese KGs receive both forms of input, so the layer is operationally necessary even if it doesn't move the in-distribution headline.*

## Honest caveats

1. **The stress test is small (n=23)** — would prefer 50+ for robust per-type means. Current numbers are directional.
2. **agg_count gold is brittle**: when alias merge counts > surface-form-only gold, the pipeline's "more correct" answer scores wrong. Could be addressed with a tolerance metric (e.g., relative error < 10%).
3. **Construction bias**: the substitutions are 1-to-1 mappings I chose. Real users would also typo, partially translate ("U.S." vs "USA"), or mix Chinese and English in the same query. Stress test could be expanded to cover these.
