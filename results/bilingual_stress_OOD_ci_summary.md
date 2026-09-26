# Bilingual Stress Set — 95% Bootstrap CI

N = 26 questions; 1000 resamples; seed = 42.
Δ column is paired bootstrap (same resample indices for ON and OFF).

| Type | n | Bilingual ON (%) | Bilingual OFF (%) | delta(ON-OFF) (pp) |
|---|---:|---:|---:|---:|
| agg_count | 6 | 33.3 [0.0, 66.7] | 16.7 [0.0, 50.0] | +16.7 [+0.0, +50.0] |
| agg_enum | 6 | 50.0 [16.7, 83.3] | 33.3 [0.0, 66.7] | +16.7 [+0.0, +50.0] |
| attr_filter | 5 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] |
| negation | 3 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | +0.0 [+0.0, +0.0] |
| relation_inverse | 6 | 33.3 [0.0, 66.7] | 16.7 [0.0, 50.0] | +16.7 [+0.0, +50.0] |
| **OVERALL** | **26** | **38.5 [19.2, 57.7]** | **26.9 [11.5, 42.3]** | **+11.5 [+0.0, +26.9]** |

**Headline**: Bilingual layer ON vs OFF on OOD stress set = +11.5 pp [+0.0, +26.9] → *not significant at 95%*.