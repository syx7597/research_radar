# KQA Pro High-Cardinality Count Stress Test

- Subset: Count questions with gold answer >= 30
- N = 77

## Headline
- Baseline (BM25 top-20 + LLM count): **0/77 = 0.0%**
- Strategy (exhaustive operator):     **0/77 = 0.0%**
- Delta (Strategy - Baseline):        **+0.0 pp**

## By answer-cardinality bucket

| Bucket | n | Baseline | Strategy | Delta |
|---|---:|---:|---:|---:|
| 30-50 | 41 | 0/41=0% | 0/41=0% | +0pp |
| 51-100 | 16 | 0/16=0% | 0/16=0% | +0pp |
| 101-500 | 18 | 0/18=0% | 0/18=0% | +0pp |
| 501+ | 2 | 0/2=0% | 0/2=0% | +0pp |