# KQA Pro End-to-End Mini Comparison

- N = 502 (stratified ≤45/KQA-Pro-fn, Find-resolvable only)
- Baseline (BM25 top-8 on per-Q 2-hop subgraph) → answerer
- Strategy (oracle-operator via KQAPRO_TO_OURS_V2 + program → args) → answerer

## Headline
- Baseline accuracy: **142/502 = 28.3%**
- Strategy (oracle-op) accuracy: **141/502 = 28.1%**
- Δ (Strategy − Baseline): **-0.2 pp**

## Per-type breakdown

| KQA Pro fn | n | Baseline | Strategy-oracle | Δ |
|---|---:|---:|---:|---:|
| QueryRelationQualifier | 45 | 11% | 9% | -2pp |
| QueryRelation | 45 | 24% | 18% | -7pp |
| VerifyStr | 45 | 53% | 53% | +0pp |
| SelectBetween | 45 | 76% | 78% | +2pp |
| QueryAttr | 45 | 20% | 22% | +2pp |
| VerifyNum | 45 | 42% | 42% | +0pp |
| VerifyYear | 45 | 20% | 20% | +0pp |
| QueryAttrQualifier | 45 | 0% | 0% | +0pp |
| Count | 45 | 36% | 36% | +0pp |
| SelectAmong | 45 | 4% | 7% | +2pp |
| What | 45 | 18% | 18% | +0pp |
| VerifyDate | 7 | 71% | 71% | +0pp |