# KQA Pro End-to-End Mini Comparison

- N = 139 (stratified ≤12/KQA-Pro-fn, Find-resolvable only)
- Baseline (BM25 top-8 on per-Q 2-hop subgraph) → answerer
- Strategy (oracle-operator via KQAPRO_TO_OURS_V2 + program → args) → answerer

## Headline
- Baseline accuracy: **40/139 = 28.8%**
- Strategy (oracle-op) accuracy: **42/139 = 30.2%**
- Δ (Strategy − Baseline): **+1.4 pp**

## Per-type breakdown

| KQA Pro fn | n | Baseline | Strategy-oracle | Δ |
|---|---:|---:|---:|---:|
| QueryRelationQualifier | 12 | 8% | 17% | +8pp |
| QueryRelation | 12 | 17% | 17% | +0pp |
| VerifyStr | 12 | 50% | 50% | +0pp |
| SelectBetween | 12 | 75% | 75% | +0pp |
| QueryAttr | 12 | 33% | 33% | +0pp |
| VerifyNum | 12 | 50% | 50% | +0pp |
| VerifyYear | 12 | 17% | 17% | +0pp |
| QueryAttrQualifier | 12 | 8% | 0% | -8pp |
| Count | 12 | 17% | 25% | +8pp |
| SelectAmong | 12 | 8% | 8% | +0pp |
| What | 12 | 8% | 17% | +8pp |
| VerifyDate | 7 | 71% | 71% | +0pp |