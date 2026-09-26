# KQA Pro End-to-End Mini Comparison

- N = 502 (stratified ≤45/KQA-Pro-fn, Find-resolvable only)
- Baseline (BM25 top-8 on per-Q 2-hop subgraph) → answerer
- Strategy (oracle-operator via KQAPRO_TO_OURS_V2 + program → args) → answerer

## Headline
- Baseline accuracy: **139/502 = 27.7%**
- Strategy (oracle-op) accuracy: **140/502 = 27.9%**
- Δ (Strategy − Baseline): **+0.2 pp**

## Per-type breakdown

| KQA Pro fn | n | Baseline | Strategy-oracle | Δ |
|---|---:|---:|---:|---:|
| QueryRelationQualifier | 45 | 9% | 11% | +2pp |
| QueryRelation | 45 | 31% | 24% | -7pp |
| VerifyStr | 45 | 53% | 53% | +0pp |
| SelectBetween | 45 | 69% | 80% | +11pp |
| QueryAttr | 45 | 20% | 20% | +0pp |
| VerifyNum | 45 | 42% | 42% | +0pp |
| VerifyYear | 45 | 20% | 20% | +0pp |
| QueryAttrQualifier | 45 | 2% | 0% | -2pp |
| Count | 45 | 31% | 22% | -9pp |
| SelectAmong | 45 | 4% | 4% | +0pp |
| What | 45 | 16% | 22% | +7pp |
| VerifyDate | 7 | 71% | 71% | +0pp |