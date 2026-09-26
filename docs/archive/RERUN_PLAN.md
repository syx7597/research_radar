# KG 更新后的实验重跑计划

**版本**: v1
**日期**: 2026-04-29
**目标**: 当 KG 从 v1（5610 triples）迁移到 v2（10000 边 + 3000 属性）后，列清楚每个产物是否需要重跑、跑哪个脚本、估算时间和成本，方便决策"全量重跑 vs 部分重跑"。

---

## 1. 依赖关系图（什么依赖什么）

```
                merged_triples.json (KG)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   lexicon (固定)   生成数据集        所有实验产物
                        │               │
              ┌─────────┴─────┐
              │               │
       qa_500.json      bilingual_stress.json
              │               │
              ├──────────────┐│
              │              │
   ┌──────────┴────┐    ┌────┴───────────┐
   │               │    │                │
 主结果           ablation             双语
 e2e_30           ablation_strat       bilingual_stress
 e2e_3way         ablation_biling
 e2e_qa500_subset
 router_eval
 router_error_analysis (无 LLM, 纯分析)
 cost_latency
```

**关键观察**：所有实验产物都依赖 KG，只有 `router_eval`（题型分类）和 `BUGS.md` / `PROJECT_REPORT.md` 等文档不依赖。

---

## 2. 哪些一定要重跑 / 可以不重跑

按依赖关系判断：

### 必须重跑（KG 改变直接影响）

| 产物 | 影响原因 |
|---|---|
| `evaluation/qa_500.json` | gold answer 基于 KG 内容（agg_count 数字、enum 列表、negation 验证集合）|
| `evaluation/bilingual_stress_30.json` | 派生自 qa_500 |
| `results/qa500_3way_100_v2.json` | 三方对比的 strategy/baseline/RoG 答案都基于 KG |
| `results/ablation_strategies_100.json` | strategy 行为依赖 KG |
| `results/bilingual_stress_results.json` | 同上 |
| `results/router_error_analysis.json` | 派生自 3way 结果 |
| `results/cost_latency.json` | 包含 KG-bound 的真实运行时间 |

### 可以不重跑（KG-independent 或弱依赖）

| 产物 | 原因 |
|---|---|
| `lexicon/relations.json` | 关系 schema 是设计层，KG 内容变了关系定义不变（除非加新关系类型） |
| `lexicon/entity_aliases.json` | 别名字典是查询表层，独立于 KG 实例 |
| `lexicon/question_types.json` | 题型分类规范 |
| `results/router_30_results.json` | router 准确率只跟 prompt + 题目本身相关，跟 KG 内容无关 |
| `BUGS.md` / `PROJECT_REPORT.md` / 论文草稿 | 文档 |

### 部分重跑（要看 v2 schema 里关系类型有没有变）

如果 v2 增加了新关系类型（如 `coDeployedWith`、`hasSubsystem`），需要：

| 产物 | 必要更新 |
|---|---|
| `lexicon/relations.json` | 加入新关系的 zh_label / 同义词 / 模板 / 类型签名 |
| `qa_strategy_pipeline.py` | parser prompt 中关系列表自动同步（从 lexicon 加载，不用改代码）|
| `qa_router.py` | 同上 |
| 测试脚本 | 不动 |

---

## 3. v2 重跑总流水线（按依赖顺序）

### 阶段 0 — 数据迁移与生成（不需要 LLM）

| # | 脚本 | 用途 | 时间 | 成本 |
|---|---|---|---|---|
| 0.1 | `scripts/migrate_v1_to_v2.py`（待写）| literal→属性下沉 + 数据规范化 | 30 min | $0 |
| 0.2 | `scripts/kg_enrichment_v2.py`（待改）| 推导 coDeployedWith / competitorOf 等 | 5 min | $0 |
| 0.3 | `import_wikidata.py`（已存在，需扩展）| 厂商 / 标准 / 武器 富化 | 1-2 hr | $0 |
| 0.4 | `scripts/extract_subsystems.py`（待写）| LLM 抽取 subsystem / component | 4-6 hr | ~$2 |
| 0.5 | `scripts/generate_qa_500.py`（已有，更新 schema 适配）| 重新生成 ~500 题 | 5 min | $0 |
| 0.6 | `scripts/build_bilingual_stress.py`（已有）| 重新生成 23 题压力集 | 1 min | $0 |

**阶段 0 总时间**: 5-8 小时；**成本**: ~$2

### 阶段 1 — 实验重跑（依赖 LLM）

| # | 脚本 | 题量 | LLM 调用 | 时间 | 成本 |
|---|---|---|---|---|---|
| 1.1 | `experiments/run_e2e_30.py` | 30 (手工集) | baseline + strategy 各 30 | 10 min | <$0.05 |
| 1.2 | `experiments/run_e2e_3way.py` | 30 (手工集) | 3 系统 × 30 | 15 min | $0.05 |
| 1.3 | `experiments/run_router_eval.py` | 30 | 30 router 调用 | 3 min | <$0.01 |
| 1.4 | `experiments/run_e2e_3way_qa500.py` | 100 (qa_500 抽样) | 3 系统 × 100 | **~50 min** | $0.15 |
| 1.5 | `experiments/run_strategy_ablation.py` | ~80 | 5 ablation × 各自子集 | **~40 min** | $0.12 |
| 1.6 | `experiments/run_bilingual_stress_ablation.py` | 23 × 2 = 46 | ON/OFF 各 23 | 15 min | $0.04 |
| 1.7 | `experiments/run_router_error_analysis.py` | 0 LLM | 纯分析 1.4 输出 | 1 min | $0 |
| 1.8 | `experiments/run_cost_latency.py` | 20 × 3 系统 | timing 测量 | 8 min | $0.03 |

**阶段 1 总时间**: ~2.5 小时；**成本**: ~$0.4

### 阶段 2 — 复现产物（rescore）

不需要 LLM，纯重新打分：

| # | 脚本 | 用途 |
|---|---|---|
| 2.1 | `experiments/rescore_e2e.py` | 用最新 score_question 修正分数 |
| 2.2 | `experiments/rescore_qa500_subset.py` | 同上 |

**阶段 2 总时间**: 5 分钟；**成本**: $0

### 阶段 3 — 论文素材更新

| # | 动作 | 输入 | 输出 |
|---|---|---|---|
| 3.1 | 重生成 paper figures | 1.4 + 1.5 数据 | `paper/figures/fig{1,2,3}.{pdf,png}` |
| 3.2 | 重新捕获 traces | `experiments/capture_traces.py` | `results/appendix_c_traces.json` |
| 3.3 | 更新论文 markdown 中的数字 | 1.4-1.7 | `paper/strategy_routed_graphrag.md` |

**阶段 3 总时间**: 30 分钟；**成本**: $0

---

## 4. 三种重跑场景（按工作量）

### 场景 A — 最小复现（仅核心数字）

只重跑论文 main results 必需的：
- 阶段 0：0.1 + 0.5（数据迁移 + 重生成 qa_500）
- 阶段 1：1.4（100 题三方对比）
- 阶段 2：2.2（重打分）
- 阶段 3：3.1 + 3.3（更新图 2 + 论文中的 Table 1）

**总成本**: ~50 分钟 LLM + 30 分钟 figures + ~$0.15

适用：v2 KG 仅做小调整（如清理噪声、新增几条边），想确认核心数字没崩。

### 场景 B — 投稿前完整重跑（推荐）

在 v2 KG 完整迁移后做一遍完整跑，给论文表 1-4 重新填数：

- 阶段 0 全部
- 阶段 1 全部
- 阶段 2 全部
- 阶段 3 全部

**总成本**: ~3 小时 LLM + 阶段 0 数据迁移单独算（~6 小时）= **半天到一天**；**~$2.5**

适用：投稿前的完整复现 + 数字更新。

### 场景 C — 论文已投，v2 是 future work

什么都不做。论文 v1 数字保持，v2 KG 作为 future work / camera-ready 阶段优化项。

**总成本**: $0

适用：v1 论文数字已经成立（确实如此，94.0%），不动它。

---

## 5. 关键决策点

### 5.1 改 KG 之前必须明确的事

1. **v2 schema 是否冻结**：开始迁移前必须把 `KG_SCHEMA.md` 的实体/属性/关系定稿。中途改 schema 等于全部白做。
2. **是否同时改投稿目标**：v1 数字已经是 NAACL/EMNLP Industry 级别。如果 v2 没准备好，先投 v1 再说。

### 5.2 影响论文宏观结论的风险点

| 风险 | 概率 | 应对 |
|---|---|---|
| v2 上 baseline 显著提升（属性化让 BM25 更易召回属性）| 中 | baseline 真的变强了，反而是 GraphRAG 自洽的好事，论文叙事可调 |
| v2 上 strategy 不再大幅领先（top-K 在属性化 KG 上够用）| 中 | 这是论文核心论点的反例。需要在 paper 里 honest 报告"v2 上差距压缩到 +X pp" |
| v2 上 RoG 突然涨很多（属性化让 path planning 更精准）| 低 | 同 1，证明对手也变强 |
| 抽取出错把好关系丢了 | 中 | 阶段 0 / 1 之间加 sanity check：v1 vs v2 的 agg_count gold 必须高度相关 |

### 5.3 论文宏观叙事是否依赖 v1 specifically

不依赖。论文核心论点是"top-K 对 ~30% 题型结构性失败"——这与 KG 是 v1 还是 v2 无关。在 v2 KG 上做实验如果结论变弱，反而说明属性化也是缓解 top-K 问题的手段（这本身就是有趣发现）。

---

## 6. 推荐路径

我的建议：

1. **先用 v1 数据投稿一版**（NAACL Industry deadline 前），论文成稿不动。
2. **同步开始 v2 schema 落地**（按 KG_SCHEMA.md 的 4 阶段做）。
3. **v2 KG 完成后跑场景 B 完整重跑**。
4. **比对 v1 vs v2 结果**：
   - 如果 v2 上 strategy 仍领先 ≥+25pp：camera-ready 时升级到 v2 数字
   - 如果差距明显缩小：写一篇 follow-up paper 讨论"属性化 KG 对 strategy routing 的影响"，是新故事

---

## 7. 实操命令清单（场景 B 全量复现）

```bash
# 阶段 0：数据迁移
python scripts/migrate_v1_to_v2.py
python scripts/kg_enrichment_v2.py --infer all
python import_wikidata.py --enrich-manufacturers --enrich-standards
python scripts/extract_subsystems.py --source manuals/  # ~$2 LLM
python scripts/generate_qa_500.py
python scripts/build_bilingual_stress.py

# 阶段 1：实验
python experiments/run_e2e_30.py
python experiments/run_e2e_3way.py
python experiments/run_router_eval.py
python experiments/run_e2e_3way_qa500.py        # ~$0.15
python experiments/run_strategy_ablation.py     # ~$0.12
python experiments/run_bilingual_stress_ablation.py
python experiments/run_router_error_analysis.py
python experiments/run_cost_latency.py

# 阶段 2：重打分
python experiments/rescore_e2e.py
python experiments/rescore_qa500_subset.py

# 阶段 3：论文素材
python experiments/capture_traces.py
python scripts/make_paper_figures.py
# 手工：更新 paper/strategy_routed_graphrag.md 中的所有数字
```

预计总计：~半天工作 + ~$2.5 LLM 费用。

---

## 附录：跑实验时遇到 bug / 数据问题的应急方案

参见 `BUGS.md`，目前 10 个 bug 全部已修但都来自 v1 跑实验时碰到的。v2 上跑时可能出新 bug，建议每个阶段跑完后立刻人工抽查 5-10 个样本核对答案，遇到再加进 BUGS.md。
