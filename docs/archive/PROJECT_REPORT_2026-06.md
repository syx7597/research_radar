> 历史资料：保留当时的方案与结果，不代表当前完成度。当前方向见 `/docs/RESEARCH_DIRECTION.md`，进度见 `/docs/CURRENT_STATUS.md`。

# 雷达 GraphRAG 项目工作报告

**最后更新**: 2026-06-09（毕设系统化：把单次问答升级为**雷达情报分析 Agent**。双层架构落地在 `agent/`：内层「类型化算子组合执行器 + LLM 计划器」(把论文 composability 做成真东西，LLM 出 JSON 计划树→确定性求值，实体不漂移可审计)，外层「手写 ReAct 编排」(工具=组合核心/装备档案/对比报告/web_search)。可溯源情报报告每条事实挂来源+置信度(吃 KG 的 source/confidence/evidence)；图谱未收录型号(如 AN/SPY-1)经 Wikipedia 抽取+别名层实体链接补全并标"未核验"；关键数字由确定性 trace 生成、LLM 不进信任路径。**组合评测 72 题正确率 99%**(单算子做不了的聚合/差集/嵌套交集)。Streamlit Demo UI 已验证。详见 `agent/README.md`。下一步：反思节点 / 报告 faithfulness 评测 / NL2SPARQL 对照(已设计 `experiments/nl2sparql/PLAN.md`)。）

**前一更新**: 2026-06-08（v12.2 → v12.3：服务器联网后补**原版 RoG 基座 LLaMA-2-7B-Chat**。下载 modelscope/Llama-2-7b-chat-ms，同套流程 LoRA 微调（q/k/v/o_proj，max_len 2304 应对中文 token 膨胀，grad_ckpt 防 OOM）+ LLaMA-2-zs 对照。**两基座 7-way**：LLaMA-2 26.5→55.9（微调效应 +29.5 [+24.8,+34.1]）、Qwen 31.5→61.7（+30.3）、Strategy 88.6（vs LLaMA-FT +32.7 [+28.1,+37.3]、vs Qwen-FT +26.9）。**结论对基座鲁棒**：原版 LLaMA-2 + 强中文 Qwen 两基座微调都被 Strategy 领先 +27~33pp，彻底堵死"弱 baseline"+"非原版 RoG"两个攻击点。论文 §4.6/Table 11/Appendix I 升级为 7-way。）

**前一更新**: 2026-06-04（v12.1 → v12.2：补**忠实微调 RoG** 实验（reviewer 最强攻击点）。在服务器（4×4090，无外网）用缓存的 Qwen-7B-Chat LoRA 微调一个 RoG-style planner（240 条 KG 采样路径，与 499 题无重叠），只替换 planner、walker+DeepSeek reasoner 不变。加跑 Qwen-7B zero-shot 对照隔离 confound。**结果（5-way, n=499）**：RoG-zs(DeepSeek) 60.3 / RoG-zs(Qwen) 31.5 / **RoG-FT(Qwen) 61.7** / Strategy 88.6。微调效应 +30.3pp [+25.9,+34.7]（排除弱 baseline 嫌疑），但 **Strategy 仍领先最强 RoG +26.9pp [+22.6,+31.3]**。关键洞察：微调让 planner 逐题型重新发明 exhaustive 算子（agg_count 2→64 追平 Strategy），但单路径范式**无法表达交集**（attr_filter 28 vs 92）。写入论文 §4.6 (RQ6) + Appendix I + Limitations/Future Work 更新。脚本 `experiments/rog_finetune/`，结果 `results/rog_*`，详见 `experiments/rog_finetune/RESULTS.md`。论文 19 页（temp 编译验证 0 undefined；正式 PDF 待查看器关闭后重建）。）

**前一更新**: 2026-06-01（v12 → v12.1：从 `strategy_routed_graphrag_v5_revised.md` 回移补全。加 §2.3 "no formal completeness" hedge 避免过度宣称；Limitations 5 → 8 条；新加 §7 Future Work 整节（5 条）+ Artifact Release Plan；Appendix 从 9 行占位扩成 9 章实质内容（dispatcher prompt / cost decomposition / OOD dict / KG construction pipeline 等）；RoG fairness caveat 强化到 §4 setup / §6 Related Work / §6.5 Limitations 三处。详见 `paper/REVIEW_NOTES.md` v12→v12.1 段。）

**前一更新**: 2026-05-29（v11 → v12：朋友 advice 触发的 framing 重构。从"分类器 + 派发"叙事切换到"answer-geometry mismatch + 信息需求匹配的 operator suite"。Oracle Δ −0.4pp 提前到 abstract 直接堵"this is just a classifier"的质疑；6 operator 从"empirical"提升为"derived from information-requirement classes"。详见 `paper/REVIEW_NOTES.md` v11→v12 段。）

本文档是项目的导航地图——从原始动机到当前论文核心数据，列出每个 Python 文件、每个实验的目的与结果，便于追溯和复现。

---

## 0. 项目背景与转向

### 0.1 起点
原项目是"多源雷达语料 → 知识图谱 → 问答系统"。早期路线 **OntoCom** 试图在 KG 补全（KGE）层做创新（RotatE + TCNS + ORE + SAPC 三件套）。

### 0.2 OntoCom 路线被废弃的理由
- **消融实验自相矛盾**：裸 RotatE backbone MRR 0.2718 → 加全部三件套 0.2539（**降 6.6%**）
- **"+25% MRR vs PyKEEN RotatE"** 的数字其实是来自训练循环工程改动，与 TCNS/ORE/SAPC 三模块**无关**
- **创新性不足**：TCNS = type-constrained negative sampling（2019 已发表）；ORE = cluster + hierarchy L2（HAKE 同构）；SAPC = path mining（PTransE 同构）
- 文献已饱和：RoG / ToG / ToG-2 / KGMRF-LRPM / RDPG / RAR / Plan-Then-Retrieve

### 0.3 新路线：Strategy-Routed GraphRAG
**核心论点**：top-K 检索范式在结构上无法解决 ~30% 的自然 KGQA 问题（聚合/否定/多约束）。我们把问题分成 11 类，路由到 6 种检索原语（lookup / exhaustive / complement / path_plan / constrained_join / dual_subgraph）。

---

## 1. 基础设施层（lexicon/）

| 文件 | 内容 | 用途 |
|---|---|---|
| `lexicon/relations.json` | 28 关系的双语字典：zh_label / 同义词 / question_templates / 三元组中英渲染模板 / multi_valued 标记 / head_types & tail_types 签名 | parser prompt 构造、双语三元组渲染、问题→关系反向匹配 |
| `lexicon/entity_aliases.json` | 40 国 + 25 模式 + 14 体制 + 16 功能 + 7 舰艇级别的双语别名表 | 跨语言实体解析（USA→美国、AESA→有源相控阵 等）|
| `lexicon/question_types.json` | 11 题型 × 6 策略 + 路由 prompt 模板 + 决策门规范 | 题型路由 + 测试时分配检索策略 |
| `lexicon/__init__.py` | 加载器 + `build_alias_index()` (614 surface forms) + `build_relation_lookup()` (129 关键词) + `render_triple_text()` 双语三元组渲染 | 所有上层组件的依赖入口 |

---

## 2. 核心系统层

| 文件 | 角色 | 关键导出 |
|---|---|---|
| `qa_router.py` | LLM zero-shot 题型分类器 | `QuestionRouter.route(question) → (qtype, strategy, raw)` |
| `qa_strategy_pipeline.py` | 主 pipeline：router → parser → executor → answerer，6 个策略执行器 | `StrategyPipeline.run(question, ablate_strategy=None)` |
| `qa_rog_baseline.py` | RoG-style 基线：planner（带 `^-1` 反向遍历）+ KG walker + LLM reasoner | `run_rog_baseline(question, kg)` |
| `graphrag_retriever.py` | 现有 GraphRAG（BM25 + 向量 + RRF + 图扩展），用作 baseline 与 lookup fallback | `HybridRetriever.retrieve()` |
| `qa_pipeline.py` | 项目原有问答流水线（保留作历史对比） | — |

---

## 3. 数据集

| 文件 | 规模 | 用途 |
|---|---|---|
| `graphrag_index/merged_triples.json` | 5610 三元组 / 28 关系 / 3384 节点 | 全部实验的底层 KG |
| `evaluation/validation_30.json` | 30 题（手工策划，10 聚合 + 10 否定 + 10 多跳）| Section 5.1 的精挑细选验证集 |
| `evaluation/qa_500.json` | 478 题（自动生成 + canonical 修复，11 类型分层）| Section 5.2 主结果集 |
| `evaluation/bilingual_stress_30.json` | 23 题（替换中文实体为英文 alias）| Section 5.4 双语层 ablation |

---

## 4. 实验脚本与结果一览

### 4.1 数据生成

| 脚本 | 目的 | 结果 |
|---|---|---|
| `scripts/generate_qa_500.py` | 从 5262 干净三元组（filtered from 5610）按 11 题型自动生成 478 题，每题带 canonical alias 合并的金标 | 478 题（target 500，three_hop_chain KG 中链稀少只有 8 题）|
| `scripts/build_bilingual_stress.py` | 替换 qa_500 中的中文实体为英文 alias（USA / AESA / SAR / TWS 等）造双语压力测试集 | 23 题，覆盖 5 类型 |

### 4.2 Pipeline 验证（递进式）

| 脚本 | 目的 | 关键数字 |
|---|---|---|
| `experiments/run_validation_30.py` | **检索层 only** 验证：在 30 题上对比 baseline top-K vs 我们的策略检索（不调 LLM）| `agg_count` baseline recall 16.9% → strategy 100%；`agg_enum` 44% → 100%；`negation` baseline grounded 40%。决策门通过 |
| `experiments/run_router_eval.py` | LLM zero-shot router 在 30 题上的分类准确率 | 28/30 = **93.3%**；4 关键题型（agg_*, negation, attr_filter）全 ≥ 80% |
| `experiments/run_e2e_30.py` | **端到端** 30 题：baseline + strategy 全 LLM pipeline | strategy 93.3% vs baseline 60.0%（+33.3pp）|
| `experiments/rerun_failures.py` + `rescore_e2e.py` | 修复 alias suffix bug 和 _extract_int 后重测 | 修完后 strategy 仍 93.3% |
| `experiments/run_e2e_qa500_subset.py` | qa_500 自动生成集 50 题分层 strategy 单跑（不跑 baseline 节省 token）| strategy 92.0%（与手工集 93.3% 一致）|
| `experiments/rescore_qa500_subset.py` | 加上 single_hop / unanswerable / distractor 的评分分支后重打分 | 76% → 92% |

### 4.3 三方主对比（论文 Section 5.2 主结果表）

| 脚本 | 目的 | 关键数字 |
|---|---|---|
| `experiments/run_e2e_3way.py` | 30 题手工集上 baseline + RoG-style + strategy 三方对比 | strategy 96.7% / RoG 60.0% / baseline 56.7% |
| `experiments/run_e2e_3way_qa500.py` | qa_500 上 100 题分层（10/类型）三方对比 | strategy 91.0% / RoG 69.0% / baseline 61.0% |
| `experiments/rerun_3way_failures.py` | 修复 lookup KG-fallback + two_hop_bridge generator filter 后只重跑 strategy 失败的题 | **strategy 94.0%**（RoG 69.0% / baseline 61.0% 不变），strategy 在所有 11 题型 ≥ 二者 |
| `experiments/run_e2e_3way_qa500_full.py` ★ | **qa_500 全 499 题三方对比**（带 checkpoint，63 min, ~$0.20）| **strategy 88.6% / RoG 60.3% / baseline 52.7%**（+35.9pp / +28.3pp）—— 与 100Q 比 strategy 优势反而扩大 |
| `experiments/run_oracle_dispatcher.py` ★★ | **Oracle dispatcher 实验**（gold qtype 替代 LLM 路由，18 min）| **Oracle 88.2% vs LLM 88.6% → Δ −0.4pp**。意外发现：LLM dispatcher 反而稍好；three_hop_chain 上 Oracle **输 13.3pp**（LLM 把冗余 3-hop 重路由到 exhaustive 反而更准）。证明 +35.9pp 全部来自 operator suite，dispatcher noise ≈ 0 |
| `experiments/bootstrap_ci.py` ★ | 1000 次 paired bootstrap 计算 Table 1 的 95% CI | Strategy vs Baseline: **+35.9 [+31.7, +40.1]** 显著；Strategy vs RoG: **+28.3 [+23.8, +32.7]** 显著；Strategy 输给 RoG 的 2 题型（two_hop_bridge / unanswerable）CI 都跨 0 → 实为打平 |
| `experiments/bootstrap_ci_bilingual.py` ★ | 26 Q OOD bilingual stress 加 CI | Δ(ON−OFF) = **+11.5pp [+0.0, +26.9]** — **CI 下界正好 = 0**，临界显著；诚实写进 §5.3 |
| `experiments/suite_size_ablation.py` ★ | 用现有 single-op ablation 数据组合，分析 multi-op 抓与 minimal-suite | **4-op suite = 6-op suite = 94.0%**（complement / dual_subgraph 联合冗余）；load-bearing 操作**精确可加**（-19 + -10 = -29）；lookup-only 57% 是地板（main result +35.9pp 分解为 +4.3pp 改良 lookup + +37pp 三个 load-bearing） |
| `experiments/webqsp_dispatcher_probe.py` ★★ | WebQSP 246Q 跑 dispatcher（无 e2e scoring，~3 min, $0.05）| 0 错误：dispatcher 在英文上工作；**88.2% 落入 single_hop，0% 进 multi-hop / set-compare / attr-filter** → 反向证据 WebQSP 缺少 Strategy 设计的题型，不是合适的对比基准 |
| `experiments/kqa_pro_dispatcher_probe.py` ★★★ | **KQA Pro 288Q stratified probe**（cross-benchmark validation of main innovation，~6 min, $0.07）| **routing match 67.4%（v2 mapping）**；dual_subgraph 100% / complement 84% / lookup 68% — 三个 operator 几乎完美命中。distribution 非退化，6 operators 全被激活。与 WebQSP 全塌 single_hop 形成强对比 |
| `experiments/kqa_pro_probe_rescore.py` | v1 mapping 错把 QueryAttrQualifier 标 constrained_join；修正为 path_plan 后 routing match 从 61.1% → 67.4% | 实验设计问题，不是 dispatcher 失败 |
| `experiments/kqa_pro_e2e.py` ★★★ | **KQA Pro 端到端 mini comparison（B1 mechanical args 版）**（139 题，~6 min）| **Baseline 30.9% / Strategy 28.8%，Δ = −2.2pp**。诚实负面结果——揭示 cross-domain transfer 是 2 阶段问题 |
| `datasets/kqa_pro_parser.py` ★★ | **KQA Pro–aware LLM parser**（60 关系 + 30 属性 + 通用 prompt，~1 天工作）| 替代 radar parser 处理 KQA Pro questions；sanity test 4/4 题输出合理（Totoro vs Arendt 正确识出 primary+secondary+duration）|
| `experiments/kqa_pro_e2e.py` v2 ★★★★ | **KQA Pro e2e B2: 真 LLM parser 公平对比**（同 139 题，~10 min, $0.18）| **Baseline 28.8% / Strategy 30.2%，Δ = +1.4pp** ✓——换 parser 一项操作把 Δ 从 −2.2 翻到 +1.4（净 +3.6pp 位移）。Count 上 −8pp → +8pp（per-type 16pp 翻转）|
| `experiments/kqa_pro_e2e.py` B3 ★★★★★ | **KQA Pro e2e 500 题大样本验证**（n=502, ~30 min, $0.60）| **Baseline 27.7% [23.7, 31.7] / Strategy 27.9% [23.9, 31.7] / Δ = +0.2pp [−1.6, +2.2]，统计上平局**。B2 的 +1.4pp 揭示为小样本噪声。**但 per-type 出现两个统计显著真信号**：SelectBetween **+11.1pp [+2.2, +22.2]** ✓（dual_subgraph 跨域真赢），Count **−8.9pp [−17.8, −2.2]** ✗（2-hop 子图无法枚举概念类，主创新被数据层限制而非算法层）。这把 next step 从模糊指向"子图扩展"硬钉到具体题型 |
| `experiments/bootstrap_ci_kqa_pro.py` | 502 题 paired bootstrap 95% CI | overall 不显著；SelectBetween / Count 显著（CI 严格远离 0）|
| `datasets/kqa_pro_subgraph.py::build_question_subgraph_v2` ★ | 子图扩展：2-hop + 概念类成员枚举（FilterConcept 包含的所有 entities）| Q1 PA 县计数从 0 三元组 → 718（包含 39 个 PA 县）|
| `experiments/kqa_pro_e2e.py` B4 ★★★ | KQA Pro e2e 502 题 + concept-expanded 子图 | **Baseline 28.3% / Strategy 28.1% / Δ = −0.2pp [−2.4, +2.0]**。Count 从 −9pp 修到 +0pp ✓；但 baseline 也跟着涨，所以总差距没拉开。**揭示 RadarKG +35.9pp 的真实来源**：不是 "operator 设计 vs 不设计"，而是 "answer set 大到 baseline top-K 装不下" |
| `experiments/run_baseline_k20.py` ★★★★ | **RadarKG K=20 baseline ablation**（499 题，~13 min, $0.06）| **K=8 52.7% / K=20 55.1%（+2.4pp）/ Strategy 88.6%**。加 retrieval budget 几乎没用——agg_count 4%→8%（结构性失败）/ three_hop_chain 23%→20%（变差）。**Strategy 仍 +33.5pp [+29.5, +37.9] 显著赢 K=20 baseline**——证明 +35.9pp 不是 K=8 artifact，是真实算法增益 |
| `experiments/k20_paired_ci.py` | K=8 / K=20 / Strategy 三方 paired bootstrap CI | Strategy − K=20 = +33.5 [+29.5, +37.9] 显著；per-type 上 agg_count Strategy 比 K=20 baseline 还赢 +54pp |
| `datasets/kqa_pro_subgraph.py` | 每题 2-hop topic-entity 子图提取器（避免对 362K 全图建索引）| ~70% val 题在 KB 中可 resolve topic entity；FindAll-only 题（Count/SelectAmong）需全 KB 枚举，超出当前子图作用域 |

### 4.4 论文 Section 5.3 Ablation 表

| 脚本 | 目的 | 关键数字 |
|---|---|---|
| `experiments/run_strategy_ablation.py` | 逐个关掉 5 个策略（替换为 lookup），看每个策略的边际贡献 | exhaustive **-18pp**（最关键） / constrained_join **-8pp**（attr_filter -80pp）/ path_plan **-6pp**（multi-hop -25~37pp）/ complement 0pp（graceful redundancy）/ dual_subgraph 0pp（同上）|
| `experiments/run_bilingual_ablation.py` | 在 100 题集上关掉 bilingual layer（alias / suffix-strip / 等价类 / type-validation）| **0pp 掉分**（bench 设计问题——qa_500 用 KG canonical 表面形式生成，没用上双语层）|
| `experiments/run_bilingual_stress_ablation.py` | 在 23 题双语压力集上关 bilingual layer | **-21.7pp**（agg_enum -40pp，relation_inverse -40pp）；解释为何前一个实验是 0pp 的方法论贡献 |

### 4.5 论文 Section 6 Robustness & Cost

| 脚本 | 目的 | 关键数字 |
|---|---|---|
| `experiments/run_router_error_analysis.py` | 不跑 LLM，直接分析 100 题中 router 误分类的 case | router 准确率 93%；端到端 94% **高于 router 准确率**；graceful degradation 率 85.7%；只 1 题 catastrophic |
| `experiments/run_cost_latency.py` | 20 题计时，估算 USD 成本 | baseline 2.3s/$0.00011 / RoG 3.3s/$0.00023 / strategy 5.6s/$0.00041；strategy 性价比 **$0.00094/pp**（比 RoG $0.00151/pp 更高效）|

---

## 5. 论文支撑数据汇总

### 5.1 三大主张及其证据

| 主张 | 数据来源 | 数字 |
|---|---|---|
| **主创新：Strategy-Routed > 单一策略（full 499Q）★** | `qa500_3way_full.json` + `qa500_3way_full_ci_summary.md` | **Strategy 88.6% [86.2, 91.2]** vs RoG 60.3% [55.5, 64.9] vs Baseline 52.7% [48.3, 57.1]；paired Δ vs Baseline **+35.9 [+31.7, +40.1]**；vs RoG **+28.3 [+23.8, +32.7]**；CI 严格 > 0 → 统计显著 |
| 100Q 验证（与 full 一致性）| `qa500_3way_100_v2.json` | Strategy 94.0% / RoG 69.0% / Baseline 61.0%（strategy 优势在 full 反而扩大）|
| **每个策略的硬贡献** | `ablation_strategies_100.json` | exhaustive -18pp / constrained_join -8pp / path_plan -6pp |
| **副创新：双语层是真贡献** | `bilingual_stress_results.json` | -21.7pp on stress set；-40pp on agg_enum |
| **★ 新主张：+35.9pp 来自 operator suite，不来自分类器** | `oracle_dispatcher_full_summary.md` | Oracle ceiling 88.2% vs LLM 88.6% → **dispatcher noise = -0.4pp**（LLM 反而稍好）。回应 reviewer M3 攻击"多少来自路由 vs operator?"——答案：全部 operator |

### 5.2 论文每节用什么数据（v5 layout）

```
Section 1 Introduction          → e2e_30_3way_summary 证明 baseline 范式失败
Section 3 Method                → lexicon/ + qa_router.py + qa_strategy_pipeline.py
Section 4 Dataset               → qa_500.json + bilingual_stress_30.json
Section 5.1 Main results (full) → qa500_3way_full_ci_summary  ★ 表 1（499 题 + 95% CI）
                                  qa500_3way_100 作为 sanity check 段
Section 5.2 Ablation            → ablation_strategies_100_summary  ★ 表 2
Section 5.3 Bilingual layer     → bilingual_ablation_summary  ★ 表 3
Section 5.4 Dispatcher robustness → oracle_dispatcher_full_summary  ★★ 表 4 新增
                                    + router_error_analysis_summary（93% 路由准确率）
Section 5.5 Cost & latency      → cost_latency.json  ★ 表 5（用 +35.9pp 重算 $0.00084/pp）
Section 6.1 Why keep 0pp ops    → 软化为务实理由（auditability / adversarial / cost）
Section 6.2 Single-policy fail  → 用 single_hop -72.5pp / attr_filter 0% 说事
Section 6.3 Bilingual paradox   → 同前
Section 6.4 Extension           → "operator suite" 而非 "algebra"；去 Codd 强类比
Section 6.5 Limitations         → 新增 single-domain self-built bench、simplified-RoG、suite size 三条
Section 7   Future Work         → ★ 新增整节：8 条具体后续承诺
Section 8   Conclusion          → 更新数字 + 自承约束作为研究方向
```

---

## 6. 已知 Bug 与修复全记录

详见 `docs/BUGS.md`。共 10 个：

| ID | 类别 | 状态 |
|---|---|---|
| B000 | PDF 抽取 false-positive (ARINC-429) | 已修 |
| B001 | KG 噪声（Raytheon→意大利/法国）| 数据问题，未修代码 |
| B002 | `_extract_int` 正则在 Unicode 模式下漏匹配"237款" | 已修 |
| B003 | parser tail "S波段" 不匹配 KG "S" | 已修（_alias_resolve_tail 加 suffix-strip）|
| B004 | answerer max_tokens 截断 44-entity 枚举 | 已修（400→1200）|
| B005 | parser 选 type-incompatible 关系（developedBy 美国）| 已修（3 层 prompt+validate+equiv）|
| B006 | "合成孔径" multi-type alias index 错选 | 已修（multi-type 兼容）|
| B007 | mh_07 gold 用 countryOfOrigin 但 KG 子图分离 | 已修（gold 改用 operatedBy）|
| B008 | scorer 缺 single_hop / unanswerable / distractor 分支 | 已修 |
| B009 | lookup 用 BM25 漏召回 "MM/SPQ-2" 这种带特殊符的型号 | 已修（KG-fallback）|
| B010 | gen_two_hop_bridge 允许 Manufacturer 当 head | 已修（head_type filter）|

---

## 6.5 论文 v4 → v5 自审修订（2026-05-22）

模拟 NAACL/EMNLP Industry Track 严格审稿人评议后落实的 6 项修改。详细 review 见 `paper/REVIEW_NOTES.md`。

| 审稿问题 ID | 处理 | 影响 |
|---|---|---|
| **M2** RoG 是稻草人（zero-shot ≠ 原 RoG 微调）| 全文重命名 "RoG-style" → "Zero-shot Path Planner (RoG-inspired)"；§5.1 加 explicit 声明"This is not a faithful RoG reproduction" + 解释为何用 zero-shot 变体；Limitation 单列一条 | 免被指责造假对比 |
| **M3** 缺 oracle dispatcher 上界 | `run_oracle_dispatcher.py` 跑全 499Q；新表 4 + §5.4 重写 | **意外发现：LLM dispatcher 反而比 oracle 稍好**（−0.4pp 但 LLM 在 three_hop_chain 上 +13.3pp）。彻底回应"多少来自路由 vs operator"的质疑 |
| **M4** Algebraic closure 是循环论证 | 保留 algebra 作为论文 hook（title 不动），删全文 "closed cover" / "design closure" / "identity element"；§6.1 改名 "Why Keep the 0pp-Drop Operators?" 给三条务实理由（auditability / adversarial / cost）；§6.4 去掉 Codd 强类比 | 去掉一个攻击面，论文更工程派 |
| **M5** 无统计显著性 | `bootstrap_ci.py` 跑 1000 次 paired bootstrap；Table 1 全部加 95% CI | 主结果都 significant；Strategy 输的 2 题型 CI 跨 0 → 实为打平 → **9 wins + 2 ties，反而是更强主张** |
| **m10** 缺 Future Work | 新增 §7 整节，8 条具体后续（真 RoG / WebQSP / oracle / 对抗集 / suite ablation / 跨域 / 延迟 / A/B） | 给审稿人 roadmap |
| **m1** WebQSP 公开 benchmark | 评估 ~2 天工作（KG ingest + relation 字典 + 别名 + scorer 全要改），当前 scope 外；明确写进 §7.2 + REVIEW_NOTES | 诚实承诺，不糊弄 |

**自评分**（NAACL Industry 1-5）：v4 borderline-reject → v5 **weak accept / borderline accept**。Soundness 2.5→3.5（oracle + CI + 自承限制）、Empirical 3.5→4。

## 6.6 论文 v5 → v6 二轮自审修订（2026-05-25）

| 审稿问题 ID | 处理 | 数据/工具 |
|---|---|---|
| **m1** Bilingual stress CI | bootstrap_ci_bilingual.py 跑出 +11.5pp [+0.0, +26.9] | **诚实承认 CI 下界 = 0**，临界显著；不掩饰 |
| **m2** Suite-size ablation | suite_size_ablation.py 复用 single-op 数据组合 | 4-op = 6-op = 94.0%；load-bearing 精确可加 |
| **m6** Cost-per-pp vs RoG | 文字加算 ($0.00041−$0.00023)/28.3 = **$0.00064/pp** | 比 vs-Baseline 的 $0.00084 还便宜 |
| **m7** Artifact release plan | §7.1 新章节：Apache-2.0 / CC-BY-4.0 / 数据列表 / 复现 $1.50 | — |
| **M1+M2 部分** WebQSP probe | webqsp_dispatcher_probe.py 在 246Q 上跑 dispatcher，0 错误 | **88.2% → single_hop, 0% → multi-hop/filter**。强论点：WebQSP 不是合适对比基准 |

**自评分 v6**：v5 weak accept → **clear accept / borderline strong accept**。新增点：
- bilingual CI 显示作者愿意诚实报告非显著结果（反而增强可信度）
- suite-size ablation 直接消除"6 操作怎么定的"攻击面
- WebQSP probe 反向论证为何当前 benchmark 必要

## 6.7 论文 v6 → v7：KQA Pro 跨域实验（2026-05-26）

| 步骤 | 完成 | 关键数字 |
|---|---|---|
| KQA Pro 下载 + KB ingest（362K triples, 123K entities, 990 relations）| ✓ | KB 规模 30× RadarKG |
| KQA Pro 9 类 → 我们 11 类 mapping（v1 → v2 修正后）| ✓ | mapping 错误自暴：QueryAttrQualifier 应映射 path_plan 而非 constrained_join |
| **Dispatcher probe**（n=288, stratified, 6 min, $0.07）| ✓ | routing match **67.4%**；dual_subgraph **100%**, complement **84%**, lookup 68%；6 operators 全激活 |
| **E2E mini comparison**（n=139, Find-resolvable, oracle-op + mechanical program→args, 6 min, $0.12）| ✓ | Baseline 30.9% / Strategy 28.8% / Δ = **−2.2pp**；多数题型 tied (lookup fallback) |

**核心发现**：**Two-Stage Portability** —— framework 干净拆分为可移植与不可移植两部分：

| 组件 | 跨域可移植? | KQA Pro 证据 |
|---|---|---|
| 11-way typology | ✓ | 0 路由错误，分布合理 |
| Zero-shot LLM dispatcher | ✓ | 67.4% match，100% on SelectBetween，84% on Verify |
| Operator suite mechanism | ✓ in principle | Operators run，lookup fallback works |
| **Parser (args extraction)** | ✗ | Radar-locked；需要 per-domain re-tuning |

**论文写法**：新增 §5.6 "Cross-Domain Transfer (KQA Pro)" 整节呈现两实验 + Table 4-portability。Future Work 第 2 条改为最高优先级的 "Schema-derived parser"，直接 unlocks medical/materials KG 转移。

**自评分 v7**：v6 clear accept → **stronger clear accept**。新增 cross-benchmark evidence + 干净的可移植性 dissection。Reviewer 攻击 "你只在自建 benchmark 上测" 现在有了三层回应：
1. WebQSP probe（反向证据：benchmark 不合适）
2. KQA Pro dispatcher probe（正向证据：typology 跨域）
3. KQA Pro e2e（诚实负面证据：parser 是 portability bottleneck，指明 future direction）

## 6.9 论文 v11 → v12：Advice-driven framing 重构（2026-05-29）

### 起因
朋友 advice.md 指出 v11 论文叙事让 reviewer 把工作误读为"只是分类器 + 手写策略"。oracle 实验 (Δ −0.4pp) 已经证明 dispatcher noise ≈ 0，但论文结构仍把 dispatcher / typology 当核心摆在前面。这是叙事自我矛盾——已经用实验证伪了"分类器是 contribution"，但 framing 还在卖分类器。

### 核心 framing 切换

| 旧 (v11) | 新 (v12) |
|---|---|
| "Strategy-Routed GraphRAG: A Typed Operator Suite" | "Beyond Top-K: A Typed Operator Suite for **Answer-Geometry Mismatches**" |
| "top-K 是错的" | "top-K embeds an implicit answer-geometry assumption; assumption fails on specific computable patterns (information-theoretic, not LLM-capacity)" |
| dispatcher / typology 是卖点 | dispatcher 是 O(1) 查表；operator suite 是 analytical contribution |
| 6 operator 是 "empirical" | 6 operator 从 6 个 information-requirement class **derived**——derivation 决定了 number 和 content 都不是经验拼凑 |
| 跟 Adaptive-RAG / ByoKG-RAG 关系不清 | 显式：他们变 depth/source（保持 ranking 不变），**我们变 semantics**（每个 operator 是结构性不同的 KG 操作） |

### v12 具体改动（七处）

1. **Title + Abstract** 重写：去 "Strategy-Routed" 前缀，加 "answer-geometry mismatch" 概念；oracle Δ −0.4pp 提前到 abstract 明示 "this is a paper about the operators, not the classifier"
2. **Introduction** 重构为 4 步（phenomenon → diagnosis → insight → system），加 OLAP/B-tree 数据库类比作为理论 ground
3. **§2.3 新加 Information Requirement Profile 表**：明示 6 operator 是从 6 个 info-requirement class derived
4. **Related Work** 加 Routing-based retrieval 段落：显式对比 Adaptive-RAG (Jeong et al. 2024) 和 ByoKG-RAG，强调"vary semantics, not depth/source"
5. **§4.2 RQ2** 重写为 "Structural Failure → Operator"：per-operator ablation 表加 AGM-class 列，做 1:1 mapping
6. **§4.5 新加 RQ5: "Does AGM exposure predict Δ?"**：cross-bench 表（RadarKG + KQA Pro）按 AGM 类型分组（card/comp/cmpl/none）；AGM-exposed median Δ = +58pp，non-exposed = 0pp
7. **§5 Discussion** 加 Composability 子节，强化 operator suite 在自然组合下闭合的论证

### 新工件
- `scripts/cardinality_delta_analysis.py` + `results/cardinality_vs_delta.tsv`：per-type AGM-exposure vs Δ 跨基准数据
- `references.bib` 加 Adaptive-RAG (jeong2024adaptiverag) 和 ByoKG-RAG (byokgrag2025, TODO verify) 引用

### 自评分 v12
- v11 weak accept / borderline strong → **v12 clear accept / borderline strong accept**
- Main Track 也从"偏弱"变为"可投"
- 关键变化：v11 是"工程项目带方法论"，v12 是"analytical contribution（AGM）+ system contribution（operator suite）+ methodological finding（bilingual paradox）三位一体"

### 与 advice 对照
| advice 建议 | 落实情况 |
|---|---|
| #1 Restructure narrative: 结构性失败 + 算子设计 | ✓ Title + Abstract + Intro 全改 |
| #2 加 information-requirement 分析框架 | ✓ §2.3 加表 |
| #3 算子可组合性 | ✓ §5 加 Composability 表 |
| #4 typology 粒度消融 (3/6/11 类) | 未做（任务 #8 没拉，预算考量；suite-size ablation 部分覆盖此问题） |
| #5 去 "algebra" | ✓ Title + §5 改 |
| #6 KQA Pro reframe (cardinality vs Δ 图) | ✓ §4.5 新加 RQ5 整节 + 跨基准表 |
| #7 Structural Failure Analysis | ✓ §4.2 重写 + §4.5 AGM 分析联合实现 |

---

## 6.8 论文 v9 → v10：把 RadarKG-QA-499 升级为 benchmark contribution（2026-05-28）

### 起因
2026-05-28 系统调研了 7 个主流 KGQA 公开 benchmark 的 Count 答案分布：
| Benchmark | Count 中位 | 中位 ≥30 | 备注 |
|---|---:|---:|---|
| **RadarKG-QA-499** | **14** | **18%** | 我们的 |
| KQA Pro val | 2 | 6% | 已跑 ±0.2pp |
| Mintaka dev | 4 | 3% | 太小 |
| WebQSP | — | ≈0% | 88% single_hop |
| LC-QuAD 2.0 | — | — | 无 gold（要打 SPARQL）|
| HotpotQA 系 | — | 0% | 文本，非 KG-native |
| GraphRAG-Bench | — | 0% | textbook MCQ |

**结论**：**没有任何公开 benchmark 设计了 RadarKG-QA-499 的 high-cardinality 分布**。这从 reviewer 攻击点变成论文卖点。

### 论文 v10 改动
- **§1 contribution #4**：RadarKG-QA-499 从"open bilingual benchmark"升级为"**the first KGQA benchmark explicitly designed to expose top-K structural failures**（3.4× median Count cardinality of next-closest public benchmark）"
- **§4.4 新章节"Benchmark Design Philosophy"**：附 cross-benchmark distribution 表，论证为什么需要新 benchmark
- **abstract**：补一句把 RadarKG-QA-499 当方法论贡献释放
- **§7 Future Work item 3**：从"在 WebQSP/CWQ/GrailQA 验证"改为"Community develop high-cardinality KGQA benchmarks"——把缺失变成 research gap

### 自评分 v10
- v9 weak accept → **v10 clear accept**
- "在公开 benchmark 上没赢"不再是弱点：因为**没有公开 benchmark 有合适的分布**——这本身就是 community gap，我们的 benchmark 是 gap 的第一块拼图

### 还在做 / 选项
| 任务 | 工时 | 收益 |
|---|---|---|
| 论文 ACL 格式 + 字数压缩到 8 页 | 1 天 | 投稿前必做 |
| 真 RoG finetuned 复现 | 2 天 + GPU | rebuttal 备弹 |
| 自造 Wikidata 高基数 benchmark | 2 天 | 加固 §4.4，但有自造嫌疑 |

---

## 7. 未完成工作

| # | 任务 | 状态 | 估计工时 |
|---|---|---|---|
| #5 | 抽取层 `upgradeOf` precision 0.143 修复 | 已决定不做（小修，论文不依赖）| — |
| 文档 | 论文撰写（10-12 页 conf paper）| v5 完成（reviewer-style 自审 + 改）；剩字数压缩到 conf 格式 | 1 天 |
| 加分 | qa_500 全 499 题三方对比 | **完成（2026-05-22）** | — |
| 加分 | v2 KG Stage C 全量扩展（120→331 雷达）| **完成（2026-05-22）** | — |
| 加分 | Oracle dispatcher 上界实验 | **完成（2026-05-22）** — 反常发现 LLM 反而稍好 | — |
| 加分 | Bootstrap 95% CI on Table 1 | **完成（2026-05-22）** | — |
| 加分 | 真 RoG 复现（finetuned LLaMA-2 / DeepSeek-V3 on WebQSP）| 推荐（rebuttal 用）| 2 天（需 GPU）|
| 加分 | WebQSP 公开 benchmark 切片（200 题英文） | 推荐 | 2 天（KG ingest 改 Freebase 子图）|
| 加分 | 跨域转移到 medical/materials KG | 战略性 | 1 周 |

---

## 8. 一图说话：项目当前状态

```
                 ┌──────────────────────────────────────────────┐
                 │  RadarKG  (5610 triples, 28 relations)       │
                 └──────────────────────────────────────────────┘
                                       │
                  ┌────────────────────┼────────────────────────┐
                  │                    │                        │
            Lexicon Layer       Strategy Pipeline           Datasets
            ├ relations.json    ├ qa_router.py             ├ validation_30
            ├ entity_aliases    ├ qa_strategy_pipeline     ├ qa_500 (478)
            └ question_types    └ qa_rog_baseline          └ bilingual_stress
                                       │
                  ┌────────────────────┼────────────────────────┐
                  │                    │                        │
            实验：3-way (full 499) 实验：ablation            实验：robustness
            B 52.7% [48.3, 57.1]  -exhaustive  -18pp       router 93%, e2e 88.6%
            R 60.3% [55.5, 64.9]  -con_join     -8pp       Oracle e2e 88.2% (Δ −0.4pp!)
            S 88.6% [86.2, 91.2]★ -path_plan    -6pp       → dispatcher noise = 0
            ΔB +35.9 [+31.7, +40.1]  -bilingual -21.7pp(*) cost $0.00041/Q
            ΔR +28.3 [+23.8, +32.7]                        $0.00084/pp marginal
            (paired bootstrap, 1000 resamples)
                                       │
                                       ▼
                              论文 main table + ablation table 数据齐全
                              (主创新 + 副创新 + robustness 全有硬证据)
```

---

## 9. 这套数据能投哪个会议

按预估投稿等级：

- **Tier-1 NLP 会议（ACL/EMNLP main）**：需要补 RoG 在 WebQSP 上的复现数字（验证基线实现），并把 qa_500 全 478 题数据补齐（n=100 略显单薄）。当前数据接近但不完全到位。
- **应用类会议（NAACL Industry / EMNLP Industry / KDD Applied Data Science）**：当前数据完全够投。bilingual + 工业 KG 这一对组合非常对口。
- **领域会议（ICTAI / IJCKG）**：数据更够。
- **期刊（TKDE / TASLP / TOIS）**：扩到 500 题 + 增加 OOD 评测后可投。

我的建议：先把数据补齐到 478 全集 + RoG WebQSP 复现，瞄准 NAACL Industry 或 EMNLP Industry 2026。
