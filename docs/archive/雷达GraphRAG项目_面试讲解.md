> 历史资料：保留当时的方案与结果，不代表当前完成度。当前方向见 `/docs/RESEARCH_DIRECTION.md`，进度见 `/docs/CURRENT_STATUS.md`。

# 雷达知识图谱 GraphRAG 系统 —— 面试讲解手册

> 一句话定位：一个**面向军事雷达情报的全栈 GraphRAG 系统**，从"多智能体抽取建图 → 混合检索 → 双层分析智能体 → 置信度感知的 RL 多轮检索策略"四层递进，贯穿主线是**可信、可溯源、LLM 不进事实信任路径**。

---

## 0. 项目总览：四层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  ① 建图层  pipeline/v3/   多智能体 LLM 抽取 + 确定性双轨 → kg_v3       │
│     语料/PDF ─► 分块 ─► [确定性抽取 ‖ 多agent LLM抽取] ─► 裁判 ─► 融合  │
├─────────────────────────────────────────────────────────────────────┤
│  ② 检索层  graphrag_retriever/reranker/retrieve_v3                     │
│     BM25 + 向量 + RRF融合 + 图扩展 + Cross-Encoder精排 + 图文联动      │
├─────────────────────────────────────────────────────────────────────┤
│  ③ 分析层  agent/ + qa_strategy_pipeline                              │
│     外层 Plan-and-Execute 编排 ‖ 内层算子组合执行器 ‖ 可溯源报告 ‖ EW  │
├─────────────────────────────────────────────────────────────────────┤
│  ④ 推理层  ca_agraphrag/  置信度感知 Agentic GraphRAG（RL）           │
│     KG环境 + 工具集 + SFT冷启动 + GRPO + trust奖励塑形（毕设主脊柱）    │
└─────────────────────────────────────────────────────────────────────┘
```

**贯穿全系统的设计哲学**：
1. **LLM 只做"规划/复述"，绝不进入事实信任路径**——所有事实由确定性引擎从 KG 算出。
2. **全程可溯源**——每条边/属性/结论都带 `evidence + source + tier/confidence`。
3. **保守优先**——消歧、模糊合并已被实验证伪，只做词表命中和写法撞车归并。
4. **每个机制都有实验依据**——两段式抽取、窄批判裁判、类型感知加权等都对应错误分析或消融。

---

# 第一部分：知识图谱构建 Agent（`pipeline/v3/`）

## 1.1 作用

把异构语料（维基/RadarTutorial/装备指南/两本 PDF 手册）抽成一张**带类型、带证据、带置信分层**的雷达知识图谱 `kg_v3`（实体 + 关系边 + 属性 + 冲突记录）。

## 1.2 架构：S1→S2(+S2b)→S3→S4→S6→S5→S8（`run_v3.py` 编排）

**核心是"双轨抽取"**：结构化源走**零 LLM 的确定性规则**，非结构化叙述文本走**多 agent LLM 抽取 + 忠实度裁判**。

```
S1 分块 ─┬─ struct块 ─► S2 确定性抽取（infobox/规格表，零LLM）─┐
         │              S2b Wikidata SPARQL 第三方源           │
         └─ text块  ─► S3 多agent抽取 ─► S4 忠实度裁判 ─► S6 类别归一 ─┤
                                                                    ▼
manual/  PDF ─► 图片 ─► VLM转录 ─► 确定性解析 ─► 映射边/属性 ──► S5 融合
                                                                    ▼
                                              S8 全局质量自检（只读不改）
```

## 1.3 各阶段的算法与机制

**S1 分块（`s1_ingest.py`）**：按文档类型路由。wiki 按 `== 章节 ==` 切块（超长再按段落/句子二级切），RT 卡整卡一块，规格表单独产 `struct` 记录。每块带 `radar_hint`（页面主型号）**仅作提示不强制当 head**。

**S2 确定性抽取（`s2_deterministic.py`）**：零 LLM，处理 infobox/规格表键值对。核心机制是**属性/关系源头分流**——数值型键（range/PRF/power）进属性表，实体型键（country/manufacturer）进边表。每条挂 `evidence=原始"key:value"`。`parse_measure` 做单位归一（括号内公制优先、欧式小数逗号、千分位空格）。

**S2b Wikidata（`s2b_wikidata.py`）**：对已有雷达批量查 SPARQL，用 **`RADARISH` 类型门控**（匹配 P31 类型标签是否像雷达/传感器）排除同名歧义。查询做型号变体展开（去后缀、拆 NATO 代号、拆括号编号）提命中率。独立第三方源 → 天然多源印证。

**S3 多 agent LLM 抽取（`s3_extract.py`）—— 建图核心**：
- **Agent① 侦察员（scout）**：先列出文本里实际出现的实体清单，不推断。
- **Agent② 抽取员（extract）**：**两段式抽取**（KGGen 式）——`head` 只能从①盘点的型号里选，治"张冠李戴"（金标错误分析发现）。关系清单按**章节标题动态路由子集**（ODKE+ 式，`schema.route_relations`），缩小选择空间提准确率。prompt 由 `schema.py` 从 `lexicon/relations.json` **自动编译**（单一事实源）。
- **Agent④ schema 提案**：零成本——②输出带 `proposals` 字段收集清单外的真实关系，供人工终审扩展 schema（这是"受控schema扩展"验证有效的来源）。
- WEG 装备指南走独立 `extract_weg`，用 `partOfSystem` 把内嵌雷达挂到系统上，避免系统名污染成雷达节点。

**S4 忠实度裁判（`s4_critic.py`，Agent③，验证 +5pp）—— 两级门控**：
1. **词法硬门控（零成本）**：tail（含别名变体）必须在块文本中字面出现，数值型要求数字组全命中；
2. **LLM 窄裁判**：只对词法未过、但**引文确实存在于原文**（防幻觉引文骗过裁判）的三元组，判"原文是否支持该论断"——**明确禁止评价关系类型选得合理与否**（宽批判已证伪）。
同时记录 head 接地状态：在文中→`grounded`；只是页面主体→`hinted`（降置信层级）。

**S6 类别归一（`s6_canonicalize.py`）**：把 LLM 抽出的自由文本 `hasFunction/hasTechType/hasMode` tail 合并到受控词表（去后缀/括号缩写→多标签匹配），让不同雷达共享同一功能/体制节点提升图连通密度；归不进词表的独特描述**降级为属性**而非留孤立边。

**S5 融合（`s5_fuse.py`）—— 另一核心**：
- 全库唯一 junk filter；
- **雷达名归一三层**：精确 norm 匹配语料标题 → 放宽索引（去通用词后仍唯一才收）→ **写法撞车归并**（同 norm 多写法自动归并，跳过带斜杠的歧义型号）；
- **厂商归一**：`maker_core` 剥公司后缀/部门词取核心，同核心归到最简洁写法（厂商是 hub，碎片化是低密度主因）；
- **置信度 = 分层先验**（`TIER_PRIOR`：wikidata 0.92 / manual 0.93 / struct 0.90 / llm_grounded 0.85 / llm_hinted 0.75 / judge_passed 0.70），**不是模型自评分**；
- `(head,relation,tail)` 去重合并 provenance，`corroborated = 来源kind≥2 或 doc≥2`；
- **单值关系一头多尾 → `conflicts.json` 显式记录，不消解**；
- 唯一派生边 `headquarteredIn`（厂商总部国=其研制雷达的多数国别），要求 ≥2 部雷达且 ≥60% 一致——有据聚合非规则灌水。

**S8 全局质量自检（`s8_quality.py`）**：把历次人肉发现的问题模式编码成自动检测器（垃圾型号/厂商平台碎片/悬空引用/类型违规/孤立叶子占比），只读产出 `quality_report.md`，新数据源接入跑一遍就暴露问题，不再依赖逐个人肉巡查。

**manual/ 子流程**：两本 PDF 手册独立走"PyMuPDF 渲染成图片 → Qwen3-VL **多页滑动窗口视觉转录**（窗口重叠防跨页雷达被切两半，带模型链自动降级）→ 确定性正则解析结构化 Markdown → 映射到 v3 schema（tier=v3_manual 0.93）"。设计分工：**乱文本走多 agent 流水线，结构化 md 走确定性 parse**。

---

# 第二部分：混合检索链路（`graphrag_retriever.py` / `reranker.py` / `retrieve_v3.py`）

## 2.1 作用

给定自然语言查询，从 KG 三元组 + 手册叙述文本中召回最相关证据，组装成 LLM 上下文。

## 2.2 架构：四层 + 精排

```
query
 ├─ 层1 BM25 关键词召回 (Top-30)   ┐
 ├─ 层2 FAISS 向量召回 (Top-30)    ├─► 层3 RRF 融合 (Top-10, type-aware)
 │                                 │
 │                       层4 图扩展 (NetworkX/Neo4j, 1-2跳)
 │                                 │
 │                 Cross-Encoder 精排 (可选懒加载, Top-5)
 │                                 │
 └──────────────► context 组装 + entity_hits ─► LLM
```

## 2.3 各层的算法/机制/工具

**层1 BM25（`BM25Retriever`，rank_bm25）**：稀疏检索。关键是**自定义中英混合分词**——正则 `[a-zA-Z0-9/\-\.]+|[一-鿿]` 把 `AN/SPY-1` 这种带斜杠连字符的型号当**一个整词**保留、中文按单字切。三元组先转"头 关系中文 尾。证据"自然语言句再索引。

**层2 向量（`VectorRetriever`，FAISS + `bge-small-zh-v1.5`）**：稠密检索。**自适应索引**——<2000 条用 `IndexFlatIP`（精确内积≈余弦），≥2000 条切 `IndexIVFFlat`（倒排+聚类近似，`nlist=√n`、`nprobe=nlist/4`）权衡精度速度；带落盘缓存 + 数量变化自动重建。

**层3 RRF 融合（`rrf_fusion`）**：倒数排名融合 `score(d)=Σ 1/(k+rank_i(d))`，k=60。**只用排名不用原始分**，天然免归一化（BM25 分和向量余弦量纲不可比）。创新点 **type-aware 加权**：`infer_answer_type` 正则从问题推断期望答案类型（问"谁研制"→Manufacturer），命中的 `tail_type` 匹配时分数 `×(1+0.30)`——**用乘法不用加法**，避免类型匹配压倒相关性本身。

**层4 图扩展（`InMemoryGraphExpander`，NetworkX MultiDiGraph）**：从 RRF Top-5 头尾实体出发 1-2 跳 BFS 邻域扩展（进出边都走），低置信边过滤。这是 GraphRAG 相对普通 RAG 的核心——**多跳问题**（"X 的研制方所在国"）纯语义检索召不全，图扩展补桥接实体。提供 `Neo4jGraphExpander` 可选后端（APOC 子图查询），连不上自动降级内存图。**图扩展结果不进 RRF**，作为补充候选直接拼接（它是结构补全不是相关性排序）。

**精排层（`reranker.py`，Cross-Encoder `bge-reranker-base`）**：原理对比是重点——Bi-Encoder（层2）query 和文档**分开** encode，能预建索引所以快但精度低；Cross-Encoder query 和文档**拼接一起过 attention**，精度高但慢。所以是"Bi-Encoder 粗筛 20 条 → Cross-Encoder 精排 5 条"两阶段。工程细节：**懒加载**（第一次 rerank 才载模型）；给 Cross-Encoder 的文本**更丰富**（带类型+证据）。`compare_rrf_vs_reranker` / `rerank_with_analysis` 供论文消融。

## 2.4 v3 图文联动（`retrieve_v3.py`，最近工作）

`V3Retriever` 把检索器迁到 kg_v3（独立索引，不动 v2 冻结索引保证旧实验可复现）。真正的新东西是**图文联动**：KG 存结构化三元组，但手册"技术特点/工作原理"那种叙述段无法三元组化——另建 `narrative_index`（bge-small-zh FAISS 叙述向量库）。检索时**图命中雷达型号后，自动去叙述库按该型号定向召回技术描述段**（`radar=` 过滤），不足再全局补。这样"原理/结构"类问题也能答，联动键是"图命中的实体"，比纯向量拼接更精准。

## 2.5 消融框架（`run_ablation`）

内置 6 配置对比：BM25-only / Vector-only / BM25+Vector / Full(+Graph) / **NaiveRAG（原文分块 BM25）** / **LLM-only（无检索裸问）**，按 single/multi/comparison 分题型算 Recall@K。后两个是关键 baseline——证"结构化 KG 检索 vs 原文分块 RAG vs 纯大模型记忆"的差距。

---

# 第三部分：分析 Agent（双层智能体，`agent/` + `qa_strategy_pipeline.py`）

## 3.1 作用

把"单次问答"升级为能做**复杂分析任务**（多约束筛选、计数聚合、多跳、对比、生成可溯源报告、电子战对抗推演）的智能体。

## 3.2 架构：双层 + 共享框架

```
用户问题
  │
  ▼ 外层 RadarAgent (agent_loop.py) ── Plan-and-Execute
  │   ①指代消解 → ②LLM拆解带数据流的计划 → ③确定性执行工具 → ④验证+自修正 → ⑤综述
  │
  ├─ composition_query → 内层 CompositionAgent (planner.py + composition.py)
  ├─ entity_dossier / comparison_report → RadarReporter (report.py)
  ├─ counter_advisor / eob_wargame → EW 三件套
  └─ web_search → WebTool (web_tool.py)

共享地基 qa_strategy_pipeline.py：KGIndex + 别名层 + 策略路由 + 类型校验
```

## 3.3 外层 RadarAgent（`agent_loop.py`）—— Plan-and-Execute

不是简单单轮问答，是 **Plan-and-Execute**（比 ReAct 更适合可控执行）：
1. **指代消解（`_resolve_coref`）**：多轮对话里"它/前者"用 LLM 依历史改写成自包含问题（decontextualization），带正则前置门控省调用。
2. **任务规划（`_make_plan`）**：LLM 把请求拆成步骤 JSON，**带数据流**——`$s1` 引用步骤 s1 结果，`foreach:$s1`+`$item` 逐个处理。**LLM 只出计划不执行**。规划失败降级单步。
3. **确定性执行（`_exec_steps`）**：实体集合在步骤间以 Python set/list 流转，**不经 LLM 重新转录**（防实体名漂移），foreach 有 `MAX_FOREACH=6` 扇出上限。
4. **验证+一轮自修正（`_verify`/`_revise_plan`）**：用**确定性规则**（非 LLM 自评）扫 trace，`_GAP_MARKERS`（"未收录/共0项/出错"）命中判缺口，喂回 LLM 出修正计划（型号没收录→加 web_search；空结果→放宽约束）再跑。
5. **综述+可溯源交付（`_finalize`）**：LLM 只基于确定性观察综述，严禁添加数字/型号；**真正给用户的报告 artifacts 逐字原样附在综述后**——用户看到的权威内容是确定性产物。

## 3.4 内层 CompositionAgent（`planner.py`+`composition.py`）—— 算子组合

**分析能力的核心，也是毕设算法贡献点**（Answer-Geometry Mismatch / 派生算子框架）。

**`planner.py`**：LLM 把问题翻译成一棵 **JSON 算子组合计划树**（9 种带类型算子），**只出计划树不执行不碰数字**。

**`composition.py`（`CompositionExecutor`）**：确定性递归求值器。4 种类型流转：`EntitySet`/`Scalar`/`Bool`/`Compare`。9 算子：`constraint`（满足 (?,rel,value)）、`intersect/union/difference`（交并差=AND/OR/NOT）、`path`（多跳，`r^-1` 反向）、`count/enumerate`、`aggregate`（数值聚合，**且必报覆盖率** `covered/total`，因数值属性全库仅 ~20-25%）、`compare`（双子图 same/only_a/only_b）、`contains`（成员判断）。组合查询靠**嵌套**，如"中国研制的 S 波段雷达有多少种"= `count(intersect(constraint(countryOfOrigin,中国), constraint(hasFrequencyBand,S)))`。

**反思重规划（`reflect`+`is_degenerate`）**：结果空时**用 KG 真实取值当接地提示**重规划——如约束写"在役"但图谱存"服役中"，`distinct_tails` 喂真实取值让 LLM 改。区分"计划错"和"真没答案"，最多 2 轮。

**权威答案分离（`deterministic_summary`）**：**只从执行值和 trace 构造答案，数字永不过 LLM**，`answer` 让 LLM 复述但严令不得增减数字。

## 3.5 共享框架（`qa_strategy_pipeline.py`）

分析层地基，agent 复用其 `KGIndex`（两个倒排索引 O(1) 查）、别名解析、算子叶子。它本身也是一条完整的**策略路由 GraphRAG 流水线**（agent 之前的方案 + 现作对比基线）：
- **策略路由（`qa_router.py`）**：零样本把问题分 11 题型 → 映射 6 策略（exhaustive/complement/path_plan/constrained_join/dual_subgraph/lookup）——**不同题型走不同检索策略**（Answer-Geometry Mismatch 思想）。
- **双语别名层 + 等价类回退**：中文"美国"↔KG 写法、"S波段"→"S"、去后缀。`RELATION_EQUIV_CLASSES`：`countryOfOrigin` 和 `operatedBy` 都指向 Country，交集空时自动并（抽取噪声致子图割裂的补救）。带 `BILINGUAL_DISABLED` 开关做消融。
- **类型校验（`_validate_args`）**：LLM 把"美国研制"错填 `developedBy`（tail 应是 Manufacturer）且 KG 零命中时，自动换成 tail_type 匹配的最高频关系——**只在"类型全不匹配且零命中"时才换**，避免过度纠正。
- `LLM_URL/MODEL/KEY` 环境变量可换——default DeepSeek，可指向**本地 Ollama**（保密边界内无外部 API）。

## 3.6 可溯源报告（`report.py`）

`RadarReporter` 产出**装备档案 dossier** 和**对比报告 compare**。核心 `ProvenanceKG`——每条三元组连 `source/confidence/evidence` 一起索引，报告里**每条事实带〔来源+置信度+证据〕**，低置信（<0.75）标 ⚠ 且排除在自动概述外。`_faithfulness` 轻量校验概述里的数字/型号 token 是否都在引用事实里出现。

## 3.7 电子战三件套（EW，`threat_profile`/`ew_advisor`/`ew_wargame`）

面向"雷达对抗决策辅助"落地场景：
- **`threat_profile.py`**：从 KG 派生**威胁画像**（用途/跟踪体制/扫描/频率捷变/所用 ECCM）。结构化字段高置信 0.9，描述文本关键词推断中置信 0.6。**规则抽取真实 KG 值非编造**，每字段带证据。
- **`ew_advisor.py`**：确定性 doctrine 推理。匹配 `doctrine_map.json` 规则推荐/避免干扰样式，**对方 ECCM 抵消哪些推荐**（`defeats`，被克制置信×0.4），加威胁定级、按本机参数（波段/功率/距离）的差异化交战研判。**无 LLM 进信任路径**，明确标"非实测·doctrine级启发式"。
- **`ew_wargame.py`**：EOB 级推演。多部敌雷达——威胁排序 + **贪心加权集合覆盖**求最小装备包 + 干扰样式经济性 + 能力缺口 + SEAD。

## 3.8 联网补全（`web_tool.py`）

KG 没覆盖的型号：搜 Wikipedia → LLM schema 约束抽三元组 → 别名层链接 KG 规范写法 → 标 `source=wikipedia, confidence=0.6`，**明确和 curated KG 分离**，答案注明"未核验"。

---

# 第四部分：置信度感知 Agentic GraphRAG（RL，`ca_agraphrag/`）—— 毕设主脊柱

## 4.1 作用与定位

**把已建好的 KG 当环境，用执行反馈强化学习训一个 7B 级多轮检索策略**，让它面对复杂/多跳/多约束问题时**自主选择检索动作**；并把**置信度/冲突信号放进奖励**，学出"偏好印证证据、冲突时审慎"的行为。第三部分的 6 算子在这里**降级为 agent 的工具**（不再是创新点）。

> 研究定位（`agentic_graphrag_rl_plan_v1.md`）：硕士论文在 Agentic GraphRAG + RL 检索主线上，加自己的想法（**置信度/冲突奖励塑形**）+ 扎实工程量。前置方向"属性可修复"已被证伪（缺失71%、可回收仅2.4%），置信度重排/自演化/过程奖励均已否——**只保留"训练时奖励塑形"这一条可量化消融的主张**。

## 4.2 架构

```
问题 ─► [策略 πθ (Qwen2.5-7B + LoRA)] ─多轮─► think → 选工具 → 执行 → 观察 → …→ finish
              │                            │
              │ 动作空间=工具集             │ 每步返回观察（后续附证据 tier/冲突标记）
              ▼                            ▼
        KG 环境（只读 kg_v3，确定性判分）
              │
              ▼
   r = correctness（主）+ trust（创新，可消融）+ format − step_penalty
              │
              ▼
   GRPO 更新 πθ（SFT 冷启动 → RL）；4×4090 + LoRA
```

## 4.3 各模块的实现

**① KG 工具层（`kg_tools.py`）—— agent 的动作空间**：只读 kg_v3，确定性执行。7 个工具：`lookup(entity,relation)`（取尾集，单跳/多跳/否定用）、`find(relation,value,type)`（反向取头集，计数/枚举/多约束用）、`intersect/union/difference`（集合运算）、`count`、`finish`。关键：**工具用与金标生成一致的过滤视图**（`filtered=True`：干净实体名 + 高置信边 + 频段白名单 + 国家白名单 + 单值关系必印证），保证"工具查询视图 == 金标视图，环境才自洽"。

**② 环境（`env.py`）—— gym 式多轮环境**：`reset/step`。奖励 `r = correctness − step_penalty·(steps-1)`，`finish` 时结算。`score_answer` **按题型确定性判分**：count 精确整数、bool 布尔匹配、scalar 集合相等、set 用 **F1**、compare 判 same。非法工具调用扣 format（-0.05），超步硬罚（-0.1）。

**③ 数据集生成（`gen_dataset.py`/`gen_probe.py`）**：从 kg_v3 确定性生成 7 题型带金标的题（single_hop/two_hop/count/enumerate/comparison/negation/multi_constraint）。三个关键机制：
  - **上游 KG 脏值清洗**：人工核验发现 4 类系统性问题（字段粘连冒号/频段书写不一/developedBy 混入非公司/公司名子公司变体），逐类规则治理；
  - **高置信金标**：单值关系（developedBy 等）金标**必须多源印证**（治手册按使用国分章把操作国污染成原产国）；
  - **实体级切分防泄露**：同一雷达/分组用**稳定哈希**只落一个桶，`test 与 train 不共享实体`（末尾有跨桶泄露检查断言=0）。这是"测真泛化非记忆"的关键。

**④ SFT 冷启动轨迹（`gen_sft_traj.py`）**：为每题构造金标工具序列（think→tool→observe→finish），**finish 用工具实际算出的值**，且**每条轨迹回放进环境确认 reward=1 才写盘**（自洽可执行）。

**⑤ SFT 训练（`train_sft.py`）**：Qwen2.5-7B + LoRA（r=16，7 个 target module），**只对 assistant 轮算 loss，tool 观察作输入不算 loss**（labels=-100 掩码）。tool 观察作 user 消息喂回（便携不依赖各模型 tool-role 模板）。带 `--dry` CPU 冒烟验证分词/掩码管线。

**⑥ GRPO 强化（`train_grpo.py`）—— 工程主体**：策略=基座+SFT LoRA（可训），参考=冻结同权重。**简化 GRPO（组相对优势 + KL 到参考）**：
```
A_i = (r_i − mean(r_grp)) / (std(r_grp)+eps)          # 组内相对优势，免价值网络
L   = −(1/G)Σ A_i·(1/|o_i|)Σ logπθ(o_t) + β·KL(πθ‖πref)
```
每题采样 G=8 条轨迹（temp=0.9 探索），`rollout_group` 记录整条对话 token + **生成掩码**（只有 assistant 生成位置计入 loss），`seq_logprobs` 算生成位置 token logπ，`grpo_step` 按题分组算优势更新。梯度裁剪 + KL 系数 0.02。**训练慢在环境交互（每步查图）不在梯度**——真实系统工程量。

**⑦ 评测（`eval_policy.py`/`test_env_oracle.py`）**：协程式策略接口，`run_policy` 驱动策略与环境交互。两个桩策略：`oracle`（走金标序列应得~100%，验证框架正确）、`empty`（空答下限）。`test_env_oracle` 用工具按金标重算验证"工具能复现金标 & 环境给满分"，逐题型报命中率——全高才说明**环境+工具+奖励自洽可用于 RL**。

**⑧ 问句润色（`polish_llm.py`）**：模板句→自然中文（Qwen 模型链+缓存），**金标保护**——改写后必须仍含关键 token（型号/值），否则回退模板句，`gold_answer` 绝不变。

## 4.4 奖励设计（论文"自己的想法"）

`r = α·correctness + β·trust + γ·format − δ·step_penalty`
- **correctness（主项）**：EM/集合 F1/数值容差；
- **trust（创新，可消融）**：答案建立在低层级证据且无印证→扣分，建立在印证过的高层级证据→加分；冲突题若识别冲突并对冲/弃权→加分，武断取值取错→重罚。**目的是让策略"学出"审慎行为而非事后重排，±trust 消融量化其价值**。

## 4.5 实验设计（四档基线 + 四组消融）

- **基线**：①单轮混合检索（第二部分，增益下限）②裸 7B+工具 prompt-only ReAct（判训练值不值）③SFT-only（判 RL 增量）④RoG/ToG 式对照。
- **消融**：−RL（退 SFT）/ **−trust 奖励项**（核心想法价值）/ −工具子集 / −多轮（限单轮）。
- **指标**：EM、集合 F1（漏答与编造分列）、分题型准确率、冲突识别率、成本（工具调用数/token）、公开集迁移（WebQSP/CWQ/MetaQA 各 500 守通用性下限）。3 seed + 配对 bootstrap 显著性 + test 封盘。

---

# 第五部分：面试怎么介绍（分层递进）

## 电梯版（30秒）

> 我做的是一个**面向军事雷达情报的全栈 GraphRAG 系统**，四层：① 多智能体 LLM 抽取 + 确定性双轨建了一张带证据、带置信分层的雷达知识图谱；② 四层混合检索链路（BM25+向量+RRF+图扩展+Cross-Encoder），做了图谱事实与手册叙述的图文联动；③ 一个双层分析智能体，外层 Plan-and-Execute 带数据流的多步计划自验证，内层把问题编译成**类型化算子组合树**由确定性引擎执行；④ 毕设主脊柱是**置信度感知的 Agentic GraphRAG**——把 KG 当环境，用 GRPO 强化学习训一个 7B 多轮检索策略，把置信度/冲突信号放进奖励。全系统主线是 **LLM 只做规划复述、绝不碰事实，所有结论从图谱确定性算出、逐条可溯源**。

## 详版（分四层讲，每层一个"为什么"）

**第一层 建图——"为什么不直接用大模型抽三元组一把梭"**：
军事情报要求可溯源、低幻觉。我做了双轨——结构化规格走零 LLM 确定性规则，叙述文本走多 agent（侦察→抽取→裁判）。核心是**两段式抽取**（head 只能从盘点实体选，治张冠李戴）+ **两级忠实度门控**（词法硬门控免费拦截，只有引文真实存在才送 LLM 窄裁判，实测 +5pp）。置信度是**分层先验**不是模型自评，冲突显式记录不消解。

**第二层 检索——"为什么 GraphRAG 优于普通 RAG"**：
普通向量 RAG 对多跳、多约束召回不全。我做四层混合——BM25 保型号精确匹配、向量保语义、RRF 免归一化融合、**图扩展补桥接实体**（多跳答案的关键）、Cross-Encoder 两阶段精排。还做了图文联动：图命中雷达后定向拉它的手册技术描述段，让"原理类"问题也能答。

**第三层 分析 Agent——"为什么算子组合优于让 LLM 直接生成查询"**：
把复杂分析（计数/多约束/对比）变成**类型化算子树的确定性执行**。LLM 只出计划树，实体集合在步骤间以 Python set 流转不经 LLM 转录——**没有实体名漂移、数字永不过 LLM**。外层再套 Plan-and-Execute 做多步编排 + 确定性验证 + 一轮自修正。所有交付物（档案/对比/对抗建议）逐条带来源置信度可审计。

**第四层 RL 推理——"为什么要训一个策略而不是手写规则路由"**：
手写策略路由（第三层的做法）覆盖不了开放的复杂题。我把 KG 当环境、6 算子当工具，用 **SFT 冷启动 + GRPO** 训 7B 策略自主选择多轮检索动作，并把**置信度/冲突放进奖励**塑形出审慎行为——赌注是"子技能上训出的组合能力泛化到复杂题优于单轮检索"，用完整消融验证。工程上用 4×4090 + LoRA，实体级切分防泄露。

---

# 第六部分：被深挖怎么回答

## 建图部分

**Q：两段式抽取和普通 LLM 抽三元组区别？**
普通抽取 LLM 可以凭世界知识补出文本没有的 head，导致"张冠李戴"（把 B 雷达的事实挂到页面主角 A 上）。两段式先让 Agent① 盘点文本实际出现的实体，Agent② 的 head **硬约束只能从这个清单选**，从机制上杜绝了跨实体污染。这是金标错误分析里发现的头号错误类型。

**Q：为什么置信度用分层先验不用 LLM 自评？**
LLM 自评分数是"自我标注当金标"——记忆里我们已经删过一个 87% 的 LLM 自标结果。分层先验按**来源可靠性**（Wikidata 独立第三方 / 手册权威 / 结构化规格 / LLM接地 / LLM提示 / 裁判通过）赋值，是可解释、可回归校准的，且和多源印证正交——两个信号叠加比单一自评稳健得多。

**Q：忠实度裁判为什么只判"引文支持论断"，不判关系类型对不对？**
这是踩过坑的——宽批判（让 LLM 同时评关系类型合理性）实测**证伪**（没有增益甚至负向）。窄批判（只判事实是否被原文表达）+5pp。因为关系类型的判断需要 schema 全局视角，LLM 在单条上下文里评容易主观误判；而"这句话是否表达了这个事实"是个客观得多的判断。

## 检索部分

**Q：为什么用 RRF 而不是加权求和？**
BM25 分和向量余弦量纲不可比、分布差异大，加权求和要调归一化和权重且不稳定。RRF 只用排名、无量纲、无需调参、对异常分数鲁棒（TREC 验证过）。代价是丢了分数绝对值，所以后面用 Cross-Encoder 精排把绝对相关性找回来。

**Q：图扩展为什么不进 RRF 一起排序？**
图扩展是**结构补全**不是相关性排序——它补的是多跳答案的桥接/终点实体，本身和 query 文本相关性可能不高，塞进 RRF 会被打低分淘汰反而丢答案。所以它作为补充候选直接拼进上下文，交给 Cross-Encoder 和 LLM 判断。

**Q：type-aware 加权为什么用乘法不用加法？**
加法会让"类型匹配"这个布尔信号直接叠加一个固定分，可能压倒 BM25/向量的相关性排序（一个类型对但内容无关的三元组排到前面）。乘法 `×(1+0.3)` 是**按比例微调**，保持相关性主导、类型只做相对提升，排序更稳。

**Q：Cross-Encoder 慢，线上怎么用？**
默认关闭+懒加载，只在需要高精度时开。粗筛已把候选压到 20 条内，Cross-Encoder 只对这 20 条打分，单次几十毫秒可接受。它和 Bi-Encoder 是互补两阶段不是替代。

## 分析 Agent 部分

**Q：算子组合 vs 让 LLM 直接生成 Cypher/SPARQL？**
LLM 生成查询语言是"文本→文本"，错了难定位、执行前无类型保证、数字可能被 LLM 二次改写。我的算子树是**类型化的**（4 类值），每算子输入输出类型确定、每步可审计（trace），实体集合以 Python set 流转不经 LLM 转录所以**无实体名漂移**、数字永不过 LLM。本质是把"生成正确查询"拆成"LLM 出结构 + 引擎保正确"。

**Q：aggregate 只覆盖 20% 数据，结果不就不可信？**
正因不可信才**强制报覆盖率**——`avg=350（12/48 个实体有该属性）`。我不假装完整，而是把覆盖率作为答案一部分暴露，让用户判断可信度。比悄悄用 20% 数据算个平均当全量诚实得多。

**Q：自验证会不会越修越错/死循环？**
两个约束：一是**确定性触发**（规则命中"共0项/未收录/报错"客观信号才触发，不是 LLM 自我怀疑），二是**最多一轮（外层）/两轮（内层）**硬上限。反思用 KG 真实取值当接地提示，是修**接地错误**（写法没对上）不是让 LLM 瞎猜，所以收敛。

**Q：LLM 只做规划，规划错了怎么办？**
三层兜底：①规划失败（JSON 解析不出）降级单步；②执行空结果触发反思重规划；③最终答案的事实来自确定性 trace，即使规划次优答出的也是 KG 真实事实。规划错顶多"没答全"不会"答错"。

## RL 推理部分（ca_agraphrag）

**Q：为什么用 GRPO 不用 PPO？**
PPO 需要额外训一个价值网络（critic）估计基线，7B 上再挂一个 critic 显存和调参成本高。GRPO 用**组内相对优势**（同一题采样 G 条轨迹，用组内奖励均值方差归一化当基线）**免掉价值网络**，`A_i=(r_i−mean)/std`。对我们这种**确定性判分、奖励稀疏但可批量采样**的场景特别合适，4×4090 上更可行。这也是 DeepSeek-R1 用 GRPO 的原因。

**Q：为什么必须 SFT 冷启动，不能纯 RL？**
纯 RL 从随机策略起步，7B 一开始根本产不出合法的多轮工具调用 JSON，奖励几乎全 0 没有梯度信号，样本效率极低。SFT 先用金标轨迹教会它"格式+基本工具序列"，给 RL 一个能拿到正奖励的起点，RL 才能在此基础上探索更优策略。而且我的 SFT 轨迹是**回放进环境确认 reward=1 才留**，保证冷启动数据自洽。

**Q：trust 奖励如果消融打平（没增益）怎么办？**
诚实报告——plan 里就写了这个风险对策：trust 降为分析章，主贡献转"多轮检索策略工程 + 领域 KG 环境"。但 trust 的地基是**真实存在的信号**（441 冲突 + 分层置信度，和已证伪的属性地基不同），所以有东西可挂。关键是它是**训练时奖励塑形**（改变学到的策略）不是推理时重排（那个已否），一个 ±trust 消融就能量化。

**Q：实体级切分为什么重要？随机切分不行吗？**
随机切分会让同一个雷达的 single_hop 题进 train、two_hop 题进 test，模型可能**记住这个实体的事实**而不是学会推理，test 分数虚高。实体级切分用稳定哈希保证**同一雷达/分组只落一个桶**，test 实体 train 从没见过，测的是**真泛化**。代码末尾有跨桶泄露断言=0。

**Q："又是 Graph-R1/Search-R1"怎么回应？**
不辩全新。delta 讲清楚：① **领域 KG 环境**（雷达情报，不是通用 KGQA）；② **置信度/冲突奖励塑形**（把 KG 质量信号引入 RL reward，这是我 KG 建图工作和 RL 的结合点，别人没有）；③ 6 算子作工具的完整工具集；④ 完整消融 + 公开集迁移。工程价值 + 领域结合立论，符合硕士论文"在主线上加自己想法+扎实工程量"的定位。

**Q：四层是怎么串起来的、有没有割裂？**
一条数据流贯穿：建图层产出的 `kg_v3`（带 tier/corroborated/conflict）**同时是**检索层的索引源、分析层的 KGIndex、RL 层的环境。而建图层沉淀的**置信度分层和冲突信号**，正是 RL 层 trust 奖励的地基——这是四层最紧的耦合点，也是"置信度感知"这个毕设标题的由来：**KG 的质量元数据不是丢弃，而是一路传导到最终的策略奖励里**。

---

## 附：一个要修的坑（面试前处理）

`qa_strategy_pipeline.py:42` 和 `qa_router.py:25` 有**硬编码的 DeepSeek API key 默认值**。代码评审/仓库公开会暴露，建议改成纯环境变量读取（去掉默认 key）。这和记忆里"泄露key作废"的待办对应。
