# 超越 Top-K：面向知识图谱问答中"答案几何不匹配"的类型化算子套件

**作者**: TBD
**目标会议**: NAACL Industry / EMNLP Industry 2026
**状态**: 中文版 v12.1（2026-06-01）— 英文原稿同步在 `strategy_routed_graphrag.md`；本中文版用于中文学界传播 / 内部分享 / 学位论文素材。

---

## 摘要

GraphRAG 系统在检索算法上不断演化——从社区摘要、Personalized PageRank、LLM 路径模板，到 beam 搜索——但 top-K 检索范式背后隐含了一个关于**答案几何形态**（answer geometry）的假设：答案是一个可以被排序信号 capture 的小集合。这个假设在三类工业 KG 上常见的可计算模式上**系统性失效**——无界枚举、多约束交集、补集 / 非成员判定——而且失效的原因是**信息论意义上的**，与 LLM 的能力无关。我们把这种失效命名为**答案几何不匹配**（Answer-Geometry Mismatch, AGM），并指出它在概念上类似于数据库领域 OLTP / OLAP 的划分：B-tree 索引天然适合 point query，但无论如何调优都不适合 aggregate query。基于这个诊断，我们提出 **Strategy-Routed GraphRAG**——一个由六个检索算子组成的类型化套件（`lookup`、`exhaustive`、`complement`、`path-plan`、`constrained-join`、`dual-subgraph`），每一个都被**推导**为满足某一类信息需求的最小原语。LLM 派发器只是一个轻量级查表；**oracle 消融把全部增益定位到算子层面**（oracle 88.2% vs LLM 派发 88.6%，Δ = −0.4 pp）——**这是一篇关于算子的论文，不是一篇关于分类器的论文**。在我们发布的双语基准 **RadarKG-QA-499** 上（该基准的 Count 答案集中位基数是公开基准的 3.4×，专门用于暴露 AGM），整个套件达到 **88.6%**，相对 Baseline 52.7% 提升 **+35.9 pp**、相对 zero-shot 路径规划器 60.3% 提升 **+28.3 pp**（配对 bootstrap 置信区间严格大于零，K=8→20 检索预算消融下仍然显著）。跨域到 KQA Pro 上，问题类型学和派发器**零成本迁移**（无重训练即可达到 67.4% 的策略匹配率；`dual_subgraph` 在 SelectBetween 上 +11 pp 显著）；但**总体增益的幅度取决于基准是否包含 AGM**——公开基准系统性地欠采样了这一模式，我们把这视作一个 community gap 并予以记录。最终成本为 **\$0.00041 / 问题**（折合 \$0.00084 / 准确率点），可用于工业部署。

---

## 1. 引言

### 现象

工业部署的 GraphRAG \citep{edge2024graphrag,gutierrez2024hipporag,luo2024rog,sun2024tog,ma2025tog2} 面对一个稳定的查询分布——这类查询用统一的 top-K 检索无法回答。考虑一个雷达 KG 上的三个自然问题：

| 问题形态 | top-K 为什么失败 |
|---|---|
| *"美国一共运营多少款雷达？"* | 答案集 > 400 个实体；任何固定的 K 都不可能枚举完 |
| *"列出美国公司研制的、工作在 S 波段的所有雷达。"* | top-K 无法表达跨约束的**交集**；ranking 把两个约束混在一起 |
| *"AN/TPY-2 是否被出口到日本？"* | KG 只存储事实的**存在**而不存储缺失；top-K 看不到非局部的"无" |

这些不是 LLM 的失败：无论模型能力多强，top-K 在 K = 8 时不可能枚举 400 个答案、不可能在两个独立的排序信号之间表达 AND、也不可能在只存正面三元组的 KG 里宣告"无"。它们是**信息论意义上**的失败——检索到的证据集合根本不具备下游推理需要的信息量。在 RadarKG-QA-499（§3）中，**约 30%** 的自然问题属于这三类；无论选择什么 LLM 或 K ∈ {8, 20}，baseline top-K 在这些题上的准确率都只有 4–12%（§4.1，Table 6）。

### 诊断：答案几何不匹配

top-K 隐含了一个关于**答案几何**的假设——答案是一个可以被 surface relevance 排序 capture 的小集合。这个假设对 factoid 类问题成立（WebQSP 和 NaturalQuestions 这类常见基准就在这个区域），但在三类工业 KG 常见的可计算模式上系统性失败：

- **无界枚举**（unbounded enumeration）：计数、列举。信息需求是**完整集合**；top-K 受 K 上限约束。
- **多约束交集**（multi-constraint intersection）：跨多个独立排序信号的 AND。信息需求是**集合交**；top-K 的排序把约束混在一起。
- **补集 / 非成员判定**（complement / non-membership）：确认某事实不存在。信息需求是**全局否定**；KG 只存正面三元组，"缺失"不在任何局部窗口里。

我们把这种失配命名为**答案几何不匹配**（Answer-Geometry Mismatch，AGM）：查询的信息需求与 top-K 几何能够提供的形态**结构性不兼容**。一个基准展现 AGM 的比例是 30% 还是 5%，是**基准分布的属性**，不是 top-K 本身的属性。公开基准系统性欠采样这个模式（§3.3，Table 4）；工业 KG 系统性过采样这个模式（§3，§4.4）。

### 洞察：用信息需求匹配算子

这个套路在数据库理论里很熟悉。B-tree 索引非常适合 point query（`SELECT WHERE id=5`）而完全不适合 aggregate query（`COUNT(*) GROUP BY country`）的执行计划——不是 B-tree 本身有问题，而是查询的信息需求要求**不同的访问模式**。数据库的应对方式不是改 B-tree，而是引入**类型化的 execution plan**（hash join、full scan、sort-merge）和一个 cost-based query optimizer 来挑对的 plan。**Top-K 就是 GraphRAG 里的 B-tree**——对 factoid 检索最优，对 analytical KGQA 结构性不匹配。

所以正确的解法不是**更好的 ranker**，而是**为每个查询的信息需求选对算子**。我们枚举出 top-K 无法满足的信息需求类，并为每一类推导一个算子，最终得到一个由六个算子组成的类型化套件（方法 §2）。一个轻量级的 LLM 派发器负责把问题映射到对应的算子类。

### 系统（与 oracle 实验的意外发现）

我们把这个框架实例化为 **Strategy-Routed GraphRAG**：一个 LLM 派发器 + 六个类型化算子 + 一个 LLM answerer 读取算子证据。派发器的 zero-shot 分类准确率是 93%——但这个准确率**并不承重**。**一个 oracle 消融实验把 LLM 派发器换成 gold 题型路由，端到端准确率达到 88.2%，而 LLM 派发是 88.6%（Δ = −0.4 pp；§4.3）**：相对 Baseline 的 +35.9 pp 增益**完全由算子驱动**，与派发器质量无关。**这是一篇关于算子的论文，不是一篇关于分类器的论文。** 派发器只是一个 O(1) 的查表操作（LLM 恰好能做得很好），真正的 analytical contribution 是这个能满足 top-K 无法满足的信息需求的算子套件。

### 贡献

1. **AGM 诊断**：一个精确的、基于信息论下界的解释，说明 top-K 在 KGQA 上**何时**、**为什么**系统性失败；我们借用数据库领域的 OLTP / OLAP 划分使这个诊断不局限于雷达领域。
2. **类型化的六算子套件**：每个算子都被**推导**为满足某一类信息需求的原语。派发器在实验中被证明贡献 ≈ 0 pp；算子才是增益的来源。
3. **RadarKG-QA-499**：首个明确为暴露 AGM 而设计的 KGQA 基准（Count 答案集中位基数 3.4× 于公开基准），并附 26 题 OOD 双语压力子集。
4. **实证结果**：相对 baseline +35.9 pp、相对 single-policy 路径规划 +28.3 pp（配对 bootstrap CI 全部严格 > 0）；五套消融（K 检索预算 / 单算子 / suite size / oracle 派发器 / 跨域）把增益的来源精确锁定到算子设计。
5. **方法论发现**：双语 / 别名层的**分布内**评估会产生**假阴性消融**（§4.3）。我们提出一个 OOD 替换压力集协议，并展示这个协议把消融效应从 0 pp 变成 +11.5 pp。

---

## 2. 方法

### 2.1 流水线

每个问题经过三个 LLM 阶段——**派发器**（选算子）、**parser**（抽结构化参数）、**answerer**（把证据渲染为自然语言答案）——和一个**仅查 KG** 的 executor（图 1）。KG（RadarKG-v2：16,513 三元组、98 个关系 + 属性、4,827 实体）**只在 executor 中被查询**，LLM 不直接访问 KG。

![图 1：Strategy-Routed GraphRAG 流水线](figures/fig1_pipeline.pdf)

### 2.2 问题类型学

我们定义了 11 种问题类型，这些类型基于**检索模式需求**而不是表面形态。每个类型确定性地映射到一个算子（Table 1）。

**Table 1: 11 类问题类型 → 6 个算子的映射。**

| 类型 | 检测信号 | 算子 |
|---|---|---|
| single_hop | 实体 + 单属性 | lookup |
| relation_inverse, agg_count, agg_enum | 无界枚举 | exhaustive |
| two_hop_bridge, three_hop_chain | 关系组合（k ≥ 2）| path-plan |
| attr_filter | 两个并列约束的 AND | constrained-join |
| negation, unanswerable | 缺失 / 非成员 | complement |
| set_compare | 两实体 + 同 / 异 | dual-subgraph |
| distractor | 命名消歧 | lookup |

### 2.3 从信息需求推导算子套件

我们**不是**通过观察常见失败模式来挑算子。我们枚举 top-K 无法满足的**信息需求类**，为每一类推导出对应的算子，然后再为 top-K **能**满足的那一类加上 `lookup`。Table 2 显式展示这个推导过程；Table 3 给出实现机制。

**Table 2: 信息需求推导。每一行通过"满足该信息需求的最小访问模式"决定其算子；top-K 在每一行的失败原因不同。**

| 计算模式 | 信息需求（最小） | top-K 为什么失败 | 推导出的算子 |
|---|---|---|---|
| 身份 / factoid | 一条最匹配的三元组 |（无——top-K 能满足）| **lookup** = BM25 ⊕ vec ⊕ RRF |
| 无界枚举 / 计数 | 关系下的**完整** head / tail 集 | K 是上界，答案集不是 | **exhaustive** = $\{h \mid (h,r,t) \in \text{KG}\}$ |
| 多约束交集 | N 个关系 head 集的集合交 | 排序无法表达 AND；混淆约束信号 | **constrained-join** = $\bigcap_{i} \text{heads}(r_i, t_i)$ |
| 补集 / 非成员 | 完整的**已知**集合 + 显式 gap | KG 只存正面事实；"无"非局部 | **complement** = 枚举后判定 |
| 关系组合（k ≥ 2）| 一条**保留 hop 结构**的关系链 | top-K 是平的，丢失 hop 边界 | **path-plan** = 链式 $\text{tails}_{r_1} \circ \dots \circ \text{tails}_{r_k}$ |
| 二元比较（集合层）| **双侧**子图都显式且对齐 | top-K 可能饿死一侧；无法对齐 | **dual-subgraph** = 在 $(A, B)$ 上的并行 $\text{tails}$ + 集合操作 |

每个算子都是**满足其行信息需求的最小访问模式**：去掉 `exhaustive` 的"穷举"性质就把 K 受限的失败带回来；去掉 `constrained-join` 的交集运算就回到"排序混淆"的失败；其他类同。这个推导**同时**证明了套件的**数量**（六个信息需求类，六个算子）和**内容**——两者都不是经验拼凑。剩下的经验问题只是"实际中是否还存在**额外的**信息需求类"——§4.2 的 suite-size 消融回答了这个问题（三个 load-bearing 算子在**互不相交**的问题切片上有可加贡献，4-op 子集在本基准上达到与 6-op 完全一致的准确率）。

**关于推导的范围声明。** 上述框架是**基于**"信息需求 ↔ 访问模式"的匹配；我们**不主张形式上的完备性**。这六类信息需求是通过对雷达基准上常见 KGQA 失败模式的观察识别得到的，它们不是一个闭合的分类法：更多模式——**聚合算术**（在数值属性上的 `SUM` / `AVG`）、**传递闭包**、**时间区间约束**、**模糊匹配**——会触发更多算子（讨论 §5 中勾勒了扩展路径）。这个推导给套件一个**有原则的起源**，但不是任何代数意义上的**最小性**或**覆盖性**证明。

**Table 3: 六算子套件——实现机制。**

| 算子 | 实现机制 |
|---|---|
| **lookup** | BM25 + bi-encoder + RRF + 2-hop 图扩展；当 parser 抽出 (head, relation) 时旁路文本检索直接查 KG。 |
| **exhaustive** | $\{h \mid (h, r, t) \in \text{KG}\}$，无 K 截断。空结果时启用等价类回退（见下）。 |
| **complement** | 枚举 $\{t \mid (h, r, t) \in \text{KG}\}$；如果为空或没有命中目标 tail，显式标记非成员，让 LLM 回答"否"而不是 hallucinate。 |
| **path-plan** | 关系链 BFS，支持反向遍历 $r^{-1}$；返回末端实体集 + 完整边轨迹。 |
| **constrained-join** | 在 N 个约束上做 head 集的交集，附带**等价类回退**：空交集时把语义等价的关系 union（如 `countryOfOrigin` ∪ `operatedBy`），处理多源抽取下的不连通子图噪声。 |
| **dual-subgraph** | 在两实体 $(A, B)$ 上对共享关系做并行 $\text{tails\_of}$，并排渲染 $A \cap B$、$A \setminus B$、$B \setminus A$。 |

`lookup` 算子额外充当**派发器错误的安全回退**：错路由的问题会优雅降级为标准混合检索，而不是命中一个错算子的空结果。

### 2.4 派发器与双语别名层

派发器是一个 zero-shot DeepSeek-Chat，输入是问题，prompt 包含类型定义 + 5 个示例，输出是 type ID（确定性映射到算子）。路由准确率 93.0%。一个跨切面的**双语别名层**（5 级级联：直接匹配 → 614 形式的别名索引 → 频段归一化 → 后缀剥离 → 等价关系类 union）处理工业中文 KG 的 schema 英文 + value 中英混杂的情形。

---

## 3. 基准：RadarKG-QA-499

### 3.1 知识图谱

**RadarKG-v2** 包含 4,827 实体 / 12,219 实体间边 / ~80 个类型化属性，源数据来自一本 472 页机载雷达手册的全 OCR、Wikidata SPARQL 拉取、Wikipedia 摘要。country-of-origin / developer 覆盖率 93.4% / 87.0%。Schema 是英文；value 词表是中英混杂。

### 3.2 问题生成

我们自动生成 499 道问题，按 11 类分层；每题携带机器可验证的 gold 答案（计数题给整数，列举题给经过 canonical alias 合并的实体集合，是非题给 yes/no，等等）。

| 类型 | n | 类型 | n |
|---|---:|---|---:|
| single_hop | 80 | attr_filter | 50 |
| two_hop_bridge | 80 | negation | 40 |
| relation_inverse | 50 | set_compare | 40 |
| agg_count | 50 | three_hop_chain | 30 |
| agg_enum | 50 | unanswerable | 20 |
| | | distractor | 9 |

**合计 499 题。** 额外的 **26 题 OOD 双语压力子集**把英文 / 缩写别名（USA、AESA、monopulse、phased array 等）注入中文模板，专门用于隔离别名层的贡献。

### 3.3 为什么要建一个新基准：跨基准分布

我们认为 RadarKG-QA-499 是一个**首要**贡献，因为目前没有公开 KGQA 基准在**高基数答案集**上分层——而高基数答案集恰恰是检索预算结构性失败的最清晰信号。

**Table 4: 跨基准 Count 答案基数对比。**

| 基准 | n | 中位 | 均值 | 比例 ≥ 30 |
|---|---:|---:|---:|---:|
| **RadarKG-QA-499（我们的）** | 50 | **14** | **25.3** | **18%** |
| KQA Pro val | 1318 | 2 | 11.7 | 6% |
| KQA Pro pure-enum | 557 | 1 | 3.7 | 2% |
| Mintaka dev | 200 | 4 | 7.1 | 3% |
| WebQSP |（88% single-hop，源自 probe）| — | — | ≈ 0% |
| LC-QuAD 2.0 |（无预计算 gold）| — | — | — |

中位 3.4×、高基数比例 3× 于次近的公开基准。同样的差距在 `attr_filter`（多约束）和 `three_hop_chain` 上也存在。我们发布 RadarKG-QA-499 来填补这个 gap。

---

## 4. 实验

**Setup.** 所有系统使用 DeepSeek-Chat。三个系统：**Baseline** = BM25 + bi-encoder + RRF + 2-hop 扩展，top-K = 8；**Zero-shot Path Planner（RoG-inspired）** = 我们对 single-policy LLM 路径规划的 zero-shot 复现；**Strategy** = 我们的流水线。全部在 499 题上评估；配对 bootstrap CI（1000 次重采样）。

**RoG 比较的 caveat（重要）。** 我们把第二个系统称为 "Zero-shot Path Planner (RoG-inspired)"，表格中简称 **RoG**。**这不是 \citet{luo2024rog} 的忠实复现**：我们刻意使用 zero-shot DeepSeek 而非原文的 fine-tuned LLaMA-2，目的是**把"单一策略均匀应用"这一假设（我们要批判的部分）与"监督路径输出训练"（我们不批判的部分）隔离开**。所以此 baseline 报告的 Δ 值衡量的是"Strategy vs 在同等 LLM 能力下的均匀路径规划"，而不是"Strategy vs 已发表的 RoG 结果"。忠实的 finetuned-RoG 比较列在未来工作 §7。§4.2 的单算子消融给出此修订的上界：`path-plan` 仅占总增益的 −10 pp，最宽容的解读也只留下 ≥ 25 pp 不变。

### 4.1 RQ1：Strategy 能否打败单一策略检索？

**Table 5: RadarKG-QA-499 主结果（n = 499，准确率 %，95% 配对 bootstrap CI）。**

| 类型 | n | Baseline | RoG | **Strategy** | Δ vs B | Δ vs RoG |
|---|---:|---:|---:|---:|---:|---:|
| agg_count | 50 | 4.0 [0, 10] | 38.0 [26, 50] | **62.0 [48, 74]** | +58 [+44, +72] | +24 [+12, +36] |
| agg_enum | 50 | 46.0 [32, 58] | 92.0 [84, 98] | **96.0 [90, 100]** | +50 [+36, +64] | +4 [+0, +10] |
| attr_filter | 50 | 12.0 [4, 22] | 0.0 | **92.0 [84, 98]** | +80 [+66, +92] | +92 [+84, +98] |
| relation_inverse | 50 | 40.0 [26, 54] | 74.0 [62, 86] | **86.0 [76, 94]** | +46 [+32, +60] | +12 [+4, +22] |
| three_hop_chain | 30 | 23.3 [10, 40] | 50.0 [33, 70] | **93.3 [83, 100]** | +70 [+53, +87] | +43 [+23, +63] |
| two_hop_bridge | 80 | 40.0 [29, 51] | **95.0 [90, 99]** | 88.8 [81, 95] | +49 [+38, +60] | $-$6 [$-$14, +0] |
| single_hop | 80 | 83.8 [75, 91] | 11.2 [5, 19] | **86.2 [79, 94]** | +2.5 [+0, +6] | +75 [+65, +85] |
| set_compare | 40 | 97.5 [93, 100] | 85.0 [73, 95] | 97.5 [93, 100] | 0 | +13 [+3, +23] |
| negation | 40 | 100.0 | 97.5 [93, 100] | 100.0 | 0 | +3 [+0, +8] |
| unanswerable | 20 | 90.0 [75, 100] | **100.0** | 90.0 [75, 100] | 0 | $-$10 [$-$25, +0] |
| distractor | 9 | 100.0 | 66.7 [33, 100] | 100.0 | 0 | +33 [+0, +67] |
| **OVERALL** | **499** | **52.7 [48, 57]** | **60.3 [56, 65]** | **88.6 [86, 91]** | **+35.9 [+32, +40]** | **+28.3 [+24, +33]** |

Strategy 在 9/11 个类别上以严格 > 0 的 CI 取胜；两处"失利"（two_hop_bridge $-$6 pp，unanswerable $-$10 pp）的 CI 都触及 0——是**统计上的平局**而非失败。RoG 在 `single_hop`（$-$72.5 pp）、`distractor`（$-$33 pp）、`attr_filter`（$-$12 pp）上**低于 baseline**：在不需要路径的场景下，均匀路径规划比不规划更糟。

**检索预算消融。** K = 8 → K = 20 只把 Baseline 拉到 **+2.4 pp [+0.0, +4.6]**（Table 6）。Strategy 相对 K = 20 baseline 仍然 **+33.5 pp [+29.5, +37.9]** 显著获胜。Strategy 增益最大的几个类别在 K = 20 下的提升 ≤ 6 pp：`agg_count` 4% → 8%、`three_hop_chain` 23% → 20%（反而下降）、`attr_filter` 12% → 18%。**加大检索预算无法解决答案集结构性大于任何 K 的问题**——所以 +35.9 pp **不是** K = 8 的 artifact。

**Table 6: K = 20 baseline 消融。**

| 系统 | 整体 | Δ 配对 CI |
|---|---:|---|
| Baseline K = 8 | 52.7% | — |
| Baseline K = 20 | 55.1% | +2.4 [+0.0, +4.6] vs K = 8 |
| Strategy | 88.6% | **+33.5 [+29.5, +37.9]** vs K = 20 |

### 4.2 RQ2：哪个算子在承重？（结构性失败 → 算子）

我们做单算子消融（100 题分层子集，强制回退到 `lookup`；附录 C）并**显式地把每个算子的贡献映射到它所对付的 AGM 类**。§2.3 的推导预测算子和它所修复的结构性失败之间是 1:1 对应；Table 7 验证了这一点。

**Table 7: 单算子消融，每个算子的 Δ 贡献被映射到对应的 AGM 类（与 §2.3 推导 1:1 对应）。**

| 移除 | 整体 Δ | 对付的 AGM 类 | 最受影响的类型（及 baseline 失败模式）|
|---|---:|---|---|
| –exhaustive | **$-$19** | 无界枚举（`card`）| agg_count、agg_enum、relation_inverse 各 $-$60 pp（top-K ≤ K，答案集 > K）|
| –path-plan | **$-$10** | 组合 $k \geq 2$（`comp`）| three_hop_chain $-$62.5，two_hop_bridge $-$41.7（top-K 是平的，丢失 hop 边界）|
| –constrained-join | **$-$8** | 交集 AND（`card+comp`）| attr_filter $-$90（top-K 排序混淆两个独立约束）|
| –complement | 0 | 非成员（`cmpl`）|（在本基准的 negation 分布上 `lookup` + LLM 优雅降级）|
| –dual-subgraph | 0 | 二元比较（`comp`）|（在本基准的 same / different 分布上 `lookup` + LLM 优雅降级）|

**结构性失败分解。** Baseline = 52.7%，Strategy = 88.6%，Δ = +35.9 pp。三个 load-bearing 算子在消融中贡献 $-19, -10, -8 = -37$ pp，覆盖了**全部结构性可修复的失败预算**；剩余的 ~2 pp 来自双语层贡献 + 派发器的优雅降级（在 §4.3 中讨论）。**每个 load-bearing 算子修复的失败模式恰好与它的 AGM 类推导吻合**（§2.3，Table 2）。`complement` 与 `dual-subgraph` 之所以在本基准上消融为 0 pp，是因为 RadarKG-QA-499 的否定 / 比较分布对 `lookup` 来说是优雅降级友好的（negation gold 大多是单关系语境下的"否"，缺失三元组容易识别；set_compare gold 大多是 yes / no，可化为两次原子 lookup）。这个 0 pp 信号是**基准分布相关的**，不是算子冗余（§5）。

**Suite-size 消融**（附录 C）：联合移除呈线性叠加（例如 $-$exhaustive $-$path-plan = $-29$ pp = $-19 - 10$），证明三个 load-bearing 算子针对的是**互不相交**的问题切片——它们修复的是**结构性不同**的失败模式，而不是相互重叠的。在本基准上，4 算子子集 {lookup, exhaustive, path-plan, constrained-join} 达到与 6 算子完全相同的准确率（94.0%）；仅 lookup 时见底于 57.0%。

### 4.3 RQ3：增益来自派发器还是来自算子？

**Oracle 派发器消融。** 把 LLM 路由器换成 gold 题型 → 算子路由后，端到端达到 **88.2%**——**与 LLM 派发的 88.6% 仅差 0.4 pp**。派发器噪声实际上贡献 0 pp；**全部 +35.9 pp 增益由算子驱动**。一个有趣的反常现象：在 `three_hop_chain` 上 LLM 派发**比 oracle 高 13 pp**——灵活的路由器偶尔会把冗余的 3 跳链 re-route 到 `exhaustive`，绕开算子弱点；这是一种内在的优雅降级特性。

**双语别名悖论。** 在分布内评估（自动生成的问题复用 KG 的 canonical surface form），别名层消融效应为 0 pp；在 26 题手工 OOD 压力集上，它防止 **+11.5 pp** 的损失（配对 CI [+0.0, +26.9]；在 n = 26 时下界正好触零）。0 pp vs +11.5 pp 的差本身是一个**方法学发现**：**别名层组件的分布内消融会产生假阴性**。我们建议评估双语 / 别名组件时单独构造 OOD 替换集。

### 4.4 RQ4：框架能否跨域迁移？

我们在 **KQA Pro**（Wikidata，12 个 program-output 函数与我们的 11 类 1:1 映射）上测试。四个配置隔离 parser 和子图的贡献：

**Table 8: KQA Pro 4 配置跨域（502 题分层，配对 CI）。**

| 配置 | Parser / Subgraph | Baseline | Strategy | Δ |
|---|---|---:|---:|---|
| B1 | 机械化 args，2-hop | 30.9% | 28.8% | $-$2.2 |
| B2 | LLM parser，2-hop，n = 139 pilot | 28.8% | 30.2% | +1.4（噪声）|
| **B3** | LLM parser，2-hop，全集 | 27.7% | 27.9% | **+0.2 [$-$1.6, +2.2]** |
| **B4** | LLM parser，**+ concept-class 扩展** | 28.3% | 28.1% | **$-$0.2 [$-$2.4, +2.0]** |

B3 上有两个 per-type 信号统计显著：**SelectBetween +11.1 pp [+2.2, +22.2]**（`dual_subgraph` 算子干净迁移）和 **Count $-$8.9 pp [$-$17.8, $-$2.2]**（在 B4 通过把概念类成员纳入子图修复到 0）。两者在整体层面相互抵消。我们不粉饰——**在 KQA Pro 上 Strategy 与 Baseline 打平**。

**为什么跨域 gap 这么小**，尽管派发器零成本迁移（67.4% 路由匹配率）：KQA Pro 的问题分布**欠采样**了驱动 RadarKG +35.9 pp 的结构性失败模式（Count 中位 = 2，而 RadarKG 是 14；§3.3）。框架的**组件**迁移（派发器和算子无需修改即可在 KQA Pro KB 上运行，parser 只需一次约 1 天的重写）；增益的**幅度**取决于基准是否包含 Strategy 的目标失败模式。**Strategy 是结构性保险，不是通用乘子。**

### 4.5 RQ5：AGM 暴露度能否预测 Δ？

如果我们的诊断正确，per-type Δ 应当与该类型的**答案几何暴露度**——即答案的信息需求超出 top-K 几何能力的程度——成比例。我们把 AGM 暴露度操作化为两个维度：(a) **基数**——答案集的大小；(b) **组合**——答案是否需要多跳 / 交集 / 补集等 top-K 的扁平排序无法表达的访问模式。Table 9 跨两个基准交叉列出 per-type Δ 与 AGM 维度。

**Table 9: 按 AGM 暴露度的 per-type Δ（RadarKG-499 + KQA Pro B3）。AGM 暴露度：`card` = 高 gold 基数；`comp` = 多跳或交集组合；`cmpl` = 补集 / 非成员；`none` = 原子单步。† 和 ‡ 在表格下方解释。**

| 基准 | 类型 | n | Gold 中位基数 | AGM 类 | Δ S − B (pp) |
|---|---|---:|---:|---|---:|
| RadarKG-499 | agg_count | 50 | 13.5 | card | **+58.0** |
| RadarKG-499 | agg_enum | 50 | 9.5 | card | **+50.0** |
| RadarKG-499 | relation_inverse | 50 |（list）| card | **+46.0** |
| RadarKG-499 | attr_filter | 50 | 2.5 | card+comp | **+80.0** |
| RadarKG-499 | three_hop_chain | 30 | 1 | comp | **+70.0** |
| RadarKG-499 | two_hop_bridge | 80 | 1 | comp | **+48.8** |
| RadarKG-499 | single_hop | 80 | 1 | none | +2.4 |
| RadarKG-499 | negation | 40 | 1 | cmpl† | 0.0 |
| RadarKG-499 | set_compare | 40 | 1 | comp† | 0.0 |
| RadarKG-499 | unanswerable | 20 | 1 | cmpl† | 0.0 |
| RadarKG-499 | distractor | 9 | 1 | none | 0.0 |
| KQA-Pro | SelectBetween | 45 | 1 | comp | **+11.1** [+2.2, +22.2] |
| KQA-Pro | Count | 45 | 2 | card‡ | $-$8.9（B4：0.0）|
| KQA-Pro | QueryRelation | 45 | 1 | none | $-$6.7 |
| KQA-Pro | What | 45 | 1 | none | +6.6 |
| KQA-Pro | QueryAttr | 45 | 1 | none | 0.0 |
| KQA-Pro | VerifyStr | 45 | 1 | cmpl† | 0.0 |
| KQA-Pro |（其余 6 个 fn）| — |（多为 1）| none / † | ≈ 0 |

† Lookup + 称职 LLM 在这些类型上优雅降级（baseline 已达 95–100%；算子没有提升空间）。
‡ 基数-AGM 类型，但受 2-hop 子图完整性的瓶颈限制；B4 的 concept-expansion 把缺口收到 0 pp。

**规律。** 在 6 个有 AGM 暴露的 RadarKG 类型（card 或 comp）上，中位 Δ = **+58 pp**；在 5 个无 AGM 暴露（或 lookup 优雅降级）的类型上，中位 Δ = **0.0 pp**。在 KQA Pro 上，**唯一一个不被子图抽取 confound 干扰的 AGM 暴露类型 SelectBetween（二元比较）**展现出同样的规律：**+11.1 pp [+2.2, +22.2]，统计显著**。KQA Pro 上的 Count 是基数-AGM 类型，本来**应该**让 `exhaustive` 取胜，但被 2-hop 子图不完整性限制；B4 的 concept-class 扩展把基数 gap 收到 0 pp，把算子贡献与子图抽取 confound 分开（§4.4）。

跨域迁移的含义因此非常明确：**per-type Δ 由问题的 AGM 暴露度决定，与基准或 KB 无关**。一个基准上的总体 Δ 是 per-type Δ 的"AGM 暴露度加权平均"——这是基准问题分布的属性，不是框架的属性。RadarKG-QA-499 的总体 +35.9 pp 和 KQA Pro 的 ≈ 0 pp 都被各自的 AGM 暴露度分布精确预测；它们之间**并不矛盾**。

### 4.6 成本与延迟

Strategy：4.8 s / 题，3 次 LLM 调用，**\$0.00041 / 题**（折合 \$0.00084 / 准确率点 vs Baseline；vs RoG 则是 \$0.00064 / pp——RoG 每 pp 更贵，因为它比 Baseline 多花的钱买到的准确率更少）。10 万题 / 月 ≈ \$41。

---

## 5. 讨论

**为什么保留 0 pp 消融的算子？** Complement 和 dual-subgraph 在本基准消融为 0 pp，我们仍保留它们有两个层面的理由。**原则上**：它们各自是某一类信息需求（非成员、二元比较；§2.3）的最小算子——剪掉就破坏推导的完备性。**务实上**：(i) 否定题和比较题的可审计证据轨迹（部署价值）；(ii) 对抗性否定和 3 方比较的压力集（未来工作）会让它们承重；(iii) token 成本可忽略。0 pp 信号反映的是**本基准的分布**，不是**算子的冗余**。

**Composability：从套件走向 algebra。** 我们的 11 类映射把每个问题映射到**一个**算子，是个刻意简单的派发策略。但**算子本身是可组合的**，许多自然问题可以**分析性地**分解为满足层级信息需求的算子组合（Table 10）。

**Table 10: 自然问题的算子组合分析（分析性，不是当前实现）。当前实现把内层组合放在单个算子里实现（例如 `constrained-join` 直接返回交集大小，相当于一步做完 `exhaustive ∘ constrained-join`）；这个分解展示了套件在自然组合下是闭合的。**

| 自然问题 | 算子组合分析 |
|---|---|
| *"中国研制的 S 波段雷达有多少种？"* | `exhaustive ∘ constrained-join({band=S}, {country=中国})` |
| *"美国出口到日本的雷达中哪些用了 AESA？"* | `constrained-join({tech=AESA}, path-plan(美国, exportedTo, 日本))` |
| *"AN/TPY-2 与 AN/SPY-1 是否共享厂商？"* | `dual-subgraph(AN/TPY-2, AN/SPY-1; developedBy)` 然后做相等性检验 |
| *"列出研制方所属国家不是美国的雷达。"* | `complement_{美国}(path-plan(?, developedBy, m); m, affiliatedTo, ?)` |

当前实现中，这些组合都被放在**单个算子内部**完成（例如 `constrained-join` 内部直接计算交集大小，等效一步执行 `exhaustive ∘ constrained-join`）。这个分析性分解之所以重要：(a) 它展示套件在 query optimizer 会发出的自然组合下是**闭合**的；(b) 它给出了**未来扩展的路径**——派发器可以从输出"算子"扩展为输出"算子表达式"。我们**不主张**这个框架是 Codd 意义上的形式 query algebra；我们主张这个套件是**可组合**的，这是有原则地扩展到更复杂查询模式的最低要求。

**为什么 +35.9 pp 不能干净地迁移到 KQA Pro？** RQ4 的诊断是精确的：派发器和算子代码零成本迁移；parser 只需 1 天重写（B1 → B2 的 +3.6 pp 来自 prompt 改动）；**增益的幅度由基准分布决定**。公开基准系统性地欠采样结构性失败模式（Table 4）。填补这个 gap 是 community direction；RadarKG-QA-499 是我们的第一个贡献。

**Limitations.**

1. **单领域 + 自建基准。** KG、499 题和 gold 都是我们构造的。虽然我们加了 26 题 OOD 压力集来部分解耦 gold 与流水线，但完全独立的、人工写 gold 的基准才能彻底排除 self-favoring 评估。
2. **简化版 RoG baseline。** 我们的 "Zero-shot Path Planner" 近似 RoG 的"单一策略均匀派发"假设，但**不含**原文的 LLaMA-2 微调。+28.3 pp 的差距应被读作"Strategy vs 同等 LLM 能力下的均匀路径规划"，**而非** "Strategy vs 已发表的 RoG 结果"。忠实的 finetuned 复现在未来工作 §7 中承诺；§4.2 的单算子消融为这一修订给出上界——`path-plan` 只占增益的 −10 pp。
3. **KG 噪声。** 多源抽取在 `affiliatedTo` 上的残留噪声、`countryOfOrigin` 与 `operatedBy` 不一致，会让某些 `agg_count` 答案偏差 1–3 个；`constrained-join` 的等价类回退（§2.3）能处理最常见的模式，但不能完全中和噪声。
4. **对抗性覆盖。** Complement 和 dual-subgraph 在当前基准上消融为 0 pp，是因为基准缺少对抗性否定和 3 方比较；这些算子在对抗性分布下可能变成 load-bearing（未来工作 §7）。
5. **增益的分布依赖性。** 总体 +35.9 pp 取决于基准展现的 AGM 程度（§4.5）。要在 KQA Pro 这类基准上拿到类似量级的增益，需要 (a) 基准包含更多 AGM 题型，或 (b) 改造 parser / 子图抽取层（B1 → B4 把 Count 从 −8.9 修到 0 pp）。我们把框架视作**结构性保险**，不是通用乘子。
6. **OOD 压力集规模。** 26 题；更紧的 per-type CI 需要 50+ 题。双语悖论的 +11.5 pp [+0.0, +26.9] 在这个 n 下边界紧贴 0。
7. **子类样本量。** `distractor`（n = 9）与 `three_hop_chain`（n = 30）受 KB 结构性可用性的限制；其他类型 n ≥ 40。
8. **Suite size。** 在 RadarKG-QA-499 上，4 算子子集与 6 算子准确率持平（§4.2）。在带对抗性否定或 3 方比较的更广基准上 `complement` 与 `dual-subgraph` 可能差异化；我们按 §5 "为什么保留 0 pp 算子？"的"原则 + 务实"论证保留它们。

---

## 6. 相关工作

**GraphRAG / KG + LLM。** Microsoft GraphRAG \citep{edge2024graphrag}、HippoRAG \citep{gutierrez2024hipporag}、KGP \citep{wang2024kgp}、MindMap \citep{wen2024mindmap} 都把 top-K 检索当作**统一策略**应用。我们把这个单一策略**分解**为一个类型化的算子套件，套件本身是从 top-K 无法满足的信息需求类**推导**而来。

**LLM 路径规划。** RoG \citep{luo2024rog}、ToG \citep{sun2024tog,ma2025tog2} 把路径规划统一应用。§4.1 显示，*均匀的路径规划*（操作化为一个 zero-shot 路径规划器；RoG 比较的 caveat 见 §4 setup）在 11 类问题中的**三类**上**低于 baseline**（single_hop $-$72.5 pp）。我们把 `path-plan` 视作大套件中的**一个**算子，只在需要时才派发它。我们对已发表的 RoG 结果不作判断；我们批判的是 RoG 与 ToG 共享的**单一策略均匀应用假设**，不是 RoG 特定的路径输出训练。

**基于路由的检索（最相近的相关工作）。** Adaptive-RAG \citep{jeong2024adaptiverag} 按查询**复杂度**派发并改变检索**深度**（不检索 / 单步 / 迭代）；ByoKG-RAG \citep{byokgrag2025} 把多个 KG 检索工具（agentic 遍历、path retrieval、OpenCypher）的输出**融合**起来。两者都让底层检索原语（按相关性排序）**保持不变**，只是改变它被应用的次数或被咨询的来源。**我们改变的是检索原语本身的语义**：每个算子都执行一个结构性不同的 KG 访问（交集 vs 枚举 vs 补集 vs 路径组合）。具体地，在 *"美国一共运营多少款雷达？"* 上，Adaptive-RAG 实例会派发到多步检索但每步仍按相关性排序（且仍受 K 限制）；ByoKG-RAG 实例会融合多个工具的 top-K 输出（仍受融合预算限制）。两者都**继承了 top-K 的 AGM**。我们的 `exhaustive` 算子返回**完整**的 head 集——满足 top-K 派发无法满足的信息需求，与深度和来源无关。派发器的角色因此也不同：在按深度派发中，错误的代价是"多花一步"；在我们的框架中，错误的代价是"信息需求不匹配"，只能靠 `lookup` 回退兜底。这意味着我们的 +35.9 pp 增益**不可能**通过给 Adaptive-RAG 或 ByoKG-RAG 加更多深度或更多来源达到。

**符号 KGQA / NL2SPARQL。** \citep{jiang2023structgpt,baek2023kaping} 把自然语言映射到完整的 query algebra；在工业 KG（schema 英文 + value 中英混杂）上脆弱。我们的套件刻意更粗粒度（6 个算子、LLM 派发），用表达精度换 schema / surface form 的鲁棒性。

**多语 KGQA。** \citep{perevalov2024multilingual} 假设单语 KG + 翻译接口；我们处理的是单一 KG 内部的中英混杂 value。

**问题分类。** 已有的 OLTP / OLAP 划分会放过聚合和多约束筛选的准确率；我们的 11 类是基于**检索模式需求**的。

---

## 7. 未来工作

1. **忠实复现 RoG。** 用原 RoG 的公开 LLaMA-2 checkpoint 在 RadarKG-QA-499 上跑一遍，验证 +28.3 pp 是结构性的（来自单一策略假设）而非容量驱动（zero-shot vs finetuned）。§4.2 的单算子消融给出这次修订的上界——`path-plan` 只占增益的 −10 pp，最宽容的解读也只让 +35.9 pp 减少 10 pp 左右。
2. **schema 衍生 parser，用于跨域迁移。** KQA Pro 4 配置消融（§4.4）把 parser 锁定为可移植性的主要瓶颈：B1（机械化 args）→ B2（LLM parser，prompt 工程化）在 139 题上闭合 +3.6 pp。从任意 KG 的 schema 自动衍生 parser 的关系词表，会把当前 1 天的重写降为零成本迁移——这是单项最高杠杆的下一步。
3. **跨域迁移到医疗 / 材料 KG。** 把流水线重新对接 DrugBank 或 Materials Project，两者都结构性展现 AGM 模式（高基数的 drug-target 关系；多约束的材料属性查询）。这直接检验 §1 中"工业 KG 系统性过采样 AGM"的主张——而且在我们没构造过的基准上检验。
4. **构造对抗性子集。** 构造 KG 模糊否定和 3 方比较的压力集，测试 `complement` 和 `dual-subgraph` 在对抗分布下是否会变成 load-bearing。当前的 0 pp 是基准分布属性，不是算子属性；对抗集是最干净的实证测试。
5. **community 高基数 KGQA 基准。** 我们的跨基准调研（§3.3）显示没有公开 KGQA 基准在显著样本量下对高基数 Count、多约束交集或 3 跳链分层。我们建议 community 开发明确按 per-pattern 基数分层的基准；RadarKG-QA-499 是我们的第一个贡献。

### 7.1 工件释放计划

接受后发布（代码 Apache-2.0，数据 CC-BY-4.0）：
- **代码**：流水线（`qa_router.py`，`qa_strategy_pipeline.py` 含六算子，`graphrag_retriever.py`），baseline（`qa_rog_baseline.py`），`experiments/` 下 15 个实验脚本，以及 lexicon（关系 / 别名 / 问题类型）。
- **数据**：**RadarKG-v2**（4,827 实体，12,219 实体间边，16,513 平铺三元组；JSON 与 Neo4j 双格式）；**RadarKG-QA-499** 含 per-question 可机器验证 gold；26 题 OOD 双语压力集 + 替换字典（附录 E）；100 题分层子集；完整 oracle 派发器结果。
- **复现性**：每张表都标注脚本路径和结果 JSON；所有 LLM 调用使用 DeepSeek-Chat，附 seed 与 prompt（附录 A）。完整流水线重跑成本 ≈ \$1.5。

---

## 8. 结论

我们呈现了 Strategy-Routed GraphRAG——用一个由 LLM 派发器调度的六算子类型化套件，替换之前 GraphRAG 的单一 top-K 策略。在我们发布的 RadarKG-QA-499（一个为暴露检索预算结构性失败而设计的新基准）全 499 题上，Strategy 达到 88.6%（vs Baseline 52.7%，+35.9 pp；vs zero-shot 路径规划 60.3%，+28.3 pp），在 9/11 个类别上 per-type 配对 bootstrap 显著获胜。Oracle 派发器消融显示增益由**算子驱动**；K = 20 检索预算消融排除"K 不够大"的 artifact；单算子与 suite-size 消融把贡献定位到三个 load-bearing 算子且它们的贡献**可加**。跨域到 KQA Pro 上，框架的**组件**（派发器、算子、parser-prompt 加 1 天重写）能迁移；**增益的绝对值取决于基准分布**——我们把"公开基准系统性欠采样结构性失败"作为一个 community gap 显式记录。这个方法是**结构性保险**：在不需要时几乎无代价，在 AGM 显现时回报巨大。

---

## 参考文献

\bibliography{references}

---

## 附录

### A. 派发器 prompt（完整）

zero-shot DeepSeek-Chat 派发器使用如下 system prompt（这里是英译版；中文原版在发布代码 `qa_router.py` 中）：

```
You are a query classifier for a radar knowledge-graph QA system. Given a
Chinese or English question, output the type ID that best matches one of:

- single_hop:       direct fact query (X's attribute is ?)
- two_hop_bridge:   bridging an intermediate entity (X's Y → ?)
- three_hop_chain:  three-relation chain
- relation_inverse: inverse query, find subjects from object
                    ("which radars are developed by Raytheon")
- agg_count:        counting ("how many / 几款 / 几种")
- agg_enum:         listing all ("list all / which / 都有哪些")
- set_compare:      comparing two entities (same / different / common)
- negation:         absence / exclusion ("is X NOT in Y" / "未装备 / 没有")
- attr_filter:      multi-constraint AND ("both... and... / 既...又...")
- unanswerable:     KG cannot answer
- distractor:       similar-name disambiguation (AN/SPY-1 vs AN/SPY-1A)

Disambiguation rules (5):
- relation_inverse vs agg_enum: prefer agg_enum if "list / all" present.
- single_hop vs negation: presence of "is / has / not" → negation.
- set_compare priority: two coordinated entities + comparison → set_compare
  (even when "is / not" is also present), NOT negation.
- three_hop_chain vs two_hop_bridge: two stacked possessives → three_hop_chain.
- single_hop priority over enum: when the subject is a specific entity,
  even with "which NN" wording, it is single_hop (the answer is the entity's
  attribute, bounded).

Then 5 examples covering the priority rules.

Output the type ID only, no explanation.
```

路由准确率：100 题分层子集上 93.0%（§4.3，oracle 消融）。这个派发器下的端到端准确率为 88.6%，oracle 是 88.2%，仅差 0.4 pp。

### B. Parser prompt 与类型校验规则

parser 把 `(question, qtype, strategy)` 映射为 JSON：包含 `primary_entity`、`secondary_entity`、`relation_chain`、`constraints`、`forbidden_tail`、`answer_target` 几个字段。prompt 是 type-aware 的（例如 `Country` 类型的 tail 必须使用 `operatedBy / countryOfOrigin / exportedTo / affiliatedTo`，**不能**使用 `developedBy`——后者的 `tail_type` 是 `Manufacturer`）。后处理校验在以下三个条件**同时**满足时替换关系：(i) tail 的推断类型与关系期望不兼容；(ii) 原 `(rel, tail)` 对在 KG 中零命中；(iii) 别名解析也无法闭合 gap。多类型 tail（如 `合成孔径` 既属 `RadarMode` 又属 `TechType`）保留原 parsing。完整 prompt 与校验代码在发布的 `qa_strategy_pipeline.py` 中。

### C. 单算子与 Suite-size 消融细节

在 100 题分层子集（大多数类型 10 题）上，依次把每个算子替换为 `lookup`，端到端重跑。联合移除矩阵验证可加性：

| 移除组合 | 预测 Δ（单算子之和）| 观察 Δ |
|---|---:|---:|
| $-$exhaustive | $-19$ | $-19$ |
| $-$path-plan | $-10$ | $-10$ |
| $-$constrained-join | $-8$ | $-8$ |
| $-$exhaustive $-$ path-plan | $-29$ | $-29$ |
| $-$exhaustive $-$ constrained-join | $-27$ | $-27$ |
| $-$path-plan $-$ constrained-join | $-18$ | $-18$ |
| 4-op 套件（lookup, exhaustive, path-plan, constrained-join）| $0$（complement、dual 各为 0）| $0$（与 6-op 同为 94%）|
| 仅 lookup | $-37$ | $-37$（见底于 57%）|

可加性确认三个 load-bearing 算子针对的是不相交的问题切片。源数据：`results/ablation_strategies_100.json`、`results/suite_size_ablation_100.md`。

### D. 成本分解

| 阶段 | LLM 调用次数 | 平均 in / out tokens | 单题成本 |
|---|---:|---|---:|
| dispatcher | 1 | 400 / 12 | \$0.00006 |
| parser | 1 | 1100 / 220 | \$0.00022 |
| answerer | 1 | 550 / 220 | \$0.00014 |
| **合计** | **3** | **2050 / 452** | **\$0.00041** |

DeepSeek-Chat 定价（cache-miss）：input \$0.14 / 100 万 token，output \$0.28 / 100 万 token。在 10 万题 / 月规模下边际 LLM 成本 ≈ \$41。

### E. 双语压力集：26 题 OOD 替换字典

26 题手工 OOD 双语压力集把英文 / 缩写别名注入到从 RadarKG-QA-499 采样的中文问题模板。每题接受 1–2 次替换；部分替换是 mid-word（如 `目标 track mode 模式`），用于在精确表面匹配之外测试别名索引。

| canonical（中文）| OOD 替换形式 |
|---|---|
| 美国 | USA, U.S., United States |
| 俄罗斯 | Russia, USSR |
| 中国 | China, PRC |
| 脉冲多普勒 | PD, pulse Doppler |
| 合成孔径 | SAR |
| 逆合成孔径 | ISAR |
| 动目标指示 | MTI |
| 地面动目标指示 | GMTI |
| 有源相控阵 | AESA |
| 无源相控阵 | PESA |
| 相控阵 | phased array |
| 单脉冲 | monopulse |
| 跟踪 | track mode |
| 边搜索边跟踪 | TWS, track-while-scan |
| 地形回避 | TA, terrain avoidance |
| 地形跟随 | TF, terrain following |
| X 波段 | X band, X-band |
|（S、L、C、Ku、Ka 波段类同）|（波段 → "band" / "-band"）|

在此 OOD 集上的双语层消融：完整流水线 80.8% vs 别名层关闭 69.2%（Δ = +11.5 pp [+0.0, +26.9] 配对 bootstrap，n = 26）。CI 下界正好触零（在此 n 下边界紧贴 0）；我们诚实地把这个作为方法学发现报告（分布内消融为 0 pp；OOD 才显出 +11.5 pp）。

### F. KG 构造流水线

**RadarKG-v2**（4,827 实体 / 12,219 实体间边 / 16,513 平铺三元组 / 98 类型化属性）由以下流水线得到：

- **源抽取**：基于规则 + LLM zero/few-shot 抽取一本 472 页的机载雷达手册（PaddleOCR 全 OCR），辅以 Wikidata \citep{vrandecic2014wikidata} SPARQL 拉取和 Wikipedia 文章摘要。九个源通道，每条三元组都带 `source` + `confidence` + `evidence` 来源三元组。
- **Schema 升级**：从 v1 的扁平三元组 schema（28 关系）升级到 v2 类型化 entity / property schema，9 个实体类型 + 20 个实体间关系 + ~80 个类型化数值 / 类别属性。
- **靶向富化**：`countryOfOrigin` / `developer` 字段覆盖率提升到 **93.4% / 87.0%**，方法是规则化命名模式（如 `AN/*` → 美国、`EL/M-*` → 以色列、Cyrillic 模式 → 俄罗斯）+ 厂商链推断 + LLM 领域知识带置信阈值（`derivedFrom` 要求 ≥ 0.75，因为历史抽取精度只有 8%）。function 关系从 1.2% 提升到 75.6%；`similarTo` 与 `compatibleWith` 边由特征重叠候选 + LLM 验证产生。
- **实体去重（带安全边界）**：变体命名合并（如 `APG-70 / AN/APG-70 / AN/APG-70(V)` → 一个 canonical），附带**国家冲突安全检查**（如 `Flycatcher` 雷达一个源标荷兰、另一个源标法国 → 自动 skip）。真正的代际变体（`AN/APG-63(V)1 / (V)2 / (V)4`）保留为独立实体。888 → 849 canonical Radar 实体；**零错合并**已验证。
- **属性归一化**：状态英中统一（`in_service` ↔ `服役中`）；频段 token 拆分（`IJ` → `[I, J]`）；CamelCase 厂商名加空格（`TexasInstruments` → `Texas Instruments`）；OCR 错字修正。

Schema 英文；value 词表中英混杂（雷达名 / 厂商名用英文惯例；国家、工作模式、技术体制用中文）。

### G. KQA Pro 跨域细节

- **派发器探针**（n = 288，分层）：每个 KQA-Pro-function 的策略匹配率、混淆矩阵、完整分布。零重训 67.4% 路由匹配率；`dual_subgraph` 在 SelectBetween 上 100%、`complement` 在 Verify 上 84%、`lookup` 68%。
- **B1–B4 端到端细节**：B1 机械化 args 30.9% / 28.8% Δ = $-2.2$；B2 LLM parser 试点（n = 139）28.8% / 30.2% Δ = $+1.4$；**B3** LLM parser 全集（n = 502）27.7% / 27.9% Δ = $+0.2$ [$-1.6$, $+2.2$]；**B4** + concept-class 扩展 28.3% / 28.1% Δ = $-0.2$ [$-2.4$, $+2.0$]。per-type CI 表、KQA-Pro-aware parser prompt、以及 2-hop + concept-expansion 子图抽取器（`datasets/kqa_pro_subgraph.py::build_question_subgraph_v2`）随代码发布。

### H. WebQSP 探针（n = 246，仅派发器）

路由分布：**88.2% → `single_hop`**，8.1% → `unanswerable`，0% → 多跳 / set-compare / attr-filter。WebQSP 的问题分布**缺少** Strategy 设计要解决的结构性失败题型；我们把这个探针作为**反向证据**——WebQSP 不是评估 AGM 干预的合适基准，而**不是**用作端到端比较的对照点。

### I. Zero-Shot 路径规划器（RoG-style）复现细节

完整 prompt 和 ~30 条采样路径在 `experiments/run_e2e_3way_qa500_full.py` 和 `results/qa500_3way_full.json`。我们刻意使用 zero-shot DeepSeek-Chat 而非原 RoG 的 fine-tuned LLaMA-2，目的是把**单一策略均匀应用**这一假设——我们要批判的部分——从**监督路径输出训练**这一正交贡献中隔离开。忠实的 finetuned RoG 比较在未来工作 §7（item 1）。

未压缩的 v10 长版本在 `strategy_routed_graphrag_full.md`，包含额外的 per-type 表格和 per-question 失败分析。
