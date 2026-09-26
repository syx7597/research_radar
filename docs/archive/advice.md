Let我先仔细读一下你的论文。我仔细读完了整篇论文。你的担忧很准确——这篇文章目前最大的风险就是reviewer会说"这不就是做了个问题分类器，然后对不同类型的问题手写了不同的retrieval策略吗？" 下面我从**根本问题诊断**和**具体优化建议**两个层面来分析。

---

## 核心诊断：为什么现在读起来像"只是分类"

你的系统其实有一个很强的洞察——**top-K retrieval在结构上无法处理某些query pattern**——但论文的叙事结构把这个洞察埋在了 dispatcher / typology 的包装之下。目前读起来的逻辑链是：

> 定义11种问题类型 → 训练分类器 → 每类问题派发到对应算子

这个叙事框架本身就在暗示"分类是核心贡献"。讽刺的是，你自己的oracle实验（Table 4）其实已经证明了**分类不重要**（LLM dispatcher vs oracle只差0.4pp），真正重要的是算子本身。但你没有把这个insight变成论文的主旋律。

---

## 具体优化建议（按优先级排序）

### 1. 重构叙事主线：从"分类+派发"转向"结构性不可解 + 算子设计"

现在论文的title和framing都在强调"Strategy-Routed"（路由/分类），建议整体转向：

**核心论点不应该是"我们做了更好的routing"，而是"top-K是结构性错误的，我们设计了能覆盖不同计算模式的算子集，routing只是一个轻量级的index"**

具体做法：

- Introduction的三个例子（counting / intersection / negation）非常好，但应该进一步提炼为一个**不可能性论证**（impossibility argument）：不是"top-K表现不好"，而是"对于answer set > K的问题，top-K在*信息论意义上*不可能正确"。这个论证可以很简单——如果answer set有400个entity，K=8只能采样2%，任何基于这2%的计数都是错的，这不是LLM能力的问题，是信息量不足。
- 把Section 5.4的oracle实验**提前到Introduction的argument里**作为预告："我们的实验证明，整个+35.9pp的gain几乎全部来自算子设计，而非分类质量"。这直接堵住"这就是分类"的质疑。

### 2. 给算子设计增加理论深度：引入"计算模式-信息需求"的分析框架

现在六个算子的选择是"by inspection of common KGQA failure modes"，这听起来很ad hoc。建议增加一个分析维度：

对每个算子，明确其**信息需求特征**（information requirement profile）：

| 计算模式 | 信息需求 | 为什么top-K失败 | 算子的设计原理 |
|---|---|---|---|
| 计数/枚举 | 需要**完整集合** | K是上界，集合无上界 | exhaustive: 无K截断 |
| 多约束过滤 | 需要**交集语义** | ranking不能表达AND | constrained_join: 集合交运算 |
| 否定/缺失 | 需要**全局否定** | KG只存在正面三元组 | complement: 枚举已知尾部，验证非成员 |
| 关系组合 | 需要**路径语义** | top-K是flat的，不保留hop结构 | path_plan: 链式遍历 |

这样每个算子的设计就不是"我们试了试发现这样好"，而是"给定这种计算模式的信息需求，这是唯一合理的retrieval策略"。这个框架把算子设计从engineering artifact提升为了一个有分析基础的设计选择。

### 3. 证明算子的可组合性（Composability）

现在的架构是严格的1-question → 1-operator映射，这确实看起来像分类。如果能展示**算子组合**的场景，就真正像一个algebra了：

- 比如一个问题"中国研制的、工作在X波段的雷达有多少种？"可以分解为 `exhaustive(constrained_join(country=中国, band=X))`——先做交集过滤，再做计数。
- 再比如"美国出口到日本的雷达中，哪些具有AESA技术？"可以是 `constrained_join(path_plan(美国→export→日本), tech=AESA)`

即使你现在的系统实际上是用constrained_join一步完成的，把它**分析性地展示为算子组合**也能大大增强"algebra"的说服力。至少可以在Discussion里加一个小节讨论组合性，并给2-3个例子。

### 4. 增加typology粒度的消融实验

一个很有说服力的实验是：**不同粒度的分类对结果的影响**。比如：

- 粗粒度：3类（lookup / exhaustive / structured）→ 准确率多少？
- 中粒度：6类（对应6个算子）→ 准确率多少？
- 细粒度：11类（你现在的方案）→ 88.6%

如果3类→6类有大幅提升，但6类→11类提升很小，那说明**关键不是分类的精细度，而是算子的结构性差异**——直接反驳"这只是分类"的批评。如果11类明显优于6类，也说明typology的设计本身有价值。无论哪种结果都能给出有意义的分析。

### 5. 弱化"algebra"的说法，强化"operator decomposition"

你在论文里多次小心地disclaiming "不是formal algebra"，但title还是叫"Retrieval Operator Algebra"。这种反差会让reviewer不舒服。建议：

- Title改为类似 **"Strategy-Routed GraphRAG: Typed Retrieval Operators for Structural Failures in Knowledge Graph QA"**
- 正文中用"operator decomposition"或"operator suite"替代"algebra"（你已经在部分地方这样做了，但不一致）
- 如果想保留algebra的类比，用一个专门的小节（比如在Discussion中）讨论与relational algebra的对应关系和区别，而不是放在title里

### 6. 解决KQA Pro上0pp gain的叙事问题

这是目前论文最大的实验弱点。你在5.6.2的分析其实很诚实也很好，但reviewer的第一反应会是"你的方法在别人的benchmark上没用"。建议：

- 把这个结果**重新框架**为：我们的方法是为**top-K结构性失败的问题模式**设计的保险机制。KQA Pro的answer set恰好足够小，top-K不会结构性失败，所以不需要保险。这不是方法无效，是benchmark没有压力测试到需要这种方法的场景。
- 制作一个分析图：X轴是answer set cardinality，Y轴是Strategy vs Baseline的Δ。预期会看到一个明确的拐点——当answer set超过K时，Δ急剧上升。这个图会非常直观地说明"我们的方法在什么时候、为什么有效"。

### 7. 增加一个"Structural Failure Analysis"小节

在Section 5之后、Section 6之前，加一个专门的分析节，做定量的failure mode解剖：

- 对Baseline的所有错误案例做分类：有多少是因为answer set > K？有多少是因为缺少intersection语义？有多少是因为path不够长？
- 展示这些failure mode与你的算子**一一对应**——每种structural failure恰好被一个算子修复
- 这样论文的逻辑就变成了：**先诊断failure（分析贡献）→ 再设计对应的算子（系统贡献）→ routing只是连接二者的胶水**

---

## 总结一下核心优化方向

你的工作**真正的贡献**其实是：

1. 发现了top-K GraphRAG在若干计算模式下的**结构性不可能性**
2. 为每种模式设计了**信息需求匹配的retrieval算子**
3. 证明了**gain来自算子，不来自分类**（oracle实验）

但论文目前的叙事结构把重心放在了typology和dispatcher上，这反而让最强的贡献（算子设计 + structural failure分析）被弱化了。最高优先级的改动是**调整叙事主线**：把"我们做了一个更聪明的routing"变成"top-K在结构上是错的，我们设计了对的retrieval原语"。分类/routing只是一个O(1)的查表操作——它的存在是为了证明算子选择可以自动化，而不是说分类本身有技术深度。

你这个顾虑非常合理。让我先看看相关工作是怎么处理类似的"挑战主流方法"的定位问题的。你的直觉非常好。搜了一圈之后，我对整个landscape有了更清晰的认识。让我给你一个整体判断。

---

## 你的担心是对的：不应该说"top-K是错的"

但原因不仅仅是"太激进"——而是**不准确**。top-K对很多场景是完全正确的。你真正发现的问题更精确也更有价值。让我帮你找到一个既准确又有力的定位。

## 从相关工作看"分类+路由"这条路线的接受度

我搜到了几篇关键的对标论文，它们对你的定位有直接启示：

**Adaptive-RAG（NAACL 2024）** 做的也是"分类问题 → 选择检索策略"，把查询分成简单/中等/复杂三类，分别路由到无检索、单步检索、多步迭代检索。这篇发在了NAACL主会，说明"分类+路由"这个模式本身**审稿人是接受的**。但请注意，Adaptive-RAG变的是**检索深度**（0步/1步/N步），每一步用的是同一种检索原语（相似度搜索）。

**ByoKG-RAG（EMNLP 2025，Amazon）** 跟你的工作精神上最接近——它也用了多种检索策略（agentic traversal、path retrieval、OpenCypher graph query），并且明确提到 aggregation queries 是 agentic retrieval 的失败场景。它发在了EMNLP主会。**但是**，ByoKG-RAG是把所有策略的结果fusion在一起，没有你这种"问题类型→算子"的显式mapping。

**"Mitigating KG Quality Issues"（2025）** 开篇就说了一句值得你参考的话："distinct deficiency patterns require different solution strategies"——不同的缺陷模式需要不同的解决策略。

这些论文告诉你：你的大方向是被认可的，关键在于怎么frame。

---

## 核心建议：找到你真正的定位——"answer geometry mismatch"

你不需要说"top-K是错的"。你需要说的是：

> **Top-K retrieval对answer的几何形状（answer geometry）做了一个隐含假设——answer是一个可以被ranking capture的小规模集合。这个假设对factoid查询是成立的，但对若干在工业KG中常见的查询模式系统性地不成立。**

这个framing比"top-K是错的"好在三个地方：

**第一，它精确。** 你不是说top-K不好，你是说top-K隐含了一个假设（answer set ≪ K 且可rank），而你明确了什么时候这个假设不成立。

**第二，它有数据库理论的precedent。** 数据库领域早就区分了 **point query（点查询）** 和 **analytical query（分析查询）**，对应OLTP和OLAP的区分。没人会说"SELECT * WHERE id = 5 用B-tree索引是错的"——但如果你用同一个B-tree去做 `COUNT(*) GROUP BY country`，那确实是结构性不匹配。你可以直接借用这个framing：top-K is the B-tree of GraphRAG——适合point queries，不适合analytical queries。

**第三，它定义了你的scope而不是攻击别人。** 你不是在否定MS GraphRAG / HippoRAG / RoG——你是在说：这些工作都在优化point-query retrieval（让ranking更好、让path更准），但它们**没有覆盖analytical query patterns**。你的工作补上这个gap。

---

## 怎么描述"这类领域/场景"的特征

你需要一个术语来描述"雷达KG这类场景"。不应该只说"雷达领域"，而应该抽象为一组**可检验的结构特征**。我建议这样描述：

> **知识图谱具有以下结构特征时，top-K retrieval会系统性失效：**
>
> 1. **高基数关系（high-cardinality relations）**：某些关系的尾部集合很大（比如"美国运营的雷达"有400+个实体）。这意味着完整枚举或计数无法被任何固定K覆盖。
>
> 2. **多约束查询需求（multi-constraint query patterns）**：用户自然地提出涉及两个以上独立约束的交集查询（"S波段且中国研制"）。Top-K的ranking把这些约束混在一起，无法表达AND语义。
>
> 3. **否定/缺失查询需求（negation/absence patterns）**：用户需要确认某个事实不存在。KG只存储正面三元组，top-K窗口无法保证覆盖全局否定。

这三个特征**不是雷达独有的**。医疗KG（"列出所有治疗糖尿病的药物"——高基数）、材料KG（"找到硬度>X且密度<Y的合金"——多约束）、军事/工业KG都天然具有这些特征。这样你就从"雷达领域的工程trick"上升到了"一类KG上的结构性问题"。

---

## 你跟Adaptive-RAG和ByoKG-RAG的本质差异

这个对比非常重要，因为reviewer一定会问"跟Adaptive-RAG/ByoKG-RAG有什么区别"：

| 维度 | Adaptive-RAG | ByoKG-RAG | 你的工作 |
|---|---|---|---|
| 分什么 | 查询**复杂度**（简单/中等/复杂） | 不显式分类 | 查询的**计算模式**（factoid/enum/intersection/negation/...） |
| 变什么 | 检索**深度**（0/1/N步） | 检索**来源**（多工具结果融合） | 检索**语义**（每种算子执行不同的KG操作） |
| 隐含假设 | 所有检索步骤用同一种原语 | 最终需要LLM从混合结果中推理 | 每种计算模式有唯一匹配的检索原语 |
| 失败场景 | 当单步足够时，多步浪费 | 当多策略产生矛盾信号时 | 当某计算模式在分布中缺失时（你的KQA Pro结果） |

你的核心差异点是：**你变的不是检索的"多少"（depth），而是检索的"是什么"（semantics）**。Adaptive-RAG是"要不要多查几次"；你是"这道题根本就不该用ranking来查，应该做set intersection"。

---

## 建议的新叙事结构

把Introduction重组为四步论证：

**Step 1 — 问题观察（phenomenon）**：在工业KG（如雷达、医疗、材料）上部署GraphRAG时，约30%的用户查询准确率显著低于其他查询。

**Step 2 — 原因分析（diagnosis）**：这些低准确率查询共享一个结构特征——它们的answer geometry不符合top-K的隐含假设。具体分三种：answer set太大（enumeration）、需要交集语义（intersection）、需要全局否定（complement）。**这不是LLM能力问题，是信息量不足问题。**

**Step 3 — 方法论洞察（insight）**：解决方案不是更好的ranking，而是**为每种计算模式匹配正确的检索原语**。这类似于数据库领域中point-query和analytical-query使用不同的execution plan——B-tree适合点查，hash join适合交集，全表扫描适合聚合。

**Step 4 — 系统方案（system）**：我们设计了六种检索算子，覆盖六种计算模式，用一个轻量级LLM dispatcher根据问题的计算模式选择算子。Oracle实验证明：gain的99.6%来自算子设计，0.4%来自dispatcher质量。

这个叙事的好处是：**分类/routing被降级为一个implementation detail**，真正的贡献被明确为两件事——(1) 对top-K隐含假设的分析（这是analytical contribution），(2) 为每种失效模式设计的算子（这是system contribution）。

---

## 一句话总结

不要说"top-K是错的"，要说"**top-K对answer geometry做了一个隐含假设，这个假设在具有高基数关系和异构查询模式的工业KG上系统性不成立——我们的工作识别了这些不成立的模式，并为每种模式设计了信息需求匹配的检索算子**"。这比"top-K是错的"更准确，比"我们做了个分类器"更深刻，比"只在雷达领域work"更可推广。