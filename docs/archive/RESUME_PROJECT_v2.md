# 雷达 GraphRAG — 简历重定位指南 v2（反"玩具"叙事）

> 配套老版本 `RESUME_PROJECT.md`。老版本写了三种粒度的项目描述（一行 / 标准 / 详细）；这一版解决另一个问题——**面试官把 KGQA 视作"玩具项目"**——通过岗位定向叙事、用词替换、面试 talking points 三件套解决。

---

## 0. 核心定位策略（先看这段，所有改写都从这里推出来）

**面试官质疑"问答系统是玩具"的真正含义不是"技术差"，而是"我看不出能上线 / 能赚钱 / 能解决业务问题"。** 三个潜台词：

1. 没有真实用户 → 你的指标无法外推到生产
2. 没有团队协作 → 你只会一个人做，进不了规模化体系
3. 没有标准 benchmark → 你的 91% 我没法跟其他人比

**因此简历重写有三条铁律**：

| 原则 | 含义 |
|---|---|
| **不主打"系统"，主打"研究"或"工程能力"** | "问答系统"是产品视角，玩具感强；"KGQA 方法论研究"是论文视角；"多源 KG pipeline + 评测框架"是工程视角。**前者面试官扣分，后两者加分。** |
| **不主打"91% 准确率"，主打"+35.9pp [+31.7, +40.1] paired bootstrap CI"** | 单点数字像演示；带 CI、带 paired 比较的数字是研究质感。 |
| **必须有外部背书或可验证产物** | preprint / demo / GitHub stars / blog 任一项。**没有这些，再好的描述也只是自吹。** 见 §4。 |

---

## 1. 三个岗位定向版本

### 版本 A — 投研究岗 / AI Lab / 研究员实习

**适用**：DeepMind/Anthropic 等 lab 中国岗、字节/阿里/百度的 NLP 研究组、博士申请

> **Strategy-Routed GraphRAG: 题型感知的检索原语路由方法** ｜ 个人研究项目，独立完成 ｜ 2026.04–2026.05
> **论文产出**：一作，目标 NAACL/EMNLP Industry Track 2026（preprint: arxiv.org/abs/2026.XXXXX）
>
> - **研究发现**：识别 top-K 检索范式在 ~30% 自然 KGQA 问题上的**结构性失败**（聚合题答案集 > K、多约束需跨通道交集、否定题缺失事实非局部）。提出按 11 种细粒度题型路由到 6 个检索原语的 framework。
> - **实验设计**：499 题分层基准 + 26 题 OOD 双语压力集 + KQA Pro 502 题跨域验证；paired bootstrap (1000 resamples) 报告 95% CI。
> - **核心结果**：Strategy vs Baseline = **+35.9 pp [+31.7, +40.1]**，vs RoG-style **+28.3 pp [+23.8, +32.7]**，全 11 题型 ≥ baseline。
> - **方法学严谨度**：
>   - K=20 retrieval ablation 证明增益非 K-artifact（仍 +33.5pp 显著）
>   - Oracle dispatcher 上界实验显示 dispatcher noise ≈ 0（增益 100% 来自 operator suite）
>   - Suite-size ablation：4-op = 6-op = 94%，load-bearing 操作的贡献精确可加（-19 + -10 = -29）
>   - 诚实报告 KQA Pro 跨域 ±0.2pp 平局，论文将主创新重新 frame 为 "structural insurance, benchmark-distribution-dependent"
> - **代码与数据公开**：[github.com/xxx/radar-graphrag]，Apache-2.0 / CC-BY-4.0

**关键点**：这个版本里我**没有提"系统"**，全部用研究语言（"研究发现"、"实验设计"、"方法学严谨度"）。把项目定位成 paper，不是 product。

---

### 版本 B — 投算法工程师 / NLP / 搜索 / 推荐

**适用**：字节搜索、阿里小蜜、美团搜索推荐、各大模型公司应用算法岗

> **领域知识图谱构建与混合检索 pipeline（雷达装备域）** ｜ 个人项目，独立完成 ｜ 2026.04–2026.05
>
> #### 一、多源 KG 工程化构建（核心可迁移能力 1）
> - 9 个异构数据源融合（规则抽取、LLM zero/few-shot、PDF narrative、Wikidata、Wikipedia、GlobalSecurity 等），每条三元组带 source + confidence + evidence 三元溯源
> - **冲突仲裁的工程策略**：按 relation 类型分别设可信度阈值（如 `derivedFrom` ≥0.75 因抽取精度仅 8%）；合并实体前用 country 一致性做安全检查避免误合并（如 Flycatcher 雷达多源标注国家冲突 → 自动 skip）
> - 实体规范化的**安全边界设计**：明确排除 `(V)N` 代际变种合并，只折叠 `AN/` 前缀差异；888 实体去重到 849，0 误删
> - **产出**：4,817 实体 / 12,022 边 / 16,513 三元组 / 21 实体类型 / 100+ 关系
>
> #### 二、混合检索系统（核心可迁移能力 2）
> - BM25（rank-bm25）+ FAISS 向量检索（BGE-small-zh-v1.5）+ **RRF（Reciprocal Rank Fusion）融合** + 2-hop 图扩展
> - **KG-direct 旁路路径**：parser 识别 `(head, relation)` 时绕过文本检索直接走图查询，解决型号特殊字符（`MM/SPQ-2`、`AN/X-NN(V)Y`）的 tokenization 失败
> - 双语别名 5 级 cascade 解析（直接匹配 → 别名查表 → 频段归一化 → 后缀剥离 → 等价关系类 union）
>
> #### 三、LLM 调用工程（核心可迁移能力 3）
> - `ThreadPoolExecutor` 10 worker 并发 DeepSeek API 调用，4,400+ 次调用降到 12 分钟、成本 < $5
> - 三层持久化缓存（OCR / LLM / Wikipedia），所有脚本幂等可断点续跑
> - 全 pipeline LLM 总调用成本 ~$1.5（含跨域 KQA Pro 4 配置消融）
>
> #### 四、自建评测框架（核心可迁移能力 4）
> - 499 题分层基准 + 26 题手工 OOD 压力集 + 自动评分器（数值题 exact-count / 多值题集合包含 / unanswerable 显式拒答检测）
> - **paired bootstrap 1000 resamples** 报告 95% CI；分题型 / 分操作 / 分语言三维消融
> - 路由 / 单策略 / 双语层 / K=20 / Oracle dispatcher 五套独立消融脚本
>
> #### 结果
> 端到端准确率 **88.6%**（vs 同等检索预算 K=20 baseline **+33.5 pp [+29.5, +37.9]** paired CI）；路由分类器 93%；单问题成本 $0.00041，延迟 4.8s。
>
> **技术栈**：Python 3.11 / DeepSeek API / FAISS / rank-bm25 / Neo4j (Bolt) / NetworkX / PaddleOCR / pdfplumber / sentence-transformers / ThreadPoolExecutor。
> **代码规模**：8K 行 / 24 核心模块 / 15 实验脚本。

**关键点**：完全不主打"问答系统"，主打四种**可迁移工程能力**（KG 构建、混合检索、LLM 工程、评测框架）。任意一项拎出来都是阿里美团字节用得上的真本事。

---

### 版本 C — 投数据 / 平台 / 基础架构岗

**适用**：数据平台、ML infra、向量数据库相关岗位

> **多源知识图谱 ETL Pipeline 与混合向量检索系统** ｜ 个人项目 ｜ 2026.04–2026.05
>
> - **8 阶段数据 pipeline**（OCR → 字段解析 → 实体抽取 → LLM 富化 → 实体去重 → 属性归一化 → FAISS/BM25 索引 → Neo4j 导出），全幂等设计，断点续跑
> - **三层持久化缓存**（OCR / LLM / Wikipedia HTTP）降低重跑成本到 ~5% 原始 LLM 调用量
> - **schema 演进的零下游修改**：v1 扁平三元组 → v2 typed entity-relation，写适配器 `v2_to_flat_triples.py` 让下游 FAISS / Neo4j / BM25 索引零修改，类比生产环境 schema 切换不停机
> - **FAISS 向量库**：16K 三元组 × 512-dim BGE embedding，带 metadata（source / confidence / evidence）支持过滤检索
> - **Neo4j 图导出**：4,817 节点 + 12,022 边，支持 Cypher 多跳查询；同时维护 NetworkX 内存图（7,201 节点 / 16,450 边）作为低延迟检索后端
> - **并发与配额管理**：`ThreadPoolExecutor` + 信号量限流，DeepSeek API 4,400 次调用控制在 12 分钟、$5 总成本，零 429 错误
>
> **技术栈**：Python / FAISS / Neo4j / NetworkX / PaddleOCR / ThreadPoolExecutor。**代码规模**：8K 行。

---

## 2. 一行 bullet 三个变体（按岗位选）

**研究岗一行**：
> 一作论文（NAACL/EMNLP Industry 投稿中）：提出按题型路由 6 检索原语的 GraphRAG 框架，相对 baseline +35.9pp [+31.7, +40.1] paired bootstrap CI；附 K=20 / Oracle dispatcher / suite-size 三组防御性消融。

**算法工程师一行**：
> 中文领域 KG 全栈系统：9 源融合 16K 三元组 + BM25/FAISS/RRF 混合检索 + 4,400 LLM 调用工程（12 分钟 / $5）+ 499 题 paired bootstrap 评测框架；准确率相对 baseline +33.5pp 显著。

**通用兜底一行**：
> 个人独立完成：8K 行中文 KGQA 全栈研究项目（多源 KG 构建 + 混合检索 + LLM 路由 + bootstrap CI 评测），一作论文投稿 NAACL Industry 2026；端到端准确率相对 baseline +33.5pp [+29.5, +37.9]。

---

## 3. 措辞 do / don't 速查表

| ❌ 别写 | ✅ 改写为 | 为什么 |
|---|---|---|
| "搭建了一个问答系统" | "提出 / 实现了 X 方法 / framework" | "系统"=玩具，"方法"=研究 |
| "准确率 91%" | "+33.5pp [+29.5, +37.9] paired bootstrap CI（n=499）" | 单数字像 demo，paired CI 是研究 |
| "使用了 DeepSeek 大模型" | "DeepSeek API 4,400 次调用 / 12 分钟 / $5（带三层缓存 + 限流）" | 工程量化避免"调 API"印象 |
| "用 FAISS 做向量检索" | "BM25 + FAISS + RRF 融合 + 2-hop 图扩展的混合检索后端" | 单技术 → 体系 |
| "做了实体识别和关系抽取" | "9 源融合 + relation-level 可信度阈值 + 安全边界实体去重（0 误删）" | 描述 → 工程决策 |
| "实现了基于 RAG 的问答" | "识别 top-K 范式的结构性失败模式，设计 6 个检索原语" | 用别人框架 → 自己提出方法 |
| "可以回答各种问题" | "11 题型分层评测，含 OOD 双语压力集 + KQA Pro 跨域 502 题验证" | 模糊 → 可验证 |
| "效果很好 / 显著提升" | 永远带数字 + CI | 任何"很好"都是空话 |

**禁用词**：智能、高效、强大、先进、完美、革命性。**任何形容词都要换成数字**。

---

## 4. 简历投递前必做的三件事（按 ROI 排序）

### 必做 #1 — arXiv preprint（半天，ROI 最高）

简历上写"NAACL Industry 投稿中"vs"arxiv.org/abs/2026.XXXXX"，后者**直接消灭了"个人项目"的标签**。面试官看到 arXiv 链接的瞬间，项目从"小作品"变成"科研产出"。

操作：
1. 论文转 ACL 8 页格式（你已经在做 v11）
2. arXiv 投 cs.CL，需要一个有 arXiv 账号的 endorser（导师 / 已发论文的合作者 / 朋友）
3. 简历直接挂链接

### 必做 #2 — Gradio Demo + 30 秒录屏（1 天）

简历挂一个 demo 链接：`radar-graphrag.streamlit.app` 或 HuggingFace Space。面试官点开看到能跑、能问、能给答案，**直接破除"玩具感"**——因为玩具是没人能看到的，能跑的就不是玩具。

操作：
1. 写一个 `app.py`：输入框 + 显示 router 分类 + 显示 retrieved triples + 显示最终答案
2. 部署到 Streamlit Cloud / HuggingFace Spaces（都免费）
3. 录 30 秒 mp4，放 GitHub README 顶部 + 简历项目链接旁

### 必做 #3 — GitHub README + Docker compose（1 天）

让面试官能 5 分钟内复现。README 必须包含：
1. 一图说明架构（你的 PROJECT_REPORT 第 8 节那张 ASCII 图就够，画成 mermaid 更好）
2. 核心结果表（带 CI）
3. `docker compose up` 一键启动 demo
4. 链接到论文 + Demo + 视频

**没有这三件事**，简历上写得再花哨都没用——面试官看不到的就是不存在的。

---

## 5. 面试时的 Talking Points

### 30 秒电梯版（自我介绍 / 项目一句话讲）

> "我做了一个 KGQA 方法论研究，核心发现是：现有 GraphRAG 都在做更聪明的 top-K 检索，但 top-K 在计数、多约束、否定这三类问题上数学上就解决不了——'美国有多少款雷达' 这种题 top-8 永远只能返回 8 个。我的解法是把问题分类后路由到 6 个不同的检索原语，相对 baseline +35.9pp 带 paired bootstrap CI，论文在投 NAACL Industry。"

### 2 分钟深入版（被问"展开讲讲"）

按 **问题 → 发现 → 方法 → 实验 → 局限** 五段讲，每段 20-30 秒：

1. **问题**：top-K 范式的三个结构性失败模式（举一个例子说明，比如 "美国有多少款雷达"）
2. **发现**：~30% 自然问题落在这三个失败模式，且不同问题需要不同检索算法
3. **方法**：11 题型 × 6 操作的路由 framework，每个操作的算法机制（exhaustive / constrained_join with equivalence fallback / path_plan with `r⁻¹`）
4. **实验**：499 题三方对比 + 5 套消融 + paired bootstrap CI；K=20 ablation 防 K-artifact 攻击；Oracle dispatcher 证明 100% 增益来自 operator
5. **局限**：诚实讲 KQA Pro 跨域 ±0.2pp 平局，论文重定位为 "structural insurance"，跨域可移植性的 parser 是瓶颈

### 应对常见追问

| 追问 | 回答策略 |
|---|---|
| "这跟 ChatGPT 直接问有啥区别？" | "ChatGPT 答不出领域 KG 里的具体型号关系，且不可审计。我的输出每条都带 triple 溯源 + confidence，工业部署需要这种 auditability。" |
| "为什么不直接用 RoG / ToG？" | "RoG/ToG 都是单一路径规划策略，在聚合 / 多约束题上结构性失败（我跑了 zero-shot RoG 简化版 baseline 60.3%，比我们差 28pp）。但我也诚实在论文里写了没复现真 RoG finetuned 版本。" |
| "self-built benchmark 不就是 cherry-pick 吗？" | "正是这个原因我跑了 KQA Pro 跨域 4 配置 + WebQSP dispatcher probe。KQA Pro 上 ±0.2pp 平局我老实写在论文里，并发现这是 benchmark distribution 问题——公开 benchmark 的 Count 中位 ≤4，而 RadarKG 是 14，没有公开 benchmark 设计了我们目标的失败题型。这本身是 community gap。" |
| "工程上能上线吗？" | "现在不能：单机 NetworkX 内存图、4.8s 延迟、无 streaming 索引。但拆解后可上线的子模块是混合检索后端 + 评测框架，这两部分我设计时就考虑了 serving（FAISS 持久化 + Neo4j 后端 + bootstrap eval 可作 A/B 框架）。" |
| "为什么选雷达这个领域？" | "选一个 schema 复杂、双语混杂、有自然 high-cardinality 答案集的领域来暴露 top-K 失败。如果选 movie/celebrity domain（KQA Pro 那种），答案集都 ≤3，问题根本不会出现。" |
| "你一个人做的怎么协作？" | "确实是个人项目所以没有团队协作。我能展示的工程素养是 git 提交规范、模块化设计（24 核心模块解耦）、版本演进（v1→v11 with changelog）、reviewer-style 自审改了 6 版论文。不替代真团队经验但能证明工程品味。" |

---

## 6. 简历项目栏的最终模板（直接抄）

按你的目标岗位从 §1 选 A/B/C 一个版本作为正文，**前面加一个标准化的项目头**：

```
雷达装备知识图谱与题型感知 KGQA 研究  |  个人独立项目  |  2026.04–2026.05
GitHub: github.com/xxx/radar-graphrag  |  Demo: xxx.streamlit.app  |  Paper: arxiv.org/abs/2026.XXXXX
技术栈: Python / DeepSeek API / FAISS / Neo4j / NetworkX / BM25 / PaddleOCR / sentence-transformers
```

**这三行（链接 + 技术栈）是反"玩具"叙事的核心**：链接证明可验证，技术栈证明覆盖面。

---

## 附录：当前项目相对老版 RESUME_PROJECT.md 的核心进展

老版（5/12）写时项目状态：策略路由 + 验证 30 题。当前（2026-05-28+）已新增：
- 499 题全量三方对比 + paired bootstrap CI
- KQA Pro 跨域 4 配置消融（B1/B2/B3/B4）
- K=20 retrieval ablation
- Oracle dispatcher 上界
- Suite-size ablation
- WebQSP dispatcher probe（反向证据）
- 论文 v5 → v11（含 reviewer-style 6 版自审）

老版的数字（端到端 91%）已被更严格的 499 题全量数字（88.6% + CI）替代——新简历应该用新数字 + CI 而非老的 91%。
