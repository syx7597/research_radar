# 面向雷达情报的可解释知识图谱问答与对抗决策支持方法研究

---

## 摘要

随着现代战场电磁环境日趋复杂，雷达情报的规模、异构性与时效性持续增长，如何对海量雷达装备知识进行有效组织、可靠问答与进一步的对抗决策辅助，已成为军事智能化的重要课题。近年来，大语言模型（Large Language Model, LLM）与知识图谱（Knowledge Graph, KG）相结合的检索增强生成（Retrieval-Augmented Generation, RAG）方法在开放域问答中取得显著进展，但直接迁移到雷达情报这一**专业性强、可解释性要求高、且最终服务于决策**的领域时，仍面临三方面困难：其一，通用 GraphRAG 对不同"问题形态"缺乏针对性，检索召回与问题所需的图结构不匹配；其二，生成式模型存在幻觉，难以满足情报分析对结论可溯源、可审计的刚性要求；其三，从"知道是什么"到"建议怎么办"的对抗决策，需要将稀疏且异构的电子战学理知识形式化，而现有方法多为面向特定干扰样式的强化学习黑箱，缺乏可解释性与知识可溯源性。

针对上述问题，本文以雷达情报领域为背景，构建了一条"**构建知识图谱 → 可解释问答 → 可审计分析 → 对抗决策支持**"的递进式技术路线，主要工作与创新点如下：

**（1）雷达领域可溯源知识图谱构建。** 设计了面向雷达装备的属性图模式，融合领域手册、百科与开放知识库进行多源抽取，为每条事实附加来源、置信度与证据元数据，形成规模达万级三元组、支持可溯源查询的领域知识图谱，为后续问答与决策提供统一知识底座。

**（2）策略路由的 GraphRAG 与"答案-几何失配"诊断。** 提出并形式化了"答案-几何失配"（Answer-Geometry Mismatch, AGM）现象，揭示通用 GraphRAG 在不同问题形态上失效的结构性根因；据此设计 O(1) 复杂度的问题分发器与六类**类型化检索算子**，使检索结构与问题所需几何相匹配。在自建的、显式暴露该失配的领域基准 RadarKG-QA-499 上，方法取得 88.6% 的总体准确率，较 Top-K 基线提升 35.9 个百分点、较 RoG 提升 28.3 个百分点；消融实验表明增益由算子设计驱动（分发器贡献约 0 个百分点）。

**（3）双层可审计分析智能体。** 将单次问答升级为"内层类型化算子组合执行器 + 外层 ReAct 编排"的双层智能体，并提出"大语言模型不进入信任路径"的可审计架构原则：关键事实与数字由确定性执行轨迹生成，模型仅承担受约束的自然语言复述。算子组合评测正确率 99%，情报报告逐条引证覆盖率与概述可溯源率均达 100%。

**（4）面向雷达对抗的可解释决策支持。** 将电子战对抗学理形式化为知识图谱上的推理：以反制关系刻画干扰与抗干扰的博弈，给出"可行/被抵消/应避免"三分输出的确定性推理算子并证明其可审计性；将战场级对抗方案归约为加权集合覆盖问题，给出具有 $H_n$ 近似保证的贪心算法；构建覆盖九个国家、三十余个防空体系、八十余部威胁雷达的可溯源威胁库与对抗本体，并以已解密战史作为独立检验。教科书级对抗案例测试通过率 100%，全部画像满足引擎不变量。

**（5）面向自适应雷达的知识引导在线对抗学习。** 针对认知电子战中威胁雷达**反应式切换抗干扰**这一序贯、部分可观测特性，将对抗手段选择形式化为隐藏抗干扰(ECCM)状态下的在线学习问题，提出以 doctrine 反制图为**结构先验**的贝叶斯学习算法——经由每轮干扰成败沿反制关系反推雷达隐藏的抗干扰，其样本复杂度与抗干扰数相关而**与候选手段数无关**；并以变点检测应对雷达变招、以主雷达图谱属性提供逐雷达先验，使知识图谱由"被查询"升级为"驱动学习"。在合成、**真实 doctrine 图**与**真实威胁雷达属性**上的仿真表明，该学习器稳定优于静态学理与强知识盲在线学习基线；本文并诚实界定其相对静态策略的密度-时长适用边界，以及"知识真实、对抗动态为仿真、效能非实测"的性质。

本文工作贯穿"**类型化算子**"与"**可解释、可溯源、不确定性显式建模**"两条主线，验证了在专业性强、知识稀疏、决策代价高的领域中，构造结论可审计、不确定性可传播、并具理论保证的智能分析与决策支持系统的可行性。

**关键词：** 知识图谱；检索增强生成；大语言模型；可解释问答；电子战；雷达对抗决策；集合覆盖

---

## Abstract

As the electromagnetic battlespace grows increasingly complex, the scale, heterogeneity, and timeliness of radar intelligence continue to rise, making the effective organization, reliable question answering, and downstream countermeasure decision support over massive radar knowledge a key topic in military intelligentization. Recently, Retrieval-Augmented Generation (RAG) methods that combine Large Language Models (LLMs) with Knowledge Graphs (KGs) have achieved notable progress in open-domain question answering. However, transferring them directly to radar intelligence—a domain that is highly specialized, demands strong explainability, and ultimately serves decision-making—still faces three difficulties: (i) generic GraphRAG is not tailored to different "question shapes," so retrieval recall mismatches the graph structure a question actually requires; (ii) generative models hallucinate, failing the rigid requirement of traceable and auditable conclusions in intelligence analysis; and (iii) moving from "knowing what" to "advising what to do" requires formalizing sparse, heterogeneous electronic-warfare doctrine, whereas existing methods are mostly reinforcement-learning black boxes targeting specific jamming patterns, lacking explainability and provenance.

To address these problems, this thesis develops, in the radar-intelligence domain, a progressive technical roadmap of "**building a knowledge graph → explainable question answering → auditable analysis → countermeasure decision support**." The main contributions are:

**(1) A provenance-aware radar-domain knowledge graph.** A property-graph schema for radar equipment is designed; multi-source extraction fuses domain handbooks, encyclopedias, and open knowledge bases, attaching source, confidence, and evidence metadata to each fact, yielding a domain KG of tens of thousands of triples that supports traceable queries.

**(2) Strategy-routed GraphRAG and the Answer-Geometry Mismatch (AGM) diagnosis.** We formalize the AGM phenomenon that explains the structural root cause of generic GraphRAG's failures across question shapes, and design an O(1) dispatcher with six typed retrieval operators that align retrieval structure with the geometry a question requires. On RadarKG-QA-499, a domain benchmark built to expose this mismatch, the method attains 88.6% overall accuracy—35.9 points above the Top-K baseline and 28.3 points above RoG—with ablations showing the gain is operator-driven (the dispatcher contributes about 0 points).

**(3) A two-layer auditable analysis agent.** Single-shot QA is upgraded to a two-layer agent—an inner typed operator-composition executor plus an outer ReAct orchestrator—under the architectural principle that "the LLM does not enter the trust path": critical facts and numbers are produced by a deterministic execution trace, while the model only paraphrases under constraints. Operator composition reaches 99% accuracy, with 100% citation coverage and 100% summary faithfulness in intelligence reports.

**(4) Explainable countermeasure decision support against radar.** EW doctrine is formalized as reasoning over a KG: a counter-relation models the jamming/anti-jamming game; a deterministic operator yields a three-way output (viable / neutralized / avoid) and is proven auditable; battlefield-level planning is reduced to weighted set cover with a greedy $H_n$-approximation algorithm. A provenance-aware threat library and doctrine ontology covering 80+ threat radars across 9 countries and 30+ air-defense systems are built and validated against declassified combat history. Textbook-level countermeasure cases pass at 100%, and all profiles satisfy the engine invariants.

This thesis is unified by two threads—**typed operators** and **explainable, traceable, uncertainty-aware reasoning**—demonstrating the feasibility of building auditable, theoretically grounded intelligent analysis and decision-support systems for specialized, knowledge-sparse, high-stakes domains.

**Keywords:** Knowledge Graph; Retrieval-Augmented Generation; Large Language Model; Explainable Question Answering; Electronic Warfare; Radar Countermeasure Decision; Set Cover
