# 符号表与缩略语

## 主要符号

| 符号 | 含义 | 首次出现 |
|---|---|---|
| $G=(V,E,A,\ell,\mathrm{prov})$ | 属性图：实体、边、属性、类型标注、可溯源元数据 | §3.2 |
| $(h,r,t)$ | 三元组：头实体、关系、尾实体 | §2.1 |
| $\mathrm{prov}(x)=(\mathrm{source},\mathrm{conf},\mathrm{evidence})$ | 声明 $x$ 的来源/置信度/证据 | §3.4 |
| $I(q)$ | 问题 $q$ 的最小信息需求 | §4.2 |
| $K$ | Top-K 检索的截断常数 | §4.1 |
| $\mathcal{G}$ | 计划树文法（可嵌套算子代数） | §5.3 |
| $\mathrm{eval}(P)$ | 计划树 $P$ 的确定性求值（值 + 轨迹） | §5.3 |
| $\delta(P)$ | 由执行轨迹生成的权威答案 | §5.4 |
| $\psi$ | 大模型对权威答案的受约束自然语言复述 | §5.4 |
| $\pi(r)$ | 威胁雷达 $r$ 的威胁画像（属性上的部分函数） | §6.2 |
| $\mathcal{A}$ | 威胁画像属性集（用途/跟踪体制/频段/频率/功率/捷变…） | §6.2 |
| $J,\ E$ | 干扰技术集合、抗干扰（ECCM）技术集合 | §6.2 |
| $\mathcal{D}\subseteq E\times J$ | 反制关系：$(e,j)\in\mathcal{D}$ 表 $e$ 克制 $j$ | §6.2 |
| $\mathcal{P},\ \rho$ | 学理规则集；单条规则 $(\mathrm{cond}_\rho,\mathrm{rec}_\rho,\mathrm{avoid}_\rho,c_\rho,\mathrm{src}_\rho)$ | §6.2 |
| $F(r)$ | 对威胁 $r$ 点火的规则集 | §6.5 |
| $\widehat{R}(r),\ A(r),\ N(r)$ | 推荐集、避免集、被对方 ECCM 中和的集合 | §6.5 |
| $\mathrm{Viab}/\mathrm{Cnt}/\mathrm{Avoid}(r)$ | 三分输出：可行 / 被抵消 / 应避免 | §6.5 |
| $c(j\mid r)$ | 干扰技术 $j$ 对威胁 $r$ 的置信度 | §6.5 |
| $\Gamma(r)$ | 按本机定量参数生成的交战研判 | §6.5 |
| $\mathrm{cover}(s,r)$ | 装备 $s$ 可交战威胁 $r$ 的谓词 | §6.6 |
| $w(r)$ | 威胁 $r$ 的威胁度权重 | §6.6 |
| $H_n=\sum_{k=1}^n 1/k$ | 第 $n$ 调和数（集合覆盖近似比 $\le\ln n+1$） | §6.6 |

## 缩略语

| 缩写 | 全称 | 中文 |
|---|---|---|
| KG | Knowledge Graph | 知识图谱 |
| LLM | Large Language Model | 大语言模型 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| GraphRAG | Graph Retrieval-Augmented Generation | 图检索增强生成 |
| KGQA | Knowledge Graph Question Answering | 知识图谱问答 |
| AGM | Answer-Geometry Mismatch | 答案-几何失配 |
| CoT | Chain-of-Thought | 思维链 |
| ReAct | Reasoning + Acting | 推理-行动（智能体范式） |
| RoG | Reasoning on Graphs | 图上推理 |
| ToG | Think-on-Graph | 图上思考 |
| BM25 | Best Matching 25 | 概率检索排序函数 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| LoRA | Low-Rank Adaptation | 低秩适配（微调） |
| EW | Electronic Warfare | 电子战 |
| ESM | Electronic Support Measures | 电子支援措施 |
| ECM | Electronic Countermeasures | 电子对抗（干扰） |
| ECCM / EP | Electronic Counter-Countermeasures / Electronic Protection | 抗干扰 / 电子防护 |
| SEAD | Suppression of Enemy Air Defenses | 压制敌防空 |
| EOB | Electronic Order of Battle | 电子战斗序 |
| IADS | Integrated Air Defense System | 一体化防空系统 |
| SAM | Surface-to-Air Missile | 地空导弹 |
| TWS | Track-While-Scan | 边扫描边跟踪 |
| RGPO / RGPI | Range Gate Pull-Off / Pull-In | 距离门拖引（拖出/拖入） |
| VGPO | Velocity Gate Pull-Off | 速度门拖引 |
| DRFM | Digital Radio Frequency Memory | 数字射频存储 |
| AESA / PESA | Active / Passive Electronically Scanned Array | 有源/无源相控阵 |
| ARM | Anti-Radiation Missile | 反辐射导弹 |
| PRF | Pulse Repetition Frequency | 脉冲重复频率 |
| ERP | Effective Radiated Power | 有效辐射功率 |
| LPI | Low Probability of Intercept | 低截获概率 |
| WEG | Worldwide Equipment Guide | （美陆军）世界装备指南 |
