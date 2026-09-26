# 内部审稿意见 — Strategy-Routed GraphRAG v4

**日期**: 2026-05-22
**视角**: 模拟 NAACL/EMNLP Industry Track 严格审稿人
**目的**: 提交前自查清单 + 修改优先级

---

## 优点（保留这些卖点）

- **+35.9pp 在 499Q 全集上比 100Q 还扩大** —— 不是 cherry-pick，可信度高
- **Cost & latency 表** —— 工业类论文该有的，但 90% 投稿都不写
- **6.3 bilingual-bench paradox 自我反思** —— 承认 in-distribution 评测有盲区
- **5.1 主动报告 Strategy 输的两个题型**（two_hop_bridge / unanswerable）—— 不藏数据

---

## ★ Major Issues（不解决会被拒/major revision）

### M1. 单 benchmark + 自建 + 自动生成 gold = 三重未受控

**问题**：KG 是自抽取/富化的、499Q 是脚本从 KG 反向生成的、Gold 答案与 pipeline parser 共享同一套别名层。

**例证**：attr_filter 中 RoG 0%——任何 NL2SPARQL 系统都不会 0%；最可能是 RoG 的输出格式与 gold scorer 不兼容，而非算法本身这么差。

**审稿人会要求**：
1. 至少跑一个**公开 benchmark**（WebQSP / CWQ / MetaQA / GrailQA）切片
2. 外部标注者重写 ≥50 题手工 gold
3. "Gold 生成器"和"pipeline parser"显式分离

---

### M2. RoG baseline 是稻草人

**问题**：原 RoG 微调 LLaMA-2 在 `<PATH>r1<SEP>r2</PATH>` 上输出 path。你用的是 DeepSeek **zero-shot** 出 path——single_hop 11.3% 像 prompt-following 失败而非算法失败。

**应对**：
- 跑真 RoG（作者公开 LLaMA-2 checkpoint）
- 或把 "RoG-style" 改名为 **"Zero-shot Path Planner (RoG-inspired)"**，明确不代表 RoG 原论文数字
- 加 ToG / HippoRAG / KGP 至少一个的实际对比

---

### M3. 缺 Oracle Dispatcher 上界实验

**问题**：现有 93% routing + 88.6% e2e。如果 dispatcher 完美 (oracle，用 gold qtype)，e2e 上界是多少？

没有这个数字，无法分离：
- "operator 本身的 ceiling" vs "dispatcher 噪声"
- 多少 +35.9pp 来自 operator、多少来自 LLM 猜对题型

**1 行代码的实验**（用 q["type"] 直接选 strategy），缺失会让方法论分数大降。

---

### M4. "Algebraic closure" framing 风险大于收益

**问题**：
- 没形式化定义什么叫 "closed"
- 没证明 6 是 minimal——可能 4 个就够（complement → lookup，dual_subgraph → constrained_join）
- "0pp drop 不是 redundancy 而是 closure" 是循环论证

**审稿人嗅觉**：over-mathematization。Codd algebra 的类比让对比刺眼——Codd 给了完备性证明，你没有。

**建议**：删 "operator algebra" 命名，改成 "Strategy Suite" / "Operator Library"。承认是工程选择不是数学结构。

---

### M5. 没有统计显著性 / 置信区间

`distractor` n=9、`unanswerable` n=20、`three_hop_chain` n=30——±10pp 完全可能是噪声。整个论文 0 个 CI / p-value。

**1 小时工作量**：bootstrap 1000 次报告 95% CI。

---

## ☆ Minor Issues（状态更新 2026-05-25）

| ID | 问题 | 工时 | 状态 |
|---|---|---|---|
| m1 | Bilingual stress set 26 题太小，加 CI | 1h | **完成 2026-05-25**：+11.5pp [+0.0, +26.9]，**CI 下界正好 = 0** — 临界显著；诚实写进 §5.3 |
| m2 | 11 题型 / 6 operator 数量没 ablation | 半天 | **完成 2026-05-25**：suite-size ablation 表 2b；**4-op 子集 = 6-op suite**；load-bearing 操作精确可加（-19 + -10 = -29） |
| m3 | Table 1 是 11×6 数字墙；改成 Δ 双列图 + OVERALL 行 | 1h | 未做（fig2 已经可视化，但表保留为详尽参考）|
| m4 | KG 抽取 precision 没量化；多少 +35.9pp 来自方法 vs KG 质量 | 半天 | 未做（v2 KG 已经是产物，质量评估需另写脚本）|
| m5 | Latency 单线程；可并行讨论 | 1h | 已在 §7.7 提到（"projects Strategy latency to ~3.0s"）|
| m6 | cost-per-pp 也算 vs RoG ($0.00064/pp) | 10 min | **完成 2026-05-25** |
| m7 | Artifact release plan | 1h | **完成 2026-05-25**：§7.1 加 Apache-2.0 / CC-BY-4.0 / 数据列表 / 复现成本 |
| m8 | 自我吹捧（"paradox" / "identity element"）| 1h | 部分完成（identity element 删了；paradox 保留作为方法论命名）|
| m9 | Related Work 漏 ChatKBQA / KAPING / KG-RAG / NL2SPARQL-LLM | 半天 | 部分完成（KAPING 已在 bib + Related Work §2.3 含 NL2SPARQL）|
| m10 | 缺 Future Work 章节 | 1h | **完成 2026-05-22**：8 条具体后续 |

---

## 预估初始评分（NAACL Industry 1-5）

- **Soundness**: 2.5/5（M1+M2+M3 三重问题）
- **Novelty**: 3/5（typology→routing 不算新颖，但实测扎实）
- **Empirical**: 3.5/5（量够、ablation 全、缺公开 benchmark）
- **Presentation**: 3.5/5（清楚但过度 framing）

**Overall**: Borderline Reject → Major Revision

---

## 优先级修改 Roadmap（一周内）

| 优先级 | 任务 | 工时 | 影响 | 状态 |
|---|---|---|---|---|
| 1 | Oracle dispatcher 实验（gold qtype 而非 LLM 预测）| 半天 | M3 | **完成 2026-05-22** — 反常发现 LLM 反而稍好（−0.4pp）|
| 2 | 公开 benchmark 切片（WebQSP ~200 题）跑 Strategy | 2 天 | M1+M2 | **dispatcher probe 完成 2026-05-25**：246 题英文路由 0 错误，但 88.2% 落入 single_hop → WebQSP 缺 multi-hop / count / filter 类，**反向证明它不是合适的对比基准**。Full e2e scoring 仍在 §7.2 future work |
| 2b | **KQA Pro 替代公开 benchmark（120K Wikidata）** | 3 天预估 | M1+M2 主创新跨域 | **probe（n=288）**：routing match 67.4%；dual_subgraph 100% / complement 84%。**E2E B1（n=139, mech args）**：−2.2pp。**E2E B2（n=139, LLM parser）**：+1.4pp。**E2E B3（n=502, 2-hop）**：Δ = +0.2 [−1.6, +2.2] 平局；SelectBetween +11.1 [+2.2, +22.2] 显著 ✓，Count −8.9 [−17.8, −2.2] 显著。**E2E B4（n=502, +concept expansion）**：Δ = −0.2 [−2.4, +2.0]，Count 修到 0pp 但 baseline 同步涨，总差距未拉开。结论：framework 跨域 OK；**+35.9pp 的真实来源是 RadarKG 的高答案基数题型，不是 baseline 弱**——这点用 §5.6.1 K=20 ablation 验证 |
| 2c | **K=20 baseline ablation（RadarKG）** | 完成 2026-05-27 | M1 主创新非 K-artifact | **K=8 52.7% / K=20 55.1% / Strategy 88.6%**。K=20 只涨 +2.4pp [+0.0, +4.6]；Strategy 仍 +33.5pp [+29.5, +37.9] 显著赢 K=20 baseline。agg_count 4→8% / three_hop 23→20% / attr_filter 12→18%——加 retrieval budget 救不了结构性失败题。**证明 +35.9pp 是真实算法差，不是 K=8 artifact**。这条加固论文 §5.6.1 |
| 2d | **Mintaka inspection + 公开 benchmark survey** | 完成 2026-05-28 | M1 公开 benchmark 适配性 | Mintaka Count 中位 4 / max 137——比 KQA Pro 还小；LC-QuAD 2.0 无 gold；GrailQA 需 Freebase ingest；HotpotQA 系是文本非 KG。**结论：没有公开 KGQA benchmark 有 RadarKG 的 high-cardinality 分布**。把 RadarKG-QA-499 重定位为 benchmark contribution（§4.4 新章节 + abstract + intro 重写）|
| 3 | Bootstrap CI 加到 Table 1 | 1h | M5 | **完成 2026-05-22**（主表）+ **2026-05-25**（bilingual stress）|
| 4 | 删 "algebra closure" framing，改 "Operator Library" | 半天 | M4 | **完成 2026-05-22**（保守路线：保留 algebra 作为 hook，去掉无证明的 closure 主张和循环论证）|
| 5 | 改 baseline 命名为 "Zero-shot Path Planner (RoG-inspired)" | 10 min | M2 | **完成 2026-05-22** |
| 6 | 加 Future Work 章节 | 1h | m10 | **完成 2026-05-22** |
| 7 | Suite-size ablation（M4 follow-up）| 半天 | m2 | **完成 2026-05-25** — 4-op minimal suite = 6-op suite；load-bearing 精确可加；lookup-only 57% 是地板 |
| 8 | Artifact release plan | 1h | m7 | **完成 2026-05-25** |
| 9 | Cost-per-pp vs RoG ($0.00064) | 10 min | m6 | **完成 2026-05-25** |

---

## WebQSP 公开 benchmark 转移工作量评估（M1+M2，scope 外）

实测可下载性：`rmanluo/RoG-webqsp` 在 HuggingFace 公开，文件结构：
- `data/test-00000-of-00002-*.parquet` (~91MB)
- `data/test-00001-of-00002-*.parquet` (~93MB)
- `data/validation-00000-of-00001-*.parquet` (~24MB)
- `data/train-00000/00001` (~154MB ×2)

转移代价（每项是 1–3 hour 工作）：

1. **下载 + 解析 parquet** → JSON（要装 pyarrow）
2. **Freebase 子图 ingest**：每题带局部子图（topic_entity + 2-hop 邻域），需要把它当成 mini-KG 喂给 Strategy 而不是当成全局 KG
3. **Relation 字典**：Freebase 有 ~200 高频关系；需要给每个关系起 `head_type` / `tail_type` / `question_templates` 才能让 parser 工作（我们 lexicon 现在 28 个 radar 关系，需要补 ~200 个）
4. **别名层**：Freebase entity 是 MID（如 `m.0d3k14`），需要 surface form lookup（Freebase dump 已废弃，需要从 WebQSP 自带的 entity_names 字段拿）
5. **题型映射**：WebQSP 没有原生的 11 题型标签，需要手工或 LLM 重新分类
6. **Gold scorer**：WebQSP 用 F1 over answer set，与我们的 exact-count / set-inclusion scorer 不一样

合理执行计划（~2 天）：拿 validation 集 200 题切片做一个 minimal pilot，每条用 Freebase 子图当本地 KG，跳过别名层，只测 dispatcher + 基础 operator。预期结果：单一 KGQA 论文里 WebQSP 单跳精度天花板 ~75%；如果 Strategy 在英文上也能维持相对 Baseline 的优势，就足够回应审稿人。

---

## v11 → v12 修订（2026-05-29）— Advice-driven framing 重构

**触发**：朋友 advice.md 指出 v11 论文的叙事让 reviewer 把工作误读为"只是分类器 + 手写策略"。oracle 实验本身已证明 dispatcher noise ≈ 0，但论文结构仍把 dispatcher / typology 当卖点摆在前面。我之前的 v11 自评也低估了 operator suite 的 analytical contribution（错把 6 个 operator 当"简单集合代数 + C 档工程"评价）。

### 核心 framing 切换

| 旧 framing (v11) | 新 framing (v12) |
|---|---|
| "Strategy-Routed GraphRAG: A Typed Operator Suite" | "Beyond Top-K: A Typed Operator Suite for **Answer-Geometry Mismatches**" |
| "top-K 是错的" | "top-K embeds an implicit assumption about answer geometry; the assumption fails on specific computable patterns" |
| dispatcher / typology 当核心贡献 | dispatcher 是 O(1) 查表，operator suite 是核心 analytical contribution |
| operator 选 6 是 "empirical" | 6 个 operator 是从 6 个 information-requirement class **derived**，不是经验拼凑 |
| 比 Adaptive-RAG / ByoKG-RAG 差异不清 | 显式：他们变 **depth/source**（保持 ranking 不变），我们变 **semantics**（每个 operator 是不同 KG 操作） |

### v12 七项具体改动

1. **Title + Abstract 重写**：去 "Strategy-Routed" 前缀，加 "answer-geometry mismatch" 概念，Oracle Δ −0.4pp 提前到 abstract，明示 "this is a paper about the operators, not the classifier"。
2. **Introduction 重构为 4 步**：phenomenon → diagnosis (AGM) → insight (OLAP/B-tree 类比) → system (oracle surprise + operators-driven)。Contributions 列表新增"AGM diagnosis"和"methodological finding (bilingual paradox)"两条。
3. **§2.3 新加 Information Requirement Profile 表**：6 个 operator 从"经验选择"提升为"unique solution to information requirement class"。注明每个 row 中 top-K 失败的具体原因。
4. **Related Work 加 Routing-based retrieval 段落**：显式对比 Adaptive-RAG (变 depth)、ByoKG-RAG (变 source)、我们 (变 semantics)，并给出具体例子说明他们的 +35.9pp 不可达。
5. **§4.2 重写为 "Structural Failure → Operator"**：per-operator ablation 表显式加 AGM-class 列，做 1:1 mapping。失败模式分解：load-bearing 三件套 -19/-10/-8 = -37pp 恰好覆盖 +35.9pp 主增益。
6. **新加 §4.5 RQ5: "Does answer-geometry exposure predict Δ?"**：用 `scripts/cardinality_delta_analysis.py` 产出 RadarKG + KQA Pro 跨基准 per-type Δ 表，按 AGM 类型分组（card / comp / cmpl / none）。证据：AGM-exposed 类型 median Δ = +58pp；non-exposed = 0pp。KQA Pro 跨域 ±0.2 平局 = AGM-exposure 分布弱，**不是方法失败**。
7. **§5 Discussion 新加 Composability 子节**：表格化展示 operator 组合（如 `exhaustive ∘ constrained_join`），强调 suite 在自然组合下闭合，未来 dispatcher 可扩展到 operator expressions。
8. **去 algebra 措辞**：§5 "no formal-algebraic-closure claim" 改成 "operators are derived from minimal information-requirement satisfaction; suite is composable but not formal algebra in Codd sense"。

### 关键数据/工件

- 新脚本 `scripts/cardinality_delta_analysis.py`，输出 `results/cardinality_vs_delta.tsv`
- references.bib 加 Adaptive-RAG (Jeong et al. 2024 NAACL) 引用；ByoKG-RAG 加 TODO-verify 占位

### 评估：v11 → v12 的自评分变化

| 维度 | v11 | v12 |
|---|---|---|
| Soundness | 3.5–4 | **4**（AGM diagnosis 提供理论 ground，operator derivation 不再 "empirical"） |
| Empirical | 4 | 4（实验数据相同；§4.5 是数据 re-presentation，不是新实验） |
| Novelty / Positioning | 3 | **3.5–4**（AGM 概念 + OLAP 类比 + Adaptive-RAG/ByoKG-RAG 差异化让审稿人难再说"only a classifier"） |
| Presentation | 4 | **4.5**（叙事自洽：dispatcher 自降级 → operator 才是 contribution，跟 oracle 实验对齐） |

**整体**：v11 weak accept / borderline strong accept → **v12 clear accept / borderline strong accept**（NAACL/EMNLP Industry Track）。Main Track 也变得可投（v11 是偏弱的）。

### 还需在投稿前做的事

- [ ] **核对 ByoKG-RAG 引用**：references.bib 当前是 TODO 占位，提交前要查实际 title/authors/venue
- [ ] **arXiv preprint**（最高 ROI，0.5 天）
- [ ] 拆分 short paper "Bilingual Paradox in Multilingual KGQA Eval"（可单独投 EMNLP Workshop on Eval）
- [x] ~~WebQSP 200-Q pilot~~ / **忠实微调 RoG**（2026-06-04 完成，见下）

---

## v12.1 → v12.2 增订（2026-06-04）— 忠实微调 RoG（消除最强攻击点）

**触发**：reviewer 最强攻击"你的 RoG baseline 是 zero-shot DeepSeek，被人为打弱了（single_hop 11%、attr_filter 0%）"。v12.1 只用"path-plan 消融 −10pp 上界"做空泛辩护，不够。

**做法**：服务器（华师大 4×RTX4090，**无外网**，校园网 Srun 认证未过，但实验不需要外网）。LLaMA-2 下不了，改用缓存的 **Qwen-7B-Chat**（HF 格式，mixrag conda 环境 transformers 4.49 + 复制来的 peft 0.7.1）。LoRA（r16/α32/c\_attn）微调一个 RoG-style planner，训练数据 240 条从 KG 结构采样（`mine_paths.py`，与 499 评测题字符串无重叠）。**只替换 RoG 的 planner**，walker + DeepSeek reasoner + scorer 与 zero-shot RoG 逐字节一致。加跑 Qwen-7B **zero-shot** planner 对照，隔离"微调效应"vs"换 base 效应"。

**5-way 结果（n=499）**：

| | Baseline | RoG-zs(DeepSeek) | RoG-zs(Qwen) | RoG-FT(Qwen) | Strategy |
|---|---:|---:|---:|---:|---:|
| OVERALL | 52.7 | 60.3 | 31.5 | **61.7** | 88.6 |

配对 bootstrap CI：微调效应 **+30.3pp [+25.9,+34.7]**；Strategy−RoG-FT **+26.9pp [+22.6,+31.3]**，均显著。

**对论文的价值**（强化，非削弱）：
1. 微调**大幅有效**（+30.3pp），追平强闭源 zero-shot planner → **彻底驳倒"弱 baseline"**。
2. 微调让 planner **逐题型重新发明 exhaustive 算子**：agg_count 2→64（追平 Strategy 62）、agg_enum 2→96、relation_inverse 6→80。
3. **但单路径范式无法表达集合代数算子**：attr_filter 仅 0→28（Strategy 92，+64pp 结构性缺口）；过拟合 ≤2-hop 训练分布（three_hop 36.7→13.3）。
4. 最强 RoG 变体仍落后 Strategy **+26.9pp**。"微调 planner 只能逐题型重新推导我们按设计提供的算子，无法在单路径范式内复制交集/补集。"

**confound 诚实披露**：RoG-FT 用 Qwen-7B（非 LLaMA-2，因服务器无外网）。但 Qwen-FT 已追平强 DeepSeek planner 且残差缺口是结构性的（交集/补集），故 LLaMA-2 不会改变结论。Future Work item 1 改为"LLaMA-2 planner replication"。

**论文落点**：§4.6 (RQ6) 新增 + Table 11 + Appendix I 全 5-way 表 + Limitations item 2 / Future Work item 1 更新 + §4 setup caveat 重写。论文 18→19 页。

**工程坑记录**：(a) Qwen1 modeling 代码需 transformers_stream_generator，4.57 下该包又坏（缺 BeamSearchScorer）→ 改用 transformers 4.49 的 mixrag 环境 + 复制 peft；(b) 首次训练 loss=0，因 system prompt 1365 token + max_len=1280 把训练目标截掉 → max_len 提到 1536 修复。

---

## v12 → v12.1 增订（2026-06-01）— 从 v5_revised 回移补全

**触发**：朋友审 v12 提出"v12 锐度好但 Limitations / Future Work / Appendix 缩水太狠；'principled derivation' 有过度宣称风险；RoG fairness caveat 只出现一次太隐蔽"。从 `strategy_routed_graphrag_v5_revised.md` 回移以下内容到 v12：

### 五处回移 / 增订

1. **§2.3 "Scope of the derivation claim" hedge**：在 derivation 主叙事后加一段：六个 information-requirement class 是 by inspection identified、不是 closed taxonomy；future patterns (aggregation arithmetic / transitive closure / temporal-range / fuzzy matching) 会 motivate 更多 operator。明示 "we do not claim formal completeness"。
2. **Limitations 5 → 8 条**：补 KG noise（具体到 1–3 unit 影响）、adversarial coverage、distribution-dependence of gain、sub-type sample sizes；保留 simplified-RoG / OOD size / suite size 原三条；用 enumerate 而非 inline 提高可读性。
3. **新加 §7 Future Work 整节 + §7.1 Artifact Release Plan**：5 条具体路线（faithful RoG / schema-derived parser / medical+materials cross-domain / adversarial subset / community high-card benchmarks）；Conclusion 顺延为 §8。
4. **Appendix 实质化**：从 v5 回移完整 dispatcher prompt（英译版）、parser prompt 说明、per-operator additivity 矩阵、cost decomposition 表（dispatcher \$0.00006 + parser \$0.00022 + answerer \$0.00014）、26-Q OOD substitution dictionary、KG construction pipeline 5 段（含 country coverage 93.4%/87.0% 数据来源 + entity dedup safety boundary）、KQA Pro B1–B4 per-config 数字、WebQSP probe 分布、RoG details。从原来 9 行标题占位扩到 9 章实质内容。
5. **RoG fairness caveat 强化到 3 处**：(a) §4 setup 改名 "Zero-shot Path Planner (RoG-inspired)" + 独立段落 explicit disclaim "not a faithful reproduction"；(b) §6 Related Work LLM-path-planning 段加 "We make no claim about the published RoG result"；(c) §6.5 Limitations 第 2 条独立项。三处互相 cross-reference。

### v12.1 对反 reviewer 攻击的强化

| Reviewer 可能攻击点 | v12 状态 | v12.1 加固 |
|---|---|---|
| "你说 operator 是 derived，过度宣称" | 风险 | §2.3 加 hedge：grounded but not formally complete |
| "RoG 比的是稻草人" | 单点提及 | 三处 explicit caveat + per-op ablation 给上限 |
| "Limitations 太短了" | 5 条 inline | 8 条 enumerated，覆盖 KG noise / adversarial / distribution |
| "Future Work 哪去了" | 完全省 | §7 整节 5 条 + §7.1 artifact release |
| "Appendix 是不是空的" | 9 行占位 | 9 章实质内容（prompts / cost / dict / KG pipeline） |
| "reproducibility 不够" | 暗示 | Appendix F KG construction 5 段 + Appendix D cost table |

### 自评分 v12 → v12.1

- v12 clear accept / borderline strong → **v12.1 clear accept（更稳）**
- 没动主结果数字（实验数据完全没变）；只动文字密度
- 主结构：md 397 行 → 488 行（+91）；tex 504 行 → ~620 行（+116）
- ACL 8 页正文不超（Appendix 不算页数）

**当前决定**：不在此 sprint 做。已在论文 §7 (Future Work) 第 2 条明确承诺，并在 §6.5 Limitations 第 2 条标注"single-domain self-built benchmark"作为已知问题。
