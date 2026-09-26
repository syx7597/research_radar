> 历史资料：保留当时的方案与结果，不代表当前完成度。当前方向见 `/docs/RESEARCH_DIRECTION.md`，进度见 `/docs/CURRENT_STATUS.md`。

# 技术方案 v2:覆盖率归因驱动的属性感知图检索增强生成
## Attribute-aware, Feedback-Routed GraphRAG(AFR-RAG)v2

> **本文档取代 afr_rag_tech_plan_v1.md**(v1 保留存档,其 MINRELAX 降级为本版消融对照 A3)。
> 与 `thesis_roadmap.md`(置信度方向,未采纳)、`kg_agent_diagnosis_protocol_v1.md`(诊断协议,
> 可独立执行,其 T2/T3/T4 题层可为本方案供给领域评测集)相互独立,互不覆盖。
>
> **一句话主张**:在开放世界、属性部分覆盖(实测 54%)的领域知识图谱上,把属性过滤的
> 空集结果**归因**为「约束过严 / 覆盖缺失 / 落地错误」三种成因,并据此驱动
> **符号信号门控的异质三通道(属性/关系/文本)推理工作流**——而不是盲目松弛约束
> 或让 LLM 每步自由决策。

---

## 0. v2 相对 v1 的三处实质变更

| 变更 | v1 | v2 | 动因 |
|---|---|---|---|
| C2 重定义 | 最小约束松弛(MINRELAX,闭世界假设) | **空集溯因 + 覆盖率感知修复**(开放世界) | why-not/合作式查询是 1990s 老话题;且 54% 属性缺失下"空集≠约束过严",旧 C2 分不清两种空集 |
| C3 重定义 | 规则切换表 + 模仿学习小分类器 | **符号状态机工作流**:结构化状态 + 异质通道动作 + Verify 门控,LLM 仅用于 IR 编译与最终生成 | agentic 图推理是红海;差异化=「符号驱动、可审计、低 LLM 成本」对打「LLM 每步决策」 |
| 贡献收敛 | 三个松散创新点 C1/C2/C3 | **两个咬合贡献**:C1 承重墙 + CORE(C2+C3 合流) | 三个都不够硬 → 一墙一核 |

**贡献声明(v2)**
- **C1(承重墙,系统贡献)**:属性三重索引 + 数值/单位结构化 + 可执行约束编译。
- **CORE(算法核心)**:覆盖率归因驱动的符号化异质通道图推理工作流
  = 空集溯因(§4)⊕ 符号门控路由(§5)。溯因是工作流的一等动作,不再是子程序。
- novelty 保护序:**覆盖率归因 > 符号门控异质路由 > C1**。若只能保一个,保归因
  (why-not 文献无覆盖率概念,agentic 图推理文献无空集归因——组合空白待 G4 查实)。

---

## 1. 总体架构

```
                    ┌─────────────────────────────────────────────┐
                    │                离线索引层(新建, §3)           │
                    │  [只读复用] 实体嵌入 / 关系嵌入 / 叙述块嵌入    │
                    │  ┌───────────────────────────────────────┐  │
                    │  │ 属性三重索引 (C1)                        │  │
                    │  │  ①结构化属性长表  ②属性言语化嵌入          │  │
                    │  │  ③单位归一数值列(可范围查询)             │  │
                    │  │  + 属性覆盖率统计 cover(key, type) ←CORE 地基 │  │
                    │  └───────────────────────────────────────┘  │
                    └───────────────────┬─────────────────────────┘
                                        │
 用户问题 ──► IR 编译 (§2, 唯一入口 LLM 调用) ──► 工作流引擎 (§5)
                                        │  初始计划 = f(IR 形态) 确定性推出
                                        ▼
        ┌────────────────────────────────────────────────────────┐
        │  符号状态机工作流引擎 (C3-CORE, §5)                        │
        │  状态 s = (IR, 约束满足位图, 各通道已探信号, 预算, 候选集分桶) │
        │  动作(异质三通道,非"算子"):                              │
        │   属性通道: AttrLookup / AttrFilter / Abduce(§4)          │
        │   关系通道: RelHop / PathExpand                          │
        │   文本通道: TextSearch                                   │
        │   门控/终止: Verify / Answer / Abstain                    │
        └───────────────┬────────────────────────────────────────┘
                        │ 每步返回: 结果 + 符号信号
                        │ (EMPTY / AMBIG / KEY_MISSING / HOP_BROKEN / LOW_SCORE / TYPE_VIOLATION)
                        ▼
              路由器读符号信号 → 续走 / 切通道 / 触发 Abduce / 终止
                        │
                        ▼
        融合作答 (§6): RRF 多通道融合 + 强制溯源 + (若经 Abduce) 归因报告
                        │  ← 出口第二次、也是最后一次 LLM 调用(仅生成)
```

**关键立场(写进论文的观点)**:LLM 只出现在**两端**(IR 编译、最终生成),中间的路由与
修复**全部由符号信号驱动**。这是对主流"LLM 每步自由决策的 agentic 图推理(ToG/Graph-CoT)"
的有意对立面——可复现、低成本、可审计,契合"可溯源决策支持"的应用背景。

---

## 2. 查询中间表示(IR)——工作流不退化的关键,且是 G1 硬门对象

自然语言问题先编译为类型化 IR(**一次** LLM 调用 + schema 接地校验):

```
QueryIR {
  target:      (type: EntityType|ValueType, card: one|set|count)
  anchors:     [(surface, canonical_id | UNGROUNDED)]
  rel_chain:   [(relation, direction)]          # 关系约束链, 可空
  attr_cons:   [(key_norm, op, value, unit)]    # op ∈ {=, >, <, between, ≈}
  aggregation: none | count | argmax(key) | compare(key, e1, e2)
}
```

- `anchors` 落地:[只读复用] `kg_v3` 别名记录 + 图内唯一性校验;
- `attr_cons.key_norm` 经属性键归一表(§3.1)映射;`value` 经单位归一;
- IR 编译失败或字段冲突(target 类型与 rel_chain 签名不符)→ 打回重编译一次,
  再失败走纯文本通道兜底。

**IR 三重作用**:(a) 初始路由计划由 IR 形态**确定性**推出(有 attr_cons 无 rel_chain →
属性通道先行;皆有 → 先关系收窄再属性过滤),可审计可复现;(b) 中途切换时 IR 提供
"还剩哪些约束未满足"的精确记账,切换是**续算**不是重来;(c) IR 是"可分解可审计"claim 的实体。

> **G1 硬门**:IR 编译是全案单点故障。开工前必须在 ≥100 题上测 **IR 编译准确率**
> (target/anchors/rel_chain/attr_cons 逐字段人工核),低于阈值则先修 IR,不进 §3 之后。

---

## 3. C1:属性感知检索(承重墙)

### 3.1 属性三重索引

> **数据实况(已核 `kg_v3/entities.json`)**:Radar 实体的属性是节点上的 `attributes`
> **列表**(`{attr, value_raw, unit?, tier, unmapped}`),**不是边**。3057/5674 实体有属性(54%)。
> 属性分两类,**本模块只吃第一类**:
> - **数值规格键(结构化对象)**:range 1747(1601 含数值)/ frequency 1887 / peak_power 1104 /
>   pulse_width 885 / prf 854 / beamwidth 856 / avg_power 438 / weight 227 / mtbf 226 …(约 10–15 个核心键)
> - **自由文本描述键(unmapped)**:function_description 911 / description 880 /
>   techtype_description 495 …→ **不进属性索引,归 TextSearch 通道**(§5)。

对每个实体的**数值规格键**建三重索引:

- **① 结构化属性长表** `(entity_id, key_norm, value_lo, value_hi, unit_norm, value_raw, tier, source)`
  - key 归一:构建属性键别名表(探测距离/range/detection range → `detection_range`),
    嵌入聚类 + 人工审定(核心键 <200,一天可完成);
  - 数值解析:区间"50–80 km"→(50,80);约值"约 300"→(300,±δ);上界"up to X"→(-∞,X];
  - 单位归一:换算 SI 存储,保留 value_raw 供溯源。
- **② 属性言语化嵌入**:每条属性生成模板句("<实体>的<键中文名>为<值><单位>")入向量索引,
  服务模糊问法("X 能看多远")到键的软对齐,与①硬过滤互补。
- **③ 数值列索引**:对 (value_lo, value_hi) 建可范围查询列索引(DuckDB/SQLite),
  支撑 AttrFilter 毫秒级执行。

### 3.2 属性覆盖率统计(CORE 的地基,§4 依赖)

索引期同时计算 **cover(key, type) = 该 type 下拥有 key 的实体数 / 该 type 实体总数**,
落表 `attr_coverage.parquet`。这是 §4 空集溯因区分"约束过严"与"覆盖缺失"的**唯一判据**,
必须离线预算好。

### 3.3 可执行约束编译

IR.attr_cons + IR.target.type → 过滤程序 σ = ⋀ᵢ (keyᵢ opᵢ vᵢ) ∧ (type ≤ T),
type 按实体类型层级下推包含(NavalVessel ≤ Platform)。σ 在③上执行返回 id 集合。

---

## 4. CORE 之一:覆盖率感知的空集溯因(取代 v1 MINRELAX)

### 4.1 问题重定义(与 why-not 文献的 delta 就在这里)

σ(G) = ∅ 时,**不假设"约束过严"**。在开放世界 + 部分覆盖下,空集有三种本质不同成因,
用 cover(·) 与 IR 落地状态**归因**:

| 成因 | 判据(可机判) | 修复动作 | 语义 |
|---|---|---|---|
| **OVER_CONSTRAINED** | 所有约束键 cover(keyᵢ,T) ≥ θ_hi(默认 .5) 但无实体同时满足 | 代价松弛 widen/lift(即旧 MINRELAX,仅此分支触发) | 真·闭世界式过严 |
| **COVERAGE_GAP** | ∃ 约束键 cover(keyᵢ,T) < θ_lo(默认 .3) | **切文本通道**:对满足其余约束的候选逐个取证补该属性 → 补全后重算 σ | 空集来自数据缺失,非不可满足 |
| **GROUNDING_ERROR** | anchor=UNGROUNDED 或 key 归一多义 | 回 §2 重编译,**不动数据** | 空集是编译假象 |

### 4.2 覆盖率加权代价(把静态代价表变成数据驱动)

松弛动作代价与覆盖率挂钩:`cost(widen(cᵢ)) = base_cost / max(cover(keyᵢ,T), ε)`。
低覆盖键"松弛"边际收益低(放宽了也没数据),best-first 搜索**自然优先"转文本补全"而非"继续放宽"**。
Dijkstra 最优性(代价非负)仍成立,但求解语义从"闭世界最小松弛"变为"开放世界最小代价修复"。

### 4.3 算法

```
function ABDUCE(σ, T, G, β):
    cov ← [cover(keyᵢ, T) for cᵢ in σ]
    if any(anchor UNGROUNDED in σ.related_IR): return (GROUNDING_ERROR, ∅, report)
    if min(cov) < θ_lo:                                  # COVERAGE_GAP 优先
        cand ← EXEC(σ \ low_cov_constraints, G)          # 放开低覆盖约束得候选
        aug  ← TEXT_FETCH_VERIFY(cand, low_cov_keys)     # 文本通道逐个取证补属性
        R    ← EXEC(σ, aug)                              # 补全后重算
        return (COVERAGE_GAP, R, report(补了哪些实体的哪些键))
    else:                                                # OVER_CONSTRAINED
        # 覆盖率加权 best-first,动作 widen/lift/drop
        PQ ← {(0, σ)}
        while PQ:
            (c, σ') ← pop-min(PQ)
            if c > β: return (NO_SOLUTION_TRUE, ∅, report)   # 覆盖充足下的真无解 → 高可信拒答
            R ← EXEC(σ', G)
            if |R| ≥ 1: return (OVER_CONSTRAINED, R, report(σ→σ'))
            for a in ACTIONS(σ'): push(PQ, (c+cost(a,cov), a(σ')))
    return (NO_SOLUTION_TRUE, ∅, report)
```

### 4.4 可写进论文的性质与价值

- **最优性**:代价非负 + best-first ⇒ OVER_CONSTRAINED 分支返回代价最小修复(Dijkstra 引理);
- **单调性**:widen/lift/drop 均使 σ'(G) ⊇ σ(G),搜索无需回溯校验;
- **拒答语义厘清(解决 v1 遗留的 NO_SOLUTION 混淆)**:只在 **覆盖充足**(非 COVERAGE_GAP)
  且松弛耗尽时判 `NO_SOLUTION_TRUE`(高可信拒答);COVERAGE_GAP 分支永不冒充"真无解",
  未补全成功则输出低置信 + 明示"因数据缺失无法确定"。**这一条正是 why-not / agentic 图推理
  两支文献都没有的点**(它们都假设数据完整)。
- **归因报告** REPORT 进入生成措辞与 Abstain 判据,服务可解释与幻觉抑制。

---

## 5. CORE 之二:符号信号门控的异质通道图推理工作流(取代 v1 规则表+小分类器)

### 5.1 状态与动作(刻意区别于 ToG 的自由文本 scratchpad)

- **状态 s(结构化、可机读、可复现)** = (QueryIR, 约束满足位图, 各通道已探信号集, 剩余预算, 候选集大小分桶)。
  **不是** LLM 自写的思考文本 —— 这是"可审计"claim 的实体。
- **动作空间 = 异质三通道**(见 §1 图),每个动作带前置条件与后置**符号信号**:

| 动作 | 通道 | 成功 | 失败信号(路由输入) |
|---|---|---|---|
| Ground | — | 唯一命中 | EMPTY / AMBIG |
| AttrLookup | 属性 | 值存在 | KEY_MISSING |
| AttrFilter | 属性 | 非空集 | FILTER_EMPTY →触发 Abduce(§4) |
| Abduce | 属性 | 归因+修复集 | NO_SOLUTION_TRUE / COVERAGE_GAP_UNRESOLVED |
| RelHop | 关系 | 非空 | HOP_BROKEN |
| PathExpand | 关系 | 达目标类型 | BEAM_EMPTY |
| TextSearch | 文本 | 含目标实体/键 | LOW_SCORE |
| Verify | 门控 | 通过 | TYPE_VIOLATION / UNIT_VIOLATION |
| Answer / Abstain | 终止 | — | — |

### 5.2 路由(v0 符号规则版——本方案主打,不是过渡)

- 初始计划:IR 形态 → 通道序,确定性映射(附录写死);
- 切换规则(符号信号 → 动作):

| 信号 | 动作 |
|---|---|
| KEY_MISSING | TextSearch(verbalize(entity,key)) → Verify → 命中则以文本证据作答 |
| FILTER_EMPTY | Abduce(§4);按归因走松弛 / 文本补全 / 回 IR |
| HOP_BROKEN | 换 schema 近义关系重试 → 仍断则 TextSearch 定向取证 |
| AMBIG | 用 IR 其余约束消歧 → 仍多义则并行展开或澄清 |
| LOW_SCORE | 若 IR 含未用 rel_chain → 回关系通道 |
| TYPE/UNIT_VIOLATION | 回退上一步,该分支剪除 |

- **Verify 门控**:[只读复用] `pipeline/v3/s8_quality.py` 的类型签名检测器 + 单位相容检查;
  agent 每步先过 Verify,不合法立即回退。这是 ToG 类没有的"符号护栏"。
- 预算:总动作数 ≤ B(默认 10),每类动作单独上限,防循环。

### 5.3 LLM 用量纪律(卖点所在)

LLM **仅**在 §2 IR 编译(1 次)与 §6 最终生成(1 次)出现,**中间零 LLM 决策**。
对照实验(§7 的 A-workflow)将证明:符号驱动路由用 ~1/3 的 LLM 调用打平/打赢
"LLM 每步自由决策"——**成本×可靠性的 Pareto 前移就是 CORE 的核心量化证据**。

---

## 6. 与既有资产对接 + 命名防混淆(硬纪律)

> **本方案严禁改动既有产物。** 全部新代码落 **顶层新目录 `afr_rag/`**(下述),
> 对 `kg_v3/`、`agent/`、`pipeline/v3/`、`graphrag_retriever.py` **只读引用,不写回**。

### 6.1 命名去歧义(防止与项目原有概念混淆)

| 原有概念(勿混) | 位置 | 本方案对应物 | 区别 |
|---|---|---|---|
| **6 个"算子"** (constraint/intersect/…/aggregate) | `agent/composition.py` | 本方案叫**"通道动作(actions)"** | 不复用不继承;动作是异质通道 IO,非集合代数;**文档/代码一律不用"算子/operator"指代本方案动作** |
| **strategy-routed dispatcher**(11 型→6 策略) | `agent/`, thesis §4 | 本方案的**符号工作流**(§5) | 旧=入口一次性分类;新=符号信号驱动的多轮状态机;二者是对照,不是同一物 |
| **v3 混合检索**(BM25+FAISS+RRF+图扩展+rerank) | `graphrag_retriever.py` | 只读复用为**关系/文本通道底座** | 本方案在其上加属性通道 + 工作流,不改其代码 |
| **kg_v3 数据** | `kg_v3/` | 只读数据源 | 导出为 `afr_rag/data/` 副本,不动原文件 |

### 6.2 只读复用清单

| 既有资产 | 角色 | 复用方式 |
|---|---|---|
| `kg_v3/entities.json` 的 `attributes` 列表 | C1 属性索引数据源 | 只读导出 |
| `kg_v3` 别名记录 | Ground + 属性键别名种子 | 只读 |
| `pipeline/v3/s8_quality.py` 类型签名/单位检测 | Verify 动作 | import 调用,零新开发 |
| 1,432 段叙述文本 | TextSearch 通道语料 | 只读 |
| 实体类型层级 | lift 松弛 + type 下推 | 只读 |
| 诊断协议 T2/T3/T4 题层 | 领域评测集 | 若已构造则复用 |

### 6.3 新建目录结构

```
afr_rag/                      # 顶层新包,自包含
├── index/                    # C1: 属性三重索引 + 覆盖率统计
│   ├── build_attr_index.py   #   读 kg_v3 → 长表/嵌入/数值列/attr_coverage
├── ir/                       # §2: IR 编译 + schema 校验
├── channels/                 # §5 动作实现: attr / rel / text 三通道
├── abduce/                   # §4: 覆盖率感知空集溯因(CORE-1)
├── workflow/                 # §5: 符号状态机引擎 + 路由(CORE-2)
├── verify.py                 # 薄封装 → 调 pipeline/v3/s8_quality
├── fuse.py                   # §6 RRF 融合 + 溯源 + 归因报告
├── configs/                  # 所有阈值(θ_hi/θ_lo/β/B/代价)yaml,禁硬编码
├── baselines/                # §7 基线封装
├── eval/                     # 指标 + 显著性
└── experiments/              # 每次实验: config 快照 + results.json + predictions.jsonl
```

---

## 7. 实验设计

### 7.1 数据集

| 数据集 | 角色 | 说明 |
|---|---|---|
| STaRK(amazon/mag/prime) | 主力 | 结构+文本+属性混合;Hit@1 / Recall@20 / MRR。**G2 先验证数值约束题密度** |
| KQA Pro / GrailQA 数值·比较·计数子集 | 攻坚 | 数值约束题(A2 的公开集证据);EM(数值容差)/ 集合 P/R/F1 |
| WebQSP / MetaQA | sanity | 关系多跳为主,只证不掉点 |
| 雷达 KG → **自建 QA 集**(见 7.1.1) | 领域 | EM + 集合 F1 + LLM-judge win-rate;数值属性最密,是 A2 与空集溯因的主证场 |

### 7.1.1 领域 QA 评测集构造(自建 —— KG 本身不是评测集)

> **关键澄清**:雷达 KG 是**知识源/检索库**(被查询的对象),**不是评测集**。领域评测必须
> 构造带金标的 QA 对 `{question, gold_answer, gold_path, gold_evidence, expected_channel, type}`。
> 公开集(STaRK 等)**自带 QA,无需构造**;领域集**必须自建,现状:未构造** ← 待补的活。

- **复用现成蓝图**:直接执行 `kg_agent_diagnosis_protocol_v1.md` 的分层构造规程(T1–T7)。
  **一套题两用**:先当诊断探针(坐实失血点),再当本方案领域评测集。题层 ↔ 本方案组件:
  T3→属性通道/C1;**T3c + T4→空集溯因 CORE-1 + 完备枚举**;T2→关系通道;T6→文本通道。
- **构造前必并入的 4 处校准(依图谱实况,否则金标有坑)**:
  1. **T3/T3c/T4 金标限定"具有该属性/关系的覆盖子集"**,并记 `covered/total`——属性仅覆盖 54%,
     否则漏答被系统性误判为系统失血;
  2. **T1 别名层砍西里尔子层**(实测仅 1 条,出不了 8 题),保留北约代号 + 系列指针,
     先过滤 OCR 垃圾与 (V) 后缀;
  3. **T2 ≥4 跳先枚举确认实例数**,不足则并入 3 跳;
  4. **构造时即记录 `expected_channel` 与 `gold_path`/`gold_evidence`**——通道路由与 CORE
     对齐全靠这两字段,事后补不了。
- **为本方案新增两字段**:`attr_coverage_at_gold`(金标涉及属性的覆盖率,喂 §4 归因评测)、
  `gold_channel`(= expected_channel,喂 CORE-2 路由评测)。
- **产出**:`afr_rag/data/probe_set_v1.jsonl`,冻结 + hash 存档;dev/test = 7:3,test 封盘。

### 7.2 基线(四档,后两档是 CORE 的直接对手)

1. **GraphRAG 主线**:LightRAG / HippoRAG 2 / MS GraphRAG / Practical-GraphRAG(RRF 底座);
2. **属性平铺文本 RAG**(C1 的死敌):属性拼成句子当普通块检索——输给它则 C1 不成立,**必须实现得强**;
3. **裸 ReAct 多工具 agent**:同样三工具,无 IR、无符号路由、无 Abduce(判死刑基线);
4. **LLM 每步决策路由**(A-workflow 对手):同样动作空间,但每步由 LLM 自由选下一动作
   ——直接回答"你这符号工作流是不是不如让 LLM 自己走"。

### 7.3 消融(每行绑一个 claim)

| 消融 | 去掉 | 证明 |
|---|---|---|
| A1 | 属性通道整体 | C1 存在价值 |
| A2 | 数值/单位结构化(退为字符串匹配) | C1 结构化必要性 |
| A3 | 覆盖率归因(空集即失败,= v1 盲松弛) | **CORE-1 的增益**(核心) |
| A-workflow | 符号路由(退为 LLM 每步决策) | **CORE-2:符号驱动的成本 Pareto**(核心) |
| A5 | Verify 门控 | 符号护栏对幻觉/类型违规的作用 |
| A6 | RRF 融合(单通道) | 三通道互补性 |

### 7.4 指标

- 端到端:EM(数值 ±5%)/ F1;集合题 P/R/F1(**漏答与编造分列**);
- 检索层:**各通道 recall@k**(属性证据被检回率 = C1 直接证据);
- **成本:LLM 调用数 / token / 延迟 → 准确率-成本 Pareto 图(CORE-2 主图)**;
- 拒答:应拒题拒答准确率 + `NO_SOLUTION_TRUE` 判定精度(CORE-1 附加价值);
- 归因报告人工小样本评价:报告是否如实反映空集成因与修复。

### 7.5 统计

- 每配置 3 seed(或 temp=0 两遍一致性);主对比报均值±std;
- 与最强基线做题级配对 bootstrap 显著性;公开集用原始切分,领域集沿用诊断冻结纪律。

---

## 8. 前置硬门(G1–G4,全过才允许进 §9 全量开发)

> 纪律:这四道门任一不过,**改方案而非硬闯**。均为低成本、无需 GPU、开工前可完成。

- **G1 IR 编译准确率**:≥100 题人工核字段级准确率。不达阈值先修 IR(§2)。
- **G2 STaRK 数值题密度**:确认主力公开集含足量数值约束题;不足则 A2 主证场移至
  KQA Pro/GrailQA 数值子集 + 领域集,并在论文如实声明"数值通用性证据主要来自 X"。
- **G3 空集成因二分布(CORE-1 地基)**:在雷达 KG 上统计"属性过滤空集"案例中
  COVERAGE_GAP(涉及键 cover<θ_lo)与 OVER_CONSTRAINED 各占比。
  **两类都有可观占比 → 覆盖率归因有真实分布支撑,CORE-1 成立;若几乎全一类 → 归因无意义,CORE-1 降级**。
- **G4 组合 delta 查新(CORE 保命)**:精读 ToG / RoG / PoG / Graph-CoT / GraphAgent / Readi
  + cooperative query answering / why-not 综述,确认"符号信号驱动 + 异质通道 + 覆盖率空集归因"
  三词组合未被占。判断倾向为空,但**必须查实**;若被占,按 §10 退路收缩。

---

## 9. 里程碑(约 11 周,C1 → CORE-1 → CORE-2 依赖顺序)

| 周 | 任务 | 出口标准 |
|---|---|---|
| 0 | **前置硬门 G1–G4** | 四门全过并存档;否则改方案 |
| 1–2 | 底座复现(LightRAG/Practical-GraphRAG 于 STaRK 一子集) | 复现指标 ≈ 原报告 ±2 点 |
| 3–4 | C1:属性三重索引 + 覆盖率统计 + IR 编译 + AttrFilter | 领域 T3 题 recall@k 相对底座正向 ← **go/no-go** |
| 5–6 | CORE-1:ABDUCE 实现 + 三分支 + 归因报告 + 性质引理成文 | 空集案例专项测试通过;A3 可跑 |
| 7 | CORE-2:符号工作流引擎 + 路由 + Verify 门控;裸 ReAct & LLM每步 基线搭好 | 工作流跑通;A-workflow 可跑 |
| 8–9 | 全量:STaRK + KQA Pro/GrailQA 子集 + sanity 集 | 主表成型 |
| 10 | 领域集 + 消融补全(A1/A2/A3/A-workflow/A5/A6) | 全消融数据齐 |
| 11 | 成本 Pareto + 显著性 + 图表锁定 | 全部图表冻结 |

---

## 10. 风险与兜底退路(预写死,防中途慌乱)

| 风险 | 对策 |
|---|---|
| G3 空集几乎全一类 | CORE-1 归因降级为普通松弛,承重转 C1 + CORE-2;标题去掉"归因" |
| A-workflow 显示符号路由无成本优势 | CORE-2 降为工程组件,叙事转"属性感知检索 + 空集归因" |
| G4 组合已被占 | 收缩到"覆盖率空集归因"单点(§0 保护序第一位),其余降为系统 |
| C1 在 STaRK 不显著(第 4 周) | 先查键归一/言语化模板;仍无效则主战场移 KQA Pro/GrailQA 数值子集 + 领域集 |
| 属性平铺基线赢了 C1 | C1 不成立,全案重议(概率低,属性盲是实测普遍现象) |

**兜底底线**:C1(墙)+ CORE 中至少一半(空集归因 **或** 符号工作流)成立,即达"1–2 算法创新点"的毕设要求。novelty 保护序:覆盖率归因 > 符号门控异质路由 > C1。

---

## 11. 与项目其它文档的关系(一次说清,避免混淆)

- **取代**:`afr_rag_tech_plan_v1.md`(其 MINRELAX = 本版 A3 消融对照);
- **未采纳**:`thesis_roadmap.md`(置信度感知方向);
- **可前置**:`kg_agent_diagnosis_protocol_v1.md` —— 若先跑诊断,其错误分布可**客观坐实**
  本方案打在真痛点上(E2 通道↔工作流、E3a 覆盖↔空集归因、E4a 聚合漏答↔完备枚举);
  诊断的 T2/T3/T4 题层直接供给本方案领域评测集。**建议先诊断再上本方案**;
- **不改动**:`agent/`(6 算子 + strategy dispatcher + EW)、`pipeline/v3/`(KG 构建)、
  `graphrag_retriever.py`、`kg_v3/`、`thesis/` 已写章节 —— 全部只读。

---

*v2 | 承 v1 骨架,C2 升级为开放世界空集溯因、C3 升级为符号门控异质工作流,二者合流为单一算法核心。
开发前必须过 G1–G4 四道硬门(尤其 G3 覆盖率归因地基、G4 组合查新)。*
