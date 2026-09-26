# Radar KG — quality evaluation

Fills the gap in thesis §3.5 (which had only "100% provenance coverage" — a trivial
metric). Implements the standard KG-quality dimensions (accuracy / consistency;
Zaveri et al.; sampling-based accuracy, Gao et al. VLDB'19) — the **automatable**
parts here, plus a stratified sheet for the **manual** gold-precision step.

KG: 16,513 triples, every triple carries source / confidence / evidence.

## (A) Consistency — syntactic accuracy (`kg_quality.py`)
- **Type-signature violations: 0.07% (12 triples)** after normalising a `Radar`/
  `RadarSystem` naming inconsistency in the schema (raw 6.49% was almost entirely
  that naming issue — itself a minor schema-hygiene finding). The 12 genuine errors
  are e.g. the entity "Thales" typed as `Manufacturer` but carrying radar attributes.
- **Self-loops: 0.04% (7).**
- **Malformed / noise tails: 3.2% (528)** — truncated extractions ("theu", "the").
- **Functional-relation contradictions: 222 heads** carry >1 value on a single-valued
  relation (developedBy / countryOfOrigin), e.g. `AN/MPQ-65 developedBy {raytheon,
  "theu"}`, `AN/APY-9 developedBy {5 surface variants}`. **This is the actionable
  finding** — it quantifies the entity-resolution + noise problem (the same one that
  caps agg_count accuracy in §4) and motivates a domain-specific entity-resolution step.

## (B) Evidence grounding — faithfulness of extractions to cited text
For sources whose `evidence` is **quoted source text**, does the cited text literally
contain the asserted tail? (a lexical lower bound — cross-lingual/normalised tails undercount)

| source | grounded | total | rate |
|---|---:|---:|---:|
| pdf_narrative_llm | 1523 | 1736 | **87.7%** |
| pdf_spec_block | 966 | 1255 | 77.0% |
| llm_zero_shot | 29 | 45 | 64.4% |
| llm_few_shot | 98 | 194 | 50.5% |
| **quoted-text overall** | 2616 | 3230 | **81.0%** |

The remaining sources are correctly excluded (their `evidence` is **not** quoted text):
~8,600 are **derivation notes** (inferences, low-confidence by design), ~2,460 are
**page-pointers** (checkable against the PDF, not by string match). So the 81% is a
real grounding signal for the text-extraction pipeline (PDF-narrative strongest), and
the rule/enrich tiers are honestly separated rather than mislabelled as hallucinated.

## (C) Independent cross-check vs Wikidata (`wikidata_xcheck.py`)
An independent reference (no human, no LLM): link non-Wikidata-sourced heads to
Wikidata and compare developedBy (P176) / countryOfOrigin (P495).

- **Coverage is very low** — only **8** developedBy and **11** countryOfOrigin radar
  heads were found in Wikidata at all (radar models are sparsely covered, as expected).
  So this is a **spot-check, not a primary metric**.
- **countryOfOrigin agreement: 8/11 (73%)** on the covered subset; some "misses" are
  label artifacts (`捷克`≈`Czechoslovakia`).
- **developedBy agreement: low and noisy** at n=8 (empty WD labels, `Westinghouse`
  spelling, Ericsson→Saab acquisition all depress it artificially).
- **The real value — it surfaced GENUINE errors:** `AN/APG-81` and `AN/APG-77` are
  listed as **Raytheon** in the KG but are actually **Northrop Grumman** products;
  `Globus III` country is likely wrong (KG 俄罗斯 vs Wikidata Norway). An independent
  source found real factual bugs — concrete evidence that accuracy evaluation is needed.

**Takeaway:** Wikidata is too sparse for radar to be the primary accuracy measure, but
works as a complementary independent spot-check that caught real errors. The **manual
annotation (§D) remains the gold-standard precision**; cite Wikidata as the independent
cross-check that surfaced N mis-attributions.

## (D) Stratified sample for MANUAL accuracy annotation
`annotation_sheet.csv`: 305 triples (~15 per source), each with head/relation/tail/
source/evidence and an empty `correct(1/0)` column. The student annotates by hand;
**precision = mean correct, with a 95% CI (Wald/bootstrap), reported overall and per
source** (the gold-standard accuracy number, per Gao et al. VLDB'19). Per-source
precision also validates whether the confidence tiers (§3.4) are empirically justified.

## (D′) LLM-assisted PRELIMINARY precision (`llm_annotate.py`)
A DeepSeek pre-label of the 120 verifiable facts (1/0/?), instructed to answer "?"
when it does not actually know — **a screen to speed the human pass, NOT the gold
number** (an LLM judging an often-LLM-extracted KG is partly circular).

- distribution: 75 correct / 11 wrong / **34 "?" unverifiable (28%)**
- **preliminary precision ≈ 87% (95% CI 80–94%)** on the 86 verifiable facts
- by evidence category (judged subset): **WIKI 100%** (n=28), **TEXT 90%** (n=30),
  **POINTER 86%** (n=22), **ATTR 17%** (n=6 — most ATTR numerics went to "?", so this
  is undetermined, not a reliable 17%).

Reading (honest): the *believable pattern* is curated/text sources (WIKI/TEXT) are
highly accurate, page-table extractions (POINTER) a bit lower, and **numeric attributes
are largely unverifiable** (28% of the whole sample) — which both (a) flags numeric
specs as the weakest, hardest-to-check part of the KG, and (b) suggests the uniform
0.95 confidence on `v2_attribute` is **over-confident** (a calibration finding). The
human must still review — especially the 11 zeros and the numeric "?"s (against the
original manuals) — to set the final precision. The pre-labels are written into
`annotation_facts.csv` (label + `[LLM]` reason) as the review starting point.

## (E) 改进抽取流水线验证(`extract_pipeline.py`) — 多智能体 vs 单次,同文本同模型(25 篇)
| 流水线 | 抽取率 | 接地率 | 合法关系率 |
|---|---:|---:|---:|
| 单次抽取(旧方法) | 13.0 条/篇 | **86%** | 97% |
| 多智能体(抽取+批判) | 11.7 条/篇 | 82% | 96% |

**诚实负面结果:多智能体的"批判 agent"没有提升质量,反而略降**(接地 86%→82%、产量 13→11.7)。
原因:云端抽取**本就高保真(86%)**,批判 agent 提升空间小,反而误毙了一些合法三元组
(如 `AMES Type 82 -hasFunction-> early warning` 本是对的也被毙)。→ **不建议加多智能体批判层**
(成本↑、质量↓);这是"基线已够好,加层无益"的又一例(同 GVR/真值发现)。

**但"受控 schema 扩展"有真价值**:抽取 agent 提议了现有 28 关系**装不下的真实关系**——
`replacedBy`、`partOf`、`developedAt`、`usedInConflict`、`designationSystem` 等。这是 schema 的
真实缺口。→ **建议采纳此一项**(人工审核后把好的新关系加入 schema),低风险、不破坏类型核心。

→ 修正建议:**① 多智能体批判:实测无益,弃。② 受控 schema 扩展:有效,采纳(审核+加关系)。
③ 实体消歧/规范化(222 冲突):仍是最高价值的未验证项,值得做。**

## (F) 实体消歧/规范化验证(`canonicalize.py`) — 诚实负面
确定性的保守变体合并 + 噪声清除,实测:
- 功能性关系冲突: **222 → 217**(只减 5,≈2%);
- agg_count 精确率: **64% → 50%(变差!)**;平均计数误差 0.86 → 5.92(大幅变差)。

**结论:规范化在这个 KG 上弊大于利,弃。** 两个原因(诚实诊断):
1. **那 222 个"冲突"多数不是简单变体**——而是合法多值(联合研制)或**不同来源给的不同值**
   (如 S-300 的 MZiK vs NPO Almaz),变体合并根本碰不到 → 冲突只减 2%;
2. **原始计数本就接近金标**(平均误差仅 0.86,64% 完全精确)——**去重头部空间很小**;
   激进合并反而**误并了不同型号**(如 `AIM-7F/M`→`AIM-7F`),把计数搞偏 → 精确率不升反降。

→ 这是第三个"实测不奏效"的改进(继多智能体、真值发现):**一致的原因是 KG/系统本就够好,
头部空间小,激进"改进"反而引入误差**。**实体消歧:弃。**

## (G) 受控 schema 扩展验证(`schema_expand.py`,60 篇)— 正面,采纳
让抽取器在 60 篇上提议现有 28 关系装不下的新关系,按频次聚合(候选清单,人工审核):

**① 真正的关系缺口(有向边、可类型化、Radar→X)——建议加为新关系:**
| 候选 | 频次 | 例 | 评 |
|---|---:|---|---|
| `hasVariant` | 11 | Type 281→Type 281B | 最高频最干净,型号变体(逆于 upgradeOf) |
| `replaces`/`replacedBy`/`successorOf` | 5+4+2 | Voronezh replaces Daryal | 服役更替,合并成一对 replaces/replacedBy |
| `installedAt` | 5 | AN/FPS-17→Pirinçlik 基地 | 固定部署**地点**,与 deployedOn(平台)互补不同型 |

**② 可并入现有关系(不必新增):** `mountedOn`→`deployedOn`;`designedBy`→`developedBy`;
`basedOn`/`influenced`→`derivedFrom`;`monitors`/`supportsMission`/`hasCapability`→`hasFunction`/`hasMode`。

**③ 数值/类别属性(是属性不是关系,应进属性 schema;数值类是 §D 最弱可验证项,标低置信):**
`hasRange`/`hasDetectionRange`/`hasMaxTrackTargets`/`hasPulseWidth`/`hasPowerLevel`/`hasFrequencyRange`/
`hasAccuracy`/`hasCost`(数值);`hasAntennaType`/`hasDisplayType`/`hasTransportMode`/`hasDesignation`/
`hasDesignationSystem`(类别)。

**结论(对比 ①③ 的负面):schema 扩展是唯一实测有真价值的改进** —— 它**只增不毁**,
surfaced 了 `hasVariant`(11x)、服役更替(11x)、`installedAt`(5x)三个真实关系缺口。
建议**采纳 ① 三个结构关系**(定向补抽填充),数值属性可选(标低置信)。这给论文一个干净的故事:
**多智能体/真值发现/实体消歧都无益(基线已好),但受控 schema 扩展提升了图谱覆盖度(coverage 维度)。**

### (G′) 落地结果(`extract_newrels.py`,全 249 篇,已并入 KG)
采纳 **3 结构关系 + 5 属性**(已写入 `lexicon/relations.json`,34→43;数值属性标 `low_confidence`),
全语料定向补抽,**接地门控**(值须在原文出现)+ provenance(`source=schema_expand_llm`/置信分层/引文证据):

| | |
|---|---|
| 提议 → 接地 → 去重入库 | 603 → **585(97%)** → 583 条新边 |
| hasVariant / installedAt / replaces+replacedBy | 187 / 122 / 77 |
| hasAntennaType / hasTransportMode | 110 / 65 |
| 数值(maxTrack/pulse/accuracy) | 14 / 3 / 5(稀少,印证 §D 数值最弱) |
| **KG 规模** | 16513 → **17096**(+583);190 雷达获新边;+110 地点节点 |

接地率 **97%**(高于原始 86%,因门控滤掉无原文支撑项)。**既有关系查询不变**(developedBy Thales=37、
Marconi=23、S 波段=50,加性合并不破坏旧 QA),新关系可查(`EL/M-2080→Green Pine Block-B` 等)。
备份 `merged_triples.pre_schema_expand.bak.json`。→ **schema 扩展:已采纳并入库,覆盖度↑、忠实度门控保证。**

## (H) 多智能体抽取 + 批判门控:干净 vs 噪声消融(`multi_agent_extract.py`,30 篇)
把抽取重构为三角色流水线 **Extractor → Critic(忠实度门控) → Schema-Proposer**,
对**同一文档**变化输入噪声(clean vs OCR 模拟降质 p=0.18),接地一律对**干净原文**判(真值):

| 输入 | Extractor 接地 | +Critic 接地 | 增益 | 产量 |
|---|---:|---:|---:|---|
| 干净 | 83% | **88%** | **+5.1pp** | 376→320 |
| 噪声 | 81% | 85% | +3.4pp | 366→301 |

**三条诚实结论:**
1. **忠实度门控型批判确实提升接地**(+5.1pp 干净 / +3.4pp 噪声),代价是约 15% 产量↓ ——
   precision↑ 换 recall↓,**正合可审计情报 KG 的取舍**(宁缺毋滥)。
2. **增益不随噪声上升**:词级 OCR 噪声只把抽取从 83%→81%(模型能读穿错字),
   故批判的价值是"忠实度-精度",而非"噪声恢复"。
3. **范围决定成败**:本实验批判**只判"值是否在原文"**→ 有效;而 §E 那个**还判"关系类型是否合理"**
   的更宽批判**误毙合法领域事实**(如 `AMES Type 82 -hasFunction-> early warning`)→ 接地 86%→82%。
   **设计教训:批判须窄化到 value-in-text,不要让它二次猜测 schema。**

→ 修正总账:**多智能体 = Extractor + 窄范围忠实度 Critic + Schema-Proposer**。Critic(窄范围)正面、
Proposer 正面(+583 边)、宽范围 Critic 负面。这给抽取章一个**有方法论分量且诚实**的故事,
而非"单次调 LLM"。

## (I) 多源印证 / 一致投票(`source_agreement.py`)— Knowledge-Vault 思路,此 KG 上受限
按**独立来源**(wikipedia / 手册PDF / Wikidata / GlobalSecurity)统计每条事实被几个来源印证:

| 来源 | 事实数 |
|---|---:|
| wikipedia(规则+LLM) | 1046 |
| manual(PDF 手册) | 4323 |
| wikidata | 257 |
| globalsecurity | 71 |

- 并集 5680 条,**仅 17 条(0.3%)被 ≥2 独立来源印证**;
- 验证信号方向正确但样本极小:1 来源接地 44%、2 来源 75%(n=8);
- 功能性关系跨源取值"一致"仅 20%(且多为**归一化假象**:Raytheon vs Raytheon Company、美国 vs USA)。

**诚实结论:多源投票在本 KG 上几乎不适用** —— 因为各来源**互补(覆盖不同雷达)而非冗余**,
重叠极少。这反而**正面论证了本文的"逐源置信分层"设计**(§3.4)优于依赖跨源印证的方案。
(又一个"方法本身合理、但此数据上不奏效"的诚实记录。)

## (J) NLI 语义接地(`nli_grounding.py`)— 比字符串接地更硬的忠实度
审计**引文本身**:给 LLM 裁判【引文(premise)】+【三元组论断】,只判原文是否支持(读理解非世界知识,
缓解"LLM 判 LLM 图谱"的循环)。分层样本 n=178(8 类引文型来源):

| | |
|---|---:|
| 字符串接地(样本内) | 115/178 = 65% |
| **NLI 语义支持** | **102/178 = 57%** |

**2×2(字符串命中 × NLI):**
| | NLI 支持 | NLI 不支持 |
|---|---:|---:|
| 字符串命中 | 78(真接地) | **37 ← NLI 揪出**(字面在但关系不成立) |
| 字符串未命中 | **24 ← NLI 找回**(改写/翻译) | 39(确未支持) |

- **NLI 双向纠正字符串法**:找回 24(`GE`→General Electric、`shipborne`→水面舰艇、`英国`隐含),
  揪出 37(`AN/APG-73 operatedBy Switzerland`、`Voronezh replaces Volga`——引文只列了雷达名、并未说"取代")。
- 揪出 > 找回 → **字符串接地高估了忠实度**;NLI 57% 是更硬的(且严格,引文片段常很短,含"欠完整引文"与少量裁判误判,属下界)。
- **连我自己加的 schema 扩展边也被揪出可疑项**(Voronezh/Volga)——评测对自身产物同样有效。

**结论:NLI 语义判定 > 字符串接地**,既补字符串的漏判又揪其误判,是可写进论文的忠实度指标;
裁判仍非完美(个别误拒),故定位为"强自动代理"而非金标。

## (K) 人工金标准确率(`make_gold_sheet.py` / `score_gold.py`)— 真·金标,已完成
按 Gao(VLDB'19)分层抽样,领域专家对 **173 条**逐条人工核对(每行带可判断证据),核实率 **99%**:

| 估计 | 精度 | 95% CI |
|---|---:|---|
| **分层加权(按来源占全库比)** | **85.1%** | [75.1%, 95.0%] |
| 合并 Wilson(交叉验证) | 90.7%(156/172) | [85.4%, 94.2%] |

- **分层<合并**:最大来源层 `v2_attribute`(数值属性,占库≈1/3)精度仅 **64%**,加权后拉低整体——
  这是 KG 的真实代表性准确率,并**用人工金标证实"数值属性是最弱层"**(此前 §D 仅由 LLM 自评推测)。
- **per-source 验证置信分层**:`stage_c`/`pdf_spec`/`wikidata`/`schema_expand` 100%、`pdf_narrative` 92%;
  数值/性能属性最弱(`v2_attribute` 64%、`extract_perf_data` 75%)。**§3.3.1 新增的 schema 扩展边人工核验 100%。**
- **16 个判错的错误类型学,与自动指标互证**:①张冠李戴(APQ-120 平台挂到 APQ-36/AWG-11)=**NLI 同型**;
  ②收购方≠原研(Northrop Grumman vs Westinghouse/Bendix)=**Wikidata 交叉核验同型**;③关系混淆
  (吸取经验≠衍生自、developed *for*≠研制方);④数值拼接/错值;⑤劣质头实体("ATR,"、地名当雷达)。

**这取代了此前来路不明的"87%":现在有一个真实、分层、带 CI、且与自动评估相互印证的人工金标准确率。**
弃用的 LLM 自标注(`annotation_template_labeled.json`)不作金标。

### (K′) 按来源层分解准确率(诚实:85% 不是全库,推断层未测)
单一 85% 会**混合强手册层与弱属性层,且完全不含推断层**。按层分解:

| 来源层 | KG占比 | 人工精度 | 95%CI |
|---|---:|---:|---|
| 手册(PDF/OCR) | 42% | **93%** | [84,97] |
| Wikidata | 2% | 92% | [76,98] |
| 百科 LLM | 2% | 90% | [78,95] |
| schema 扩展 | 3% | 100% | [76,100] |
| **属性摊平** | 25% | **69%** | [42,87] |
| **规则/推断富化** | **25%** | **未测** | 低置信,设计上应过滤 |

- **抽取层普遍 90–93%**(比混合的 85% 高);拖后腿的是**属性摊平层 69%**(见 (L) 清洗)。
- **推断富化层占库 25% 却零标注** —— 85% 只是**可核验抽取部分(约占库 75%)**的加权值;
  **下游若不按 §3.4 置信过滤推断层,实际可用准确率将低于 85%**。这是须诚实声明的**使用边界**。
- `similarTo`(256)几乎全来自推断层(255/256=`enrich_similar_llm`),是"同频段+同用途+同国"的**共类兜底**,
  低信息量 → 是"降解为精确关系(variant/derivedFrom/replaces)"的首要目标(92 条同型号族可降解)。

## (L) v2_attribute 数据清洗(`clean_v2attr.py` / `repair_v2attr.py`)— 分层,诚实
金标诊断:最弱层 `v2_attribute`(人工 64%)的错误集中在 `*_description` **OCR 乱码(9%)**+ 劣质头(3%),
**数值字段干净**。据此分层清洗:

- **层1 确定性(安全,已应用)**:删 **187** 条"乱码描述 + 已有干净数值兄弟"的冗余错误副本(**零信息损失**)
  + 修 8 头 + 2 浮点 + 将 **329** 条无兄弟乱码 OCR **降置信(conf→0.3)**(不删不臆造)。**KG 17096→16909**;
  样本 v2_attribute 精度 **64%→70%**(删错 1/删对 0,方向理想)。
- **层3 LLM 接地重抽(诚实负面)**:330 条无兄弟乱码里仅 **30 条 head 在语料**,重抽+接地门控后**仅修复 1 条**。
  原因:乱码是**手册细参数**(脉宽/PRF/接收机/方位),清洁源(原始手册)不在语料、wiki 也无此细节 → 无据可抽。
  **结论:OCR 乱码的手册参数在缺原始手册时无法自动修复;正确处置是"删冗余 + 降权不可验证",而非编造。**
- QA 核心不受影响(只动字面属性;Thales=37 / Marconi=23 / S=50 不变)。备份 `merged_triples.pre_v2clean.bak.json`。

这是"验证再落地"的又一例:确定性去噪有实效且安全;花哨的 LLM 自动修复经实测在此数据上近乎无效(1/330),如实报告。

## How this answers "the KG is just LLM extraction"
- Adds the **missing accuracy/consistency evaluation** §3.5 lacked.
- Surfaces **real, quantified defects** (222 conflicts, 3.2% noise) → motivates a
  genuine non-LLM method (entity resolution) and validates the citation-grounding idea.
- The **81% evidence-grounding** + **citation-verification filter** (move from §6 to §3)
  are the "real method" beyond prompting: extractions are traceable to source, measurably.
