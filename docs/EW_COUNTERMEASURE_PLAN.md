# 雷达对抗决策辅助：真实场景调研 + 可落地方案

**版本**: v1 draft　**日期**: 2026-06-09
**目标**: 把系统从「雷达图谱问答」推进到「雷达对抗决策辅助」——给定敌方雷达布控（EOB），
推荐我方该用什么装备对抗、怎么干扰、怎么抗干扰，并给出**可溯源、带置信**的依据。

---

## 0. 一句话定位

不是做"作战火控系统"，而是做一个 **doctrine 级（教科书级）的电子战决策支持助手**：
用开源知识构建"威胁→对抗手段"的知识库 + 确定性匹配推理 + 可审计报告。
所有效能判断都是**公开 doctrine 的启发式**，显式标注"非实测"，不进硬信任路径——
这与现有系统"关键结论由确定性 trace 生成、LLM 只润色"的原则完全一致。

---

## 1. 真实场景：电子战决策链（调研结论）

电子战的对抗决策是一条标准 kill chain（参考美军 EW doctrine、radartutorial、FAS Navy EW 手册）：

```
①探测/截获        ②识别/编目            ③决策                ④行动        ⑤评估
ESM/ELINT 收信号 → 比对威胁库定型号/类 → 选对抗手段+战术几何 → 干扰/规避/摧毁 → 评估效果→回填威胁库
(RWR告警)         (建立 EOB 电子战斗序)   (本方案的核心)                         (闭环学习)
```

### 1.1 识别一部敌方雷达，关心哪些参数（→ 这就是"威胁画像"字段）
- **工作频段** (A–J / L,S,C,X,Ku…)：决定我方干扰机/雷达频段是否覆盖
- **PRF（脉冲重复频率）**、**脉宽/脉压**：决定欺骗干扰的时序设计
- **天线扫描方式**（圆扫/扇扫/相控阵）：决定能否用扫描相关的欺骗
- **跟踪体制**（单脉冲 monopulse / 圆锥扫描 conical scan / TWS 边扫边跟）：**最关键**，直接决定用哪种欺骗干扰
- **ERP（有效辐射功率）**：决定 burnthrough（烧穿）距离
- **频率/PRF 捷变、LPI（低截获概率）**：决定干扰是否要用"跟随式/宽带拦阻"
- **用途/威胁等级**（搜索/引导/火控/SAM 制导/GCI/预警）：决定威胁优先级

### 1.2 对抗手段三大类（→ 这就是"干扰技术本体"）
- **压制式干扰（noise）**：瞄准(spot)/拦阻(barrage)/扫频(sweep) —— 抬高噪声底，降低探测距离
- **欺骗式干扰（deception）**：距离门拖引(RGPO)、速度门拖引(VGPO)、角度欺骗(inverse-gain/cross-eye/cross-pol)、假目标
- **无源 + 平台**：箔条(chaff)、诱饵(decoy)、隐身/RAM；以及**反辐射摧毁**（ARM 反辐射导弹，如 HARM）

### 1.3 抗干扰（ECCM，敌方雷达的反制 → "抗干扰技术本体"）
频率捷变、PRF 抖动、旁瓣对消/旁瓣匿影(SLC/SLB)、单脉冲（抗角度欺骗）、脉压、MTI/CFAR、多普勒鉴别、EMCON。
**关键 doctrine 事实**：单脉冲跟踪天然抗大多数角度欺骗 → 要用 cross-eye/cross-pol；频率捷变 → 瞄准式失效，要用拦阻式。

### 1.4 战术几何与烧穿（→ 给出建议时要附的约束）
自卫干扰(SSJ)/远距支援(SOJ)/随队(SFJ)；burnthrough range 由 雷达 ERP vs 干扰功率 决定。
本方案在"建议"里把这些作为**附注/约束条件**呈现，不做精确电磁计算。

> **核心洞察**：真实世界里"跟踪体制 → 推荐欺骗技术""频段 → 干扰机选型""频率捷变 → 干扰样式"
> 这些大多是 **doctrine 确定性映射**（教科书级），天然适合做成一张**可审计规则库**——这正是知识图谱 + 算子的强项。

---

## 2. 落地用例（系统对外能力）

**输入**：敌方雷达布控 EOB —— 一组 {雷达型号或类型, 平台/角色, （可选）位置}。
**输出**：对每个威胁，给出三类建议 + 整体对抗方案报告，每条带 *依据规则 + 来源 + 置信*：

| 子问题 | 系统回答 |
|---|---|
| **用什么对抗** | 我方可用的克制装备/平台（干扰机/反辐射武器/我方雷达），需频段覆盖且体制对路 |
| **怎么干扰对方** | 针对其跟踪体制/扫描/捷变，推荐干扰样式（spot/barrage/RGPO/VGPO/cross-eye…）+ 战术几何附注 |
| **怎么抗对方干扰** | 若敌方是干扰源/我方雷达被压制，推荐 ECCM 要点（捷变/单脉冲/SLB…） |

示例交互："敌方部署了 1 部 X 波段单脉冲火控雷达 + 1 部 S 波段搜索雷达，我方该怎么应对？"
→ 威胁排序 → 火控雷达：cross-eye 角度欺骗（单脉冲抗 inverse-gain）+ 反辐射打击候选；搜索雷达：拦阻噪声（注意其频率捷变）→ 推荐我方覆盖该频段的干扰机 → 生成可溯源对抗方案报告。

---

## 3. Gap 分析：现有 KG 缺什么

现有 16.5k 三元组是**装备名录（描述性）**：型号、参数、研制国、平台、模式、功能。
要做对抗决策，缺的是**对抗性知识**：

| 缺失 | 说明 |
|---|---|
| 威胁画像参数 | `scan_type` / `tracking_method` / `erp` / `freq_agile` / `prf_agile` / `lpi` / `purpose(威胁类)` 基本没有（PRF/频段已部分有） |
| EW 装备本体 | 干扰机/ESM/RWR/诱饵（AN/ALQ-99、NGJ、Krasukha、AN/ALR-67…）几乎不在库 |
| 干扰技术本体 | spot/barrage/sweep/RGPO/VGPO/cross-eye/chaff… 不存在 |
| 抗干扰技术本体 | 现仅 `eccm_description` 51 条自由文本，没有结构化的 ECCMTechnique 实体 |
| 对抗关系 | `vulnerableTo / counteredBy / employsECCM / defeats / effectiveAgainst / suppresses` 全无 |
| doctrine 映射 | "跟踪体制→推荐干扰""频段→干扰机选型"这张教科书表不存在 |

---

## 4. 需要补充的知识（schema 扩展，在 v2 基础上加）

### 4.1 新实体类型
| 类型 | 关键属性 | 例 |
|---|---|---|
| `EWSystem` | ew_type(jammer/ESM/RWR/decoy), bands[], platform, country, technique_support[] | AN/ALQ-249 NGJ、Krasukha-4、AN/ALR-67 |
| `JammingTechnique` | category(noise/deception/passive), sub(spot/barrage/RGPO/VGPO/cross-eye/chaff), targets(scan/track/search) | RGPO、cross-eye |
| `ECCMTechnique` | category(parameter/processing/operational), defeats[] | 频率捷变、单脉冲、SLB |
| `RadarPurpose`（或在 Radar 上做 enum 属性） | search/acquisition/track/fire_control/SAM_guidance/GCI/AEW/EW | — |

### 4.2 Radar 新增"威胁画像"属性
`scan_type`(circular/sector/phased/raster) · `tracking_method`(monopulse/conical/lobe/TWS/none) ·
`erp_dbw` · `freq_agile`(bool) · `prf_agile`(bool) · `lpi`(bool) · `purpose`(enum) · `threat_priority`(派生)

### 4.3 新增关系（对抗本体）——共约 8 种
| 关系 | 域 → 范围 | 含义 |
|---|---|---|
| `vulnerableTo` | Radar → JammingTechnique | 该雷达易受某干扰技术 |
| `resistantTo` | Radar → JammingTechnique | 该雷达对某干扰技术鲁棒 |
| `employsECCM` | Radar → ECCMTechnique | 该雷达采用的抗干扰技术 |
| `defeats` | ECCMTechnique → JammingTechnique | 抗干扰技术克制哪类干扰 |
| `effectiveAgainst` | JammingTechnique/EWSystem → (tracking_method/purpose) | 干扰对哪类体制有效 |
| `coversBand` | EWSystem → FrequencyBand | 干扰机/接收机频段覆盖 |
| `suppresses` | Weapon(ARM) → Radar/purpose | 反辐射武器压制 |
| `counterAsset`（派生）| Threat → 我方 EWSystem/Radar/Weapon | 综合推荐的克制资产 |

### 4.4 doctrine 映射表（核心、且大部分可规则派生）
一张教科书级确定性表，是整套推理的"规则知识"：
- `tracking_method=monopulse` → 推荐 cross-eye / cross-pol；inverse-gain 失效
- `tracking_method=conical/lobe` → inverse-gain / AGC 欺骗有效
- `tracking_method=range-gate` → RGPO；`velocity/PD` → VGPO；`TWS` → 假目标群
- `purpose=search` → 噪声压制（spot 若窄带、barrage 若捷变）
- `freq_agile=true` → spot 失效 → barrage / 跟随式；`lpi=true` → 难截获，提示 ESM 局限
- `employsECCM 含 SLB/SLC` → 旁瓣干扰失效，需主瓣对抗
每条映射带"教科书出处/规则 id"，可人工审核。

---

## 5. 数据来源（怎么补，按成本排序）

- **(a) 规则派生（0 成本，复用 `kg_enrichment.py` 模式）**：`vulnerableTo / resistantTo` 大量可由
  `tracking_method + freq_agile + employsECCM` 经 doctrine 表派生，**每条带规则出处与置信**。
- **(b) doctrine 本体（手工 + LLM 审核，~80 条，权威）**：JammingTechnique / ECCMTechnique 本体 + `defeats` / `effectiveAgainst` 映射表。一次性建好，是知识不是凑数。
- **(c) Wikipedia/公开抽取（复用 `web_tool.py` + LLM）**：具体 EW 装备（干扰机/RWR/ESM）及其频段覆盖、所属平台、国别。
- **(d) 威胁画像参数补全（LLM 从现有语料）**：`scan_type / tracking_method / purpose / agility` 从已有 `eccm_description / hasMode / hasFunction` 文本里抽，部分（PRF/频段）已在库。

> **诚信说明（重要，针对此前"反对为刷边数做规则推断"的顾虑）**：这里的规则派生**不是为了凑边数指标**，
> 而是编码真实 EW doctrine（教科书确定性知识），每条带规则出处 + 人工审核 + 置信标注，
> 用途是支撑可解释推理，不是充数。效能评级一律标注"doctrine 启发式、非实测"。

---

## 6. 推理与工具（agent 怎么用，最大复用现有架构）

### 6.1 新增确定性算子/工具：`recommend_counter`
输入一个威胁雷达（型号或威胁类）→ 在 KB 上做确定性多跳匹配：
1. 取威胁的 `tracking_method / band / purpose / employsECCM`；
2. 沿 `vulnerableTo` + doctrine 表得候选干扰技术（剔除被 `employsECCM→defeats` 克制的）；
3. 与"我方可用资产"求交：`operatedBy=我方` 的 EWSystem 且 `coversBand` 命中 且 `effectiveAgainst` 对路；
4. 输出候选列表，每条挂 *依据规则 + 来源 + 置信 + 战术附注（几何/烧穿提示）*。

完全复用现有 `composition.py` 的 `path/constraint/intersect/difference` 算子 + 新增 doctrine 规则查询；
保持**可审计**：结论来自确定性 trace，LLM 只组织叙述。

### 6.2 新增 agent 工具（挂到 `agent_loop.py` 的工具箱）
- `threat_assessment(EOB)`：录入敌方布控 → 威胁画像 + 优先级排序
- `counter_recommendation(threat)`：单威胁的对抗建议（上面的匹配引擎）
- `jamming_plan(threat)` / `eccm_advice(my_radar, threat)`：干扰样式 / 抗干扰要点
- 反思节点：若我方无覆盖该频段资产 → 提示能力缺口 / 触发 `web_search` 补 EW 装备

### 6.3 可溯源"对抗方案报告"
复用 `report.py` 模式，新增 `EWPlanReporter`：威胁清单 → 每威胁的建议表（技术/资产/依据/置信）→ 总体方案 + 缺口提示。

---

## 7. 场景化 UI（在现有 Streamlit 上加"对抗推演"模式）
- 新模式：录入敌方 EOB（多部雷达型号/类型 + 平台/角色，可下拉或文本）→ 一键"生成对抗方案"
- 输出：威胁排序卡片 + 每威胁的"对抗装备 / 干扰技术 / 抗干扰要点" + 可下载的对抗方案报告
- 复用现有实时 ReAct 轨迹与溯源徽章；效能评级显式标 "doctrine 级·非实测"

---

## 8. 边界与诚信（毕设答辩防御点）
- **定位**：决策**支持** + 教学/研究；doctrine 级开源知识；**非作战火控、非实弹/精确电磁参数**。
- **数据**：全部开源（Wikipedia / 公开 EW 教科书 / radartutorial / FAS 等），无涉密。
- **可信路径**：效能/克制判断是 doctrine 启发式，标注"非实测"，不进硬信任路径；与现有"LLM 不进信任路径"原则一致。
- **学术贡献点**：把电子战 doctrine 形式化为**可审计知识图谱 + 确定性匹配推理**，是"知识工程 + 可解释决策支持"的应用创新（区别于纯检索问答）。

---

## 9. 分阶段落地计划

| 阶段 | 内容 | 产出 | 估时 |
|---|---|---|---|
| **P0 本体与 schema** | 定义 EWSystem/JammingTechnique/ECCMTechnique 实体 + 8 种对抗关系；建 doctrine `defeats`/`effectiveAgainst` 映射表（~80 条手工审核） | `EW_SCHEMA.md` + `data/ew/doctrine_map.json` | 1–2 天 |
| **P1 威胁画像补全** | LLM 从现有语料抽 `scan_type/tracking_method/purpose/agility`；规则派生 `vulnerableTo/resistantTo`（复用 enrichment，带出处） | 扩充 triples + 派生报告 | 2–3 天 |
| **P2 我方 EW 装备入库** | `web_tool` 抽干扰机/ESM/RWR + 频段覆盖 + 平台 + 国别 | EWSystem 实体集 | 2 天 |
| **P3 推理算子 + 工具** | `recommend_counter` 匹配引擎 + 4 个 agent 工具 + `EWPlanReporter` 可溯源报告 | 代码 + 单元/组合评测 | 3 天 |
| **P4 场景 UI** | Streamlit "对抗推演"模式 | app.py 扩展 | 1–2 天 |
| **评测** | 构造 N 个 EOB 场景，与 doctrine 标准答案对照（技术选型正确率），类比现有组合评测 99% 的做法 | `eval_ew.py` + 报告 | 1 天 |

**最小可行版（MVP，约 1 周）**：P0 + P1（仅规则派生）+ P3 的 `recommend_counter` + 在现有 UI 加一个对抗查询入口。
先用"敌方雷达型号 → 推荐干扰样式 + 抗干扰要点（带 doctrine 依据）"跑通闭环，再逐步加我方装备库与场景 UI。

---

## 10. 与现有系统的衔接（不重造轮子）
- 算子：复用 `composition.py` 的 path/constraint/intersect/difference；新增 doctrine 规则查询节点。
- 溯源：复用 `report.py` 的 `ProvenanceKG` + 置信标注；新增 `EWPlanReporter`。
- 联网：复用 `web_tool.py` 抽 EW 装备。
- agent：复用 `agent_loop.py` 工具箱机制 + 反思节点 + 可审计 `_finalize`。
- 评测：复用 `gen_composition_eval.py` / `eval_composition.py` 的"执行器算 gold"范式，造 EOB→对抗 的评测集。
