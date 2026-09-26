# EW 本体 Schema：雷达对抗知识扩展

**版本**: v1　**日期**: 2026-06-09
**定位**: 在 `KG_SCHEMA.md`（v2）基础上扩展"对抗性知识"，支撑雷达对抗决策辅助。
**配套**: 机器可读的 doctrine 映射表见 `data/ew/doctrine_map.json`。

> 范围声明：本体内容为 **doctrine / 教科书级开源知识**（Schleher《EW in the Information Age》、
> Adamy《EW 101–103》、radartutorial.eu、FAS Navy EW 手册等）。效能/克制判断是**启发式、非实测**，
> 用于可解释决策支持与教学研究，不构成作战火控参数。

---

## 1. 新增实体类型

### 1.1 EWSystem（电子战装备）
干扰机 / 电子支援(ESM) / 雷达告警(RWR) / 诱饵。

| 属性 | 类型 | 必填 | 示例 |
|---|---|---|---|
| `id` | str | ✓ | `an_alq-249` |
| `name_en` | str | ✓ | `AN/ALQ-249 NGJ` |
| `name_zh` | str |  | `下一代干扰吊舱` |
| `ew_type` | enum | ✓ | `jammer / esm / rwr / decoy / ecm_pod` |
| `bands` | [str] |  | `["S","C","X","Ku"]`（→ `coversBand` 边） |
| `techniques` | [str] |  | jamming_technique id 列表（→ `supportsTechnique` 边） |
| `platform` | str |  | 搭载平台（→ `deployedOn`） |
| `country` | str |  | `美国` |
| `erp_dbw` | float |  | 有效辐射功率 |
| `status` | enum |  | in_service / development / retired |
| `description` | str |  | |

### 1.2 JammingTechnique（干扰技术本体）
压制 / 欺骗 / 无源三大类。**实例集见 doctrine_map.json `jamming_techniques`**。

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `rgpo` |
| `name_zh / name_en` | str | `距离门拖引 / Range Gate Pull-Off` |
| `category` | enum | `noise / deception / passive` |
| `subcategory` | str | `range_deception` |
| `effective_against` | [str] | 命中的 tracking_method / purpose / scan（见 §2.2） |
| `defeated_by` | [str] | ECCMTechnique id 列表（冗余索引，权威在 doctrine_map） |
| `geometry_note` | str | 战术几何/烧穿附注 |
| `description` | str | |

### 1.3 ECCMTechnique（抗干扰技术本体）
雷达侧反制。**实例集见 doctrine_map.json `eccm_techniques`**。

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `monopulse` |
| `name_zh / name_en` | str | `单脉冲测角 / Monopulse` |
| `category` | enum | `parameter / processing / antenna / operational` |
| `defeats` | [str] | JammingTechnique id 列表（它克制哪些干扰） |
| `description` | str | |

### 1.4 RadarPurpose（用途/威胁类）
作为 Radar 的 enum 属性 `purpose`（不单独建实体，便于查询）：
`search / acquisition / track / fire_control / SAM_guidance / GCI / AEW / EW / navigation / multifunction`

---

## 2. Radar 新增"威胁画像"属性 + 对抗关系

### 2.1 威胁画像属性（加到现有 Radar 实体）
| 属性 | 类型 | 取值 | 来源 |
|---|---|---|---|
| `purpose` | enum | 见 §1.4 | LLM 抽取 / 规则 |
| `tracking_method` | enum | `monopulse / conical_scan / lobe_switching / sequential_lobing / TWS / range_gate / none` | LLM 抽取 |
| `scan_type` | enum | `circular / sector / raster / phased / track` | LLM 抽取 |
| `freq_agile` | bool |  | 抽取自 eccm_description / mode |
| `prf_agile` | bool |  | 同上 |
| `lpi` | bool | 低截获概率 | 抽取 |
| `erp_dbw` | float |  | 抽取（可空） |
| `threat_priority` | int | 1–5（派生） | 由 purpose + tracking 派生 |

### 2.2 新增对抗关系（8 种）
| 关系 | 域 → 范围 | 含义 | 主要来源 |
|---|---|---|---|
| `vulnerableTo` | Radar → JammingTechnique | 易受某干扰 | 规则派生（doctrine_map 推荐规则） |
| `resistantTo` | Radar → JammingTechnique | 对某干扰鲁棒 | 规则派生（employsECCM→defeats） |
| `employsECCM` | Radar → ECCMTechnique | 采用的抗干扰技术 | LLM 抽取 eccm_description |
| `defeats` | ECCMTechnique → JammingTechnique | 抗干扰克制干扰 | doctrine 表（手工审核） |
| `effectiveAgainst` | JammingTechnique/EWSystem → (enum:tracking/purpose) | 干扰对哪类体制有效 | doctrine 表 |
| `coversBand` | EWSystem → FrequencyBand | 频段覆盖 | 抽取 |
| `supportsTechnique` | EWSystem → JammingTechnique | 装备支持的干扰样式 | 抽取 |
| `suppresses` | Weapon(ARM) → Radar/purpose | 反辐射武器压制 | 抽取/规则 |

> `counterAsset`（威胁 → 我方资产）不持久化为边，由 `recommend_counter` 算子在查询时**确定性派生**。

---

## 3. 推理派生约定（供 P1/P3 实现）

- `vulnerableTo` 派生：对每部 Radar，按其 `tracking_method / scan_type / freq_agile / purpose`
  查 doctrine_map 的 `recommendation_rules`，得推荐干扰技术集；再剔除被其 `employsECCM → defeats` 命中的，
  剩余即 `vulnerableTo`，被剔除的入 `resistantTo`。**每条派生边带 `rule_id + source + confidence`**。
- 置信：手工审核的 doctrine 表条目默认 0.85；纯文本抽取的画像属性按抽取置信；规则派生取规则置信 × 抽取置信。
- 一律标注 `evidence_type = "doctrine_heuristic"`，在报告中显式提示"非实测"。

---

## 4. 与现有 schema 的对齐
- 复用现有 `hasFrequencyBand` / `deployedOn` / `operatedBy` / `compatibleWith`。
- `employsECCM` 是现有 `eccm_description`（自由文本）的结构化版本，二者并存（文本作 evidence）。
- 武器 `compatibleWith` 中的反辐射型号（ARM）通过 `suppresses` 接入对抗推理。

---

## 5. 产出文件（P0）
```
EW_SCHEMA.md                 # 本文档：本体定义
data/ew/doctrine_map.json    # 机器可读：干扰/抗干扰本体 + defeats/effectiveAgainst + 推荐规则
```
后续阶段（P1+）再产出威胁画像抽取、EW 装备入库、派生边与推理算子。
