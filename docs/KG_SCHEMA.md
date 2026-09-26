# RadarKG 重设计：实体类型 / 属性 / 关系 / 抽取扩展方案

**版本**: v2 schema draft
**日期**: 2026-04-29
**目标**: 把当前"扁平化三元组"重设计为标准的 property graph，实体带属性、边只表达实体间语义；规模目标 3384 实体 + 1w+ 边 + ~3k 属性。

---

## 0. 当前问题诊断

| 维度 | 现状 | 问题 |
|---|---|---|
| 实体属性 | 不存在 | 雷达的"重量/频率/造价/服役年代"等被表达成 14 种 `hasXXX` 关系 + Literal tail（共 1241 条三元组），不利于查询和检索 |
| 关系语义 | 28 种关系混杂"属性"和"边" | 14 种 literal-tail（应是属性）+ 14 种 entity-tail（真正的边），界线没分清 |
| 边密度 | 真正的边只有 4351 条 / 3384 节点 = **平均度 1.3** | 远低于工业 KG（典型 3-8）；图扩展、多跳推理、子图检索能力受限 |
| 实体类型粒度 | 17 种 | RadarSystem / Radar 两套并存，平台细分混乱（Platform / NavalVessel / AircraftPlatform / GroundPlatform / Aircraft 五套）；没有 Subsystem / Component / Conflict / Standard 这些有用的类型 |
| 数据噪声 | extraction artifacts | "the same company" 当 Manufacturer，OCR 残片当属性值 |

## 1. 目标 schema 总览

```
─────────────────────────────────────────────────────────────────
  ENTITIES (with properties)         EDGES (inter-entity only)
─────────────────────────────────────────────────────────────────
  Radar / RadarSystem                28 entity-edge relations
  ├─ Subsystem (NEW)                 ├─ 既有 14 种保留
  ├─ Component (NEW)                 └─ 14 种新增（详见 §3）
  Manufacturer
  ├─ Subsidiary (NEW, soft type)
  Country
  Platform                           PROPERTIES (per entity)
  ├─ NavalVessel                     ├─ Radar  ~30 attrs
  ├─ AircraftPlatform                ├─ Manufacturer ~8 attrs
  └─ GroundPlatform                  ├─ Country ~5 attrs
  FrequencyBand                      └─ ...
  TechType
  RadarMode
  Function
  Weapon
  Standard (NEW, replaces blacklist) Total target:
  Conflict (NEW)                       3,384 nodes
  Location (NEW)                       10,000+ edges
─────────────────────────────────────────────────────────────────       3,000+ properties
```

---

## 2. 实体类型与属性

每个实体类型给出：**核心属性**（必有）/ **可选属性** / **示例**。属性命名小写下划线，类型标注（str / int / float / [str]…）。

### 2.1 Radar（核心实体）

继承自 RadarSystem 但做合并——v2 中只保留 Radar 一种，用 `system_level: bool` 区分系统级 vs 子型号级。

| 属性 | 类型 | 必填 | 来源 | 示例 |
|---|---|---|---|---|
| `id` | str | ✓ | 命名规范化 | `an_apg-68` |
| `name_en` | str | ✓ | 抽取保留 | `AN/APG-68` |
| `name_zh` | str |  | 别名表 | `AN/APG-68 雷达` |
| `aliases` | [str] |  | 抽取 | `["APG-68", "AN/APG-68(V)"]` |
| `system_level` | bool | ✓ | 推断 | `false` |
| `country_of_origin` | str |  | 关系折叠 | `美国` |
| `developer` | str |  | 关系折叠 | `Northrop Grumman` |
| `operator_primary` | str |  | 关系折叠 | `美国` |
| `frequency_GHz_min` | float |  | 抽取 | `8` |
| `frequency_GHz_max` | float |  | 抽取 | `12.5` |
| `frequency_band` | str |  | 抽取 | `X` |
| `range_km` | float |  | 抽取 | `185` |
| `range_description` | str |  | 抽取 | `185km(空对空)` |
| `peak_power_kW` | float |  | 抽取 | `25` |
| `power_consumption_W` | float |  | 抽取 | `200` |
| `weight_kg` | float |  | 抽取 | `120` |
| `mtbf_hours` | int |  | 抽取 | `220` |
| `lru_count` | int |  | 抽取 | `4` |
| `antenna_gain_dB` | float |  | 抽取 | `32` |
| `cooling_method` | str |  | 抽取 | `风冷+液冷` |
| `eccm_description` | str |  | 抽取 | `频率捷变；脉冲压缩` |
| `price_usd` | float |  | 抽取 | `1_200_000` |
| `status` | enum |  | 抽取 | `in_service` (in_service/retired/development/cancelled/unknown) |
| `researched_in` | str |  | 抽取 | `1970s` (年代或具体年) |
| `deployed_in` | str |  | 抽取 | `1981` |
| `tech_types` | [str] |  | 关系折叠 | `["Pulse Doppler", "AESA"]` |
| `frequency_bands` | [str] |  | 关系折叠 | `["X", "Ku"]` |
| `description` | str |  | 抽取 | 一句话描述 |

> **设计权衡**：把"主要 developer / operator / country"作为属性方便单跳查询；多对多关系（多个 frequency_band、多个 tech_type）用 list 属性；对应关系仍保留为边（用于反向查询和多跳）。

### 2.2 Subsystem（新增）

雷达系统的子模块（天线、收发机、信号处理器、显控等）。

| 属性 | 类型 | 必填 |
|---|---|---|
| `id` | str | ✓ |
| `name` | str | ✓ |
| `subsystem_type` | enum | ✓ | `antenna / transmitter / receiver / signal_processor / display / power_supply / cooling` |
| `parent_radar` | str |  | radar id |

### 2.3 Component（新增，更细粒度）

LRU 级别。从 `hasLRUCount` 的引申。

| 属性 | 类型 |
|---|---|
| `id` | str |
| `name` | str |
| `parent_subsystem` | str |
| `weight_kg` | float |
| `replaceable` | bool |

### 2.4 Manufacturer

| 属性 | 类型 | 来源 | 示例 |
|---|---|---|---|
| `id` | str | 规范化 | `raytheon` |
| `name_en` | str | 主形 | `Raytheon` |
| `name_zh` | str | 别名表 | `雷神公司` |
| `aliases` | [str] | 别名表 | `["RTX", "Raytheon Technologies"]` |
| `country` | str | 关系折叠 | `美国` |
| `founded_year` | int | Wikidata | `1922` |
| `headquarters_city` | str | Wikidata | `Waltham, MA` |
| `parent_company` | str | 关系 | (空 or RTX) |
| `subsidiaries` | [str] | 关系反向 | `["Hughes Aircraft Company"]` |
| `description` | str |  | |

### 2.5 Country

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `usa` |
| `name_zh` | str | `美国` |
| `name_en` | str | `United States` |
| `iso2` | str | `US` |
| `iso3` | str | `USA` |
| `region` | str | `North America` |
| `nato_member` | bool | `true` |
| `aliases` | [str] | `["USA", "U.S.", ..., "苏联", "Russian Federation"]` |

> 当前 entity_aliases.json 的 `countries` 块直接迁移过来。

### 2.6 Platform（含子类）

```
Platform (abstract)
├── NavalVessel       displacement_t, length_m, propulsion, country, class
├── AircraftPlatform  aircraft_role (fighter/bomber/AEW/...), max_speed_mach
└── GroundPlatform    mobility (mobile/fixed), vehicle_chassis
```

### 2.7 FrequencyBand

| 属性 | 类型 | 示例 |
|---|---|---|
| `code` | str | `X` |
| `freq_low_GHz` | float | `8` |
| `freq_high_GHz` | float | `12` |
| `wavelength_cm_low` | float | `2.5` |
| `wavelength_cm_high` | float | `3.75` |
| `nato_designation` | str | `X` |
| `ieee_designation` | str | `X` |
| `name_zh` | str | `X 波段` |

> 当前 frequency_bands 块的 `valid` / `normalize` / `noise_to_drop` 转化为属性 + 一份噪声黑名单。

### 2.8 TechType

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `aesa` |
| `name_zh` | str | `有源相控阵` |
| `name_en` | str | `Active Electronically Scanned Array` |
| `abbreviation` | str | `AESA` |
| `category` | enum | `array_design / waveform / processing / multifunction` |
| `evolved_from` | str | 母体技术 id |

### 2.9 RadarMode

| 属性 | 类型 |
|---|---|
| `name_zh / name_en / abbreviation` | str |
| `mode_category` | enum (`search / track / mapping / nav / ECCM / multi-target`) |
| `serves_function` | str | 主要服务的 Function id |

### 2.10 Function

| 属性 | 类型 |
|---|---|
| `name_zh / name_en` | str |
| `category` | enum (`detection / tracking / navigation / EW / fire_control / surveillance`) |

### 2.11 Weapon

| 属性 | 类型 |
|---|---|
| `model` | str |
| `type` | enum (`AAM / SAM / ASM / ARM / ECM_pod / ...`) |
| `country` | str |
| `manufacturer` | str |
| `range_km` | float |

### 2.12 Standard（新增——把现有黑名单的标准号转为实体）

之前 `NON_RADAR_ENTITIES` 黑名单里的 ARINC-429 / MIL-STD-1553 / PESA 等其实是有用的标准实体，重新加回来作为 `Standard` 类型。

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `mil_std_1553` |
| `code` | str | `MIL-STD-1553` |
| `category` | enum | `data_bus / interface / qualification / cooling / ...` |
| `issued_by` | str | `US DoD` |
| `description` | str | |

### 2.13 Conflict（新增）

历史冲突 / 战争，用于建立 "雷达 X 在 Y 战中使用" 的关系。

| 属性 | 类型 | 示例 |
|---|---|---|
| `id` | str | `gulf_war_1991` |
| `name_zh / name_en` | str | `海湾战争` |
| `start_year / end_year` | int | |
| `parties` | [str] | country ids |

### 2.14 Location（新增）

固定站点、覆盖区域、城市等。

| 属性 | 类型 |
|---|---|
| `name` | str |
| `country` | str |
| `lat / lon` | float |
| `location_type` | enum (`station / region / city / theater`) |

---

## 3. 关系类型目录

属性化之后，关系**只保留实体间的边**（Radar→X 的 X 是另一个实体）。共 28 种，分 5 组。

### 3.1 既有保留（14 种）

| 关系 | 域 → 范围 | 当前规模 | 说明 |
|---|---|---|---|
| `developedBy` | Radar → Manufacturer | 386 | 研制方 |
| `operatedBy` | Radar → Country | 625 | 使用国 |
| `exportedTo` | Radar → Country | 114 | 出口去向 |
| `countryOfOrigin` | Radar → Country | 283 | 原产国 |
| `affiliatedTo` | Manufacturer → Country | 78 | 厂商所属国 |
| `deployedOn` | Radar → Platform | 462 | 部署平台 |
| `upgradeOf` | Radar → Radar | 117 | 升级自 |
| `derivedFrom` | Radar → Radar | 120 | 衍生自 |
| `similarTo` | Radar → Radar | 1 | 类似 |
| `compatibleWith` | Radar → Weapon | 43 | 武器兼容 |
| `hasFrequencyBand` | Radar → FrequencyBand | 177 | 工作频段（可多）|
| `hasTechType` | Radar → TechType | 179 | 技术体制（可多）|
| `hasMode` | Radar → RadarMode | 1737 | 工作模式（可多）|
| `hasFunction` | Radar → Function | 26 | 功能（可多）|
| **小计** | | **4348** | |

### 3.2 新增——内部组成（3 种）

| 关系 | 域 → 范围 | 估算来源 | 估算条数 |
|---|---|---|---|
| `hasSubsystem` | Radar → Subsystem | 抽取 + Wikidata + 推断 | ~600 |
| `hasComponent` | Subsystem → Component | 同上 | ~400 |
| `hasInterface` | Radar/Subsystem → Standard | 抽取黑名单恢复 + 手册 | ~200 |
| **小计** | | | **~1200** |

### 3.3 新增——厂商演化（4 种）

| 关系 | 域 → 范围 | 估算来源 | 估算条数 |
|---|---|---|---|
| `subsidiaryOf` | Manufacturer → Manufacturer | 现有 COMPANY_ALIASES + Wikidata | ~50 |
| `mergedWith` | Manufacturer → Manufacturer | 现有规范化字典 + Wikidata | ~20 |
| `acquiredBy` | Manufacturer → Manufacturer | Wikidata | ~30 |
| `headquarteredIn` | Manufacturer → Location | Wikidata | ~80 |
| **小计** | | | **~180** |

### 3.4 新增——结构推断（5 种，可由现有 KG 自动推断）

| 关系 | 域 → 范围 | 推断规则 | 估算条数 |
|---|---|---|---|
| `coDeployedWith` | Radar ↔ Radar | 同 platform → 互相 coDeployedWith | **~3000** |
| `competitorOf` | Radar ↔ Radar | 同 freq + 同 platform-class + 不同 country | ~400 |
| `coOperatedBy` | Country ↔ Country | 共用同一雷达型号 | ~150 |
| `replacedBy` | Radar → Radar | upgradeOf 反向 + 时间证据 | ~120 |
| `precededBy` | Radar → Radar | upgradeOf / derivedFrom 反向 | ~250 |
| **小计** | | | **~3920** |

> `kg_enrichment.py` 现有 `CompetitorInferenceModule` 和 `CoDeploymentInferenceModule` 已经实现，扩展到 v2 schema 即可。

### 3.5 新增——使用与历史（3 种）

| 关系 | 域 → 范围 | 估算来源 | 估算条数 |
|---|---|---|---|
| `usedIn` | Radar → Conflict | 手册 + 维基 | ~150 |
| `deployedAt` | Radar → Location | 抽取（固定站尤其相关）| ~80 |
| `coversTheater` | Radar → Location | 抽取 | ~50 |
| **小计** | | | **~280** |

### 3.6 新增——技术演化（2 种）

| 关系 | 域 → 范围 | 估算来源 | 估算条数 |
|---|---|---|---|
| `evolvedFrom` | TechType → TechType | 手工拓扑 + LLM | ~30 |
| `serves` | RadarMode → Function | 手工映射 | ~80 |
| **小计** | | | **~110** |

### 总计

```
既有保留        4,348
内部组成      +~1,200
厂商演化      +~180
结构推断      +~3,920
使用与历史    +~280
技术演化      +~110
─────────────────────
v2 总边数 ≈ 10,038         ← ✓ 达到 1w+ 目标
```

加上属性化的 ~3000 个 entity attribute，总图谱"声明数"约 **13,000**。

---

## 4. 抽取扩展方案

按"成本从低到高"排序，分 4 阶段：

### 阶段 A — 自动重构（0 成本，1-2 小时）

1. **literal-tail 三元组转属性**：14 种 hasX literal 的关系全部下沉到 Radar 实体上的属性
2. **结构推断的 5 种新关系**：扩展 `kg_enrichment.py`，从现有 KG 派生 coDeployedWith / competitorOf / coOperatedBy / replacedBy / precededBy
3. **数值规范化**：把 "20km(RCS1m²)" 拆成 `range_km=20.0` + `range_description="RCS1m²夏船，海态3"`，这步用正则 + 单位库
4. **OCR 残片清理**：剔除 11 个自循环 + "the same company" 等 garbage

预期产出：~5500 边 + ~2000 属性

### 阶段 B — Wikidata 富化（低成本，1 天）

依赖现有 `import_wikidata.py`，扩展查询：

5. **厂商演化**：subsidiaryOf / mergedWith / acquiredBy / headquarteredIn / founded_year
6. **武器属性**：weapon.country / weapon.manufacturer / weapon.range_km
7. **标准实体**：MIL-STD-XXXX / ARINC-XXX 的元信息

预期产出：~280 边 + ~600 属性

### 阶段 C — LLM 抽取扩展（中成本，2-3 天 + ~$5 LLM 费用）

7. **Subsystem / Component**：从手册（《机载雷达手册》）按章节抽取 RX/TX/天线/信号处理器/显控等，挂到对应雷达上；用 DeepSeek + few-shot prompt
8. **使用历史**：抽取 `usedIn` / `deployedAt` / `coversTheater`（手册的"作战使用"章节）
9. **技术演化**：让 LLM 列出 TechType 之间的母子关系（30 条手工审核）
10. **Mode-Function 映射**：80 条手工 + LLM 协作的"哪些模式服务哪些功能"

预期产出：~1500 边 + 修正 ~500 属性

### 阶段 D — 拓展数据源（高成本，可选）

11. 抓取 Jane's / GlobalSecurity 更多型号详情
12. 引入军用卫星雷达 / 太空雷达扩展类型
13. 加入 ESM / RWR 等 EW 关联实体

预期产出：再 +1000-2000 边

---

## 5. 迁移路线图

```
当前 KG (5610 三元组, 扁平)
    │
    ▼
[A1] generator 重构: 关系→属性下沉
    输出: data/v2/radarkg_v2_attrs.json + data/v2/radarkg_v2_edges.json
    │
    ▼
[A2] 结构推断: kg_enrichment_v2.py
    输入: edges.json
    输出: edges.json (扩展为 ~8000)
    │
    ▼
[B] wikidata 富化（manufactor / weapon / standard）
    输出: edges.json (~8300) + attrs.json (~2600)
    │
    ▼
[C] LLM 抽取: subsystem / component / usedIn / mode-function
    输出: edges.json (~10000) + attrs.json (~3000)
    │
    ▼
[D] 可选: 多源补充（不必为论文 v1 做）
```

每阶段产出**独立的 JSON 文件 + 评测报告**（沿袭现有 extraction_results 模式），可以增量审核。

---

## 6. 在 v2 schema 上 GraphRAG 的影响

属性化后，以前的 lookup 会更高效：

- **single_hop 属性查询**："X 的 frequency 是？" 直接读 `radar.frequency_band`，不需要图扩展
- **attr_filter** 仍走 constrained_join，但谓词可以用 `radar.country_of_origin == 美国 AND "X" in radar.frequency_bands`
- **agg_count / agg_enum** 走属性聚合，O(N) 扫描而不是 O(E) 索引遍历

预期 strategy pipeline accuracy：v1 是 94.0% → v2 应能 **≥96%**（属性查询不再依赖检索召回）。

但同时，**当前论文实验在 v1 KG 上跑出的 +33pp 数字依然有效**——v2 KG 是工业落地优化，不是论文版本前提。

---

## 7. 输出 deliverable 清单

迁移完成后会有这些新文件：

```
data/v2/
├── radarkg_v2_entities.json    # 实体 + 属性
├── radarkg_v2_edges.json       # 边
├── radarkg_v2_schema.json      # 机器可读 schema
└── migration_report.md         # 迁移过程的统计 + 问题列表

scripts/
├── migrate_v1_to_v2.py         # 阶段 A
├── kg_enrichment_v2.py         # 阶段 A2 + B 推断
├── extract_subsystems.py       # 阶段 C
└── validate_v2_schema.py       # 模式验证 + 一致性检查

KG_SCHEMA.md                    # 本文档
```

---

## 8. 建议的工作顺序

如果**目标是先发论文再迭代**：先发当前 v1 paper，KG 重设计列入 future work。
如果**目标是 v2 KG 上重做实验**：执行顺序见 `RERUN_PLAN.md`，预计 ~1 周工作 + ~$10 LLM 费用。
