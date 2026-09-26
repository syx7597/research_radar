# 雷达事实核验与独立问答标注规范 v0

本规范连接论文的数据构建与问答评价。原始资料和旧图谱不修改；新核验记录与待复核样例单独保存。模板位于[templates/radar_review](../../templates/radar_review)。当前没有独立人工金标。

## 1. 本轮来源追踪样例

`kg_v3/entities.json`的第0个实体为AN/MPQ-65，第2个属性为`service_entry`，原值含1981和1984两个事件，规范值仅保留1981。其`doc_id=an_mpq-65`可定位`pipeline/v3/work/chunks.jsonl`的`an_mpq-65#ibx`和`radar_corpus/raw/an_mpq-65.json`。

原始记录的`source_en`指向MIM-104 Patriot系统页面，不能据此确认1981就是该雷达型号的服役年份。该例至少存在实体归属与事件区分待核查的问题。此次只追踪本地来源链，未独立核实真实服役日期，不发布一个替代年份。结构化记录见[radar_source_review.json](../../artifacts/thesis_direction_review/radar_source_review.json)。

## 2. 事实表

使用[facts.csv](../../templates/radar_review/facts.csv)。一行是一条有明确主体、属性及来源条件的事实；不同型号版本、不同事件、不同条件分行。

| 字段组 | 含义与要求 |
|---|---|
| fact_id、entity_id、entity_name、variant、attribute、event_type | 保留稳定ID；明确雷达/整套系统/平台的主体区别，年份必须区分设计、服役、初始能力等事件 |
| value_raw、value_kind、value_status、value、min_value、max_value、unit_raw、unit_std | 原文不可覆盖；区间保留双端点。kind为`text/scalar/range/lower_bound/upper_bound/approximate/date/event/unknown`；status为`stated/not_recorded/not_applicable/conflicting/ambiguous`，不得用0填未知。单位换算只有语义一致时进行 |
| condition_status、condition_raw、conditions_json | `explicit/not_stated/ambiguous`；原文缺条件不能自行补充，空值不能解释为所有条件成立 |
| doc_id、source_uri、source_file、source_sha256、locator、evidence_text | 分开记录网页URL与本地快照；哈希对应source_file，locator明确页码、chunk或JSON pointer，不能只给无法回溯的片段 |
| review_status、reviewer、review_notes | `unreviewed/needs_review/accepted/rejected`；AI初筛标`AI`，不得假充人工；accepted须经人工核验并指定核验人和理由 |

不能直接将现有实体中所有属性当作accepted。来源可追踪不等于事实正确；信息框可能属于系统级对象或混合多个事件。冲突记录保留并说明，不能默默挑选有利数值。

## 3. 独立问答表

使用[questions.csv](../../templates/radar_review/questions.csv)。由核验过的原文/事实独立设计问题，记录`support_fact_ids`、冻结的`knowledge_version`、答案依据、题型、答案状态、来源/型号家族分组及标注和复核人员。

- 训练/开发可使用旧自动生成问答；独立评价题不能从旧抽取图自动得出答案后宣称人工金标。
- `answer_status`使用`answerable/unknown/conflicting`。未记录参数不等于不满足，不将局部资料计数写成现实世界全部数量。
- 先确定划分组，避免同一来源或型号近似题跨开发与评价；冻结后不据效果替换问题。
- 语义问题和程序实现分开核验：程序能回放答案不能替代题目是否忠实于原文。
- 雷达方法评测先在核验知识图上进行，再测原始抽取图的端到端效果；两项分开报告。

## 4. 首批执行范围

先按型号/系统归属、数值单位/区间、关系和文本解释四类准备小型核验包，再根据证据质量扩展至20–30型号和约100–150独立问题。该规模是应用验证起点，不是确定的最终数量或泛化保证。本轮只提供模板和一个待复核样例，未把AI检查标成完成的人工作业。
