雷达独立评价空模板（设计版，未冻结）

本目录没有任何真实问题、答案或验收行。详细方案见：
artifacts/thesis_direction_review/radar_evaluation_plan_v1.json

现在可以做什么
1. 用户本人先用已有 12 条事实包做首轮核验，暂时没有第二个审阅者也可以推进事实与开发接口。
2. 将本目录模板复制到 data/radar_evaluation_private/ 后填写；该目录已被现有 /data/ 规则忽略。
3. 未来独立 QA 需另一名真人交叉复核。用户说将找同学；目前不代填身份、不记录已完成。
4. 本目录在公开仓库中始终只保留空模板。原文、最终题目/答案和逐题输出留在私有工作目录。

填写顺序
source_inventory.csv → family_assignments.csv → fact_acceptance.csv → split_assignments.csv
→ questions.csv → question_reviews.csv → freeze_manifest.json
预测运行后再填 prediction_reviews.csv 和 answer_claim_reviews.csv。

字段通则
- CSV 是 UTF-8，复杂列表/对象用合法 JSON，空列表填 []，待填字段留空。留空不是 unknown，也不是通过。
- *_sha256 必须是文件内容的 SHA-256；question_record_sha256/fact_record_sha256 是整行所有字段的规范 JSON 编码哈希。
  规范编码：原 CSV 值均保留字符串，键按 Unicode 字典序排序，ensure_ascii=false，分隔符为逗号和冒号且不加空格，UTF-8，无尾换行。
  已有事实审阅记录如采用相同编码直接复用；若采用另一明确编码，记录其版本，不把不同编码哈希混用。
- *_kind: human 或 AI；实际验收身份必须 human，签名由本人填写。*_id 是真实、稳定的署名/匿名化参与者编号。
  私有身份对照可与公开匿名编号分离；不能把模型昵称声明为真人。软件无法代替身份真实性核验。
- *_at 为带时区 ISO 8601 时间。当前没有审阅发生，不预填任何时间。
- 布尔字段填写 true/false；评分不适用时用 not_applicable。0 分母在聚合指标中为 null，不算 0% 或 100%。
- 全部列表按稳定顺序保存；支持来源/事实必须完整，不只写第一个来源、主实体或 anchor。

来源、家族和事实
source_inventory:
 source_id 是快照标识；canonical_document_id 是原始文档标识。镜像、翻译、不同快照、引用同一原文的转述通过
 derivative_of_source_ids_json 记录派生关系。local_snapshot_file 保留完整内容在本地，snapshot_sha256 绑定字节；
 source_class 按原核验流程分类（primary / secondary_web_snapshot / other），primary_status 为
 pending / verified_primary / not_primary。出版日期未知留空并在 review_notes 说明，不猜测。
 source_scope 写明文档所覆盖主体与边界。整本手册不能靠分页面变成不同独立来源。
family_assignments:
 先处理别名、型号版本、系统/部件关系，family_definition 记录为何同组。assignment_status 为 pending / reviewed。
 并非任意实体都有型号家族；公共单位和纯 schema 常量若共享，必须在切分审计中明确列白名单，不携带装备事实。
fact_acceptance:
 仅索引已有的完整事实版本及其验收记录，不代替原事实表与核验表。acceptance_status 为 pending / accepted /
 needs_changes / rejected；acceptance_scope 为 independent_fact 或 source_snapshot_reading。
 独立知识事实准入需要 independent_fact + accepted + human，且原审阅记录的主体、版本、事件、值、单位、条件和
 一手来源检查完整通过。首批用户本人可以完成这一核验；QA 的异人复核是另一个门禁。
 equivalent_fact_group_id 将同一陈述的转引/重复事实归并；不是将不同条件/版本的数据混成一个数值。
 source_snapshot_reading 只说明某二手快照写了什么，不能充当已核验装备参数。
 对冲突事实，应核验“某来源在某条件下作了某陈述”，保留冲突与归属；不要将相斥参数都写成无条件真实值。
split_assignments:
 split: adaptation_train / development_only / final_evaluation。
 一个来源/家族传递连通组只能有一个 split。forced_development_reasons_json 记录已见开发题及依赖闭包等强制理由。
 组划分限制适配问题、答案和提示开发。所有核验事实仍在共用 KB/文本库可查询，包含最终题所需事实。

独立问题及答案（本模板所有行为空）
question_group: fact_subject / relation_set / condition_numeric / insufficient_conflict。
answer_status: answerable / unknown / conflicting；未标注时留空，不把未知标签当占位符。
annotation_origin: independent_human_from_sources / development_human / development_ai_assisted。
annotation_status: draft / ready_for_review / needs_changes / accepted / rejected。
最终独立题必须 independent_human_from_sources，model_outputs_seen=false，author_kind=human。
出题依据是核验原文，不看旧自动题答案、模型输出或执行答案；先人工定语义，再用执行回放检查。
questions.answer_json 的约定（仅结构，不含答案实例）：
 {status, items, conflict_alternatives, reason, scope}
 items 中每项包含 item_id、value_raw、value_kind、value、min_value、max_value、unit、entity_ids、conditions、support_fact_ids；
 按题型用必要字段，无关字段为 null；值类型/单位沿用事实模式。unknown 的 items 为 []，reason 和 scope 必填；
 conflicting 记录带各自事实/来源依赖的 conflict_alternatives，不静默选值。
 required_answer_items_json 是评分所需最小答案项的列表（item_id、语义要求、support_fact_ids），不是关键词抽样。
 required_conditions_json 记录题目要求/事实成立必须保留的主体版本、事件、部件、条件；无要求填 []。
 expected_numeric_slots_json 列出所需数值项的 item_id、quantity_kind、允许单位、区间形式和已声明容差引用；
 不含所需数值时填 []，实际正确值放 answer_json。不能默认百分比误差容许范围或将区间变成中点。
 support_fact_ids_json 与 source_ids_json 记全体实际支持依赖；scope_source_ids_json 记为确认范围/未知状态而审阅的来源，
 可以大于直接支持集合。unknown 即使没有支持事实，也不能省略资料范围。
 full_entity_ids_json 包括问题实体、关系目标、集合候选和条件中的实体；semantic_request_signature 表示正规化实体、
 属性/关系、投影、条件与资料范围。paraphrase_group_id 将等价信息请求归组，不将普通知识库操作语法当成题目泄漏。
 当前 12 开发题、旧 5401 自动 QA 和它们的改写/翻译/重编号永不升级 final_evaluation。

question_reviews:
 所有 *_check 字段为 pass / fail / not_applicable；独立作者、答案、事实支持、范围、泄漏检查不能写 not_applicable。
 decision: pending / accepted / needs_changes / rejected。accepted 需实际 human reviewer 且不同于 author_id，检查无 fail，
 每项 not_applicable 有理由；未解决分歧不得 accepted。条件/单位不适用须有依据。
 修改 questions 整行后必须重新计算 hash 并复核，不能复用旧决定。adjudicator_* 仅在有争议裁决时填，不能凭空署名。

逐题输出与断言评分
prediction_reviews 用相同冻结金标复核各 run 的完整输出。predicted_answer_status:
 answerable / unknown / conflicting / execution_failure。failure_type:
 none / timeout / empty_output / parse_failure / execution_failure。
 answer_correct 包括必要状态、答案项与条件，不以程序能运行或最终字符串碰巧相同代替。
 substantive_answer_given 仅在状态 answerable 且有具体答案时为 true；明确资料不足/冲突可以正确但不计确定答案覆盖。
 condition_faithful 仅对 gold=answerable 且 required_conditions 非空的题评分，其他为 not_applicable。
 all_required_items_supported 仅对 gold=answerable 评分；漏答/拒答/失败为 false。
answer_claim_reviews 将一条输出拆成可核验的实质断言，固定同一拆分规则。required_answer_item_ids_json 指其覆盖哪些金标项；
 新增的无关/多余断言仍计实质断言，不能为提高分数忽略。is_numeric 仅指参数数值，来源页码或引用编号不算。
 numeric_correct 包含数值、单位、上下界、主体/条件匹配；非数值填 not_applicable。
 citation_ids_json 列出该断言输出的全部引用 ID；valid_citation_ids_json 必须是其中能够精确定位的子集。
 同一引用可支持多个断言；定位率按 run_id/qid/citation_id 去重，断言支持率按 claim_id 计数。
 source_supports_subject 包含必要主体/版本/事件；source_supports_value_unit 包含原值、单位、范围；
 source_supports_conditions 包含必要条件。不适用可写 not_applicable，但 claim_supported 必须按适用项全部满足才 true。
 claim_supported 同时需要真实支持和可定位来源；能打开链接不算支持。计数/集合的支持可包括可回放操作与全部事实依赖。
 在 gold=answerable 的必要答案项支持率中，缺失输出贡献 0；在输出断言支持率中，未引用断言仍在分母。

冻结前必须仍完成的工作
- 事实、家族和来源审阅；完整组件划分与旧自动 QA 依赖过滤。
- 独立人工出题及异人复核；实际题数/组件数/各评分分母登记。
- 确定同源 RAG/P/两种子的 A/C 的模型、适配状态、推理/检索预算和所有哈希。
- 保存单独最终答案，模型推理和检索只能见问题/知识，不能见金标或参考查询。
- 所有 freeze_manifest 核心绑定填齐后才能产生正式冻结版本；本目录 manifest 的 null 表示未完成。
