# 首批雷达事实审阅与独立 QA 准入

更新：2026-10-05。本文件补充[原标注规范](RADAR_DATA_ANNOTATION.md)。原规范和生成脚本是首批包的哈希输入，保持不变；只增加审阅层，不改旧图谱、12 条候选事实或来源快照。

## 当前采用AI交叉审阅推进开发

用户最新说明没有相关专业知识，要求另开子agent代审。已由未继承会话的新子agent逐条读取绑定来源，再由主agent复核，完成[12条AI审阅结果](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/index.html)、[原始子agent报告](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/reviewer_report.json)和[开发读法记录](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/development_readings.json)。主agent有原项目上下文，并接收了新来源线索；候选自带旧备注也可被子agent看到，因此不称双盲、不同模型或统计独立性验证。

子agent建议开发保留4条、隔离5条、修订3条。主agent按开发用途裁决为5条修订/隔离型号断言、5条保留限定来源读法、2条保留读法并补查精度/选项范围。09–10的隔离针对型号真值提升，不禁止忠实记录快照；12采纳结构化unknown_scope建议。两套判断逐条保留，不用简单多数票掩盖不同范围。

本轮不再要求用户先判断专业参数才能继续。AI审阅记录统一标为development_only，可用于候选修订、限定来源问答及开发回放；人工验收计数仍为0，不改变原accepted门禁。若以后采用AI参考答案作应用探索评价，须明确标注这一等级，不与独立人工评价混称。

此前的[人工审阅页](../../artifacts/thesis_direction_review/radar_review_ui_v1/index.html)仍可选用：对照原始快照填写真实署名及理由后导出CSV草稿。页面没有自动上传或仓库写入；没有真人实际完成时保持待审，不由AI代填姓名。

页面的“确认”只对应 `source_snapshot_reading`，确认这份资料的读法；不自动确认现实参数或独立事实。含主体/事件歧义的候选可标需要修订，一手来源和新事实版本按下述门禁继续处理。未填写或不确定时不代填验收结果。当前页面默认12条待审，原决定文件未改变。

已有01–03的[机构来源AI预审](../../artifacts/thesis_direction_review/radar_primary_screening_001_003/ai_screening.json)已附到页面。新增[本地抓取记录](../../artifacts/thesis_direction_review/radar_primary_screening_001_003/capture_manifest_v1.json)：DSCA文件全文与短引文绑定成功；Redstone 1981连接失败，1984响应未含原引文，不能把它们当已绑定的一手证据。全文只保存在本地 `data/`，不提交公开仓库，且未据此填写替代事实或问答答案。

[领域评价设计版](../../artifacts/thesis_direction_review/radar_evaluation_plan_v1.json)及[空模板说明](../../templates/radar_evaluation/README.txt)记录下一阶段的独立QA、来源/家族分组、同源对照和评分规则。目前尚未冻结知识或最终评价集。

只读[查询接口](../../experiments/radar_domain/adapter.py)会先验证全包及决定。`source_preview` 返回保留条件/区间/引用的候选读法，`accepted-independent` 默认只返回完整通过独立事实验收的记录。当前10条开发查询的双模式回放通过，后者全部无可用事实；这不是自然语言模型评测。

## 当前交付与边界

原[12 条事实包](../../artifacts/thesis_direction_review/radar_review_batch/facts.csv)由 9 条旧属性拆分，涉及 6 个实体标签和 6 份 Wikipedia 本地快照。12 条均为 AI 预填的 `needs_review`。每条已有原始属性、网页 URL、文件 SHA-256、JSON pointer、Unicode 字符区间和 chunk 定位。原包构建阶段仅离线复核这些绑定，未判定真实参数；后续补充来源预审及抓取情况见上文，未改变原候选的验收状态。

这 12 条是有目的选择的流程问题样例，不能估计全图错误比例。其现有来源全是二手网页快照；来源能定位、数字在原文出现，都不足以说明主体、配置或实际参数正确。

新增[流程目录](../../artifacts/thesis_direction_review/radar_review_workflow)含：

| 文件 | 用途 | 当前状态 |
|---|---|---|
| `review_decisions.csv` | 12 条事实的独立审阅表，绑定整行事实哈希和来源哈希 | 全部 `pending`；真人、时间、理由及检查栏均留空 |
| `development_questions.csv` | 12 道用于检验标注流程的题目草稿 | 全部 `development_only`，答案与答案状态留空，AI 草拟、未经复核 |
| `workflow_manifest.json` | 固定原包版本、二手来源分类和流程限制 | 不是人工验收或新的知识库版本 |
| `audit_report.json` | 机器检查结果 | 分别统计快照读法验收、独立事实准入和开发题数量 |

开发题仅检查主体、事件、区间、部件标签、情境、量纲和来源不足的表述。它们没有经过独立人工出题，不能计入未来独立评价；后续即使填写答案，也必须另存开发版本，不能修改 split 后改称测试题。

## 首批人工审阅顺序

| 事实编号尾号 | 首先核对什么 | 不可直接接受的解释 |
|---|---|---|
| 01–03 | 来源主体是 Patriot 系统还是 AN/MPQ-65 雷达；两个年份分别对应什么事件 | 直接把系统类型、服役或初始能力年份赋给某雷达型号 |
| 04 | 频率区间是型号实际参数还是一般频段说明；型号版本是否明确 | 因 Frequency 栏出现区间就断言整段都是该型号工作范围 |
| 05–06 | Illuminator 与 Tracking Radar 标签的部件归属、配置和两段范围 | 合并部件范围，或默认为所有版本通用 |
| 07–08 | surface/air 的原文含义、数值形式和缺失条件 | 用标签补造目标、测试场景或适用条件 |
| 09–10 | 频率与脉宽的量纲、单位和区间；回查一手出处 | 沿用旧图谱的错误量纲，或把区间压成一个值 |
| 11 | Canceled 对应取消事件，是否存在其他独立服役证据 | 把取消年份改名为服役年份 |
| 12 | 本快照是否提供数值；缺失结论的范围 | 将未记录写成 0，或推断现实参数不存在 |

人工先确认主体，再核对版本、事件、值、单位和条件；原文未说明时明确记录 `not_specified_in_source`，不能用空字段表示已经确认。能取得出版手册、厂商原始规格或机构原始文件时，在本地保留引用文件、稳定 URI/书目信息、哈希和具体位置；不要因网页转引某文件就把转引页本身标成一手来源。

## 审阅表与机器门禁

`fact_record_sha256` 对候选事实整行的规范 JSON 编码计算；任一字段改变都会使旧决定失配。`source_sha256` 同时绑定原快照。审阅决定不直接修改原候选。

- `pending`：尚未审阅，不填写真人身份或完成时间。
- `needs_changes/rejected`：记录 `reviewer_kind=human/AI`、真实署名或 AI 标记、带时区的 ISO 时间及理由。AI 可以记录待修改意见。
- `accepted`：必须声明真人核验人、时间和理由，并逐项填写 `subject_check/variant_check/event_check/value_check/unit_check/condition_check`。允许值为 `confirmed/not_applicable/not_specified_in_source`。软件拒绝 AI 身份及已知 AI 名称标记冒充 accepted。

验收分两个范围：

1. `source_snapshot_reading`：只表示人工确认了“这份二手快照写了什么”。它可以保留原文歧义，**不计入独立事实准入**。
2. `independent_fact`：除上述字段外，需要另一个被人工认定的一手来源、短证据、来源文件哈希与精确定位；主体必须确认。当前 `ambiguous/conflicting` 候选需先形成新版本、处理解释或冲突，不能用一行 accepted 覆盖不确定性。

一手来源字段为 `independent_source_class/uri/file/sha256/locator` 与 `independent_evidence_text`。本包已知二手快照不能仅改标签就通过一手来源门禁。证据最多 180 字符；文件正文留在本地，公开只保存必要短摘录与定位。

定位支持 UTF-8 文本和 JSON pointer，必须填写 `char_start/char_end/offset_unit=unicode_codepoint/end_exclusive=true`。PDF 来源先保存本地可定位文本片段，locator 另填 `original_file/original_sha256/page`（PDF 页码从 1 开始），绑定原 PDF 和对应页；扫描图与转录是否一致仍由人工核对。当前验证器不解析 PDF 版面，也不判断哪个机构必然可靠。

**软件检查的是声明是否完整、版本是否匹配、引文是否精确，不能证明署名者实际完成了审阅、来源确属一手或事实真实。** 本轮不会填写真人验收字段，也没有产出可导入最终测试集的金标。

## 独立 QA 的入组与冻结协议

1. 先按原始来源与型号家族分组，再划分开发与评价。涉及同一来源或家族的连通记录放在同组；两个标签不相同不代表已经独立。当前 6 组只是流程标签，未经覆盖审计，不能宣称完成独立划分。
2. 先完成事实审阅并冻结知识版本，再由标注者基于核验原文设计问题和答案。标注者不得先看旧图答案、生成器答案或执行结果来定金标；另一个人复核题意、范围、支持事实及答案。机器执行回放仅在语义标注之后用于发现冲突。
3. 事实表与问答表分开验收。QA 记录 `support_fact_ids`、知识版本、来源组、型号家族、出题人和复核人；每项答案都应追溯至已验收事实。`unknown` 只能针对已冻结资料范围，`conflicting` 要有明确冲突依据，不能把未标注状态当作 unknown。
4. 模型可查询评价涉及的知识与原文；隔离的是适配训练问题、答案和近似问法。旧自动 QA 在复用前须检查完整实体、支持事实与来源，而不是只检查 anchor。不能确定来源归属的自动题不进入需要严格来源隔离的训练。
5. 先固定题型范围、来源/家族划分、标注表版本、评分方式、unknown/conflicting 处理及冻结哈希，再开展独立评价；最终答案由独立评分流程保管。开发样例和当前 12 道流程题永久排除在独立评价之外。本轮不生成、不读取最终测试答案，也不读取公开基准留出集。
6. 按全部问题、可回答问题、错误数值、错误拒答、证据支持完整度分别报告结果。核验图与原抽取图同题比较，不能把人工纠错收益归因于模型训练；文本与图对照使用同源资料。

首批包结构检查与AI语义审阅均已完成，可以继续开发修订。若要进入独立人工事实层，仍需真实的主体和一手来源核验及版本化决定；在此之前保留为明确标识的AI开发读法，不扩大到全库重建或生成所谓独立人工金标。

## 运行

```bash
# 复验新子agent报告、主agent裁决和AI开发读法的来源绑定与确定性导出
python3 scripts/summarize_radar_ai_review.py --check-only

# 重建可填写的本地审阅页；不产生审阅决定
python3 scripts/build_radar_review_ui.py

# 验证已抓取的补充来源，默认不联网；输出区别文件绑定与引文匹配
python3 scripts/capture_radar_primary_sources.py

# 验证10条已有开发查询的确定性演示，不调用模型
python3 -m experiments.radar_domain.pilot --check-only

# 已有12条原包：只读验证，无网络和模型调用
python3 scripts/prepare_radar_review_packet.py --check-only

# 新审阅层：只读检查已有表
python3 scripts/audit_radar_review_workflow.py

# 首次创建（已有审阅文件时拒绝覆盖）
python3 scripts/audit_radar_review_workflow.py --initialize

# 显式写出机器检查报告
python3 scripts/audit_radar_review_workflow.py \
  --report artifacts/thesis_direction_review/radar_review_workflow/audit_report.json

# 仅用合成事实检查验收门禁，不触及公开基准
python3 -m unittest discover -s tests -p test_radar_review_workflow.py
```
