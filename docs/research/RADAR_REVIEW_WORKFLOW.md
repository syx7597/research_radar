# 首批雷达事实审阅与独立 QA 准入

更新：2026-10-07。本文件补充[原标注规范](RADAR_DATA_ANNOTATION.md)。原规范和生成脚本是首批包的哈希输入，保持不变；后续审阅及新来源开发包均单独版本化，不改旧图谱、12条候选事实或来源快照。

## 来源准备已执行：进入统一读法整理

2026-10-07的[归档汇总](../../artifacts/thesis_direction_review/radar_sources_v2/archive_summary.json)及[来源准备复核](../../artifacts/thesis_direction_review/radar_sources_v2/source_preparation_review.json)已完成，8组有主件，Simrad失败与MRR替补分别保留。新原件与旧语料型号重叠分开登记，历史排查不是当前权重污染认证。

接下来严格使用[预先声明的标注范围](../../artifacts/thesis_direction_review/radar_sources_v2/annotation_scope.json)：先完整整理13页PDF和13段HTML内符合统一规则的主型号/部件读法，另记遗漏与歧义，再独立编题。源版本、父级、表头和脚注随记录保存；不以规范化或程序输出默补缺单位、消除真实冲突。JRC/Raymarine/METEK辅助网页不参与本轮事实输入。新题编写者不看旧答案，另由审计者核查旧QA/开发题衍生；排除规则在看模型分数前执行。当前读法/题目尚未生成，准入标志仍为false。

## 下一批材料：覆盖设计与元数据划分审计

2026-10-06已形成[覆盖设计v2](../../artifacts/thesis_direction_review/radar_coverage_v2/design.json)，目标为8个来源/家族组、96题，设计登记时新归档/新题均为0。现有官方候选只有入口筛查，尚未通过版本、家族、原字节、表格和语义准入。先完成资料身份与覆盖检查，再整理新题，不先跑模型再挑题。

[已见登记](../../artifacts/thesis_direction_review/radar_coverage_v2/exposure_registry.json)将原12目标及后续24题的来源和家族闭包一并登记；编号、翻译、问法文本指纹不是独立题数量。新增[source_split_audit](../../experiments/radar_domain/source_split_audit.py)只接收来源/问题元数据：规范文档ID、URI、哈希、家族、依赖来源、等价请求、用途、参考等级和已见状态。来源、家族、等价请求形成连通分量；跨开发/候选评价混用、候选与已见闭包相交、重复/悬空ID、未解决审阅及AI材料冒充人工金标均阻断。

该检查不读取原文、不验证实际文件哈希、不判断事实正确或实际审阅人的独立性；CLI另绑定实际输入元数据字节与代码哈希。未登记的别名、派生文档、语义改写和预训练暴露不能由无交集自动排除。[CPU回归](../../artifacts/thesis_direction_review/radar_coverage_v2/design_check.json)明确区分开发复用可接受与伪装成未见评价必须阻断，不能拿结构审计通过代替完整事实准入。

下一轮flat与bound须能还原完全相同的关系元组和重复次数，保留每个主体、条件、数值、单位、部件及引用的对应，不只比较数字集合。语义参考仍须先据原文声明，CPU往返只能证明表示保真。AI参考等级和人工事实验收边界继续采用下文规则，不改旧包或旧决定。

## 新厂商来源包：固定评价与双AI正文审阅完成

[新资料清单](../../artifacts/thesis_direction_review/radar_sources_v1/manifest.json)及[来源索引](../../artifacts/thesis_direction_review/radar_sources_v1/source_index.json)绑定Vaisala WRM200、Leonardo METEOR 735C、Furuno WR2120、JRC JMA-1030系列4份厂商PDF；JRC包含JMA-1032与JMA-1034，共4个来源/家族、5个型号、24道中文单意图题、62条来源读法和15个完整原文页。另一个AI agent依据原PDF核对主体、表格列、量值、离散选项和条件，[来源审阅通过](../../artifacts/thesis_direction_review/radar_sources_v1/independent_ai_audit.json)，24条参考程序CPU精确检查通过。随后按[固定协议](../../results/radar_domain/source_eval_v1/protocol.json)完成11个GPU任务及双AI正文审阅。

原PDF按字节与SHA-256保存，完整页文本和读法同时绑定页码、文本哈希及Unicode定位。JRC保留横向空白以保存型号列，机器抽取的间距和字体瑕疵不隐去。先根据原文声明语义参考，再执行程序检查，不由执行器结果制造答案。62条中含同一脉宽/PRF配置的正反向字段表示，不当作62份独立事实证据。全局KB保留所有型号和干扰读法；问题只有编号与题面，参考程序、支持事实及语义断言另存。原PDF、全文、真实QA及结构化读法放在忽略的`data/radar_sources_v1/`，公开清单、索引和审阅元数据。

本包与原12条实际调试来源及家族闭包隔离，但WRM200、METEOR 735C家族和WR2120已在历史语料出现，旧自动QA完整来源血缘为`not_certified`。这是来源隔离的AI探索评价，不称全项目或预训练未见，也不称最终独立人工金标。4个来源/家族是主要相关单元；本批没有未知或事件题，不能据此宣称对应能力通过验证。用户授权AI代审支持当前工作；原人工事实验收身份不被改写。

固定适配后的P/A1/C1/A2/C2选中记录只映射其原文页，BM25 top4 RAG检索同一15页材料。六路均由未经本项目适配的Qwen2.5-3B-Instruct根据原文生成`answer_text`、断言和引用，不把记录字段、中文审阅提示或固定模板证据卡作为回答。全部120条选择与144条回答的CPU重放、证据输入和解析检查零差异，见[机械汇总](../../results/radar_domain/source_eval_v1/mechanical_summary.json)。

| 证据入口 | 精确选择／24 | 支持页／24 | 正文正确／24（AI） | 正确且获引用支持／24 | 选择＋回答总token |
|---|---:|---:|---:|---:|---:|
| RAG | 不适用 | 23 | 13 | 9 | 88,520 |
| P | 15 | 19 | 13 | 13 | 48,368 |
| A1 | 21 | 22 | 16 | 16 | 212,651 |
| C1 | 19 | 23 | 15 | 15 | 176,433 |
| A2 | 20 | 22 | 16 | 16 | 169,768 |
| C2 | 19 | 24 | 16 | 16 | 143,625 |

[语义汇总](../../results/radar_domain/source_eval_v1/ai_semantic_summary.json)来自两个隐藏方法标签的AI agent；题目、所给证据与生成文本完全相同的输出去重后审58条，再还原到144个原始输出。两者的正文正确、引用和条件保留判定完全一致；5条唯一答案仅在限定的缺证据拒答是否有支持上分歧。主agent按其限定措辞裁决支持为真，正确性与引用未支持仍为假，[原判与分歧全部保留](../../results/radar_domain/source_eval_v1/ai_semantic_verdicts.json)。主agent已见汇总分数，不能把裁决称盲审或把双AI一致称成人工事实验证。

严格断言字段匹配均为0/24，这是规范字段诊断，不是正文准确率为0；同义表达、分解方式与字段使用可能导致不匹配。选错结构化记录也可能返回含答案的同一原文页，因此选择、证据覆盖、正文和引用分别计分。显式查询条件子集分母保持8题，按RAG/P/A1/C1/A2/C2顺序的条件保留数为4/6/6/5/7/5；运行后另报告参考中带必要限定的9题子集（含第15题双波束限定），分子相同。新诊断不静默替换原8题定义，条件保留也不等于答案正确。

准备侧曾查看参考并观察固定检索支持页23/24，`source_zh_19`未命中；[预冻结可见性记录](../../artifacts/thesis_direction_review/radar_sources_v1/pre_freeze_retrieval_observation.json)公开保留，没有因此改题、调整分块或调参，检索不能称完全盲测。[执行记录](../../results/radar_domain/source_eval_v1/execution_summary.json)确认11项任务均由`syx`完成，无新训练；北京时间22:04:05–22:06:41，实耗0.103 GPU小时，累计24.040/72，低于本轮3.25上限。总token包含记录选择和回答，Agent在本轮高于RAG，不用较短的回答上下文宣称端到端省token；知识整理成本亦应另列。

[阶段决定](../../results/radar_domain/source_eval_v1/stage_decision.json)保留公开2000题方法结论，领域部分不宣称C算法领先。C1正文比A1少1题，C2与A2持平；C2取得全部24题支持页仍答错8题。[错误分析](../../results/radar_domain/source_eval_v1/failure_diagnosis.json)将缺页与已有证据仍答错分开，量值、条件、部件绑定和描述完整性是可观察的短板，不能据此证明某种新训练机制有效。停止追加领域训练、RL及提示搜索，先完成应用章与证据表，再补数据构建章衔接。24道AI题只是最小验证，完整论文仍需补充来源/题型覆盖和评价可信度，扩展范围应按事先定义的来源多样性及条件类别规划，不依照本批分数挑选。未来若确需比较原文页、规则模板和条件绑定结构化证据，须另立新题及协议，本轮未建立或执行该实验。

## 当前采用AI交叉审阅推进开发

用户最新说明没有相关专业知识，要求另开子agent代审。已由未继承会话的新子agent逐条读取绑定来源，再由主agent复核，完成[12条AI审阅结果](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/index.html)、[原始子agent报告](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/reviewer_report.json)和[开发读法记录](../../artifacts/thesis_direction_review/radar_ai_cross_review_v1/development_readings.json)。主agent有原项目上下文，并接收了新来源线索；候选自带旧备注也可被子agent看到，因此不称双盲、不同模型或统计独立性验证。

子agent建议开发保留4条、隔离5条、修订3条。主agent按开发用途裁决为5条修订/隔离型号断言、5条保留限定来源读法、2条保留读法并补查精度/选项范围。09–10的隔离针对型号真值提升，不禁止忠实记录快照；12采纳结构化unknown_scope建议。两套判断逐条保留，不用简单多数票掩盖不同范围。

本轮不再要求用户先判断专业参数才能继续。AI审阅记录统一标为development_only，可用于候选修订、限定来源问答及开发回放；人工验收计数仍为0，不改变原accepted门禁。若以后采用AI参考答案作应用探索评价，须明确标注这一等级，不与独立人工评价混称。

此前的[人工审阅页](../../artifacts/thesis_direction_review/radar_review_ui_v1/index.html)仍可选用：对照原始快照填写真实署名及理由后导出CSV草稿。页面没有自动上传或仓库写入；没有真人实际完成时保持待审，不由AI代填姓名。

页面的“确认”只对应 `source_snapshot_reading`，确认这份资料的读法；不自动确认现实参数或独立事实。含主体/事件歧义的候选可标需要修订，一手来源和新事实版本按下述门禁继续处理。未填写或不确定时不代填验收结果。当前页面默认12条待审，原决定文件未改变。

已有01–03的[机构来源AI预审](../../artifacts/thesis_direction_review/radar_primary_screening_001_003/ai_screening.json)已附到页面。新增[本地抓取记录](../../artifacts/thesis_direction_review/radar_primary_screening_001_003/capture_manifest_v1.json)：DSCA文件全文与短引文绑定成功；Redstone 1981连接失败，1984响应未含原引文，不能把它们当已绑定的一手证据。全文只保存在本地 `data/`，不提交公开仓库，且未据此填写替代事实或问答答案。

[领域评价设计版](../../artifacts/thesis_direction_review/radar_evaluation_plan_v1.json)及[空模板说明](../../templates/radar_evaluation/README.txt)保留最终独立QA、来源/家族分组、同源对照和评分规则。最终独立人工知识与评价集仍未形成；新增厂商资料已按单独AI探索版本验收，状态见上节。

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

## 共同接口适配：合成材料准入及评价完成

两次已见雷达开发检查后，已按用户授权完成[合成接口包](../../artifacts/thesis_direction_review/radar_interface_synthetic_v1/manifest.json)及[另一个AI agent的审计](../../artifacts/thesis_direction_review/radar_interface_synthetic_v1/independent_ai_audit.json)。100个虚构查询意图先按60/20/20划分训练、开发和保留组，每组中英两题，共200题；实体别名、知识实例与题面模板跨组隔离。每个匿名知识路由返回完整的3实体、32记录知识包，包含属性、条件、事件和已知/未知干扰项，不按问题预裁成目标记录。

独立AI审计从中英题面解释实体、字段和限定词，再与参考对照；200条Agent和200条一次性程序轨迹均经CPU真实执行核对。来源文件、JSON定位、引文与全部查询字段逐项一致，错实体、错字段、错条件的查询与目标分离，未知记录与空查询不同。CPU验收验证数据和工具约定，不是模型效果，也不形成真实装备金标。

原12条雷达材料的真实名称、来源URI、文本答案和标准化数值答案从合成数据排除；`Find/QueryAttr/QueryAttrUnderCondition/finish`、字段名、单位和条件/事件标签是明示的共享接口词汇。合成数值没有物理真实性。任务类别、程序结构与公共措辞仍有意共享，因此合成保留组只检验虚构新实例和新题面，不代表真实文献来源独立或未见推理结构。

已冻结[共同适配训练协议](../../results/radar_domain/interface_adapt_v1/protocol.json)（SHA前缀`b536a79d42f9`）和[前后评价协议](../../results/radar_domain/interface_adapt_v1/evaluation/protocol.json)（`715e94249707`），以`8ee7cf6`在GPU运行前推送。只有训练组120条进入分词缓存：A/C每轮输入232,598 token、监督9,734 token；固定5轮、学习率5e-5、batch 2、累积8，各A/C检查点已完成40次更新、48,670监督token。两组A/C的样本、实际顺序、监督量和配置匹配复核通过。P完成同样40次更新，使用同题程序轨迹，监督量4,200×5=21,000，与Agent不同。Agent只监督成功的三次动作，不增加恢复或偏好目标；这是共同接口接入，不包装成新算法创新。

评价已完成520条新预测：五模型适配前各40道合成保留题，适配后各40道合成题加24个已暴露雷达中英问法；雷达适配前结果直接复用旧120条冻结预测。开发组没有用于调参；合成包含20个双语意图组，每语言20题，雷达两语言共享12个目标，语言版本不能算独立样本。全部10路完成、520条CPU重放零差异后才读取参考。原人工验收门禁、12条来源读法与已见题身份均不改变，最终独立雷达QA仍未形成。

| 检查点 | 合成精确选择：适配前→后／40 | 适配后已见雷达：中文／12 | 英文／12 |
|---|---:|---:|---:|
| P | 1→40 | 11 | 11 |
| A1 | 0→39 | 12 | 12 |
| C1 | 7→40 | 12 | 12 |
| A2 | 3→40 | 12 | 12 |
| C2 | 8→40 | 12 | 11 |

见[统一结果](../../results/radar_domain/interface_adapt_v1/evaluation/summary.json)。15个GPU作业全部以`syx`完成，结束检查四卡空闲；[执行记录](../../results/radar_domain/interface_adapt_v1/execution_summary.json)为北京时间19:39:54–19:59:20，本轮0.916 GPU小时、累计23.937/72，归档后本地再次核验520条重放零差异。结果支持共同接口适配有效，普通A同样达到高分，不能证明恢复C有额外领域优势；已见题的记录选择也不是真实装备事实准确率。模型只选择记录，来源证据卡仍由模板展示，不计为独立生成语义回答。[阶段决定](../../results/radar_domain/interface_adapt_v1/stage_decision.json)固定全部模型，停止训练和提示搜索，保留此前失败证据。

后续新厂商来源的固定六入口评价及双AI正文审阅现已完成，具体结果与收束决定见文首。原12题和衍生问法永久排除在新评价之外。领域章节可以如实报告主方法没有额外优势，不以接口适配或合成高分替代方法创新和事实核验。

## 运行

### AI开发数据与中文模型查询（2026-10-05）

首批审阅意见已应用为单独的 [v2开发版本](../../artifacts/thesis_direction_review/radar_development_v2/manifest.json)，原候选、人工决定表和答案留空的原开发题保持不变。01–03将历史检索锚点与Patriot系统来源主体分开；11将取消记录的属性改为`lifecycle_event`；12把未知范围限定为绑定来源字段。区间、部件/情境标签、事件和单位独立保存，09–10的第三方转录疑点只保留为AI审阅注记，不覆盖快照或合成新参数。

[开发执行环境](../../experiments/radar_domain/development_environment.py)复用公开实验的`step/finish`调用形式，支持`Find`、`QueryAttr`和`QueryAttrUnderCondition`。P的一次性程序与A/C逐步调用共用执行语义，没有按问题编号查答案或自动修复查询。全局提供一致的实体锚点、字段释义和限定词目录；问题仅增加中性的来源文档上下文，不包含逐题目标字段、条件或参考查询。

模型的任务是选取证据记录。选中记录再由固定模板展示中文证据卡，保留来源主体、范围、单位、条件、短引文及定位。模板中的限定说明来自AI审阅资料，**不能记作模型独立进行事实判断或生成答案的能力**。12题均为已暴露开发题，永久不进入独立评价。

[五检查点开发协议](../../results/radar_domain/development_probe_v1/protocol.json)固定P、两个普通续训种子A和两个恢复续训种子C，无新训练、无按成绩重跑。每模型12题、greedy、上下文8192、生成总预算2048，Agent最多24调用。计分区分精确目标集合、目标覆盖、允许背景记录、无关额外记录和未完成；02/03可保留另一事件作比较背景，但严格集合指标仍单独报告。先确认全部完成并重放轨迹，再打开独立保存的AI参考文件。

```bash
# 检查新版本与全部原始来源/审阅输入一致（需要本地原始资料）
python3 -m experiments.radar_domain.development_data --check

# 在配好原检查点的 syx GPU 环境中，通过现有 run_job 监管器运行；示例P
python -m experiments.agent_feedback.run_job --name radar_dev_v1_P --gpus 0 \
  --max-hours 0.25 -- python -m experiments.radar_domain.development_probe generate --label P

# 五路全部完成后做CPU重放和开发诊断；--check仅复验既有报告
python3 -m experiments.radar_domain.development_analysis
python3 -m experiments.radar_domain.development_analysis --check

# 首次迁移检查后另存的中英单意图材料；只执行12条参考程序，不调用模型
python3 -m experiments.radar_domain.task_calibration --check
```

下列命令保留原始包和显式查询演示的复验入口：

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

### 一次中英单意图对齐复测

用户于2026-10-05授权执行已准备的校准材料。[运行前协议](../../results/radar_domain/lookup_probe_v1/protocol.json)固定P和A/C两个续训种子，每模型分别运行同一12题的中文与英文版本，共10组120条流程。新[运行器](../../experiments/radar_domain/lookup_probe.py)沿用原模型权重、提示、工具、知识版本、解析方式及生成预算，仅更换题面文件、语言标记和输出路径。每组由原`run_job`监管，最多15分钟，总上限2.5 GPU小时；实际耗时另记，不把上限当作实耗。

英文题仍共享原双语系统提示和中文知识注记。因此中英差异只描述这12组AI改写题上的题面语言关联，不能推广为完整英文与中文系统优劣。与上一轮相比又改变了题意范围，只能作为任务改写诊断，不能全部归因于翻译。

[统一分析器](../../experiments/radar_domain/lookup_analysis.py)先确认10组完成与语言、模型、配置及输出哈希，再重放全部120条流程，最后读取AI参考。分别报告严格目标选择、完成/空结果、调用错误和成本，并区分“从未取到目标”与“曾取到目标却未正确结束”。按同一模型的12个目标配对列出中英都对、仅中文对、仅英文对和都错，不把24个语言版本或不同检查点当独立样本。

```bash
# 仅在原syx GPU环境中运行；每个名称/输出只能首次创建
python -m experiments.agent_feedback.run_job --name radar_lookup_v1_P_zh --gpus 0 \
  --max-hours 0.25 -- python -m experiments.radar_domain.lookup_probe generate --label P --language zh

# 十组全部完成后统一CPU回放和诊断；不调用GPU
python3 -m experiments.radar_domain.lookup_analysis
python3 -m experiments.radar_domain.lookup_analysis --check
```

这次用尽预定的单次对齐复测机会，不根据结果继续改提示或挑种子。其后用户授权的合成材料验收和共同接口适配见上节；原复测协议与结果保持不变，不将这批已见衍生题转成最终评价。
