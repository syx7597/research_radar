# 复杂知识问答：公开基准验证与雷达知识应用

**论文定位：电子信息专业硕士，以一项可归因的算法或训练改进为研究核心，雷达知识资源和领域应用提供支撑。** 建议题目为“基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用”。已调研9篇具有学位依据的硕士论文，具体文献、贡献边界、七章框架和六个月安排见[论文主线](docs/THESIS_ROUTE_REVIEW.md)。

**当前实验：固定2000题项目留出评测完成，恢复轨迹SFT的收益保留。** 两个续训种子分别由78.25%提升到82.90%、79.25%提升到84.10%；预先固定的两种子平均增益为**+4.75个百分点，题目配对95%区间[3.53,5.98]**。两轮共享初始模型和语料，属于条件续训种子复现；不是两个完整端到端种子。雷达接口和来源回答已完成最小接通，独立人工参考、更广来源覆盖和可靠条件理解仍待补足；不能把既有大图和自动题视为已验证成果。

[导师式方法审查](docs/THESIS_ROUTE_REVIEW.md#8-导师式方法审查将工程主线落实为可检验的训练改进)中的“语义核验辅助的轨迹偏好训练”已按用户要求暂缓，不在当前队列。恢复轨迹SFT已获得本项目留出证据，停止公开集追分，转入雷达核验与应用验证；工程完整性不能替代核心方法证据。

主线：**保留主体、条件、区间和来源的雷达知识资源 → 小模型生成可执行查询 → 答案与依据 → 公开基准和独立雷达评价**。复用现有数据、执行器和系统，不重建整个项目。配对续训复现、回放复核及B/D配对RL均已完成。RL增量未达到扩展标准，保留恢复轨迹SFT为主方案；语义核验偏好训练继续暂缓。领域核验与论文写作并行推进；范围和队列见[执行计划](docs/ROADMAP.md)。

**2026-10-07：数据构建章及下一轮评价设计已完成复核，本轮没有新训练或模型推理。** [第三章草稿](thesis/current/03_radar_data_construction.tex)区分历史库存、旧12条开发读法、合成100组和新来源材料，附6张表、109项可重建数据字段；当前第三至六章均有工作稿，PDF尚未编译，论文整体尚未完成。

[下一轮设计](artifacts/thesis_direction_review/radar_coverage_v2/design.json)暂定8个新来源/家族组、至少5家厂商、96道中文题。[8个官方候选入口](artifacts/thesis_direction_review/radar_coverage_v2/source_candidates.json)只完成网页级筛查，尚未归档或出题。核心对照是同一完整证据的逐行表示与条件绑定组织；原文页作系统参照、规则模板作资料展示参照，不通过删掉条件削弱基线。至少48题存在真实竞争读数、覆盖6组，且12题需要双限定或多值对应，才启动该对照；不足则取消，避免做单记录格式试验。

[独立AI设计审阅](artifacts/thesis_direction_review/radar_coverage_v2/design_review.json)未发现阻断问题，17项CPU审计测试及三组论文表格复核通过；设计审阅不等于来源事实验收或模型运行就绪。

[已见登记](artifacts/thesis_direction_review/radar_coverage_v2/exposure_registry.json)覆盖10份已见来源/10个家族组及全部已登记问法版本；[CPU划分审计](experiments/radar_domain/source_split_audit.py)检查显式闭包、跨组混用和AI身份声明，只证明元数据一致性。下一步先归档候选来源并核验家族/版本/表格可读性，再整理新读法与题目；达到数据和CPU门槛后另行冻结一次最多2 GPU小时的三路LLM评价。当前只是设计，既没有96道新题，也没有运行协议冻结或新GPU任务。

**前一轮：新厂商来源固定评价和双AI正文审阅完成，未观察到恢复C的额外领域正确性收益。** [资料包](artifacts/thesis_direction_review/radar_sources_v1/manifest.json)来自Vaisala WRM200、Leonardo METEOR 735C、Furuno WR2120和JRC JMA-1030系列的4份厂商PDF，共4个来源/家族、5个型号、24道中文题、62条来源读法和15个完整原文页。[来源AI审阅](artifacts/thesis_direction_review/radar_sources_v1/independent_ai_audit.json)通过。新资料与原12条实际调试来源/家族闭包隔离，但3个家族已在历史语料出现，旧自动QA完整来源血缘未认证；这是小规模AI探索评价，不是最终独立人工评价。

六种证据入口为固定适配后的P/A1/C1/A2/C2和BM25 top4 RAG，统一由未经本项目适配的Qwen2.5-3B-Instruct根据原文页生成正文、断言与引用，中文模板证据卡不算回答。120条选择及144条回答的CPU重放/输入核验均零差异，见[机械检查](results/radar_domain/source_eval_v1/mechanical_summary.json)和[AI语义汇总](results/radar_domain/source_eval_v1/ai_semantic_summary.json)。

| 证据入口 | 精确记录选择／24 | 支持页覆盖／24 | 正文正确／24（AI审阅） | 选择＋回答总token |
|---|---:|---:|---:|---:|
| RAG | 不适用 | 23 | 13 | 88,520 |
| P | 15 | 19 | 13 | 48,368 |
| A1 | 21 | 22 | 16 | 212,651 |
| C1 | 19 | 23 | 15 | 176,433 |
| A2 | 20 | 22 | 16 | 169,768 |
| C2 | 19 | 24 | 16 | 143,625 |

两个隐藏方法标签的AI审阅者对58个去重答案的正文正确性完全一致；5条分歧仅涉及缺证据拒答是否有支持，按限定措辞裁决为支持，不改变其答错和引用未支持判定，原判保留。主agent裁决已见汇总结果，不称盲裁决或人工金标。严格断言字段匹配均为0/24，**这不是正文准确率为0**。显式查询条件子集保留8题分母，另公开运行后增加的9题参考限定诊断，包含第15题双波束说明，不静默换分母。

[运行记录](results/radar_domain/source_eval_v1/execution_summary.json)：11个GPU任务均由`syx`完成，无新训练；北京时间22:04:05–22:06:41，实耗0.103 GPU小时，累计24.040/72。表中总token包含选择和回答，不能因Agent回答阶段上下文较短就称其比RAG省token；资料整理成本也不同。[预冻结检索23/24](artifacts/thesis_direction_review/radar_sources_v1/pre_freeze_retrieval_observation.json)已披露，未据此调参。

接口已可用，但C1正文比A1少1题，C2与A2持平；C2虽覆盖全部24题支持页，仍有8题在已有证据下答错。当前短板集中在表格量值、条件/部件绑定与回答，不由本批推定某个新算法已有效。按[阶段决定](results/radar_domain/source_eval_v1/stage_decision.json)保留公开2000题方法结论，领域不宣称算法领先；停止追加领域训练、RL及提示搜索，先收束应用章节、来源证据表与[失败案例](results/radar_domain/source_eval_v1/failure_diagnosis.json)，再补数据构建章衔接。本批没有未知或事件题，不能宣称对应能力已验证。24道AI题只是最小验证，完整论文仍需按预定义来源多样性、题型和参考可信度补充评价工作，当前不按分数扩样追分。

**2026-10-05：一次共同接口适配与固定前后评价已完成，接口使用明显改善。** [合成材料](artifacts/thesis_direction_review/radar_interface_synthetic_v1/manifest.json)含100组虚构意图，按60/20/20划分训练、开发和保留组，中英共200题；[另一个AI agent审计](artifacts/thesis_direction_review/radar_interface_synthetic_v1/independent_ai_audit.json)及CPU执行检查通过。四个A/C检查点共用120条成功调用轨迹，固定5轮、学习率5e-5、batch 2、累积8，均完成40次更新、48,670个监督token；P完成同样40次更新、21,000个监督token，不能视为与Agent等预算。两组A/C的样本、实际顺序、监督量及配置匹配复核通过。该步骤用于教会新接口，不新增算法创新点。[训练](results/radar_domain/interface_adapt_v1/protocol.json)与[评价协议](results/radar_domain/interface_adapt_v1/evaluation/protocol.json)已在GPU运行前以`8ee7cf6`推送。

| 检查点 | 合成精确选择：适配前→后／40 | 适配后已见雷达：中文／12 | 英文／12 |
|---|---:|---:|---:|
| P | 1→40 | 11 | 11 |
| A1 | 0→39 | 12 | 12 |
| C1 | 7→40 | 12 | 12 |
| A2 | 3→40 | 12 | 12 |
| C2 | 8→40 | 12 | 11 |

新增520条预测全部CPU重放零差异，本地归档后再次复验一致；雷达适配前对照直接复用旧120条结果。15个作业全部由`syx`完成，结束检查四卡空闲。[执行记录](results/radar_domain/interface_adapt_v1/execution_summary.json)：北京时间19:39:54–19:59:20，本轮0.916 GPU小时，该阶段结束时累计23.937/72。见[适配结果](results/radar_domain/interface_adapt_v1/evaluation/summary.json)和[阶段决定](results/radar_domain/interface_adapt_v1/stage_decision.json)。合成40题来自20个双语意图组，每语言20题；雷达两语言共享12个已见目标，不按独立样本累计。模型只选择来源记录，证据卡由模板展示，不计作独立生成语义回答。共同接口适配有效，但普通A同样达到高分，**未证明恢复C在领域上的额外优势，也不能称为真实雷达准确率**。全部检查点已固定，停止该轮训练和提示搜索；后续的新厂商来源评价与双AI审阅已完成，结果见上文。AI参考与独立人工金标分开标注，公开方法收益与领域应用边界分别报告。

此前预定的一次中英对齐复测已完成，停止未经适配的直接迁移。在同12个查询目标上，P/A1/C1/A2/C2中、英文精确记录选择均为1/1/2/3/0；C1和A2的中英成功题部分不同。10组120条流程重放零差异，仍有大量参数、字段/条件和句柄错误，没有可靠语言优势或领域C优势。见[复测结果](results/radar_domain/lookup_probe_v1/summary.json)和[阶段决定](results/radar_domain/lookup_probe_v1/stage_decision.json)。该次复测0.240 GPU小时，当时累计23.021/72；没有训练，不再改提示追分。

此前首轮中文开发检查也完整保留：[AI交叉审阅](artifacts/thesis_direction_review/radar_ai_cross_review_v1/index.html)的12条修订已应用为[独立开发版本](artifacts/thesis_direction_review/radar_development_v2/manifest.json)，原始数据保持不变。中文问题→模型查询→真实执行→固定模板来源证据卡已接通，首轮目标记录选择为P 0/12、A 0/12与0/12、C 3/12与1/12，60条重放零差异，实耗0.139 GPU小时。见[首次负结果](results/radar_domain/development_probe_v1/summary.json)。两轮都是已暴露AI开发材料，固定模板不代表模型独立完成来源裁决，不能称独立雷达准确率。

新主线的[方法与公开实验两章草稿](thesis/README.md)及8张可追溯结果表已完成。[领域评价设计](artifacts/thesis_direction_review/radar_evaluation_plan_v1.json)和[空标注模板](templates/radar_evaluation/README.txt)已备齐，尚未冻结最终独立QA；本次合成接口适配不改变这一状态。用户的专业审阅不作为开发前提；AI材料与人工金标分开，当前仍无已验收独立事实，原接口默认事实模式返回无可用数据。

**已完成两轮开发和KQA Pro完整官方validation的11,797题评测。** 同一5,000题训练的BART下，4候选执行筛选79.90%，普通保守字段修复81.25%，局部保守修复81.39%，8候选执行筛选81.44%。局部保守修复相对4候选纠正178题、改错2题，但未证明优于8候选；局部约束相对普通修复的额外增量很小。

后验补算相同保护的beam8为81.74%，进一步削弱当前字段修复的必要性；字段修复已停止开发。完整正负结果、单种子和数据使用边界见[修复验证报告](docs/QUERY_REPAIR_VALIDATION.md)。历史首轮排序负结果保留在[最小验证报告](docs/MINIMAL_VALIDATION.md)。

## 阅读入口

当前方向和队列以[论文主路线](docs/THESIS_ROUTE_REVIEW.md)及[执行计划](docs/ROADMAP.md)为准；此前小算法论证和训练建议保留为历史，不再作为新路线前置条件。

| 文档 | 用途 |
|---|---|
| [论文整体路线](docs/THESIS_ROUTE_REVIEW.md) | 专硕定位、9篇硕士论文、研究贡献、七章框架和实证计划 |
| [公开方法子实验原计划](docs/THESIS_AGENT_PILOT_PLAN.md) | 保留Agent/SFT/RL与恢复训练的具体机制、对照和停止规则 |
| [执行计划](docs/ROADMAP.md) | 当前任务队列与六个月里程碑 |
| [此前小算法方法论证](docs/research/METHOD_FEASIBILITY_BRIEF.md) | 历史字段约束/负例假设，已停止作为当前入口 |
| [字段约束接口与覆盖检查](docs/research/SCHEMA_CONSTRAINT_BASELINE.md) | 已有技术基线C的范围、分词器核验与接入边界 |
| [字段约束500题对照](docs/research/SCHEMA_PILOT_RESULTS.md) | 实际收益、改错、成本与停止决定 |
| [雷达标注规范与模板](docs/research/RADAR_DATA_ANNOTATION.md) | 事实核验与独立问答分离、来源追踪待复核样例 |
| [雷达事实审阅与独立QA准入](docs/research/RADAR_REVIEW_WORKFLOW.md) | 12条事实审阅表、12道流程开发题、来源绑定与人工验收边界 |
| [查询修复完整验证](docs/QUERY_REPAIR_VALIDATION.md) | 两轮开发、完整官方val、强对照、误修改与成本 |
| [首轮最小验证结果](docs/MINIMAL_VALIDATION.md) | 历史排序实验与负结果 |
| [历史字段约束方向](docs/RESEARCH_DIRECTION.md) | 保留旧假设与停止理由，当前入口为论文主线 |
| [公开基准专项调研](docs/research/PUBLIC_BENCHMARK_REVIEW.md) | 数据协议、近邻方法、源码核验和公平比较 |
| [当前状态](docs/CURRENT_STATUS.md) | 实际数据规模、已实现内容及证据限制 |
| [训练就绪检查](docs/TRAINING_READINESS.md) | 已修复环境问题与开训前阻断项 |
| [仓库地图](docs/REPOSITORY_MAP.md) | 当前模块、历史成果和本地数据位置 |
| [数据与产物政策](docs/ARTIFACT_POLICY.md) | 公开范围、数据恢复和清理重建 |
| [开题原稿](docs/research/proposal.md) | 保留原结构；贡献表述以新研究方向审查为准 |

## 当前事实（2026-10-04）

**公开实验已完成。** 五路2000题留出推理于2026-10-04 22:44（北京时间）结束，22:51完成全部10000条轨迹/程序回放与统一评分，零回放差异，该阶段GPU任务全部结束。方法/统计在读取问题前冻结，全部预测保存后才读取答案。

| 项目留出2000题 | 首续训种子 | 第二续训种子 |
|---|---:|---:|
| A：普通轨迹SFT | 1565／78.25% | 1585／79.25% |
| C：恢复轨迹SFT | 1658／82.90% | 1682／84.10% |
| C−A | +4.65个百分点 [3.10,6.25] | +4.85个百分点 [3.35,6.40] |
| P：一次性程序基线 | 1523／76.15% | 共用同一P，无第二训练种子 |

主估计先在每道题内平均两组C−A，再按2000道题bootstrap：A平均78.75%、C平均83.50%，增益+4.75个百分点 [3.53,5.98]。它支持当前训练设定下的恢复续训增量，不是官方隐藏测试或普遍语义纠错能力的证明。C比A总推理token减少8.84%/4.17%，但生成token增加，且总token仍是P的23.64/24.72倍，不能宣称普遍更省或更快。见[完整留出结果](results/agent_feedback/holdout_v1_analysis.json)、[回放审计](results/agent_feedback/holdout_v1_execution_audit.json)及[阶段决定](results/agent_feedback/holdout_stage_decision_v1.json)。

以下保留同一500题开发集的全部对照；不与留出题混合统计。

| 方法 | 首种子正确数／准确率 | 第二续训种子 |
|---|---:|---:|
| P：一次性程序基线 | 376／75.2% | 未追加 |
| A：普通轨迹SFT | 392／78.4% | 386／77.2% |
| C：恢复轨迹SFT | 404／80.8% | 412／82.4% |
| B：A + RL | 390／78.0% | 未追加 |
| D：C + 同配置RL | 410／82.0% | 未追加 |

恢复训练C/A在两个续训种子中分别+2.4、+5.2个百分点，题目配对95%区间分别[-0.2, 5.0]、[2.2, 8.2]。首种子D/B为+4.0个百分点，区间[1.2, 6.8]，说明本轮RL后仍观察到恢复训练的对照优势；不能据此把全部差异归因于RL。

**停止扩展RL，主方法保留恢复轨迹SFT。** B/A为-0.4个百分点，D/C为+1.2个百分点；两者区间均跨零，均未达到原先的继续投入门槛。D/C总token还增加2.4%。本轮不再追加RL种子或调整奖励；这不证明所有RL方案不可行。见[阶段结论](results/agent_feedback/rl_stage_decision_v1.json)和[完整RL复核](results/agent_feedback/rl_paired_review_v1.json)。P训练表示与预算不同，且Agent总token远高于P，应报告质量与成本取舍。

错误诊断消融：将具体error/detail换成通用失败内容后，A/C首种子正确数392→394、404→405，第二种子386→387、412→413；差值+0.2～+0.4个百分点，配对区间均包含零。四组正常/遮蔽轨迹共4000条回放通过，但C有5条在首次遮蔽前出现分歧，其中3条原本从未报错；不能把全部变化归因于报错内容。没有证据支持“具体报错诊断带来稳定增益”，也不能据此否定恢复状态训练。保持原正常反馈作为主方案，不采用消融配置追分。见[诊断汇总](results/agent_feedback/feedback_diagnostic_summary_v1.json)。

第二轮A/C及首轮B/D各1000条轨迹均精确回放无差异，四组RL配对比较复算一致。第二轮A/C净增26题中，完成状态切换对应+23题、双方完成作答对应+3题；主要收益伴随格式错误和无效循环减少。62条变化案例AI预审发现部分答对路径仍有条件遗漏、绕过或实体替换，不能把答案正确等同于语义正确，更不能将AI预审直接转为训练标签。见[第二轮复核](results/agent_feedback/replication_review_seed20261004_v2.json)与[案例分析](results/agent_feedback/replication_case_analysis_seed20261004.json)。

正式RL前，两路固定16题、64轨迹、两次更新的短测均确认有效梯度和真实LoRA更新。随后首种子B/D各完成200更新、200个相同训练问题组、800条轨迹，分别有47/43个组产生有效学习信号；不是遍历全部4999个合格训练题。正式训练与dev于10月4日16:00（北京时间）全部结束，实际合计2.524 GPU小时，实验累计13.883/72 GPU小时，全部以syx执行。原始SFT权重未被短测或RL覆盖。见[短测](results/agent_feedback/rl_signal_v1_summary.json)、[RL冻结协议](results/agent_feedback/protocol_rl_paired_v1.json)及[完成记录](results/agent_feedback/rl_paired_v1_summary.json)。

机制诊断已收束；[五模型留出协议](results/agent_feedback/protocol_holdout_v1.json)于2026-10-04 20:29（北京时间）冻结，并在运行前推送仓库。留出推理共用6.885 GPU小时，累计22.641/72 GPU小时。固定P和两个续训种子的A/C，使用原正常反馈，不按结果重选模型或追加调参。下一步是整理公开方法章节、核验雷达事实并接通最小领域问答与来源链路。语义核验偏好训练继续暂缓。

- 公开KQA Pro：固定一个BART训练种子、5,000题监督数据；完成v1首次2000题检查及v2完整official val 11,797题评估。旧2000题在v2中仅作已见诊断；official val有历史使用，不能称hidden test。
- 当前图谱21,928条边、10,744个实体；RL过滤后保留11,432条边记录。
- train/dev/test 为4,277/582/542；SFT轨迹4,277条。
- 5,401题工具oracle检查通过，证明数据和工具自洽，不是模型正确率。
- 旧固定策略系统499题88.6%属于历史结果，不能作为新方法实验结论。
- 新公开实验固定5,000训练题、500开发题及2,000项目留出题；后者排除此前已用组件，仍不是官方隐藏测试集。见[冻结协议](results/agent_feedback/protocol.json)与[切分清单](results/agent_feedback/split_manifest.json)。
- 逐步执行与原整程序执行在5,500条金标轨迹上完全一致；训练/开发各有1条既有答案不一致，仅排除训练异常。4,999条训练轨迹通过真实Qwen分词检查。这些是基础设施验证，不是模型正确率。
- 新入口为`experiments/agent_feedback/`：共享初始SFT、同监督预算A/C续训、同模型程序基线、真实交互推理与GPU预算记录。GRPO运行短测已通过，原生工具循环预算见独立修订协议；正式初始模型与程序基线已训练并完成开发评测，恢复训练两次续训种子及一次配对RL对照已完成，当前保留SFT主方案。

2026-10-03第一阶段已完成：初始Agent SFT训练81.4分钟，程序基线训练42.1分钟；500题开发评测和5,000题训练轨迹收集均正常退出，累计约6.03 GPU小时（包括失败及运行短测）。全部模型任务使用`syx`，管理员未执行训练。

| 开发集500题 | 初始Agent（2轮SFT） | 同3B程序基线（3轮SFT） |
|---|---:|---:|
| 正确数／准确率 | 378／75.6% | 376／75.2% |
| 未完成作答 | 61 | 83 |
| 评测进程耗时 | 26.3分钟 | 4.5分钟 |
| 平均输入加生成token | 11,664 | 403 |

Agent相对程序基线纠正42题、改错40题，净增2题；差异0.4个百分点，配对bootstrap 95%区间为[-3.2, 3.8]，不能宣称可靠提升。Agent每轮重新处理历史，计入的输入加生成token约为程序方案28.9倍；这不是FLOPs比值。两者训练形式和预算不同，这只是初始模型与必要强基线的比较，尚未检验恢复训练增量。证据见[完整配对比较](results/agent_feedback/initial_vs_program_dev.json)。

已从5,000题自由生成中构造418组配对恢复样本，覆盖训练题8.36%，占659条通过回放验证的失败轨迹63.4%。418对已固定执行种子再次复核；另有两条因进程集合顺序产生的回放差异被原构造器排除，本轮保持排除并记录原因。恢复点是失败轨迹中首次偏离参考步骤的位置，不保证该步骤本身语义错误；414处仍可执行，4处被执行器拒绝。见[恢复对统计](results/agent_feedback/recovery_pairs.json)及[独立回放审计](results/agent_feedback/recovery_pair_audit.json)。

A/C对照已完成，从同一初始adapter出发，每组5,772条记录、788,534个监督token，其中一半来自配对后缀。418道配对题各复用7–8次；重复不算新增独立样本。两组题目、顺序、复用次数和监督量一致，输入量与计算量不相同。预算裁剪差异已量化，见[语料清单](results/agent_feedback/continuation_data.json)与[监督预算审计](results/agent_feedback/continuation_token_audit.json)。项目留出已在方法冻结后完成评分，此后不能再当作未见集调参。

SFT、八题交互推理和一步原生GRPO运行短测均已完成，见[运行检查](results/agent_feedback/runtime_smoke_checks.json)。GRPO短测奖励、梯度及首步预热学习率为零，只证明兼容性；正式RL前须用完整SFT分支检查学习信号。后续GRPO命令需携带[私有头文件CPATH](results/agent_feedback/python_headers.json)。仓库中的[训练快照](results/agent_feedback/running_snapshot.json)带采集时间，不是实时状态。

历史故障：前两次短测未完成优化步骤，分别遇到Transformers 5.18预热接口变化和CUDA设备不可用；接口修复保持3%预热。此前GPU重置返回`No devices were found`，并记录GSP固件初始化失败，后经上述整机重启恢复。这些属于运行环境故障，不是方法效果负结果。历史证据见[管理员重置尝试](results/agent_feedback/gpu_reset_attempt.json)和[故障期GPU状态](results/agent_feedback/gpu_readiness.json)，模型来源见[权重校验记录](results/agent_feedback/weights_verified.json)。

## 目录

```text
docs/                 当前方向、状态、计划；archive/ 为旧路线资料
ca_agraphrag/          既有RL环境与策略原型，暂不作为新实验主入口
agent/                旧类型化组合执行器、分析Agent与演示
pipeline/v3/          知识构建、手册处理与v3图文检索
qa_*.py               旧固定策略问答与RoG风格基线
graphrag_retriever.py  共用混合检索
experiments/ results/  历史实验代码和结果
  condition_consistency/ 已结束的BART公开基准实验与结果
  agent_feedback/       新3B工具Agent受控实验入口与记录
evaluation/ datasets/  旧评测集与公开集适配代码
paper/ thesis/         历史论文和旧版毕设稿
reports/               汇报生成脚本
scripts/ tests/        审计、清理、发布检查与离线测试
artifacts/             数据SHA-256清单与清理记录
archive/               已废弃OntoCom等历史代码
```

为保持运行路径，本次未大规模移动代码。原始手册、全文语料、图谱、训练数据和下载基准数据保留原位置，但不随公开代码仓库发布。

## 验证与使用

离线边界测试只需Python 3.10+标准库，不需数据、GPU或API：

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/check_public_tree.py
```

按[产物政策](docs/ARTIFACT_POLICY.md)恢复本地数据后：

```bash
python3 -B scripts/audit_workspace.py --verify
python3 -B -m ca_agraphrag.test_env_oracle
python3 -B -m ca_agraphrag.eval_policy --split dev --policy oracle
```

`--verify`核对清单内全部本地文件；只恢复训练所需子集会报告其他文件缺失。
不带参数运行`audit_workspace.py`可只检查当前训练所需图谱和题集。
只有确认建立新数据版本后才用`--write`更新清单，不能用它掩盖意外变化。

历史`requirements.txt`不是锁定并完整验证过的环境。需要API的旧脚本从环境变量取密钥，参见[.env.example](.env.example)。旧SFT/RL原型恢复训练前需通过[训练就绪门槛](docs/TRAINING_READINESS.md)；新公开实验按[执行计划](docs/ROADMAP.md)建立环境与基线。

## 发布与历史

公开仓库包含源码、文档、历史结果和数据清单，**不提供完整研究数据下载**。
本轮清理947,099,791字节（约903.2 MiB）可重建PDF渲染产物，原始输入保留，见[清理清单](artifacts/cleanup_manifest.json)。
旧Git历史含凭据和大型原始资料，留在本地历史分支；公开`main`从干净根提交开始。不要整体推送旧分支或标签。
