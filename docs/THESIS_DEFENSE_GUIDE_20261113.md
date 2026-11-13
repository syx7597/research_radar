# 开题答辩研究介绍与追问准备

更新：2026-11-13。题目为**《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》**。本稿依据已保存的实验与当前开题v3整理，供口头介绍和答辩准备使用。本文中的A为普通配对续训，C为恢复轨迹续训，I为两组共享的初始模型。

## 一、约三分钟的口头研究介绍

各位老师好，我的课题是《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》。研究关注小模型在多步查询中走偏以后，怎样继续利用已有结果完成任务。

程序式问答需要模型依次定位实体、查找属性、筛选条件并提交答案。常规训练主要提供正确执行轨迹，实际运行却可能出现额外查询、空结果或无效调用。我的研究思路是利用初始模型真实产生的失败历史，构造适合当前状态的后续监督，让模型学习在偏离历史之后继续求解。

具体方法包含三个环节：首先定位与参考程序一致的共同成功前缀，保留其后的真实分歧及反馈；其次，将参考后缀的依赖映射到当前结果句柄；最后实际重放后缀并核验答案，将后续助手动作作为训练目标。已有历史样本的离线诊断也核对了映射对后缀构造的作用。普通组和恢复组共享初始模型、问题、成功材料及监督预算，比较不同历史上下文带来的学习效果。

目前，公开知识图谱任务的主要评价已经完成。在KQA Pro的2000题项目留出集上，两个配对续训种子的平均准确率由78.75%提高到83.50%，增加4.75个百分点。执行记录显示，改善主要伴随未完成回合和无效调用的减少。接下来将把两组模型放入相同的正常历史和偏离历史，进一步检验它们从同一状态继续求解的表现。

雷达是应用验证场景。技术参数通常同时关联型号、部件、工作模式和来源，适合研究条件化查询。我已整理八来源502条读法和96道题；相同语义记录采用条件绑定表示后，正文正确数由73题提高到80题。进一步的域内任务已准备576条合成训练题、108道开发题，以及分别报告的24道基础评价题和12道组合题。

后续将从同一个领域初始模型开展普通与恢复配对训练，直接评价包含参数、条件和出处的执行答案。整个研究形成“来源知识与任务、恢复监督方法、公开效果与行为分析、雷达应用验证”的完整链条。

介绍中主要数字的依据：[公开留出结果](../results/agent_feedback/holdout_stage_decision_v1.json)、[雷达表示结果](../results/radar_domain/coverage_v2/semantic_summary.json)、[领域划分](../results/radar_domain/frozen_20261107/summary.json)、[组合评价补充](../results/radar_domain/compositional_supplement_20261108/summary.json)、[离线构造诊断](../results/agent_feedback/construction_diagnostic_20261113/summary.json)。

## 二、十个常见追问及回答

### 1. 为什么选择雷达领域？

雷达资料中的同名参数经常对应不同型号、部件和工作配置，范围、离散选项、典型值与上限也有不同含义。例如，核对环境温度时需要区分扫描器与显示部件。这样的资料同时要求正确选择事实、保持条件对应和给出出处，适合检验多步条件查询。应用目标是技术参数检索、配置条件核查和资料追溯。

证据：[数据构建章节](../thesis/current/03_radar_data_construction.tex)、[真实开发温度案例](../results/radar_domain/programs_dev_20261107/development_case.json)。

### 2. 知识图谱在哪里？为什么雷达查询采用类型表？

公开任务直接使用KQA Pro的实体、关系、属性及限定条件，执行KoPL图查询。雷达已有主体—规格读法的图投影，类型视图从同一来源记录构建，通过记录标识和出处对应。实际核对中，502条原始读法均可还原，1015组来源字段合取查询在两种视图中选中相同记录。类型表进一步显式保存范围、选项和配对量值，支持组合运算。两类环境共享只读操作、显式依赖和可重放后缀，使同一恢复构造思想能够进入领域任务。

证据：[KoPL执行器](../experiments/condition_consistency/executor.py)、[雷达图投影](../experiments/radar_domain/kopl_record_bridge.py)、[类型视图](../experiments/radar_domain/typed_sources.py)、[同源视图核对](../results/radar_domain/dual_view_audit_20261113/summary.json)。

### 3. 方法贡献是什么？为什么采用3B模型和LoRA？

研究贡献集中在恢复监督的构造：把真实偏离历史与正确参考后缀衔接起来，处理依赖编号变化，通过执行核验形成训练目标，并在匹配监督量的条件下评价效果。3B模型用于考察有限模型规模下的多步查询能力，LoRA提供可重复的参数高效适配。相关工作已经研究状态采集、轨迹修订和部分掩码；本课题重点研究参考程序在当前偏离状态下的依赖一致衔接与可执行性。

证据：[方法第四章](../thesis/current/04_execution_feedback_method.tex)、[相关工作第二章](../thesis/current/02_background_related_work.tex)、[开题v3正文](../thesis/proposal/review_20261113/content_submission.json)。

### 4. 句柄映射如何保证恢复目标可靠？它是否证明语义正确？

映射只登记与参考一致且成功执行的共同前缀，以及重新执行的参考后缀结果。真实分歧产生的结果保留在历史中，后续参考依赖指向对应的有效结果。构造过程核对依赖、类型、执行观察和最终参考答案，保证目标在固定只读环境下可执行。检查对象是相对于已验证参考的一致性；来源本身及问题语义另由数据复核负责。当前418对覆盖回放有效失败的约63.4%，其余按缺少共同前缀等原因记录。

补充证据是本轮CPU构造诊断：固定418个历史配对，其中414个能严格重现比较起点，另外4个因本地错误文字措辞不同单列保留。在同一414个起点上，直接拼接有65条后缀全部调用通过、9条最终答值匹配；依赖映射后414条均通过并与参考一致。直接拼接的9条答值匹配仍存在依赖来源差异。映射候选与验收后的414条序列相同，验收没有新增拒绝；该结果呈现构造规则的作用，模型学习收益由模型实验另行评价。

证据：[公开恢复构造](../experiments/agent_feedback/recovery.py)、[原构造计数](../results/agent_feedback/recovery_pairs.json)、[CPU构造诊断](../results/agent_feedback/construction_diagnostic_20261113/summary.json)、[独立复算与重放审计](../results/agent_feedback/construction_diagnostic_20261113/independent_audit.json)。

### 5. A/C对照怎样保证公平？提升会不会只是多训练了？

A/C从同一个I开始，使用相同成功材料和同一逻辑参考后缀，匹配问题、记录顺序、重复次数和逐记录监督token。A从共同前缀继续，C额外读取一次真实分歧及反馈，再学习映射后的后缀。两组各使用788534个监督token；恢复历史增加的输入长度单独计入成本。公开结果中C相对A的累计总token下降约8.84%和4.17%，而相对一次性程序P，C的累计token约为其24倍，论文同时报告质量与成本。

证据：[监督预算及训练设置](../thesis/current/04_execution_feedback_method.tex)、[效果与成本分析](../thesis/current/05_public_experiments.tex)、[留出阶段汇总](../results/agent_feedback/holdout_stage_decision_v1.json)。

### 6. 同状态实验具体检验什么？为什么既看Δ_D又看差分？

由固定初始模型I采集历史并筛选合格状态，同一问题设置正常历史N和保留真实分歧的历史D。普通A与恢复C分别在相同N、相同D中续行，并匹配剩余调用及新增生成预算。主指标Δ_D是偏离条件下C减A的正确率差；同时报告正常条件增量Δ_N及Δ_D−Δ_N。前者直接检验同一偏离状态下的求解效果，差分帮助分析收益是否在偏离历史中更突出。状态池和协议已固定，实际评价待GPU运行。

证据：[冻结同状态协议](../results/agent_feedback/same_state_v1/design_protocol.json)、[状态构造](../experiments/agent_feedback/same_state.py)、[差分与配对统计](../experiments/agent_feedback/same_state_analysis.py)。

### 7. 具体报错文字诊断结果较小，说明什么？

该诊断在推理时替换失败观察中的错误类别和详情，保留真实执行状态、句柄及全部成功观察。四个模型的变化为+0.2至+0.4个百分点，区间均包含零，表明这一设置下详细报错文字没有显示稳定的额外收益。恢复训练418对中有414次分歧可以执行，主要覆盖合法偏离之后的继续求解。同状态实验将进一步分析模型对这类历史的适应。

证据：[反馈文字诊断协议](../results/agent_feedback/protocol_feedback_diagnostic_v1.json)、[诊断结果](../results/agent_feedback/feedback_diagnostic_decision_v1.json)、[恢复样本构成](../results/agent_feedback/recovery_pairs.json)。

### 8. 雷达评价只有24题加12题，可信性怎样保障？

这两部分基于相同四个来源家族和156条读法，分别作为基础评价与组合评价报告，保留全部问题、来源组计数、改正与退步。训练使用明确的合成事实，开发和评价目录按来源、主体及记录分组隔离。另有八来源96题用于条件表示研究。对其中固定的24题，新一轮非作者AI先读原件独立作答、封存后再比对，24题均与参考一致，既有型号列引用修订得到确认，并保留一项原文范围歧义。该审阅登记为AI来源核验，真人完成记录仍为0/24。

证据：[正式划分复核](../results/radar_domain/frozen_20261107/independent_audit.json)、[组合题冻结汇总](../results/radar_domain/compositional_supplement_20261108/summary.json)、[本轮独立AI来源审阅](../artifacts/thesis_direction_review/radar_source_review_20261113/summary.safe.json)、[固定人工核查协议](../results/radar_domain/coverage_v2_citation_sensitivity_v1/external_review_protocol.json)。

### 9. 强化学习在论文中是什么位置？

强化学习是已经实施的扩展对照。第一续训种子的A/C分别进行相同规模的短程GRPO，得到B/D。开发集上，B相对A变化−0.4个百分点，D相对C增加1.2个百分点，两项区间均包含零；结果支持继续以恢复轨迹监督作为主要方案。论文保留RL的训练设置、结果和成本，用于分析结果奖励在当前设置下的增量作用。

证据：[RL阶段决定](../results/agent_feedback/rl_stage_decision_v1.json)、[公开实验扩展部分](../thesis/current/05_public_experiments.tex)。

### 10. 已经完成了多少？接下来怎样形成完整论文？

已完成公开方法效果、状态与成本分析、雷达条件表示实验，以及新领域数据和运行入口。待完成的两块模型实测是公开同状态评价和雷达I_domain→A_domain/C_domain配对训练。当前服务器暂不可用，本地未找到基础权重和五个原公共检查点副本，下一步先核查原存储或备份并验证Linux/CUDA环境。随后完成既定评价、来源核查及误差分析，将实际结果补入现有七章稿。720条参考、46项测试和216条脚本演练属于运行准备证据。

证据：[运行准备清单](RADAR_RUNTIME_READINESS_20261108.md)、[本地资产核对](../results/radar_domain/runtime_readiness_20261108/asset_inventory.json)、[实验矩阵](../results/radar_domain/compositional_supplement_20261108/experiment_matrix.json)、[当前同状态状态](../results/agent_feedback/same_state_v1/round_status.json)。

## 三、答辩时使用的“主张—证据—状态”表

| 主张或待回答问题 | 对应证据 | 当前状态与口径 |
|---|---|---|
| 恢复监督提高了固定公开任务的完整问答准确率 | [2000题留出汇总](../results/agent_feedback/holdout_stage_decision_v1.json)：A平均78.75%，C平均83.50%，+4.75个百分点，题级区间[3.53,5.98] | **已完成模型实测**；两个续训种子共享初始模型和语料 |
| 改善主要伴随完成率提高和无效调用减少 | [状态转移汇总](../results/agent_feedback/state_transitions_v1/summary.json)、[第五章](../thesis/current/05_public_experiments.tex) | **已完成分析**；状态转移是描述性分解 |
| 依赖映射对参考后缀接入的作用 | [构造诊断](../results/agent_feedback/construction_diagnostic_20261113/summary.json)、[独立审计](../results/agent_feedback/construction_diagnostic_20261113/independent_audit.json)：414可比起点上直接／映射最终答值匹配9／414 | **已完成CPU构造诊断**；固定418分母另含4个措辞差异未比较，9个答值相同仍有依赖来源差异 |
| 相同偏离历史下，C是否更能继续完成任务 | [同状态协议](../results/agent_feedback/same_state_v1/design_protocol.json)及[统计实现](../experiments/agent_feedback/same_state_analysis.py) | **待模型实测**；Δ_D为主，Δ_N与差分为次级诊断 |
| 条件绑定组织改善所选雷达资料的回答 | [96题表示实验](../results/radar_domain/coverage_v2/semantic_summary.json)：正文73→80、绑定错误17→11 | **已完成模型实测与AI判分**；结果归属于表示组织 |
| 统一引用处理后两种表示的联合正确数不同 | [引用解析分析](../results/radar_domain/coverage_v2_citation_sensitivity_v1/semantic_summary.json)：55／68 | **已完成事后分析**；原始17／52结果另行保存 |
| 原公开恢复检查点经接口适配，是否改善旧来源回答 | [旧流程诊断](RADAR_METHOD_ALIGNMENT_REVIEW_20261104.md)：C相对A少1题／持平 | **已完成模型实测及流程诊断**；后续采用域内训练与直接执行答案 |
| 新领域任务和恢复训练入口已准备 | [运行准备证据](../results/radar_domain/runtime_readiness_20261108/completion.json)：576训练、720参考、46测试、216脚本控制 | **CPU准备已完成**；真实域内训练、采样及正式恢复配对仍为0 |
| 雷达图投影与类型视图有可核对的同源关联 | [双视图检查](../results/radar_domain/dual_view_audit_20261113/summary.json)：502记录还原、1015组来源字段查询一致 | **已完成CPU身份与选择核对**；覆盖来源字段合取查询 |
| 恢复监督在雷达组合查询上有多大增量 | [固定域内实验安排](../results/radar_domain/compositional_supplement_20261108/experiment_matrix.json) | **待模型实测**；24基础题与12组合题分开报告 |
| 固定24题参考的来源支持及定位修订 | [本轮独立AI复核](../artifacts/thesis_direction_review/radar_source_review_20261113/summary.safe.json)：24题一致、36处引用片段通过，保留1项原文范围歧义 | **AI审阅已完成**；原判与真人模板保持，真人完成记录仍为0/24 |

## 四、上台前核对的五个表述

- 使用“提升4.75个百分点”，并说明对象是KQA Pro的项目留出2000题。
- 使用“两个续训种子”，区分题级bootstrap与续训随机性。
- 将雷达“正文73→80”的结果归到条件表示研究。
- 将原24道基础题、12道组合题分别报告；人工核查24题另有抽样来源。
- 将公开同状态和雷达域内训练列为待完成实测，CPU参考回放与脚本控制列为准备工作。

本稿中的模型成绩沿用既有记录。本轮新增的是CPU构造诊断、同源视图核对和AI来源审阅，未增加模型训练组或产生新的模型问答成绩。
