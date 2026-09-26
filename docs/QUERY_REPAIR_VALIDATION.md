# 查询修复验证记录

后续补充：已用既有缓存补算“beam8＋相同原查询保护”，为81.74%；这是评估后的诊断，未修改下文冻结成绩，也不算新的独立验证。其对研究范围的影响见[下一步决策](research/NEXT_EXPERIMENT_DECISION.md)。

更新：2026-09-26。v1首次2,000题检查集评估和v2完整official val的11,797题评估均已完成；按[第二轮有界协议](research/QUERY_REPAIR_V2_PROTOCOL.md)，本轮停止调参。第1–5节保留v1历史结果，第6节起记录完整官方validation结果。

**最新结论：保守修复在完整官方validation上相对beam4纠正178题、改错2题，准确率从79.9017%升至81.3936%；但没有超过beam8的81.4360%，也没有稳健证明局部约束优于同样保护的全局修复。** 当前方法价值更适合定位为降低误修改的保守查询修复，不能把局部约束包装为已经成立的核心创新。结果来自一个生成器种子；official val是旧项目使用过的公开validation，不是hidden test。

以下为v1阶段结论，保留当时的结果与判断：

**v1获得了原四候选之外的正确答案，相对beam4执行筛选净增加18题；但局部修复尚未证明优于普通全局修复，准确率也低于beam8执行筛选。** 因而结果支持继续检查误修改机制，尚不足以宣布拟议的局部约束方法成立。

## 1. 实验对象与数据隔离

基础模型是首轮使用5,000条KQA Pro训练样本微调的BART-base，生成器训练种子为20260926；本轮没有重新训练模型。模型仅接收问题，生成KoPL查询程序，再由固定知识库与执行器返回答案。新增的是推理阶段的单字段修复，而非新网络架构。

原1,000题reranker数据固定分成design 500和calibration 500；原500题dev已用于开发诊断。新2,000题从剩余train组件中随机冻结，排除原6500题所属组件及与official val精确问题/程序重叠的组件。所有划分以归一化问题或完整程序连接组件进行去重。生成输入只含ID和问题；金标程序与答案另存，修复、候选执行及模型重评分不读取金标。

global/local在同一calibration上分别从73项策略中选择参数，并在新2,000题答案评估前保存冻结策略。global最终为`alpha=0.5, min_match=0.8, margin=-0.25`，local为`0.5, 0.4, 0.0`。calibration基线395/500，所选global为396/500，local为400/500；这些校准成绩不是独立有效性证据。

首次冻结回执记录时间为2026-09-26 07:10:07 UTC，明确当时尚未评估2,000题答案。所有方法使用完整问题分母，未按执行成功或修复触发情况删除题目。来源：[划分清单](../results/condition_consistency/query_repair/split_manifest.json)、[冻结策略](../results/condition_consistency/query_repair/frozen_policy.json)、[冻结回执](../results/condition_consistency/query_repair/freeze_receipt.json)。具体方法、并列规则和停止条件见[v1协议](research/QUERY_REPAIR_PROTOCOL.md)。

## 2. 执行器修正与历史结果边界

实验中发现官方`Find`可能原地追加同名概念ID，污染后续执行状态。项目适配器为单次调用提供临时列表副本并恢复索引，官方源码保持原样；所有方法统一使用这一隔离修正，并固定`PYTHONHASHSEED=20260926`。

原6500条金标程序分别正序、逆序运行，两遍均6498题答案匹配，逐ID输出差异为0；原有两条金标不匹配保留，不改标签、不删题。详见[执行器验证](../results/condition_consistency/query_repair/executor_validation.json)。它验证的是回放稳定性，不是模型性能或任意程序语义正确性。

原pilot的dev first-valid为400/500，本轮统一修正执行条件下为401/500；top1从历史365变为本轮366。历史缓存及报告保留，不能将变化归因于修复算法。下文各方法都与本轮同一执行条件的基线比较，不把80.2%减去旧80.0%当作算法收益。

## 3. 首次冻结结果

准确率按全部500/2,000题计算。修复方法共用相同beam4候选，每题最多增加4个单字段候选；beam8是独立的8束解码。执行筛选按模型顺序选第一个执行合法的候选。

| 方法 | 已见dev，500题 | 首次holdout，2,000题 |
|---|---:|---:|
| BART top1 | 366，73.20% | 1,472，73.60% |
| beam4 first-valid | 401，80.20% | 1,585，79.25% |
| beam8 first-valid | 399，79.80% | **1,614，80.70%** |
| global repair | 403，80.60% | 1,594，79.70% |
| local repair | **411，82.20%** | 1,603，80.15% |
| global，无接受阈值 | 388，77.60% | 1,518，75.90% |
| local，无接受阈值 | 398，79.60% | 1,524，76.20% |
| global候选答案覆盖上限 | 428，85.60% | 1,677，83.85% |
| local候选答案覆盖上限 | 428，85.60% | 1,693，84.65% |

覆盖上限使用金标判断候选池中是否存在正确答案，仅用于离线诊断，不能当作可部署系统成绩。无接受阈值消融仍要求候选执行合法且schema干净，但只要存在这样的修复，就选择评分最高者，取消关闭开关、最低匹配阈值和接受增量阈值。

以下区间为同一批题上的2,000次配对bootstrap，随机种子20260926；差值单位为**百分点**，不是相对提升百分比。

| 比较 | dev差值与95%区间 | 首次holdout差值与95%区间 |
|---|---:|---:|
| global − beam4 | +0.40 [0.00, +1.00] | +0.45 [+0.10, +0.85] |
| local − beam4 | +2.00 [+0.80, +3.40] | +0.90 [+0.15, +1.65] |
| beam8 − beam4 | −0.40 [−1.80, +0.80] | +1.45 [+0.80, +2.20] |
| local − global | +1.60 [+0.40, +2.80] | +0.45 [−0.25, +1.15] |
| local − beam8 | +2.40 [+0.80, +4.00] | −0.55 [−1.50, +0.45] |

dev上的优势没有完整延续到首次holdout。local相对beam4有小幅净收益；相对global的区间跨0，无法据此确认局部约束的额外优势。beam8点估计高于local，但local与beam8差值区间也跨0；不能把“尚未证明差异”解释为等效或非劣。

| 相对beam4的转换，首次holdout | 纠正 | 改错 | 净增加 |
|---|---:|---:|---:|
| global repair | 11 | 2 | +9 |
| local repair | 38 | 20 | +18 |
| beam8 first-valid | 40 | 11 | +29 |
| global，无接受阈值 | 40 | 107 | −67 |
| local，无接受阈值 | 52 | 113 | −61 |

取消接受阈值后global和local分别比beam4下降3.35、3.05个百分点，配对区间分别为[−4.60, −2.15]、[−4.40, −1.80]。更多可执行修复并不自动带来更多正确答案；误修改控制是当前明确暴露的问题。

完整指标及逐题判定见[dev结果](../results/condition_consistency/query_repair/dev_metrics.json)、[首次holdout结果](../results/condition_consistency/query_repair/holdout_metrics.json)。

## 4. 新候选确有作用，但局部合法性不足以保证正确

local在2,000题中为378题提出1,175个新增候选，最终接受85次修复，其中38次纠正、20次改错、27次未改变答案正确与否。38个纠正案例的原四候选均无正确答案，说明收益确实包括新增正确候选，不只是重排原池。34个纠正后程序与规范化金标程序完全一致。

这仍不能推出其余程序语义正确。例如`male demonym → demonym`虽然答对，却仍缺少男性限定步骤；部分不同方向或逆关系程序同答，也没有证明普遍语义等价。所有案例由脚本机械核查并由AI作事后对照阅读，尚无独立人工语义评审。

20个改错案例中，19个覆盖了原本schema干净的first-valid候选；其中18个原基线程序与金标完全一致。修复只保证相对其父候选修改一个字段，父候选未必就是first-valid，最终替换可能改变基线的整个查询含义。案例包括把`place of birth`改成`date of birth`、把`date of marriage`改成`date of birth`，表明字段存在于局部事实中并不保证它符合问题所需答案类型。

事后类别统计中，数值/极值选择组仅净增加1题，时间组净增加2题；集合与计数组均净增加0题。这些类别由金标定义且相互重叠，只能描述错误，不能据此宣传已解决数值、复杂条件或雷达领域推理。

全部38个纠正、20个改错及语义局限见[案例审计](research/QUERY_REPAIR_CASE_AUDIT.md)与[结构化审计](../results/condition_consistency/query_repair/outcome_audit.json)。此前仅用于选择研究范围的[覆盖审计](research/QUERY_REPAIR_COVERAGE.md)使用原pilot执行条件，不应与本轮修正后的错误数量混算。

## 5. 成本记录

以下是首次holdout的实际缓存构建与评分记录，不是统一串行、同负载的端到端速度基准。各任务曾并行运行，计时范围也不同，不能据此宣称local比beam8更快。

| 计数 | global | local | beam8 |
|---|---:|---:|---:|
| 原生成候选数 | 8,000 | 8,000 | 16,000 |
| 新修复候选数 | 2,197 | 1,175 | 0 |
| 全程序执行次数 | 10,197 | 9,175 | 16,000 |
| 额外局部前缀调用 / 执行步骤 | 0 / 0 | 1,056 / 1,478 | 0 / 0 |
| 字段匹配打分次数 | 388,130 | 62,330 | 不适用 |
| 教师强制目标token数 | 451,203 | 402,321 | 不使用重评分 |
| 教师强制输入token数 | 230,273 | 205,185 | 不使用重评分 |

全程序执行次数包含为评估构建缓存时对所有候选的执行，不等于线上first-valid提前停止后的调用数；前缀调用不能与完整程序调用简单视作相同工作量。local减少了全局字段遍历及新增候选，但引入局部前缀计算，故需综合报告成本。

beam4返回候选的token总数为348,806，beam8为694,473；两者输入问题token总数均44,848。这是返回候选计数，不包括搜索期间所有未保留beam的计算，也不等价于FLOPs。生成各由两个1,000题分片完成：beam4分片耗时52.92/52.41秒，beam8为66.53/69.08秒。

本次global/local的CPU修复与执行阶段分别149.92/152.66秒；额外GPU教师强制评分分别13.68/12.26秒。beam8全候选执行阶段113.31秒。这些分段计时仅记录本次运行，既不能直接相加当作并行作业总墙钟时间，也不能与其他方法的某一个阶段横比后得出效率优势。

来源为各`holdout_*_part*.jsonl.meta.json`、[global修复meta](../results/condition_consistency/query_repair/holdout_global.jsonl.meta.json)、[local修复meta](../results/condition_consistency/query_repair/holdout_local.jsonl.meta.json)、[global评分meta](../results/condition_consistency/query_repair/holdout_global_scored.jsonl.meta.json)、[local评分meta](../results/condition_consistency/query_repair/holdout_local_scored.jsonl.meta.json)及[beam8执行meta](../results/condition_consistency/query_repair/holdout_beam8_executed.jsonl.meta.json)。

## 6. v2完整官方validation结果

由v1事后错误启发，v2增加一个保护规则：如果原first-valid已经执行合法且schema干净，就直接保留；否则沿用原修复评分与接受策略。global/local应用同一规则，并仅在原500题calibration上重新选择同一73项网格的参数。两者最终均选择`alpha=0.5, min_match=0.0, margin=-0.25`；策略在完整val评分前冻结，未依据val结果调参。

**旧2,000题已用于提出v2假设，因此后续只作已见诊断，不能再作为独立有效性证据。** official val的11,797题完整保留，生成与修复只接收问题；源文件、模型权重、执行器、代码、缓存及策略均进行身份核验。official val是公开validation，旧项目曾使用它，不能称为hidden test或项目从未见过的数据。

同一份候选与执行缓存同时应用已冻结的v1/v2选择策略。下面的v1数值是在完整official val上的对照，不覆盖第3节的首次2,000题历史结果。

| 方法 | 正确 / 11,797 | 准确率 | 相对beam4纠正 / 改错 |
|---|---:|---:|---:|
| BART top1 | 8,662 | 73.4254% | 不作为修复增量对照 |
| beam4 first-valid | 9,426 | 79.9017% | 基准 |
| beam8 first-valid | **9,607** | **81.4360%** | 273 / 92 |
| v1 global repair | 9,459 | 80.1814% | 48 / 15 |
| v1 local repair | 9,534 | 80.8172% | 173 / 65 |
| v2 global repair | 9,585 | 81.2495% | 160 / 1 |
| v2 local repair | 9,602 | 81.3936% | 178 / 2 |
| global候选答案覆盖上限 | 9,993 | 84.7080% | 不可部署的金标诊断 |
| local候选答案覆盖上限 | 10,028 | 85.0047% | 不可部署的金标诊断 |

配对区间仍为2,000次问题级bootstrap，种子20260926，单位为百分点：

| 比较 | 差值 | 95%配对区间 |
|---|---:|---:|
| beam8 − beam4 | +1.5343 | [+1.2206, +1.8564] |
| v2 global − beam4 | +1.3478 | [+1.1444, +1.5682] |
| v2 local − beam4 | +1.4919 | [+1.2715, +1.7123] |
| v2 local − v1 local | +0.5764 | [+0.3899, +0.7714] |
| v2 local − v2 global | +0.1441 | [0.0000, +0.2882] |
| v2 local − beam8 | −0.0424 | [−0.4238, +0.3306] |

v2 local相对beam4净增加176题，且只改错2个原本正确答案。beam8净增加181题，准确率仍高5题，但其相对beam4改错92题。这里描述的是不同方法对固定基线的保留/纠正取舍，不意味着v2的所有答案在语义上更可靠。

v2 local相对同样保护的global只净增加17题，区间下界包含0，不能稳健宣称局部约束优势。它与beam8差值区间跨0；“没有确认差异”既不是等效证明，也不是非劣证明，本实验没有预先设定等效/非劣界值。

完整结果、逐题判定和全部对照见[官方val首次评估](../results/condition_consistency/query_repair/v2/official_val_metrics.json)，策略见[v2冻结策略](../results/condition_consistency/query_repair/v2/frozen_policy.json)，数据来源及交叠检查见[official val清单](../results/condition_consistency/query_repair/v2/official_val_manifest.json)。

## 7. v2实际支持的观察与归因边界

保护规则在10,371题上保留schema干净的first-valid，占完整val的87.9122%；其中可能包含原答案错误的问题。规则并不知道这些答案是否正确，它只是按可观测schema/执行条件限制覆盖。其代价是可能放弃已有合法查询上的语义纠正。

在相同完整val上，local相对beam4的改错从v1的65题降至v2的2题，纠正从173题变为178题；global也从48纠正/15改错变为160纠正/1改错。结果支持进一步研究保守接受与误修改控制，但**v2同时加入保护并重新校准阈值，不能将全部变化因果归为单个guard**。同样，不能将global也获得的收益归因于局部事实约束。

v1/v2使用相同字段修复算法、生成器、候选预算与执行器，未训练新模型。这一结果更适合将方法贡献表述为“在固定生成器之上的保守查询修复及可靠性分析”，并把局部约束作为待进一步核验的设计因素。当前未证明它比扩大到beam8更准确，也未证明具有跨种子稳定性或更低总推理开销。

事后核查178个纠正中148个程序精确匹配金标，177题的原四候选均无正确答案。两次改错均来自不同于基线的父候选，其中一例保留了父程序原有的错误人口阈值，说明单字段编辑不能保证符合题意。全部变化的机械诊断及固定案例AI阅读见[v2案例审计](research/QUERY_REPAIR_V2_CASE_AUDIT.md)，仍需独立人工语义核验。

## 8. 完整val的计算记录

v1/v2共享本次完整val候选执行和重评分缓存。保护只发生在最终选择阶段，所有预算内候选仍已生成、执行并评分，因此**当前v2没有通过guard实际节省这些计算**；不能用保留的10,371题推算已经省去的GPU或执行器开销。

| 完整val计数 | global | local | beam8 |
|---|---:|---:|---:|
| 原生成候选数 | 47,188 | 47,188 | 94,376 |
| 新修复候选数 | 12,138 | 6,199 | 0 |
| 有新增候选的问题数 | 3,344 | 2,044 | 不适用 |
| 全程序执行次数 | 59,326 | 53,387 | 94,376 |
| 额外局部前缀调用 / 步骤 | 0 / 0 | 5,731 / 7,870 | 0 / 0 |
| 字段匹配打分次数 | 2,117,109 | 338,145 | 不适用 |
| 教师强制目标token数 | 2,652,042 | 2,364,869 | 不使用重评分 |
| 教师强制输入token数 | 1,346,130 | 1,198,309 | 不使用重评分 |

beam4/beam8返回候选token数分别为2,078,335和4,129,547，输入问题token数均264,938；仍不包括全部未保留搜索分支的计算。local减少了匹配范围和新增候选，同时增加前缀查询；这是工作量构成变化，不能直接等同于总耗时优势。

四GPU生成完整beam4与beam8的合并阶段墙钟时间为390.95秒。12个CPU分片任务的整个并行执行阶段为262.84秒，global/local/beam8合并meta记录的是同一段时间，不能当成三种方法各自的独立延迟。global/local各自所有CPU分片耗时之和为998.60/1,000.49秒，beam8为1,028.65秒；这些和不是墙钟时间。

随后两个GPU评分任务并行运行，global/local记录75.25/68.80秒；包含合并和评分的整个`execute`阶段墙钟时间为353.26秒。没有进行相同负载下的串行端到端延迟比较，本报告不宣称修复比beam8更快。

来源：[生成阶段](../results/condition_consistency/query_repair/v2/generate_stage.json)、[执行阶段](../results/condition_consistency/query_repair/v2/execute_stage.json)、[global执行meta](../results/condition_consistency/query_repair/v2/val_global.jsonl.meta.json)、[local执行meta](../results/condition_consistency/query_repair/v2/val_local.jsonl.meta.json)、[beam8执行meta](../results/condition_consistency/query_repair/v2/val_beam8_executed.jsonl.meta.json)、[global评分meta](../results/condition_consistency/query_repair/v2/val_global_scored.jsonl.meta.json)和[local评分meta](../results/condition_consistency/query_repair/v2/val_local_scored.jsonl.meta.json)。

## 9. 本轮结束与仍需补足的证据

完整official val首次计分完成后，本轮已停止调参，未根据其错误追加规则或修改阈值。v1首次holdout和v2首次公开validation结果均保留，失败对照不删。

所有阶段指标、冻结文件哈希和完整val成本集中在[机器可读汇总](../results/condition_consistency/query_repair/summary.json)；复现入口见[运行说明](research/QUERY_REPAIR_RUNBOOK.md)。本轮任务已结束，四张GPU均无实验进程占用；模型与原始缓存保留在授权主机，公开仓库仅保存代码、指标和来源记录。

目前只有一个BART生成器训练种子。问题级bootstrap描述这一个固定模型的题目变化，不能证明重新训练后的稳定收益。v2假设受旧2,000题结果启发，official val也有旧项目访问历史，这些开发过程必须在论文披露。BF16、硬件、内核和批次/填充形状可能造成微小浮点差异，跨环境复现须记录差异，不应声称逐数值完全一致。

人工语义抽查、多生成器种子、雷达数据和中文查询适配均尚未完成。当前不能主张已改善单位/条件推理或证明雷达迁移有效。若后续继续，应以保守修复的收益与误修改取舍为可检验问题，另冻结稳定性验证协议；局部约束是否有必要保留，须由更完整对照决定。
