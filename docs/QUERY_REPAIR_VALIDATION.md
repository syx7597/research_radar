# 查询修复验证记录

更新：2026-09-26。v1已完成首次冻结的2,000题检查集评估；v2按[第二轮有界协议](research/QUERY_REPAIR_V2_PROTOCOL.md)执行中，尚未在本文报告其完整官方validation成绩。下文保留v1结果，后续v2不得覆盖这次首次评估。

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

## 6. 第二轮的边界与当前可主张结论

由v1事后错误启发，v2只增加一个保护规则：如果原first-valid已经执行合法且schema干净，就直接保留；否则沿用原修复评分与接受策略。global/local应用同一规则，并只在原500题calibration上重新选择同一73项网格的参数。该规则也可能放弃可纠正的合法错误，效果必须重新验证。

**v2已经受到旧2,000题观察的启发，因此旧2,000题从此只作已见诊断，不能再次作为独立有效性证据。** v1首次holdout结果原样保留；v2在新冻结协议下对全部official val的11,797题评估。official val是公开validation，旧项目曾使用它，不能称为hidden test或项目从未见过的数据。第二轮完整评估一次后停止调参，无论结果是否积极。

本轮只有一个生成器训练种子，尚无多种子稳定性结论，也没有雷达自建数据上的方法有效性证据。当前能主张的是完成了可复现的字段修复实验、获得少量新增正确答案并定位了误修改问题；不能提前主张新算法稳定超过强基线、改善了单位/条件推理，或已证明雷达迁移有效。第二轮完整结果和最终研究决策将在完成后另行补充。
