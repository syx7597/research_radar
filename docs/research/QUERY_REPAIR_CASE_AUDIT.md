# 查询修复结果案例审计

对象：已冻结评分的 `holdout`，共 2000 题。此报告由脚本机械核对全部纠正和改错案例，**不是人工语义评审，也不是新一轮算法实验**。没有改动方法、阈值、标签或选取评测子集。

## 整体转换与程序匹配

first_valid 正确 1585 题，局部修复正确 1603 题；纠正 38 题、改错 20 题，净变化 +18 题。共接受 85 次修复，其中 27 次未改变答案正确与否。

纠正案例中，34 个被选程序与规范化金标程序完全相同；38 个案例的原四候选均无正确答案。程序匹配仅比较函数、依赖和参数，不做语义等价证明。

所有纠正和改错均实际核验了父程序到修复程序的单字段限制。基线是否干净、父候选相对顺序等汇总见结构化结果的 `transition_profile`。

**答案相同不保证程序语义正确；程序与金标不同也不必然错误。** 数据金标可能有问题。本报告不从少量成功案例推导整个方法的语义可靠性，仍需独立人工抽查。单字段限制是相对于修复的父候选；父候选未必等于 first_valid，因此最终输出与基线的差异可能不止一个字段。

## 补充阅读与机制诊断

本节是 AI 对题目、所选程序和金标的事后对照阅读，**不是独立人工标注**。它补充脚本的机械统计，没有修改 v1 结果或执行新的接受规则。

38 个纠正案例中，19 个原列表没有可执行输出，13 个基线有 schema 异常，6 个基线 schema 干净。20 个改错案例全部来自其他父候选：19 个父候选排名更后，1 个排名更前；19 个原基线 schema 干净，18 个原基线程序与金标完全一致。这说明单个父程序的字段修复得到合法查询后，仍可能不该替换原本可靠的 first_valid。局部上下文合法不等于问题语义正确。

四个“答对但程序不等于金标”的纠正案例分别是：

| ID | 仍存在的程序差异 | 可以主张的结论与限制 |
|---|---|---|
| train:76154 | `shares border with` 的 forward / backward 方向不同 | 当前答案一致；可能是对称邻接对应，未证明不同方向普遍语义等价。 |
| train:4361 | 修复 `male demonym` 为 `demonym`，仍缺少金标 `QFilterStr(applies to part, male)` | 答案 French 正确，但没有恢复男性限定条件，不能称为完整的条件语义修复。 |
| train:80626 | `twinned administrative body` 的方向不同 | 当前路径可同答；不能将此推广为关系方向无关紧要。 |
| train:24624 | 修复结果为 `found / backward`，金标为 `founded by / forward` | 可能是逆关系的另一种正确表达，不能仅因非精确金标而判错；仍需核查实际中间实体集合。 |

20 个回退也暴露出几个具体限制：`train:60897` 把 `place of birth` 当成属性修为 `date of birth`，地点答案变成日期；`train:8758` 把 `date of marriage` 修为 `date of birth`；`train:57395` 把 `region` 修为 `duration`，国家答案变成时长。字段角色和局部可用性不足以保证问题所需的答案类型。

`train:11529` 的基线本身包含非法字段却碰巧答对。被选修复将 `military personnel` 改为 `employees`，保留数值字符串 `29700`，而金标的值为 `29700 active duty military personnel`。结果由 62 改为 81。**只保留数字和程序结构，不能保证保留原错误字段里隐含的单位或修饰条件**；这与前述 `male demonym` 案例属于同一类语义保护缺口。

下一轮可验证“已有干净 first_valid 时更严格限制其他父候选覆盖”的接受机制；优先保护已有输出，把修复用于无输出或有独立异常证据的基线。旧开发与校准集各有一个相同回退现象，且其纠正均未发生在干净基线上，因此该机制值得小范围开发验证。这里没有给出新规则在当前 2000 题上的重算成绩；新规则须仅在旧开发/校准数据确定，当前 2000 题已见，不能再用于独立有效性证明。

## 全部纠正与改错案例

只公开 ID、必要字段片段和机械诊断，不公开完整题目或程序。表中“精确金标”是规范依赖后完整程序一致，“原池正确”表示原四候选已含正确答案。

| ID | 转换 | 编辑字段 | 触发 | 父候选为基线 | 基线schema干净 | 精确金标 | 原池正确 |
|---|---|---|---|---|---|---|---|
| train:50630 | 改错 | relation / Relate: `contains municipalities` → `contains administrative territorial entity` | 角色非法 | 否 | 是 | 否 | 是 |
| train:4943 | 改错 | relation / Relate: `causes of death` → `manner of death` | 角色非法 | 否 | 是 | 否 | 是 |
| train:79763 | 纠正 | qualifier / QueryAttrQualifier: `earliest` → `earliest date` | 角色非法 | 是 | 否 | 是 | 否 |
| train:66916 | 改错 | relation / Relate: `contains` → `country` | 角色非法 | 否 | 是 | 否 | 是 |
| train:77012 | 改错 | attribute / FilterStr: `short film title` → `short name` | 角色非法 | 否 | 是 | 否 | 是 |
| train:84853 | 改错 | attribute / FilterNum: `offset` → `cost` | 角色非法 | 否 | 是 | 否 | 是 |
| train:46977 | 改错 | relation / Relate: `instrument used` → `influenced by` | 角色非法 | 否 | 是 | 否 | 是 |
| train:46646 | 纠正 | relation / Relate: `terrain feature` → `located on terrain feature` | 角色非法 | 是 | 否 | 是 | 否 |
| train:35686 | 纠正 | attribute / QueryAttr: `FIPS 10-4 (countries and regions) code` → `FIPS 10-4 (countries and regions)` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:46717 | 改错 | attribute / FilterYear: `establishment` → `start time` | 角色非法 | 否 | 是 | 否 | 是 |
| train:12346 | 纠正 | relation / Relate: `jurisdiction` → `applies to jurisdiction` | 角色非法 | 是 | 否 | 是 | 否 |
| train:32030 | 纠正 | attribute / FilterStr: `Dewey Decimal number` → `Dewey Decimal Classification` | 角色非法 | 是 | 否 | 是 | 否 |
| train:62902 | 纠正 | qualifier / QueryRelationQualifier: `statement has part` → `statement is subject of` | 角色非法 | 否 | 是 | 是 | 否 |
| train:6605 | 改错 | relation / Relate: `country for origin` → `country of citizenship` | 角色非法 | 否 | 是 | 否 | 是 |
| train:67483 | 纠正 | attribute / QueryAttrQualifier: `reserve` → `total reserves` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:60897 | 改错 | attribute / QueryAttr: `place of birth` → `date of birth` | 角色非法 | 否 | 是 | 否 | 是 |
| train:11595 | 纠正 | attribute / FilterStr: `soundex` → `Soundex` | 角色非法 | 是 | 否 | 是 | 否 |
| train:76401 | 纠正 | qualifier / QueryRelationQualifier: `terrain feature` → `located on terrain feature` | 角色非法 | 是 | 否 | 是 | 否 |
| train:70151 | 纠正 | attribute / FilterStr: `Nintendo Game ID` → `Nintendo GameID` | 角色非法 | 否 | 否 | 是 | 否 |
| train:12586 | 纠正 | relation / Relate: `famous work` → `notable work` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:8739 | 纠正 | relation / Relate: `discipline` → `award disciplines or subjects` | 角色非法 | 否 | 是 | 是 | 否 |
| train:20508 | 纠正 | qualifier / QueryRelationQualifier: `subject of` → `statement is subject of` | 角色非法 | 否 | 是 | 是 | 否 |
| train:38440 | 改错 | attribute / QueryAttr: `spouse name` → `birth name` | 角色非法 | 否 | 是 | 否 | 是 |
| train:8599 | 纠正 | attribute / FilterStr: `IPTC Newcode` → `IPTC Newscode` | 角色非法 | 是 | 否 | 是 | 否 |
| train:60684 | 纠正 | relation / Relate: `ideology` → `political ideology` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:76154 | 纠正 | attribute / QueryAttr: `FIPS 10-4 (countries and regions) code` → `FIPS 10-4 (countries and regions)` | 角色非法 | 否 | 无输出 | 否 | 否 |
| train:64755 | 纠正 | relation / Relate: `languages spoken` → `languages spoken, written or signed` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:91834 | 纠正 | attribute / QueryAttr: `Hornbostel-Sachs classification code` → `Hornbostel-Sachs classification` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:14226 | 纠正 | attribute / FilterStr: `FIPS 6-4` → `FIPS 6-4 (US counties)` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:75809 | 纠正 | attribute / FilterStr: `name in Japanese` → `name in kana` | 角色非法 | 否 | 是 | 是 | 否 |
| train:18224 | 纠正 | qualifier / QueryRelationQualifier: `role` → `object has role` | 角色非法 | 是 | 否 | 是 | 否 |
| train:60155 | 改错 | relation / Relate: `authored by` → `after a work by` | 角色非法 | 否 | 是 | 否 | 是 |
| train:48560 | 改错 | relation / Relate: `country for work` → `country for sport` | 角色非法 | 否 | 是 | 否 | 是 |
| train:18688 | 纠正 | qualifier / QueryAttrQualifier: `relation type` → `mapping relation type` | 角色非法 | 是 | 否 | 是 | 否 |
| train:67448 | 纠正 | qualifier / QueryRelationQualifier: `valid in time zone` → `valid in period` | 角色非法 | 否 | 是 | 是 | 否 |
| train:4361 | 纠正 | attribute / FilterStr: `male demonym` → `demonym` | 角色非法 | 否 | 无输出 | 否 | 否 |
| train:46669 | 纠正 | relation / Relate: `work` → `notable work` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:75742 | 纠正 | attribute / SelectAmong: `death number` → `number of deaths` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:86630 | 纠正 | attribute / FilterStr: `local area code` → `local dialing code` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:28878 | 纠正 | relation / Relate: `film director` → `film editor` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:8758 | 改错 | attribute / QueryAttr: `date of marriage` → `date of birth` | 角色非法 | 否 | 是 | 否 | 是 |
| train:11529 | 改错 | attribute / FilterNum: `military personnel` → `employees` | 角色非法 | 否 | 否 | 否 | 是 |
| train:38685 | 纠正 | qualifier / QueryAttrQualifier: `donation method` → `donated by` | 角色非法 | 否 | 是 | 是 | 否 |
| train:7285 | 纠正 | relation / Relate: `distribution region` → `film distribute region` | 角色非法 | 是 | 否 | 是 | 否 |
| train:55451 | 改错 | qualifier / QueryRelationQualifier: `statement has part` → `statement is subject of` | 角色非法 | 否 | 是 | 否 | 是 |
| train:47796 | 改错 | attribute / SelectBetween: `unemployed population` → `population` | 角色非法 | 否 | 是 | 否 | 是 |
| train:24046 | 纠正 | attribute / QueryAttr: `named after` → `native label` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:57519 | 改错 | relation / Relate: `location of event` → `filming location` | 角色非法 | 否 | 是 | 否 | 是 |
| train:58414 | 纠正 | relation / Relate: `site of formation` → `location` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:69611 | 纠正 | relation / Relate: `located on continent` → `continent` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:57395 | 改错 | attribute / QueryAttr: `region` → `duration` | 角色非法 | 否 | 是 | 否 | 是 |
| train:24647 | 纠正 | attribute / FilterStr: `fleet/registration number` → `fleet or registration number` | 角色非法 | 否 | 无输出 | 是 | 否 |
| train:80626 | 纠正 | relation / Relate: `twin administrative body` → `twinned administrative body` | 角色非法 | 否 | 无输出 | 否 | 否 |
| train:67123 | 纠正 | attribute / FilterNum: `frequency` → `frequency of event` | 角色非法 | 是 | 否 | 是 | 否 |
| train:24624 | 纠正 | relation / Relate: `founders` → `found` | 角色非法 | 是 | 否 | 否 | 否 |
| train:11377 | 改错 | relation / Relate: `work` → `after a work by` | 角色非法 | 否 | 是 | 否 | 是 |
| train:68239 | 改错 | relation / Relate: `country of formation` → `location of formation` | 角色非法 | 否 | 是 | 否 | 是 |
| train:39306 | 纠正 | attribute / QueryAttr: `website` → `official website` | 角色非法 | 否 | 无输出 | 是 | 否 |

## 按金标函数划分的事后诊断

以下类别相互重叠，只用于解释结果，不能相加，也不是预注册的子群有效性结论。数值组包括数量过滤/验证及极值选择，时间组单独列出；限定符组可与两者重叠。

| 子集 | 题数 | first_valid 正确 | 全局修复正确 | 局部修复正确 | beam8 正确 | 局部纠正 / 改错 |
|---|---:|---:|---:|---:|---:|---:|
| 限定符 | 457 | 301 | 303 | 310 | 309 | 13 / 4 |
| 数值或极值选择 | 525 | 449 | 449 | 450 | 454 | 4 / 3 |
| 时间条件 | 265 | 216 | 217 | 218 | 221 | 3 / 1 |
| 集合操作 | 536 | 389 | 393 | 389 | 404 | 10 / 10 |
| 计数 | 237 | 154 | 155 | 154 | 156 | 11 / 11 |
| 至少两步 Relate | 176 | 110 | 112 | 112 | 114 | 4 / 2 |

## 旧开发与校准数据上的同类现象

下表仍使用已经冻结的 v1 选择策略，只统计其行为，**没有运行新的基线保护规则，也没有重新调参**。机制建议已受到当前检查集的观察启发，当前检查集此后必须标记为已见。

| 集合 | first_valid / 局部正确 | 纠正 / 改错 | 纠正中基线干净 | 改错中基线干净 |
|---|---|---|---|---|
| dev_diagnostic | 401 / 411 | 11 / 1 | 0 | 1 |
| calibration | 395 / 400 | 6 / 1 | 0 | 1 |

## 复现与来源

脚本：[audit_repair_outcomes.py](../../scripts/audit_repair_outcomes.py)。结构化结果：[outcome_audit.json](../../results/condition_consistency/query_repair/outcome_audit.json)。脚本先核对冻结评测、缓存和金标 SHA256/ID/题目，再重算已选答案正确性；不重新选择候选。

完整输入及脚本哈希、逐例编辑核查和子集定义见结构化结果。原数据来源和许可见 [DATA.md](../../experiments/condition_consistency/DATA.md)。
