# v2 官方验证集修复案例审计

对象：完整 KQA Pro 官方验证集 11797 题，已经完成冻结策略评分。本报告是**事后机械诊断，不是独立人工语义标注，也没有进行新一轮调参**。v2 策略冻结后才运行本次评测，但早期项目曾使用官方 val，不能把它宣称为项目从未接触的隐藏测试集。

## 答案转换与程序匹配

first_valid 正确 9426 题，v2 局部修复正确 9602 题。纠正 178 题、改错 2 题，净变化 +176 题。共接受 297 次修复，117 次未改变答案正确与否。

纠正中的 148 个修复程序与规范化金标完全一致；177 个纠正案例原四候选均无正确答案。所有变化案例均核对了相对父候选的真实单字段修改。

规范化仅统一推断出的依赖与程序结构，不判断语义等价。**答案相同不保证程序语义正确；程序不等于金标也不必然错误。** 单字段保护相对于父候选，父候选可能不同于 first_valid；这不是最终答案只发生单一语义变化的保证。

保护规则保留了 10371 个已有干净基线，逐题确认均未被替换；其中 985 个基线答案错误，说明 schema 干净只是一项保守门控证据。

| 编辑角色 | 纠正 | 改错 |
|---|---:|---:|
| attribute | 74 | 1 |
| relation | 85 | 1 |
| qualifier | 19 | 0 |

## 事后阅读补充：两个回退与同答案风险

本节为 AI 对固定案例的题目、程序和金标进行的补充阅读，不是独立人工标注；没有更改标签、阈值或算法。178 个纠正中，136 个原候选池没有可执行输出，42 个原 first_valid 存在 schema 冲突；没有覆盖任何受保护的干净基线。下列两条回退也都发生在存在 schema 冲突的基线上。

- **val:4284**：基线使用非法关系 `production method`，碰巧得到正确计数 0。被选修复来自排名 3 的父程序，将 `produced by` 改为局部可用关系 `industry`，计数变为 1；金标实际使用 `manufacturer / backward`。局部字段存在不能证明它表达题意，schema 有异常也不保证原答案错误。
- **val:7166**：题目、金标和原 first_valid 的人口阈值是 `11000`，但排名 2 的父程序已经写成 `110000`。修复只替换 FIPS 字段名，因此仍保留了父程序的错误阈值，最终将答案由 1 改为 6。**相对父候选保留数字，不等于相对题目或已有基线保留数字。** 这不违反当前单字段代码约束，但说明其语义保护范围有限。

十个固定纠正示例中，`val:73` 使用 `derivative work / backward`，而金标为 `based on / forward`；修复程序使用概念 `film`，金标使用 `visual artwork`，这个概念差异已存在于父候选，并非本次字段编辑造成。可能存在逆关系或概念包含下的另一条正确路径，不能仅凭非精确匹配判错。`val:449` 仍使用 `< 1793`，金标使用 `!= 1793`，题面则含“not … before 1793”；至少存在题意与比较表达需核查的疑点。`val:449` 的金标与修复程序均得到计数 1，不能据此证明修复恢复了时间条件。

整体上，精确金标匹配的 148 个纠正提供了较直接的程序修复证据；另外 30 个纠正不能一概当作语义错误，也不能一概宣称完整语义正确。独立人工评审、关系等价和中间实体集合核查属于后续工作。本轮按协议停止调参，这些事后发现不用于回调当前官方验证集成绩。

## 全部改错案例

以下列出全部回退，不挑选或省略失败案例。排名从 0 开始，缺少基线显示为“无”。

| ID | 编辑字段 | 触发 | 父排名 / 基线排名 | 基线schema干净 | 修复精确金标 | 基线精确金标 |
|---|---|---|---|---|---|---|
| val:4284 | relation / Relate: `produced by` → `industry` | 角色非法 | 3 / 0 | 否 | 否 | 否 |
| val:7166 | attribute / FilterStr: `FIPS 6-4 (US counties) code` → `FIPS 6-4 (US counties)` | 角色非法 | 2 / 0 | 否 | 否 | 否 |

## 最多十个纠正示例

按属性、关系、限定符轮流取例，每类保持冻结评测中的原顺序，最多十例；不按匹配程度、答案好看与否或人工偏好选取。示例分布不代表总体比例，全部纠正 ID 和机械诊断保存在 JSON。

| ID | 编辑字段 | 触发 | 父排名 / 基线排名 | 基线schema干净 | 修复精确金标 | 基线精确金标 |
|---|---|---|---|---|---|---|
| val:35 | attribute / QueryAttr: `British Museum person-institution designation` → `British Museum person-institution` | 角色非法 | 1 / 无 | 无输出 | 是 | 否 |
| val:59 | relation / Relate: `location of filming` → `location` | 角色非法 | 0 / 无 | 无输出 | 是 | 否 |
| val:682 | qualifier / QueryAttrQualifier: `role` → `object has role` | 角色非法 | 0 / 0 | 否 | 是 | 否 |
| val:208 | attribute / QueryAttr: `GRIN UR` → `GRIN URL` | 角色非法 | 0 / 无 | 无输出 | 是 | 否 |
| val:73 | relation / Relate: `fictional profession` → `derivative work` | 角色非法 | 0 / 无 | 无输出 | 否 | 否 |
| val:694 | qualifier / QFilterYear: `place of filming` → `retrieved` | 角色非法 | 1 / 无 | 无输出 | 是 | 否 |
| val:423 | attribute / FilterStr: `Dewey Decimal Classification system` → `Dewey Decimal Classification` | 角色非法 | 0 / 0 | 否 | 是 | 否 |
| val:115 | relation / Relate: `has properties` → `said to be the same as` | 角色非法 | 1 / 0 | 否 | 是 | 否 |
| val:829 | qualifier / QFilterStr: `statement is based on` → `together with` | 角色非法 | 2 / 无 | 无输出 | 是 | 否 |
| val:449 | attribute / FilterStr: `local dialing area code` → `local dialing code` | 角色非法 | 1 / 0 | 否 | 否 | 否 |

## 金标函数子集的事后统计

类别相互重叠，只用于诊断，不得相加或当作预注册的子群优越性结论。数值组包含数量过滤/验证及极值选择，时间组另列。

| 子集 | 题数 | first_valid 正确 | v2全局正确 | v2局部正确 | beam8 正确 | 局部纠正 / 改错 |
|---|---:|---:|---:|---:|---:|---:|
| 限定符 | 2806 | 1868 | 1913 | 1915 | 1935 | 48 / 1 |
| 数值或极值选择 | 3095 | 2692 | 2729 | 2721 | 2724 | 31 / 2 |
| 时间条件 | 1551 | 1223 | 1235 | 1238 | 1248 | 15 / 0 |
| 集合操作 | 3436 | 2544 | 2592 | 2595 | 2603 | 52 / 1 |
| 计数 | 1318 | 878 | 895 | 900 | 889 | 24 / 2 |
| 至少两步 Relate | 1051 | 619 | 632 | 636 | 641 | 17 / 0 |

## 来源与复现边界

脚本：[audit_repair_v2_outcomes.py](../../scripts/audit_repair_v2_outcomes.py)。结构化结果：[v2/outcome_audit.json](../../results/condition_consistency/query_repair/v2/outcome_audit.json)。

脚本读取最终评分后，先核验原始 metrics、金标、缓存与冻结策略的哈希；仅在内存将 v2 方法名映射到既有审计函数需要的别名。v1/v2 指标文件、金标、阈值及算法文件均未修改。JSON 的 `method_aliases` 明确说明 `local_repair/global_repair` 在本审计中分别代表 v2 局部/全局方法。

所有案例、输入和源码哈希及子集定义见结构化结果。原数据来源与许可见 [DATA.md](../../experiments/condition_consistency/DATA.md)。
