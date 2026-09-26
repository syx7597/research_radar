# 单字段查询修复的覆盖审计

日期：2026-09-26。对象为 `pilot_bart5k_seed20260926` 的全部 500 条开发题及固定四候选缓存。本报告由 [audit_query_repair.py](../../scripts/audit_query_repair.py) 机械生成诊断后汇总，**不是人工语义标注，不是新增模型成绩，也不是修复算法的效果验证**。它补充此前的 [错误审计](MINIMAL_ERROR_AUDIT.md)。

## 结论与边界

first_valid 正确 400 / 500。剩余 100 题中，48 题选到了可执行但答错的程序，52 题四个候选全部执行失败。后者不存在 first_valid 程序，以下取 top1 **仅作失败诊断参考**。

- 对参考程序，恰好修改一个字段即可与金标程序完全一致的有 **20 题**；其中属性、关系、限定符合计 **18 题**，概念字段 2 题。
- 20 题中，原字段在对应角色的全局 schema 中不存在的只有 **9 题**，其中核心三类字段 **7 题**。其余字段本身合法，不能仅凭合法性检查定位。
- 若允许从四个原始候选中任取一个错误程序修改，单字段到金标的覆盖为 **30 题**，核心三类 **28 题**；同时满足原字段全局非法的覆盖为 **17 题**，核心三类 **15 题**。这 15 题原候选池均无正确答案。
- 因而，可以继续开展限定预算的字段修复验证，但不适合预设较大提升。应从四候选中提出修复，并保留普通全局字段匹配对照，实际检查局部约束是否有增量。

上述“到金标”需要读取金标程序并重放，因此是**离线的参考程序覆盖诊断**，不是可以实现的准确率，也不是所有语义正确修复的全局上界。可能存在与金标结构不同、但答案正确的程序；反之，答案相同也不证明语义等价。不得把金标替换写入实际修复算法。

## 可复现定义

运行：

```bash
python3 -B scripts/audit_query_repair.py
```

默认只读取固定 dev 缓存、固定三份定向排序器评测、原始 KB 和原版 KQA Pro 执行器；不读取新留出集，不训练、不生成新候选、不修改已有候选顺序。脚本重新计算 first_valid 和答案正确性，与冻结评测逐题核对，并检查缓存 SHA256、KB 和后端一致。对剩余错误和排序器回退涉及的金标做独立执行重放，每程序上限 5 秒。

“单字段”同时满足：函数序列相同、依赖相同、所有其他参数完全相同、恰好一个 schema 字符串不同、原候选答案错误、金标重放答案正确。字段角色包括 attribute / relation / qualifier / concept，其中“核心三类”排除 `FilterConcept` 的概念字段。`Find` 实体、方向、比较符、数字和条件值单独归类，不混入字段修复。

推理可见触发仅检查：**字段字符串是否存在于该参数对应角色的全局 KB schema**。这一判断不使用问题金标、答案、空集或计数 0，也不把“当前实体上没有这个字段”当作异常。局部不存在记录可对应合法空答案，不能据此偏好非空查询。值类型清单随 schema 发布，但本轮没有把类型判断算作已验证触发。

所有剩余题都保留在统计分母内。100 题金标重放有 99 题答对，`train:40416` 重放得到 `None`，而数据标签为 `Batman & Robin`，与先前基础设施检查一致。它不是修复目标的成功样本，亦未从总体错误分母剔除。

## 全部 100 个错误的互斥机械分类

优先比较结构，再比较参数；“结构差异”仅指函数序列或依赖不同，不等于已经证明该结构在语义上错误。

| 与金标的差异 | 有 first_valid 输出 | 无有效候选，以 top1 诊断 | 合计 |
|---|---:|---:|---:|
| 函数序列或结构不同 | 17 | 23 | 40 |
| 无法解析为合法程序 | 0 | 5 | 5 |
| 恰好一个字段不同 | 15 | 5 | 20 |
| 多个字段不同 | 2 | 3 | 5 |
| 恰好一个实体不同 | 1 | 5 | 6 |
| 多个实体或实体次序不同 | 2 | 2 | 4 |
| 恰好一个关系方向不同 | 2 | 0 | 2 |
| 一个含数字、日期或标识符的值不同 | 3 | 3 | 6 |
| 一个其他字面值不同 | 1 | 0 | 1 |
| 多类参数同时不同 | 5 | 6 | 11 |
| **合计** | **48** | **52** | **100** |

数字/日期/标识符合并是机械分类，**不能称作 6 个数值推理错误**。例如题面与金标年份冲突、标识符少字符、单位字符串变化都可能落入该类，需要后续独立核查。

## 20 个单字段案例完整清单

下面不挑选成功示例，而是列出满足定义的全部 20 题。替换箭头仅描述离线与金标的差异，不代表算法已找到替换。

| ID | 字段角色与差异 | 参考程序 | 原字段全局非法 | 原四候选有正确答案 |
|---|---|---|---|---|
| train:74006 | 属性 `discontinue` → `discontinued date` | first_valid | 是 | 否 |
| train:1668 | 关系 `winner` → `award received` | first_valid | 否 | 否 |
| train:76264 | 关系 `twin administrative body` → `twinned administrative body` | first_valid | 是 | 否 |
| train:30739 | 属性 `edition number` → `point in time` | first_valid | 否 | 否 |
| train:5086 | 关系 `followed by` → `follows` | first_valid | 否 | 是 |
| train:5281 | 属性 `Erdős value` → `Erdős number` | 失败 top1 | 是 | 否 |
| train:87706 | 关系 `award received` → `winner` | first_valid | 否 | 是 |
| train:73915 | 关系 `participant` → `participant of` | first_valid | 否 | 是 |
| train:37226 | 关系 `language used` → `languages spoken, written or signed` | first_valid | 否 | 否 |
| train:93430 | 关系 `based on` → `characters` | first_valid | 否 | 否 |
| train:9442 | 关系 `founded by` → `found` | first_valid | 否 | 是 |
| train:83510 | 关系 `located at the administrative territorial entity` → `country` | 失败 top1 | 是 | 否 |
| train:9382 | 关系 `cast member` → `voice actor` | first_valid | 否 | 否 |
| train:33771 | 概念 `military conflict` → `war` | first_valid | 是 | 否 |
| train:87675 | 概念 `EDM` → `electronic dance music` | first_valid | 是 | 是 |
| train:42943 | 关系 `distribution format` → `distribution` | first_valid | 是 | 否 |
| train:73789 | 关系 `language of work or name` → `languages spoken, written or signed` | 失败 top1 | 否 | 否 |
| train:36499 | 属性 `official website` → `full work available at` | 失败 top1 | 否 | 否 |
| train:76902 | 属性 `legal age` → `marriageable age` | 失败 top1 | 是 | 否 |
| train:30041 | 限定符 `valid in time` → `valid in period` | first_valid | 是 | 否 |

这里包含不同难度：`twin` 与 `twinned` 接近词形修正；`winner` 与 `award received` 都合法，涉及角色语义；`located at ...` 与 `country` 名称差异很大，单靠字符串相似度未必能找到。候选存在不等于能安全识别正确候选，也不能事后据此手写这些题的专用别名。

## 原池中已有正确答案的 12 题

这些题体现排序空间，与“新增正确候选”的收益应分开。五题仅差一个字段，只有 `train:87675` 的概念字段全局非法；四个核心字段案例都不能被全局非法字段触发直接覆盖。

| 类型 | 数量 | 全部 ID |
|---|---:|---|
| 单字段 | 5 | train:5086、train:87706、train:73915、train:9442、train:87675 |
| 单关系方向 | 2 | train:38871、train:67548 |
| 多实体或实体次序 | 1 | train:11247 |
| 函数序列或结构 | 3 | train:82310、train:39013、train:76510 |
| 实体次序与关系字段混合 | 1 | train:93374 |

## 定向排序器新增的 10 个错误

定向排序器三个训练种子 17 / 29 / 43 的回退集合完全一致。以下比较它选中的错误候选与原本答对的 first_valid，不能将 first_valid 当作语义金标。

| ID | 错误候选相对 first_valid 的变化 | 错误候选有全局非法字段 |
|---|---|---|
| train:78288 | 函数序列改变，限定值查询变为验证操作 | 否 |
| train:16356 | 限定符 `statement is subject of` → `for work` | 否 |
| train:52489 | 概念 `language` → `languages` | 是 |
| train:32935 | 概念 `state of Germany` → `states of Germany` | 是 |
| train:82094 | 末尾 `QueryAttr` → `Count` | 否 |
| train:66954 | 概念 `college` → `colleges` | 是 |
| train:5451 | 属性 `visitors per year` → `visitor per year` | 是 |
| train:70267 | 限定值查询被改成另一段分支与交集程序 | 否 |
| train:28629 | 概念 `science` → `sciences` | 是 |
| train:30674 | 结构增加关系遍历，另出现 `class of awards` | 是 |

其中 6 题错误候选含非法 schema 字段，但这只是说明合法性信息可能帮助避免部分排序回退。首版修复若采用原 first_valid 作为基线，本来就没有这 10 个排序器回退，不能把“恢复它们”再次算作相对 first_valid 的收益。

## 触发安全性与下一轮验证要求

在全部 500 题的参考程序上，全局角色非法检查触发 **31 题**：29 题 first_valid 基线错误，另 **2 题基线答案正确**。两题均正确回答 0：`train:74975` 含不存在的概念 `organism group`，`train:63399` 含不存在的关系 `interest`。因此“检测到非法字段”与“最终答案错误”不是等价关系，修复仍可能改坏原本正确的答案。

下一轮应固定相同生成器和候选预算，比较全局角色字段修复与局部事实限制修复；分别记录触发、提出候选、候选含正确答案、最终接受、纠正和改错。对本轮案例只能用作开发诊断，正式效果须在未查看的新留出集验证。不能根据“哪个替换会得到非空结果”选择候选，也不能利用测试金标确定哪一段前缀可信。

## 产物与来源

- [summary.json](../../results/condition_consistency/query_repair_audit/summary.json)：汇总、分组 ID、完整输入与脚本 SHA256、执行器版本。
- [schema.json](../../results/condition_consistency/query_repair_audit/schema.json)：从原 KB 提取的 629 个属性、363 个关系、275 个限定符、791 个概念及值类型清单。只含 KB 字段，没有从题目金标归纳别名。
- [regressions.json](../../results/condition_consistency/query_repair_audit/regressions.json)：三个种子的全部新增错误和差异摘要。
- `details.jsonl`、`triggers.jsonl`：逐题完整机械诊断，仅保留本地，依现有规则不提交公开仓库。

候选缓存 SHA256：`502c449e4c8c902e544781c1fdd1ef1e0e30d4b136547c0675c65ab3f3990ecf`。KB SHA256：`04da7408320c5cb7023c44372cce32846d56d369d8865d2e61a18c3956661a7c`。执行器为原版 KQA Pro Baselines，版本 `14d87cd22eb79f702fd4ad5c09240bef126d9dce`。数据镜像、原作者、许可及验证边界见 [DATA.md](../../experiments/condition_consistency/DATA.md)；公开 schema 是该 KB 的派生摘要，沿用原数据来源与许可说明。
