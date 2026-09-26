# 最小实验错误审计

审查日期：2026-09-26。对象是 `pilot_bart5k_seed20260926`：5,000 条训练问题、10 个 epoch、每题 4 个实际生成候选，评估 train 内部冻结的 500 条 dev。**这是 AI 对缓存结果的事后语义审查，不是独立人工标注，也不是模型的新一轮实验。** 查阅金标只用于解释错误，未改算法、权重、候选、划分或答案。

## 样本与检查范围

1. 按 `dev_executed.jsonl` 文件原有顺序，取 `dev_metrics.json` 判为 top1 错误的**最前 30 条**，不挑案例；它们分布在文件第 1–92 行。
2. 逐条检查规则相对 top1 的全部 35 条答案修正，以及相对 first_valid 的全部 12 条答案回退。
3. 核对规则特征得分差，区分“执行过滤的收益”和“条件规则的增量”。

缓存位置：`results/condition_consistency/pilot_bart5k_seed20260926/{dev_executed.jsonl,dev_metrics.json}`。原始问题/程序保留本地，本报告只给 ID、差异摘要和必要片段。

核验版本 SHA256：`dev_executed.jsonl` 为 `502c449e4c8c902e544781c1fdd1ef1e0e30d4b136547c0675c65ab3f3990ecf`；`dev_metrics.json` 为 `fbbb450dff7cfd3eb6a00609444f260ee3b8b3a7364417258e0691906070f8e8`。数据来源与许可见 [DATA.md](../../experiments/condition_consistency/DATA.md)。

## 整体结果与可以确认的归因

| 方法 | 答案正确数 / 500 | 准确率 |
|---|---:|---:|
| 生成器 top1 | 365 | 73.0% |
| first_valid：按生成顺序取首个能执行的候选 | 400 | 80.0% |
| 固定条件规则 | 388 | 77.6% |
| oracle@4：四个候选中存在正确答案 | 412 | 82.4% |

first_valid 相对 top1 修正 35 条、改错 0 条。条件规则相对 first_valid 修正 0 条、改错 12 条。**规则相对 top1 的 35 条“修正”全部选择了与 first_valid 完全相同的候选**，因此当前证据只支持执行过滤有效，不能据此声称条件一致性规则有效。

完整缓存中 top1 错题 135 条，其中 97 条执行或解析失败。但“执行失败”并不等于“语法错误”：字段名错误、方向错误、实体无匹配等也会触发运行异常。本次固定 30 条中，21 条 top1 执行失败，只有 1 条主要属于参数格式错误；其余大量是 grounding 或语义结构问题。

这 30 条中，oracle@4 可恢复 10 条，first_valid 实际恢复 8 条。**以下类别计数只描述这 30 条，不外推整个 dev 或 KQA Pro 的错误比例。**

## 前 30 条固定错题的主因分类

每题只给一个主类；多个问题并存时在摘要中说明。类别是 AI 解释性判断，未做双人一致性检验：

- A：schema、关系方向、实体/参数或答案投影错误，20 条。
- B：数字/日期的字符串转换错误，1 条。
- C：遗漏或错误组织条件、事实限定符与关系路径，6 条。
- D：程序参数格式错误，1 条。
- E：问题文本与金标程序存在直接可见的不一致，2 条；不直接归责模型。

| 顺序 | 文件行 | ID | 主类 | 对比 top1 与问题/金标的诊断 |
|---:|---:|---|---|---|
| 1 | 1 | train:4475 | B | March 17 被写为 `1924-04-17`；候选 2 为正确月份。 |
| 2 | 2 | train:74006 | A | 属性 `discontinued date` 被写成问题动词 `discontinue`，返回错误计数 0。 |
| 3 | 5 | train:21332 | A | 使用不存在/不匹配的 `total revenue`，另有关系方向差异；**题面 revenues 与金标 reserves 也不一致**，模型责任不能完全分离。 |
| 4 | 7 | train:75060 | A | `political ideology` 被写成 `ideology`，关系方向也反了。 |
| 5 | 9 | train:1668 | A | 应沿 `award received` 查获奖记录，却选择 `winner`。 |
| 6 | 13 | train:52215 | C | Scooby-Doo 的 `character role` 限定符被遗漏，且交集移动到错误路径阶段；并非仅一处条件错绑。 |
| 7 | 16 | train:76264 | A | `twinned administrative body` 被写成 `twin administrative body`，查询不到限定值。 |
| 8 | 21 | train:82184 | E | 题面写 1935，金标写 1835；题面“started in 1843”而金标还使用 `!=`。top1 复制 1935 不能简单判为数字推理失败。 |
| 9 | 23 | train:32253 | C | 应查询死亡地点事实上的行政区限定符，却改成普通关系遍历，限定符问题结构丢失。 |
| 10 | 32 | train:46629 | C | 应读取网站属性事实的语言限定符，生成器改为“网站过滤→creative endeavor 关系→语言实体”。 |
| 11 | 34 | train:30739 | A | 时间属性选 `start time`，金标为 `point in time`。 |
| 12 | 35 | train:838 | A | `notable work/forward` 写为 `famous people/backward`；候选中存在正确路径。 |
| 13 | 41 | train:94364 | A | `mapping relation type` 少了 `type`，且 risk factor 方向与金标不同。 |
| 14 | 47 | train:78511 | C | 丢失“包含 Ballarat 的 Victoria”路径，直接比较 Ballarat；同时 Zürich 实体大小写也变了。 |
| 15 | 49 | train:29890 | C | 出生地点事实上的 country 限定符变成普通死亡关系，Rome 人口条件亦遗漏。 |
| 16 | 51 | train:80702 | A | 把 `named after` 关系理解为 `name in native language` 属性；另漏金标 English 限定符，但该语言条件未在题面明确出现。 |
| 17 | 62 | train:37266 | A | 将完整实体名 King's Lynn and West Norfolk 拆成两个实体，丢失实际比较对象 West Dorset。 |
| 18 | 64 | train:11247 | A | `QueryRelation` 两个实体次序颠倒，得到空结果；有正确候选，但首个合法候选仍错。 |
| 19 | 66 | train:79319 | A | 要求 birth name，却返回实体常用名 Cher；最后一步答案投影错误。 |
| 20 | 70 | train:52312 | D | `UN/LOCODE` 和 `SMSAI` 少了 `<arg>` 分隔，FilterStr 参数数不符。 |
| 21 | 71 | train:25233 | A | 错误起点 Argentina 替代 Israel、方向改变，另把 GDP 数值改成 1260000000；属于多项混合错误。 |
| 22 | 72 | train:6674 | A | `work period (start)` 被截短为 `work period`。 |
| 23 | 74 | train:19732 | A | 将 native-language 名称直接作 Find 标签，并把 cause of death 关系当普通属性查询。 |
| 24 | 75 | train:5086 | A | `follows/backward` 被写为 `followed by/backward`，关系含义改变；有正确候选但过滤不足以选择它。 |
| 25 | 78 | train:36144 | E | 题面 CANTIC-ID 为 `10450877`，金标为 `a10450877`；模型照抄题面而无结果，需人工核对标注。 |
| 26 | 81 | train:3132 | A | 概念 `estate in land` 分别写为 `land` 和 `estate`，生成了不匹配的 schema 名称。 |
| 27 | 83 | train:86449 | A | inception 被理解为 `point in time`。 |
| 28 | 86 | train:93107 | A | 从作品找作者的 notable work 方向写反。 |
| 29 | 87 | train:35187 | C | 把 `depicted by` 事实上的 `present in work` 限定符改成普通关系，还错换了实体概念。 |
| 30 | 92 | train:75418 | A | 题面的 Holly Marie 未链接至完整实体标签 Holly Marie Combs。 |

C 类确实说明限定符/复杂条件组织值得研究，但这里六条都伴随执行失败或较大路径重写，**不是六条“程序合法、仅事实绑定错误”的纯净证据**。不能将所有结构差异都包装为拟议方法的目标问题。

## 全部 35 条答案修正的核查

逐条确认：原 top1 均执行或解析失败；规则与 first_valid 选择同一候选。以下分组只压缩展示这些病例，不构成新的采样统计。

| 主要变化 | 数量 | 全部 ID |
|---|---:|---|
| 日期月份转换 | 1 | train:4475 |
| 参数分隔或缺失分支修复 | 2 | train:52312、train:14397 |
| 条件/比较方向或限定符组织变化 | 4 | train:9711、train:69537、train:5335、train:61509 |
| schema、方向、实体表示或查询结构变化 | 28 | train:838、train:80702、train:37266、train:6674、train:86449、train:93107、train:93356、train:39543、train:24823、train:46591、train:26446、train:50519、train:78759、train:77、train:82094、train:80733、train:69590、train:15218、train:92115、train:6637、train:14190、train:93062、train:40292、train:14965、train:53223、train:34324、train:92183、train:10946 |

答案修正也不自动等于完整语义修复。例如 `train:93356` 的正确答案为否，但被选候选仍把比较字符串写成 `Austin 01228`，而问题问的是拨号码 `01228`；它可以因错误比较目标而碰巧输出同样的否定答案。类似样本提醒：不能只用“与金标答案相同”证明程序语义等价。

## 全部 12 条规则改错的核查

下表比较的是 first_valid 已正确的候选与规则选中的错误候选。这 12 条原 top1 也均正确。重新计算固定规则特征发现：**每条改错的得分增量都仅来自 `lexical_coverage`；数字覆盖、额外/遗漏数字、比较方向项没有变化。** 规则偏好复制问题表面词形，却缺乏 schema 是否匹配和答案投影是否正确的判别。

| ID | 被错误偏好的变化 | 结果变化 |
|---|---|---|
| train:78288 | 新增题面 Dutch 使字面覆盖更高，但把“查询 end time”改成“VerifyStr 官方名称” | 1978 → yes |
| train:52489 | 合法概念 `language` 改为题面复数 `languages` | 2 → 1 |
| train:32935 | `state of Germany` 改为 `states of Germany` | 2 → 1 |
| train:66954 | `college` 改为 `colleges` | 1 → 0 |
| train:5451 | 合法属性 `visitors per year` 改为题面 `visitor per year` | 7 → 0 |
| train:58839 | 关系 `nominated for` 改为 `nominee` | 24th Academy Awards → 空 |
| train:28629 | `science` 改为 `sciences` | 1 → 0 |
| train:74572 | `spouse` 改为问题表面的 `husband` | 2011-06-22 → 空 |
| train:79727 | 关系 `cast member` 改为题面 `child`，但 child 属于人物的另一个条件 | Arthur Weasley → 空 |
| train:30674 | `class of award` 改为 `class of awards` | 10 → 1 |
| train:14402 | 限定符 `number of subscribers` 改为 `number of followers` | 539 → 空 |
| train:47259 | `award received / statement is subject of` 改为 `winner / for work` | 79th Academy Awards → 空 |

这些不是“合法空答案一定不应选择”的证据：Count 为 0、关系查询为空本来可能合法。问题是本轮规则无法区分知识库字段和自然语言同义词，也没有建模问题所需答案类型。**本次审计没有据此补规则后重新报同一 dev 的成绩。**

另一个答案等价风险是 `train:30674`：原本答对的候选也没有完整保留金标第一分支的获奖关系遍历。应将当前分数称作答案准确率，不能称作全部程序语义正确率。

## 对后续最小实验的含义

- 生成器已足以暴露可研究的真实错误，不必把“继续训练大模型”作为唯一下一步。当前最可靠收益来自已有候选的执行过滤。
- 固定词面覆盖规则失败，不能作为毕业论文的方法增益；应以 first_valid 的 80.0% 为实际下限对照，保留负结果。
- 同一四候选缓存的剩余答案上限为 12 / 500，即 2.4 个百分点。限定符子集 122 题中，first_valid 为 87、oracle 为 89，只有 2 题余量。若只研究限定符重排，当前候选覆盖是实质限制。
- 合理的小改进应先证明对同一候选预算下的真实错误有效，并控制原本正确样本的回退。schema 词形/同义表达与条件角色的区分可能有用，但本轮审计只提出诊断，不预先宣布该方法有效。
- 数据本身存在题面与程序不一致、多个程序偶然同答等问题。正式结论应增加独立人工抽查；这 30 条 AI 分类不能替代人工质量控制，也不能推出整个数据集的缺陷比例。
