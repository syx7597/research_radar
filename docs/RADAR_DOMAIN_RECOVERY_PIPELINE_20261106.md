# 域内成功轨迹与恢复配对构造

日期：2026-11-06。新增 [recovery_data.py](../experiments/radar_domain/recovery_data.py)，基于域集合接口构造成功轨迹与普通／恢复配对；原公共模块和旧实验输出保持不变。CPU 记录见 [summary.json](../results/radar_domain/recovery_pipeline_20261106/summary.json)。

## 输入与答案契约

参考记录严格包含四个字段：id、question、actions、answer。actions 是由 name 和 arguments 组成的工具调用列表，最后一个动作明确使用 finish 或 finish_records；前面的 step 均需成功执行。answer 保存实际 RecordSetEpisode.answer 的完整对象。

自由轨迹至少包含 id、messages、events、calls、answer。messages 以域问题提示开始，随后是实际助手动作和工具观察；允许成功结束后附加 Done.。构造器重新执行所有动作，逐项核对工具观察、事件、调用数和实际答案；观测或错误消息被改写的轨迹会被排除。

答案比较保留全部记录、来源映射、量值、单位和条件。记录集合、主体集合、条件合取及引用片段的顺序可以规范化；其他内容和字段类型保持严格。input_entity_count 属于执行诊断，不影响相同记录集合的语义相等性；unique_record_count 必须与完整记录列表一致。未提交答案、空记录集合和标量零分别处理。

## 构造与监督

构造器先验证参考轨迹和自由轨迹。对最终未正确作答的回合，定位与参考一致的共同成功前缀及实际首次分歧。普通组从共同前缀接入参考后缀；恢复组额外保留真实分歧及反馈，再接入同一后缀。

所有后续 step 依赖，以及 finish 的 answer_handle、finish_records 的 record_handle，均通过参考结果到实际句柄的映射更新。被拒绝动作不新增句柄；额外成功动作按实际产生的句柄处理。后缀逐步重放，观察和最终结构化答案须与参考对应，且在同一调用上限内结束。

成功参考轨迹监督全部助手动作和 Done.。配对轨迹仅监督参考后缀中的助手动作和 Done.，问题、共同前缀、实际分歧及所有工具观察均屏蔽。当前匹配的是目标动作及其依赖结构，token 预算检查与匹配单独进行。

## 已完成的 CPU 验证

- 七道开发参考全部执行成功，包含六个集合答案（含空集）和一个 Count 零答案。
- 每题分别构造“额外有效句柄”和“被拒绝且不增句柄”的人工测试轨迹，共十四条，并成功构造十四组普通／恢复配对。
- 十一项恢复构造测试覆盖句柄重映射、终结类型、空集／零、完整答案与合法顺序、观测篡改、预算不足、已成功轨迹排除和训练准入。连同接口与桥接测试，三组共三十一项通过。

这些人工轨迹均标记 hand_constructed_cpu_fixture、training_eligible=false，私有文件位于 data/radar_domain_recovery_v1/recovery_cpu_fixtures_20261106/。其中 success.reference.jsonl、fixture.clean.jsonl 和 fixture.recovery.jsonl 均包含 messages 与 supervise，可用于本地分词和掩码检查。

**真实模型 rollout 数为 0，正式模型恢复配对数为 0，正式训练样本数为 0。** 人工轨迹用于检验构造程序的行为。

## 已完成的分词与预算预检

本地仅下载并核对原Qwen版本的两个分词器文件，使用Transformers 5.18.0、tokenizers 0.23.2和域内三工具定义，实际检查七条参考轨迹与二十八条普通／恢复测试轨迹。完整上下文最大为3,702 token，单步监督目标最大34 token，全部满足8,192上下文、192单步及2,048续写目标预算。工具观察仅进入上下文，监督标签只覆盖助手目标；超长输入和对工具观察施加监督均会被拒绝。

十四组构造测试的两侧监督token数相等；正式训练语料的采样和预算匹配仍需在真实数据产生后执行。检查结果见[token_preflight.json](../results/radar_domain/recovery_pipeline_20261106/token_preflight.json)和[配对计数](../results/radar_domain/recovery_pipeline_20261106/fixture_supervision_count_check.json)。本轮没有下载模型权重或调用模型生成。

## 正式训练准入

生产入口要求同时提供参考、rollout、训练划分、模型生成来源清单、完整知识图与来源块。划分使用 radar_domain_split_manifest_v1，含 train/dev/holdout 的 ids 和 group_by_id；训练部分绑定 questions 与 references 的文件路径和哈希，三部分的题号与来源分组均须隔离。

生成清单使用 radar_domain_rollout_manifest_v1，要求 completed、origin=model_generated、split=train、max_calls=24，以及 generation_id、划分哈希、question_input、rollouts（含行数）、model_files_sha256、implementation_files_sha256、kb 和 source_chunks。每条 rollout 的 provenance 必须与清单的模型生成标识一致。题目输入只允许 id 和 question，全量顺序与训练划分一致。人工 fixture 或开发来源不能通过该入口。

当前加载器要求完整的 502 条读法，生产入口也是面向这一图结构的训练准入实现。正式新来源或合成训练库还需要独立完整源清单与相应参数化加载；本轮没有形成这些正式训练材料。后续工作接续正式划分、训练参考、全部题型的组合解析和真实失败采集。

## 暂无服务器时的工作顺序

1. 完成任务与答案规范：明确简单查询、条件筛选、配置比较、集合和空集的评价对象，将完整任务与取证子任务分别登记；进一步区分空结果、资料未声明和执行失败，并确定空答案的检索范围与来源支持要求。
2. 完善类型语义：24条严格界限保留比较符，统一最大值的quantity_role视图；范围、离散选项和配对值按来源关系表达。
3. 建立正式数据划分：历史开发材料与新来源评价分开，按来源家族和共享事实分组，再确定训练、开发与最终评价清单。
4. 扩充并复核成功参考程序，完成结构化评分和分词预算检查；真实失败轨迹由后续I_domain生成。
5. 整理数据构造、错误类型、恢复样本与应用案例，接入论文的方法和数据章节。

2026-11-06用户已要求暂不使用服务器。本地数据、执行器、轨迹和评价准备继续推进。
