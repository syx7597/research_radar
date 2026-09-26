# 字段合法性约束基线 C：接口、覆盖检查与下一阶段

日期：2026-09-26。C是已有技术的基础对照，不是本项目的算法创新。本阶段只完成接口和CPU分词器检查，没有加载模型、生成新候选、执行查询或训练；不报告新问答准确率。

## 机制与范围

[schema_constraints.py](../../experiments/condition_consistency/schema_constraints.py)提供Hugging Face的`prefix_allowed_tokens_fn`接口。每次从当前beam的完整前缀恢复函数和参数位置，在关系、属性、限定字段槽位内用角色词典树限制下一个token。词表来自固定KB快照，共关系363、属性629、限定字段275个；不从问题或评价金标补词。

概念、实体、字面值、方向、比较符和函数语法仍不约束。因此它不能解决所有结构错误，也不能在两个合法关系之间判断题意。不能称其完整复现Candidate Expressions；机制来源和研究假设见[方法论证](METHOD_FEASIBILITY_BRIEF.md)。

| 边界 | 实现行为 |
|---|---|
| 合法字段互为前缀 | 短字段完整后同时允许结束和继续生成长字段 |
| 字段尚未完整 | 禁止EOS及参数/函数分隔符 |
| 字段内无合法续写 | 报错停止，不回退为无限制生成 |
| 未知函数、非字段参数 | 保持自由生成；未知函数另记数，最终仍须语法校验 |
| 分词及空格 | 支持原分词器的规范编码，字段两端各0或1个ASCII空格；不声称涵盖任意BPE分解及空白 |
| 特殊token | 函数识别与最终输出采用相同的特殊token删除语义，避免函数名伪装 |
| 用普通BPE碎片拼出分隔符 | 识别到完整非原子分隔符后明确失败；不悄悄跳过参数约束 |
| 最大长度 | 未完成输出必须标记失败；离线重放尚未验证实际生成终止交互 |

字段进入词典树前的其他文本不被强制改写。词典树限制的是名称合法性，不是数值类型、局部可达性、执行非空或答案正确性。

## 实际覆盖检查

[审计脚本](../../scripts/audit_schema_constraints.py)在原检查点的`BartTokenizerFast`上运行，校验了六个分词器文件及模型配置的哈希。只使用原5,000题训练集和已见500题开发集，未读取官方validation。详细来源、代码哈希、首个拒绝位置见[覆盖报告](../../artifacts/thesis_direction_review/schema_constraint_coverage.json)。

| 输入 | 数量 | 被约束拒绝 |
|---|---:|---:|
| KB全部字段×四种边界空格组合 | 5,068 | 0 |
| 原训练金标程序 | 5,000 | 0 |
| 原开发金标程序 | 500 | 0 |
| 旧开发beam4候选文本 | 2,000 | 264 |
| 旧开发beam8候选文本 | 4,000 | 585 |

候选是从已存文本重新分词，不是原始生成token轨迹；对截断文本补EOS仅为边界探针。264/585条均含对应角色词表之外的字段，没有发现不含词表外字段的候选被挡；空或残缺字段也计入词表外。这些是旧候选的拒绝数，**不是纠正题数、预期收益或新的候选覆盖率**。两组旧候选分别有559/1600条语法检查失败、115/567条含未知函数，与字段拒绝可能重叠。通过词表约束不等于程序可执行或答案正确。

14项标准库行为测试覆盖不同字段角色、前缀碰撞、半字段结束、beam重排、特殊token和非原子分隔符。实际分词器另外检查这些绕过路径。当前结论仅为：在记录的规范编码和已见金标范围内未发现误挡，可以进入小规模生成接入检查。

## 接入方式及配置差异

保持历史`baseline.py`及`executor.py`原样。新调用方使用：

```python
constraint = SchemaFieldConstraint(tokenizer, schema["names_by_role"],
                                   decoder_start_token_id=config.decoder_start_token_id)
# 对实验组和无约束对照都显式设置，并记入新的运行协议。
generation_config.forced_eos_token_id = None
kwargs = constraint.generation_kwargs(generation_config)
outputs = model.generate(**question_inputs, **kwargs)
```

这段是未来接入示例，本轮没有调用`model.generate`。`generation_kwargs`拒绝强制EOS仍开启的配置；它不偷偷修改原实验配置。新两组均关闭强制EOS，不能直接把新C与旧缓存的差值全算作约束效果。完整语法、结束状态及字段检查须在生成后执行，错误不得自动改成合法程序后再计分。

当前接口中，意外的不支持分词路径会使一次调用失败。后续小规模接入检查必须记录这种失败；不能仅删除失败题，或悄悄对失败题改用无约束模型。还需实测回调耗时和最大长度行为，本轮CPU重放用时不是GPU生成成本。

## 下一阶段的固定小试范围

先接入并检查固定开发文件前20题的运行行为；这20题仍属于已见开发集。通过后才在原500题上比较以下三组，全部沿用同一检查点、问题输入、执行器和候选选择规则，不新增训练：

1. 无约束beam4，4候选；
2. C约束beam4，4候选；
3. 无约束beam8，8候选，作为增加搜索预算的对照。

三组统一关闭强制EOS，其他配置从固定检查点原生成记录复制；开跑前将输入、代码、配置和选择规则写入独立manifest。记录完整500题准确率、纠正/改错、合法字段率、语法失败、无答案/截断、候选覆盖和实际耗时。旧保护beam8成绩作为历史背景，不能冒充这次同配置对照。

本阶段预算上限为单卡累计1小时；超时或出现接口错误立即保存失败记录，不扩充题集、训练或扫描参数，不将部分完成结果外推到500题。小试用于判断C是否留下足够合法字段语义混淆，不能当作新的独立泛化测试。当前尚未启动此GPU阶段。

训练假设H仍需通过实际混淆案例、近邻区别与训练样本量三个门槛；即使C覆盖检查通过，也不自动批准四组继续训练。

## 复核命令

```bash
python3 -B -m unittest experiments.condition_consistency.test_schema_constraints -v
# 需原检查点的小型分词器文件、固定数据和transformers环境；不需要权重加载。
python3 -B scripts/audit_schema_constraints.py \
  --tokenizer results/condition_consistency/pilot_bart5k_seed20260926/generator/final \
  --output /tmp/schema_constraint_coverage.json
python3 -B scripts/prepare_radar_review_packet.py --check-only
```

审计拒绝覆盖已有结果；重新运行可另指定输出。同期交付的[12条雷达事实核查包](../../artifacts/thesis_direction_review/radar_review_batch/README.md)全部待人工核验，不能用作方法评分金标。
