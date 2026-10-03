# 复杂知识问答：公开基准验证与雷达知识应用

**当前阶段：服务器重启后GPU已恢复，真实两步训练短测通过，初始Agent与同模型程序基线的正式SFT已启动；尚无方法收益结论。** 采用已有的小模型工具Agent与SFT→GRPO框架，唯一改进研究“执行反馈恢复训练”；KQA Pro承担主要方法验证，雷达承担知识构建和独立应用评价。模型、文献、同模型对照及三周/72 GPU小时首轮上限见[完整路线](docs/THESIS_ROUTE_REVIEW.md)。字段修复和约束小试已结束，不继续围绕小规则追分；历史结果如实保留。

硕士毕业设计研究工作区，预计六个月后提交。当前主线：**雷达知识资源构建 → 小模型工具问答 → 公开方法验证 → 独立雷达应用评价**。复用现有知识、执行器和系统，BART作为历史对照；允许借鉴已有研究，方法效果由受控实验判断。

**已完成两轮开发和KQA Pro完整官方validation的11,797题评测。** 同一5,000题训练的BART下，4候选执行筛选79.90%，普通保守字段修复81.25%，局部保守修复81.39%，8候选执行筛选81.44%。局部保守修复相对4候选纠正178题、改错2题，但未证明优于8候选；局部约束相对普通修复的额外增量很小。

后验补算相同保护的beam8为81.74%，进一步削弱当前字段修复的必要性；字段修复已停止开发。完整正负结果、单种子和数据使用边界见[修复验证报告](docs/QUERY_REPAIR_VALIDATION.md)。历史首轮排序负结果保留在[最小验证报告](docs/MINIMAL_VALIDATION.md)。

## 阅读入口

当前方向和队列以[论文主路线](docs/THESIS_ROUTE_REVIEW.md)及[执行计划](docs/ROADMAP.md)为准；此前小算法论证和训练建议保留为历史，不再作为新路线前置条件。

| 文档 | 用途 |
|---|---|
| [论文整体路线](docs/THESIS_ROUTE_REVIEW.md) | Agent/SFT/RL与恢复训练、文献、数据、对照和停止规则 |
| [执行计划](docs/ROADMAP.md) | 当前任务队列与六个月里程碑 |
| [此前小算法方法论证](docs/research/METHOD_FEASIBILITY_BRIEF.md) | 历史字段约束/负例假设，已停止作为当前入口 |
| [字段约束接口与覆盖检查](docs/research/SCHEMA_CONSTRAINT_BASELINE.md) | 已有技术基线C的范围、分词器核验与接入边界 |
| [字段约束500题对照](docs/research/SCHEMA_PILOT_RESULTS.md) | 实际收益、改错、成本与停止决定 |
| [雷达标注规范与模板](docs/research/RADAR_DATA_ANNOTATION.md) | 事实核验与独立问答分离、来源追踪待复核样例 |
| [查询修复完整验证](docs/QUERY_REPAIR_VALIDATION.md) | 两轮开发、完整官方val、强对照、误修改与成本 |
| [首轮最小验证结果](docs/MINIMAL_VALIDATION.md) | 历史排序实验与负结果 |
| [研究方向与近邻工作](docs/RESEARCH_DIRECTION.md) | 主线、价值、贡献边界与风险 |
| [公开基准专项调研](docs/research/PUBLIC_BENCHMARK_REVIEW.md) | 数据协议、近邻方法、源码核验和公平比较 |
| [当前状态](docs/CURRENT_STATUS.md) | 实际数据规模、已实现内容及证据限制 |
| [训练就绪检查](docs/TRAINING_READINESS.md) | 已修复环境问题与开训前阻断项 |
| [仓库地图](docs/REPOSITORY_MAP.md) | 当前模块、历史成果和本地数据位置 |
| [数据与产物政策](docs/ARTIFACT_POLICY.md) | 公开范围、数据恢复和清理重建 |
| [开题原稿](docs/research/proposal.md) | 保留原结构；贡献表述以新研究方向审查为准 |

## 当前事实（2026-10-03）

- 公开KQA Pro：固定一个BART训练种子、5,000题监督数据；完成v1首次2000题检查及v2完整official val 11,797题评估。旧2000题在v2中仅作已见诊断；official val有历史使用，不能称hidden test。
- 当前图谱21,928条边、10,744个实体；RL过滤后保留11,432条边记录。
- train/dev/test 为4,277/582/542；SFT轨迹4,277条。
- 5,401题工具oracle检查通过，证明数据和工具自洽，不是模型正确率。
- 旧固定策略系统499题88.6%属于历史结果，不能作为新方法实验结论。
- 新公开实验固定5,000训练题、500开发题及2,000项目留出题；后者排除此前已用组件，仍不是官方隐藏测试集。见[冻结协议](results/agent_feedback/protocol.json)与[切分清单](results/agent_feedback/split_manifest.json)。
- 逐步执行与原整程序执行在5,500条金标轨迹上完全一致；训练/开发各有1条既有答案不一致，仅排除训练异常。4,999条训练轨迹通过真实Qwen分词检查。这些是基础设施验证，不是模型正确率。
- 新入口为`experiments/agent_feedback/`：共享初始SFT、同监督预算A/C续训、同模型程序基线、真实交互推理与GPU预算记录。GRPO适配已有CPU检查，原生工具循环预算见独立修订协议；基座权重已校验，短测已产生训练后的adapter，正式基线SFT已启动，尚无完整效果对照。

2026-10-03恢复与训练记录：经用户授权的一次正常整机重启后，SSH已恢复，新boot ID为`222cdbd8-e2ea-4bbc-85d9-1058c6639e75`。四张GPU通过CUDA实际计算及BF16检查；模型训练由`syx`（UID 1001）执行，管理员未执行训练。`sft_smoke_v3`完成真实两步优化，训练loss为1.0584，实际监督506 tokens，训练循环约3.05秒；这些数据证明训练链路可运行，不是问答准确率或方法收益证据。GPU预算按监督进程账本结算，不能以训练循环耗时替代全部进程耗时。

正式初始Agent SFT（2 epochs，GPU 0）和同3B一次性程序基线（3 epochs，GPU 1）已并行启动。状态与证据见[重启记录](results/agent_feedback/server_reboot_attempt.json)、[短测结果](results/agent_feedback/sft_smoke_v3/run_result.json)、[短测账本](results/agent_feedback/jobs/sft_smoke_v3.json)、[Agent训练账本](results/agent_feedback/jobs/initial_sft_v1.json)及[程序基线账本](results/agent_feedback/jobs/program_sft_v1.json)。完成情况以对应账本和结果文件为准。

另外用空闲GPU完成八题批量交互推理和一步原生GRPO短测，见[运行检查](results/agent_feedback/runtime_smoke_checks.json)。GRPO首次因缺少Python头文件失败，已在syx实验目录内补齐，系统环境及训练算法未修改；后续GRPO命令需要保留[依赖记录](results/agent_feedback/python_headers.json)中的`CPATH`。GRPO短测使用两步SFT权重，四条采样奖励及梯度均为零，首个预热步骤学习率也为零；只证明执行链路兼容，未验证有效参数更新或RL收益。正式RL前须用完成SFT的分支检查学习信号。仓库中的[训练快照](results/agent_feedback/running_snapshot.json)带采集时间，不是实时状态。

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
