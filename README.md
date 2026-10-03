# 复杂知识问答：公开基准验证与雷达知识应用

**论文定位：电子信息专业硕士，以一项可归因的算法或训练改进为研究核心，雷达知识资源和领域应用提供支撑。** 建议题目为“基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用”。已调研9篇具有学位依据的硕士论文，具体文献、贡献边界、七章框架和六个月安排见[论文主线](docs/THESIS_ROUTE_REVIEW.md)。

**当前实验：初始Agent 75.6%，同3B程序基线75.2%，尚无可靠优势；普通续训／恢复续训对照正在运行。** 恢复训练与可选GRPO属于公开方法子实验，不再作为整篇论文的唯一创新。继续按已冻结协议检验，若简单程序方案更好就采用它。雷达独立金标、条件与区间表示、连续证据链和新模型的领域接入仍需完成，不能把既有大图和自动题视为已验证成果。

新增[导师式方法审查](docs/THESIS_ROUTE_REVIEW.md#8-导师式方法审查将工程主线落实为可检验的训练改进)：优先论证“语义核验辅助的轨迹偏好训练”，先检查自然训练轨迹是否存在足够可改变的标签。该项仍是候选，未实现、未启动训练；工程完整性不能替代核心方法证据，现有A/C协议保持不变。

主线：**保留主体、条件、区间和来源的雷达知识资源 → 小模型生成可执行查询 → 答案与依据 → 公开基准和独立雷达评价**。复用现有数据、执行器和系统，不重建整个项目。当前训练不中断，领域核验与论文写作并行推进；范围和队列见[执行计划](docs/ROADMAP.md)。

**已完成两轮开发和KQA Pro完整官方validation的11,797题评测。** 同一5,000题训练的BART下，4候选执行筛选79.90%，普通保守字段修复81.25%，局部保守修复81.39%，8候选执行筛选81.44%。局部保守修复相对4候选纠正178题、改错2题，但未证明优于8候选；局部约束相对普通修复的额外增量很小。

后验补算相同保护的beam8为81.74%，进一步削弱当前字段修复的必要性；字段修复已停止开发。完整正负结果、单种子和数据使用边界见[修复验证报告](docs/QUERY_REPAIR_VALIDATION.md)。历史首轮排序负结果保留在[最小验证报告](docs/MINIMAL_VALIDATION.md)。

## 阅读入口

当前方向和队列以[论文主路线](docs/THESIS_ROUTE_REVIEW.md)及[执行计划](docs/ROADMAP.md)为准；此前小算法论证和训练建议保留为历史，不再作为新路线前置条件。

| 文档 | 用途 |
|---|---|
| [论文整体路线](docs/THESIS_ROUTE_REVIEW.md) | 专硕定位、9篇硕士论文、研究贡献、七章框架和实证计划 |
| [公开方法子实验原计划](docs/THESIS_AGENT_PILOT_PLAN.md) | 保留Agent/SFT/RL与恢复训练的具体机制、对照和停止规则 |
| [执行计划](docs/ROADMAP.md) | 当前任务队列与六个月里程碑 |
| [此前小算法方法论证](docs/research/METHOD_FEASIBILITY_BRIEF.md) | 历史字段约束/负例假设，已停止作为当前入口 |
| [字段约束接口与覆盖检查](docs/research/SCHEMA_CONSTRAINT_BASELINE.md) | 已有技术基线C的范围、分词器核验与接入边界 |
| [字段约束500题对照](docs/research/SCHEMA_PILOT_RESULTS.md) | 实际收益、改错、成本与停止决定 |
| [雷达标注规范与模板](docs/research/RADAR_DATA_ANNOTATION.md) | 事实核验与独立问答分离、来源追踪待复核样例 |
| [查询修复完整验证](docs/QUERY_REPAIR_VALIDATION.md) | 两轮开发、完整官方val、强对照、误修改与成本 |
| [首轮最小验证结果](docs/MINIMAL_VALIDATION.md) | 历史排序实验与负结果 |
| [历史字段约束方向](docs/RESEARCH_DIRECTION.md) | 保留旧假设与停止理由，当前入口为论文主线 |
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
- 新入口为`experiments/agent_feedback/`：共享初始SFT、同监督预算A/C续训、同模型程序基线、真实交互推理与GPU预算记录。GRPO运行短测已通过，原生工具循环预算见独立修订协议；正式初始模型与程序基线已训练并完成开发评测，恢复训练对照尚未完成。

2026-10-03第一阶段已完成：初始Agent SFT训练81.4分钟，程序基线训练42.1分钟；500题开发评测和5,000题训练轨迹收集均正常退出，累计约6.03 GPU小时（包括失败及运行短测）。全部模型任务使用`syx`，管理员未执行训练。

| 开发集500题 | 初始Agent（2轮SFT） | 同3B程序基线（3轮SFT） |
|---|---:|---:|
| 正确数／准确率 | 378／75.6% | 376／75.2% |
| 未完成作答 | 61 | 83 |
| 评测进程耗时 | 26.3分钟 | 4.5分钟 |
| 平均输入加生成token | 11,664 | 403 |

Agent相对程序基线纠正42题、改错40题，净增2题；差异0.4个百分点，配对bootstrap 95%区间为[-3.2, 3.8]，不能宣称可靠提升。Agent每轮重新处理历史，计入的输入加生成token约为程序方案28.9倍；这不是FLOPs比值。两者训练形式和预算不同，这只是初始模型与必要强基线的比较，尚未检验恢复训练增量。证据见[完整配对比较](results/agent_feedback/initial_vs_program_dev.json)。

已从5,000题自由生成中构造418组配对恢复样本，覆盖训练题8.36%，占659条通过回放验证的失败轨迹63.4%。418对已固定执行种子再次复核；另有两条因进程集合顺序产生的回放差异被原构造器排除，本轮保持排除并记录原因。恢复点是失败轨迹中首次偏离参考步骤的位置，不保证该步骤本身语义错误；414处仍可执行，4处被执行器拒绝。见[恢复对统计](results/agent_feedback/recovery_pairs.json)及[独立回放审计](results/agent_feedback/recovery_pair_audit.json)。

已启动A/C对照，分别使用GPU 0/1，从同一初始adapter出发，每组5,772条记录、788,534个监督token，其中一半来自配对后缀。418道配对题各复用7–8次；重复不算新增独立样本。两组题目、顺序、复用次数和监督量一致，输入量与计算量不相同。预算裁剪差异已量化，见[语料清单](results/agent_feedback/continuation_data.json)与[监督预算审计](results/agent_feedback/continuation_token_audit.json)。留出集尚未使用。

SFT、八题交互推理和一步原生GRPO运行短测均已完成，见[运行检查](results/agent_feedback/runtime_smoke_checks.json)。GRPO短测奖励、梯度及首步预热学习率为零，只证明兼容性；正式RL前须用完整SFT分支检查学习信号。后续GRPO命令需携带[私有头文件CPATH](results/agent_feedback/python_headers.json)。仓库中的[训练快照](results/agent_feedback/running_snapshot.json)带采集时间，不是实时状态。

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
