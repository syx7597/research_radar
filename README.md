# 复杂知识问答：公开基准验证与雷达知识应用

**当前阶段：基础对照已完成，训练假设尚未成立。** 最新500题开发比较：普通beam4为80.2%，字段约束beam4为80.4%，普通beam8为79.8%。约束净增1题、未显示可靠优势，本次实现明显更慢；不继续调这个模块或立即开训。见[完整结果](docs/research/SCHEMA_PILOT_RESULTS.md)。论文范围仍以[整体路线](docs/THESIS_ROUTE_REVIEW.md)和[具体方法论证](docs/research/METHOD_FEASIBILITY_BRIEF.md)为准；另有12条雷达待审事实。

硕士毕业设计研究工作区，预计六个月后提交。当前主线：**雷达知识资源构建 → 可执行复杂问答 → 公开方法验证 → 独立雷达应用评价**。研究问题与算法试验分开，复用现有数据、BART、执行器和系统。

**已完成两轮开发和KQA Pro完整官方validation的11,797题评测。** 同一5,000题训练的BART下，4候选执行筛选79.90%，普通保守字段修复81.25%，局部保守修复81.39%，8候选执行筛选81.44%。局部保守修复相对4候选纠正178题、改错2题，但未证明优于8候选；局部约束相对普通修复的额外增量很小。

后验补算相同保护的beam8为81.74%，进一步削弱当前字段修复的必要性；字段修复已停止开发。完整正负结果、单种子和数据使用边界见[修复验证报告](docs/QUERY_REPAIR_VALIDATION.md)。历史首轮排序负结果保留在[最小验证报告](docs/MINIMAL_VALIDATION.md)。

## 阅读入口

总路线和当前队列以以下前三份文件为准；[此前训练建议](docs/research/NEXT_EXPERIMENT_DECISION.md)保留为历史候选，不自动触发实验。

| 文档 | 用途 |
|---|---|
| [论文整体路线](docs/THESIS_ROUTE_REVIEW.md) | 从知识构建到问答与三层验证的完整范围 |
| [实验前方法论证](docs/research/METHOD_FEASIBILITY_BRIEF.md) | 真实瓶颈、近邻区别、基础对照与开训/停止门槛 |
| [执行计划](docs/ROADMAP.md) | 当前任务队列与六个月里程碑 |
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

## 当前事实（2026-09-26）

- 公开KQA Pro：固定一个BART训练种子、5,000题监督数据；完成v1首次2000题检查及v2完整official val 11,797题评估。旧2000题在v2中仅作已见诊断；official val有历史使用，不能称hidden test。
- 当前图谱21,928条边、10,744个实体；RL过滤后保留11,432条边记录。
- train/dev/test 为4,277/582/542；SFT轨迹4,277条。
- 5,401题工具oracle检查通过，证明数据和工具自洽，不是模型正确率。
- 旧固定策略系统499题88.6%属于历史结果，不能作为新方法实验结论。
- 新SFT/RL代码存在训练阻断项，本地未发现对应权重或完整对照实验。

## 目录

```text
docs/                 当前方向、状态、计划；archive/ 为旧路线资料
ca_agraphrag/          既有RL环境与策略原型，暂不作为新实验主入口
agent/                旧类型化组合执行器、分析Agent与演示
pipeline/v3/          知识构建、手册处理与v3图文检索
qa_*.py               旧固定策略问答与RoG风格基线
graphrag_retriever.py  共用混合检索
experiments/ results/  历史实验代码和结果
  condition_consistency/ 当前公开基准实验入口与结果
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
