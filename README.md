# 复杂知识问答：公开基准验证与雷达知识应用

硕士毕业设计研究工作区。当前顺序：**公开KGQA基准 → 错误诊断 → 局部约束查询修复 → 公开评测 → 雷达迁移**。
下一轮聚焦字段角色与局部事实约束下的单字段修复，检验条件保护能否减少修复造成的改错；这是待验证机制。自建图谱承担应用验证，主要有效性证据来自公开数据。

**首轮最小验证已完成：500题开发集上，BART基线73.0%，执行筛选80.0%，条件定向排序79.0%。**
公开训练与评测链路已跑通，当前条件机制尚未超过有效基线。下一步完成单字段可修复覆盖审计和普通字段匹配对照，再测局部约束的增量。详见[实测报告](docs/MINIMAL_VALIDATION.md)和[主线与创新边界](docs/RESEARCH_DIRECTION.md)。

## 阅读入口

| 文档 | 用途 |
|---|---|
| [首轮最小验证结果](docs/MINIMAL_VALIDATION.md) | 实际GPU实验、负结果、可用结论与下一步 |
| [研究方向与近邻工作](docs/RESEARCH_DIRECTION.md) | 主线、价值、贡献边界与风险 |
| [公开基准专项调研](docs/research/PUBLIC_BENCHMARK_REVIEW.md) | 数据协议、近邻方法、源码核验和公平比较 |
| [当前状态](docs/CURRENT_STATUS.md) | 实际数据规模、已实现内容及证据限制 |
| [执行计划](docs/ROADMAP.md) | 分阶段交付、验收门槛与调整路线 |
| [训练就绪检查](docs/TRAINING_READINESS.md) | 已修复环境问题与开训前阻断项 |
| [仓库地图](docs/REPOSITORY_MAP.md) | 当前模块、历史成果和本地数据位置 |
| [数据与产物政策](docs/ARTIFACT_POLICY.md) | 公开范围、数据恢复和清理重建 |
| [开题原稿](docs/research/proposal.md) | 保留原结构；贡献表述以新研究方向审查为准 |

## 当前事实（2026-09-26）

- 公开KQA Pro：5,000/1,000/500题用于生成训练/排序训练/开发；实际完成四卡BART训练及同候选对照。此为开发预实验，尚非正式公开报告集成绩。
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
