# 组合评价补充与论文主线对齐：完成记录

日期：2026-11-08。本轮完成两项本地工作：冻结组合评价补充，统一论文研究问题、技术路线与实验矩阵。未连接服务器、未训练或运行问答模型。

## 1. 组合评价已经形成完整可用材料

按编题前登记的设计，四个既定新来源家族各编制三题，共12题。题目使用原156条完整来源目录，保留原24题、576条训练题和108条开发题。新12题与原24题分别冻结和统计，共享四个来源家族。

| 来源 | 新增组合题 | 参考调用数范围 |
|---|---:|---:|
| FURUNO DRS4W | 3 | 6—11 |
| OTT RLS | 3 | 8 |
| smartmicro UMRR-11 | 3 | 5—11 |
| Campbell CS475A | 3 | 9—11 |

组合题覆盖配置配对、多参数筛选、范围包含、模式条件、空结果与联合配置支持。准入以来源信息需求和必要事实为依据。实际提交路径中，7题使用Join、10题使用Filter、6题使用Derive、5题使用Expand、1题使用Assess，题型存在交叉。依赖图检查确认全部操作参与最终答案，冗余步骤为0。

两位作者先阅读原件，保存题面与预期答案，再编写程序。一位非作者AI独立阅读固定八页原件，先作答并封存，再比对作者参考和执行结果。12题的数值、条件和结论均通过来源核对。

审阅中修订了一项必要性说明：OTT同一电压下的功耗与电流属于相关表示，按Measurement与Rest两个状态事实组计量；两个筛选条件在当前候选集合中的分类相同。修订保留原稿和哈希，题面、数值、程序与格式规范均未改变。另一项存储任务的输出粒度经格式规范核对明确。所有来源审阅登记为AI。

材料入口：

- [预登记设计](../results/radar_domain/compositional_supplement_20261108/design.json)
- [独立来源审阅](../results/radar_domain/compositional_supplement_20261108/source_review.safe.json)
- [冻结汇总](../results/radar_domain/compositional_supplement_20261108/summary.json)
- [集成审计](../results/radar_domain/compositional_supplement_20261108/integration_audit.json)
- 私有完整包：`data/radar_domain_recovery_v2/compositional_supplement_20261108/`，包括question-only、完整参考、格式规范及原题元数据、执行记录、分词记录和manifest。

## 2. 执行、预算和隔离检查

12条参考程序全部通过，合计99次调用。实际分词检查的最大上下文为7770 token、最大单步目标89 token、最大动作目标总量608 token，符合原16384上下文、192单步、2048总生成、24调用预算，全程无截断。

新增准入测试与相关执行、评分、恢复测试共31项通过。验证覆盖无值格式规范、拒绝答案计数注入、忽略不参与答案的冗余操作、依赖合法性和来源文件篡改拒绝。固定分词器重新构建8个冻结产物，与保存字节全部一致。

集成复核检查原冻结19个绑定文件、新材料29项输入及设计9项输入：原训练、开发和24题评价未变，新目录与原完整156读法逐字节一致；新题号与原分组隔离。新12题属于评价材料，两个误作训练输入的检查均被拒绝。此次结构审计由其中六题作者实施，其角色与非作者来源审阅分别记录。

复核命令（项目根目录）：

```bash
python3 -m unittest tests.test_compositional_supplement tests.test_radar_table_program tests.test_radar_table_scoring tests.test_table_recovery_data
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 cache/domain_cpu_env/bin/python -m experiments.radar_domain.freeze_compositional_supplement --check
python3 thesis/current/build_thesis.py --check-only
```

## 3. 论文的主线与实验已对齐

主线为：**来源知识与任务构建 → 恢复监督方法 → 公开有效性和状态分析 → 雷达域内应用验证。**

- 第一章更新核心问题、主要研究内容与总体路线图；新增表1.2，逐项列明对照、指标、结论对象和完成状态。
- 第3.6.3节说明组合评价的准入、来源审阅、事实组与单独冻结。
- 第6.6.4节报告参考执行、操作覆盖、分词预算及基础/组合两张结果表的评价安排。
- 摘要、章节概述和第七章同步当前数据准备成果与后续实验。

当前[完整PDF](../thesis/current/build/main.pdf)为72页、30条引用；六组证据导出检查通过，无越界盒、未解析引用或缺字警告。技术路线、实验矩阵、新来源分布及操作覆盖等关键页已目视检查。修订前69页版本完整保存于私有 `manuscript_before_20261108/`。

[机器可读实验矩阵](../results/radar_domain/compositional_supplement_20261108/experiment_matrix.json)给出四项核心实验：公开A/C效果、公开同状态评价、条件表示效果、域内A/C恢复对照。公开+4.75个百分点及96题表示结果保持原实验记录。

## 4. 接下来的工作

本地评价材料与论文主线的两项工作已完成。下一阶段按现有队列，在计算环境恢复后：

1. 核验普通实验账号、原模型和运行环境，完成旧开发短测。
2. 用已有模型完成冻结的公开同状态评价，分析正常与偏离历史下的A/C差异。
3. 用576条合成成功轨迹训练I_domain，采集训练题真实失败，再开展A_domain/C_domain配对续训；在开发部分诊断并决定同方案复现。
4. 在原24道基础题和新12道组合题上分别报告完整答案、改正与退步、来源分组和成本，补入实际模型结果。

域内训练的精确参数和软件/模型身份将在运行环境短测后、正式训练前绑定。原领域开发8 GPU小时、公开同状态9小时5分钟及项目72 GPU小时上限继续执行。本轮新增模型实验与GPU用量为0。
