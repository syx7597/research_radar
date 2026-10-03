# 当前状态与证据边界

更新：2026-10-03；历史资产基线审计为2026-09-26。主线见[论文主线与框架](THESIS_ROUTE_REVIEW.md)，下一步见[执行计划](ROADMAP.md)。以下区分当前训练与历史资产，不能把旧结果视作新方法成果。

## 1. 当前阶段与历史证据

论文定位为电子信息专业硕士的知识构建、可执行问答与系统实证。恢复轨迹训练是公开方法子实验，不再代表整篇论文的唯一贡献。领域知识中的主体/条件/区间/来源缺口已有实例，但新的表示、连续证据链和独立人工评价尚未完成。

2026-10-03 23:29（北京时间）确认A/C训练及开发评测均完成。A为392/500（78.4%），C为404/500（80.8%）；C纠正29题、改错17题，净增12题，差异+2.4个百分点，配对bootstrap 95%区间[-0.2, 5.0]。达到预登记继续投入门槛，但单训练种子且区间跨0，不能宣称稳定有效。未完成54→24，无效调用1026/3957→395/3732；总token下降6.1%，评测生成进程耗时却为24.9→26.8分钟，不能据此声称整体提速。C相对程序P准确率+5.6个百分点，但输入加生成token约25.6倍且训练预算不同。证据见[同预算A/C比较](../results/agent_feedback/recovery_vs_clean_dev.json)及[程序基线比较](../results/agent_feedback/recovery_vs_program_dev.json)。

初始Agent/程序基线、5,000题训练轨迹收集、418组恢复对及每组788,534监督token续训均已完成。下一步优先当前恢复训练的复现，新的语义核验候选暂不启动。

GRPO运行短测通过，但奖励和梯度为零，不能证明训练有效；正式RL须检查学习信号并遵守原预算。2,000题项目留出尚未使用。最新证据入口见[README](../README.md)、[初始配对比较](../results/agent_feedback/initial_vs_program_dev.json)、[恢复样本统计](../results/agent_feedback/recovery_pairs.json)及[续训语料清单](../results/agent_feedback/continuation_data.json)。

以下字段约束、BART、旧雷达Agent和数据oracle均为历史证据。KQA Pro本地val共11,797题，其中2,806含限定相关函数；雷达16,609属性中未发现结构化条件字段，见[任务清单](../artifacts/public_benchmark_profile.json)。

历史字段约束基线小试已完成：同一500题开发集，普通beam4为401/500、约束beam4为402/500、普通beam8为399/500；约束相对beam4纠正5题、改错4题，未显示可靠优势，本次生成耗时约16.8倍。三卡并行推理，含接入失败和诊断累计10.84 GPU分钟，无训练。停止继续调旧字段约束基线C，旧训练假设H不再进入任务队列。此处C与新恢复续训实验组C不是同一编号体系。见[实际结果](research/SCHEMA_PILOT_RESULTS.md)。

此前5,500条金标与5,068个字段边界覆盖检查保留；这与真实生成结果是不同证据。12条雷达事实已附原文定位，全部待人工复核。

后验同保护beam8为81.74%，不属于新的独立测试。当前方法与预算见[整体路线](THESIS_ROUTE_REVIEW.md)；此前[小算法方法论证](research/METHOD_FEASIBILITY_BRIEF.md)已归为历史，雷达规范见[标注规范](research/RADAR_DATA_ANNOTATION.md)。

已有知识构建、固定策略问答、类型化组合执行器和演示。公开实验已完成一个BART-base生成器的5000题训练、首轮排序探针，以及两轮查询修复开发与完整official val评估。

在同一执行条件下，11,797题的beam4 first-valid为79.90%，v2普通保守修复81.25%，v2局部保守修复81.39%，beam8 first-valid为81.44%。局部保守修复纠正178题、改错2题，相对beam4净增176题；尚未证明优于beam8，局部约束的额外收益较小。详见[最新报告](QUERY_REPAIR_VALIDATION.md)。这些数字来自单个生成器种子，官方val有旧项目使用历史，不能声称hidden test成绩或多次训练稳定性。

v2受到v1首次2000题错误分析启发，旧2000从此仅为已见诊断。执行器状态隔离和固定哈希种子用于所有对照，属于基础设施修正，不计算法创新。历史首轮73.0%/80.0%/79.0%及旧缓存保留，见[首轮报告](MINIMAL_VALIDATION.md)。本轮已停止调参；雷达迁移尚未验证。

旧`ca_agraphrag`仍未完成统一变量协议、多粒度反馈及可靠SFT/RL训练，不能将新BART结果算作旧RL成果。

## 2. 实际数据

| 对象 | 实测 | 口径 |
|---|---:|---|
| `kg_v3/edges.json` | 21,928 | 当前边记录 |
| `kg_v3/entities.json` | 10,744 | 实体记录 |
| 关系类型 | 23 | 当前edges中的relation |
| RL过滤边记录 | 11,432 | `KGTools(filtered=True).n_edges` |
| train/dev/test | 4,277/582/542 | 共5,401题、7题型 |
| SFT轨迹 | 4,277 | 对应训练问题 |
| dev/test润色版 | 582/542 | 仍需语义一致性抽检 |

构建报告22,241边、16,609属性属于历史口径，未随清洗同步。相同tier的confidence基本固定，不能当作校准概率。
文件身份见[数据清单](../artifacts/data_manifest.json)。

train/test的anchor交集为0；但从`gold_support`的`head/a/b`字段提取的测试315个显式查询实体中，有143个也在训练中。
该检查未计入答案实体和隐式引用；目前只能称锚点划分，不能称严格实体隔离，程序组合隔离尚未建立。

## 3. 实现链路

| 链路 | 入口与数据 | 边界 |
|---|---|---|
| 旧问答/Agent | `qa_strategy_pipeline.py`、`agent/agent_loop.py` → `graphrag_index/merged_triples.json` | 默认仍用旧图谱 |
| v3图文检索 | `pipeline/v3/retrieve_v3.py` → `kg_v3`及独立索引 | 单独入口，支持叙述联动 |
| 新策略环境 | `ca_agraphrag/kg_tools.py`、`env.py` → v3过滤视图 | 尚未统一计划树、变量引用和文本通道 |

现有SFT轨迹把中间实体集合复制到后续参数，最多4步，仅含lookup/find/count/intersect/finish，没有union/difference训练。
所以现有数据和工具满分，不能证明长程决策或组合泛化。

## 4. 可引用证据

| 证据 | 结果 | 解释 |
|---|---|---|
| 旧RadarKG-QA-499 | Strategy88.6%、baseline52.7%、RoG风格60.3% | 历史固定算子方法，非新RL结果 |
| 旧KQA Pro 502题 | 27.9% vs 27.7%，差异CI跨0 | 无显著总体跨域优势 |
| 旧组合Agent | 正确71/72，计划/执行72/72 | 小规模提示式规划基础 |
| 本轮工具oracle | 5,401/5,401 | 工具可复现金标，不是模型性能 |
| 本轮完整金标策略 | dev582/582、test542/542 | 交互和金标动作链自洽 |
| 本轮保存SFT回放 | 4,277/4,277 | 保存动作可执行，不保证问题语义和泛化 |
| 本轮边界测试 | 16/16 | 空交集、严格计数、非法动作、回合终止 |

历史结果本轮未调用LLM重跑。依据位于`results/qa500_3way_full_summary.md`、`results/kqa_pro_e2e_500_ci_summary.md`、`results/composition_eval_results.json`。

## 5. 修复与阻断项

本轮修复：交集忽略空集合；计数将小数/含糊文本截为整数；非法动作崩溃；结束后继续领奖励。
数据和金标未变，oracle结果保持全通过。

旧`ca_agraphrag`训练器的历史阻断项包括：SFT对话遗漏assistant头；GRPO参考模型/KL/多卡实现；环境与训练器奖励口径；大集合复制与token预算冲突。不能直接恢复该旧入口。
新的`experiments/agent_feedback`已另行完成逐步环境、训练mask、预算及运行短测，并实际执行SFT；不能把旧阻断项误写成新实验尚未启动。历史审查见[训练就绪检查](TRAINING_READINESS.md)。

## 6. 整理动作

历史方案归档至`docs/archive/proposals/`；旧报告移至`docs/archive/PROJECT_REPORT_2026-06.md`；新开题移至`docs/research/proposal.md`。
代码保持原路径。旧paper/thesis/Agent标注为历史成果。
删除5个可重建渲染目录，共947,099,791字节；PDF、转录、抽取缓存、图谱快照和唯一实验记录保留。
公开仓库排除原始/全文/大型数据及旧凭据历史，以SHA-256清单记录数据身份。原有大量整文件差异来自CRLF，发布文本统一换行。
