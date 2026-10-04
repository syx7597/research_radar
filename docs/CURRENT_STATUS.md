# 当前状态与证据边界

更新：2026-10-04；历史资产基线审计为2026-09-26。主线见[论文主线与框架](THESIS_ROUTE_REVIEW.md)，下一步见[执行计划](ROADMAP.md)。以下区分当前训练与历史资产，不能把旧结果视作新方法成果。

## 1. 当前阶段与历史证据

**2026-10-04：所有公开训练、原dev500评测与一次四模型错误诊断消融完成。** RL流水线于16:00结束，随后四组结果全部复算，B/D共1000条轨迹回放零差异。错误诊断消融19:26启动、20:00完成（均为北京时间）。见[RL复核](../results/agent_feedback/rl_paired_review_v1.json)、[RL阶段决定](../results/agent_feedback/rl_stage_decision_v1.json)和[诊断汇总](../results/agent_feedback/feedback_diagnostic_summary_v1.json)。

| 同一500题开发集 | 首种子 | 第二续训种子 |
|---|---:|---:|
| P：程序基线 | 75.2%（376） | 未追加 |
| A：普通SFT | 78.4%（392） | 77.2%（386） |
| C：恢复SFT | 80.8%（404） | 82.4%（412） |
| B：A + RL | 78.0%（390） | 未追加 |
| D：C + RL | 82.0%（410） | 未追加 |

恢复训练C/A两次分别+2.4、+5.2个百分点，题目配对95%区间[-0.2, 5.0]与[2.2, 8.2]；共用初始模型、语料及500题，不是1000道独立题。D/B为+4.0个百分点，区间[1.2, 6.8]，是RL后的恢复训练对照；RL本身的增量B/A=-0.4、D/C=+1.2个百分点，区间均跨零，D/C总token增加2.4%。两项RL增量均未达到原投入标准，停止扩展RL，恢复轨迹SFT为当前主方案；不拿D与第二种子C作配对，也不宣称普遍否定RL。

两次A/C及本轮B/D保存轨迹都已精确回放。第二种子净增26题中，完成状态变化对应+23题、双方完成作答对应+3题，仍主要伴随执行稳定性改善。62条变化案例已做AI预审：7条C改对路径有显式条件遗漏、绕过或实体替换标记，另1条别名待核；4条A原计分正确路径也有条件风险。这些不是独立人工金标，不改变基准答案或训练奖励。见[第二轮审计v2](../results/agent_feedback/replication_review_seed20261004_v2.json)与[案例分析](../results/agent_feedback/replication_case_analysis_seed20261004.json)。v1审核的浮点均值尾差误报已在v2修正，旧报告与实验统计均保留。

正式RL前的两路16题短测确认非零奖励差异、有效梯度及真实参数更新；正式B/D各200更新、200个相同训练问题组、800轨迹，分别有47/43个有效学习信号组。采样来自4999个合格训练题，不是遍历全题库。正式训练和dev共2.524 GPU小时，含既往实验累计13.883/72 GPU小时。所有训练均为syx，未使用管理员训练；无自动追加轮次。

一次反馈机制诊断已完成：GPU 0–3分别运行原两个种子的A/C，全部syx、完整dev500；保留原真实事件，仅将失败观察的error/detail通用化。A/C首种子正确数392→394、404→405，第二种子386→387、412→413，差值分别+0.4/+0.2/+0.2/+0.2个百分点，区间均包含零。正常/遮蔽共4000条轨迹回放通过。A首次干预前和无错负对照均一致；C合计5条提前分歧，其中3条原本无错，因此C的变化不能全归因于报错内容。诊断耗费1.874 GPU小时，累计15.757/72 GPU小时。见[运行前协议](../results/agent_feedback/protocol_feedback_diagnostic_v1.json)。

原418恢复对仅4个分歧动作被拒，414个可执行；这个探针只检验错误诊断内容，不是整个恢复方法的成败门槛。结果不支持具体报错文字提供稳定增益；保留原正常反馈和恢复轨迹SFT，不追加消融、训练或奖励搜索。

20:29（北京时间）已[冻结五模型留出协议](../results/agent_feedback/protocol_holdout_v1.json)：P和两续训种子的A/C，不选最佳种子。冻结时未读取留出文件，协议先以提交`3e06845`推送GitHub；20:37启动question-only推理，全部使用syx。先保存全部2000题预测并CPU回放，再读取答案，按题保留两种子联合结果计算区间。四路Agent各2.5小时、随后P0.5小时为最大预算，不是耗时预测；不追加训练。生成控制器不会自动评分，完成后由显式分析阶段统一回放和评分。当前尚无留出成绩。

五条提前分歧的[CPU复核](../results/agent_feedback/feedback_batch_drift_review_v1.json)发现批内其他题已被遮蔽、批次人数/补齐长度发生变化，支持但不能证实数值漂移解释。见[诊断收束决定](../results/agent_feedback/feedback_diagnostic_decision_v1.json)。语义核验偏好训练继续暂缓；领域条件/区间/来源表示和人工验收仍待完成。

雷达首批12条事实的来源绑定全部通过，新增12条独立审阅记录和12道答案留空的流程开发题。原资料是6份二手网页快照，真人验收和独立事实准入均为0；机器检查不能证明事实真实。01–03另补了DSCA及Redstone机构来源的[AI预审](../artifacts/thesis_direction_review/radar_primary_screening_001_003/ai_screening.json)，用于区分系统/雷达与交付/部署事件，仍不能直接据此填雷达服役或初始能力年份。见[审阅工作流](research/RADAR_REVIEW_WORKFLOW.md)和[审计](../artifacts/thesis_direction_review/radar_review_workflow/audit_report.json)。

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
