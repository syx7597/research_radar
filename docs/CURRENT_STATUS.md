# 当前状态与证据边界

审计日期：2026-09-26。依据为本地文件、源码和离线执行。主线见[研究方向](RESEARCH_DIRECTION.md)，下一步见[执行计划](ROADMAP.md)。

## 1. 阶段判断

已有知识构建、固定策略问答、类型化组合执行器和演示；新方向有题集、工具环境与训练原型。
尚未完成统一变量协议、多粒度可验证反馈、严格组合泛化数据和可靠训练实验。
本地未找到新方向权重、训练曲线与完整对照结果；不排除其他服务器有尚未同步的产物。

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

训练前仍需解决：SFT对话遗漏assistant头；GRPO参考模型/KL/多卡实现；环境与训练器奖励口径；大集合复制与token预算冲突。
本轮未擅自重写训练算法，详见[训练就绪检查](TRAINING_READINESS.md)。

## 6. 整理动作

历史方案归档至`docs/archive/proposals/`；旧报告移至`docs/archive/PROJECT_REPORT_2026-06.md`；新开题移至`docs/research/proposal.md`。
代码保持原路径。旧paper/thesis/Agent标注为历史成果。
删除5个可重建渲染目录，共947,099,791字节；PDF、转录、抽取缓存、图谱快照和唯一实验记录保留。
公开仓库排除原始/全文/大型数据及旧凭据历史，以SHA-256清单记录数据身份。原有大量整文件差异来自CRLF，发布文本统一换行。
