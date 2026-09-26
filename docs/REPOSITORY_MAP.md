# 仓库地图

## 当前入口

- `README.md`：总入口；`CURRENT_STATUS.md`：完成度；`RESEARCH_DIRECTION.md`：研究边界；`ROADMAP.md`：执行顺序。
- `experiments/condition_consistency/`：当前公开数据训练、执行、负例及评测入口；`docs/MINIMAL_VALIDATION.md`：已完成首轮的实测报告。
- `ca_agraphrag/`：已有策略学习原型；保留复用，暂不启动RL。新主实验先走公开KGQA与条件一致性诊断。
- `agent/composition.py`：可复用类型化执行器；`agent/planner.py`：现有提示式计划器。
- `pipeline/v3/retrieve_v3.py`：独立v3图文检索；同目录其他模块负责知识构建。
- `tests/`：不需数据或模型的边界测试。
- `scripts/audit_workspace.py`：统计和数据哈希；`check_public_tree.py`：发布索引检查。

## 历史成果

| 位置 | 后续用途 | 不代表什么 |
|---|---|---|
| 根目录`qa_*.py`、`lexicon/` | 固定策略基线和别名层 | 新策略模型 |
| `agent/` | 组合执行、报告和演示 | 已学习的长程策略 |
| `experiments/`、`results/`中除`condition_consistency/`外部分 | 旧对照、负结果及评测代码 | 当前KG上的新方法结果 |
| `experiments/ew_game/` | 旧电子战应用探索 | 最新论文核心算法 |
| `paper/`、`thesis/` | 旧论文、图表、写作素材 | 新主线已完成 |
| `archive/ontocom/` | 废弃KGE路线及失败原因 | 待投入主线 |
| `docs/archive/` | 路线变更和历史证据 | 当前工作指令 |

运行代码被多个脚本直接导入，本次保留位置。以后包结构迁移应单独提交并检查导入，避免整理破坏旧实验。

## 文档优先级

1. `MINIMAL_VALIDATION.md`：最新实测与决策；`RESEARCH_DIRECTION.md`、`ROADMAP.md`：方法边界与后续计划。
2. `CURRENT_STATUS.md`、`TRAINING_READINESS.md`：实测事实和实现问题。
3. `research/proposal.md`：开题原稿，不能覆盖新的新颖性审查。
4. `archive/proposals/`：被替代的AFR-RAG、置信度奖励等路线。
5. `archive/PROJECT_REPORT_2026-06.md`、paper/thesis：历史成果。

本轮替代的RL中心方向和计划保留在`archive/RESEARCH_DIRECTION_before_public_first.md`与`archive/ROADMAP_before_public_first.md`。

## 数据与新实验

数据留在运行所需原相对路径，公开仓库以`artifacts/data_manifest.json`记录身份，恢复方式见`ARTIFACT_POLICY.md`。
新实验使用`experiments/condition_consistency/`与`results/condition_consistency/<run_id>/`，记录代码和数据哈希、模型、seed、配置、成本和评分器。
不能覆盖旧结果；oracle、部分题量和未训练基线必须明确标记。

汇报PPT/PDF与预览保留本地，生成脚本仍在`reports/`。多个“最终版”只是历史命名，不保证内容符合最新方向。
