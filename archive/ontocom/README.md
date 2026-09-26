# 归档：OntoCom（已废弃方向）

**OntoCom = Ontology-Constrained Knowledge Graph Completion**（本体约束的知识图谱补全 / 链路预测）。
这是本项目早期探索的一个**独立研究方向**，与当前主线（雷达情报知识图谱构建 → 策略路由 GraphRAG/AGM
问答 → 双层可审计 Agent → 雷达对抗决策支持）在范式上不同：OntoCom 是嵌入式（KGE）的链路预测/补全，
而主线是符号化、可解释、可溯源的检索与推理。为保持毕设主线的连贯性与"可解释/可溯源"主旨，OntoCom
**不纳入毕业论文**，在此归档保留（论文与代码完整），便于日后作为独立成果继续使用。

归档日期：2026-06-16。

## 内容与原始位置

| 归档路径 | 原始位置 | 说明 |
|---|---|---|
| `paper/ontocom_paper.tex` `paper/tables.tex` | `paper/` | OntoCom 论文正文与表格（`ontocom_paper.tex \input{tables.tex}`） |
| `models/` | `models/`（整目录） | ontocom.py 主模型、baselines.py(PyKEEN)、path_miner.py(SAPC)、type_constraint.py(TCNS/ORE) |
| `run_wn18rr.py` | 项目根 | 在 WN18RR 上运行 OntoCom |
| `experiments/run_link_prediction.py` 等 4 个 | `experiments/` | 链路预测 / 结构恢复 / QA 集成 / 组件消融 |
| `evaluation/analysis.py` | `evaluation/` | 类型违例率等分析 |
| `scripts/finalize_paper.py` | `scripts/` | 定稿论文表格 |
| `datasets/loader.py` | `datasets/` | KGE 数据集加载器(FB15k-237 / WN18RR / RadarKG) |
| `results/{link_prediction,structural_recovery,analysis,ablation}/` `results/analysis_notes.md` | `results/` | OntoCom 实验结果与笔记 |
| `logs/{wn18rr_ontocom,scaling,structural_recovery,type_violation}.log` | `logs/` | 运行日志 |

## 重新运行提示
归档脚本中的相对导入（`from models import ...`、`from datasets.loader import ...`）依赖原项目目录结构。
如需复跑，请将相应文件移回原位置，或在 `archive/ontocom/` 下补一个把 `models/`、`datasets/` 加入 `sys.path`
的入口；所用数据集 `datasets/kqa_pro`、`graphrag_index/merged_triples.json` 仍在主项目中。
