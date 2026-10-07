> 当前论文主线：**《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》**。下面的 `current/` 为新稿入口；本目录原有八章、合并稿与图片属于历史路线，不能直接作为当前论文结论或混入新稿。

## 当前主线写作入口

| 文件 | 用途与状态 |
|---|---|
| [current/03_radar_data_construction.tex](current/03_radar_data_construction.tex) | 数据构建章节：历史库存、旧开发读法、合成实例、新来源材料及表示、审阅、划分边界 |
| [current/data_review.tex](current/data_review.tex) | 第三章独立 XeLaTeX 审阅入口 |
| [current/data_export.py](current/data_export.py)、[current/data_tables.tex](current/data_tables.tex) | 从公开清单与审计汇总重建数据表，不访问私有原始资料或 QA |
| [current/data_results.csv](current/data_results.csv)、[current/data_evidence_manifest.json](current/data_evidence_manifest.json) | 数据数量、来源字段及章节/实现哈希；库存数量不视为已核验事实数量 |
| [current/04_execution_feedback_method.tex](current/04_execution_feedback_method.tex) | 方法章节：执行状态、恢复后缀、句柄重映射、监督掩码及匹配预算；绑定冻结实现 |
| [current/05_public_experiments.tex](current/05_public_experiments.tex) | 公开实验章节：2,000 题留出结果、两个续训种子、成本、RL 与反馈诊断、局限 |
| [current/results_tables.csv](current/results_tables.csv) | 机器可读结果字段、来源 JSON 路径与派生公式；不含逐题数据 |
| [current/evidence_manifest.json](current/evidence_manifest.json) | 汇总来源、冻结实现、章节及生成表格的 SHA-256 |
| [current/export_evidence.py](current/export_evidence.py) | 重建表格和清单；`--check` 检查数字及文件一致性 |
| [current/review.tex](current/review.tex) | 两章的独立 XeLaTeX 审阅入口；需要 `ctexrep`、`amsmath`、`booktabs` 等常用包 |
| [current/06_radar_application.tex](current/06_radar_application.tex) | 应用章节：来源表示、直接迁移诊断、共同接口适配、四来源六路线评价及失败边界 |
| [current/domain_review.tex](current/domain_review.tex) | 第六章独立 XeLaTeX 审阅入口，含四份制造商资料的来源链接 |
| [current/domain_export.py](current/domain_export.py) | 只读公开聚合与来源索引，重建应用章表格；不访问私有 QA 或运行训练 |
| [current/domain_tables.tex](current/domain_tables.tex)、[current/domain_results.csv](current/domain_results.csv) | 六路线完整结果、成本、迁移前后及来源分组；CSV 记录来源字段和派生口径 |
| [current/domain_evidence_manifest.json](current/domain_evidence_manifest.json) | 应用章输入、章节、导出结果的哈希绑定与适用边界 |

新稿已有依据的核心结论：恢复轨迹 SFT 在两个共享初始模型的续训种子上，相对同监督 token 预算的普通续训，项目留出集准确率平均提高 **4.75 个百分点**，问题配对 95% 区间 **[3.53, 5.98]**。它不是官方隐藏测试结果，不代表全流程两种子独立复现；C/P 总推理 token 约为 24 倍，详细报错文字和短程 RL 的额外收益均未得到证实。

领域应用已完成一轮最小验证：共同合成接口适配有效；四份新来源的 24 道 AI 题上，RAG/P/A1/C1/A2/C2 正文正确数为 **13/13/16/15/16/16**，C 相对 A 为 **−1/0 题**，未建立额外领域收益。C2 支持页齐全 24/24 而正文正确 16/24，显示来源阅读与条件绑定仍会失败。六路线共用同一未适配基础回答模型；AI 审阅不是人工金标，四个相关来源、历史型号重叠与知识库整理成本均限制结论。停止本批题上的追加训练，转入证据整理；后续覆盖与评价可信度工作仍未完成，不能将 24 道题视为整篇论文的充分验证。

从仓库根目录执行：

```sh
python3 thesis/current/data_export.py --check
python3 thesis/current/data_export.py
python3 thesis/current/export_evidence.py --check
python3 thesis/current/export_evidence.py
python3 thesis/current/domain_export.py --check
python3 thesis/current/domain_export.py
xelatex -output-directory=thesis/current thesis/current/data_review.tex
xelatex -output-directory=thesis/current thesis/current/data_review.tex
xelatex -output-directory=thesis/current thesis/current/review.tex
xelatex -output-directory=thesis/current thesis/current/review.tex
xelatex -output-directory=thesis/current thesis/current/domain_review.tex
xelatex -output-directory=thesis/current thesis/current/domain_review.tex
```

三组 Python 命令分别检查和重建数据构建、公开实验、领域应用的聚合导出，均不读取私有逐题问答、原始预测或运行训练。第三章明确：旧 10,744 个实体不是已核验雷达型号，原单位和版本归档不等于查询接口已支持这些字段，AI 多代理审阅不等于人工金标。当前写作环境没有 `xelatex`，尚未编译或目视检查新稿 PDF。章节编号暂定三至六章；正式成稿仍需学校模板、核实后的相关文献，以及预先定义覆盖范围的后续来源与评价核验。已有旧稿的参考文献未自动迁入，以免把未经本轮核对的条目当成新稿证据。

最新项目状态见 [项目首页](../README.md) 和 [当前状态](../docs/CURRENT_STATUS.md)。

---

## 以下为历史路线写作记录（保留供追溯）

# 硕士毕业论文 · 写作进度与索引

**题目：** 面向雷达情报的可解释知识图谱问答与对抗决策支持方法研究

主线：**构建知识图谱 → 可解释问答 → 可审计分析 → 对抗决策支持**；
贯穿原则：**类型化算子** + **可解释/可溯源/不确定性显式建模**。

## 文件与章节进度

| 文件 | 章节 | 状态 | 说明 |
|---|---|---|---|
| `00_abstract.md` | 题目 + 中英文摘要 + 关键词 | ✅ 初稿 | 四大贡献与硬指标；数字已与第四章对齐(88.6%/+35.9pp) |
| `01_introduction.md` | 第一章 绪论 | ✅ 初稿 | 背景、国内外研究现状(真实文献)、问题挑战、创新点、结构 |
| `02_background.md` | 第二章 相关理论与技术基础 | ✅ 初稿 | KG/RAG/GraphRAG/CoT/ReAct/KGQA/雷达与电子战 |
| `03_kg_construction.md` | 第三章 雷达领域可溯源知识图谱构建 | ✅ 初稿 | 含属性图形式化、抽取流程图、图3-2/3-3(真实数据) |
| `04_strategy_routed_graphrag.md` | 第四章 策略路由 GraphRAG 与 AGM | ✅ 初稿 | AGM形式化、六算子、实验表、图4-1/4-2 |
| `05_agent.md` | 第五章 双层可审计分析智能体 | ✅ 初稿 | 计划树文法、可审计性定理、架构图5-1 |
| `06_ew_countermeasure.md` | 第六章 雷达对抗可解释决策支持 | ✅ 初稿 | 自草稿迁入，已统一为第六章/6.x，含图6-1 |
| `07_cognitive_ew.md` | 第七章 面向自适应雷达的知识引导在线对抗学习 | ✅ 初稿 | 新增；序贯对抗形式化、doctrine 信念学习器、CUSUM、双图协同、命题7.2–7.4、实验五项；实证见 experiments/ew_game/ |
| `08_conclusion.md` | 第八章 总结与展望 | ✅ 初稿 | 总结五贡献+两主线、创新点(4)、局限、展望 |
| `references.md` | 参考文献 | ✅ 初稿 | **32 条**，含真实会议/期刊与 arXiv 编号 |
| `figures/` | 插图 + 生成脚本 | ✅ | gen_figures.py 由真实数据出图；**9 张 PNG** 已生成并嵌入各章 |

**全文初稿八章已成，完成通审完善、端到端案例、符号表，并合并为单文件。**

- `notation.md`：符号表 + 缩略语（前置，已并入合并稿）
- `thesis_full.md`：**合并单文件**（题目→符号表→第一~八章→参考文献，便于排版）
- 第六章新增 §6.9 端到端案例（S-400 防空群对抗推演，输出由系统实跑生成）
- 第七章（新增）：把对抗推进到自适应雷达下的在线学习；定位"知识真实、动态仿真、非实测"，实验代码与结果在 `experiments/ew_game/`（`cognitive.py`/`solver.py`/`theory.py`/`cusum.py`/`kg_prior.py` + `RESULTS_cognitive.md`）
- `md2tex.py` + `thesis.tex` + **`thesis.pdf`**：Markdown→LaTeX 转换器与 XeLaTeX 编译产物（**54 页 PDF，含目录/图表/公式，中文正常**）

### 重新生成 PDF
```
python thesis/figures/gen_figures.py     # 生成插图
python thesis/md2tex.py                   # 合并稿 thesis_full.md → thesis.tex
xelatex thesis.tex && xelatex thesis.tex  # 编译两遍(目录)
```

### 插图清单（均嵌入正文）
- 图1-1 总体技术路线（架构路线图）
- 图3-1 多源抽取流程（图框）·图3-2 关系分布·图3-3 来源分布（真实数据）
- 图4-1 总体准确率·图4-2 逐算子消融·图4-3 分类别对比（论文数据）
- 图5-1 双层架构（图框）·图5-2 智能体评测
- 图6-1 威胁库国别构成·图6-2 结论多样性提升（真实数据）

## 待办（打磨）
- [ ] 全文统一记号表；中英文术语与图号空格格式终校
- [ ] 参考文献按 GB/T 7714 终校页码/卷期；研究现状可补本领域中文文献
- [ ] 各章可按导师意见加深/补端到端案例 walkthrough
- [ ] 转 Word/LaTeX 排版（公式与图已就绪）

## 备注
- 文献综述中的引用均来自联网核对的真实论文（GraphRAG[5]、RoG[6]、ToG[7]、ReAct[8]、CoT[9]、认知电子战综述[12] 等），定稿前请按学校模板二次核对页码/卷期。
- 第六章定位反复声明为"doctrine 级、开源、决策支持、非实测"，答辩时应主动说明数据来源与保密边界。
