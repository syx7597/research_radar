> 当前论文主线：**《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》**。下面的 `current/` 为新稿入口；本目录原有八章、合并稿与图片属于历史路线，不能直接作为当前论文结论或混入新稿。

## 当前主线写作入口

2026-10-08 已将中英文摘要、绪论、相关工作、原有四章正文和总结整合为**七章工作稿**，采用通用审阅版式实际编译 PDF。新增文献库含 23 条已核对条目（10 篇方法论文、13 份厂商来源），访问范围与版本限制见文献审计。这仍不是按学校模板完成、经导师审定或经人工金标验证的定稿。

| 文件 | 用途与状态 |
|---|---|
| [current/build/main.pdf](current/build/main.pdf) | 本工作区生成的完整 PDF；编译目录不进入 Git，其他机器需从源稿重建 |
| [current/main.tex](current/main.tex)、[current/build_thesis.py](current/build_thesis.py) | 七章总入口与构建脚本；先核对四组证据表，再调用已有编译器 |
| [current/00_abstract.tex](current/00_abstract.tex)、[current/01_introduction.tex](current/01_introduction.tex) | 中英文摘要、研究问题、贡献范围与技术路线 |
| [current/02_background_related_work.tex](current/02_background_related_work.tex)、[current/07_conclusion.tex](current/07_conclusion.tex) | 相关方法比较、结论与局限 |
| [current/references.bib](current/references.bib)、[current/literature_audit.json](current/literature_audit.json) | 23 条引用及原始文献、版本和核对范围 |
| [current/manuscript_manifest.json](current/manuscript_manifest.json)、[current/manuscript_review.json](current/manuscript_review.json) | 源稿与 PDF 哈希、编译检查、抽样页面目视复核；不替代学术或人工事实审查 |
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

领域应用已完成一轮最小验证：共同合成接口适配有效；四份新来源的 24 道 AI 题上，RAG/P/A1/C1/A2/C2 正文正确数为 **13/13/16/15/16/16**，C 相对 A 为 **−1/0 题**，未建立额外领域收益。C2 支持页齐全 24/24 而正文正确 16/24，显示来源阅读与条件绑定仍会失败。六路线共用同一未适配基础回答模型；AI 审阅不是人工金标，四个相关来源、历史型号重叠与知识库整理成本均限制结论。停止本批题上的追加训练；后续八来源覆盖评价见下文，人工核验与外部有效性仍待补足，不能将这 24 道题视为整篇论文的充分验证。

2026-10-07的八来源覆盖评价也已完成：502条来源读法、96道AI参考题，raw/flat/bound各运行96题。正文正确60/73/80，联合正确56/17/52，绑定错误26/17/11；bound对flat达到预设门槛，但未在联合指标上超过raw。37个联合改善有29个原本正文都对，因此不能把全部收益称为语义推理改进。数据、执行与判分边界见[阶段决定](../results/radar_domain/coverage_v2/stage_decision.json)。

新增 [current/06b_evidence_representation.tex](current/06b_evidence_representation.tex)，已接入应用章；[current/coverage_export.py](current/coverage_export.py)从八个公开汇总/协议生成四张表与79个可追溯数据字段，输出 [CSV](current/coverage_results.csv)、[LaTeX表](current/coverage_tables.tex)及[证据清单](current/coverage_evidence_manifest.json)。不访问私有原文、QA或模型答案。2026-10-08补入统一引用解析事后对照：联合正确raw/flat/bound为56/55/68，bound−flat净13题，原始净35题完整保留；正文与错绑评分不变。第三章同步八来源502条读法、96题与固定24题来源核验流程，数据导出更新为7张表/124个字段。核验包只准备材料，当前真人核验数为0。当前没有新增GPU任务或训练；七章已合并编译，表格与引用均已接入。

从仓库根目录执行：

```sh
# 静态检查四组证据导出、章节、标签与引用，无需 TeX 编译器。
python3 thesis/current/build_thesis.py --check-only
# 使用已有 Tectonic；首次运行可能需要下载 TeX 包。
python3 thesis/current/build_thesis.py --engine-path /path/to/tectonic
# 或使用已有 XeLaTeX、BibTeX 和 ctex/Fandol 环境。
python3 thesis/current/build_thesis.py --engine xelatex
```

修改第 3–6 章后，需要运行对应的 `data_export.py`、`export_evidence.py`、`domain_export.py` 或 `coverage_export.py`（不带 `--check`）更新章节哈希，再构建。它们只读取公开聚合资料，均不读取私有逐题问答、原始预测或运行训练。旧 10,744 个实体不是已核验雷达型号，原单位和版本归档不等于查询接口已支持这些字段，AI 多代理审阅不等于人工金标。

本次采用临时目录中的 Tectonic 0.17.0 实际构建；没有全局安装编译器。`build/` 保存 PDF、日志及抽查图片，Git 仅保存源稿、构建工具和审计清单。完整 PDF 没有未解析引用、缺字或越界盒警告；目视复核的页码与范围另行登记。正式成稿仍需学校模板、导师评阅和实际来源核验；旧稿文献及旧路线结果没有自动迁入。

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
