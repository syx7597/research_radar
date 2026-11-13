# 错误恢复小论文：中英文完整稿

完成日期：2026-11-14。

**中文题目：面向小模型可执行知识问答的依赖一致恢复监督**

**English title: Dependency-Consistent Recovery Supervision for Executable Knowledge Question Answering with Small Language Models**

本稿从毕业论文中提取公开知识问答与恢复训练研究，形成可独立阅读的实证论文。研究对象是小模型在多步知识查询中偏离参考程序后的继续执行：利用已有程序，将参考依赖映射到实际结果句柄，重放核验后缀，并以保留分歧历史的后缀监督训练模型。中英文版本表达同一项研究，正文均不含雷达应用。

## 阅读文件

| 版本 | PDF 阅读版 | Word 编辑版 | Markdown 正文 |
|---|---|---|---|
| 中文，PDF 9 页 | [中文 PDF](recovery_supervision_zh.pdf) | [中文 Word](recovery_supervision_zh.docx) | [中文正文](manuscript_zh.md) |
| English, PDF 10 pages | [English PDF](recovery_supervision_en.pdf) | [English Word](recovery_supervision_en.docx) | [English manuscript](manuscript_en.md) |

两版均有摘要、引言、相关工作、方法、实验设置、结果分析、讨论、结论及13条参考文献；包含方法图与4张表。Word 中的公式可编辑。作者和单位留待填写。PDF 采用通用阅读排版，选定投稿渠道后可改用相应模板。

## 本文形成的研究贡献

1. **恢复目标构造。** 从固定初始模型的失败回合中保留共同成功前缀及一次实际参考分歧，以参考程序提供后续逻辑动作，通过依赖句柄映射和确定性重放构造可执行监督。
2. **受控续训比较。** 在共享初始化、成功材料、题目顺序及监督 token 数的条件下，比较普通续训与恢复续训。Qwen2.5-3B-Instruct 在 KQA Pro 项目留出集的两个续训种子上，平均准确率由78.75%升至83.50%，提高4.75个百分点；按题目配对重采样的95%区间为[3.53, 5.98]个百分点。
3. **执行与构造分析。** 报告未完成回合、无效调用及累计输入/生成成本，并对历史训练配对做CPU构造诊断。414个可比样例中，直接拼接的全部后缀调用可执行数为65，依赖映射后为414；这些数值衡量目标构造，不计作模型答题率。

## 文献对比与论文定位

已核查 DAgger、Agent-R、STeP、OEC、ReTerm、SCoRe、Fission-GRPO 等训练近邻，并整理推理时修订与基础模型文献。最近的方法比较见[文献核查与定位](literature_review.md)，逐项来源和阅读范围见[literature_audit.json](literature_audit.json)。

[OEC](https://proceedings.mlr.press/v306/lauffer26a.html)和[ReTerm](https://arxiv.org/html/2610.18484v2)已经采用学生历史上的专家接续与恢复监督。本稿将贡献落实为**给定参考程序时的依赖一致目标构造，以及匹配条件下的小模型续训实证**。相关工作表比较监督来源和构造机制；本轮未复现这些论文的实验分数。

现有结果已能组成一篇有方法、对照、统计和机制诊断的完整研究稿。若继续面向投稿，优先完成已登记的**同状态模型评价**：让A/C分别从相同普通历史和相同分歧历史接续，检验恢复能力增量。它能将自由执行的整体收益与恢复行为直接联系起来。该项尚无实测结果，正文按待开展分析说明。

仅当后续希望声称某个组件具有独立增益时，再开展对应训练消融；当前准确率结果评价完整训练方案。论文基于一个3B模型、一个基准和两个共享初始模型及语料的续训种子。研究稿已经完成，投稿适配与接收结果仍取决于目标渠道及同行评审。

## 数据与核验

- 5000/500/2000为项目训练、开发及留出划分，来源为KQA Pro官方训练数据；初始训练及一次性程序基线使用4999条通过验收的成功示范。新留出集排除历史题目/程序组及官方验证集精确组匹配。
- 两组续训分别使用相同788534个监督token；一次性程序基线另有训练配置，作为执行形式及成本参照。
- [证据说明](evidence_notes.md)、[证据审计](evidence_audit.json)与[共享数字表](tables_data.json)保存来源和口径。235项数字绑定到公开汇总文件的路径、指针及哈希。
- 独立AI审阅完成双语数字和方法表述核对，见[双语复核记录](bilingual_review.json)。本轮另核对25个已保存模型输出及相关文件的哈希；没有重新生成这些预测。
- 本轮新增模型运行数为0，未访问服务器。现有模型权重未在本机找到，已存预测及汇总证据可用。原始逐题数据、下载论文全文、编译预览和工具缓存均不随此稿提交。
- [构建清单](build_manifest.json)记录正文与成品哈希；[交付核验](review_manifest.json)记录PDF、Word与抽查页面情况。AI审阅身份如实保留。

## 本地重建

从仓库根目录运行，只检查数字表、来源哈希、双语正文与引用：

```sh
python3 papers/recovery_short_20261114/build_paper.py --check-only
```

完整构建：

```sh
python3 papers/recovery_short_20261114/build_paper.py
```

当前环境使用Tectonic 0.17.0、Pandoc 3.14和macOS Swift/PDFKit。工具分别位于`cache/thesis_build_tools/tectonic_0.17.0/tectonic`与`cache/paper_build_tools/pandoc`；可用`--tectonic`、`--pandoc`覆盖路径。Word/Markdown引用使用本地`cache/paper_build_tools/ieee.csl`，来源为[官方CSL样式库](https://github.com/citation-style-language/styles/blob/master/ieee.csl)。这些依赖属于本地缓存，需在新环境自行准备。

LaTeX入口为`paper_zh.tex`、`paper_en.tex`；正文为`manuscript_zh.tex`、`manuscript_en.tex`。`export_tables.py`从同一份核验数字生成双语表格。修改正文后重新构建，并更新独立审阅和交付记录，避免沿用旧哈希。
