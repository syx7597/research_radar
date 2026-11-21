# 恢复监督小论文投稿选择与适配稿

核查日期：2026-11-21。目标是在2027年6月毕业前取得正式接收；内部安排以2027-05-15为目标，给学校流程留出时间。学校认定类别、接收凭证标准及费用上限仍待作者提供。

当前建议：**有开放获取预算时优先考虑Discover Computing；若学校认可会议接收函且能够承担7月到会安排，IEA/AIE 2027提供明确的3月通知节点。** 无APC需求可考虑SN Computer Science，其当前首次决定周期较长。以下稿件已按前两个候选分别适配，尚未提交。

## 候选渠道与时间依据

| 渠道 | 适配与官方时间信息 | 费用及需要落实的条件 |
|---|---|---|
| **Discover Computing** | AI、机器学习与NLP在范围内，拟投Research；官网首次决定中位数22天，列明SCIE/Scopus。[期刊主页](https://link.springer.com/journal/10791) | 2027官方费表列USD2390、EUR1990、GBP1690，另计适用税费；费用按接收日核定。[2027费表，物理页14](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/53793606/data/v3) |
| **IEA/AIE 2027** | 2026-12-15截止，2027-03-10通知；长文12页含参考、双盲、LNCS/LNAI。[官方CFP](https://www.i-somet.org/iea-aie2027/cfp.html) | 注册费尚未公布。至少一位作者须参加7月20—22日那不勒斯会议；通知节点在毕业前，会议在7月，出版日期尚未确认；须核实学校是否认可接收函 |
| **SN Computer Science** | Original Research，AI/ML/信息检索范围；首次决定中位数96天；官网列EI Compendex/Scopus。[主页](https://link.springer.com/journal/42979) | 订阅出版路线无APC。[出版选项](https://link.springer.com/journal/42979/how-to-publish-with-us)。时间安排需预留返修余量 |
| **IEEE Access** | 官网平均约4周作接收／拒绝决定，通常4—6周发表；该速度描述不决定个稿结果。[审稿说明](https://ieeeaccess.ieee.org/about/rapid-peer-review/) | 当前APC USD2160＋税。[费用](https://ieeeaccess.ieee.org/about/article-processing-charges/)。稿件的单模型、单基准证据仍需接受技术评审 |

首次决定包含编辑筛选、拒稿及后续评审决定，和最终接收是不同事件。检索到的相关已发表文章显示，Discover的两个例子为69天、87天，另一个为316天；SN例子为47、105和317天。这些是定向选择的文章历史，未用于推算接收概率。逐篇题目、Received/Accepted日期和DOI见[springer_venue_audit.json](springer_venue_audit.json)。

其他备选保存在[会议核查](conference_venue_audit.json)与[IEEE/PeerJ核查](other_journals_audit.json)。ICIC 2027官网列3月15日截止、4月22日通知，但注册日期存在5月8日／12日冲突，费用与页限尚未完整核实，暂列后备。PeerJ部分官网访问受限，未采用第三方费用数字。

## 已完成的稿件适配

| 文件 | 用途 |
|---|---|
| [Discover Computing PDF](journal/discover_computing_review.pdf) | 12pt Arial期刊审阅稿，作者信息及声明待核定 |
| [Discover Computing Word](journal/discover_computing_review.docx) | 可编辑正文、表格及原生公式 |
| [期刊LaTeX源码包](journal/discover_computing_sources.zip) | 单文件正文、参考文献及矢量图 |
| [Cover letter草稿](journal/cover_letter_draft.txt) | 研究定位、主要结果及相关版本说明 |
| [IEA/AIE匿名PDF](conference/anonymous.pdf)、[会议源码包](conference/anonymous_source.zip) | 官方LNCS格式，12页含参考文献；175词摘要，匿名元数据与身份字符串检查通过 |
| [复现材料清单](reproducibility_inventory.json) | 划分、训练预算、结果来源和现有开放范围 |

期刊稿按[Discover作者指南](https://link.springer.com/journal/10791/submission-guidelines)调整为至少12pt字体、三级以内标题、连续数字引文、内嵌表格及可编辑公式。摘要175词、关键词5个。方法图提供嵌入字体的PDF/EPS矢量文件及1200dpi位图；已有DOI补充为完整链接。采用符合初投稿要求的作者稿排版，未将其称为出版社生产模板。

正文新增依赖映射不变量的解释，将辅助报错文字诊断移入讨论，集中呈现恢复监督构造、配对续训和执行成本。所有模型成绩保持原记录：平均78.75%→83.50%、增量4.75个百分点；414条诊断衡量后缀构造。未新增同状态、第二模型或雷达结果，原中英文稿保留在[原版本目录](../recovery_short_20261114/README.md)。

期刊所需资金、利益冲突、作者贡献及数据可用性声明已经设置。作者身份、资助情况、利益情况与未公开材料访问方式用明确待填标记保留；不会自动填成“无资助”“无利益冲突”或“全部数据公开”。AI使用说明记录本项目实际的代码、文献、写作及核查协助范围，供作者最终核定。

## 按毕业节点安排投稿

1. **11月下旬：**确定学校认可的收录／接收凭证、费用上限、作者顺序与通讯作者，完成导师和作者审阅，在期刊与会议之间选择一个首投渠道。
2. **12月上旬：**完成声明及数据访问安排，提交选定渠道。若计算条件及时恢复，优先补已经登记的同状态评价；当前稿件的主要结论仍建立在已完成的公开问答与构造诊断上。
3. **选择IEA/AIE时：**12月15日前提交；3月10日依据结果安排定稿或后续投稿，4月10日前注册、4月15日前交定稿。实际注册与到会费用需在提交前核实。
4. **选择期刊时：**为一至两轮返修预留时间，收到意见后集中完成回应；2月中旬根据实际编辑状态评估进度。转投前确认原稿已结束评审或撤稿完成，两个适配版本分别用于择一投稿。
5. **5月中旬：**目标取得可供学校使用的正式接收凭证。审稿进度由编辑及同行评审决定，本计划保留时间余量，不作接收日期承诺。

现有稿件适合定位为“参考程序驱动的恢复监督构造与受控实证”。最有价值的内容补强是同状态模型评价，它直接对应恢复能力问题；期刊格式调整已经完成，后续研究仍沿用原路线和预算。

## 重建与复核

```sh
python3 papers/submission_20261121/prepare_content.py
cache/proposal_env/bin/python papers/submission_20261121/render_figures.py
python3 papers/submission_20261121/build_journal.py
python3 papers/submission_20261121/build_conference.py
```

工具使用本地Tectonic、Pandoc和Python文档环境；LNCS官方模板下载记录见会议审计。编译中间文件位于各版本的忽略目录。投稿前以本次作者指南核查和作者核定信息再做最终检查。
