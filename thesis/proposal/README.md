# 开题报告当前提交入口

当前建议使用 **2026-11-21主线收束稿v4**，题目保持《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》。

- [Word：主线收束稿v4](review_20261121/华东师范大学研究生学位论文开题报告登记表_主线收束稿_v4.docx)
- [PDF：8页阅读版](review_20261121/华东师范大学研究生学位论文开题报告登记表_主线收束稿_v4.pdf)
- [当前研究主线与六章框架](../../docs/THESIS_ROUTE_REVIEW.md)
- [后续执行计划](../../docs/ROADMAP.md)
- [独立AI内容复核](review_20261121/independent_scope_review.json)
- [v3意见取舍与收尾修订](../../docs/PROPOSAL_REVIEW_DECISIONS_20261113.md)
- [上一轮外部评价的核对与取舍](../../docs/PROPOSAL_EXTERNAL_REVIEW_RESPONSE_20261110.md)
- [原提交审阅意见](../../docs/PROPOSAL_SUBMISSION_REVIEW_20261110.md)

v4按照用户确定的“恢复方法为主，保留有限雷达应用”修改七项表单正文。研究内容先写恢复监督构造与公开验证，雷达来源知识和问答适配合并到一章；论文大纲由七章收束为六章。计划优先完成公开同状态评价，再按既定预算开展一次域内配对与固定评价，是否追加配对种子由开发信号决定。领域以完成应用效果和适用性分析收尾。

原题目、方法、数据冻结范围、同状态主指标及监督预算保持。六个月为原相对阶段安排，不从本次修订日重新起算提交期限。3009个正文汉字、10条参考文献，OEC条目采用已核验的ICML2026正式记录，并纳入恢复训练近邻ReTerm。本轮未新增模型运行或访问服务器。

正文七项已填写，个人信息、学生类别、学习形式及签字日期由本人填写。原空白模板、旧精简版及v1至v3稿均保留。毕业论文已按同一主线整合为[56页六章研究稿](../revisions/20261121/毕业论文_六章整合稿.pdf)，已完成成果与待做模型评价分别呈现。

从原模板重建v4 Word：

```bash
cache/proposal_env/bin/python thesis/proposal/build_proposal.py --content thesis/proposal/review_20261121/content_submission.json --output thesis/proposal/review_20261121/华东师范大学研究生学位论文开题报告登记表_主线收束稿_v4.docx --record thesis/proposal/review_20261121/build_record.json --join-section-openings
```

PDF由Apple Pages 13.2导出，已检查全部8页和正文完整性；验证记录位于对应版本目录。
