# 开题报告当前提交入口

当前建议使用 **2026-11-10提交修订稿v2**，题目保持《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》。

- [Word：提交修订稿v2](review_20261110_v2/华东师范大学研究生学位论文开题报告登记表_提交修订稿_v2.docx)
- [PDF：8页预览](review_20261110_v2/华东师范大学研究生学位论文开题报告登记表_提交修订稿_v2.pdf)
- [外部评价的核对与取舍](../../docs/PROPOSAL_EXTERNAL_REVIEW_RESPONSE_20261110.md)
- [原提交审阅意见](../../docs/PROPOSAL_SUBMISSION_REVIEW_20261110.md)

v2补充了恢复构造的前提与映射规则、A/C配对定义、公开图查询与雷达类型视图的联系、同状态和失败分层分析、种子与来源计数口径，以及已有反馈文字诊断。题目、数据冻结范围、两块待做核心实验和六个月计划保持。

正文七项已填写，个人信息、学生类别、学习形式及签字日期由本人填写。原空白模板、旧精简版和第一版提交修订稿均保留。

从原模板重建v2 Word：

```bash
cache/proposal_env/bin/python thesis/proposal/build_proposal.py --content thesis/proposal/review_20261110_v2/content_submission.json --output thesis/proposal/review_20261110_v2/华东师范大学研究生学位论文开题报告登记表_提交修订稿_v2.docx --record thesis/proposal/review_20261110_v2/build_record.json --join-section-openings
```

PDF由Apple Pages 13.2导出，已检查全部8页和正文完整性；验证记录位于对应版本目录。
