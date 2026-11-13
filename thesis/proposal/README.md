# 开题报告当前提交入口

当前建议使用 **2026-11-13提交完善稿v3**，题目保持《基于执行反馈的小模型知识图谱问答方法研究及雷达领域应用》。

- [Word：提交完善稿v3](review_20261113/华东师范大学研究生学位论文开题报告登记表_提交完善稿_v3.docx)
- [PDF：8页预览](review_20261113/华东师范大学研究生学位论文开题报告登记表_提交完善稿_v3.pdf)
- [本轮意见取舍与收尾修订](../../docs/PROPOSAL_REVIEW_DECISIONS_20261113.md)
- [上一轮外部评价的核对与取舍](../../docs/PROPOSAL_EXTERNAL_REVIEW_RESPONSE_20261110.md)
- [原提交审阅意见](../../docs/PROPOSAL_SUBMISSION_REVIEW_20261110.md)

v3在v2基础上补清已有方法的输入、映射式和样本准入，明确2×2同状态指标及其预登记主次，说明同源图投影与类型视图的用途，并写入既定24题人工核查流程。题目、数据冻结范围、两块待做核心实验和六个月计划保持。

正文七项已填写，个人信息、学生类别、学习形式及签字日期由本人填写。原空白模板、旧精简版及v1、v2提交修订稿均保留。

从原模板重建v3 Word：

```bash
cache/proposal_env/bin/python thesis/proposal/build_proposal.py --content thesis/proposal/review_20261113/content_submission.json --output thesis/proposal/review_20261113/华东师范大学研究生学位论文开题报告登记表_提交完善稿_v3.docx --record thesis/proposal/review_20261113/build_record.json --join-section-openings
```

PDF由Apple Pages 13.2导出，已检查全部8页和正文完整性；验证记录位于对应版本目录。
