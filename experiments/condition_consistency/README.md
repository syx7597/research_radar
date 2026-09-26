# 公开 KGQA 最小验证

用于毕业论文的首轮可复现实验。先确认公开数据、执行语义、模型训练和同候选对照可运行，再判断条件一致性机制是否值得扩大。

- [首轮协议](MINIMAL_PROTOCOL.md)：训练预算、比较对象和结论范围。
- [数据准备](DATA.md)：固定镜像、哈希、去重划分、官方执行器及残余差异。
- [生成器与执行入口](BASELINE.md)：训练、断点、候选导出及离线执行。

## 运行

先按数据准备说明取得固定输入和官方执行器，在独立虚拟环境安装本目录的`requirements.txt`。下载`facebook/bart-base`固定revision并放到`models/bart-base/`，含模型、配置和tokenizer文件；可使用Hugging Face Hub下载工具，权重SHA256应为`1bd3ac8e5b3ac71c77cf18fbcd8b113e0bfceced94c5cfbbb9a2b4bb4781190b`。

在实验工作区根目录启动四卡流程：

```bash
bash experiments/condition_consistency/run_pilot.sh
```

默认输出`results/condition_consistency/pilot_bart5k_seed20260926/`，可用`RUN_DIR`指定新目录。
此包装脚本用于新运行；需要断点恢复时使用`baseline.py`的显式`--resume`接口，不直接覆盖已有运行。
`status.json`记录阶段；每个阶段有独立日志。训练后分四卡生成同K候选，依次做官方执行、规则诊断、同题同输出类型的负例匹配及三个轻量排序种子。
匹配样本为零时保留这一诊断并跳过排序，不将其误报成算法已经有效。

公开内容包括配置、来源、输入/代码哈希和汇总指标。权重、完整题目/预测行和训练日志保留在实验主机或本地忽略目录，不进入Git。

## 指标解释

`top1`是生成器排序首位候选，`first_valid`只过滤不可执行候选；`condition_rule`是固定的数值/比较/词项覆盖规则。`oracle_at_k`使用答案离线检查候选上限，不能部署。
`natural`与`targeted`使用相同问题、同一轻量线性评分器与训练步数，只替换负例来源；直接配对比较保存在`pilot_summary.json`。
此处线性特征模型只是低成本探针，未实现原方向中完整的神经事实作用域评分器。不能把规则或线性探针的结果直接描述成该方法已获验证。

500题来自train内部开发留出。三个排序种子共用一个生成器和候选缓存，不能视作三个独立完整系统运行。正结果需要冻结方法后补充正式评测，负结果也须保留。

## 无GPU协议检查

```bash
python3 -B -m unittest experiments.condition_consistency.test_ranking -v
```

这些检查验证金标隔离、评测分母、负例匹配及执行时限，不验证模型准确率。
