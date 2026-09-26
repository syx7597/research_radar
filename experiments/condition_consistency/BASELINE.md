# 最小公开基线：BART → KoPL

`baseline.py` 是本项目的轻量复现实验入口，不是官方基线成绩的复现声明。首轮采用 `facebook/bart-base`，固定 5,000 条官方训练样本、10 个 epoch，再导出每题 4 个 beam 候选。模型只接收问题文本，不接收答案、选项、金标实体或金标程序。

程序序列遵循 [KQA Pro 官方 BART 预处理](https://github.com/shijx12/KQAPro_Baselines/blob/master/Bart_Program/preprocess.py)：`Find <arg> London <func> Count`。`<func>`、`<arg>` 作为普通新增 token，解码时保留；分支依赖由官方 KoPL 执行器重建。训练输入可为官方 JSON 数组或准备好的 JSONL（每行含 `id/question/program`，可另有 `answer/source_index`）。

## 环境与运行

参考环境：Python 3.10+、已有 CUDA 对应的 PyTorch 2.6.0、`transformers==4.48.3`、`accelerate==1.3.0`。该入口本身不依赖 `datasets`。使用已有隔离环境，避免替换共享服务器的系统环境。

```bash
python experiments/condition_consistency/baseline.py inspect \
  --input data/condition_consistency/splits/generator_train.jsonl

torchrun --standalone --nproc_per_node=4 \
  experiments/condition_consistency/baseline.py train \
  --train-file data/condition_consistency/splits/generator_train.jsonl \
  --dev-file data/condition_consistency/splits/dev.jsonl \
  --output-dir runs/condition_consistency/bart_5k_seed20260926 \
  --model facebook/bart-base \
  --revision aadd2ab0ae0c8268c7c9693540e9904811f36177 \
  --epochs 10 --learning-rate 5e-5 --batch-size 16 --precision bf16

CUDA_VISIBLE_DEVICES=0 python experiments/condition_consistency/baseline.py generate \
  --input data/condition_consistency/splits/dev.jsonl \
  --model runs/condition_consistency/bart_5k_seed20260926/final \
  --output runs/condition_consistency/dev_candidates.jsonl \
  --candidates 4 --beams 4 --batch-size 8 --precision bf16

python -m experiments.condition_consistency.execute_predictions \
  --predictions runs/condition_consistency/dev_candidates.jsonl \
  --gold data/condition_consistency/splits/dev.jsonl \
  --kb datasets/kqa_pro/kb.json \
  --output runs/condition_consistency/dev_executed.jsonl \
  --expected-candidates 4

python -m experiments.condition_consistency.evaluate \
  --predictions runs/condition_consistency/dev_executed.jsonl \
  --output runs/condition_consistency/dev_metrics.json
```

路径是示例；以 `prepare_data.py` 实际输出及本轮运行清单为准。4 卡每卡 batch 16、梯度累积 1 时有效 batch 为 64；使用 DDP 同步训练。训练过程中验证集只记录 loss，最终使用末次 checkpoint，不根据最终评估数据挑 checkpoint。

中断后用相同训练命令追加 `--resume`（查找输出目录最后的 `checkpoint-*`）或 `--resume /absolute/checkpoint-path`。生成支持 `--resume`，会核对输入 SHA256、checkpoint 配置、生成设置和已完成记录的顺序。正常进程中断可续跑；如果文件最后一行发生写入中断，先保留原文件并修复残缺行。训练 output 必须是新目录或显式续跑，不会默默覆盖。

生成可用 `--start-index N --limit M` 划分不重叠片段，每张卡各写一个文件；合并时按准备数据中的 `id` 检查唯一性和完整性。训练 `--limit` 表示已准备文件前 N 条，不负责重新随机抽样。避免把仅加载模型权重并重新创建优化器误称为断点续训。

## 输出与验证边界

- `run_config.json`：输入哈希、训练参数、随机种子、环境版本、有效 batch、精确的数据集源/目标 token 数。
- `metrics.json`：loss、当前调用耗时、累计优化步数；支持的 Transformers 版本还记录训练输入 token 位置累计数（包含 padding）。数据集 token 数不能当作实际训练消耗 token 数。
- `final/`：可直接加载的模型和 tokenizer；周期 checkpoint 包含优化器与 Trainer 状态。
- 预测 JSONL：`id/question/candidates`。每个候选有 `program_text`、顺序、累计生成 token log probability、每 token 平均 log probability、beam score、生成长度和长度上限标记。累计 log probability 包含首次 EOS，不计后续 padding；beam score 采用 Transformers 的长度惩罚定义，不能与两种 log probability 混为一谈。
- `.meta.json`：生成配置、输入哈希、环境、进度、当前调用耗时及 token 数。生成 token 数仅统计返回的候选，不包括 beam 搜索时被淘汰的内部路径，因此不是计算成本的完整量度。

默认训练最大问题/程序长度为 256/512，生成最多新增 256 token；输入或监督目标将被截断时默认报错，只有显式 `--allow-truncation` 才允许并记录。生成使用全新的 `GenerationConfig`，不会继承预训练 checkpoint 的摘要任务参数。显式设置 `no_repeat_ngram_size=0`、`encoder_no_repeat_ngram_size=0`、两种 repetition penalty 为 1、最小生成长度为 0，并关闭词语抑制。KoPL 允许重复关系、参数和子程序，禁止重复 n-gram 会错误排除合法程序。生成参数完整记录在 `.meta.json`；解码起始、EOS、PAD 和模型配置中的 forced BOS/EOS token ID 仍保留。达到生成长度上限时可能强制补 EOS，`ended_with_eos=true` 不保证自然结束，应同时检查 `hit_generation_limit` 和执行有效性。

保留 beam 搜索原有顺序和重复候选，不填入金标程序，也不利用答案改变候选。后续评估需同时报告执行成功率、原始 top-1 准确率、oracle@4、候选去重比例以及重排准确率。oracle@4 是离线上限诊断，不能当作系统成绩。当前程序没有在生成时加入语法约束或实体链接，因此不能直接对标采用金标实体链接的论文分数。

`execute_predictions.py` 在生成结束后按 ID 离线关联金标，拒绝重复/缺失 ID 和问题不一致。金标只供后续评估，不参与解析或执行。解析错误、执行错误和空候选保留，不从分母删除；每个候选使用相同的 5 秒执行时限，统一记录有效率和错误类型。原候选文本及生成分数完整保留。

执行默认使用 `--backend baseline`，即与数据集同期发布的官方 `Program/executor_rule.py`。`--backend modern` 则显式切换到现代 KoPL 实现，仅用于单独比较；两者有语义差异，不能混用缓存或成绩。元数据记录实际 backend、相应固定源码 commit 及执行器源码文件 SHA256。`--executor-source` 可指定所选 backend 的源码目录；旧参数名 `--kopl-source` 保留为别名，目录必须与 backend 匹配。默认后端在本轮 6,500 条金标程序回放中有 6,498 条与原答案一致，剩余两条保留原标签并单独披露，不通过改标签提高回放率。

公开官方验证集不参与该最小预实验的调参；train、开发及后续评估的划分与去重由数据准备入口和 manifest 管理。待开发实验稳定后，才按照预先冻结的协议使用完整官方验证集。
