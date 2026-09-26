# 查询修复实验运行说明

从仓库根目录运行。协议与已冻结参数见[实验协议](QUERY_REPAIR_PROTOCOL.md)。以下为复现实验入口，不会重新训练BART；原始数据、候选缓存、模型权重不在公开Git仓库中，需要在实验机上已有固定版本。不要将SSH凭据写入脚本或结果文件。

## 1. 路径与环境

使用此前实验虚拟环境及固定检查点；依赖版本见首轮环境记录。示例另建结果目录，避免覆盖原pilot和本轮首份冻结报告。

```bash
export PYTHON_BIN=.venv/bin/python
export PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PILOT_DIR=results/condition_consistency/pilot_bart5k_seed20260926
export MODEL_DIR="$PILOT_DIR/generator/final"
export RUN_DIR=results/condition_consistency/query_repair_reproduction
mkdir -p "$RUN_DIR"
```

需要的原始文件是`datasets/kqa_pro/{train,val,kb}.json`、原6500题划分、官方baseline源码，以及`$MODEL_DIR`中的固定模型和tokenizer。`results/condition_consistency/pilot_bart5k_seed20260926/checkpoint_manifest.json`记录模型身份，评估器会核验缓存使用的权重/配置。

## 2. 冻结划分与执行器检查

```bash
"$PYTHON_BIN" -m experiments.condition_consistency.prepare_repair_splits
"$PYTHON_BIN" -m unittest \
  experiments.condition_consistency.test_repair \
  experiments.condition_consistency.test_repair_selection \
  experiments.condition_consistency.test_executor_state
"$PYTHON_BIN" scripts/check_executor_stability.py \
  --output "$RUN_DIR/executor_validation.json"
```

划分脚本只接受与已冻结内容完全一致的输出。正逆序回放只用已见的6500题，不读取新holdout金标。`audit_query_repair.py`用于原pilot错误的离线诊断；其中含金标对照信息，不能把审计结果作为推理输入。

## 3. 开发候选、修复与评分

新生成calibration的4个候选和dev的8个候选；dev的4个候选直接复用首轮`$PILOT_DIR/dev_generated.jsonl`。生成脚本只接收question-only文件。

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m experiments.condition_consistency.baseline generate \
  --input data/condition_consistency/repair_splits/calibration.questions.jsonl \
  --output "$RUN_DIR/calibration_beam4_generated.jsonl" \
  --model "$MODEL_DIR" --local-files-only \
  --beams 4 --candidates 4 --batch-size 8 --precision bf16

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m experiments.condition_consistency.baseline generate \
  --input data/condition_consistency/repair_splits/dev_diagnostic.questions.jsonl \
  --output "$RUN_DIR/dev_beam8_generated.jsonl" \
  --model "$MODEL_DIR" --local-files-only \
  --beams 8 --candidates 8 --batch-size 8 --precision bf16

bash experiments/condition_consistency/run_repair_stage.sh development
```

最后一条命令并行处理dev/calibration的global/local模式，使用4张可见GPU重评分，并以相同执行器执行dev beam8。`repair.py`提出每题最多4个新增候选，`score_repairs.py`由固定BART重评分，`execute_unlabeled.py`执行beam8。三个入口均不读取金标。

## 4. 校准并冻结策略

```bash
"$PYTHON_BIN" -m experiments.condition_consistency.evaluate_repairs calibrate \
  --global-cache "$RUN_DIR/calibration_global_scored.jsonl" \
  --local-cache "$RUN_DIR/calibration_local_scored.jsonl" \
  --gold data/condition_consistency/repair_splits/calibration.gold.jsonl \
  --output "$RUN_DIR/frozen_policy.json"

"$PYTHON_BIN" -m experiments.condition_consistency.evaluate_repairs evaluate \
  --split dev_diagnostic \
  --global-cache "$RUN_DIR/dev_global_scored.jsonl" \
  --local-cache "$RUN_DIR/dev_local_scored.jsonl" \
  --beam8-cache "$RUN_DIR/dev_beam8_executed.jsonl" \
  --gold data/condition_consistency/repair_splits/dev_diagnostic.gold.jsonl \
  --policy "$RUN_DIR/frozen_policy.json" \
  --output "$RUN_DIR/dev_metrics.json"
```

校准命令严格核验calibration文件的哈希与ID，不能传holdout替代。开发诊断显式使用`--split dev_diagnostic`；不写该参数时评估器默认要求holdout。当前首份冻结策略为global `(alpha=0.5, min_match=0.8, margin=-0.25)`、local `(0.5, 0.4, 0.0)`；复现实验不得根据holdout结果修改策略。校准缓存和策略文件保留全部尝试，并在评估时核验代码和数据身份。

## 5. 生成holdout候选，再一次性评分

以下阶段应在算法、执行器与策略冻结后运行。生成和修复不打开holdout金标；只有最后一个评估命令读取其答案。

```bash
bash experiments/condition_consistency/run_repair_generation.sh
bash experiments/condition_consistency/run_repair_stage.sh holdout

"$PYTHON_BIN" -m experiments.condition_consistency.evaluate_repairs evaluate \
  --split holdout \
  --global-cache "$RUN_DIR/holdout_global_scored.jsonl" \
  --local-cache "$RUN_DIR/holdout_local_scored.jsonl" \
  --beam8-cache "$RUN_DIR/holdout_beam8_executed.jsonl" \
  --gold data/condition_consistency/repair_splits/holdout.gold.jsonl \
  --policy "$RUN_DIR/frozen_policy.json" \
  --output "$RUN_DIR/holdout_metrics.json"
```

`run_repair_generation.sh`专用于当前2,000题holdout：4张GPU分别生成beam4/beam8各两个1000题分片，再按原顺序合并。它不是任意长度数据的通用分片器；不要直接替换成500题或其他长度的输入。保留各分片meta和日志，核验合并后ID完整、无重复。

最终报告使用全部2000题分母，包含top1、first-valid、beam8、global/local修复、接受策略消融、候选覆盖上限，以及相对对照的净纠正、改错和配对置信区间。时间、实际执行次数、生成和重评分token数分别从对应生成/修复/评分meta汇总；4+4候选与8候选不代表算力开销相等。

入口通常拒绝覆盖已存在的候选、策略或报告。需要重现时使用新的`RUN_DIR`，并保留首次冻结报告；不能通过改输出路径把已经看过的holdout重新描述成未接触评估。

## 6. v2：保护原查询后的完整官方validation复现

v2仅增加选择阶段的保护规则，方法和停止条件见[第二轮协议](QUERY_REPAIR_V2_PROTOCOL.md)。它复用v1修复器、评分器、执行器及固定BART，依赖已保留的v1 calibration候选缓存、各缓存的`.meta.json`、v1冻结策略、训练/校准划分清单和模型清单。公开仓库不包含原始候选缓存和模型权重，只有报告文件时无法直接运行这些命令。

旧2,000题已用于提出v2假设，此后只能称已见诊断；完整official val也不是hidden test，旧项目使用过它。不得把复现或另建目录描述为重新获得一份独立未接触集。

### 6.1 明确v1输入和v2输出目录

下例读取本轮保留的正式v1数据，在新目录写复现输出，避免覆盖正式v2结果。

```bash
export QR_V1_DIR=results/condition_consistency/query_repair
export QR_V2_DIR=results/condition_consistency/query_repair_v2_reproduction
export QR_SPLIT_MANIFEST=results/condition_consistency/query_repair/split_manifest.json
export QR_VAL_MANIFEST=results/condition_consistency/query_repair/v2/official_val_manifest.json
mkdir -p "$QR_V2_DIR"

"$PYTHON_BIN" -m experiments.condition_consistency.prepare_repair_val
```

准备命令复核官方val源文件，生成11797条`official_val.questions.jsonl`及独立`official_val.gold.jsonl`，不执行或分析金标。已有输出必须与冻结字节一致。上例显式引用正式val清单；也可将清单原样复制到新目录并修改`QR_VAL_MANIFEST`，不可自行编辑其中路径、ID或哈希。

前五节若使用其他`RUN_DIR`重新生成了v1缓存和策略，应将`QR_V1_DIR`设为那一个对应目录，并在后续参数中保持一致；不能混用新缓存与不匹配的旧策略。`RUN_DIR`环境变量不会自动改变下面Python入口的`--run-dir`、`--v1-policy`或清单默认值，故全部显式指定。Python驱动的默认v2目录是本轮正式的`results/condition_consistency/query_repair/v2`，复现时务必传入新的`--run-dir`。

### 6.2 只在原500题calibration重新校准

```bash
"$PYTHON_BIN" -m experiments.condition_consistency.evaluate_repair_v2 calibrate \
  --global-cache "$QR_V1_DIR/calibration_global_scored.jsonl" \
  --local-cache "$QR_V1_DIR/calibration_local_scored.jsonl" \
  --gold data/condition_consistency/repair_splits/calibration.gold.jsonl \
  --v1-policy "$QR_V1_DIR/frozen_policy.json" \
  --manifest "$QR_SPLIT_MANIFEST" \
  --val-manifest "$QR_VAL_MANIFEST" \
  --checkpoint-manifest "$PILOT_DIR/checkpoint_manifest.json" \
  --output "$QR_V2_DIR/frozen_policy.json"
```

该命令使用原73项网格，同时冻结v1/v2代码、模型权重、缓存身份及官方val清单。它不会根据旧2,000题或official val的答案选择阈值。`--output`必须是尚不存在的新文件；若直接复用正式v2策略，应原样保留其依赖的v1策略和清单，不再覆盖式校准。

### 6.3 生成官方val，再执行和重评分

```bash
"$PYTHON_BIN" -m experiments.condition_consistency.run_repair_val generate \
  --run-dir "$QR_V2_DIR" \
  --input data/condition_consistency/repair_splits/official_val.questions.jsonl \
  --model "$MODEL_DIR"

"$PYTHON_BIN" -m experiments.condition_consistency.run_repair_val execute \
  --run-dir "$QR_V2_DIR" \
  --input data/condition_consistency/repair_splits/official_val.questions.jsonl \
  --model "$MODEL_DIR"
```

`generate`使用4张GPU，每张先后完成自己分片的beam4与beam8；四片题数为3000、3000、3000、2797。合并要求`val:0`到`val:11796`完整且无重复。`execute`要求同目录已有冻结v2策略；它并行运行12个CPU任务（4片×global/local/beam8），合并后用2张GPU分别重评分global和local。两个阶段都只读问题和候选，不打开val金标。

主要产物为`val_global_scored.jsonl`、`val_local_scored.jsonl`、`val_beam8_executed.jsonl`及对应meta。每个分片的meta和日志也应保留。合并失败可能留下未完成文件；不要将其当作可用结果，使用新输出目录重新复现并保留失败日志。驱动的部分日志/阶段文件不是覆盖保护对象，新目录是避免混入历史产物的必要措施。

### 6.4 完整官方validation一次性评估

```bash
"$PYTHON_BIN" -m experiments.condition_consistency.evaluate_repair_v2 evaluate \
  --split official_val \
  --global-cache "$QR_V2_DIR/val_global_scored.jsonl" \
  --local-cache "$QR_V2_DIR/val_local_scored.jsonl" \
  --beam8-cache "$QR_V2_DIR/val_beam8_executed.jsonl" \
  --gold data/condition_consistency/repair_splits/official_val.gold.jsonl \
  --v1-policy "$QR_V1_DIR/frozen_policy.json" \
  --policy "$QR_V2_DIR/frozen_policy.json" \
  --manifest "$QR_SPLIT_MANIFEST" \
  --val-manifest "$QR_VAL_MANIFEST" \
  --checkpoint-manifest "$PILOT_DIR/checkpoint_manifest.json" \
  --output "$QR_V2_DIR/official_val_metrics.json"
```

本轮正式运行使用相同文件名，位于`results/condition_consistency/query_repair/v2/`，包括最终`v2/official_val_metrics.json`；上例将复现报告分开保存。必须显式指定`--split official_val`。v2评估器默认的`holdout_seen_diagnostic`是已见旧检查集，不能用于宣称独立验证。评估器同时核验所有11797题的ID/题目、金标与缓存哈希、候选预算、v1/v2冻结身份，并报告v1/v2方法和beam4/beam8对照。

完整官方val首次评估后即结束这一轮开发，不根据其错误增补规则或调整阈值。另起路径重复运行不改变这一停止条件。

### 6.5 复现精度与成本口径

固定生成器种子、软件版本、模型权重、批大小及分片方式；生成和教师强制评分采用BF16。不同GPU环境、CUDA内核或批次/填充形状仍可能带来微小浮点差异，阈值附近的选择可能因此变化，不能承诺跨环境逐浮点值完全一致。需保留实际配置、哈希和复现差异，不按val结果更改阈值以追平历史成绩。

当前v2保护规则发生在最终选择阶段：系统仍先生成、执行并重评分全部预算内候选。因此，**v2并未实测通过该guard节省运行成本**，不能把“保留原输出的题数”直接换算为省去的执行或GPU开销。

合并meta中的`elapsed_seconds`是整段并行CPU阶段墙钟时间，不是每种方法独立延迟；`sum_shard_elapsed_seconds`是分片耗时之和，也不是墙钟时间。GPU评分同样并行。分别报告实际候选数、前缀/完整程序调用、token及分阶段耗时，不能用这些并行分段数值直接宣称某方法更快。
