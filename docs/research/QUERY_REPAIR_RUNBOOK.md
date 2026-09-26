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
