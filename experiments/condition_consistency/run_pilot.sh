#!/usr/bin/env bash
# Run from the experiment repository root after preparing data, model and .venv.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-.venv/bin/torchrun}"
RUN_DIR="${RUN_DIR:-results/condition_consistency/pilot_bart5k_seed20260926}"
MODEL_DIR="${MODEL_DIR:-models/bart-base}"
SPLIT_DIR="${SPLIT_DIR:-data/condition_consistency/splits}"
KB_PATH="${KB_PATH:-datasets/kqa_pro/kb.json}"
mkdir -p "$RUN_DIR"
status() {
  "$PYTHON_BIN" -c 'import json,sys,time,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})+"\n")' "$RUN_DIR/status.json" "$1"
}
trap 'exit_code=$?; if [ "$exit_code" -ne 0 ]; then status failed; fi' EXIT

status recording_environment
"$PYTHON_BIN" -m experiments.condition_consistency.record_environment --output "$RUN_DIR/environment.json" --model "$MODEL_DIR"
status training
"$TORCHRUN_BIN" --standalone --nproc_per_node=4 experiments/condition_consistency/baseline.py train \
  --train-file "$SPLIT_DIR/generator_train.jsonl" --dev-file "$SPLIT_DIR/dev.jsonl" \
  --output-dir "$RUN_DIR/generator" --model "$MODEL_DIR" \
  --revision aadd2ab0ae0c8268c7c9693540e9904811f36177 --local-files-only \
  --epochs 10 --learning-rate 5e-5 --batch-size 16 --precision bf16 \
  --save-steps 200 --logging-steps 20 --seed 20260926 > "$RUN_DIR/train.log" 2>&1

status generating
pids=()
for gpu in 0 1 2 3; do
  if [ "$gpu" -lt 2 ]; then
    split=dev; start=$((gpu * 250)); count=250
  else
    split=reranker_train; start=$(((gpu - 2) * 500)); count=500
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m experiments.condition_consistency.baseline generate \
    --input "$SPLIT_DIR/$split.jsonl" --output "$RUN_DIR/${split}_part${gpu}.jsonl" \
    --model "$RUN_DIR/generator/final" --local-files-only --start-index "$start" --limit "$count" \
    --candidates 4 --beams 4 --batch-size 8 --max-new-tokens 256 --precision bf16 \
    > "$RUN_DIR/generate_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then exit 1; fi
cat "$RUN_DIR/dev_part0.jsonl" "$RUN_DIR/dev_part1.jsonl" > "$RUN_DIR/dev_generated.jsonl"
cat "$RUN_DIR/reranker_train_part2.jsonl" "$RUN_DIR/reranker_train_part3.jsonl" > "$RUN_DIR/reranker_train_generated.jsonl"

status executing
for split in dev reranker_train; do
  "$PYTHON_BIN" -m experiments.condition_consistency.execute_predictions \
    --predictions "$RUN_DIR/${split}_generated.jsonl" --gold "$SPLIT_DIR/$split.jsonl" \
    --kb "$KB_PATH" --output "$RUN_DIR/${split}_executed.jsonl" --expected-candidates 4 \
    > "$RUN_DIR/execute_${split}.log" 2>&1
done
"$PYTHON_BIN" -m experiments.condition_consistency.evaluate --predictions "$RUN_DIR/dev_executed.jsonl" \
  --output "$RUN_DIR/dev_metrics.json" > "$RUN_DIR/evaluate.log" 2>&1

status matching_negatives
"$PYTHON_BIN" -m experiments.condition_consistency.mutations \
  --data "$SPLIT_DIR/reranker_train.jsonl" --kb "$KB_PATH" --split train --limit 1000 \
  --output "$RUN_DIR/targeted_pairs.jsonl" --natural-cache "$RUN_DIR/reranker_train_executed.jsonl" \
  --natural-output "$RUN_DIR/natural_pairs.jsonl" > "$RUN_DIR/mutations.log" 2>&1

status ranking
pair_count=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["targeted_pairs"])' "$RUN_DIR/targeted_pairs.summary.json")
if [ "$pair_count" -gt 0 ]; then
for source in natural targeted; do
  for seed in 17 29 43; do
    "$PYTHON_BIN" -m experiments.condition_consistency.rerank \
      --train-pairs "$RUN_DIR/${source}_pairs.jsonl" --output "$RUN_DIR/${source}_ranker_seed${seed}.json" --seed "$seed" \
      > "$RUN_DIR/${source}_train_seed${seed}.log" 2>&1
    "$PYTHON_BIN" -m experiments.condition_consistency.evaluate \
      --predictions "$RUN_DIR/dev_executed.jsonl" --model "$RUN_DIR/${source}_ranker_seed${seed}.json" \
      --output "$RUN_DIR/${source}_metrics_seed${seed}.json" > "$RUN_DIR/${source}_eval_seed${seed}.log" 2>&1
  done
done
fi
"$PYTHON_BIN" -m experiments.condition_consistency.summarize_pilot --run-dir "$RUN_DIR" \
  > "$RUN_DIR/summary.log" 2>&1
status complete
