#!/usr/bin/env bash
# Question-only, fixed-checkpoint generation. No holdout answers are read.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-results/condition_consistency/query_repair}"
MODEL_DIR="${MODEL_DIR:-results/condition_consistency/pilot_bart5k_seed20260926/generator/final}"
INPUT_PATH="${INPUT_PATH:-data/condition_consistency/repair_splits/holdout.questions.jsonl}"
mkdir -p "$RUN_DIR"
pids=()
for gpu in 0 1 2 3; do
  if [ "$gpu" -lt 2 ]; then beams=4; shard="$gpu"; else beams=8; shard=$((gpu - 2)); fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m experiments.condition_consistency.baseline generate \
    --input "$INPUT_PATH" --output "$RUN_DIR/holdout_beam${beams}_part${shard}.jsonl" \
    --model "$MODEL_DIR" --local-files-only --start-index "$((shard * 1000))" --limit 1000 \
    --beams "$beams" --candidates "$beams" --batch-size 8 --precision bf16 \
    > "$RUN_DIR/holdout_beam${beams}_part${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then exit 1; fi
for beams in 4 8; do
  cat "$RUN_DIR/holdout_beam${beams}_part0.jsonl" "$RUN_DIR/holdout_beam${beams}_part1.jsonl" \
    > "$RUN_DIR/holdout_beam${beams}_generated.jsonl"
done
