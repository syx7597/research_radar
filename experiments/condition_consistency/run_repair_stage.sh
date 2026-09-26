#!/usr/bin/env bash
# Run from repository root; generation is performed separately.
set -euo pipefail
export PYTHONHASHSEED=20260926 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-results/condition_consistency/query_repair}"
PILOT_DIR="${PILOT_DIR:-results/condition_consistency/pilot_bart5k_seed20260926}"
stage="${1:?development or holdout}"
if [ "$stage" = development ]; then
  splits=(dev calibration)
elif [ "$stage" = holdout ]; then
  test -f "$RUN_DIR/frozen_policy.json"
  splits=(holdout)
else
  exit 2
fi
gpu=0
pids=()
for split in "${splits[@]}"; do
  input="$RUN_DIR/${split}_beam4_generated.jsonl"
  if [ "$split" = dev ]; then input="$PILOT_DIR/dev_generated.jsonl"; fi
  for mode in global local; do
    (
      "$PYTHON_BIN" -m experiments.condition_consistency.repair --predictions "$input" \
        --kb datasets/kqa_pro/kb.json --mode "$mode" --output "$RUN_DIR/${split}_${mode}.jsonl" \
        > "$RUN_DIR/${split}_${mode}_repair.log" 2>&1
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m experiments.condition_consistency.score_repairs \
        --input "$RUN_DIR/${split}_${mode}.jsonl" --output "$RUN_DIR/${split}_${mode}_scored.jsonl" \
        --model "$PILOT_DIR/generator/final" > "$RUN_DIR/${split}_${mode}_score.log" 2>&1
    ) &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  if [ "$split" != calibration ]; then
    "$PYTHON_BIN" -m experiments.condition_consistency.execute_unlabeled \
      --input "$RUN_DIR/${split}_beam8_generated.jsonl" --output "$RUN_DIR/${split}_beam8_executed.jsonl" \
      > "$RUN_DIR/${split}_beam8_execute.log" 2>&1 &
    pids+=("$!")
  fi
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then exit 1; fi
