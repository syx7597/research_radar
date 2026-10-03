"""Run the fixed initial SFT jobs after verified weights and a short GPU check.

This controller does not evaluate the holdout or choose hyperparameters. Every
GPU process, including the smoke check, goes through the shared budget ledger.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .download_weights import FILES, REVISION, ROOT as MODEL_ROOT


def run_job(name, gpu, command, hours):
    full = [sys.executable, "-u", "-m", "experiments.agent_feedback.run_job",
            "--name", name, "--gpus", str(gpu), "--max-hours", str(hours),
            "--", sys.executable, "-u", "-m", *command]
    subprocess.run(full, check=True)


def require_idle_gpus():
    busy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("GPU processes are present; inspect allocation before launching")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-minutes", type=int, default=120)
    args = parser.parse_args()
    root = Path("results/agent_feedback")
    marker = root / "initial_runs_started.json"
    # Exclusive creation prevents two controllers from scheduling the same jobs.
    with marker.open("x") as handle:
        json.dump({"started_at": time.time(), "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()}, handle)
    deadline = time.monotonic() + args.wait_minutes * 60
    weights = root / "weights_verified.json"
    while not weights.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Weights did not become ready; no GPU job launched")
        time.sleep(10)
    manifest = json.loads(weights.read_text())
    if manifest["revision"] != REVISION or {
        x["file"]: x["sha256"] for x in manifest["files"]} != FILES:
        raise ValueError("Weight verification manifest differs from the pinned model")
    for record in manifest["files"]:
        path = MODEL_ROOT / record["file"]
        if not path.exists() or path.stat().st_size != record["bytes"]:
            raise ValueError("Verified weight disappeared or changed size")
    require_idle_gpus()
    base = ["experiments.agent_feedback.train_sft", "--model", str(MODEL_ROOT)]
    smoke_dir = root / "sft_smoke_v1"
    run_job("sft_smoke_v1", 0, base + ["--output", str(smoke_dir),
            "--limit", "32", "--max-steps", "2", "--batch-size", "2",
            "--accumulation", "1"], .25)
    result = json.loads((smoke_dir / "run_result.json").read_text())
    if (result["status"] != "completed" or result["actual_supervised_tokens"] <= 0
            or not math.isfinite(result["metrics"]["train_loss"])
            or not (smoke_dir / "model/adapter_model.safetensors").exists()):
        raise ValueError("GPU smoke did not produce a finite trained adapter")
    # Both adapters start from the untouched pinned backbone, not smoke weights.
    tasks = [
        ("initial_sft_v1", 0, base + ["--output", str(root / "initial_sft_v1"),
                                      "--epochs", "2"], 6),
        ("program_sft_v1", 1, base + ["--output", str(root / "program_sft_v1"),
            "--program", "--data", "data/agent_feedback/train.gold.jsonl",
            "--epochs", "3"], 6),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, *task) for task in tasks]
        for future in futures:
            future.result()
    print("INITIAL_SFT_JOBS_COMPLETED", flush=True)
    require_idle_gpus()
    generation = ["experiments.agent_feedback.inference", "generate",
                  "--model", str(MODEL_ROOT), "--batch-size", "8"]
    initial_adapter = str(root / "initial_sft_v1/model")
    tasks = [
        ("initial_dev_v1", 0, generation + ["--adapter", initial_adapter,
            "--questions", "data/agent_feedback/dev.questions.jsonl",
            "--output", str(root / "initial_dev_v1.jsonl")], 3),
        ("program_dev_v1", 1, generation + ["--adapter", str(root / "program_sft_v1/model"),
            "--questions", "data/agent_feedback/dev.questions.jsonl", "--program",
            "--output", str(root / "program_dev_v1.jsonl")], 3),
    ]
    for shard in range(2):
        name = f"initial_train_v1_shard{shard}"
        tasks.append((name, 2 + shard, generation + ["--adapter", initial_adapter,
            "--questions", "data/agent_feedback/train.questions.jsonl",
            "--offset", str(shard * 2500), "--limit", "2500",
            "--output", str(root / (name + ".jsonl"))], 6))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_job, *task) for task in tasks]
        for future in futures:
            future.result()
    expected = [json.loads(line)["id"] for line in
                Path("data/agent_feedback/train.questions.jsonl").read_text().splitlines()]
    combined = []
    for shard in range(2):
        combined += (root / f"initial_train_v1_shard{shard}.jsonl").read_text().splitlines()
    if [json.loads(line)["id"] for line in combined] != expected:
        raise ValueError("Training rollout shards do not cover the complete ordered split")
    predictions = root / "initial_train_v1.jsonl"
    with predictions.open("x") as handle:
        handle.write("\n".join(combined) + "\n")
    for name in ("initial_dev_v1", "program_dev_v1"):
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.inference", "score",
            "--predictions", str(root / (name + ".jsonl")),
            "--gold", "data/agent_feedback/dev.gold.jsonl"], check=True)
    subprocess.run([sys.executable, "-m", "experiments.agent_feedback.recovery",
                    "--predictions", str(predictions)], check=True)
    print("INITIAL_DEV_AND_RECOVERY_COLLECTION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
