"""Repeat only frozen A/C continuation with seed 20261004 after a reviewed gate.

This is conditional continuation-seed replication, not a second end-to-end seed:
the initial adapter, corpus, Python hash seed, inference seed and program baseline
are reused. No RL or held-out evaluation is scheduled. Failed runs are retained.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from . import run_job as ledger
from .continuation_runs import (ADAPTER, DATA, DEV_GOLD, DEV_QUESTIONS, MANIFEST,
                                ROOT, require_dev_predictions, require_syx, validate_training)
from .download_weights import FILES, REVISION
from .initial_runs import MODEL_ROOT, require_idle_gpus, run_job
from .train_sft import digest

SEED = 20261004
ORIGINAL_SEED = 20261003
REVIEW = ROOT / "continuation_review.json"
PROTOCOL = ROOT / "protocol_paired_replication_seed20261004.json"
MARKER = ROOT / "replication_seed20261004_started.json"
RUNS = (("A", "clean_sft_seed20261004", "clean_dev_seed20261004", 0),
        ("C", "recovery_sft_seed20261004", "recovery_dev_seed20261004", 1))
OLD_RUNS = (("A", "clean_sft_v1", "clean_dev_v1"),
            ("C", "recovery_sft_v1", "recovery_dev_v1"))
COMPARISONS = (("A", "clean_dev_seed20261004", "recovery_vs_clean_dev_seed20261004.json"),
               ("P", "program_dev_v1", "recovery_vs_program_dev_seed20261004.json"))
CODE_FILES = ("train_sft.py", "training_data.py", "inference.py", "environment.py", "prepare.py",
              "compare.py", "run_job.py", "initial_runs.py", "continuation_runs.py", "download_weights.py")


def read(path):
    return json.loads(path.read_text())


def reviewed_paths():
    """Minimum files the external review must bind in its inputs_sha256 map."""
    paths = [ROOT / "protocol.json", MANIFEST, ROOT / "split_manifest.json",
             ROOT / "continuation_code_manifest_v1.json",
             ROOT / "recovery_vs_clean_dev.json", ROOT / "recovery_vs_program_dev.json",
             ADAPTER / "adapter_config.json", ADAPTER / "adapter_model.safetensors",
             ROOT / "program_dev_v1.jsonl"]
    for _, name, dev_name in OLD_RUNS:
        paths.extend([ROOT / name / "run_result.json", ROOT / name / "input_manifest.json",
                      ROOT / f"{dev_name}.jsonl", ROOT / f"{dev_name}.runtime.json"])
    return paths


def require_hashes(expected):
    for name, sha in expected.items():
        if digest(name) != sha:
            raise ValueError(f"Frozen or reviewed file changed: {name}")


def code_identity():
    directory = Path(__file__).resolve().parent
    return {str(Path("experiments/agent_feedback") / name): digest(directory / name)
            for name in (*CODE_FILES, "replication_runs.py")}


def training_config(arm, name, seed):
    return {"model": str(MODEL_ROOT), "data": str(DATA / f"{arm}.jsonl"),
            "output": str(ROOT / name), "adapter": str(ADAPTER), "program": False,
            "pretokenized": True, "continuation_manifest": str(MANIFEST), "epochs": 1.0,
            "max_steps": -1, "batch_size": 2, "accumulation": 8, "max_length": 8192,
            "learning_rate": 1e-4, "seed": seed, "limit": 0, "tokenize_only": False}


def inference_config(name, dev_name):
    return {"command": "generate", "model": str(MODEL_ROOT), "adapter": str(ROOT / name / "model"),
            "questions": str(DEV_QUESTIONS), "output": str(ROOT / f"{dev_name}.jsonl"),
            "kb": "datasets/kqa_pro/kb.json", "batch_size": 8, "limit": 0, "offset": 0,
            "seed": ORIGINAL_SEED, "max_context": 8192, "max_generated": 2048, "program": False}


def validate_dev(name, dev_name, ids):
    predictions = ROOT / f"{dev_name}.jsonl"
    require_dev_predictions(predictions, ids)
    runtime = read(predictions.with_suffix(".runtime.json"))
    if (runtime["status"] != "completed" or runtime["selected_questions"] != 500
            or runtime["source_questions"] != 500 or runtime["totals"]["questions"] != 500
            or runtime["config"] != inference_config(name, dev_name)):
        raise ValueError(f"Incomplete or changed development generation: {dev_name}")


def require_new_outputs():
    paths = [MARKER, PROTOCOL, *(ROOT / row[2] for row in COMPARISONS)]
    for _, name, dev_name, _ in RUNS:
        paths.extend([ROOT / name, ROOT / "jobs" / f"{name}.json", ROOT / "jobs" / f"{name}.log",
                      ROOT / "jobs" / f"{dev_name}.json", ROOT / "jobs" / f"{dev_name}.log"])
        paths.extend(ROOT / (dev_name + suffix) for suffix in
                     (".jsonl", ".runtime.json", ".metrics.json", ".scored.jsonl"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Replication output or job already exists; do not resume or overwrite")


def validate_inputs():
    require_new_outputs()
    review = read(REVIEW)
    if review.get("gate", {}).get("allow_paired_replication") is not True:
        raise ValueError("Completed continuation review does not allow paired replication")
    reviewed = review["inputs_sha256"]
    if not {str(path) for path in reviewed_paths()} <= set(reviewed):
        raise ValueError("Continuation review does not bind all required input identities")
    require_hashes(reviewed)
    if read(ROOT / "initial_sft_v1/run_result.json")["status"] != "completed":
        raise ValueError("Initial SFT is not completed")
    manifest = read(MANIFEST)
    arms, budget = manifest["arms"], manifest["budget"]
    if (set(arms) != {"A", "C"} or manifest["seed"] != ORIGINAL_SEED
            or manifest["max_context"] != 8192 or type(budget["total"]) is not int
            or budget["total"] < 2 or budget["paired_suffix"] != budget["total"] // 2
            or budget["ordinary"] != budget["total"] - budget["paired_suffix"]):
        raise ValueError("Continuation manifest is not the frozen matched-token protocol")
    hashes = dict(reviewed)
    for arm, spec in arms.items():
        path = DATA / f"{arm}.jsonl"
        if (Path(spec["path"]).resolve() != path.resolve() or digest(path) != spec["sha256"]
                or spec["records"] < 1 or spec["supervised_tokens"] != budget["total"]
                or spec["ordinary_supervised_tokens"] != budget["ordinary"]
                or spec["paired_suffix_supervised_tokens"] != budget["paired_suffix"]):
            raise ValueError(f"Frozen continuation arm {arm} changed")
        hashes[str(path)] = spec["sha256"]
    if arms["A"]["records"] != arms["C"]["records"]:
        raise ValueError("Continuation record counts differ")
    manifest_hash = digest(MANIFEST)
    adapter_hash = digest(ADAPTER / "adapter_model.safetensors")
    versions = None
    for arm, name, _ in OLD_RUNS:
        validate_training(arm, name, manifest, manifest_hash, adapter_hash)
        inputs, result = read(ROOT / name / "input_manifest.json"), read(ROOT / name / "run_result.json")
        if (inputs["config"] != training_config(arm, name, ORIGINAL_SEED)
                or result["seed"] != ORIGINAL_SEED
                or result["script_sha256"] != digest(Path(__file__).with_name("train_sft.py"))
                or inputs["continuation"]["adapter_config_sha256"] != digest(ADAPTER / "adapter_config.json")):
            raise ValueError(f"Previous {arm} continuation used different inputs or training code")
        if versions is not None and versions != inputs["versions"]:
            raise ValueError("Previous A/C training package versions differ")
        versions = inputs["versions"]
    dev = read(ROOT / "split_manifest.json")["splits"]["dev"]
    if dev["count"] != 500 or len(dev["ids"]) != 500 or len(set(dev["ids"])) != 500:
        raise ValueError("Expected the frozen 500-question development split")
    for key, path in (("questions", DEV_QUESTIONS), ("gold", DEV_GOLD)):
        if Path(dev[key]["path"]).resolve() != path.resolve() or digest(path) != dev[key]["sha256"]:
            raise ValueError(f"Frozen development {key} changed")
        hashes[str(path)] = dev[key]["sha256"]
    for _, name, dev_name in OLD_RUNS:
        validate_dev(name, dev_name, dev["ids"])
    require_dev_predictions(ROOT / "program_dev_v1.jsonl", dev["ids"])
    current_code = code_identity()
    previous_code = read(ROOT / "continuation_code_manifest_v1.json")["files"]
    if any(previous_code.get(name) != sha for name, sha in current_code.items()
           if not name.endswith("/replication_runs.py")):
        raise ValueError("Training/evaluation implementation changed since the first paired continuation")
    hashes[str(REVIEW)] = digest(REVIEW)
    hashes[str(ROOT / "initial_sft_v1/run_result.json")] = digest(ROOT / "initial_sft_v1/run_result.json")
    return {"manifest": manifest, "manifest_sha256": manifest_hash, "dev_ids": dev["ids"],
            "initial_adapter_weights_sha256": adapter_hash, "inputs_sha256": hashes,
            "code_sha256": current_code, "versions": versions}


def require_runtime(snapshot):
    for package, expected in snapshot["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Installed package differs from first continuation: {package}")
    weights = read(ROOT / "weights_verified.json")
    if weights["revision"] != REVISION or {r["file"]: r["sha256"] for r in weights["files"]} != FILES:
        raise ValueError("Pinned backbone identity differs")
    for row in weights["files"]:
        if (MODEL_ROOT / row["file"]).stat().st_size != row["bytes"]:
            raise ValueError("Previously verified backbone weight is missing or changed size")
    for name, sha in snapshot["manifest"]["tokenizer"]["files_sha256"].items():
        if digest(MODEL_ROOT / name) != sha:
            raise ValueError("Frozen tokenizer identity differs")
    if snapshot["manifest"]["tokenizer"]["transformers"] != snapshot["versions"]["transformers"]:
        raise ValueError("Tokenizer construction and training versions differ")


def require_budget(hours):
    """Read the shared ledger under its existing lock; do not reserve new jobs."""
    ledger.ROOT.mkdir(parents=True, exist_ok=True)
    with (ledger.ROOT / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [read(path) for path in ledger.ROOT.glob("*.json")]
        if any(row.get("finished_at") is None or row.get("status") not in {"completed", "failed"}
               for row in rows):
            raise RuntimeError("Unresolved GPU jobs must be reconciled before replication")
        used = ledger.total_gpu_seconds(time.time()) / 3600
        if used + hours > 72:
            raise RuntimeError(f"Insufficient shared GPU budget: {72 - used:.4f}h remain; need {hours}h")
    return used


def training_command(arm, name):
    return ["experiments.agent_feedback.train_sft", "--model", str(MODEL_ROOT),
            "--adapter", str(ADAPTER), "--pretokenized", "--epochs", "1", "--max-steps", "-1",
            "--limit", "0", "--continuation-manifest", str(MANIFEST), "--batch-size", "2",
            "--accumulation", "8", "--max-length", "8192", "--learning-rate", "0.0001",
            "--seed", str(SEED), "--data", str(DATA / f"{arm}.jsonl"), "--output", str(ROOT / name)]


def generation_command(name, dev_name):
    return ["experiments.agent_feedback.inference", "generate", "--model", str(MODEL_ROOT),
            "--batch-size", "8", "--questions", str(DEV_QUESTIONS), "--seed", str(ORIGINAL_SEED),
            "--max-context", "8192", "--max-generated", "2048", "--limit", "0", "--offset", "0",
            "--adapter", str(ROOT / name / "model"), "--output", str(ROOT / f"{dev_name}.jsonl")]


def require_unchanged(snapshot):
    require_hashes(snapshot["inputs_sha256"])
    if code_identity() != snapshot["code_sha256"]:
        raise ValueError("Replication implementation changed after freezing")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_syx()
    snapshot = validate_inputs()
    require_runtime(snapshot)
    require_idle_gpus()
    used = require_budget(6)
    require_unchanged(snapshot)
    jobs = [(name, gpu, training_command(arm, name), 2) for arm, name, _, gpu in RUNS]
    dev_jobs = [(dev_name, gpu, generation_command(name, dev_name), 1) for _, name, dev_name, gpu in RUNS]
    protocol = {"version": "paired_continuation_replication_seed20261004", "frozen_at": time.time(),
                "scope": "conditional continuation-seed replication; NOT a second end-to-end seed",
                "training_seed": SEED, "corpus_seed": ORIGINAL_SEED, "python_hash_seed": ORIGINAL_SEED,
                "inference_seed": ORIGINAL_SEED, "inference_decoding": "greedy",
                "base_protocol": str(ROOT / "protocol.json"), "review": str(REVIEW),
                "initial_adapter": str(ADAPTER), "program_baseline": str(ROOT / "program_dev_v1.jsonl"),
                "supervised_tokens_per_arm": snapshot["manifest"]["budget"]["total"],
                "maximum_additional_gpu_hours": 6, "global_gpu_hours_cap": 72,
                "budget_enforcement": "run_job: per-job timeouts, 10s checks and 20s termination grace",
                "gpu_hours_used_before_start": used, "training_jobs": jobs, "development_jobs": dev_jobs,
                "inputs_sha256": snapshot["inputs_sha256"], "code_sha256": snapshot["code_sha256"],
                "versions": snapshot["versions"], "rl": False, "held_out_evaluation": False}
    # The marker is created only after all CPU, identity, GPU and budget checks.
    # It is never removed automatically, even when a later step fails.
    with MARKER.open("x") as handle:
        json.dump({"started_at": time.time(), "script_sha256": digest(__file__),
                   "protocol": str(PROTOCOL), "review_sha256": snapshot["inputs_sha256"][str(REVIEW)]}, handle, indent=2)
    with PROTOCOL.open("x") as handle:
        json.dump(protocol, handle, indent=2)
    for stage, tasks in (("SFT", jobs), ("DEV", dev_jobs)):
        if stage == "DEV":
            require_idle_gpus()
            require_budget(2)
        require_unchanged(snapshot)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_job, *task) for task in tasks]
            for future in futures:
                future.result()
        require_unchanged(snapshot)
        if stage == "SFT":
            for arm, name, _, _ in RUNS:
                validate_training(arm, name, snapshot["manifest"], snapshot["manifest_sha256"],
                                  snapshot["initial_adapter_weights_sha256"])
                inputs, result = read(ROOT / name / "input_manifest.json"), read(ROOT / name / "run_result.json")
                if (inputs["config"] != training_config(arm, name, SEED) or result["seed"] != SEED
                        or inputs["versions"] != snapshot["versions"]
                        or result["script_sha256"] != snapshot["code_sha256"]["experiments/agent_feedback/train_sft.py"]
                        or inputs["continuation"]["adapter_config_sha256"] != digest(ADAPTER / "adapter_config.json")):
                    raise ValueError(f"Replication {arm} did not use the frozen configuration")
        print(f"PAIRED_REPLICATION_{stage}_COMPLETED", flush=True)
    for _, name, dev_name, _ in RUNS:
        validate_dev(name, dev_name, snapshot["dev_ids"])
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.inference", "score",
                        "--predictions", str(ROOT / f"{dev_name}.jsonl"), "--gold", str(DEV_GOLD)], check=True)
    for label, baseline, output in COMPARISONS:
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.compare",
            "--baseline", str(ROOT / f"{baseline}.jsonl"),
            "--candidate", str(ROOT / "recovery_dev_seed20261004.jsonl"), "--gold", str(DEV_GOLD),
            "--baseline-label", label, "--candidate-label", "C", "--output", str(ROOT / output)], check=True)
    print("PAIRED_REPLICATION_DEV_COMPARISONS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
