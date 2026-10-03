"""Run the frozen A/C continuation and development comparison through the ledger.

Only syx may launch this controller. Existing runs are never resumed or
overwritten: a failure leaves its marker, artifacts and job accounting intact.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

from .initial_runs import MODEL_ROOT, require_idle_gpus, run_job
from .train_sft import digest
from .training_data import read_jsonl

ROOT = Path("results/agent_feedback")
DATA = Path("data/agent_feedback/continuation")
MANIFEST = ROOT / "continuation_data.json"
ADAPTER = ROOT / "initial_sft_v1/model"
DEV_QUESTIONS = Path("data/agent_feedback/dev.questions.jsonl")
DEV_GOLD = Path("data/agent_feedback/dev.gold.jsonl")
RUNS = (("A", "clean_sft_v1", "clean_dev_v1", 0),
        ("C", "recovery_sft_v1", "recovery_dev_v1", 1))


def require_syx():
    identities = (os.getuid(), os.geteuid())
    if any(uid == 0 or pwd.getpwuid(uid).pw_name != "syx" for uid in identities):
        raise PermissionError("Continuation training must run as syx, never an administrator account")


def require_dev_predictions(path, ids):
    rows = read_jsonl(path)
    actual = [row["id"] for row in rows]
    if len(actual) != len(ids) or set(actual) != set(ids):
        raise ValueError(f"Predictions do not cover the complete frozen development split: {path}")


def validate_inputs():
    initial = json.loads((ROOT / "initial_sft_v1/run_result.json").read_text())
    if initial["status"] != "completed":
        raise ValueError("Initial SFT is not completed")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER / name).is_file() or (ADAPTER / name).stat().st_size == 0:
            raise FileNotFoundError(f"Completed initial adapter is missing {name}")
    # The builder writes the manifest last; its presence plus both matching
    # artifacts is the completion contract, not an invented status field.
    manifest = json.loads(MANIFEST.read_text())
    arms, budget = manifest["arms"], manifest["budget"]
    if (set(arms) != {"A", "C"} or manifest["seed"] != 20261003
            or manifest["max_context"] != 8192
            or type(budget["total"]) is not int or budget["total"] < 2
            or budget["paired_suffix"] != budget["total"] // 2
            or budget["ordinary"] != budget["total"] - budget["paired_suffix"]):
        raise ValueError("Continuation completion manifest has an invalid frozen budget")
    for arm, spec in arms.items():
        path = DATA / f"{arm}.jsonl"
        if (Path(spec["path"]).resolve() != path.resolve()
                or not path.is_file() or digest(path) != spec["sha256"]):
            raise ValueError(f"Frozen continuation artifact {arm} is missing or changed")
        if (spec["records"] < 1 or spec["supervised_tokens"] != budget["total"]
                or spec["ordinary_supervised_tokens"] != budget["ordinary"]
                or spec["paired_suffix_supervised_tokens"] != budget["paired_suffix"]):
            raise ValueError(f"Continuation arm {arm} has an unmatched supervision budget")
    if arms["A"]["records"] != arms["C"]["records"]:
        raise ValueError("Continuation arms have different record counts")
    dev = json.loads((ROOT / "split_manifest.json").read_text())["splits"]["dev"]
    if dev["count"] != 500 or len(set(dev["ids"])) != 500:
        raise ValueError("Expected the frozen 500-question development split")
    for key, path in (("questions", DEV_QUESTIONS), ("gold", DEV_GOLD)):
        if (Path(dev[key]["path"]).resolve() != path.resolve()
                or digest(path) != dev[key]["sha256"]):
            raise ValueError(f"Frozen development {key} file differs")
    require_dev_predictions(ROOT / "program_dev_v1.jsonl", dev["ids"])
    outputs = [ROOT / "recovery_vs_clean_dev.json", ROOT / "recovery_vs_program_dev.json"]
    for _, train_name, dev_name, _ in RUNS:
        outputs.extend([ROOT / train_name, ROOT / "jobs" / f"{train_name}.json",
                        ROOT / "jobs" / f"{dev_name}.json"])
        outputs.extend(ROOT / (dev_name + suffix) for suffix in
                       (".jsonl", ".runtime.json", ".metrics.json", ".scored.jsonl"))
    if any(path.exists() for path in outputs):
        raise FileExistsError("Continuation job or result already exists; do not overwrite or duplicate")
    return manifest, dev["ids"], digest(ADAPTER / "adapter_model.safetensors")


def validate_training(arm, name, manifest, manifest_hash, adapter_hash):
    output = ROOT / name
    result = json.loads((output / "run_result.json").read_text())
    inputs = json.loads((output / "input_manifest.json").read_text())
    continuation = inputs["continuation"]
    budget, spec = manifest["budget"]["total"], manifest["arms"][arm]
    if (result["status"] != "completed" or result["actual_supervised_tokens"] != budget
            or not math.isfinite(result["metrics"]["train_loss"])
            or result["data_sha256"] != spec["sha256"]
            or inputs["examples"] != spec["records"] or inputs["supervised_tokens"] != budget
            or continuation["arm"] != arm
            or continuation["manifest_sha256"] != manifest_hash
            or continuation["artifact_sha256"] != spec["sha256"]
            or continuation["expected_supervised_tokens"] != budget
            or continuation["adapter_weights_sha256"] != adapter_hash):
        raise ValueError(f"Continuation {arm} did not complete the frozen budget from the shared adapter")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = output / "model" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Continuation {arm} did not save {filename}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_syx()
    manifest, dev_ids, adapter_hash = validate_inputs()
    manifest_hash = digest(MANIFEST)
    with (ROOT / "continuation_runs_started.json").open("x") as handle:
        json.dump({"started_at": time.time(), "script_sha256": digest(__file__),
                   "manifest_sha256": manifest_hash, "initial_adapter_weights_sha256": adapter_hash,
                   "supervised_tokens_per_arm": manifest["budget"]["total"]}, handle, indent=2)
    require_idle_gpus()
    base = ["experiments.agent_feedback.train_sft", "--model", str(MODEL_ROOT),
            "--adapter", str(ADAPTER), "--pretokenized", "--epochs", "1", "--max-steps", "-1",
            "--limit", "0", "--continuation-manifest", str(MANIFEST)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, name, gpu, base + ["--data", str(DATA / f"{arm}.jsonl"),
                    "--output", str(ROOT / name)], 6) for arm, name, _, gpu in RUNS]
        for future in futures:
            future.result()
    if digest(MANIFEST) != manifest_hash or digest(ADAPTER / "adapter_model.safetensors") != adapter_hash:
        raise ValueError("Frozen manifest or shared initial adapter changed during continuation")
    for arm, name, _, _ in RUNS:
        validate_training(arm, name, manifest, manifest_hash, adapter_hash)
    print("CONTINUATION_SFT_JOBS_COMPLETED", flush=True)
    require_idle_gpus()
    generation = ["experiments.agent_feedback.inference", "generate", "--model", str(MODEL_ROOT),
                  "--batch-size", "8", "--questions", str(DEV_QUESTIONS)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, dev_name, gpu, generation + [
                    "--adapter", str(ROOT / name / "model"), "--output", str(ROOT / f"{dev_name}.jsonl")],
                    3) for _, name, dev_name, gpu in RUNS]
        for future in futures:
            future.result()
    for _, name, dev_name, _ in RUNS:
        predictions = ROOT / f"{dev_name}.jsonl"
        require_dev_predictions(predictions, dev_ids)
        runtime = json.loads(predictions.with_suffix(".runtime.json").read_text())
        if (runtime["status"] != "completed" or runtime["selected_questions"] != 500
                or runtime["source_questions"] != 500 or runtime["totals"]["questions"] != 500
                or Path(runtime["config"]["adapter"]).resolve() != (ROOT / name / "model").resolve()):
            raise ValueError(f"Incomplete or wrong-adapter development generation: {dev_name}")
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.inference", "score",
                        "--predictions", str(predictions), "--gold", str(DEV_GOLD)], check=True)
    for label, baseline, output in (("A", "clean_dev_v1", "recovery_vs_clean_dev.json"),
                                    ("P", "program_dev_v1", "recovery_vs_program_dev.json")):
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.compare",
            "--baseline", str(ROOT / f"{baseline}.jsonl"),
            "--candidate", str(ROOT / "recovery_dev_v1.jsonl"), "--gold", str(DEV_GOLD),
            "--baseline-label", label, "--candidate-label", "C", "--output", str(ROOT / output)], check=True)
    print("CONTINUATION_DEV_COMPARISONS_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
