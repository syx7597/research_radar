"""Run one explicitly approved B/D pair, then the frozen development comparisons.

The decision file is external to this controller. Preparing this code does not
approve training. Original seed-20261003 A/C adapters are used, never smoke
weights. Every GPU process uses the existing shared 72 GPU-hour job ledger.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .continuation_runs import ADAPTER, MANIFEST, require_dev_predictions, require_syx, validate_training
from .initial_runs import MODEL_ROOT, require_idle_gpus
from .replication_runs import (DEV_GOLD, DEV_QUESTIONS, generation_command, require_budget,
                               require_hashes, require_runtime, validate_dev)
from .train_grpo import NATIVE_CONTRACT, digest, protocol_check
from .training_data import read_jsonl

ROOT = Path("results/agent_feedback")
CODE = Path("experiments/agent_feedback")
DECISION = ROOT / "rl_signal_decision_v1.json"
PROTOCOL = ROOT / "protocol_rl_paired_v1.json"
MARKER = ROOT / "rl_paired_v1_started.json"
SUMMARY = ROOT / "rl_paired_v1_summary.json"
NATIVE_PROTOCOL = ROOT / "protocol_rl_native.json"
SIGNAL_PROTOCOL = ROOT / "protocol_rl_signal_v1.json"
QUESTIONS = Path("data/agent_feedback/train.questions.jsonl")
GOLD = Path("data/agent_feedback/train.gold.jsonl")
ELIGIBLE = Path("data/agent_feedback/train.success.jsonl")
SEED = 20261003
UPDATES = 200
RUNS = (("B", "clean_sft_v1", "clean_grpo_v1", "clean_grpo_dev_v1", "clean_rl_signal_smoke_v1", 0),
        ("D", "recovery_sft_v1", "recovery_grpo_v1", "recovery_grpo_dev_v1", "recovery_rl_signal_smoke_v1", 1))
COMPARISONS = (("B", "D", "clean_grpo_dev_v1", "recovery_grpo_dev_v1", "recovery_grpo_vs_clean_grpo_dev.json"),
               ("A", "B", "clean_dev_v1", "clean_grpo_dev_v1", "clean_grpo_vs_clean_sft_dev.json"),
               ("C", "D", "recovery_dev_v1", "recovery_grpo_dev_v1", "recovery_grpo_vs_recovery_sft_dev.json"),
               ("P", "D", "program_dev_v1", "recovery_grpo_dev_v1", "recovery_grpo_vs_program_dev.json"))


def read(path):
    return json.loads(Path(path).read_text())


def decision_required_paths():
    """Canonical minimum hash keys; extra reviewed files are also checked."""
    paths = [SIGNAL_PROTOCOL, ROOT / "rl_signal_v1_summary.json", NATIVE_PROTOCOL,
             ROOT / "protocol.json", ROOT / "continuation_code_manifest_v1.json",
             ROOT / "split_manifest.json", ROOT / "python_headers.json", ROOT / "weights_verified.json",
             MANIFEST, QUESTIONS, GOLD, ELIGIBLE, DEV_QUESTIONS, DEV_GOLD,
             CODE / "rl_paired_runs.py", CODE / "train_grpo_signal.py", CODE / "train_grpo.py",
             CODE / "replication_runs.py", CODE / "rl_signal_runs.py",
             ROOT / "clean_dev_v1.jsonl", ROOT / "recovery_dev_v1.jsonl", ROOT / "program_dev_v1.jsonl"]
    for _, adapter, _, _, smoke, _ in RUNS:
        paths.extend(ROOT / adapter / name for name in (
            "run_result.json", "input_manifest.json", "model/adapter_config.json", "model/adapter_model.safetensors"))
        paths.extend(ROOT / smoke / name for name in (
            "run_result.json", "run_started.json", "reward_group_signal.jsonl",
            "optimizer_signal.rank0.jsonl", "trainable_parameter_signal.rank0.json", "trainable_layout.rank0.json"))
    return paths


def timeout_hours(decision):
    hours = decision["training_timeout_hours_per_arm"]
    if type(hours) not in (int, float) or not math.isfinite(hours) or not 0 < hours <= 8:
        raise ValueError("Formal per-arm timeout must be a finite number in (0, 8] hours")
    return float(hours)


def require_new_outputs():
    paths = [PROTOCOL, MARKER, SUMMARY, *(ROOT / spec[-1] for spec in COMPARISONS)]
    for _, _, name, dev, _, _ in RUNS:
        paths.append(ROOT / name)
        paths.extend(ROOT / "jobs" / (job + suffix) for job in (name, dev) for suffix in (".json", ".log"))
        paths.extend(ROOT / (dev + suffix) for suffix in (".jsonl", ".runtime.json", ".metrics.json", ".scored.jsonl"))
    if any(path.exists() for path in paths):
        raise FileExistsError("A paired RL job or artifact exists; never resume, duplicate, or overwrite")


def validate_smoke(adapter, smoke, selected_ids):
    result, started = read(ROOT / smoke / "run_result.json"), read(ROOT / smoke / "run_started.json")
    signal = result["learning_signal"]
    config, native_config = started["config"], started["grpo_config"]
    groups = read_jsonl(ROOT / smoke / "reward_group_signal.jsonl")
    actual_ids = [row["example_id"] for row in groups]
    if (result["status"] != "completed" or result["smoke"] is not True
            or result["formal_result_eligible"] is not False or result["model_artifact"] is not None
            or result["optimizer_steps"] != 2 or result["seed"] != SEED or result["world_size"] != 1
            or result["native_contract"] != NATIVE_CONTRACT or started["native_contract"] != NATIVE_CONTRACT
            or result["sha256"] != started["sha256"] or started["training_questions"] != 16
            or any(config.get(key) != value for key, value in {"batch_size": 1, "accumulation": 32,
                   "seed": SEED, "smoke_steps": 2, "preflight": False}.items())
            or any(native_config.get(key) != value for key, value in {"generation_batch_size": 32,
                   "steps_per_generation": 32, "num_generations": 4, "max_steps": 2}.items())
            or signal["verified_nonzero_learning_signal"] is not True
            or signal["optimizer_callbacks_match_trainer_steps"] is not True
            or signal["question_groups"] != 16 or signal["rollouts"] != 64
            or signal["unique_questions_observed"] != 16 or signal["all_eligible_questions_observed"] is not True
            or len(actual_ids) != 16 or set(actual_ids) != set(selected_ids)
            or actual_ids != signal["group_question_ids_in_generation_order"]
            or signal["reward_groups_audit_sha256"] != digest(ROOT / smoke / "reward_group_signal.jsonl")
            or Path(started["config"]["adapter"]).resolve() != (ROOT / adapter / "model").resolve()):
        raise ValueError(f"Short diagnostic is incomplete or lacks verified signal: {smoke}")
    for key, path in (("adapter_config", ROOT / adapter / "model/adapter_config.json"),
                      ("adapter_weights", ROOT / adapter / "model/adapter_model.safetensors"),
                      ("script", CODE / "train_grpo_signal.py"), ("frozen_grpo_entry", CODE / "train_grpo.py"),
                      ("environment", CODE / "environment.py"), ("protocol", NATIVE_PROTOCOL)):
        if result["sha256"][key] != digest(path) or started["sha256"][key] != digest(path):
            raise ValueError(f"Short diagnostic input or implementation changed: {smoke}/{key}")
    return started["versions"], actual_ids


def validate_inputs():
    require_new_outputs()
    decision = read(DECISION)
    if decision.get("gate", {}).get("allow_formal_rl") is not True:
        raise ValueError("A separate reviewed decision has not authorized formal paired RL")
    hours = timeout_hours(decision)
    hashes = decision["inputs_sha256"]
    if not {str(path) for path in decision_required_paths()} <= set(hashes):
        raise ValueError("Formal decision must bind diagnostic artifacts, original adapters, data, and code")
    require_hashes(hashes)
    if not protocol_check(read(NATIVE_PROTOCOL)):
        raise ValueError("Native RL protocol differs")
    # Historical code and signal-specific code are checked independently. The
    # new observation entry does not replace any historical implementation.
    require_hashes(read(ROOT / "continuation_code_manifest_v1.json")["files"])
    diagnostic = read(SIGNAL_PROTOCOL)
    require_hashes(diagnostic["inputs_sha256"])
    selected = diagnostic["selected_train_ids"]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("Expected the original paired 16-question signal diagnostic")
    manifest = read(MANIFEST)
    manifest_hash, initial_hash = digest(MANIFEST), digest(ADAPTER / "adapter_model.safetensors")
    versions, paired_ids = None, None
    for arm, adapter, _, _, smoke, _ in RUNS:
        validate_training("A" if arm == "B" else "C", adapter, manifest, manifest_hash, initial_hash)
        if read(ROOT / adapter / "run_result.json")["seed"] != SEED:
            raise ValueError("Formal RL must use the original seed A/C adapters")
        actual_versions, ids = validate_smoke(adapter, smoke, selected)
        if versions is not None and (versions != actual_versions or paired_ids != ids):
            raise ValueError("Paired diagnostics used different runtime versions or question order")
        versions, paired_ids = actual_versions, ids
    split = read(ROOT / "split_manifest.json")["splits"]
    for name, expected_count, paths in (("train", 5000, (QUESTIONS, GOLD)), ("dev", 500, (DEV_QUESTIONS, DEV_GOLD))):
        spec = split[name]
        if spec["count"] != expected_count or len(spec["ids"]) != expected_count or len(set(spec["ids"])) != expected_count:
            raise ValueError(f"Frozen {name} split size differs")
        for key, path in zip(("questions", "gold"), paths):
            if Path(spec[key]["path"]).resolve() != path.resolve() or digest(path) != spec[key]["sha256"]:
                raise ValueError(f"Frozen {name} {key} changed")
    eligible_ids = [row["id"] for row in read_jsonl(ELIGIBLE)]
    if (len(eligible_ids) != 4999 or len(set(eligible_ids)) != 4999
            or not set(eligible_ids) <= set(split["train"]["ids"])
            or digest(ELIGIBLE) != read(ROOT / "grpo_smoke_v2/run_result.json")["sha256"]["eligible_ids"]):
        raise ValueError("Formal RL requires all 4999 originally eligible training questions")
    for name in ("clean_dev_v1", "recovery_dev_v1", "program_dev_v1"):
        require_dev_predictions(ROOT / f"{name}.jsonl", split["dev"]["ids"])
    frozen = dict(hashes)
    frozen.update({str(DECISION): digest(DECISION), **read(ROOT / "continuation_code_manifest_v1.json")["files"]})
    return {"inputs_sha256": frozen, "timeout_hours": hours, "versions": versions,
            "manifest": manifest, "dev_ids": split["dev"]["ids"], "eligible_ids": eligible_ids}


def training_config(adapter, name):
    return {"model": str(MODEL_ROOT), "adapter": str(ROOT / adapter / "model"),
            "questions": str(QUESTIONS), "gold": str(GOLD), "eligible_ids": str(ELIGIBLE),
            "protocol": str(NATIVE_PROTOCOL), "kb": "datasets/kqa_pro/kb.json", "output": str(ROOT / name),
            "batch_size": 1, "accumulation": 4, "seed": SEED, "preflight": False, "smoke_steps": None}


def training_command(adapter, name):
    return ["env", f"CPATH={read(ROOT / 'python_headers.json')['CPATH']}", sys.executable, "-u", "-m",
            "experiments.agent_feedback.train_grpo_signal", "--model", str(MODEL_ROOT),
            "--adapter", str(ROOT / adapter / "model"), "--output", str(ROOT / name),
            "--questions", str(QUESTIONS), "--gold", str(GOLD), "--eligible-ids", str(ELIGIBLE),
            "--protocol", str(NATIVE_PROTOCOL), "--seed", str(SEED), "--batch-size", "1", "--accumulation", "4"]


def launch(name, gpu, command, hours):
    subprocess.run([sys.executable, "-u", "-m", "experiments.agent_feedback.run_job", "--name", name,
                    "--gpus", str(gpu), "--max-hours", str(hours), "--", *command], check=True)


def validate_formal(adapter, name, snapshot):
    output = ROOT / name
    result, started = read(output / "run_result.json"), read(output / "run_started.json")
    config = started["grpo_config"]
    expected = {"max_steps": UPDATES, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
                "num_generations": 4, "beta": 0.0, "learning_rate": 5e-6, "num_iterations": 1,
                "seed": SEED, "data_seed": SEED, "generation_batch_size": 4, "steps_per_generation": 4}
    signal = result["learning_signal"]
    if (result["status"] != "completed" or result["smoke"] is not False
            or result["formal_result_eligible"] is not True or result["optimizer_steps"] != UPDATES
            or result["world_size"] != 1 or result["seed"] != SEED
            or result["native_contract"] != NATIVE_CONTRACT or not math.isfinite(result["metrics"]["train_loss"])
            or Path(result["model_artifact"]).resolve() != (output / "model").resolve()
            or started["training_questions"] != 4999 or started["config"] != training_config(adapter, name)
            or result["sha256"] != started["sha256"]
            or started["signal_audit"]["planned_question_groups"] != UPDATES
            or started["versions"] != snapshot["versions"] or any(config.get(key) != value for key, value in expected.items())
            or signal["optimizer_callbacks_match_trainer_steps"] is not True
            or signal["question_groups"] != UPDATES or signal["rollouts"] != 4 * UPDATES
            or signal["all_rewards_and_advantages_finite"] is not True
            or signal["all_rewards_binary"] is not True or signal["all_observed_gradients_finite"] is not True
            or signal["trainable_parameter_update"]["all_before_finite"] is not True
            or signal["trainable_parameter_update"]["all_after_finite"] is not True):
        raise ValueError(f"Formal RL did not complete the fixed configuration: {name}")
    paths = {"questions": QUESTIONS, "gold": GOLD, "eligible_ids": ELIGIBLE, "protocol": NATIVE_PROTOCOL,
             "script": CODE / "train_grpo_signal.py", "frozen_grpo_entry": CODE / "train_grpo.py",
             "environment": CODE / "environment.py", "adapter_config": ROOT / adapter / "model/adapter_config.json",
             "adapter_weights": ROOT / adapter / "model/adapter_model.safetensors"}
    if any(result["sha256"][key] != digest(path) or started["sha256"][key] != digest(path) for key, path in paths.items()):
        raise ValueError(f"Formal RL used changed data, adapter, or code: {name}")
    groups = read_jsonl(output / "reward_group_signal.jsonl")
    ids = [row["example_id"] for row in groups]
    if (len(groups) != signal["question_groups"] or ids != signal["group_question_ids_in_generation_order"]
            or not set(ids) <= set(snapshot["eligible_ids"])
            or digest(output / "reward_group_signal.jsonl") != signal["reward_groups_audit_sha256"]
            or any(row["all_rewards_finite"] is not True or row["all_advantages_finite"] is not True
                   or row["all_rewards_binary"] is not True or len(row["rewards"]) != 4
                   or len(row["loss_tokens_after_completion_and_tool_masks"]) != 4
                   or any(type(n) is not int or n < 0 for n in row["loss_tokens_after_completion_and_tool_masks"])
                   for row in groups)):
        raise ValueError(f"Formal generation count or training coverage audit differs: {name}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = output / "model" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Formal adapter was not saved: {path}")
    return {"optimizer_steps": result["optimizer_steps"], "question_groups": signal["question_groups"],
            "rollouts": signal["rollouts"], "unique_training_questions": len(set(ids)),
            "expected_question_groups": UPDATES, "expected_rollouts": UPDATES * 4,
            "expected_counts_match": signal["question_groups"] == UPDATES and signal["rollouts"] == UPDATES * 4,
            "learning_signal": signal, "run_result_sha256": digest(output / "run_result.json")}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_syx()
    snapshot = validate_inputs()
    require_runtime(snapshot)
    require_idle_gpus()
    total_hours = 2 * snapshot["timeout_hours"] + 2
    used = require_budget(total_hours)
    require_hashes(snapshot["inputs_sha256"])
    jobs = [(name, gpu, training_command(adapter, name), snapshot["timeout_hours"])
            for _, adapter, name, _, _, gpu in RUNS]
    dev_jobs = [(dev, gpu, [sys.executable, "-u", "-m", *generation_command(name, dev)], 1)
                for _, _, name, dev, _, gpu in RUNS]
    protocol = {"version": "paired_formal_native_rl_v1", "frozen_at": time.time(), "decision": str(DECISION),
        "decision_sha256": digest(DECISION), "native_protocol": str(NATIVE_PROTOCOL), "checkpoint_seed": SEED,
        "seed": SEED, "python_hash_seed": SEED, "optimizer_updates_per_arm": UPDATES, "group_size": 4,
        "batch_size": 1, "gradient_accumulation_steps": 4, "eligible_training_questions": 4999,
        "expected_question_groups_per_arm": 200, "expected_rollouts_per_arm": 800,
        "initial_adapters": {arm: str(ROOT / adapter / "model") for arm, adapter, *_ in RUNS},
        "smoke_weights_used": False, "maximum_additional_gpu_hours": total_hours, "global_gpu_hours_cap": 72,
        "gpu_hours_used_before_start": used, "timeout_note": "Existing run_job polling and shutdown add seconds of overhead",
        "training_jobs": jobs, "development_jobs": dev_jobs, "inputs_sha256": snapshot["inputs_sha256"],
        "versions": snapshot["versions"], "held_out_evaluation": False, "automatic_next_round": False}
    with MARKER.open("x") as handle:
        json.dump({"started_at": time.time(), "script_sha256": digest(__file__),
                   "decision_sha256": digest(DECISION), "protocol": str(PROTOCOL)}, handle, indent=2)
    with PROTOCOL.open("x") as handle:
        json.dump(protocol, handle, indent=2)
    reports = {}
    for stage, tasks in (("TRAINING", jobs), ("DEV", dev_jobs)):
        if stage == "DEV":
            require_idle_gpus()
            require_budget(2)
        require_hashes(snapshot["inputs_sha256"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, *task) for task in tasks]
            for future in futures:
                future.result()
        require_hashes(snapshot["inputs_sha256"])
        if stage == "TRAINING":
            reports = {arm: validate_formal(adapter, name, snapshot) for arm, adapter, name, *_ in RUNS}
            if (reports["B"]["learning_signal"]["group_question_ids_in_generation_order"]
                    != reports["D"]["learning_signal"]["group_question_ids_in_generation_order"]):
                raise ValueError("Formal B/D generated different training question group orders")
        print(f"PAIRED_FORMAL_RL_{stage}_COMPLETED", flush=True)
    for _, _, name, dev, _, _ in RUNS:
        validate_dev(name, dev, snapshot["dev_ids"])
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.inference", "score",
                        "--predictions", str(ROOT / f"{dev}.jsonl"), "--gold", str(DEV_GOLD)], check=True)
    for baseline_label, candidate_label, baseline, candidate, output in COMPARISONS:
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.compare",
            "--baseline", str(ROOT / f"{baseline}.jsonl"), "--candidate", str(ROOT / f"{candidate}.jsonl"),
            "--gold", str(DEV_GOLD), "--baseline-label", baseline_label, "--candidate-label", candidate_label,
            "--output", str(ROOT / output)], check=True)
    with SUMMARY.open("x") as handle:
        json.dump({"finished_at": time.time(), "protocol_sha256": digest(PROTOCOL), "training": reports,
            "development_comparisons": {f"{row[1]}_vs_{row[0]}": {"path": str(ROOT / row[-1]),
                "sha256": digest(ROOT / row[-1])} for row in COMPARISONS},
            "held_out_evaluation": False, "automatic_next_round": False}, handle, indent=2)
    print("PAIRED_FORMAL_RL_COMPLETE_NO_AUTOMATIC_NEXT_ROUND", flush=True)


if __name__ == "__main__":
    main()
