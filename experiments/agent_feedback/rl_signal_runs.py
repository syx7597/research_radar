"""Run two bounded learning-signal diagnostics after the replication audit.

Uses the original A/C checkpoints, 16 outcome-blind train IDs, and two updates.
Never starts formal RL, evaluates development/holdout, or selects checkpoints.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from .continuation_runs import require_syx
from .initial_runs import require_idle_gpus
from .replication_runs import require_budget, require_hashes
from .train_grpo import digest, protocol_check
from .training_data import read_jsonl

ROOT = Path("results/agent_feedback")
REVIEW = ROOT / "replication_review_seed20261004_v2.json"
PROTOCOL = ROOT / "protocol_rl_signal_v1.json"
MARKER = ROOT / "rl_signal_v1_started.json"
SUMMARY = ROOT / "rl_signal_v1_summary.json"
IDS = Path("data/agent_feedback/rl_signal_v1.ids.jsonl")
SEED = 20261003
COUNT = 16
ACCUMULATION = 32
HOURS = 0.75
RUNS = (("A", "clean_sft_v1", "clean_rl_signal_smoke_v1", 0),
        ("C", "recovery_sft_v1", "recovery_rl_signal_smoke_v1", 1))


def read(path):
    return json.loads(Path(path).read_text())


def select_ids(questions, eligible, count=COUNT):
    import hashlib
    all_ids = [row["id"] for row in questions]
    eligible_ids = [row["id"] for row in eligible]
    if (len(set(all_ids)) != len(all_ids) or len(set(eligible_ids)) != len(eligible_ids)
            or not set(eligible_ids) <= set(all_ids) or len(eligible_ids) < count):
        raise ValueError("Invalid eligible training IDs")
    return sorted(eligible_ids, key=lambda id: (
        hashlib.sha256(f"{SEED}:{id}".encode()).hexdigest(), str(id)))[:count]


def command(adapter, name):
    cpath = read(ROOT / "python_headers.json")["CPATH"]
    return ["env", f"CPATH={cpath}", sys.executable, "-u", "-m",
            "experiments.agent_feedback.train_grpo_signal", "--model", "models/qwen2.5-3b-instruct",
            "--adapter", str(ROOT / adapter / "model"), "--output", str(ROOT / name),
            "--eligible-ids", str(IDS), "--seed", str(SEED), "--batch-size", "1",
            "--accumulation", str(ACCUMULATION), "--smoke-steps", "2"]


def require_new():
    paths = [PROTOCOL, MARKER, SUMMARY, IDS]
    for _, _, name, _ in RUNS:
        paths.extend([ROOT / name, ROOT / f"{name}_preflight",
                      ROOT / "jobs" / f"{name}.json", ROOT / "jobs" / f"{name}.log"])
    if any(p.exists() for p in paths):
        raise FileExistsError("A signal diagnostic artifact already exists; never duplicate or overwrite")


def freeze():
    require_new()
    review = read(REVIEW)
    if review.get("gate", {}).get("conditional_signal_check_eligible") is not True:
        raise ValueError("Second-seed review has not passed the signal-check gate")
    required = {str(ROOT / name / suffix) for _, name, _, _ in RUNS
                for suffix in ("run_result.json", "input_manifest.json", "model/adapter_config.json",
                               "model/adapter_model.safetensors")}
    required.update(str(ROOT / name) for name in (
        "recovery_vs_clean_dev.json", "recovery_vs_clean_dev_seed20261004.json"))
    if (not required <= set(review.get("inputs_sha256", {}))
            or any(review.get("issues", {}).values())
            or any(review.get("replay", {}).get(arm, {}).get("checked") != 500
                   or review["replay"][arm]["mismatches"] for arm in ("A", "C"))):
        raise ValueError("Review must bind both checkpoints and both comparisons with complete replay")
    require_hashes(review["inputs_sha256"])
    protocol_path = ROOT / "protocol_rl_native.json"
    if not protocol_check(read(protocol_path)):
        raise ValueError("Native RL contract is not recorded")
    split = read(ROOT / "split_manifest.json")["splits"]["train"]
    paths = [REVIEW, protocol_path, ROOT / "split_manifest.json",
             ROOT / "python_headers.json", Path(__file__), Path(__file__).with_name("train_grpo.py"),
             Path(__file__).with_name("train_grpo_signal.py"),
             Path(__file__).with_name("environment.py"), Path(__file__).with_name("run_job.py")]
    for key in ("questions", "gold"):
        path = Path(f"data/agent_feedback/train.{key}.jsonl")
        if (path.resolve() != Path(split[key]["path"]).resolve()
                or digest(path) != split[key]["sha256"]):
            raise ValueError("Frozen train split changed")
        paths.append(path)
    eligible_path = Path("data/agent_feedback/train.success.jsonl")
    # Bind eligibility to the original successful SFT preparation, not dev results.
    old_rl = read(ROOT / "grpo_smoke_v2/run_result.json")
    if digest(eligible_path) != old_rl["sha256"]["eligible_ids"]:
        raise ValueError("Training eligibility changed since the recorded native smoke")
    paths.extend([eligible_path, ROOT / "grpo_smoke_v2/run_result.json"])
    for _, adapter, _, _ in RUNS:
        result_path = ROOT / adapter / "run_result.json"
        result = read(result_path)
        if result["status"] != "completed" or result["seed"] != SEED:
            raise ValueError("Signal runs must use the original completed SFT pair")
        paths.extend([result_path, ROOT / adapter / "input_manifest.json",
                      ROOT / adapter / "model/adapter_config.json",
                      ROOT / adapter / "model/adapter_model.safetensors"])
    chosen = select_ids(read_jsonl("data/agent_feedback/train.questions.jsonl"), read_jsonl(eligible_path))
    require_idle_gpus()
    used = require_budget(2 * HOURS)
    IDS.parent.mkdir(parents=True, exist_ok=True)
    with IDS.open("x") as handle:
        for id in chosen:
            handle.write(json.dumps({"id": id}) + "\n")
    paths.append(IDS)
    hashes = {str(p): digest(p) for p in paths}
    frozen = {"version": "paired_rl_signal_v1", "frozen_at": time.time(),
        "scope": "Learning-signal diagnostic only; no formal RL or inference evaluation",
        "checkpoint_seed": SEED, "sampling_seed": SEED, "python_hash_seed": SEED,
        "selected_train_ids": chosen,
        "selection": "eligible IDs sorted by SHA256 of UTF-8 '20261003:<id>'; no outcome filtering",
        "questions_per_arm": COUNT, "group_size": 4, "optimizer_updates": 2,
        "batch_size": 1, "gradient_accumulation_steps": ACCUMULATION,
        "expected_rollouts_per_arm": COUNT * 4,
        "diagnostic_batch_note": "Larger accumulation covers 16 questions within the existing two-update smoke limit; does not freeze formal RL batch size",
        "native_protocol": str(protocol_path), "inputs_sha256": hashes,
        "gpu_hours_used_before_start": used, "maximum_additional_gpu_hours": 2 * HOURS,
        "per_arm_timeout_hours": HOURS, "global_gpu_hours_cap": 72,
        "jobs": [{"arm": arm, "name": name, "gpu": gpu, "command": command(adapter, name)}
                 for arm, adapter, name, gpu in RUNS],
        "decision": "Require paired fixed-ID coverage, mixed within-question rewards, finite nonzero gradients and actual adapter updates before considering formal RL; zero signal is inconclusive, never auto-expand",
        "formal_rl_auto_launch": False, "holdout_evaluation": False,
        "semantic_preference_training": False}
    with PROTOCOL.open("x") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return frozen


def main():
    require_syx()
    frozen = freeze()
    # Both tokenizers/prompt schemas pass without loading weights before any GPU job.
    for _, adapter, name, _ in RUNS:
        subprocess.run(command(adapter, name + "_preflight") + ["--preflight"], check=True)
    require_hashes(frozen["inputs_sha256"])
    require_idle_gpus()
    require_budget(2 * HOURS)
    with MARKER.open("x") as handle:
        json.dump({"started_at": time.time(), "protocol_sha256": digest(PROTOCOL)}, handle)

    def launch(job):
        subprocess.run([sys.executable, "-u", "-m", "experiments.agent_feedback.run_job",
                        "--name", job["name"], "--gpus", str(job["gpu"]), "--max-hours", str(HOURS),
                        "--", *job["command"]], check=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch, job) for job in frozen["jobs"]]
        for future in futures:
            future.result()
    require_hashes(frozen["inputs_sha256"])
    results = {arm: read(ROOT / name / "run_result.json") for arm, _, name, _ in RUNS}
    with SUMMARY.open("x") as handle:
        json.dump({"finished_at": time.time(), "protocol_sha256": digest(PROTOCOL),
                   "results": results, "formal_rl_started": False,
                   "next_step": "Review signal, coverage, parameter updates and observed throughput before a separate formal decision"},
                  handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("PAIRED_RL_SIGNAL_CHECK_COMPLETED_NO_FORMAL_RL_STARTED", flush=True)


if __name__ == "__main__":
    main()
