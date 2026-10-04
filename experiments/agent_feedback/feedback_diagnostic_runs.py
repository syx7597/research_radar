"""One fixed error-diagnostic masking run on dev500 for four frozen SFT models.

Normal evaluation remains the primary result. This intervention cannot select a
checkpoint, change the training recipe, open holdout, or launch another round.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.condition_consistency.executor import KoPLExecutor
from .continuation_runs import ADAPTER, MANIFEST, require_dev_predictions, require_syx, validate_training
from .initial_runs import MODEL_ROOT, require_idle_gpus, run_job
from .replication_runs import (DEV_GOLD, DEV_QUESTIONS, inference_config, require_budget,
                               require_hashes, require_runtime, validate_dev)
from .train_sft import digest
from .training_data import read_jsonl

ROOT = Path("results/agent_feedback")
CODE = Path("experiments/agent_feedback")
KB = Path("datasets/kqa_pro/kb.json")
REVIEW = ROOT / "replication_review_seed20261004_v2.json"
PROTOCOL = ROOT / "protocol_feedback_diagnostic_v1.json"
MARKER = ROOT / "feedback_diagnostic_v1_started.json"
AUDIT = ROOT / "feedback_diagnostic_audit_v1.json"
SUMMARY = ROOT / "feedback_diagnostic_summary_v1.json"
MODE = "generic_failure_v1"
HOURS_PER_ARM = 1
RUNS = (("A", 20261003, "clean_sft_v1", "clean_dev_v1", "clean_feedback_masked_dev_v1", 0),
        ("C", 20261003, "recovery_sft_v1", "recovery_dev_v1", "recovery_feedback_masked_dev_v1", 1),
        ("A", 20261004, "clean_sft_seed20261004", "clean_dev_seed20261004", "clean_feedback_masked_dev_seed20261004", 2),
        ("C", 20261004, "recovery_sft_seed20261004", "recovery_dev_seed20261004", "recovery_feedback_masked_dev_seed20261004", 3))


def read(path):
    return json.loads(Path(path).read_text())


def require_new_outputs():
    paths = [PROTOCOL, MARKER, AUDIT, SUMMARY]
    for _, _, _, _, name, _ in RUNS:
        paths.extend(ROOT / (name + suffix) for suffix in
                     (".jsonl", ".runtime.json", ".feedback.json", ".feedback_started.json", ".metrics.json", ".scored.jsonl", ".diagnostic.json"))
        paths.extend(ROOT / "jobs" / (name + suffix) for suffix in (".json", ".log"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Feedback diagnostic artifacts already exist; no resume, overwrite or automatic repeat")


def validate_inputs():
    require_new_outputs()
    review = read(REVIEW)
    if review.get("gate", {}).get("audit_passed") is not True or any(review.get("issues", {}).values()):
        raise ValueError("The completed paired continuation audit must pass before feedback diagnostics")
    required = {str(ROOT / model / name) for _, _, model, _, _, _ in RUNS
                for name in ("run_result.json", "input_manifest.json", "model/adapter_config.json",
                             "model/adapter_model.safetensors")}
    required.update(str(ROOT / (normal + ".jsonl")) for _, _, _, normal, _, _ in RUNS)
    if not required <= set(review["inputs_sha256"]):
        raise ValueError("Continuation review must bind all four models and all normal development trajectories")
    require_hashes(review["inputs_sha256"])
    old_code = read(ROOT / "continuation_code_manifest_v1.json")["files"]
    require_hashes(old_code)
    split = read(ROOT / "split_manifest.json")
    dev = split["splits"]["dev"]
    if dev["count"] != 500 or len(dev["ids"]) != 500 or len(set(dev["ids"])) != 500:
        raise ValueError("Diagnostic requires the frozen complete dev500")
    for key, path in (("questions", DEV_QUESTIONS), ("gold", DEV_GOLD)):
        if Path(dev[key]["path"]).resolve() != path.resolve() or digest(path) != dev[key]["sha256"]:
            raise ValueError(f"Frozen development {key} changed")
    if digest(KB) != split["source_sha256"]["kb.json"]:
        raise ValueError("The execution knowledge base changed")
    manifest = read(MANIFEST)
    manifest_hash, adapter_hash = digest(MANIFEST), digest(ADAPTER / "adapter_model.safetensors")
    versions = None
    paths = [REVIEW, ROOT / "continuation_code_manifest_v1.json", ROOT / "split_manifest.json",
             ROOT / "protocol.json", ROOT / "rl_stage_decision_v1.json", ROOT / "weights_verified.json",
             ROOT / "recovery_pairs.json",
             MANIFEST, DEV_QUESTIONS, DEV_GOLD, KB, CODE / "feedback_diagnostic_runs.py",
             CODE / "feedback_mask.py", CODE / "feedback_diagnostic_analysis.py", CODE / "replication_runs.py",
             Path("external/kqa_pro_baselines/utils/value_class.py")]
    for arm, seed, model, normal, _, _ in RUNS:
        validate_training(arm, model, manifest, manifest_hash, adapter_hash)
        result, inputs = read(ROOT / model / "run_result.json"), read(ROOT / model / "input_manifest.json")
        if result["seed"] != seed or inputs["config"]["seed"] != seed:
            raise ValueError("Diagnostic must use both original continuation seeds without checkpoint selection")
        if versions is not None and versions != inputs["versions"]:
            raise ValueError("The four frozen SFT models used different runtime package versions")
        versions = inputs["versions"]
        validate_dev(model, normal, dev["ids"])
        paths.extend(ROOT / model / name for name in (
            "run_result.json", "input_manifest.json", "model/adapter_config.json", "model/adapter_model.safetensors"))
        paths.extend(ROOT / (normal + suffix) for suffix in (".jsonl", ".runtime.json", ".metrics.json"))
    hashes = {**review["inputs_sha256"], **old_code, **{str(path): digest(path) for path in paths}}
    return {"inputs_sha256": hashes, "versions": versions, "manifest": manifest, "dev_ids": dev["ids"]}


def generation_command(model, name):
    return ["experiments.agent_feedback.feedback_mask", "generate", "--model", str(MODEL_ROOT),
            "--adapter", str(ROOT / model / "model"), "--questions", str(DEV_QUESTIONS),
            "--output", str(ROOT / f"{name}.jsonl"), "--kb", str(KB), "--batch-size", "8",
            "--limit", "0", "--offset", "0", "--seed", "20261003", "--max-context", "8192",
            "--max-generated", "2048", "--feedback-mode", MODE]


def validate_job(model, name, gpu):
    job = read(ROOT / "jobs" / f"{name}.json")
    if (job.get("name") != name or job.get("status") != "completed" or job.get("returncode") != 0
            or job.get("user") != "syx" or type(job.get("uid")) is not int or job["uid"] <= 0
            or job.get("gpus") != [gpu]
            or job.get("command") != [sys.executable, "-u", "-m", *generation_command(model, name)]):
        raise ValueError(f"Incomplete or redirected diagnostic job ledger: {name}")
    for key in ("started_at", "finished_at", "gpu_seconds"):
        if type(job.get(key)) not in (int, float) or not math.isfinite(job[key]):
            raise ValueError(f"Missing or invalid job accounting: {name}")
    if (job["finished_at"] < job["started_at"] or job["gpu_seconds"] < 0
            or not math.isclose(job["gpu_seconds"], job["finished_at"] - job["started_at"], abs_tol=.001)):
        raise ValueError(f"Inconsistent job accounting: {name}")


def validate_feedback(model, name, config):
    path = ROOT / f"{name}.jsonl"
    metadata = read(path.with_suffix(".feedback.json"))
    started = read(path.with_suffix(".feedback_started.json"))
    paths = {"inference": CODE / "inference.py", "environment": CODE / "environment.py",
             "renderer": CODE / "feedback_mask.py", "questions": DEV_QUESTIONS, "kb": KB,
             "adapter_config": ROOT / model / "model/adapter_config.json",
             "adapter_weights": ROOT / model / "model/adapter_model.safetensors"}
    expected = {"feedback_mode": MODE, "masked_fields": ["error", "detail"],
                "replacement": {"error": "ExecutionError", "detail": "execution_failed"},
                "questions": 500, "config": config,
                "inputs_sha256": {key: digest(source) for key, source in paths.items()}}
    if any(record.get(key) != value for record in (metadata, started) for key, value in expected.items()):
        raise ValueError(f"Diagnostic feedback source, mode or configuration differs: {name}")
    if (metadata.get("status") != "completed" or metadata.get("predictions_sha256") != digest(path)
            or metadata.get("runtime_sha256") != digest(path.with_suffix(".runtime.json"))):
        raise ValueError(f"Diagnostic feedback metadata does not bind its completed output: {name}")
    counts = metadata.get("rendering_counts", {})
    if (any(type(counts.get(key)) is not int or counts[key] < 0 for key in
            ("observations", "failed_observations", "masked_observations"))
            or counts["masked_observations"] != counts["failed_observations"]
            or counts["failed_observations"] > counts["observations"]):
        raise ValueError(f"Invalid diagnostic rendering counts: {name}")


def validate_outputs(snapshot):
    # Import the new validator only when generation has completed. Importing
    # this controller does not run inference, initialize a KB or open any split.
    from .feedback_diagnostic_analysis import validate_diagnostic_predictions
    questions = read_jsonl(DEV_QUESTIONS)
    executor = KoPLExecutor(str(KB))
    audits = {}
    for arm, seed, model, normal, name, gpu in RUNS:
        validate_job(model, name, gpu)
        path = ROOT / f"{name}.jsonl"
        require_dev_predictions(path, snapshot["dev_ids"])
        runtime = read(path.with_suffix(".runtime.json"))
        expected_config = {**inference_config(model, name), "feedback_mode": MODE}
        if (runtime["status"] != "completed" or runtime["selected_questions"] != 500
                or runtime["source_questions"] != 500 or runtime["totals"]["questions"] != 500
                or runtime["config"] != expected_config):
            raise ValueError(f"Incomplete, redirected or incorrectly configured diagnostic: {name}")
        validate_feedback(model, name, expected_config)
        audit = validate_diagnostic_predictions(
            read_jsonl(ROOT / f"{normal}.jsonl"), read_jsonl(path), questions, executor=executor)
        if audit.get("audit_passed") is not True or audit.get("mismatches"):
            raise ValueError(f"Diagnostic execution or visible-message audit failed: {name}")
        audits[f"{arm}_{seed}"] = audit
    return audits


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require_syx()
    if os.environ.get("PYTHONHASHSEED") != "20261003":
        raise ValueError("Start the diagnostic controller with PYTHONHASHSEED=20261003 for CPU replay")
    snapshot = validate_inputs()
    require_runtime(snapshot)
    require_idle_gpus()
    used = require_budget(4 * HOURS_PER_ARM)
    require_hashes(snapshot["inputs_sha256"])
    jobs = [(name, gpu, generation_command(model, name), HOURS_PER_ARM)
            for _, _, model, _, name, gpu in RUNS]
    protocol = {"version": "error_diagnostic_feedback_ablation_v1", "frozen_at": time.time(),
        "question_scope": "same complete frozen dev500 for every model; no error-exposed-only selection",
        "feedback_mode": MODE, "intervention": {"only_when": "actual observation ok is false",
            "error": "ExecutionError", "detail": "execution_failed",
            "preserved": "ok=false, other fields, all successful observations, actual Episode state and raw events"},
        "primary_results": "Original normal-feedback results remain primary; this diagnostic cannot select a seed/model or change the method",
        "inference_seed": 20261003, "python_hash_seed": 20261003, "decoding": "greedy", "batch_size": 8,
        "budgets": {"max_calls": 24, "per_turn_generated_tokens": 192, "total_generated_tokens": 2048,
                    "max_context": 8192, "gpu_hours_per_model": 1, "maximum_additional_gpu_hours": 4,
                    "global_gpu_hours_cap": 72}, "gpu_hours_used_before_start": used,
        "timeout_note": "Existing run_job polling and shutdown can add seconds of overhead",
        "models": [{"arm": arm, "continuation_seed": seed, "adapter": str(ROOT / model / "model"),
                    "normal_predictions": str(ROOT / f"{normal}.jsonl"),
                    "masked_predictions": str(ROOT / f"{name}.jsonl")} for arm, seed, model, normal, name, _ in RUNS],
        "jobs": jobs, "inputs_sha256": snapshot["inputs_sha256"], "versions": snapshot["versions"],
        "stopping_rule": "Exactly one four-model diagnostic; retain all outcomes. No automatic repeat, training, model selection or holdout evaluation",
        "interpretation_limits": ["Masks diagnostic content, not the failure status or successful execution feedback",
            "Each model's error-exposed subset is model-dependent; full dev500 is the primary denominator",
            "Changed message lengths and distribution shift are part of the intervention, not isolated semantic information removal",
            "Pre-mask and no-error control divergences must be reported before interpreting feedback use",
            "The same 500 questions across continuation seeds are not independent extra questions",
            "Only 4 of 418 accepted recovery-pair divergence actions were rejected; this narrow error-content ablation is not a gate on the whole recovery-training method",
            "Full-prefill plus generated-token counts are not GPU FLOPs"],
        "training": False, "held_out_evaluation": False, "automatic_next_round": False}
    with MARKER.open("x") as handle:
        json.dump({"started_at": time.time(), "script_sha256": digest(__file__), "protocol": str(PROTOCOL)}, handle)
    with PROTOCOL.open("x") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_job, *job) for job in jobs]
        for future in futures:
            future.result()
    require_hashes(snapshot["inputs_sha256"])
    audits = validate_outputs(snapshot)
    with AUDIT.open("x") as handle:
        json.dump({"protocol_sha256": digest(PROTOCOL), "models": audits}, handle, ensure_ascii=False, indent=2)
    # No scoring or outcome analysis starts until all four full outputs pass the
    # structural/execution/visibility audit. Existing normal scores are retained.
    reports = {}
    for arm, seed, _, normal, name, _ in RUNS:
        predictions, report = ROOT / f"{name}.jsonl", ROOT / f"{name}.diagnostic.json"
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.inference", "score",
                        "--predictions", str(predictions), "--gold", str(DEV_GOLD)], check=True)
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.feedback_diagnostic_analysis",
                        "--normal", str(ROOT / f"{normal}.jsonl"), "--masked", str(predictions),
                        "--output", str(report)], check=True)
        reports[f"{arm}_{seed}"] = {"path": str(report), "sha256": digest(report)}
    require_hashes(snapshot["inputs_sha256"])
    with SUMMARY.open("x") as handle:
        json.dump({"finished_at": time.time(), "protocol_sha256": digest(PROTOCOL),
            "audit_sha256": digest(AUDIT), "reports": reports, "primary_results": "original normal-feedback outputs",
            "pre_intervention_equivalence_passed": all(audit.get("pre_intervention_equivalence_passed") is True
                                                        for audit in audits.values()),
            "mechanism_interpretation": "bounded_error_content_diagnostic_only" if all(
                audit.get("pre_intervention_equivalence_passed") is True for audit in audits.values())
                else "needs_numerical_batching_review",
            "model_selection": False, "training": False, "held_out_evaluation": False,
            "automatic_next_round": False}, handle, indent=2)
    print("FEEDBACK_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_TRAINING_OR_HOLDOUT", flush=True)


if __name__ == "__main__":
    main()
