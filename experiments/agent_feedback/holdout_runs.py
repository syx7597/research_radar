"""Freeze, generate, then analyze one fixed five-model project holdout.

Freezing copies split metadata without opening holdout questions or answers.
Generation reads questions only. Analysis is a separate explicit command and
cannot open answers until every fixed model has complete, bound predictions.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from . import inference
from .compare import indexed, validate_prediction
from .continuation_runs import ADAPTER, MANIFEST, require_syx, validate_training
from .holdout_analysis import analyze_predictions
from .initial_runs import MODEL_ROOT, require_idle_gpus, run_job
from .replication_runs import require_budget, require_hashes, require_runtime, training_config
from .recovery import Excluded, replay_rollout
from .train_sft import digest
from .training_data import FUNCTION_HELP, read_jsonl
from experiments.condition_consistency.executor import KoPLExecutor

ROOT = Path("results/agent_feedback")
CODE = Path("experiments/agent_feedback")
KB = Path("datasets/kqa_pro/kb.json")
QUESTIONS = Path("data/agent_feedback/holdout.questions.jsonl")
GOLD = Path("data/agent_feedback/holdout.gold.jsonl")
DECISION = ROOT / "feedback_diagnostic_decision_v1.json"
DIAGNOSTIC = ROOT / "feedback_diagnostic_summary_v1.json"
DIAGNOSTIC_PROTOCOL = ROOT / "protocol_feedback_diagnostic_v1.json"
DIAGNOSTIC_AUDIT = ROOT / "feedback_diagnostic_audit_v1.json"
REVIEW = ROOT / "replication_review_seed20261004_v2.json"
SPLIT = ROOT / "split_manifest.json"
PROTOCOL = ROOT / "protocol_holdout_v1.json"
MARKER = ROOT / "holdout_v1_started.json"
COMPLETE = ROOT / "holdout_v1_generation_complete.json"
ANALYSIS_MARKER = ROOT / "holdout_v1_analysis_started.json"
AUDIT = ROOT / "holdout_v1_execution_audit.json"
REPORT = ROOT / "holdout_v1_analysis.json"
SUMMARY = ROOT / "holdout_v1_summary.json"
SEED = 20261003
RUNS = (("A_20261003", "clean_sft_v1", "clean_holdout_v1", 0, 2.5, False),
        ("C_20261003", "recovery_sft_v1", "recovery_holdout_v1", 1, 2.5, False),
        ("A_20261004", "clean_sft_seed20261004", "clean_holdout_seed20261004", 2, 2.5, False),
        ("C_20261004", "recovery_sft_seed20261004", "recovery_holdout_seed20261004", 3, 2.5, False),
        ("P", "program_sft_v1", "program_holdout_v1", 0, .5, True))
STATISTICS = {
    "primary": "Per question, mean of C-minus-A correctness across both continuation seeds; then mean across questions",
    "uncertainty": "Paired percentile bootstrap of 2000 question units, retaining both seed outcomes together",
    "bootstrap_replicates": 5000, "bootstrap_seed": SEED, "confidence_level": .95,
    "secondary": "Each seed C versus A and C versus P; all five accuracies and inference costs",
    "subgroups": "Fixed reference-length, qualifier, set-operation and numeric/comparison groups; descriptive only",
    "practical_reference_gain_pp": 2,
    "limitations": "Conditional on two continuation checkpoints sharing the initial model and corpus; no training-seed uncertainty; 2000 question units, not 4000; token counts are not FLOPs",
    "after_results": "Report all fixed models and outcomes; no tuning, checkpoint selection, or automatic next round"}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    with path.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def require_non_holdout_hashes(hashes):
    # Even an accidentally overbroad external decision must not open the two
    # held-out files during freeze or before all five predictions are complete.
    forbidden = {QUESTIONS.resolve(), GOLD.resolve()}
    if any(Path(path).resolve() in forbidden for path in hashes):
        raise ValueError("Holdout file hashes must remain deferred split metadata, not freeze inputs")
    require_hashes(hashes)


def inference_config(model, name, program):
    return {"command": "generate", "model": str(MODEL_ROOT), "adapter": str(ROOT / model / "model"),
            "questions": str(QUESTIONS), "output": str(ROOT / f"{name}.jsonl"), "kb": str(KB),
            "batch_size": 8, "limit": 0, "offset": 0, "seed": SEED,
            "max_context": 8192, "max_generated": 2048, "program": program}


def generation_command(model, name, program):
    command = ["experiments.agent_feedback.inference", "generate", "--model", str(MODEL_ROOT),
               "--adapter", str(ROOT / model / "model"), "--questions", str(QUESTIONS),
               "--output", str(ROOT / f"{name}.jsonl"), "--kb", str(KB), "--batch-size", "8",
               "--limit", "0", "--offset", "0", "--seed", str(SEED),
               "--max-context", "8192", "--max-generated", "2048"]
    return command + (["--program"] if program else [])


def run_specs():
    return [{"label": label, "model": model, "name": name, "gpu": gpu, "timeout_hours": hours,
             "config": inference_config(model, name, program),
             "command": generation_command(model, name, program)}
            for label, model, name, gpu, hours, program in RUNS]


def require_new_outputs(include_protocol=False):
    paths = [MARKER, COMPLETE, ANALYSIS_MARKER, AUDIT, REPORT, SUMMARY]
    if include_protocol:
        paths.append(PROTOCOL)
    for _, _, name, _, _, _ in RUNS:
        paths.extend(ROOT / (name + suffix) for suffix in
                     (".jsonl", ".runtime.json", ".metrics.json", ".scored.jsonl"))
        paths.extend(ROOT / "jobs" / (name + suffix) for suffix in (".json", ".log"))
    if any(path.exists() for path in paths):
        raise FileExistsError("Holdout artifacts already exist; never resume, overwrite, or repeat")


def diagnostic_paths():
    return [DIAGNOSTIC, DIAGNOSTIC_PROTOCOL, DIAGNOSTIC_AUDIT, *(
        ROOT / f"{name}.diagnostic.json" for name in (
            "clean_feedback_masked_dev_v1", "recovery_feedback_masked_dev_v1",
            "clean_feedback_masked_dev_seed20261004", "recovery_feedback_masked_dev_seed20261004"))]


def model_paths(model):
    return [ROOT / model / name for name in (
        "run_result.json", "input_manifest.json", "model/adapter_config.json", "model/adapter_model.safetensors")]


def validate_split_metadata(split):
    spec = split["splits"]["holdout"]
    ids = spec["ids"]
    if spec["count"] != 2000 or len(ids) != 2000 or len(set(ids)) != 2000:
        raise ValueError("Expected the original complete project holdout2000 metadata")
    for other in ("train", "dev"):
        if set(ids) & set(split["splits"][other]["ids"]):
            raise ValueError("Holdout IDs overlap training or development")
    for key, path in (("questions", QUESTIONS), ("gold", GOLD)):
        entry = spec[key]
        if (Path(entry["path"]).resolve() != path.resolve() or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError(f"Invalid frozen holdout {key} metadata")
    return spec


def decision_required_paths():
    return [*diagnostic_paths(), REVIEW, SPLIT, ROOT / "continuation_code_manifest_v1.json",
            ROOT / "weights_verified.json", MANIFEST, ROOT / "protocol.json", KB,
            CODE / "holdout_runs.py", CODE / "holdout_analysis.py", CODE / "recovery.py",
            Path("experiments/condition_consistency/executor.py"),
            Path("external/kqa_pro_baselines/utils/value_class.py"),
            *(path for _, model, _, _, _, _ in RUNS for path in model_paths(model))]


def validate_freeze_inputs():
    decision = read(DECISION)
    if decision.get("gate", {}).get("allow_holdout_freeze") is not True:
        raise ValueError("Diagnostic closure decision has not authorized holdout freezing")
    hashes = decision["inputs_sha256"]
    if not {str(path) for path in decision_required_paths()} <= set(hashes):
        raise ValueError("Closure decision must bind diagnostic reports, all five original models, split metadata and analysis code")
    require_non_holdout_hashes(hashes)
    summary, audit = read(DIAGNOSTIC), read(DIAGNOSTIC_AUDIT)
    labels = {row[0] for row in RUNS[:4]}
    if (summary["protocol_sha256"] != digest(DIAGNOSTIC_PROTOCOL)
            or summary["audit_sha256"] != digest(DIAGNOSTIC_AUDIT)
            or set(summary["reports"]) != labels or set(audit["models"]) != labels
            or any(row.get("audit_passed") is not True or row.get("mismatches") for row in audit["models"].values())
            or any(summary.get(key) is not False for key in
                   ("model_selection", "training", "held_out_evaluation", "automatic_next_round"))):
        raise ValueError("The complete four-model diagnostic must be reviewed before freezing")
    for item in summary["reports"].values():
        if Path(item["path"]) not in diagnostic_paths() or digest(item["path"]) != item["sha256"]:
            raise ValueError("Diagnostic report binding differs")
    review = read(REVIEW)
    if review.get("gate", {}).get("audit_passed") is not True or any(review.get("issues", {}).values()):
        raise ValueError("The original paired continuation audit did not pass")
    reviewed = review["inputs_sha256"]
    if not {str(p) for _, model, _, _, _, _ in RUNS[:4] for p in model_paths(model)} <= set(reviewed):
        raise ValueError("All four A/C checkpoint identities must match the pre-diagnostic review")
    previous = read(ROOT / "continuation_code_manifest_v1.json")["files"]
    diagnostic_inputs = read(DIAGNOSTIC_PROTOCOL)["inputs_sha256"]
    for group in (reviewed, previous, diagnostic_inputs):
        require_non_holdout_hashes(group)
    split = read(SPLIT)
    spec = validate_split_metadata(split)
    if digest(KB) != split["source_sha256"]["kb.json"]:
        raise ValueError("Frozen knowledge base identity changed")
    manifest = read(MANIFEST)
    initial_hash = digest(ADAPTER / "adapter_model.safetensors")
    versions = None
    for label, model, _, _, _, _ in RUNS[:4]:
        arm, seed = label.split("_")
        validate_training(arm, model, manifest, digest(MANIFEST), initial_hash)
        inputs, result = read(ROOT / model / "input_manifest.json"), read(ROOT / model / "run_result.json")
        if (inputs["config"] != training_config(arm, model, int(seed)) or result["seed"] != int(seed)
                or versions is not None and inputs["versions"] != versions):
            raise ValueError("Holdout A/C checkpoints must retain both pre-reviewed continuation seeds")
        versions = inputs["versions"]
    program = ROOT / "program_sft_v1"
    inputs, result = read(program / "input_manifest.json"), read(program / "run_result.json")
    config = inputs["config"]
    if (result["status"] != "completed" or not math.isfinite(result["metrics"]["train_loss"])
            or result["seed"] != SEED or inputs["versions"] != versions
            or result["actual_supervised_tokens"] != 3 * inputs["supervised_tokens"]
            or result["script_sha256"] != digest(CODE / "train_sft.py")
            or result["data_sha256"] != split["splits"]["train"]["gold"]["sha256"]
            or any(config.get(key) != value for key, value in {
                "model": str(MODEL_ROOT), "output": str(program), "data": "data/agent_feedback/train.gold.jsonl",
                "program": True, "adapter": None, "pretokenized": False, "epochs": 3,
                "seed": SEED, "max_steps": -1, "limit": 0, "tokenize_only": False}.items())):
        raise ValueError("Program baseline must be the original completed three-epoch P adapter")
    # All entries are pre-holdout evidence. Deferred question/gold hashes stay
    # only inside split metadata and are checked in their own later stages.
    frozen = {**hashes, **reviewed, **previous, **diagnostic_inputs, str(DECISION): digest(DECISION)}
    return {"inputs_sha256": frozen, "holdout": spec, "versions": versions, "manifest": manifest}


def freeze():
    require_new_outputs(include_protocol=True)
    snapshot = validate_freeze_inputs()
    protocol = {"version": "fixed_five_model_holdout_v1", "frozen_at": time.time(), **snapshot,
                "models": run_specs(), "statistics": STATISTICS,
                "method": "Matched-supervision real-failure recovery SFT versus clean continuation; both fixed continuation seeds, shared initial adapter; P is the contextual program baseline",
                "inference": {"seed": SEED, "python_hash_seed": SEED, "decoding": "greedy", "batch_size": 8,
                              "agent_max_calls": 24, "agent_per_turn_generated_tokens": 192,
                              "total_generated_tokens": 2048, "max_context": 8192, "feedback": "original"},
                "maximum_additional_gpu_hours": 10.5, "global_gpu_hours_cap": 72,
                "schedule": "Four agent models concurrently, then P on GPU0; no automatic analysis",
                "timeout_note": "Existing supervisor polling and shutdown can add seconds of overhead",
                "gold_policy": "Hash/parse questions at run; hash/parse answers only in explicit analyze after all five complete outputs are bound",
                "execution_audit": "During explicit analyze, replay all 8000 agent trajectories and re-execute all emitted P programs using questions only; save audit and require zero differences before gold access",
                "training": False, "model_selection": False, "automatic_next_round": False}
    write_new(PROTOCOL, protocol)
    print("HOLDOUT_PROTOCOL_FROZEN_WITHOUT_OPENING_HOLDOUT_FILES", flush=True)


def read_frozen():
    protocol = read(PROTOCOL)
    if (protocol.get("version") != "fixed_five_model_holdout_v1" or protocol["models"] != run_specs()
            or protocol["statistics"] != STATISTICS or protocol["maximum_additional_gpu_hours"] != 10.5
            or protocol["global_gpu_hours_cap"] != 72
            or any(protocol.get(key) is not False for key in ("training", "model_selection", "automatic_next_round"))):
        raise ValueError("Holdout protocol differs from the fixed five-model plan")
    require_non_holdout_hashes(protocol["inputs_sha256"])
    if protocol["holdout"] != validate_split_metadata(read(SPLIT)):
        raise ValueError("Deferred holdout metadata changed after freezing")
    return protocol


def require_questions(protocol):
    if digest(QUESTIONS) != protocol["holdout"]["questions"]["sha256"]:
        raise ValueError("Frozen holdout questions changed")
    questions = read_jsonl(QUESTIONS)
    inference.validate_questions(questions)
    if [row["id"] for row in questions] != protocol["holdout"]["ids"]:
        raise ValueError("Holdout questions must retain the complete frozen order")
    return questions


def validate_output(spec, protocol):
    name = spec["name"]
    path, job_path = ROOT / f"{name}.jsonl", ROOT / "jobs" / f"{name}.json"
    job = read(job_path)
    if (job.get("name") != name or job.get("status") != "completed" or job.get("returncode") != 0
            or job.get("user") != "syx" or type(job.get("uid")) is not int or job["uid"] <= 0
            or job.get("gpus") != [spec["gpu"]]
            or job.get("command") != [sys.executable, "-u", "-m", *spec["command"]]):
        raise ValueError(f"Invalid or incomplete holdout job: {name}")
    if (any(type(job.get(key)) not in (int, float) or not math.isfinite(job[key])
            for key in ("started_at", "finished_at", "gpu_seconds"))
            or job["finished_at"] < job["started_at"] or job["gpu_seconds"] < 0
            or not math.isclose(job["gpu_seconds"], job["finished_at"] - job["started_at"], abs_tol=.001)):
        raise ValueError(f"Invalid holdout GPU accounting: {name}")
    rows = read_jsonl(path)
    if list(indexed(rows, name)) != protocol["holdout"]["ids"]:
        raise ValueError(f"Holdout output is not the complete ordered 2000 questions: {name}")
    for row in rows:
        validate_prediction(row, name)
        if (row["generated_tokens"] > 2048 or not spec["config"]["program"] and row["calls"] > 24):
            raise ValueError(f"Prediction exceeds the frozen inference budgets: {name}")
    runtime_path = path.with_suffix(".runtime.json")
    runtime = read(runtime_path)
    totals = {key: sum(row[key] for row in rows) for key in
              ("calls", "input_tokens", "generated_tokens", "total_tokens", "invalid_calls")}
    if (runtime.get("status") != "completed" or runtime.get("selected_questions") != 2000
            or runtime.get("source_questions") != 2000 or runtime.get("config") != spec["config"]
            or runtime.get("totals") != {**totals, "questions": 2000}
            or type(runtime.get("seconds")) not in (int, float) or not math.isfinite(runtime["seconds"])
            or runtime["seconds"] < 0):
        raise ValueError(f"Holdout runtime/configuration/accounting differs: {name}")
    return rows, {str(p): digest(p) for p in (path, runtime_path, job_path)}


def validate_all_outputs(protocol):
    predictions, hashes = {}, {}
    for spec in protocol["models"]:
        predictions[spec["label"]], bound = validate_output(spec, protocol)
        hashes.update(bound)
    return predictions, hashes


def replay_predictions(predictions, questions, executor):
    """Gold-free audit using the unchanged execution and replay implementations."""
    by_question = indexed(questions, "replay questions")
    inference.validate_questions(questions)
    mismatches, counts = [], {}
    for label, rows in predictions.items():
        if set(indexed(rows, label)) != set(by_question):
            raise ValueError("Replay requires complete matching question IDs")
        counts[label] = {"trajectories": len(rows), "mismatches": 0, "executed_programs": 0}
        for row in rows:
            try:
                question = by_question[row["id"]]
                if label != "P":
                    _, episode = replay_rollout(executor, row, question, 24)
                    if (episode.selected != row["selected_handle"] or episode.calls != row["calls"]
                            or sum(not event["observation"]["ok"] for event in row["events"]) != row["invalid_calls"]
                            or episode.done and row["stop_reason"] != (
                                "finished" if episode.prediction is not None else "call_budget")
                            or not episode.done and row["stop_reason"] not in {"context_budget", "generation_budget"}):
                        raise ValueError("Agent final state, stop reason or accounting differs from replay")
                else:
                    prefix = [{"role": "system", "content":
                        "Translate the question into a complete executable KoPL program. "
                        "Output only function and literal inputs separated by <arg>, with steps separated by <func>. "
                        "Dependencies follow the standard branch-stack order.\n" + FUNCTION_HELP},
                        {"role": "user", "content": question["question"]}]
                    messages = row["messages"]
                    if messages[:2] != prefix or row["events"] != [] or row["selected_handle"] is not None:
                        raise ValueError("Program prompt or event state differs")
                    if "program_result" in row:
                        if (len(messages) != 3 or messages[2].get("role") != "assistant"
                                or not isinstance(messages[2].get("content"), str)):
                            raise ValueError("Expected exactly one raw program response")
                        result = executor.execute(messages[2]["content"])
                        counts[label]["executed_programs"] += 1
                        if (result != row["program_result"] or result["prediction"] != row["prediction"]
                                or row["calls"] != 1 or row["invalid_calls"] != int(not result["valid"])
                                or row["stop_reason"] != ("program_completed" if result["valid"] else "program_error")):
                            raise ValueError("Program re-execution differs from its saved result")
                    elif (len(messages) != 2 or row["calls"] != 0 or row["invalid_calls"] != 0
                          or row["prediction"] is not None or row["stop_reason"] != "context_budget"):
                        raise ValueError("Unexecuted program has inconsistent context-stop state")
            except (Excluded, ValueError, TypeError, KeyError, IndexError) as exc:
                mismatches.append({"label": label, "id": row["id"],
                    "reason": getattr(exc, "reason", str(exc)), "detail": getattr(exc, "detail", None)})
                counts[label]["mismatches"] += 1
    return {"audit_passed": not mismatches, "mismatch_count": len(mismatches),
            "mismatches": mismatches, "models": counts, "questions": len(questions),
            "scope": "Actual recorded execution consistency, not semantic equivalence or answer correctness",
            "gold_opened": False}


def run():
    protocol = read_frozen()
    require_new_outputs()
    require_runtime(protocol)
    require_questions(protocol)
    require_idle_gpus()
    used = require_budget(10.5)
    protocol_hash = digest(PROTOCOL)
    write_new(MARKER, {"started_at": time.time(), "protocol_sha256": protocol_hash,
                       "gpu_hours_used_before_start": used, "script_sha256": digest(__file__)})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_job, spec["name"], spec["gpu"], spec["command"], spec["timeout_hours"])
                   for spec in protocol["models"][:4]]
        for future in futures:
            future.result()
    for spec in protocol["models"][:4]:
        validate_output(spec, protocol)
    require_idle_gpus()
    require_budget(.5)
    spec = protocol["models"][4]
    run_job(spec["name"], spec["gpu"], spec["command"], spec["timeout_hours"])
    _, hashes = validate_all_outputs(protocol)
    require_non_holdout_hashes(protocol["inputs_sha256"])
    require_questions(protocol)
    if digest(PROTOCOL) != protocol_hash:
        raise ValueError("Holdout protocol changed during generation")
    write_new(COMPLETE, {"status": "completed", "finished_at": time.time(), "questions_per_model": 2000,
                         "labels": [spec["label"] for spec in protocol["models"]],
                         "protocol_sha256": protocol_hash, "started_sha256": digest(MARKER),
                         "outputs_sha256": hashes, "gold_opened": False, "analysis_started": False})
    print("ALL_FIVE_HOLDOUT_PREDICTIONS_COMPLETE_GOLD_UNOPENED_RUN_ANALYZE_SEPARATELY", flush=True)


def analyze():
    protocol = read_frozen()
    completion = read(COMPLETE)
    if (completion.get("status") != "completed" or completion.get("questions_per_model") != 2000
            or completion.get("labels") != [spec["label"] for spec in protocol["models"]]
            or completion.get("protocol_sha256") != digest(PROTOCOL)
            or completion.get("started_sha256") != digest(MARKER)
            or read(MARKER)["protocol_sha256"] != digest(PROTOCOL)
            or completion.get("gold_opened") is not False or completion.get("analysis_started") is not False):
        raise ValueError("All five bound generations must complete before any gold access")
    expected_paths = {str(path) for spec in protocol["models"] for path in (
        ROOT / f"{spec['name']}.jsonl", ROOT / f"{spec['name']}.runtime.json",
        ROOT / "jobs" / f"{spec['name']}.json")}
    if set(completion["outputs_sha256"]) != expected_paths:
        raise ValueError("Completion must bind exactly the five prediction/runtime/job artifact sets")
    require_non_holdout_hashes(completion["outputs_sha256"])
    predictions, actual_hashes = validate_all_outputs(protocol)
    if actual_hashes != completion["outputs_sha256"]:
        raise ValueError("Complete prediction artifact bindings differ")
    paths = [ANALYSIS_MARKER, AUDIT, REPORT, SUMMARY, *(ROOT / (spec["name"] + suffix)
        for spec in protocol["models"] for suffix in (".metrics.json", ".scored.jsonl"))]
    if any(path.exists() for path in paths):
        raise FileExistsError("Holdout analysis already started; preserve all reports")
    questions = require_questions(protocol)
    write_new(ANALYSIS_MARKER, {"started_at": time.time(), "protocol_sha256": digest(PROTOCOL),
                               "generation_complete_sha256": digest(COMPLETE), "gold_opened_at_start": False})
    audit = replay_predictions(predictions, questions, KoPLExecutor(str(KB)))
    write_new(AUDIT, {**audit, "protocol_sha256": digest(PROTOCOL),
                      "generation_complete_sha256": digest(COMPLETE)})
    if audit["audit_passed"] is not True or audit["mismatch_count"] != 0:
        raise ValueError("Holdout CPU execution replay differs; preserve audit and do not open gold")
    # First permitted touch of answers: all five outputs AND gold-free replay
    # have passed. Failed execution audits remain available without scoring.
    if digest(GOLD) != protocol["holdout"]["gold"]["sha256"]:
        raise ValueError("Frozen holdout gold changed")
    gold = read_jsonl(GOLD)
    if list(indexed(gold, "holdout gold")) != protocol["holdout"]["ids"]:
        raise ValueError("Holdout gold does not cover the original ordered questions")
    report = analyze_predictions(predictions, gold, bootstrap_replicates=STATISTICS["bootstrap_replicates"])
    for spec in protocol["models"]:
        inference.score(SimpleNamespace(predictions=str(ROOT / f"{spec['name']}.jsonl"), gold=str(GOLD)))
    require_hashes(completion["outputs_sha256"])
    require_non_holdout_hashes(protocol["inputs_sha256"])
    write_new(REPORT, report)
    scoring = {str(path): digest(path) for spec in protocol["models"]
               for suffix in (".metrics.json", ".scored.jsonl") for path in [ROOT / (spec["name"] + suffix)]}
    write_new(SUMMARY, {"status": "completed", "finished_at": time.time(), "protocol_sha256": digest(PROTOCOL),
                        "generation_complete_sha256": digest(COMPLETE), "analysis": str(REPORT),
                        "execution_audit_sha256": digest(AUDIT), "execution_replay_mismatches": 0,
                        "analysis_sha256": digest(REPORT), "scoring_outputs_sha256": scoring,
                        "gold_sha256": protocol["holdout"]["gold"]["sha256"],
                        "training": False, "model_selection": False, "automatic_next_round": False})
    print("FIXED_HOLDOUT_ANALYSIS_COMPLETE_NO_AUTOMATIC_TRAINING_OR_SELECTION", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "analyze"))
    args = parser.parse_args()
    require_syx()
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise ValueError("Start this controller with PYTHONHASHSEED=20261003")
    {"freeze": freeze, "run": run, "analyze": analyze}[args.stage]()


if __name__ == "__main__":
    main()
