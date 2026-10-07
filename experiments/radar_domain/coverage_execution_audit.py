"""CPU-only mechanical audit of frozen generation artifacts; never semantic grading.

Raw answers are decoded/compared only inside this program. Public output uses an
explicit aggregate allowlist. Remote weights are checked through archived hash
attestations; the auditor does not claim to rehash unavailable remote weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform

from . import coverage_evaluate as evaluate
from . import coverage_runtime_execute as runtime
from .source_readings import ROOT

RESULT_KEYS = {"question_id", "arm", "status", "input_tokens", "prompt_ids_sha256",
    "generated_ids", "generated_text", "generated_tokens", "stop_reason", "seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "truncated", "format_diagnostic"}
ACCOUNTING = "Entire single-GPU child lifetime, including imports, loading, generation and cleanup; CPU preflight excluded."
DEFAULT_NORMALIZATIONS = {
    "encoder_repetition_penalty": 1.0, "return_dict_in_generate": False,
    "early_stopping": False, "remove_invalid_values": False, "min_length": 0,
    "output_scores": False, "typical_p": 1.0, "target_lookbehind": 10,
    "encoder_no_repeat_ngram_size": 0, "diversity_penalty": 0.0,
    "no_repeat_ngram_size": 0, "eta_cutoff": 0.0, "assistant_confidence_threshold": 0.4,
    "num_assistant_tokens": 20, "num_assistant_tokens_schedule": "constant",
    "epsilon_cutoff": 0.0, "max_length": 20, "assistant_lookbehind": 10,
}
PUBLIC_TOTAL_KEYS = ("question_count", "mechanical_valid_rows", "strict_format_valid_count",
    "strict_format_invalid_count", "input_tokens", "generated_tokens", "stop_reasons",
    "gpu_seconds", "gpu_hours", "generation_seconds", "peak_allocated_bytes", "peak_reserved_bytes")


class Checks:
    def __init__(self):
        self.issues = []
        self.files = {}

    def check(self, condition, code, **context):
        if not condition:
            self.issues.append({"code": code, **context})
        return bool(condition)

    def read(self, path):
        path = Path(path)
        if not self.check(path.is_file() and not path.is_symlink(), "missing_or_symlink_file", file=str(path)):
            return None
        try:
            self.files[str(path)] = runtime.digest(path)
            return runtime.strict_json(path.read_text())
        except (ValueError, OSError, UnicodeError):
            self.check(False, "invalid_json_file", file=str(path))
            return None


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def audit_row(saved, source, marker, environment, tokenizer, checks):
    """Validate token/serialization invariants without interpreting answer meaning."""
    context = {"arm": source["arm"], "question_id": source["question_id"]}
    before = len(checks.issues)
    def check(ok, code):
        return checks.check(ok, code, **context)
    info = {**context, "mechanical_passed": False}
    if not check(type(saved) is dict and set(saved) == RESULT_KEYS, "output_schema"):
        return info
    check(saved["question_id"] == source["question_id"] and saved["arm"] == source["arm"]
          and saved["status"] == "generated", "output_identity")
    check(saved["truncated"] is False and source["truncated"] is False, "truncation")
    check(type(saved["input_tokens"]) is int and saved["input_tokens"] == source["input_tokens"]
          and saved["prompt_ids_sha256"] == source["token_ids_sha256"], "prompt_identity")
    ids = saved["generated_ids"]
    vocabulary_size = len(tokenizer)
    valid_ids = check(type(ids) is list and 0 < len(ids) <= 768
          and all(type(i) is int and 0 <= i < vocabulary_size for i in ids), "generated_ids")
    if valid_ids:
        check(type(saved["generated_tokens"]) is int and saved["generated_tokens"] == len(ids), "generated_count")
        check(source["input_tokens"] + len(ids) <= 32768, "context_limit")
        eos = environment["effective_generation_config"]["eos_token_id"]
        eos = eos if type(eos) is list else [eos]
        ended = ids[-1] in eos
        check(not any(i in eos for i in ids[:-1]), "tokens_after_eos")
        check(saved["stop_reason"] == ("eos" if ended else "generation_budget")
              and (ended or len(ids) == 768), "stop_reason")
        try:
            decoded = tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False)
            check(type(saved["generated_text"]) is str and decoded == saved["generated_text"], "generated_text_decode")
        except (ValueError, TypeError, RuntimeError):
            check(False, "generated_text_decode")
        info["generated_tokens"] = len(ids)
        info["stop_reason"] = saved["stop_reason"]
    if type(saved["generated_text"]) is str:
        diagnosis = evaluate.format_diagnostic(saved["generated_text"], source)
        check(saved["format_diagnostic"] == diagnosis, "format_diagnostic_replay")
        info["strict_format_valid"] = diagnosis["valid"]
    check(number(saved["seconds"]), "case_seconds")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        check(type(saved[key]) is int and saved[key] >= 0, key)
    if all(type(saved[k]) is int for k in ("peak_allocated_bytes", "peak_reserved_bytes")):
        check(saved["peak_allocated_bytes"] <= saved["peak_reserved_bytes"]
              <= environment["gpu"]["total_MiB"] * 1024 * 1024, "gpu_memory_bounds")
    check(type(marker) is dict and set(marker) == {"question_id", "arm", "worker_seconds"}
          and marker.get("question_id") == source["question_id"] and marker.get("arm") == source["arm"]
          and number(marker.get("worker_seconds")), "started_marker")
    info.update(seconds=saved["seconds"], peak_allocated_bytes=saved["peak_allocated_bytes"],
                peak_reserved_bytes=saved["peak_reserved_bytes"], input_tokens=source["input_tokens"],
                worker_start_seconds=marker.get("worker_seconds") if type(marker) is dict else None)
    info["mechanical_passed"] = len(checks.issues) == before
    return info


def model_attestation(receipt, protocol, checks, arm):
    """Compare remote file proofs, never follow remote symlinks or load weights."""
    identity = protocol["model_identity"]
    model_expected = {w["file"]: w["sha256"] for w in identity["expected_base_weights"]["files"]}
    model_expected.update({"config.json": identity["expected_config_sha256"],
                           "generation_config.json": identity["expected_generation_config_sha256"]})
    token_expected = {**identity["expected_tokenizer_sha256"], "config.json": identity["expected_config_sha256"]}
    files, view = receipt["model_files"], receipt["model_view_files"]
    checks.check(set(view) == set(model_expected) | {"model.safetensors.index.json"}, "model_view_roster", arm=arm)
    for name, expected_sha in model_expected.items():
        target = view.get(name)
        checks.check(target in files and files[target].get("sha256") == expected_sha,
                     "model_file_hash", arm=arm, file=name)
        checks.check(target == str(PurePosixPath(protocol["model_path"]) / name), "model_source_path", arm=arm, file=name)
    model_targets = set(view.values())
    token_records = {path: value for path, value in files.items() if path not in model_targets}
    checks.check(len(token_records) == len(token_expected), "tokenizer_attestation_roster", arm=arm)
    for name, expected_sha in token_expected.items():
        found = [(p, v) for p, v in token_records.items() if PurePosixPath(p).name == name]
        checks.check(len(found) == 1 and found[0][1].get("sha256") == expected_sha
                     and found[0][0].endswith("/" + str(PurePosixPath(protocol["tokenizer_path"]) / name)),
                     "tokenizer_file_hash", arm=arm, file=name)
    sizes = {w["file"]: w["bytes"] for w in identity["expected_base_weights"]["files"]}
    for path, record in files.items():
        fp = record.get("fingerprint", {})
        checks.check(set(record) == {"sha256", "fingerprint"} and type(record["sha256"]) is str
                     and len(record["sha256"]) == 64 and set(fp) == {"bytes", "mtime_ns", "ctime_ns", "device", "inode"}
                     and all(type(v) is int and v >= 0 for v in fp.values()), "file_fingerprint", arm=arm, file=path)
        if PurePosixPath(path).name in sizes:
            checks.check(fp.get("bytes") == sizes[PurePosixPath(path).name], "weight_size", arm=arm, file=path)
    checks.check(len(files) == len(model_targets) + len(token_expected), "model_attestation_extra_file", arm=arm)


def config_equivalent(prepared, declared):
    """Only the enumerated 5.18.0 None-to-default preparation is equivalent."""
    if set(prepared) != set(declared):
        return False
    for key in declared:
        if prepared[key] == declared[key] and type(prepared[key]) is type(declared[key]):
            continue
        if not (declared[key] is None and key in DEFAULT_NORMALIZATIONS
                and prepared[key] == DEFAULT_NORMALIZATIONS[key]
                and type(prepared[key]) is type(DEFAULT_NORMALIZATIONS[key])):
            return False
    return True


def audit_runtime(report, receipt, env, child, worker, protocol, protocol_sha, arm, checks):
    qids = protocol["question_ids"]
    def check(ok, code):
        return checks.check(ok, code, arm=arm)
    check(report["schema"] == "radar_coverage_arm_execution_v1" and report["arm"] == arm
          and report["protocol_sha256"] == protocol_sha, "runtime_binding")
    check(report["status"] == "completed" and report["returncode"] == 0
          and report["error"] is None and report["process_started"] is True
          and report["process_reaped"] is True and report["GPU_hours_is_lower_bound"] is False
          and report["within_arm_budget"] is True, "runtime_completion")
    check(report["rows"] == [{"question_id": q, "status": "generated"} for q in qids]
          and report["question_denominator"] == report["generated_count"] == 96
          and report["not_run_question_ids"] == [], "runtime_inventory")
    check(report["new_training_runs"] == report["automatic_retries"] == 0
          and report["semantic_scoring_performed"] is False, "no_training_retries_scoring")
    seconds, hours = report["gpu_seconds"], report["gpu_hours"]
    check(number(seconds) and 0 < seconds <= 2400 and number(hours)
          and math.isclose(hours, seconds / 3600, rel_tol=0, abs_tol=1e-12), "gpu_time_accounting")
    check(report["accounting_basis"] == ACCOUNTING and report["formal_round_reserved_gpu_seconds"] == 7200
          and report["project_gpu_hours_before"] == protocol["execution"]["project_gpu_hours_before"]
          and report["project_gpu_hours_budget"] == 72, "budget_binding")
    check(worker == {"status": "completed", "question_count": 96}, "worker_completion")
    check(receipt["protocol_sha256"] == protocol_sha and receipt["arm"] == arm
          and receipt["all_96_prompt_ids_verified"] is True, "cpu_receipt_binding")
    check({k: env[k] for k in protocol["runtime_environment"]} == protocol["runtime_environment"], "software_identity")
    user = env["identity"]
    check(user["real_user"] == user["effective_user"] == "syx"
          and type(user["real_uid"]) is int and user["real_uid"] == user["effective_uid"] > 0, "os_user_identity")
    check(type(receipt["parent_pid"]) is int and type(child["pid"]) is int
          and receipt["parent_pid"] > 0 and child["pid"] > 0 and receipt["parent_pid"] != child["pid"]
          and child["gpu_count"] == 1 and child["max_seconds"] == 2400, "gpu_child_identity")
    freeze_time = datetime.fromisoformat(protocol["frozen_at_utc"]).timestamp()
    check(number(child["started_at_unix"], freeze_time), "run_after_protocol_freeze")
    gpu = env["gpu"]
    check(gpu["uuid"] == protocol["execution"]["gpu_by_arm"][arm]
          and 0 <= gpu["used_MiB"] <= 256 and gpu["gpu_utilization_percent"] == 0, "idle_gpu_allocation")
    for process in gpu.get("allowed_idle_display_processes", []):
        memory = process.get("used_memory", "")
        check(process.get("type") == "G" and process.get("process_name") == "/usr/lib/xorg/Xorg"
              and memory.endswith(" MiB") and 0 <= int(memory[:-4]) <= 8, "idle_display_process")
    config = env["effective_generation_config"]
    check(config_equivalent(receipt["effective_generation_config"], config), "effective_config_cpu_gpu_match")
    expected = {"max_new_tokens": 768, "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                "use_cache": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
                "repetition_penalty": 1.0, "length_penalty": 1.0}
    check(all(config.get(k) == v for k, v in expected.items()), "fixed_greedy_config")
    model_attestation(receipt, protocol, checks, arm)


def audit_arm(folder, arm, protocol, protocol_sha, tokenizer, checks):
    before = len(checks.issues)
    source_rows = evaluate.load_inputs(protocol, arm)
    # This validates the complete same-length-sensitive hash, never a text-only approximation.
    evaluate.tokenize_rows(tokenizer, source_rows)
    for row in source_rows:
        checks.check(hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()
                     == protocol["system_prompt_sha256"], "system_prompt_identity", arm=arm)
    names = {"runtime.json", "cpu_preflight.json", "environment.json", "child_started.json", "worker_result.json", "worker.log"}
    names.update(row["question_id"] + suffix for row in source_rows for suffix in (".json", ".started.json"))
    # Remote model symlinks need not be transferred; their attested targets are
    # checked below without following them or downloading the base checkpoint.
    checks.check({p.name for p in folder.iterdir()} - {"verified_model"} == names, "arm_artifact_roster", arm=arm)
    report, receipt, env, child, worker = [checks.read(folder / name) for name in
        ("runtime.json", "cpu_preflight.json", "environment.json", "child_started.json", "worker_result.json")]
    if (folder / "worker.log").is_file():
        checks.files[str(folder / "worker.log")] = runtime.digest(folder / "worker.log")
    runtime.require(all(type(value) is dict for value in (report, receipt, env, child, worker)), "Incomplete arm metadata")
    audit_runtime(report, receipt, env, child, worker, protocol, protocol_sha, arm, checks)
    rows = []
    for source in source_rows:
        saved = checks.read(folder / f"{source['question_id']}.json")
        marker = checks.read(folder / f"{source['question_id']}.started.json")
        rows.append(audit_row(saved, source, marker, env, tokenizer, checks))
    starts = [r.get("worker_start_seconds") for r in rows]
    durations = [r.get("seconds") for r in rows]
    if all(number(v) for v in starts + durations):
        checks.check(all(starts[i + 1] >= starts[i] + durations[i] - 0.001 for i in range(len(rows) - 1)),
                     "sequential_fixed_question_order", arm=arm)
        checks.check(sum(durations) <= report["gpu_seconds"] + 0.001
                     and starts[-1] + durations[-1] <= report["gpu_seconds"] + 0.001,
                     "child_lifetime_contains_all_generation", arm=arm)
    totals = {"question_count": len(rows), "mechanical_valid_rows": sum(r["mechanical_passed"] for r in rows),
              "strict_format_valid_count": sum(r.get("strict_format_valid") is True for r in rows),
              "strict_format_invalid_count": sum(r.get("strict_format_valid") is False for r in rows),
              "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
              "generated_tokens": sum(r.get("generated_tokens", 0) for r in rows),
              "stop_reasons": dict(Counter(r.get("stop_reason", "invalid") for r in rows)),
              "gpu_seconds": report["gpu_seconds"], "gpu_hours": report["gpu_hours"],
              "generation_seconds": sum(r.get("seconds", 0) for r in rows if number(r.get("seconds"))),
              "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0),
              "peak_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in rows), default=0)}
    return {"passed": len(checks.issues) == before, "totals": totals, "rows": rows,
            "execution_report": report, "execution_report_path": str(folder / "runtime.json"),
            "execution_report_sha256": runtime.digest(folder / "runtime.json"),
            "model_attestation": receipt["model_files"], "generation_config": env["effective_generation_config"],
            "prepared_generation_config": receipt["effective_generation_config"],
            "identity": env["identity"], "hardware": {k: env["gpu"][k] for k in ("name", "driver_version", "total_MiB")}}


def public_summary(private):
    """Explicitly keep answer text, per-question details and prompts private."""
    return {"schema": "radar_coverage_execution_audit_v1", "passed": private["passed"],
            "protocol_sha256": private["protocol_sha256"], "audit_code_sha256": private["audit_code_sha256"],
            "semantic_scoring_performed": False, "reference_files_opened": False,
            "local_weight_rehash_performed": False,
            "model_proof_basis": "Frozen remote CPU hash attestations, fingerprint and cross-arm equality; remote weights were not reread locally.",
            "generation_config_default_normalizations": DEFAULT_NORMALIZATIONS,
            "arms": {arm: {"passed": value["passed"], **{k: value["totals"][k] for k in PUBLIC_TOTAL_KEYS if k in value["totals"]},
                       "execution_report": {"path": str(Path(value["execution_report_path"]).relative_to(ROOT)),
                                            "sha256": value["execution_report_sha256"]}}
                     for arm, value in private["arms"].items()},
            "total_gpu_seconds": private["total_gpu_seconds"], "total_gpu_hours": private["total_gpu_seconds"] / 3600,
            "project_gpu_hours_after": private["project_gpu_hours_after"],
            "issue_count": len(private["issues"]), "issue_codes": dict(Counter(i["code"] for i in private["issues"])),
            "claim_limit": "Mechanical integrity, token/decoding identity and recorded resource accounting only. Format validity is not answer correctness, evidence support or citation support."}


def audit(protocol_path, protocol_sha, tokenizer_path, private_output, public_output):
    private_output, public_output = Path(private_output).resolve(), Path(public_output).resolve()
    runtime.require(private_output.is_relative_to(ROOT / "data") and not private_output.exists(), "Use a new private directory under data")
    runtime.require(public_output.is_relative_to(ROOT / "results") and not public_output.exists(), "Use a new public file under results")
    protocol = evaluate.load_protocol(protocol_path, protocol_sha)
    tokenizer, token_files = runtime.tokenizer_precheck(tokenizer_path, protocol["model_identity"])
    checks = Checks()
    arms = {}
    for arm in evaluate.ARMS:
        folder = evaluate.path_in_root(protocol["execution"]["output_root"]) / arm
        try:
            arms[arm] = audit_arm(folder, arm, protocol, protocol_sha, tokenizer, checks)
        except (ValueError, TypeError, KeyError, OSError, IndexError):
            checks.check(False, "arm_audit_incomplete", arm=arm)
    if len(arms) == 3:
        baseline = arms["raw"]
        for arm in ("flat", "bound"):
            for key in ("model_attestation", "generation_config", "prepared_generation_config", "identity", "hardware"):
                checks.check(arms[arm][key] == baseline[key], "cross_arm_" + key, arm=arm)
    for path, original_sha in checks.files.items():
        checks.check(runtime.digest(path) == original_sha, "artifact_changed_during_audit", file=path)
    total = sum(value["totals"]["gpu_seconds"] for value in arms.values())
    project_after = protocol["execution"]["project_gpu_hours_before"] + total / 3600
    checks.check(len(arms) == 3 and total <= 7200 and project_after <= 72, "total_gpu_budget")
    private = {"schema": "radar_coverage_execution_audit_private_v1", "passed": not checks.issues,
               "protocol_sha256": protocol_sha, "audit_code_sha256": runtime.digest(Path(__file__)),
               "audit_python": platform.python_version(), "arms": arms, "issues": checks.issues,
               "total_gpu_seconds": total, "project_gpu_hours_after": project_after,
               "artifact_sha256": checks.files,
               "input_bindings": {key: protocol[key] for key in ("inputs", "token_identity", "code_sha256", "old_dev_pass")},
               "local_tokenizer_files": token_files}
    summary = public_summary(private)
    private_output.mkdir(parents=True)
    runtime.write_new(private_output / "audit.json", private)
    summary["private_audit"] = {"path": str((private_output / "audit.json").relative_to(ROOT)),
                                "sha256": runtime.digest(private_output / "audit.json")}
    public_output.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_new(public_output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "tokenizer", "private-output", "public-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.protocol, args.protocol_sha256, args.tokenizer, args.private_output, args.public_output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
