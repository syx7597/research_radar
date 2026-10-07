"""Mechanical audit tests use only synthetic prompts, text, tokens and metadata."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import coverage_execution_audit as audit


TEXT = '{"answer_text":"synthetic fixture","citations":[{"chunk_id":"fixture:p1"}]}'


class Tokenizer:
    def __init__(self, text=TEXT):
        self.text = text
    def __len__(self):
        return 10
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        return self.text
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        return [1, 2, 3]


def fixture():
    qids = [f"fixture-{i}" for i in range(96)]
    config = {"max_new_tokens": 768, "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
              "use_cache": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
              "repetition_penalty": 1.0, "length_penalty": 1.0, "eos_token_id": [9]}
    files = {}
    weights = [{"file": f"model-{n}.safetensors", "bytes": 8, "sha256": str(n) * 64} for n in (1, 2)]
    model_hashes = {w["file"]: w["sha256"] for w in weights}
    model_hashes.update({"config.json": "c" * 64, "generation_config.json": "g" * 64,
                         "model.safetensors.index.json": "i" * 64})
    token_hashes = {"tokenizer.json": "t" * 64, "tokenizer_config.json": "u" * 64, "config.json": "c" * 64}
    view = {name: "/remote/model/" + name for name in model_hashes}
    for folder, hashes in (("/remote/model/", model_hashes), ("/remote/repository/data/tokenizer/", token_hashes)):
        for name, sha in hashes.items():
            files[folder + name] = {"sha256": sha, "fingerprint": {"bytes": 8, "mtime_ns": 1, "ctime_ns": 1, "device": 1, "inode": 2}}
    p = {"question_ids": qids, "model_path": "/remote/model", "tokenizer_path": "data/tokenizer",
         "frozen_at_utc": "2020-01-01T00:00:00+00:00", "system_prompt_sha256": audit.hashlib.sha256(b"fixture").hexdigest(),
         "model_identity": {"expected_base_weights": {"files": weights}, "expected_config_sha256": "c" * 64,
            "expected_generation_config_sha256": "g" * 64,
            "expected_tokenizer_sha256": {k: v for k, v in token_hashes.items() if k != "config.json"}},
         "runtime_environment": {"python": "fixture", "torch": "fixture", "transformers": "fixture", "tokenizers": "fixture", "cuda": "fixture"},
         "execution": {"project_gpu_hours_before": 24.0, "gpu_by_arm": {"raw": "GPU-fixture"}}}
    receipt = {"model_files": files, "model_view_files": view, "effective_generation_config": config,
               "parent_pid": 10, "protocol_sha256": "sha", "arm": "raw", "all_96_prompt_ids_verified": True}
    env = {**p["runtime_environment"], "effective_generation_config": config,
           "identity": {"real_user": "syx", "effective_user": "syx", "real_uid": 1000, "effective_uid": 1000},
           "gpu": {"uuid": "GPU-fixture", "used_MiB": 0, "gpu_utilization_percent": 0,
                   "total_MiB": 16384, "name": "fixture", "driver_version": "fixture", "allowed_idle_display_processes": []}}
    report = {"schema": "radar_coverage_arm_execution_v1", "arm": "raw", "protocol_sha256": "sha",
        "status": "completed", "returncode": 0, "error": None, "process_started": True, "process_reaped": True,
        "GPU_hours_is_lower_bound": False, "within_arm_budget": True,
        "rows": [{"question_id": q, "status": "generated"} for q in qids], "question_denominator": 96,
        "generated_count": 96, "not_run_question_ids": [], "new_training_runs": 0, "automatic_retries": 0,
        "semantic_scoring_performed": False, "gpu_seconds": 100.0, "gpu_hours": 100 / 3600,
        "accounting_basis": audit.ACCOUNTING, "formal_round_reserved_gpu_seconds": 7200,
        "project_gpu_hours_before": 24.0, "project_gpu_hours_budget": 72}
    child = {"pid": 20, "started_at_unix": 1700000000, "gpu_count": 1, "max_seconds": 2400}
    worker = {"status": "completed", "question_count": 96}
    sources, outputs, markers = [], [], []
    for i, qid in enumerate(qids):
        source = {"question_id": qid, "arm": "raw", "input_tokens": 3, "truncated": False,
                  "token_ids_sha256": audit.evaluate.token_hash([1, 2, 3]),
                  "messages": [{"role": "system", "content": "fixture"}, {"role": "user", "content": json.dumps(
                      {"question": "fixture", "evidence": {"chunks": [{"chunk_id": "fixture:p1"}]}})}]}
        saved = {"question_id": qid, "arm": "raw", "status": "generated", "input_tokens": 3,
                 "prompt_ids_sha256": source["token_ids_sha256"], "generated_ids": [1, 2, 9], "generated_text": TEXT,
                 "generated_tokens": 3, "stop_reason": "eos", "seconds": 0.01,
                 "peak_allocated_bytes": 16, "peak_reserved_bytes": 32, "truncated": False,
                 "format_diagnostic": {"valid": True, "error": None}}
        sources.append(source); outputs.append(saved)
        markers.append({"question_id": qid, "arm": "raw", "worker_seconds": 1 + i * 0.1})
    return p, receipt, env, report, child, worker, sources, outputs, markers


class ExecutionAudit(unittest.TestCase):
    def test_full_synthetic_arm_and_token_decode_replay(self):
        p, receipt, env, report, child, worker, sources, outputs, markers = fixture()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for name, value in (("runtime.json", report), ("cpu_preflight.json", receipt), ("environment.json", env),
                                ("child_started.json", child), ("worker_result.json", worker)):
                (folder / name).write_text(json.dumps(value))
            (folder / "worker.log").write_text("synthetic log")
            for source, saved, marker in zip(sources, outputs, markers):
                (folder / f"{source['question_id']}.json").write_text(json.dumps(saved))
                (folder / f"{source['question_id']}.started.json").write_text(json.dumps(marker))
            checks = audit.Checks()
            with patch.object(audit.evaluate, "load_inputs", return_value=sources):
                result = audit.audit_arm(folder, "raw", p, "sha", Tokenizer(), checks)
            self.assertTrue(result["passed"], checks.issues)
            self.assertEqual(result["totals"]["generated_tokens"], 288)
            self.assertEqual(len(checks.files), 198)
            markers[1]["worker_seconds"] = 0
            (folder / "fixture-1.started.json").write_text(json.dumps(markers[1]))
            checks = audit.Checks()
            with patch.object(audit.evaluate, "load_inputs", return_value=sources):
                audit.audit_arm(folder, "raw", p, "sha", Tokenizer(), checks)
            self.assertIn("sequential_fixed_question_order", [v["code"] for v in checks.issues])

    def test_wrong_decode_ids_count_prompt_and_truncation_rejected(self):
        _, _, env, _, _, _, sources, outputs, markers = fixture()
        changes = [{"generated_text": "changed"}, {"generated_ids": [True, 2, 9]}, {"generated_tokens": 2},
                   {"prompt_ids_sha256": "changed"}, {"truncated": True}, {"generated_ids": [1, 9, 2]},
                   {"generated_ids": [1, 2, 3], "stop_reason": "generation_budget"}]
        for change in changes:
            checks = audit.Checks()
            result = audit.audit_row({**outputs[0], **change}, sources[0], markers[0], env, Tokenizer(), checks)
            self.assertFalse(result["mechanical_passed"], change)

    def test_invalid_format_is_not_a_mechanical_or_semantic_failure(self):
        _, _, env, _, _, _, sources, outputs, markers = fixture()
        saved = outputs[0]
        saved["generated_text"] = "not JSON"
        saved["format_diagnostic"] = audit.evaluate.format_diagnostic("not JSON", sources[0])
        result = audit.audit_row(saved, sources[0], markers[0], env, Tokenizer("not JSON"), audit.Checks())
        self.assertTrue(result["mechanical_passed"])
        self.assertFalse(result["strict_format_valid"])

    def test_format_diagnostic_drift_rejected(self):
        _, _, env, _, _, _, sources, outputs, markers = fixture()
        outputs[0]["format_diagnostic"] = {"valid": False, "error": "invented"}
        checks = audit.Checks()
        audit.audit_row(outputs[0], sources[0], markers[0], env, Tokenizer(), checks)
        self.assertIn("format_diagnostic_replay", [r["code"] for r in checks.issues])

    def test_only_explicit_default_normalizations_are_allowed(self):
        declared = {key: None for key in audit.DEFAULT_NORMALIZATIONS}
        prepared = dict(audit.DEFAULT_NORMALIZATIONS)
        self.assertEqual(len(prepared), 18)
        self.assertTrue(audit.config_equivalent(prepared, declared))
        self.assertFalse(audit.config_equivalent({**prepared, "max_time": 1}, {**declared, "max_time": None}))
        self.assertFalse(audit.config_equivalent({**prepared, "min_length": 7}, declared))
        self.assertFalse(audit.config_equivalent({**prepared, "min_length": False}, declared))

    def test_user_environment_budget_and_weight_attestation_tampering(self):
        for target in ("user", "environment", "budget", "weight", "config", "unreaped"):
            p, receipt, env, report, child, worker, *_ = fixture()
            if target == "user": env["identity"]["effective_user"] = "admin"
            if target == "environment": env["torch"] = "changed"
            if target == "budget": report["gpu_hours"] = 0
            if target == "weight": receipt["model_files"]["/remote/model/model-1.safetensors"]["sha256"] = "changed"
            if target == "config": env["effective_generation_config"]["do_sample"] = True
            if target == "unreaped": report["process_reaped"] = False
            checks = audit.Checks()
            audit.audit_runtime(report, receipt, env, child, worker, p, "sha", "raw", checks)
            self.assertTrue(checks.issues, target)

    def test_partial_json_is_retained_as_an_issue_without_text_leak(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "partial.json"
            path.write_text('{"generated_text":"PRIVATE_SENTINEL')
            checks = audit.Checks()
            self.assertIsNone(checks.read(path))
            self.assertIn(str(path), checks.files)
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(checks.issues))

    def test_public_summary_is_an_explicit_nontext_allowlist(self):
        arm = {"passed": True, "totals": {"question_count": 96, "generated_tokens": 300, "accidental_text": "PRIVATE_TOTAL"},
               "rows": [{"generated_text": "PRIVATE_ANSWER", "question": "PRIVATE_QUESTION"}],
               "execution_report": {"secret": "PRIVATE_REPORT"},
               "execution_report_path": str(audit.ROOT / "data/run/raw/runtime.json"), "execution_report_sha256": "sha"}
        private = {"passed": False, "protocol_sha256": "sha", "audit_code_sha256": "sha", "arms": {"raw": arm},
                   "total_gpu_seconds": 100, "project_gpu_hours_after": 24.1,
                   "issues": [{"code": "test", "text": "PRIVATE_ERROR"}]}
        summary = audit.public_summary(private)
        self.assertNotIn("PRIVATE_", json.dumps(summary))
        self.assertFalse(summary["semantic_scoring_performed"])
        self.assertFalse(summary["local_weight_rehash_performed"])


if __name__ == "__main__":
    unittest.main()
