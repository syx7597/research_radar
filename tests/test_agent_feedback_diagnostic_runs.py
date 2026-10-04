"""CPU mocks for the one-round, full-dev, four-model feedback diagnostic."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback import feedback_diagnostic_runs as runs


class FeedbackDiagnosticControllerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, previous)
        self.calls, self.audited = [], 0
        self.bad_audit, self.bad_coverage, self.prefix_mismatch = False, False, False
        self.bad_sidecar, self.bad_job = None, None
        self.questions = [{"id": i, "question": "Q"} for i in range(500)]
        self.jsonl(runs.DEV_QUESTIONS, self.questions)
        hashes = {str(runs.DEV_QUESTIONS): runs.digest(runs.DEV_QUESTIONS)}
        for path in (runs.CODE / "inference.py", runs.CODE / "environment.py",
                     runs.CODE / "feedback_mask.py", runs.KB):
            self.write(path, {"fixture": str(path)})
        for _, _, model, normal, _, _ in runs.RUNS:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                self.write(runs.ROOT / model / "model" / filename, {"fixture": model})
            path = runs.ROOT / f"{normal}.jsonl"
            self.jsonl(path, [{"id": i} for i in range(500)])
            hashes[str(path)] = runs.digest(path)
        self.snapshot = {"inputs_sha256": hashes, "versions": {}, "manifest": {}, "dev_ids": list(range(500))}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def fake_job(self, name, gpu, command, hours):
        self.calls.append((name, gpu, command, hours))
        model = next(row[2] for row in runs.RUNS if row[4] == name)
        count = 499 if self.bad_coverage else 500
        self.jsonl(runs.ROOT / f"{name}.jsonl", [{"id": i} for i in range(count)])
        self.write(runs.ROOT / f"{name}.runtime.json", {"status": "completed", "source_questions": 500,
            "selected_questions": count, "totals": {"questions": count},
            "config": {**runs.inference_config(model, name), "feedback_mode": runs.MODE}})
        path = runs.ROOT / f"{name}.jsonl"
        sources = {"inference": runs.CODE / "inference.py", "environment": runs.CODE / "environment.py",
                   "renderer": runs.CODE / "feedback_mask.py", "questions": runs.DEV_QUESTIONS, "kb": runs.KB,
                   "adapter_config": runs.ROOT / model / "model/adapter_config.json",
                   "adapter_weights": runs.ROOT / model / "model/adapter_model.safetensors"}
        metadata = {"feedback_mode": runs.MODE, "masked_fields": ["error", "detail"],
                    "replacement": {"error": "ExecutionError", "detail": "execution_failed"}, "questions": 500,
                    "config": {**runs.inference_config(model, name), "feedback_mode": runs.MODE},
                    "inputs_sha256": {key: runs.digest(source) for key, source in sources.items()}}
        self.write(path.with_suffix(".feedback_started.json"), metadata)
        metadata.update(status="completed", predictions_sha256=runs.digest(path),
                        runtime_sha256=runs.digest(path.with_suffix(".runtime.json")),
                        rendering_counts={"observations": 500, "failed_observations": 5, "masked_observations": 5})
        if self.bad_sidecar:
            metadata[self.bad_sidecar] = "changed"
        self.write(path.with_suffix(".feedback.json"), metadata)
        job = {"name": name, "status": "completed", "returncode": 0, "user": "syx", "uid": 1001,
               "gpus": [gpu], "command": [sys.executable, "-u", "-m", *command],
               "started_at": 100, "finished_at": 110, "gpu_seconds": 10}
        if self.bad_job:
            job[self.bad_job] = "changed"
        self.write(runs.ROOT / "jobs" / f"{name}.json", job)

    def fake_validator(self, normal, masked, questions, *, executor):
        self.assertEqual(len(normal), 500)
        self.assertEqual(len(masked), 500)
        self.assertEqual(questions, self.questions)
        self.audited += 1
        return {"audit_passed": not self.bad_audit, "mismatches": ["bad"] if self.bad_audit else [],
                "coverage": 1.0, "pre_intervention_equivalence_passed": not self.prefix_mismatch,
                "pre_intervention_divergence_ids": [0] if self.prefix_mismatch else []}

    def fake_cpu(self, command, **kwargs):
        self.assertEqual(self.audited, 4)
        self.assertTrue(runs.AUDIT.exists())
        if "--output" in command:
            self.write(Path(command[command.index("--output") + 1]), {"comparison": "fixture"})

    def run_mock(self, gpu_error=None, budget_error=None):
        def idle():
            self.assertFalse(runs.MARKER.exists())
            if gpu_error:
                raise gpu_error
        with patch.object(runs, "require_syx"), patch.object(runs, "validate_inputs", return_value=self.snapshot), \
                patch.object(runs, "require_runtime"), patch.object(runs, "require_idle_gpus", side_effect=idle), \
                patch.object(runs, "require_budget", side_effect=budget_error, return_value=13.883) as budget, \
                patch.object(runs, "run_job", side_effect=self.fake_job), patch.object(runs, "KoPLExecutor"), \
                patch.object(runs.subprocess, "run", side_effect=self.fake_cpu) as cpu, \
                patch.dict(sys.modules, {"experiments.agent_feedback.feedback_diagnostic_analysis":
                    SimpleNamespace(validate_diagnostic_predictions=self.fake_validator)}), \
                patch.dict(os.environ, {"PYTHONHASHSEED": "20261003"}), \
                patch("sys.argv", ["feedback_diagnostic_runs"]):
            runs.main()
        return budget, cpu

    def test_four_frozen_models_full_dev_only_and_all_audits_precede_scoring(self):
        budget, cpu = self.run_mock()
        budget.assert_called_once_with(4)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual({row[1] for row in self.calls}, {0, 1, 2, 3})
        self.assertEqual({row[3] for row in self.calls}, {1})
        for name, _, command, _ in self.calls:
            model = next(row[2] for row in runs.RUNS if row[4] == name)
            for key, value in (("--adapter", str(runs.ROOT / model / "model")),
                               ("--questions", str(runs.DEV_QUESTIONS)), ("--limit", "0"),
                               ("--seed", "20261003"), ("--feedback-mode", "generic_failure_v1")):
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertNotIn("holdout", " ".join(command))
        self.assertEqual(cpu.call_count, 8)
        protocol, summary = runs.read(runs.PROTOCOL), runs.read(runs.SUMMARY)
        self.assertEqual(len(protocol["models"]), 4)
        self.assertFalse(summary["model_selection"])
        self.assertFalse(summary["training"])
        self.assertFalse(summary["held_out_evaluation"])
        self.assertFalse(summary["automatic_next_round"])
        runs.require_hashes(self.snapshot["inputs_sha256"])
        with self.assertRaises(FileExistsError):
            runs.require_new_outputs()

    def test_incomplete_output_is_never_scored(self):
        self.bad_coverage = True
        with self.assertRaisesRegex(ValueError, "complete frozen development"):
            self.run_mock()
        self.assertEqual(self.audited, 0)
        self.assertFalse(runs.AUDIT.exists())
        self.assertTrue(runs.MARKER.exists())

    def test_failed_execution_visibility_audit_is_never_scored(self):
        self.bad_audit = True
        with self.assertRaisesRegex(ValueError, "visible-message audit failed"):
            self.run_mock()
        self.assertFalse(runs.AUDIT.exists())
        self.assertFalse(runs.SUMMARY.exists())

    def test_prefix_mismatch_is_retained_and_limits_interpretation(self):
        self.prefix_mismatch = True
        _, cpu = self.run_mock()
        self.assertEqual(cpu.call_count, 8)
        summary = runs.read(runs.SUMMARY)
        self.assertFalse(summary["pre_intervention_equivalence_passed"])
        self.assertEqual(summary["mechanism_interpretation"], "needs_numerical_batching_review")

    def test_gpu_or_budget_failure_leaves_no_start_marker(self):
        for kwargs in ({"gpu_error": RuntimeError("busy")}, {"budget_error": RuntimeError("budget")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.run_mock(**kwargs)
            self.assertFalse(runs.MARKER.exists())
            self.assertFalse(runs.PROTOCOL.exists())
        self.assertFalse(self.calls)

    def test_incomplete_review_rejected_before_gpu(self):
        self.write(runs.REVIEW, {"gate": {"audit_passed": False}, "issues": {}})
        with patch.object(runs, "require_idle_gpus") as gpu, self.assertRaisesRegex(ValueError, "audit must pass"):
            runs.validate_inputs()
        gpu.assert_not_called()

    def test_source_mode_and_output_hashes_are_required_before_audit_or_score(self):
        self.bad_sidecar = "predictions_sha256"
        with self.assertRaisesRegex(ValueError, "does not bind"):
            self.run_mock()
        self.assertEqual(self.audited, 0)
        _, _, model, _, name, _ = runs.RUNS[0]
        metadata_path = runs.ROOT / f"{name}.feedback.json"
        metadata = runs.read(metadata_path)
        metadata["predictions_sha256"] = runs.digest(runs.ROOT / f"{name}.jsonl")
        config = {**runs.inference_config(model, name), "feedback_mode": runs.MODE}
        for field in ("feedback_mode", "inputs_sha256", "config"):
            with self.subTest(field=field):
                self.write(metadata_path, {**metadata, field: "changed"})
                with self.assertRaisesRegex(ValueError, "source, mode or configuration"):
                    runs.validate_feedback(model, name, config)

    def test_job_identity_completion_command_and_gpu_are_required(self):
        self.bad_job = "returncode"
        with self.assertRaisesRegex(ValueError, "job ledger"):
            self.run_mock()
        self.assertEqual(self.audited, 0)
        _, _, model, _, name, gpu = runs.RUNS[0]
        path = runs.ROOT / "jobs" / f"{name}.json"
        record = {**runs.read(path), "returncode": 0}
        for field in ("status", "user", "uid", "gpus", "command"):
            with self.subTest(field=field):
                self.write(path, {**record, field: "changed"})
                with self.assertRaisesRegex(ValueError, "job ledger"):
                    runs.validate_job(model, name, gpu)


if __name__ == "__main__":
    unittest.main()
