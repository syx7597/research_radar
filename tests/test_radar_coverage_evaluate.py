"""Synthetic CPU tests for one-round generation isolation and failure handling."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.radar_domain import coverage_evaluate as evaluate


def fixtures(count=2):
    rows, identities = [], []
    for i in range(count):
        for arm in evaluate.ARMS:
            row = {"question_id": f"synthetic-{i}", "arm": arm, "input_tokens": 3, "truncated": False,
                   "messages": [{"role": "system", "content": "fixture"}, {"role": "user", "content": json.dumps(
                       {"question": "fixture", "evidence": {"chunks": [{"chunk_id": "fixture:p1"}]}})}]}
            rows.append(row)
            identities.append({"question_id": row["question_id"], "arm": arm,
                               "token_count": 3, "token_ids_sha256": evaluate.token_hash([1, 2, 3])})
    return rows, identities


class EvaluationExecution(unittest.TestCase):
    def test_protocol_requires_passed_old_gate_and_reserves_project_budget(self):
        protocol = {"schema": "radar_coverage_evaluation_protocol_v1", "status": "frozen",
            "question_ids": [f"fixture-{i}" for i in range(96)],
            "inputs": {"path": evaluate.INPUT_PATH, "sha256": evaluate.INPUT_SHA},
            "inference": evaluate.INFERENCE, "model_identity": {"adapter": None},
            "execution": {"max_gpu_seconds_per_arm": 2400, "max_total_gpu_seconds": 7200,
                "automatic_retries": 0, "training_runs": 0, "project_gpu_hours_budget": 72,
                "project_gpu_hours_before": 70.1, "gpu_by_arm": {a: f"GPU-{a}" for a in evaluate.ARMS}},
            "code_sha256": {name: "sha" for name in evaluate.REQUIRED_CODE},
            "token_identity": {"path": "fixture/tokens.json", "sha256": "sha"},
            "old_dev_pass": {"path": "fixture/old.json", "sha256": "sha"}}
        with patch.object(evaluate, "digest", return_value="sha"), \
             patch.object(Path, "read_text", return_value=json.dumps(protocol)):
            with self.assertRaisesRegex(ValueError, "reserved two GPU hours"):
                evaluate.load_protocol(Path("protocol.json"), "sha")
        protocol["execution"]["project_gpu_hours_before"] = 24.05
        def digest(path):
            return evaluate.INPUT_SHA if str(path).endswith("inputs.jsonl") else "sha"
        old = {"status": "failed", "format_passed_cases": []}
        with patch.object(evaluate, "digest", side_effect=digest), \
             patch.object(Path, "read_text", side_effect=[json.dumps(protocol), json.dumps(old)]):
            with self.assertRaisesRegex(ValueError, "Old-dev runtime gate"):
                evaluate.load_protocol(Path("protocol.json"), "sha")

    def test_selection_keeps_question_order_and_all_arms_share_ids(self):
        rows, identities = fixtures()
        for arm in evaluate.ARMS:
            selected = evaluate.select_inputs(rows, identities, arm, count=2)
            self.assertEqual([r["question_id"] for r in selected], ["synthetic-0", "synthetic-1"])
            self.assertTrue(all(r["arm"] == arm for r in selected))

    def test_reference_fields_duplicates_and_reordering_are_rejected(self):
        rows, identities = fixtures()
        bad = copy.deepcopy(rows)
        bad[0]["reference_answer"] = "must not enter generator"
        with self.assertRaises(ValueError):
            evaluate.select_inputs(bad, identities, "raw", count=2)
        bad = copy.deepcopy(rows)
        bad[3:6] = bad[:3]
        with self.assertRaises(ValueError):
            evaluate.select_inputs(bad, identities, "raw", count=2)
        bad = copy.deepcopy(rows)
        bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaises(ValueError):
            evaluate.select_inputs(bad, identities, "raw", count=2)

    def test_same_length_different_token_ids_rejected_and_no_truncation(self):
        rows, identities = fixtures()
        selected = evaluate.select_inputs(rows, identities, "flat", count=2)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.assertEqual(len(evaluate.tokenize_rows(tokenizer, selected)), 2)
        self.assertIs(tokenizer.apply_chat_template.call_args.kwargs["truncation"], False)
        tokenizer.apply_chat_template.return_value = [1, 9, 3]
        with self.assertRaisesRegex(ValueError, "Prompt IDs/count"):
            evaluate.tokenize_rows(tokenizer, selected)

    def test_format_diagnostic_never_discards_original_or_scores_truth(self):
        row = fixtures()[0][0]
        good = '{"answer_text":"unverified fixture","citations":[{"chunk_id":"fixture:p1"}]}'
        self.assertTrue(evaluate.format_diagnostic(good, row)["valid"])
        self.assertFalse(evaluate.format_diagnostic("not JSON", row)["valid"])
        self.assertFalse(evaluate.format_diagnostic(good.replace("fixture:p1", "unknown:p1"), row)["valid"])
        for evidence in ([{"citation": {"chunk_id": "fixture:p1"}}],
                         [{"subjects": ["x"], "groups": [{"records": [{"citation": {"chunk_id": "fixture:p1"}}]}]}]):
            row["messages"][1]["content"] = json.dumps({"evidence": evidence})
            self.assertTrue(evaluate.format_diagnostic(good, row)["valid"])

    def test_partial_or_failed_outputs_remain_in_fixed_denominator(self):
        rows, identities = fixtures(4)
        selected = evaluate.select_inputs(rows, identities, "bound", count=4)
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            (output / "synthetic-0.json").write_text(json.dumps({"question_id": "synthetic-0", "arm": "bound", "status": "generated", "format_diagnostic": {"valid": False}}))
            (output / "synthetic-1.json").write_text('{"question_id":')
            (output / "synthetic-2.started.json").write_text("{}")
            result = evaluate.inventory(output, selected)
        self.assertEqual(len(result), 4)
        self.assertEqual([r["status"] for r in result], ["generated", "incomplete_output", "failed_or_interrupted", "not_run"])

    def test_long_watchdog_kills_only_own_cpu_session(self):
        command = [sys.executable, "-B", "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)"]
        with tempfile.TemporaryDirectory() as d:
            result = evaluate.watch(command, dict(os.environ), Path(d), seconds=0.8, grace=0.1)
        self.assertEqual(result["status"], "budget_stopped")
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertTrue(result["process_reaped"])
        self.assertFalse(result["GPU_hours_is_lower_bound"])
        self.assertLess(result["gpu_seconds"], 1.5)

    def test_unreaped_interrupt_preserves_budget_uncertainty(self):
        timeout = subprocess.TimeoutExpired("fake", 1)
        process = Mock(pid=123456, returncode=None)
        process.wait.side_effect = [InterruptedError("stopped"), timeout]
        with tempfile.TemporaryDirectory() as d, patch.object(evaluate.subprocess, "Popen", return_value=process), \
             patch.object(evaluate.runtime, "terminate_owned_group") as kill:
            result = evaluate.watch(["fake"], {}, Path(d), seconds=1, grace=0.1)
        self.assertEqual(result["status"], "cleanup_unverified")
        self.assertTrue(result["GPU_hours_is_lower_bound"])
        self.assertFalse(result["within_arm_budget"])
        self.assertTrue(all(call.args[0] is process for call in kill.call_args_list))

    def test_precheck_failure_persists_all_unrun_questions_and_slot_is_exclusive(self):
        rows, identities = fixtures()
        selected = evaluate.select_inputs(rows, identities, "raw", count=2)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            protocol = {"question_ids": [r["question_id"] for r in selected],
                        "execution": {"output_root": "evaluation", "project_gpu_hours_before": 24.05}, "runtime_environment": {"expected": "different"}}
            with patch.object(evaluate, "ROOT", root), patch.object(evaluate.runtime, "identity"), \
                 patch.object(evaluate, "load_protocol", return_value=protocol), patch.object(evaluate, "load_inputs", return_value=selected), \
                 patch.object(evaluate, "environment", return_value={}), patch.object(evaluate, "watch") as watch:
                output = root / "evaluation/raw"
                result = evaluate.execute(Path("unused"), "sha", "raw", "GPU-fixture", output)
                watch.assert_not_called()
                self.assertEqual(result["question_denominator"], 96)
                self.assertEqual(result["not_run_question_ids"], ["synthetic-0", "synthetic-1"])
                self.assertEqual(result["gpu_seconds"], 0)
                self.assertTrue((output / "runtime.json").exists())
                with self.assertRaises(FileExistsError):
                    evaluate.execute(Path("unused"), "sha", "raw", "GPU-fixture", output)
                with self.assertRaisesRegex(ValueError, "single frozen output slot"):
                    evaluate.execute(Path("unused"), "sha", "raw", "GPU-fixture", root / "retry")

    def test_input_loading_failure_keeps_all_96_protocol_ids(self):
        qids = [f"fixture-{i}" for i in range(96)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            protocol = {"question_ids": qids, "execution": {"output_root": "evaluation", "project_gpu_hours_before": 24.05}}
            with patch.object(evaluate, "ROOT", root), patch.object(evaluate.runtime, "identity"), \
                 patch.object(evaluate, "load_protocol", return_value=protocol), \
                 patch.object(evaluate, "load_inputs", side_effect=ValueError("input loading failed")), \
                 patch.object(evaluate, "watch") as watch:
                result = evaluate.execute(Path("unused"), "sha", "raw", "GPU-fixture", root / "evaluation/raw")
            watch.assert_not_called()
            self.assertEqual(result["not_run_question_ids"], qids)
            self.assertEqual(len(result["rows"]), 96)
            self.assertEqual(result["gpu_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
