"""Isolated CPU fixtures: staged five-model evaluation and deferred gold access."""
import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback import holdout_runs as runs
from experiments.agent_feedback.environment import Episode, call_message, compact, prompt_messages


class HoldoutControllerTest(unittest.TestCase):
    def setUp(self):
        directory, previous = tempfile.TemporaryDirectory(), Path.cwd()
        os.chdir(directory.name)
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, previous)
        self.calls, self.bad_model = [], None
        ids = list(range(2000))
        self.questions = [{"id": ident, "question": "Q"} for ident in ids]
        self.gold = [{"id": ident, "question": "Q", "answer": "yes", "program": []} for ident in ids]
        self.gold_text = "".join(json.dumps(row) + "\n" for row in self.gold)
        self.jsonl(runs.QUESTIONS, self.questions)
        self.protocol = {"version": "fixed_five_model_holdout_v1", "models": runs.run_specs(),
            "statistics": runs.STATISTICS, "inputs_sha256": {}, "versions": {}, "manifest": {},
            "maximum_additional_gpu_hours": 10.5, "global_gpu_hours_cap": 72,
            "training": False, "model_selection": False, "automatic_next_round": False,
            "holdout": {"count": 2000, "ids": ids,
                "questions": {"path": str(runs.QUESTIONS), "sha256": runs.digest(runs.QUESTIONS)},
                "gold": {"path": str(runs.GOLD), "sha256": hashlib.sha256(self.gold_text.encode()).hexdigest()}}}
        self.write(runs.SPLIT, {"splits": {"holdout": self.protocol["holdout"],
                                          "train": {"ids": [3000]}, "dev": {"ids": [4000]}}})
        self.write(runs.PROTOCOL, self.protocol)
        self.actual_digest = runs.digest

    def write(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def no_gold_digest(self, path):
        self.assertNotEqual(Path(path).resolve(), runs.GOLD.resolve(), "Premature gold file access")
        return self.actual_digest(path)

    def fake_job(self, name, gpu, command, hours):
        self.assertFalse(runs.GOLD.exists())
        spec = next(spec for spec in self.protocol["models"] if spec["name"] == name)
        if spec["label"] == "P":
            self.assertEqual(len(self.calls), 4)
        self.calls.append((name, gpu, command, hours))
        count = 1999 if name == self.bad_model else 2000
        rows = [{"id": ident, "prediction": "yes", "calls": 1, "input_tokens": 5,
                 "generated_tokens": 2, "total_tokens": 7, "invalid_calls": 0,
                 "stop_reason": "program_completed" if spec["config"]["program"] else "finished"}
                for ident in range(count)]
        self.jsonl(runs.ROOT / f"{name}.jsonl", rows)
        self.write(runs.ROOT / f"{name}.runtime.json", {"status": "completed", "seconds": 10,
            "source_questions": 2000, "selected_questions": count, "config": spec["config"],
            "totals": {"questions": count, "calls": count, "input_tokens": 5 * count,
                       "generated_tokens": 2 * count, "total_tokens": 7 * count, "invalid_calls": 0}})
        self.write(runs.ROOT / "jobs" / f"{name}.json", {"name": name, "status": "completed", "returncode": 0,
            "user": "syx", "uid": 1001, "gpus": [gpu], "started_at": 100, "finished_at": 110,
            "gpu_seconds": 10, "command": [sys.executable, "-u", "-m", *command]})

    def run_mock(self, idle_error=None, budget_error=None):
        with patch.object(runs, "require_runtime"), \
                patch.object(runs, "require_idle_gpus", side_effect=idle_error), \
                patch.object(runs, "require_budget", return_value=15, side_effect=budget_error) as budget, \
                patch.object(runs, "run_job", side_effect=self.fake_job), \
                patch.object(runs, "digest", side_effect=self.no_gold_digest):
            runs.run()
        return budget

    def fake_score(self, args):
        path = Path(args.predictions)
        self.write(path.with_suffix(".metrics.json"), {"total": 2000})
        self.jsonl(path.with_suffix(".scored.jsonl"), [{"id": ident, "correct": True} for ident in range(2000)])

    def test_freeze_copies_metadata_without_opening_either_holdout_file(self):
        runs.PROTOCOL.unlink()
        runs.QUESTIONS.unlink()
        snapshot = {key: self.protocol[key] for key in ("inputs_sha256", "holdout", "versions", "manifest")}
        with patch.object(runs, "validate_freeze_inputs", return_value=snapshot), \
                patch.object(runs, "read_jsonl", side_effect=AssertionError("No split parsing at freeze")):
            runs.freeze()
        frozen = runs.read(runs.PROTOCOL)
        self.assertEqual(frozen["holdout"], snapshot["holdout"])
        self.assertEqual(frozen["statistics"], runs.STATISTICS)
        self.assertEqual(len(frozen["models"]), 5)
        self.assertFalse(runs.MARKER.exists())
        self.assertFalse(runs.GOLD.exists())
        with self.assertRaises(FileExistsError):
            runs.freeze()

    def test_external_gate_and_no_holdout_input_maps(self):
        self.write(runs.DECISION, {"gate": {"allow_holdout_freeze": False}})
        with self.assertRaisesRegex(ValueError, "has not authorized"):
            runs.validate_freeze_inputs()
        for path in (runs.QUESTIONS, runs.GOLD):
            with patch.object(runs, "require_hashes") as hashing, \
                    self.assertRaisesRegex(ValueError, "deferred split metadata"):
                runs.require_non_holdout_hashes({str(path): "a" * 64})
            hashing.assert_not_called()

    def test_full_generation_has_five_fixed_jobs_and_never_touches_gold(self):
        budget = self.run_mock()
        self.assertEqual([call.args[0] for call in budget.call_args_list], [10.5, .5])
        self.assertEqual(len(self.calls), 5)
        self.assertEqual({call[1] for call in self.calls[:4]}, {0, 1, 2, 3})
        self.assertEqual(sum(call[3] for call in self.calls), 10.5)
        self.assertIn("--program", self.calls[-1][2])
        self.assertTrue(all("--program" not in call[2] for call in self.calls[:4]))
        self.assertTrue(all("--gold" not in call[2] and "train" not in call[2][0] for call in self.calls))
        complete = runs.read(runs.COMPLETE)
        self.assertEqual(len(complete["outputs_sha256"]), 15)
        self.assertFalse(complete["gold_opened"])
        self.assertFalse(runs.ANALYSIS_MARKER.exists())
        self.assertFalse(runs.GOLD.exists())
        with self.assertRaises(FileExistsError):
            runs.run()

    def test_partial_agent_failure_preserves_outputs_without_p_or_gold(self):
        self.bad_model = runs.RUNS[0][2]
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            self.run_mock()
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(runs.MARKER.exists())
        self.assertFalse(runs.COMPLETE.exists())
        with patch.object(runs, "digest", side_effect=self.no_gold_digest), self.assertRaises(FileNotFoundError):
            runs.analyze()

    def test_busy_gpu_or_insufficient_budget_never_creates_marker(self):
        for kwargs in ({"idle_error": RuntimeError("busy")}, {"budget_error": RuntimeError("budget")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.run_mock(**kwargs)
            self.assertFalse(runs.MARKER.exists())
        self.assertFalse(self.calls)

    def test_analysis_requires_bound_complete_all_five_before_reading_gold(self):
        self.run_mock()
        completion = runs.read(runs.COMPLETE)
        altered = {**completion, "outputs_sha256": {**completion["outputs_sha256"], str(runs.GOLD): "a" * 64}}
        self.write(runs.COMPLETE, altered)
        with patch.object(runs, "digest", side_effect=self.no_gold_digest), \
                self.assertRaisesRegex(ValueError, "exactly the five"):
            runs.analyze()
        self.write(runs.COMPLETE, completion)
        prediction = runs.ROOT / f"{runs.RUNS[-1][2]}.jsonl"
        original = prediction.read_text()
        prediction.write_text(original + "\n")
        with patch.object(runs, "digest", side_effect=self.no_gold_digest), \
                self.assertRaisesRegex(ValueError, "changed"):
            runs.analyze()
        prediction.write_text(original)
        runs.GOLD.write_text(self.gold_text)
        with patch.object(runs, "analyze_predictions", return_value={"questions": 2000}) as analyze, \
                patch.object(runs, "KoPLExecutor"), \
                patch.object(runs, "replay_predictions", return_value={"audit_passed": True, "mismatch_count": 0}), \
                patch.object(runs.inference, "score", side_effect=self.fake_score) as score:
            runs.analyze()
        self.assertEqual(set(analyze.call_args.args[0]), {spec["label"] for spec in self.protocol["models"]})
        self.assertEqual(analyze.call_args.kwargs["bootstrap_replicates"], 5000)
        self.assertEqual(score.call_count, 5)
        summary = runs.read(runs.SUMMARY)
        self.assertEqual(len(summary["scoring_outputs_sha256"]), 10)
        self.assertFalse(summary["model_selection"])
        self.assertFalse(summary["automatic_next_round"])
        with self.assertRaises(FileExistsError):
            runs.analyze()

    def test_execution_mismatch_is_saved_and_blocks_gold_access(self):
        self.run_mock()
        with patch.object(runs, "KoPLExecutor"), \
                patch.object(runs, "replay_predictions", return_value={"audit_passed": False, "mismatch_count": 1}), \
                patch.object(runs, "digest", side_effect=self.no_gold_digest), \
                self.assertRaisesRegex(ValueError, "CPU execution replay differs"):
            runs.analyze()
        self.assertTrue(runs.AUDIT.exists())
        self.assertFalse(runs.REPORT.exists())
        self.assertFalse(runs.GOLD.exists())

    def test_actual_agent_replay_detects_observation_and_final_state_changes(self):
        engine = SimpleNamespace(entities={}, concepts={}, FindAll=lambda deps, inputs: ([], None),
                                 Count=lambda deps, inputs: len(deps[0][0]))
        executor = SimpleNamespace(engine=engine)
        episode, messages = Episode(executor), prompt_messages("Q")
        for tool, args in (("step", {"function": "FindAll", "inputs": [], "dependencies": []}),
                           ("step", {"function": "Count", "inputs": [], "dependencies": [0]}),
                           ("finish", {"answer_handle": 1})):
            assistant = call_message(tool, args)
            observation = runs.inference.execute_response(episode, assistant["content"])
            messages.extend([assistant, {"role": "tool", "content": compact(observation)}])
        row = {"id": 1, "messages": messages, "events": episode.events, "prediction": "0",
               "calls": 3, "selected_handle": 1, "invalid_calls": 0, "stop_reason": "finished"}
        question = [{"id": 1, "question": "Q"}]
        self.assertTrue(runs.replay_predictions({"A_20261003": [row]}, question, executor)["audit_passed"])
        for key, value in (("selected_handle", 0), ("prediction", "00"), ("stop_reason", "call_budget")):
            with self.subTest(field=key):
                changed = {**row, key: value}
                report = runs.replay_predictions({"A_20261003": [changed]}, question, executor)
                self.assertEqual(report["mismatch_count"], 1)
        changed = deepcopy(row)
        changed["messages"][3]["content"] = "{}"
        self.assertFalse(runs.replay_predictions({"A_20261003": [changed]}, question, executor)["audit_passed"])

    def test_program_reexecution_is_strict_and_allows_no_generation_context_stop(self):
        prefix = [{"role": "system", "content":
            "Translate the question into a complete executable KoPL program. "
            "Output only function and literal inputs separated by <arg>, with steps separated by <func>. "
            "Dependencies follow the standard branch-stack order.\n" + runs.FUNCTION_HELP},
            {"role": "user", "content": "Q"}]
        result = {"valid": True, "prediction": "0", "program": [{"function": "Count"}]}
        executor = SimpleNamespace(execute=lambda text: deepcopy(result))
        question = [{"id": 1, "question": "Q"}]
        row = {"id": 1, "messages": prefix + [{"role": "assistant", "content": "FindAll <func> Count"}],
               "program_result": result, "events": [], "selected_handle": None,
               "prediction": "0", "calls": 1, "invalid_calls": 0, "stop_reason": "program_completed"}
        report = runs.replay_predictions({"P": [row]}, question, executor)
        self.assertTrue(report["audit_passed"])
        self.assertEqual(report["models"]["P"]["executed_programs"], 1)
        changed = deepcopy(row)
        changed["program_result"]["prediction"] = "00"
        self.assertEqual(runs.replay_predictions({"P": [changed]}, question, executor)["mismatch_count"], 1)
        absent = {"id": 1, "messages": prefix, "events": [], "selected_handle": None,
                  "prediction": None, "calls": 0, "invalid_calls": 0, "stop_reason": "context_budget"}
        self.assertTrue(runs.replay_predictions({"P": [absent]}, question, executor)["audit_passed"])


if __name__ == "__main__":
    unittest.main()
