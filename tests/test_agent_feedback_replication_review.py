"""CPU-only replay and evidence gates; no model weights or benchmark needed."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback import review_replication as audit
from experiments.agent_feedback.environment import Episode, call_message, compact, prompt_messages
from experiments.agent_feedback.inference import execute_response


class Engine:
    entities = {"a": {"name": "Alpha", "attributes": [], "relations": []}}
    concepts = {}

    def Find(self, deps, inputs):
        return (["a"] if inputs == ["Alpha"] else [], None)

    def Count(self, deps, inputs):
        return len(deps[0][0])


class ReplicationReviewTest(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(engine=Engine())
        self.gold = {"id": "dev:1", "question": "How many Alpha entities?", "answer": "1",
                     "program": [{"function": "Find", "inputs": ["Alpha"], "dependencies": []},
                                 {"function": "Count", "inputs": [], "dependencies": [0]}]}
        for module in (audit, audit.outcome.__module__):
            target = module if not isinstance(module, str) else __import__(module, fromlist=["compare_answers"])
            mock = patch.object(target, "compare_answers", side_effect=lambda a, b: b is not None and str(a) == b)
            mock.start()
            self.addCleanup(mock.stop)

    def rollout(self, name="Alpha", repeat_invalid=False):
        actions = [("step", {"function": "Find", "inputs": [name], "dependencies": []})]
        if repeat_invalid:
            actions += [("step", {"function": "Count", "inputs": [], "dependencies": [99]})] * 2
        actions += [("step", {"function": "Count", "inputs": [], "dependencies": [0]}),
                    ("finish", {"answer_handle": 1})]
        episode, messages = Episode(self.executor), prompt_messages(self.gold["question"])
        for tool, arguments in actions:
            action = call_message(tool, arguments)
            observation = execute_response(episode, action["content"])
            messages.extend([action, {"role": "tool", "content": compact(observation)}])
        return {"id": self.gold["id"], "messages": messages, "events": episode.events,
                "calls": episode.calls, "invalid_calls": 2 if repeat_invalid else 0,
                "selected_handle": episode.selected, "prediction": episode.prediction}

    def test_real_replay_checks_all_actions_and_repeat_counts(self):
        row = self.rollout(repeat_invalid=True)
        reviewed, replay, summary = audit.replay_arm(self.executor, [row], [self.gold], "A")
        self.assertEqual(replay, {"checked": 1, "mismatches": []})
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["invalid_calls"], 2)
        self.assertEqual(summary["repeated_calls"], 1)
        self.assertEqual(summary["repeated_invalid_calls"], 1)
        self.assertEqual(reviewed[self.gold["id"]]["handles_with_gold_matching_value"], [1])

    def test_tampered_selected_handle_or_observation_cannot_pass(self):
        for change in ("selected", "observation"):
            row = self.rollout()
            if change == "selected":
                row["selected_handle"] = 0
            else:
                row["messages"][3]["content"] = "{}"
            _, replay, _ = audit.replay_arm(self.executor, [row], [self.gold], "C")
            self.assertEqual(replay["checked"], 0)
            self.assertEqual(len(replay["mismatches"]), 1)

    def test_coverage_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.replay_arm(self.executor, [self.rollout(), self.rollout()], [self.gold], "A")
        row = self.rollout()
        row["id"] = "unseen"
        with self.assertRaisesRegex(ValueError, "coverage"):
            audit.replay_arm(self.executor, [row], [self.gold], "A")

    def test_descriptive_groups_and_changed_cases_retain_direction(self):
        rows = {"A": [self.rollout("Missing")], "C": [self.rollout()]}
        reviewed = {arm: audit.replay_arm(self.executor, data, [self.gold], arm)[0] for arm, data in rows.items()}
        matrix, cases, groups = audit.describe_pairs([self.gold], reviewed, rows)
        self.assertEqual(matrix, {"finished_wrong -> correct": 1})
        self.assertEqual(cases[0]["change"], "corrected")
        self.assertEqual(groups["comparison_numeric_or_count"]["net_corrected"], 1)
        self.assertEqual(cases[0]["program"], self.gold["program"])
        self.assertEqual(cases[0]["A_events"], rows["A"][0]["events"])
        self.assertEqual(cases[0]["C_events"], rows["C"][0]["events"])

    def test_only_derived_mean_roundoff_has_tolerance(self):
        summary = {"correct": 404, "questions": 500, "failed_to_finish": 26, "accuracy": 0.808,
                   "stop_reasons": {"finished": 474, "call_budget": 26},
                   "means": {"total_tokens": 10338.344}, "totals": {"total_tokens": 5169172}}
        metrics = {"correct": 404, "total": 500, "failed_to_finish": 26, "accuracy": 0.808,
                   "evaluation_scope": "complete", "gold_questions": 500, "prediction_questions": 500,
                   "missing_predictions": 0, "coverage": 1.0, "stop_reasons": summary["stop_reasons"],
                   "means": {"total_tokens": 10338.344000000001}}
        runtime = {"totals": {"total_tokens": 5169172, "questions": 500}}
        self.assertTrue(audit.verify_metrics(metrics, runtime, summary))
        runtime["totals"]["total_tokens"] += 1
        self.assertFalse(audit.verify_metrics(metrics, runtime, summary))
        runtime["totals"]["total_tokens"] -= 1
        metrics["correct"] -= 1
        self.assertFalse(audit.verify_metrics(metrics, runtime, summary))
        metrics["correct"] += 1
        for value in (10338.345, float("nan"), float("inf"), "10338.344"):
            metrics["means"]["total_tokens"] = value
            self.assertFalse(audit.verify_metrics(metrics, runtime, summary))
        metrics["means"] = {}
        self.assertFalse(audit.verify_metrics(metrics, runtime, summary))

    def test_revision_uses_new_output_names(self):
        self.assertEqual(audit.OUTPUT.name, "replication_review_seed20261004_v2.json")
        self.assertEqual(audit.CASES.name, "replication_review_cases_seed20261004_v2.jsonl")
        self.assertNotEqual(audit.OUTPUT, audit.PREVIOUS_OUTPUT)

    def test_gate_never_approves_formal_rl_or_semantic_training(self):
        comparisons = {str(seed): {"pilot_investment_decision": {"continue_investment": True}}
                       for seed in (audit.ORIGINAL_SEED, audit.SEED)}
        gate = audit.decision_gate({}, comparisons)
        self.assertTrue(gate["conditional_signal_check_eligible"])
        for flag in ("allow_rl", "allow_formal_rl", "allow_semantic_preference_training"):
            self.assertFalse(gate[flag])
        self.assertFalse(audit.decision_gate({"replay_mismatches": 1}, comparisons)["conditional_signal_check_eligible"])
        self.assertFalse(audit.decision_gate({}, {})["conditional_signal_check_eligible"])
        comparisons[str(audit.SEED)]["pilot_investment_decision"]["continue_investment"] = False
        self.assertFalse(audit.decision_gate({}, comparisons)["conditional_signal_check_eligible"])

    def test_holdout_binding_rejected_before_any_file_open(self):
        with patch.object(audit, "digest") as digest:
            with self.assertRaisesRegex(ValueError, "Holdout"):
                audit.require_frozen_hashes({"ordinary.json": "a", "data/agent_feedback/holdout.gold.jsonl": "b"})
            digest.assert_not_called()

    def test_identity_checks_training_seed_and_actual_input_accounting(self):
        arm, name = "A", "fixture"
        manifest = {"arms": {"A": {"input_tokens": 9, "records": 2}},
                    "tokenizer": {"files_sha256": {"tokenizer.json": "tokenizer-sha"}}}
        protocol = {"versions": {"transformers": "fixture"},
                    "code_sha256": {"experiments/agent_feedback/train_sft.py": "script-sha"}}
        inp = {"config": audit.training_config(arm, name, audit.SEED), "versions": protocol["versions"],
               "input_tokens": 9, "continuation": {"adapter_config_sha256": "digest",
               "tokenizer_files_sha256": manifest["tokenizer"]["files_sha256"]}}
        result = {"seed": audit.SEED, "script_sha256": "script-sha", "actual_input_tokens": 9,
                  "microbatches": 1, "actual_supervised_tokens": 4, "seconds": 1}
        def read(path):
            return inp if path.name == "input_manifest.json" else result
        with patch.object(audit, "validate_training"), patch.object(audit, "digest", return_value="digest"), \
                patch.object(audit, "read", side_effect=read):
            self.assertEqual(audit.validate_training_identity(arm, name, audit.SEED, manifest, protocol, "weights")["seed"], audit.SEED)
            result["seed"] = audit.ORIGINAL_SEED
            with self.assertRaisesRegex(ValueError, "seed or token budget"):
                audit.validate_training_identity(arm, name, audit.SEED, manifest, protocol, "weights")
            result["seed"], result["actual_input_tokens"] = audit.SEED, 8
            with self.assertRaisesRegex(ValueError, "seed or token budget"):
                audit.validate_training_identity(arm, name, audit.SEED, manifest, protocol, "weights")

    def test_exclusive_evidence_and_private_case_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cases = Path(directory) / "report.json", Path(directory) / "cases.jsonl"
            report = {"gate": {"allow_rl": False}}
            audit.write_reports(report, [{"id": "dev:1"}], output, cases)
            self.assertEqual(cases.stat().st_mode & 0o777, 0o600)
            original = output.read_bytes(), cases.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_reports({}, [], output, cases)
            self.assertEqual((output.read_bytes(), cases.read_bytes()), original)
            self.assertEqual(json.loads(output.read_text())["private_changed_cases"]["count"], 1)

    def test_wrong_hash_seed_rejected_before_input_validation(self):
        with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), patch.object(audit, "validate_identity") as validate:
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
                audit.main()
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
