from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_table_inference import FixtureTokenizer, FixtureGenerator, reference_fixture
from experiments.radar_domain.table_program import TableExecutor
from experiments.radar_domain.table_program import digest
from experiments.radar_domain.table_recovery_data import action_text
from experiments.radar_domain.table_inference import run_episode
from experiments.radar_domain.table_evaluation import evaluate_records, aggregate, paired_counts, paired_bootstrap, score_manifest


class TableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.ref, self.meta = reference_fixture()
        self.tokenizer = FixtureTokenizer()
        self.row = run_episode({"id": self.ref["id"], "question": self.ref["question"]}, TableExecutor(self.catalog),
            self.tokenizer, FixtureGenerator(self.tokenizer, [action_text(a) for a in self.ref["actions"]]))

    def evaluate(self, row):
        return evaluate_records([row], [self.ref], [self.meta], self.catalog, "dev",
                                tokenizer=self.tokenizer, generation_id="cpu_fixture")

    def test_success_replayed_and_real_development_separated(self):
        result = self.evaluate(self.row)
        self.assertTrue(result[0]["correct"])
        self.assertEqual(result[0]["population"], "real_development")
        self.assertEqual(aggregate(result)["correct"], 1)

    def test_incomplete_or_reordered_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete frozen"):
            evaluate_records([], [self.ref], [self.meta], self.catalog, "dev", tokenizer=self.tokenizer)
        wrong = deepcopy(self.row); wrong["id"] = "other"
        with self.assertRaises(ValueError):
            self.evaluate(wrong)

    def test_observation_and_answer_tampering_fail_replay(self):
        from experiments.agent_feedback.recovery import Excluded
        for field in ("events", "answer"):
            row = deepcopy(self.row)
            if field == "events": row["events"][0]["observation"]["handle"] = 42
            else: row["answer"]["source_records"]["r1"]["value"]["text"] = "999 W"
            with self.assertRaises(Excluded):
                self.evaluate(row)

    def test_token_and_context_accounting_tampering_rejected(self):
        for key in ("input_tokens", "generated_tokens", "context_tokens_after_observation"):
            row = deepcopy(self.row); row["turn_tokens"][0][key] += 1
            with self.assertRaises(ValueError):
                self.evaluate(row)
        row = deepcopy(self.row); row["turn_tokens"][0]["generated_token_ids"][0] = ord("X")
        with self.assertRaises(ValueError):
            self.evaluate(row)

    def test_failed_episode_stays_in_denominator(self):
        row = run_episode({"id": self.ref["id"], "question": self.ref["question"]}, TableExecutor(self.catalog),
            self.tokenizer, FixtureGenerator(self.tokenizer))
        result = self.evaluate(row)
        report = aggregate(result)
        self.assertEqual((report["questions"], report["correct"], report["completed"]), (1, 0, 0))
        self.assertEqual(report["totals"]["invalid_calls"], 24)

    def test_cpu_manifest_requires_explicit_fixture_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"generation.json"
            path.write_text(json.dumps({"schema": "radar_table_rollout_manifest_v1", "status": "completed", "origin": "cpu_fixture"}))
            with self.assertRaisesRegex(ValueError, "admitted generation"):
                score_manifest(path, tokenizer=self.tokenizer)

    def test_paired_counts_preserve_wins_and_regressions(self):
        a = [{"id": str(i), "correct": v} for i, v in enumerate([False, True, True, False])]
        c = [{"id": str(i), "correct": v} for i, v in enumerate([True, False, True, True])]
        result = paired_counts(a, c)
        self.assertEqual((result["corrected"], result["regressed"], result["net"]), (2, 1, 1))
        self.assertEqual(result["delta_percentage_points"], 25)
        with self.assertRaises(ValueError):
            paired_counts(a, c[::-1])

    def test_registered_bootstrap_is_paired_and_deterministic(self):
        a = [{"id": str(i), "correct": False} for i in range(24)]
        c = [{"id": str(i), "correct": i < 12} for i in range(24)]
        result = paired_bootstrap(a, c)
        self.assertEqual(result, paired_bootstrap(a, c))
        self.assertEqual((result["replicates"], result["seed"]), (5000, 20261107))
        self.assertEqual(result["delta_percentage_points"], 50)
        self.assertLess(result["ci_percentage_points"][0], 50)
        self.assertGreater(result["ci_percentage_points"][1], 50)
        self.assertEqual(paired_bootstrap(a, a)["ci_percentage_points"], [0, 0])

    def test_completed_manifest_cannot_override_timed_out_gpu_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); ledger = folder/"ledger.json"
            job = {"job_id": "actual_run", "status": "timed_out", "gpu_hours": 1}
            ledger.write_text(json.dumps({"jobs": [job]}))
            manifest = folder/"generation.json"
            manifest.write_text(json.dumps({"schema": "radar_table_rollout_manifest_v1", "status": "completed",
                "origin": "model_generated", "generation_id": "actual_run", "model_files_sha256": {"weight": "hash"},
                "model_gpu_smoke": {"finite_logits": True}, "gpu_budget_completion": {
                    "ledger_path": str(ledger), "job_id": "actual_run", "status": "completed", "job_sha256": digest(job)}}))
            with patch("experiments.radar_domain.table_job_budget.require_round_ledger"):
                with self.assertRaisesRegex(ValueError, "terminal states disagree"):
                    score_manifest(manifest, tokenizer=self.tokenizer)


if __name__ == "__main__":
    unittest.main()
