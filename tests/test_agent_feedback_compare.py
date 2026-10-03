"""Complete-pair evaluation and distinction between pilot and statistical evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback.compare import (
    DEFAULT_GO_RULE, compare_predictions, digest, investment_decision,
    paired_bootstrap, validate_dev_path, validate_frozen_dev,
)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        compare = patch("experiments.agent_feedback.compare.compare_answers",
                        side_effect=lambda gold, prediction: prediction is not None and str(gold) == prediction)
        compare.start()
        self.addCleanup(compare.stop)
        self.gold = [{"id": f"q{i}", "answer": "yes"} for i in range(4)]

    def predictions(self, answers, input_tokens=100, generated_tokens=20):
        return [{"id": f"q{i}", "prediction": answer,
                 "stop_reason": "finished" if answer is not None else "call_budget",
                 "calls": 4, "invalid_calls": int(answer is None),
                 "input_tokens": input_tokens, "generated_tokens": generated_tokens,
                 "total_tokens": input_tokens + generated_tokens}
                for i, answer in enumerate(answers)]

    def compare(self, a, c, gold=None):
        return compare_predictions(a, c, gold or self.gold, bootstrap_replicates=1000)

    def test_corrections_regressions_and_failures_use_full_paired_denominator(self):
        a = self.predictions(["yes", "no", "yes", None])
        c = self.predictions(["no", "yes", "yes", "yes"])
        report = self.compare(a, list(reversed(c)))
        self.assertEqual(report["accuracy_delta_pp"], 25)
        self.assertEqual(report["arms"]["baseline"]["accuracy"], 0.5)
        self.assertEqual(report["arms"]["candidate"]["accuracy"], 0.75)
        self.assertEqual(report["paired_outcomes"], {"corrected": 2, "regressed": 1,
                                                   "both_correct": 1, "both_wrong": 0, "net_corrected": 1})
        self.assertEqual(report["arms"]["baseline"]["failed_to_finish"], 1)
        self.assertEqual(report["coverage"], 1)

    def test_missing_extra_duplicate_or_empty_ids_are_rejected(self):
        good = self.predictions(["yes"] * 4)
        extra = deepcopy(good)
        extra[0]["id"] = "unknown"
        for bad in ([], good[:-1], good + [good[0]], extra):
            for arm in ("baseline", "candidate"):
                with self.subTest(arm=arm, size=len(bad)), self.assertRaises(ValueError):
                    self.compare(bad if arm == "baseline" else good,
                                 bad if arm == "candidate" else good)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare(good, good, self.gold + [self.gold[0]])

    def test_cost_uses_full_repeated_prefill_not_just_generated_tokens(self):
        a = self.predictions(["yes"] * 4, input_tokens=80, generated_tokens=20)
        c = self.predictions(["yes"] * 4, input_tokens=120, generated_tokens=10)
        report = self.compare(a, c)
        self.assertEqual(report["arms"]["baseline"]["totals"]["total_tokens"], 400)
        self.assertEqual(report["arms"]["candidate"]["means"]["total_tokens"], 130)
        self.assertAlmostEqual(report["pilot_investment_decision"]["total_inference_token_reduction_fraction"], -0.3)
        self.assertFalse(report["pilot_investment_decision"]["continue_investment"])

    def test_cost_and_terminal_state_inconsistencies_are_rejected(self):
        good = self.predictions(["yes"] * 4)
        changes = ({"total_tokens": 20}, {"input_tokens": -1}, {"calls": True},
                   {"invalid_calls": 9}, {"prediction": None}, {"stop_reason": "call_budget"})
        for change in changes:
            bad = deepcopy(good)
            bad[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare(good, bad)

    def test_identical_predictions_have_zero_width_paired_interval(self):
        rows = self.predictions(["yes", "no", "yes", None])
        report = self.compare(rows, deepcopy(rows))
        self.assertEqual(report["statistical_evidence"]["accuracy_delta_pp_ci"], [0, 0])
        self.assertEqual(report["statistical_evidence"]["interval_assessment"], "includes_zero")

    def test_pilot_go_does_not_claim_statistical_support(self):
        a = self.predictions(["yes", "no", "yes", None])
        c = self.predictions(["no", "yes", "yes", "yes"])
        report = self.compare(a, c)
        self.assertTrue(report["pilot_investment_decision"]["continue_investment"])
        self.assertEqual(report["statistical_evidence"]["interval_assessment"], "includes_zero")

    def test_bootstrap_is_seeded_paired_and_order_invariant(self):
        differences = [-1, 0, 1, 1, 0]
        first = paired_bootstrap(differences, replicates=1000, seed=20261003)
        second = paired_bootstrap(list(reversed(differences)), replicates=1000, seed=20261003)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence_level"], 0.95)
        with self.assertRaises(ValueError):
            paired_bootstrap(differences, replicates=10)


class InvestmentRuleTest(unittest.TestCase):
    def test_prespecified_thresholds_are_inclusive_and_independent(self):
        self.assertTrue(investment_decision(2, 100, 200, DEFAULT_GO_RULE)["continue_investment"])
        self.assertFalse(investment_decision(1.99, 100, 100, DEFAULT_GO_RULE)["continue_investment"])
        result = investment_decision(-1, 100, 80, DEFAULT_GO_RULE)
        self.assertTrue(result["efficiency_threshold_passed"])
        self.assertFalse(result["accuracy_threshold_passed"])
        self.assertFalse(investment_decision(-1.01, 100, 80, DEFAULT_GO_RULE)["continue_investment"])
        self.assertFalse(investment_decision(0, 100, 81, DEFAULT_GO_RULE)["continue_investment"])

    def test_zero_cost_baseline_disables_efficiency_branch(self):
        result = investment_decision(0, 0, 0, DEFAULT_GO_RULE)
        self.assertIsNone(result["total_inference_token_reduction_fraction"])
        self.assertFalse(result["continue_investment"])


class FrozenDevelopmentTest(unittest.TestCase):
    def test_only_complete_unchanged_development_gold_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dev.gold.jsonl"
            rows = [{"id": "dev:1", "answer": "yes"}]
            path.write_text(json.dumps(rows[0]) + "\n")
            manifest = {"splits": {"dev": {"ids": ["dev:1"], "count": 1,
                                             "gold": {"path": str(path), "sha256": digest(path)}}}}
            validate_frozen_dev(rows, path, manifest)
            # The other file need not exist: rejection occurs before opening it.
            with self.assertRaisesRegex(ValueError, "not opened"):
                validate_dev_path(Path(temp) / "holdout.gold.jsonl", manifest)
            with self.assertRaisesRegex(ValueError, "IDs"):
                validate_frozen_dev([{"id": "different", "answer": "yes"}], path, manifest)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                validate_dev_path(path, manifest)


if __name__ == "__main__":
    unittest.main()
