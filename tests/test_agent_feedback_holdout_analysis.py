"""Independent paired estimand and coverage checks, using synthetic data only."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments.agent_feedback.holdout_analysis import (
    LABELS, analyze_predictions, question_groups, seed_average_bootstrap,
)


class HoldoutAnalysisTest(unittest.TestCase):
    def setUp(self):
        for module in ("holdout_analysis", "compare"):
            mock = patch(f"experiments.agent_feedback.{module}.compare_answers",
                         side_effect=lambda gold, prediction: prediction is not None and str(gold) == prediction)
            mock.start()
            self.addCleanup(mock.stop)
        self.gold = [{"id": str(i), "answer": "yes", "program": [{"function": "Find"}]} for i in range(4)]

    def rows(self, answers):
        return [{"id": str(i), "prediction": answer, "stop_reason": "finished" if answer else "call_budget",
                 "calls": 4, "invalid_calls": int(answer is None), "input_tokens": 80, "generated_tokens": 20,
                 "total_tokens": 100} for i, answer in enumerate(answers)]

    def fixture(self):
        return {"P": self.rows(["yes", "no", None, "yes"]),
                "A_20261003": self.rows(["no", "no", "yes", "yes"]),
                "C_20261003": self.rows(["yes", "no", "yes", "yes"]),
                "A_20261004": self.rows(["yes", "no", "yes", "yes"]),
                "C_20261004": self.rows(["yes", "yes", "yes", "yes"])}

    def test_primary_keeps_question_pairs_together_and_reports_all_models(self):
        report = analyze_predictions(self.fixture(), self.gold, 1000)
        self.assertEqual(report["primary"]["questions"], 4)
        self.assertEqual(report["primary"]["accuracy_delta_pp"], 25)
        self.assertEqual(report["primary"]["accuracy_delta_pp_ci"],
                         seed_average_bootstrap([.5, .5, 0, 0], 1000)["accuracy_delta_pp_ci"])
        self.assertEqual(set(report["arms"]), set(LABELS))
        self.assertTrue(report["primary"]["both_seed_point_estimates_positive"])
        for comparison in report["paired_comparisons"].values():
            self.assertNotIn("pilot_investment_decision", comparison)

    def test_conflicting_seeds_are_not_hidden_by_selecting_better_seed(self):
        predictions = self.fixture()
        predictions["C_20261004"] = self.rows(["no"] * 4)
        report = analyze_predictions(predictions, self.gold, 1000)
        self.assertEqual(report["primary"]["accuracy_delta_pp"], -25)
        self.assertFalse(report["primary"]["both_seed_point_estimates_positive"])

    def test_missing_model_or_prediction_and_duplicate_ids_fail(self):
        for kind in ("model", "row", "duplicate"):
            predictions = self.fixture()
            if kind == "model":
                predictions.pop("C_20261004")
            elif kind == "row":
                predictions["P"].pop()
            else:
                predictions["P"].append(deepcopy(predictions["P"][0]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analyze_predictions(predictions, self.gold, 1000)

    def test_bootstrap_order_invariance_and_invalid_units(self):
        values = [-1, -.5, 0, .5, 1]
        self.assertEqual(seed_average_bootstrap(values, 1000), seed_average_bootstrap(values[::-1], 1000))
        for bad in ([], [.1], [True], [2]):
            with self.assertRaises(ValueError):
                seed_average_bootstrap(bad, 1000)

    def test_groups_overlap_and_empty_group_remains_explicit(self):
        ref = {"program": [{"function": f} for f in ["Find", "QFilterNum", "And", "Count", "What", "Find"]]}
        self.assertEqual(question_groups(ref), ["reference_steps_gt_5", "qualifier", "set_operation", "comparison_numeric_or_count"])
        report = analyze_predictions(self.fixture(), self.gold, 1000)
        empty = report["descriptive_overlapping_groups"]["qualifier"]
        self.assertEqual(empty["questions"], 0)
        self.assertIsNone(empty["accuracy"]["P"])
        self.assertEqual(report["cost_distribution"]["P"]["total_tokens"]["max"], 100)


if __name__ == "__main__":
    unittest.main()
