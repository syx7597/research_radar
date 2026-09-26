"""Protocol tests: leakage, paired denominators, and mutation label safety."""
import unittest
import time

from .evaluate import evaluate, paired_comparison
from .mutations import (_remove_unary, propose, verified_negatives, bounded_execute,
                        cached_natural_negative, validated_natural_cache)
from .rerank import choose, features, train_pairs, empty_prediction


class RankingProtocolTests(unittest.TestCase):
    def test_ranking_ignores_candidate_gold_metadata(self):
        candidate = {"program": [{"function": "FilterYear", "inputs": ["year", "1990", "="], "dependencies": [0]}],
                     "valid": True, "prediction": "x"}
        contaminated = {**candidate, "answer": "1991", "correct": False, "gold_program": []}
        self.assertEqual(features("What happened in 1990?", candidate),
                         features("What happened in 1990?", contaminated))
        self.assertEqual(choose("What happened in 1990?", [candidate, contaminated]), 0)

    def test_empty_answer_is_not_rejected_by_rule(self):
        self.assertEqual(choose("Who?", [{"program": [], "valid": True, "prediction": ""}]), 0)
        self.assertTrue(empty_prediction({"prediction": "None", "answers": []}))
        self.assertFalse(empty_prediction({"prediction": "None", "answers": ["None"]}))

    def test_training_refuses_heldout_and_evaluation_refuses_overlap(self):
        candidate = {"program": [], "valid": True, "prediction": "a"}
        pair = {"id": "train:1", "question": "Who?", "positive": candidate, "negative": candidate, "split": "dev"}
        with self.assertRaises(ValueError):
            train_pairs([pair])
        pair["split"] = "train"
        model = train_pairs([pair])
        with self.assertRaises(ValueError):
            evaluate([{"id": "train:1", "question": "Who?", "answer": "a", "candidates": [candidate]}],
                     lambda a, b: a == b, model=model, samples=100)

    def test_fixed_candidates_do_not_insert_gold_and_denominators_are_all_questions(self):
        rows = [
            {"id": "a", "question": "Who?", "answer": "yes", "candidates": [
                {"program": [], "valid": False, "prediction": "yes"},
                {"program": [], "valid": True, "prediction": "yes"}]},
            {"id": "b", "question": "Who?", "answer": "yes", "candidates": []},
        ]
        result = evaluate(rows, lambda a, b: a == b, samples=100)
        self.assertEqual(result["metrics"]["top1"]["accuracy"], 0.0)
        self.assertEqual(result["metrics"]["first_valid"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["oracle_at_k"]["accuracy"], 0.5)
        self.assertFalse(result["same_k_within_cache"])
        self.assertEqual(result["comparisons"]["first_valid_vs_top1"]["corrected"], 1)

    def test_paired_counts(self):
        result = paired_comparison([True, False, False], [False, True, True], samples=100)
        self.assertEqual((result["corrected"], result["regressed"], result["net_corrected"]), (2, 1, 1))
        self.assertEqual(result["correct_to_wrong_rate"], 1)


class MutationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.program = [
            {"function": "Find", "inputs": ["Alice"], "dependencies": []},
            {"function": "Relate", "inputs": ["award", "forward"], "dependencies": [0]},
            {"function": "QFilterYear", "inputs": ["point in time", "1990", "="], "dependencies": [1]},
            {"function": "What", "inputs": [], "dependencies": [2]},
        ]

    def test_remove_condition_reconnects_dependency(self):
        changed = _remove_unary(self.program, 2)
        self.assertEqual(len(changed), 3)
        self.assertEqual(changed[2]["dependencies"], [1])
        self.assertEqual(self.program[3]["dependencies"], [2])

    def test_key_mutation_retains_literal_type(self):
        schema = {"point in time": {"year"}, "start time": {"year"}, "name": {"string"}}
        changes = list(propose(self.program, schema))
        fields = [edit["to"] for edit, _ in changes if edit["kind"] == "qualifier_key"]
        self.assertEqual(fields, ["start time"])

    def test_same_answers_are_never_marked_negative(self):
        class Executor:
            def execute(self, program):
                return {"valid": True, "prediction": "same", "error": None}
        row = {"id": "train:1", "question": "Who won in 1990?", "program": self.program, "answer": "same"}
        negatives, stats = verified_negatives(row, Executor(), lambda a, b: a == b, {})
        self.assertEqual(negatives, [])
        self.assertGreater(stats["same_answer_excluded"], 0)

    def test_natural_cache_rejects_duplicate_ids_and_question_mismatch(self):
        source = {"id": "train:1", "question": "Who?"}
        cached = {**source, "candidates": []}
        with self.assertRaises(ValueError):
            validated_natural_cache([source], [cached, cached])
        with self.assertRaises(ValueError):
            validated_natural_cache([source], [{**cached, "question": "When?"}])

    def test_natural_cache_never_resurrects_invalid_candidate(self):
        row = {"id": "train:1", "answer": "gold"}
        invalid = {"program": self.program, "valid": False, "prediction": "wrong",
                   "error": "CandidateTimeout: exceeded deadline"}
        good = {"program": self.program, "valid": True, "prediction": "gold"}
        cache = {"candidates": [invalid, good]}
        self.assertIsNone(cached_natural_negative(row, cache, lambda a, b: a == b))
        negative = {**good, "prediction": "natural_wrong", "answers": ["natural_wrong"]}
        cache["candidates"].append(negative)
        self.assertEqual(cached_natural_negative(row, cache, lambda a, b: a == b)["prediction"], "natural_wrong")

    def test_training_execution_uses_same_bounded_deadline(self):
        class SlowExecutor:
            def execute(self, program):
                time.sleep(0.2)
                return {"valid": True, "prediction": "gold"}
        result = bounded_execute(SlowExecutor(), self.program, timeout_seconds=0.01)
        self.assertFalse(result["valid"])
        self.assertTrue(result["error"].startswith("CandidateTimeout:"))
        row = {"id": "train:1", "question": "Who?", "program": self.program, "answer": "gold"}
        negatives, stats = verified_negatives(row, SlowExecutor(), lambda a, b: a == b, {}, timeout_seconds=0.01)
        self.assertEqual(negatives, [])
        self.assertEqual(stats["timeout_count"], 1)
        self.assertEqual(stats["gold_execution_timeout"], 1)


if __name__ == "__main__":
    unittest.main()
