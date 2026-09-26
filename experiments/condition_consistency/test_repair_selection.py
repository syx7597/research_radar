"""Selection safety tests: conservative fallback and no gold-dependent choices."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .baseline import sha256
from .evaluate_repairs import baseline_index, choose_repair, load_aligned, verify_split


def original(valid=True, score=-1.0, prediction="original"):
    return {"origin": "original", "valid": valid, "tf_mean_logprob": score,
            "program_text": "original", "prediction": prediction}


def repair(valid=True, clean=True, score=-0.5, match=0.8, parent=0):
    return {"origin": "repair", "valid": valid, "schema_clean": clean,
            "tf_mean_logprob": score, "match_score": match, "parent_index": parent,
            "program_text": "repair", "prediction": "repaired"}


def record(candidates):
    return {"id": "train:1", "question": "A question?", "candidates": candidates}


POLICY = {"disabled": False, "alpha": 0.0, "min_match": 0.4, "margin": 0.0}


class RepairSelectionTests(unittest.TestCase):
    def test_first_valid_ignores_repairs_and_keeps_legal_empty_answer(self):
        row = record([repair(), original(False), original(prediction="None"), original()])
        self.assertEqual(baseline_index(row), 2)
        self.assertEqual(choose_repair(row, {**POLICY, "disabled": True}), 2)

    def test_execution_and_schema_validity_both_required(self):
        for candidate in (repair(valid=False), repair(clean=False)):
            row = record([original(), candidate])
            self.assertEqual(choose_repair(row, POLICY), 0)
            self.assertEqual(choose_repair(row, POLICY, unconditional=True), 0)
        candidate = repair()
        del candidate["schema_clean"]
        self.assertEqual(choose_repair(record([original(), candidate]), POLICY), 0)

    def test_match_threshold_and_margin_each_prevent_switch(self):
        self.assertEqual(choose_repair(record([original(), repair(match=0.3)]), POLICY), 0)
        self.assertEqual(choose_repair(record([original(), repair(score=-1.1)]), POLICY), 0)
        self.assertEqual(choose_repair(record([original(), repair(score=-1.0)]), POLICY), 1)

    def test_valid_original_reference_is_first_valid_not_repair_parent(self):
        row = record([original(False, -9.0), original(True, -0.1), repair(score=-0.5, parent=0)])
        self.assertEqual(choose_repair(row, POLICY), 1)

    def test_no_valid_original_compares_to_own_parent_and_can_abstain(self):
        row = record([original(False, -1.0), original(False, -0.2), repair(score=-0.5, parent=1)])
        self.assertIsNone(choose_repair(row, POLICY))
        row["candidates"][2]["parent_index"] = 0
        self.assertEqual(choose_repair(row, POLICY), 2)

    def test_equal_candidate_scores_preserve_candidate_order(self):
        row = record([original(), repair(), repair()])
        self.assertEqual(choose_repair(row, POLICY), 1)

    def test_selector_does_not_consult_gold_or_prediction_contents(self):
        row = record([original(), repair()])
        expected = choose_repair(row, POLICY)
        poisoned = copy.deepcopy(row)
        poisoned.update(answer="oracle", program=[{"function": "oracle"}])
        for candidate in poisoned["candidates"]:
            candidate.update(prediction="arbitrarily different answer", correct=False,
                             gold_answer="oracle", answers=[], empty_result=True)
        self.assertEqual(choose_repair(poisoned, POLICY), expected)

    def test_no_acceptance_ablation_bypasses_threshold_not_execution(self):
        row = record([original(), repair(score=-9.0, match=0.0)])
        disabled = {**POLICY, "disabled": True}
        self.assertEqual(choose_repair(row, disabled), 0)
        self.assertEqual(choose_repair(row, disabled, unconditional=True), 1)
        row["candidates"][1]["valid"] = False
        self.assertEqual(choose_repair(row, disabled, unconditional=True), 0)

    def test_empty_candidate_list_preserves_failure(self):
        self.assertIsNone(choose_repair(record([]), POLICY))


class InputIsolationTests(unittest.TestCase):
    def check_join(self, cache_rows, gold_rows):
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, cache_path = Path(tmp) / "gold.jsonl", Path(tmp) / "cache.jsonl"
            for path, rows in ((gold_path, gold_rows), (cache_path, cache_rows)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return load_aligned({"global": cache_path}, gold_path)

    def setUp(self):
        self.cache = record([original() for _ in range(4)] + [repair()])
        self.gold = {"id": "train:1", "question": "A question?", "answer": "answer", "program": []}

    def test_gold_fields_in_candidate_cache_are_rejected(self):
        for field in ("answer", "program"):
            cache = {**self.cache, field: self.gold[field]}
            with self.assertRaisesRegex(ValueError, "no gold"):
                self.check_join([cache], [self.gold])

    def test_missing_question_never_disappears_from_denominator(self):
        extra = {**self.gold, "id": "train:2"}
        with self.assertRaisesRegex(ValueError, "ID sets"):
            self.check_join([self.cache], [self.gold, extra])

    def test_question_text_join_is_verified(self):
        with self.assertRaisesRegex(ValueError, "Question mismatch"):
            self.check_join([{**self.cache, "question": "Different question"}], [self.gold])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            self.check_join([self.cache, self.cache], [self.gold])

    def test_unexecuted_candidate_is_rejected(self):
        self.cache["candidates"][0]["valid"] = 1
        with self.assertRaisesRegex(ValueError, "execution required"):
            self.check_join([self.cache], [self.gold])

    def test_repair_budget_cannot_silently_expand(self):
        self.cache["candidates"] = [original() for _ in range(4)] + [repair() for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "at most 4 repairs"):
            self.check_join([self.cache], [self.gold])

    def test_missing_original_is_not_a_smaller_baseline(self):
        self.cache["candidates"] = [original() for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "4 originals"):
            self.check_join([self.cache], [self.gold])


class FrozenSplitTests(unittest.TestCase):
    def test_gold_hash_and_id_set_must_both_match_declared_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold_path, manifest_path = Path(tmp) / "gold.jsonl", Path(tmp) / "manifest.json"
            gold_path.write_text('{"id":"train:1","question":"Question?","answer":"a"}\n', encoding="utf-8")
            expected = {"gold": {"sha256": sha256(gold_path)}, "ids": ["train:1"]}
            manifest_path.write_text(json.dumps({"splits": {"calibration": expected,
                                                          "holdout": {**expected, "ids": ["train:2"]}}}))
            args = SimpleNamespace(gold=gold_path, manifest=manifest_path)
            verify_split(args, {"train:1": {}}, "calibration")
            with self.assertRaisesRegex(ValueError, "frozen holdout split"):
                verify_split(args, {"train:1": {}}, "holdout")
            gold_path.write_text('{"id":"train:1","question":"Question?","answer":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen calibration split"):
                verify_split(args, {"train:1": {}}, "calibration")


if __name__ == "__main__":
    unittest.main()
