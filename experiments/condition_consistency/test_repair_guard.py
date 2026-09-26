"""Safety boundaries for the one-rule v2 guard; v1 remains immutable."""
import copy
from types import SimpleNamespace
import unittest

from .evaluate_repair_v2 import (calibration_grid, choose_repair, guard_applies,
                                 verify_evaluation_split, verify_originals)
from .evaluate_repairs import choose_repair as choose_v1
from .test_repair_selection import POLICY, original, record, repair


class RepairGuardTests(unittest.TestCase):
    def test_valid_clean_first_original_blocks_a_better_scored_repair(self):
        baseline = {**original(), "schema_clean": True}
        row = record([baseline, repair(score=100.0)])
        self.assertEqual(choose_v1(row, POLICY), 1)
        self.assertEqual(choose_repair(row, POLICY), 0)

    def test_guard_uses_first_valid_not_first_rank_or_repair_parent(self):
        row = record([{**original(False), "schema_clean": True},
                      {**original(True), "schema_clean": True}, repair(parent=0)])
        self.assertEqual(choose_repair(row, POLICY), 1)
        row["candidates"][1]["schema_clean"] = False
        self.assertEqual(choose_repair(row, POLICY), choose_v1(row, POLICY))

    def test_missing_valid_original_or_unclean_original_delegates_to_v1(self):
        for candidates in ([], [original(False), repair()],
                           [{**original(), "schema_clean": False}, repair()],
                           [original(), repair()]):
            row = record(candidates)
            self.assertFalse(guard_applies(row))
            self.assertEqual(choose_repair(row, POLICY), choose_v1(row, POLICY))

    def test_empty_or_zero_correctness_labels_do_not_change_guard(self):
        source = record([{**original(), "schema_clean": True}, repair()])
        for prediction in ("None", "0", "no", "an arbitrary answer"):
            row = copy.deepcopy(source)
            row.update(answer="different gold", program=[], correct=False)
            row["candidates"][0].update(prediction=prediction, correct=False, empty_result=True)
            row["candidates"][1].update(prediction="different gold", correct=True)
            self.assertEqual(choose_repair(row, POLICY), 0)

    def test_unchanged_grid_has_exactly_73_distinct_policies(self):
        grid = calibration_grid()
        self.assertEqual(len(grid), 73)
        self.assertEqual(sum(policy["disabled"] for policy in grid), 1)
        self.assertEqual(len({tuple(policy.items()) for policy in grid}), 73)

    def test_missing_schema_check_is_rejected_at_cache_validation(self):
        row = record([original() for _ in range(4)])
        caches = {"global": {row["id"]: row}, "local": {row["id"]: copy.deepcopy(row)}}
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            verify_originals(caches)

    def test_holdout_cannot_be_presented_as_an_unseen_v2_split(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            verify_evaluation_split(SimpleNamespace(split="holdout"), {}, {"calibration_ids": []})


if __name__ == "__main__":
    unittest.main()
