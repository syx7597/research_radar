"""Exact per-example budget, context preservation and TRAIN-only pairing checks."""
from collections import Counter
from copy import deepcopy
import unittest

from experiments.agent_feedback.continuation_data import (
    build_schedule, cap_supervision, match_pairs, materialize_pair, reuse_summary,
)
from experiments.agent_feedback.environment import prompt_messages


def encoded(targets, offset=0):
    # Two unsupervised prefix tokens plus one supervised contiguous action span.
    ids = list(range(offset + 10, offset + 12 + targets))
    return {"input_ids": ids, "labels": [-100, -100] + ids[2:],
            "supervised_tokens": targets, "spans": [[2, 2, len(ids)]]}


def raw_row(identifier, source="gold_success", question="Count Alpha."):
    result = {"id": identifier, "source": source, "correct": True,
              "messages": prompt_messages(question) + [{"role": "assistant", "content": "Done."}],
              "supervise": [False, False, True]}
    if source != "gold_success":
        result.update(pair_id=identifier, selection={"status": "accepted"})
    return result


class ContinuationBudgetTest(unittest.TestCase):
    def pools(self):
        ordinary = [{"source_id": "train:0", "A": encoded(5), "C": encoded(5)},
                    {"source_id": "train:1", "A": encoded(8), "C": encoded(8)}]
        pairs = [{"source_id": "train:1", "A": encoded(2), "C": encoded(4, 100)}]
        return ordinary, pairs

    def test_cap_preserves_all_context_and_only_masks_later_targets(self):
        row = encoded(5)
        before = deepcopy(row)
        result = cap_supervision(row, 3)
        self.assertEqual(row, before)
        self.assertEqual(result["input_ids"], before["input_ids"])
        self.assertEqual(result["labels"], [-100, -100, 12, 13, 14, -100, -100])
        self.assertEqual(result["spans"], [[2, 2, 5]])
        self.assertEqual(result["masked_by_budget"], 2)

    def test_cap_counts_targets_across_unsupervised_gaps(self):
        row = {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, 2, -100, -100, 5, 6],
               "supervised_tokens": 3, "spans": [[2, 1, 2], [4, 4, 6]]}
        self.assertEqual(cap_supervision(row, 2)["labels"], [-100, 2, -100, -100, 5, -100])
        with self.assertRaises(ValueError):
            cap_supervision(row, 4)
        bad = deepcopy(row)
        bad["labels"][1] = 8
        with self.assertRaises(ValueError):
            cap_supervision(bad, 1)

    def test_each_pair_and_total_budget_match_despite_different_lengths(self):
        ordinary, pairs = self.pools()
        schedule, budget = build_schedule(ordinary, pairs)
        self.assertEqual(budget["total"], 13)
        self.assertEqual((budget["ordinary"], budget["paired_suffix"]), (7, 6))
        totals = {"A": 0, "C": 0}
        kinds = Counter()
        for index, entry in enumerate(schedule):
            a, c = materialize_pair(entry, index, ordinary, pairs)
            self.assertEqual(a["id"], c["id"])
            self.assertEqual(a["source_id"], c["source_id"])
            self.assertEqual(a["supervised_tokens"], c["supervised_tokens"])
            for arm, row in [("A", a), ("C", c)]:
                totals[arm] += sum(label != -100 for label in row["labels"])
            kinds[entry["kind"]] += a["supervised_tokens"]
            if entry["kind"] == "ordinary":
                self.assertEqual(a, c)
            else:
                self.assertEqual(len(c["input_ids"]), 6)
                self.assertEqual(len(a["input_ids"]), 4)
        self.assertEqual(totals, {"A": 13, "C": 13})
        self.assertEqual(kinds, {"ordinary": 7, "paired_suffix": 6})

    def test_repeat_counts_are_explicit_and_selection_is_reproducible(self):
        ordinary, pairs = self.pools()
        schedule, budget = build_schedule(ordinary, pairs)
        self.assertEqual((schedule, budget), build_schedule(ordinary, pairs))
        summary = reuse_summary(schedule, "paired_suffix", pairs, 10)
        self.assertEqual(summary["used_unique_questions"], 1)
        self.assertEqual(summary["selected_records"], 3)
        self.assertEqual(summary["records_per_used_question"], 3)
        self.assertEqual(summary["used_fraction_of_formal_train"], 0.1)
        self.assertEqual(summary["reuse_histogram"], {3: 1})

    def test_pairing_is_by_unique_training_id_not_file_order(self):
        success = [raw_row("train:1"), raw_row("train:0")]
        clean = [raw_row("train:1", "paired_clean_suffix"), raw_row("train:0", "paired_clean_suffix")]
        recovery = [raw_row("train:0", "real_failure_recovery"), raw_row("train:1", "real_failure_recovery")]
        ordinary, pairs = match_pairs(success, clean, recovery, {"train:0", "train:1"})
        self.assertEqual([r["id"] for r in ordinary], ["train:0", "train:1"])
        self.assertEqual([r["source_id"] for r in pairs], ["train:0", "train:1"])

    def test_out_of_split_and_question_mismatch_fail_before_selection(self):
        success = [raw_row("train:0")]
        clean = [raw_row("train:0", "paired_clean_suffix")]
        recovery = [raw_row("train:0", "real_failure_recovery")]
        with self.assertRaises(ValueError):
            match_pairs(success, clean, recovery, {"train:9"})
        recovery[0] = raw_row("train:0", "real_failure_recovery", "Different question")
        with self.assertRaises(ValueError):
            match_pairs(success, clean, recovery, {"train:0"})
        with self.assertRaises(ValueError):
            match_pairs(success, clean + clean, recovery, {"train:0"})

    def test_empty_recovery_pool_does_not_turn_into_clean_only_comparison(self):
        ordinary, pairs = self.pools()
        with self.assertRaises(ValueError):
            build_schedule(ordinary, [])
        with self.assertRaises(ValueError):
            match_pairs([raw_row("train:0")], [], [], {"train:0"})


if __name__ == "__main__":
    unittest.main()
