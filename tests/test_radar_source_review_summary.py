"""Protect paired AI-review coverage and prevent silent disagreement removal."""
from copy import deepcopy
import unittest

from experiments.radar_domain import source_review_summary as review


def verdict(bid="b1", **changes):
    row = {"blind_id": bid, "answer_correct": True, "answer_supported": True,
           "citations_support_answer": True, "required_conditions_preserved": None,
           "notes_zh": "Source-grounded reason"}
    row.update(changes)
    return row


class ReviewSummaryTests(unittest.TestCase):
    def test_missing_duplicate_and_unmasked_reviews_rejected(self):
        document = {"human_review": False, "arm_labels_seen": False, "reviewer": "AI 1",
                    "records": [verdict(), verdict("b2")]}
        self.assertEqual(len(review.validate_review(document, {"b1", "b2"})), 2)
        for changes in ({"records": [verdict(), verdict()]},
                        {"records": [verdict()]}, {"human_review": True}, {"arm_labels_seen": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                review.validate_review({**document, **changes}, {"b1", "b2"})

    def test_integer_truthiness_does_not_count_as_review(self):
        with self.assertRaises(ValueError):
            review.validate_scores(verdict(answer_correct=1))
        broken = verdict()
        del broken["required_conditions_preserved"]
        with self.assertRaises(ValueError):
            review.validate_scores(broken)

    def test_all_disagreements_must_be_explicitly_resolved(self):
        first = {"b1": verdict()}
        second = {"b1": verdict(answer_correct=False)}
        with self.assertRaises(ValueError):
            review.resolve(first, second, {"records": []})
        decision = verdict(answer_correct=False, reason_zh="Missing requested value")
        final, disagreements = review.resolve(first, second, {"records": [decision]})
        self.assertFalse(final["b1"]["answer_correct"])
        self.assertEqual(disagreements, {"b1": ["answer_correct"]})

    def test_agreed_fields_cannot_be_changed_in_adjudication(self):
        first = {"b1": verdict()}
        second = {"b1": verdict(answer_correct=False)}
        decision = verdict(answer_correct=False, answer_supported=False, reason_zh="Changed consensus")
        with self.assertRaisesRegex(ValueError, "agreed score"):
            review.resolve(first, second, {"records": [decision]})
        with self.assertRaises(ValueError):
            review.resolve(first, deepcopy(first), {"records": [decision]})

    def test_fixed_condition_subset_and_original_multiplicity_preserved(self):
        rows = [{"id": "q1", "blind_id": "b1"}, {"id": "q2", "blind_id": "b1"},
                {"id": "q3", "blind_id": "b2"}]
        refs = {"q1": {"intent": {"condition_raw": "mode"}, "semantic_claims": [{"condition": "mode"}]},
                "q2": {"intent": {"condition_raw": None}, "semantic_claims": [{"condition": "both beams"}]},
                "q3": {"intent": {"condition_raw": "mode"}, "semantic_claims": [{"condition": "mode"}]}}
        verdicts = {"b1": verdict(), "b2": verdict("b2", answer_correct=False,
                                                   required_conditions_preserved=True)}
        scored = review.tally(rows, verdicts, refs)
        self.assertEqual(scored["questions"], 3)
        self.assertEqual(scored["answer_correct"], 2)
        self.assertEqual(scored["correct_supported_and_cited"], 2)
        self.assertEqual(scored["qualified_questions"], 2)
        self.assertEqual(scored["qualified_conditions_preserved"], 1)
        self.assertEqual(scored["qualified_condition_not_applicable"], 1)
        self.assertEqual(scored["posthoc_reference_condition_questions"], 3)
        self.assertEqual(scored["posthoc_reference_conditions_preserved"], 1)


if __name__ == "__main__":
    unittest.main()
