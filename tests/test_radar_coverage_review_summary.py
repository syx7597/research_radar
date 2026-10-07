import copy
import unittest

from experiments.radar_domain.coverage_review_summary import merge_reviews, summarize, validate_verdict


def verdict(qid, success=True, binding=False):
    return {"review_id": qid, "question_id": qid, "answer_correct": success,
            "supported_by_supplied_evidence": success, "citations_support": success,
            "binding_error_any": binding, "binding_error_types": ["unit"] if binding else [],
            "refusal": False, "redundancy": False, "reason_zh": "synthetic fixture", "evidence_basis": []}


class ReviewAggregationTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"denominator_each_arm": 4, "question_groups": {"q1": "g1", "q2": "g1", "q3": "g2", "q4": "g2"},
                       "registered_competing_question_ids": ["q1", "q2"], "registered_complex_question_ids": ["q2"],
                       "missing_reference_page_question_ids": ["q4"],
                       "continuation_gate": {"net_joint_correct_bound_minus_flat_at_least": 2,
                                             "source_groups_with_positive_net_gain_at_least": 2}}
        self.mapping, self.rows = {}, {}
        for arm in ("raw", "flat", "bound"):
            for q in self.policy["question_groups"]:
                rid = arm + q
                self.mapping[rid] = {"question_id": q, "arm": arm}
                row = verdict(q, (arm == "bound" and q in ("q1", "q3")))
                row["review_id"] = rid
                self.rows[rid] = row

    def test_gate_requires_spread_and_nonincreasing_binding_errors(self):
        result = summarize(self.rows, self.mapping, self.policy)
        self.assertTrue(result["continuation_gate"]["passed"])
        self.assertEqual(result["overall"]["paired_bound_minus_flat"], {"wins": 2, "losses": 0, "net": 2, "percentage_points": 50})
        self.rows["boundq2"]["binding_error_any"] = True
        self.assertFalse(summarize(self.rows, self.mapping, self.policy)["continuation_gate"]["passed"])

    def test_truth_alone_cannot_pass_joint_or_remove_missing_page(self):
        self.rows["boundq4"]["answer_correct"] = True
        result = summarize(self.rows, self.mapping, self.policy)
        self.assertEqual(result["overall"]["bound"]["answer_correct"], 3)
        self.assertEqual(result["overall"]["bound"]["joint_correct"], 2)
        self.assertEqual(result["missing_reference_pages_subset"]["bound"]["denominator"], 1)
        del self.rows["flatq4"]
        with self.assertRaisesRegex(ValueError, "Do not remove"):
            summarize(self.rows, self.mapping, self.policy)

    def test_same_net_gain_in_one_source_group_does_not_pass(self):
        self.rows["boundq2"].update(verdict("q2", True), review_id="boundq2")
        self.rows["boundq3"].update(verdict("q3", False), review_id="boundq3")
        result = summarize(self.rows, self.mapping, self.policy)
        self.assertEqual(result["overall"]["paired_bound_minus_flat"]["net"], 2)
        self.assertFalse(result["continuation_gate"]["passed"])
        self.assertEqual(result["continuation_gate"]["positive_source_groups"], 1)

    def test_disagreement_cannot_be_implicitly_accepted(self):
        first = {"x": verdict("x")}
        second = copy.deepcopy(first)
        second["x"]["citations_support"] = False
        with self.assertRaisesRegex(ValueError, "explicit adjudication"):
            merge_reviews(first, second, [])
        final, disputes = merge_reviews(first, second, [second["x"]])
        self.assertEqual(disputes, {"x": ["citations_support"]})
        self.assertFalse(final["x"]["citations_support"])
        self.assertTrue(first["x"]["citations_support"])

    def test_review_cannot_use_string_booleans_or_inconsistent_flags(self):
        row = verdict("q")
        row["answer_correct"] = "false"
        with self.assertRaisesRegex(ValueError, "actual booleans"):
            validate_verdict(row)
        row = verdict("q", binding=True)
        row["binding_error_types"] = []
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_verdict(row)


if __name__ == "__main__":
    unittest.main()
