from copy import deepcopy
import unittest

from experiments.agent_feedback.same_state_cohort import select_budgeted_pairs
from tests import test_agent_same_state as support


class SameStateCohortTest(unittest.TestCase):
    setUp = support.SameStateTest.setUp
    rollout = support.SameStateTest.rollout
    fixtures = support.SameStateTest.fixtures

    def inputs(self):
        states = self.fixtures()
        audits = [{"id": states[0]["id"], "status": "accepted"},
                  {"id": "excluded", "status": "excluded", "reason": "no_shared_successful_prefix"}]
        checks = [{"id": r["id"], "condition": r["condition"], "within_budgets": True,
                   "initial_prompt_tokens": 100, "max_reference_prompt_tokens": 200,
                   "reference_generated_tokens": 80, "max_reference_turn_tokens": 30,
                   "initial_prompt_ids_sha256": "a" * 64} for r in states]
        return states, audits, checks

    def test_all_questions_accounted_for_and_order_kept(self):
        states, audits, checks = self.inputs()
        selected, ledger, report = select_budgeted_pairs(states, audits, checks)
        self.assertEqual(selected, states)
        self.assertEqual([e["id"] for e in ledger], [a["id"] for a in audits])
        self.assertEqual(report["eligible_pairs"], 1)
        self.assertEqual(sum(report["counts"].values()), 2)
        self.assertFalse(report["AC_predictions_used"])

    def test_one_condition_over_budget_excludes_entire_pair(self):
        states, audits, checks = self.inputs()
        checks[1].update(within_budgets=False, reference_generated_tokens=2049)
        selected, ledger, report = select_budgeted_pairs(states, audits, checks)
        self.assertEqual(selected, [])
        self.assertFalse(report["evaluation_possible"])
        self.assertEqual(ledger[0]["reason"], "reference_token_budget")
        self.assertEqual(report["counts"]["no_shared_successful_prefix"], 1)

    def test_missing_reordered_or_extra_budget_rows_rejected(self):
        states, audits, checks = self.inputs()
        for bad in (checks[:1], checks[::-1], checks + checks):
            with self.assertRaises(ValueError):
                select_budgeted_pairs(states, audits, bad)

    def test_cannot_drop_replay_accepted_pair(self):
        states, audits, checks = self.inputs()
        with self.assertRaises(ValueError):
            select_budgeted_pairs([], audits, [])

    def test_bad_pass_flag_and_duplicate_ledger_rejected(self):
        states, audits, checks = self.inputs()
        for key, value in (("within_budgets", "true"), ("reference_generated_tokens", 2049),
                           ("initial_prompt_ids_sha256", "x" * 64)):
            bad = deepcopy(checks); bad[0][key] = value
            with self.assertRaises(ValueError):
                select_budgeted_pairs(states, audits, bad)
        with self.assertRaises(ValueError):
            select_budgeted_pairs(states, audits + audits, checks)

    def test_no_constructible_states_is_non_estimable(self):
        _, audits, _ = self.inputs()
        states, ledger, report = select_budgeted_pairs([], audits[1:], [])
        self.assertFalse(report["evaluation_possible"])
        self.assertEqual(states, [])
        self.assertEqual(len(ledger), 1)


if __name__ == "__main__":
    unittest.main()
