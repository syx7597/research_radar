from copy import deepcopy
import unittest

from experiments.agent_feedback.environment import compact
from experiments.agent_feedback.inference import execute_response
from experiments.agent_feedback.same_state import restore
from experiments.agent_feedback.same_state_analysis import ARMS, contrasts, summarize, validate_resumes
from tests import test_agent_same_state as support

step, finish = support.step, support.finish


class SameStateAnalysisTest(unittest.TestCase):
    setUp = support.SameStateTest.setUp
    rollout = support.SameStateTest.rollout
    fixtures = support.SameStateTest.fixtures

    def predictions(self, unfinished=None):
        states = self.fixtures()
        predictions = {}
        for arm in ARMS:
            predictions[arm] = []
            for state in states:
                episode = restore(self.executor, state)
                messages = deepcopy(state["messages"])
                if arm == unfinished:
                    actions = ["invalid action"] * state["remaining_calls"]
                else:
                    actions = [step("Count", dependencies=[0]), finish(len(episode.handles))]
                for action in actions:
                    observation = execute_response(episode, action)
                    messages.extend([{"role": "assistant", "content": action},
                                     {"role": "tool", "content": compact(observation)}])
                predictions[arm].append({"id": state["id"], "condition": state["condition"], "arm": arm,
                    "initial_state_sha256": state["state_sha256"], "prefix_calls": state["prefix_calls"],
                    "remaining_calls": state["remaining_calls"], "calls": episode.calls,
                    "continuation_calls": len(actions), "messages": messages, "events": episode.events,
                    "prediction": episode.prediction, "stop_reason": "call_budget" if arm == unfinished else "finished",
                    "invalid_calls": len(actions) if arm == unfinished else 0,
                    "input_tokens": 100, "generated_tokens": 100, "total_tokens": 200})
        return states, predictions

    def test_complete_replay_and_failures_remain_in_denominator(self):
        states, predictions = self.predictions(unfinished="A_20261003")
        validate_resumes(self.executor, states, predictions)
        report, rows = summarize(predictions, [dict(self.gold, answer="1")], states)
        self.assertEqual(report["questions"], 1)
        self.assertEqual(report["arms"]["A_20261003"]["divergent"]["finished"], 0)
        self.assertEqual(report["contrasts"]["divergent_C_minus_A"]["delta_pp"], 50)
        self.assertEqual(report["contrasts"]["difference_of_differences"]["delta_pp"], 0)
        self.assertEqual(len(rows), 1)

    def test_single_arm_requires_explicit_probe_mode(self):
        states, predictions = self.predictions()
        probe = {"A_20261003": predictions["A_20261003"]}
        with self.assertRaises(ValueError):
            validate_resumes(self.executor, states, probe)
        validate_resumes(self.executor, states, probe, probe=True)
        with self.assertRaises(ValueError):
            validate_resumes(self.executor, states, predictions, probe=True)

    def test_reject_missing_arm_missing_state_duplicate_and_reordered(self):
        states, predictions = self.predictions()
        bad = deepcopy(predictions); del bad["C_20261004"]
        invalid = [bad]
        for rows in (predictions["A_20261003"][:1], predictions["A_20261003"] * 2,
                     predictions["A_20261003"][::-1]):
            bad = deepcopy(predictions); bad["A_20261003"] = rows; invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError):
                validate_resumes(self.executor, states, bad)

    def test_reject_changed_start_prediction_and_event(self):
        states, predictions = self.predictions()
        for key, value in (("initial_state_sha256", "0" * 64), ("prediction", "invented"),
                           ("remaining_calls", 24), ("invalid_calls", 1), ("generated_tokens", 3000)):
            bad = deepcopy(predictions); bad["A_20261003"][0][key] = value
            with self.assertRaises(ValueError):
                validate_resumes(self.executor, states, bad)
        bad = deepcopy(predictions)
        bad["A_20261003"][0]["events"][-1]["observation"]["ok"] = False
        with self.assertRaises(ValueError):
            validate_resumes(self.executor, states, bad)

    def test_early_call_budget_stop_rejected(self):
        states, predictions = self.predictions()
        bad = predictions["A_20261003"][0]
        # A stopped prefix is legitimate only for an actually reached non-call budget.
        state = states[0]
        bad.update(messages=deepcopy(state["messages"]), events=deepcopy(state["events"]),
                   calls=state["prefix_calls"], continuation_calls=0, prediction=None, stop_reason="call_budget")
        with self.assertRaisesRegex(ValueError, "unused calls"):
            validate_resumes(self.executor, states, predictions)

    def test_joint_bootstrap_preserves_cross_condition_covariance(self):
        rows = []
        for direction in (0, 1):
            rows.append({"correct": {arm: {condition: int(arm.startswith("C")) ^ direction
                         for condition in ("clean", "divergent")} for arm in ARMS}})
        report = contrasts(rows)
        self.assertEqual(report["difference_of_differences"]["ci95_pp"], [0, 0])
        self.assertEqual(report["divergent_C_minus_A"]["ci95_pp"], [-100, 100])
        self.assertEqual(contrasts(rows[::-1]), report)

    def test_interaction_can_exceed_one_hundred_pp(self):
        rows = [{"correct": {a: {"clean": int(a.startswith("A")), "divergent": int(a.startswith("C"))}
                             for a in ARMS}}]
        self.assertEqual(contrasts(rows)["difference_of_differences"]["delta_pp"], 200)

    def test_empty_cohort_not_claimed_as_zero_effect(self):
        with self.assertRaises(ValueError):
            contrasts([])


if __name__ == "__main__":
    unittest.main()
