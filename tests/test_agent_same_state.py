from copy import deepcopy
import unittest

from experiments.agent_feedback.inference import execute_response
from experiments.agent_feedback.recovery import make_pair
from experiments.agent_feedback.same_state import fixture_pair, restore, object_sha
from experiments.agent_feedback.same_state_inference import validate_inventory
from tests import test_agent_feedback_recovery as support

step, finish = support.step, support.finish


class SameStateTest(unittest.TestCase):
    setUp = support.RecoveryTest.setUp
    rollout = support.RecoveryTest.rollout

    def fixtures(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Find", ["Beta"]),
                               step("Find", ["Gamma"]), step("Count", dependencies=[2]), finish(3)])
        return fixture_pair(self.executor, make_pair(self.executor, source, self.gold))

    def test_same_remaining_calls_even_with_extra_handle(self):
        clean, divergent = self.fixtures()
        self.assertEqual(clean["remaining_calls"], divergent["remaining_calls"])
        a, c = restore(self.executor, clean), restore(self.executor, divergent)
        self.assertEqual((len(a.handles), len(c.handles)), (2, 3))
        self.assertEqual((a.max_calls, c.max_calls), (23, 24))
        self.assertEqual(a.max_calls - a.calls, c.max_calls - c.calls)

    def test_replay_uses_real_handles_and_finish_on_final_call(self):
        clean, divergent = self.fixtures()
        episode = restore(self.executor, divergent)
        # 21 remaining calls, exactly 19 rejected calls + Count + finish.
        for _ in range(19):
            execute_response(episode, "invalid action")
        execute_response(episode, step("Count", dependencies=[2]))
        execute_response(episode, finish(3))
        self.assertTrue(episode.done)
        self.assertEqual(episode.prediction, "1")
        self.assertEqual(episode.calls, 24)

    def test_answer_or_program_leak_rejected(self):
        row = self.fixtures()[0]
        for key in ("answer", "program", "reference", "suffix"):
            bad = dict(row, **{key: "hidden target"})
            with self.assertRaises(ValueError):
                restore(self.executor, bad)

    def test_tampered_observation_and_internal_state_rejected(self):
        row = self.fixtures()[1]
        bad = deepcopy(row); bad["state_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            restore(self.executor, bad)
        bad = deepcopy(row); bad["events"][0]["observation"]["count"] = 9
        with self.assertRaises(ValueError):
            restore(self.executor, bad)

    def test_target_suffix_not_in_saved_messages(self):
        clean, divergent = self.fixtures()
        self.assertEqual(divergent["messages"][:-2], clean["messages"])
        self.assertEqual(len(clean["messages"]), 6)
        self.assertEqual(len(divergent["messages"]), 8)
        self.assertFalse(any(e["tool"] == "finish" for e in divergent["events"]))

    def test_digest_retains_value_order_and_tuple_type(self):
        self.assertNotEqual(object_sha(["a", "b"]), object_sha(["b", "a"]))
        self.assertNotEqual(object_sha(["a", "b"]), object_sha(("a", "b")))

    def test_missing_duplicate_or_mismatched_budget_pairs_rejected(self):
        rows = self.fixtures()
        validate_inventory(rows)
        for bad in (rows[:1], rows + rows, rows[::-1]):
            with self.assertRaises(ValueError):
                validate_inventory(bad)
        bad = deepcopy(rows); bad[1]["remaining_calls"] -= 1
        with self.assertRaises(ValueError):
            validate_inventory(bad)


if __name__ == "__main__":
    unittest.main()
