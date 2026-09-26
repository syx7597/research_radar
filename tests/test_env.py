"""判分与回合生命周期边界测试；只用标准库和确定性假工具。"""
import unittest

from ca_agraphrag.env import KGQAEnv, score_answer


class FakeTools:
    def call(self, tool, args):
        if tool != "lookup":
            raise ValueError("unknown tool")
        return ["sample"]


class ScoreTests(unittest.TestCase):
    def test_count_accepts_integer_or_integer_text(self):
        for answer in (2, "2", " 2 "):
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(answer, 2, "count"), 1.0)

    def test_count_rejects_partial_or_noninteger_answers(self):
        for answer in ("2.5", "2 or 3", "answer: 2", [2], 2.5, 2.0, True, None):
            with self.subTest(answer=answer):
                self.assertEqual(score_answer(answer, 2, "count"), 0.0)

    def test_zero_count(self):
        self.assertEqual(score_answer(0, 0, "count"), 1.0)
        self.assertEqual(score_answer(False, 0, "count"), 0.0)

    def test_empty_set_and_partial_set_f1(self):
        self.assertEqual(score_answer([], [], "set"), 1.0)
        self.assertEqual(score_answer(["a"], [], "set"), 0.0)
        self.assertAlmostEqual(score_answer(["a"], ["a", "b"], "set"), 2 / 3)


class EpisodeTests(unittest.TestCase):
    ITEM = {"question": "How many?", "gold_answer": 2, "gold_kind": "count"}
    FINISH = {"tool": "finish", "args": {"answer": 2}}
    LOOKUP = {"tool": "lookup", "args": {"entity": "x", "relation": "r"}}

    def setUp(self):
        self.env = KGQAEnv(FakeTools(), max_steps=2)

    def test_reset_required_before_step(self):
        with self.assertRaises(RuntimeError):
            self.env.step(self.FINISH)

    def test_finish_is_terminal_and_reset_starts_new_episode(self):
        self.env.reset(self.ITEM)
        _, _, done, info = self.env.step(self.FINISH)
        self.assertTrue(done)
        self.assertEqual(info["correctness"], 1.0)
        with self.assertRaises(RuntimeError):
            self.env.step(self.FINISH)
        self.env.reset(self.ITEM)
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.trace, [])
        self.assertTrue(self.env.step(self.FINISH)[2])

    def test_budget_exhaustion_is_terminal(self):
        self.env.reset(self.ITEM)
        self.assertFalse(self.env.step(self.LOOKUP)[2])
        self.assertTrue(self.env.step(self.LOOKUP)[2])
        with self.assertRaises(RuntimeError):
            self.env.step(self.FINISH)
        self.assertEqual(self.env.steps, 2)

    def test_finish_allowed_on_last_budgeted_step(self):
        self.env.reset(self.ITEM)
        self.env.step(self.LOOKUP)
        _, _, done, info = self.env.step(self.FINISH)
        self.assertTrue(done)
        self.assertEqual(info["correctness"], 1.0)

    def test_malformed_actions_become_error_observations(self):
        actions = [None, [], "lookup", {}, {"tool": []},
                   {"tool": "finish", "args": []}, {"tool": "finish", "args": "invalid"},
                   {"tool": "not_a_tool", "args": {}}]
        for action in actions:
            with self.subTest(action=action):
                self.env.reset(self.ITEM)
                obs, reward, done, info = self.env.step(action)
                self.assertIn("error", obs)
                self.assertLess(reward, 0.0)
                self.assertFalse(done)
                self.assertEqual(self.env.steps, 1)

    def test_invalid_step_budgets_rejected(self):
        for max_steps in (0, -1, 1.5, True):
            with self.subTest(max_steps=max_steps):
                with self.assertRaises(ValueError):
                    KGQAEnv(FakeTools(), max_steps=max_steps)


if __name__ == "__main__":
    unittest.main()
