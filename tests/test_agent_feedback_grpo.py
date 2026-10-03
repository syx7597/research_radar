"""Gold isolation, episode reset and audited-budget checks for native TRL use."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback.environment import prompt_messages
from experiments.agent_feedback.train_grpo import (
    JSONObservation, NATIVE_CONTRACT, RewardEnvironment, context_bound, digest,
    make_rows, protocol_check, run_settings,
)


class Engine:
    entities = {"a": {"name": "Alpha", "attributes": [], "relations": []}}
    concepts = {}

    def Find(self, deps, inputs):
        return (["a"] if inputs == ["Alpha"] else [], None)

    def Count(self, deps, inputs):
        return len(deps[0][0])


class GRPOEnvironmentTest(unittest.TestCase):
    def env(self, answer="1"):
        env = RewardEnvironment(SimpleNamespace(engine=Engine()))
        self.assertIsNone(env.reset(prompt_messages("How many?"), "example:1", answer))
        return env

    def test_gold_is_private_and_reset_returns_no_observation(self):
        rows = make_rows([{"id": "a", "question": "How many?"}],
                         [{"id": "a", "question": "How many?", "answer": "PRIVATE_TARGET",
                           "program": [{"secret": "PRIVATE_PROGRAM"}]}])
        self.assertNotIn("PRIVATE_TARGET", json.dumps(rows[0]["prompt"]))
        self.assertNotIn("PRIVATE_PROGRAM", json.dumps(rows))
        self.assertEqual(rows[0]["reference_answer"], "PRIVATE_TARGET")
        env = self.env()
        exposed = [name for name, _ in inspect.getmembers(env, inspect.ismethod)
                   if not name.startswith("_") and name not in {"reset", "get_reward"}]
        self.assertEqual(exposed, ["finish", "step"])

    def test_real_empty_result_is_not_rewarded_until_finish(self):
        env = self.env("0")
        with patch("experiments.agent_feedback.train_grpo.compare_answers",
                   side_effect=lambda answer, prediction: prediction is not None and answer == prediction):
            env.step("Find", ["Missing"], [])
            env.step("Count", [], [0])
            self.assertEqual(env._episode.prediction, None)
            # Call get_reward only after the complete rollout, as TRL does.
            self.assertEqual(json.loads(str(env.finish(1)))["answer"], "0")
            self.assertEqual(env.get_reward(), 1.0)
            self.assertEqual(env.get_reward(), 1.0)

    def test_reward_uses_answer_comparison_without_shape_bonus(self):
        env = self.env("2")
        env.step("Find", ["Alpha"], [])
        env.step("Count", [], [0])
        env.finish(1)
        with patch("experiments.agent_feedback.train_grpo.compare_answers", return_value=False) as compare:
            self.assertEqual(env.get_reward(), 0.0)
            compare.assert_called_once_with("2", "1")

    def test_group_members_and_reused_environment_have_private_state(self):
        executor = SimpleNamespace(engine=Engine())
        first, second = RewardEnvironment(executor), RewardEnvironment(executor)
        for env in [first, second]:
            env.reset(prompt_messages("How many?"), "same-id", "1")
        self.assertEqual(first.step("Find", ["Alpha"], []), second.step("Find", ["Alpha"], []))
        first.step("Count", [], [0])
        self.assertEqual(len(second._episode.handles), 1)
        first.reset(prompt_messages("Another question?"), "next-id", "0")
        self.assertEqual(first._episode.handles, [])
        self.assertIsNone(first._cached_reward)
        self.assertIsNone(first._episode.prediction)
        self.assertEqual(first._answer, "0")

    def test_observation_is_dict_and_uses_sft_json_serialization(self):
        result = self.env().step("Find", ["Alpha"], [])
        self.assertIsInstance(result, dict)
        self.assertEqual(json.loads(str(result)), result)
        self.assertIn('"ok":true', str(result))
        self.assertEqual(str(JSONObservation(ok=False, value=None)), '{"ok":false,"value":null}')

    def test_questions_may_not_smuggle_labels_and_gold_ids_must_match(self):
        with self.assertRaises(ValueError):
            make_rows([{"id": "a", "question": "Q", "answer": "A"}],
                      [{"id": "a", "question": "Q", "answer": "A"}])
        with self.assertRaises(ValueError):
            make_rows([{"id": "a", "question": "Q"}],
                      [{"id": "b", "question": "Q", "answer": "A"}])

    def test_native_budget_differences_require_protocol_record(self):
        protocol = {"rl": {"group_size": 4, "max_updates": 200, "learning_rate": 5e-6,
                    "beta": 0.0, "num_iterations": 1, "max_calls": 24,
                    "per_turn_max_new_tokens": 192, "total_completion_token_cap": 4096}}
        self.assertFalse(protocol_check(protocol))
        protocol["rl"]["native_trl_contract"] = dict(NATIVE_CONTRACT)
        self.assertTrue(protocol_check(protocol))
        protocol["rl"]["native_trl_contract"]["completion_budget_includes_tool_observations"] = False
        self.assertFalse(protocol_check(protocol))

    def test_smoke_is_bounded_and_never_saves_formal_weights(self):
        formal = run_settings(None, "runs/B")
        self.assertEqual(formal["max_steps"], 200)
        self.assertFalse(formal["smoke"])
        self.assertTrue(formal["save_model"])
        for steps in (1, 2):
            smoke = run_settings(steps, "runs/grpo_smoke")
            self.assertEqual(smoke["max_steps"], steps)
            self.assertTrue(smoke["smoke"])
            self.assertFalse(smoke["formal_result_eligible"])
            self.assertFalse(smoke["save_model"])
        for steps in (0, 3, 200, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                run_settings(steps, "runs/grpo_smoke")
        with self.assertRaises(ValueError):
            run_settings(2, "runs/B")

    def test_context_check_includes_observations_and_temporary_full_turn(self):
        self.assertEqual(context_bound([753, 833]), 833 + 4096 + 192)
        self.assertEqual(context_bound([3904]), 8192)
        with self.assertRaises(ValueError):
            context_bound([3905])
        with self.assertRaises(ValueError):
            context_bound([])

    def test_rl_amendment_preserves_original_and_strict_inference(self):
        root = Path(__file__).resolve().parents[1]
        original_path = root / "results/agent_feedback/protocol.json"
        original = json.loads(original_path.read_text())
        amended = json.loads((root / "results/agent_feedback/protocol_rl_native.json").read_text())
        self.assertEqual(amended["amendment"]["base_protocol_sha256"], digest(original_path))
        self.assertTrue(amended["amendment"]["before_any_gpu_training"])
        self.assertEqual(original["rl"]["total_generated_token_cap"], 2048)
        self.assertNotIn("total_generated_token_cap", amended["rl"])
        self.assertEqual(amended["rl"]["total_completion_token_cap"], 4096)
        self.assertEqual(amended["inference"], original["inference"])
        self.assertEqual(amended["rl"]["training_arms"], ["B", "D"])
        self.assertTrue(protocol_check(amended))
        with self.assertRaises(ValueError):
            protocol_check(original)


if __name__ == "__main__":
    unittest.main()
