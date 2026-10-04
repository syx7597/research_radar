"""Gold isolation, episode reset and audited-budget checks for native TRL use."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback.environment import prompt_messages
from experiments.agent_feedback.train_grpo import (
    JSONObservation, NATIVE_CONTRACT, RewardEnvironment, context_bound, digest,
    make_rows, protocol_check, run_settings,
)
from experiments.agent_feedback import train_grpo as original_grpo
from experiments.agent_feedback import train_grpo_signal as signal


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


class GRPOSignalTest(unittest.TestCase):
    def groups(self, tokens=None):
        return signal.reward_group_records(["a"] * 4, [0.0, 1.0, 0.0, 1.0],
            [-0.8, 0.8, -0.8, 0.8], tokens or [10, 10, 10, 10], [20, 20, 20, 20])

    def summary(self, groups=None, *, changed=1, lr=5e-6, gradient=0.1, finite=True):
        return signal.signal_summary(self.groups() if groups is None else groups,
            [{"all_gradients_finite": finite, "gradient_l2_after_clip": gradient, "learning_rates": [lr]}],
            {"all_before_finite": True, "all_after_finite": True, "changed_elements": changed,
             "l2_delta": 1e-6 if changed else 0.0, "sha256_before": "before",
             "sha256_after": "after" if changed else "before"}, ["a"])

    def test_seeded_selection_uses_only_ids_and_is_input_order_independent(self):
        ids = [f"train:{i}" for i in range(40)]
        selected = signal.select_smoke_question_ids(ids)
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected, signal.select_smoke_question_ids(reversed(ids)))
        self.assertEqual(len(set(selected)), 16)
        with self.assertRaises(ValueError):
            signal.select_smoke_question_ids(["duplicate"] * 16)
        with self.assertRaises(ValueError):
            signal.select_smoke_question_ids(ids[:15])

    def test_reward_group_requires_actual_contiguous_same_question_rollouts(self):
        with self.assertRaisesRegex(ValueError, "mixes"):
            signal.reward_group_records(["a", "b", "a", "a"], [0.0] * 4, [0.0] * 4, [1] * 4, [2] * 4)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            signal.reward_group_records(["a"] * 3, [0.0] * 3, [0.0] * 3, [1] * 3, [2] * 3)

    def test_mixed_rewards_with_no_unmasked_tokens_are_not_learning_signal(self):
        result = self.summary(self.groups([0, 0, 0, 0]))
        self.assertEqual(result["mixed_reward_groups"], 1)
        self.assertEqual(result["groups_with_unmasked_nonzero_advantage"], 0)
        self.assertEqual(result["zero_loss_token_rollouts"], 4)
        self.assertFalse(result["verified_nonzero_learning_signal"])

    def test_real_update_nonzero_finite_gradient_and_positive_lr_are_all_required(self):
        self.assertTrue(self.summary()["verified_nonzero_learning_signal"])
        for kwargs in ({"changed": 0}, {"gradient": 0.0}, {"lr": 0.0},
                       {"gradient": None, "finite": False}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.summary(**kwargs)["verified_nonzero_learning_signal"])
        zero = signal.reward_group_records(["a"] * 4, [0.0] * 4, [0.0] * 4, [10] * 4, [20] * 4)
        self.assertFalse(self.summary(zero)["verified_nonzero_learning_signal"])

    def test_invalid_reward_is_reported_without_invalid_json_or_positive_signal(self):
        groups = signal.reward_group_records(["a"] * 4, [float("nan"), 1.0, 0.0, 1.0],
            [0.0, 0.8, -0.8, 0.8], [10] * 4, [20] * 4)
        json.dumps(groups, allow_nan=False)
        self.assertFalse(self.summary(groups)["verified_nonzero_learning_signal"])

    def test_old_entry_is_unchanged_and_new_grpo_config_is_identical(self):
        self.assertEqual(digest(original_grpo.__file__),
            "0adeef4b717ea1c5b7daab4f59162d838051f99d97847bbf363512950de13d75")
        self.assertIs(signal.RewardEnvironment, original_grpo.RewardEnvironment)
        self.assertIs(signal.protocol_check, original_grpo.protocol_check)
        def config_call(module):
            tree = ast.parse(inspect.getsource(module.main))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "GRPOConfig"]
            self.assertEqual(len(calls), 1)
            return ast.dump(calls[0], include_attributes=False)
        self.assertEqual(config_call(original_grpo), config_call(signal))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Actual CPU tensor checks require installed torch")
class GRPOSignalTensorTest(unittest.TestCase):
    def model(self, dtype=None):
        import torch
        model = torch.nn.Module()
        model.register_parameter("lora_A", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype)))
        model.register_parameter("base_weight", torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False))
        return model

    def test_native_dtype_hash_and_delta_distinguish_bf16_rounding_from_update(self):
        import torch
        model = self.model(torch.bfloat16)
        before = signal.capture_trainable_parameters(model)
        with torch.no_grad():
            model.lora_A.add_(1e-8)
        unchanged = signal.parameter_update_summary(before, model)
        self.assertEqual(unchanged["changed_elements"], 0)
        self.assertEqual(unchanged["sha256_before"], unchanged["sha256_after"])
        with torch.no_grad():
            model.lora_A[0] += 0.5
        changed = signal.parameter_update_summary(before, model)
        self.assertEqual(changed["elements_by_dtype"], {"torch.bfloat16": 2})
        self.assertEqual(changed["changed_elements"], 1)
        self.assertEqual(changed["l2_delta"], 0.5)
        self.assertNotEqual(changed["sha256_before"], changed["sha256_after"])

    def test_actual_callback_observes_gradients_and_updates_without_changing_control(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            model = self.model(torch.float32)
            callback = signal.make_signal_callback(object, Path(folder), 0)
            control, state = object(), SimpleNamespace(global_step=0)
            callback.on_train_begin(None, state, control, model=model)
            model.lora_A.grad = torch.tensor([3.0, 4.0])
            self.assertIsNone(callback.on_pre_optimizer_step(None, state, control, model=model,
                              optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}])))
            self.assertEqual(callback.steps[0]["gradient_l2_after_clip"], 5.0)
            self.assertEqual(callback.steps[0]["learning_rates"], [0.0])
            with torch.no_grad():
                model.lora_A[1] += 1e-5
            callback.on_train_end(None, state, control, model=model)
            self.assertGreater(callback.parameters["l2_delta"], 0)
            self.assertEqual(callback.parameters["changed_elements"], 1)
            model.lora_A.grad[0] = float("nan")
            invalid = signal.gradient_observation(model, SimpleNamespace(param_groups=[{"lr": 5e-6}]), 2)
            self.assertFalse(invalid["all_gradients_finite"])
            self.assertIsNone(invalid["gradient_l2_after_clip"])

    def test_observation_hooks_return_identical_native_objects_and_preserve_masks(self):
        import torch
        class NativeTrainer:
            num_generations = 4
            reward_weights = torch.tensor([1.0])
            state = SimpleNamespace(global_step=0)
            def __init__(self):
                self.native_rewards = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
                self.native_result = {"completion_mask": torch.tensor([[1, 1], [0, 0], [1, 1], [1, 1]]),
                    "tool_mask": torch.tensor([[1, 0]] * 4), "advantages": torch.tensor([-0.8, 0.8, -0.8, 0.8])}
            def is_world_process_zero(self):
                return True
            def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
                return self.native_rewards
            def _generate_and_score_completions(self, inputs):
                self.returned_rewards = self._calculate_rewards(inputs, [], [], [[1, 2]] * 4)
                return self.native_result
        utils = SimpleNamespace(gather_object=lambda rows: rows)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,
                {"accelerate": SimpleNamespace(utils=utils), "accelerate.utils": utils}):
            groups = []
            trainer = signal.make_observed_trainer(NativeTrainer, Path(folder), groups)()
            before = trainer.native_result["completion_mask"].clone()
            result = trainer._generate_and_score_completions([{"example_id": "a"}] * 4)
            self.assertIs(result, trainer.native_result)
            self.assertIs(trainer.returned_rewards, trainer.native_rewards)
            self.assertTrue(torch.equal(before, result["completion_mask"]))
            self.assertEqual(groups[0]["loss_tokens_after_completion_and_tool_masks"], [1, 0, 1, 1])
            self.assertEqual(groups[0]["effective_signal_rollouts"], 3)


if __name__ == "__main__":
    unittest.main()
