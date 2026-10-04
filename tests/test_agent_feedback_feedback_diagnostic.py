"""CPU checks for a rendering-only intervention and honest diagnostic coverage."""
from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.agent_feedback import feedback_mask as renderer
from experiments.agent_feedback import feedback_diagnostic_analysis as analysis
from experiments.agent_feedback import inference
from experiments.agent_feedback.environment import Episode, call_message, compact, prompt_messages


class Engine:
    entities = {"a": {"name": "Alpha", "attributes": [], "relations": []}}
    concepts = {}

    def Find(self, deps, inputs):
        return (["a"] if inputs == ["Alpha"] else [], None)

    def Count(self, deps, inputs):
        return len(deps[0][0])


class FeedbackDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(engine=Engine())
        self.question = {"id": "dev:1", "question": "How many Alpha entities?"}

    def rollout(self, *, mode="original", error=False, name="Alpha"):
        ep, messages = Episode(self.executor), prompt_messages(self.question["question"])
        actions = [("step", {"function": "Find", "inputs": [name], "dependencies": []})]
        if error:
            actions.append(("step", {"function": "Count", "inputs": [], "dependencies": [99]}))
        actions += [("step", {"function": "Count", "inputs": [], "dependencies": [0]}),
                    ("finish", {"answer_handle": 1})]
        for tool, arguments in actions:
            action = call_message(tool, arguments)
            observation = inference.execute_response(ep, action["content"])
            messages.extend([action, {"role": "tool", "content": compact(renderer.visible_observation(observation, mode))}])
        return {"id": self.question["id"], "messages": messages, "events": ep.events,
                "prediction": ep.prediction, "selected_handle": ep.selected, "calls": ep.calls,
                "invalid_calls": int(error), "input_tokens": 1, "generated_tokens": 1,
                "total_tokens": 2, "stop_reason": "finished"}

    def test_masks_only_diagnostic_text_without_aliasing_original(self):
        raw = {"ok": False, "error": "ValueError", "detail": "incompatible_dependency_type",
               "handle": 2, "type": "entities", "nested": {"retained": True}}
        snapshot = deepcopy(raw)
        view = renderer.visible_observation(raw)
        self.assertEqual(view["error"], "ExecutionError")
        self.assertEqual(view["detail"], "execution_failed")
        self.assertEqual({k: v for k, v in view.items() if k not in {"error", "detail"}},
                         {k: v for k, v in raw.items() if k not in {"error", "detail"}})
        view["nested"]["retained"] = False
        self.assertEqual(raw, snapshot)

    def test_successful_empty_zero_no_schema_and_finish_are_preserved(self):
        observations = [{"ok": True, "handle": 0, "type": "entities", "count": 0, "sample": [],
                         "attributes": [], "relations": [], "qualifiers": []},
                        {"ok": True, "handle": 1, "type": "answer", "value": "0"},
                        {"ok": True, "handle": 2, "type": "answer", "value": "no"},
                        {"ok": True, "done": True, "answer": "None"}]
        for raw in observations:
            self.assertEqual(renderer.visible_observation(raw), raw)
        with self.assertRaises(ValueError):
            renderer.visible_observation({"ok": "false"})

    def test_wrapped_executor_keeps_real_error_and_prior_handles(self):
        original = inference.execute_response
        ep = Episode(self.executor)
        with renderer.rendering_intervention() as counts:
            inference.execute_response(ep, call_message("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []})["content"])
            observation = inference.execute_response(ep, call_message("step", {"function": "Count", "inputs": [], "dependencies": [99]})["content"])
            self.assertEqual(observation["error"], "ExecutionError")
            self.assertEqual(ep.events[-1]["observation"]["error"], "ValueError")
            self.assertEqual(len(ep.handles), 1)
            inference.execute_response(ep, call_message("step", {"function": "Count", "inputs": [], "dependencies": [0]})["content"])
            inference.execute_response(ep, call_message("finish", {"answer_handle": 1})["content"])
        self.assertIs(inference.execute_response, original)
        self.assertEqual(ep.prediction, "1")
        self.assertEqual(counts, {"observations": 4, "failed_observations": 1, "masked_observations": 1})

    def test_wrapper_restores_original_function_after_exception(self):
        original = inference.execute_response
        with self.assertRaises(RuntimeError):
            with renderer.rendering_intervention():
                raise RuntimeError("fixture")
        self.assertIs(inference.execute_response, original)

    def test_real_events_and_masked_visible_messages_replay_separately(self):
        normal, masked = self.rollout(error=True), self.rollout(error=True, mode=renderer.MODE)
        self.assertEqual(normal["events"], masked["events"])
        self.assertNotEqual(normal["messages"], masked["messages"])
        audit = analysis.validate_diagnostic_predictions([normal], [masked], [self.question], executor=self.executor)
        self.assertTrue(audit["audit_passed"])
        self.assertTrue(audit["pre_intervention_equivalence_passed"])
        self.assertEqual(audit["normal_error_exposed_ids"], ["dev:1"])
        self.assertEqual(audit["coverage"]["trajectories_replay_verified"], 2)

    def test_visible_feedback_or_true_event_tampering_is_rejected(self):
        normal, masked = self.rollout(error=True), self.rollout(error=True, mode=renderer.MODE)
        masked["messages"][5] = deepcopy(normal["messages"][5])
        audit = analysis.validate_diagnostic_predictions([normal], [masked], [self.question], executor=self.executor)
        self.assertFalse(audit["audit_passed"])
        masked = self.rollout(error=True, mode=renderer.MODE)
        masked["events"][1]["observation"]["detail"] = "invented raw error"
        audit = analysis.validate_diagnostic_predictions([normal], [masked], [self.question], executor=self.executor)
        self.assertFalse(audit["audit_passed"])

    def test_no_error_drift_is_not_hidden_as_a_feedback_effect(self):
        normal, masked = self.rollout(), self.rollout(name="Missing", mode=renderer.MODE)
        audit = analysis.validate_diagnostic_predictions([normal], [masked], [self.question], executor=self.executor)
        self.assertTrue(audit["audit_passed"])
        self.assertFalse(audit["pre_intervention_equivalence_passed"])
        self.assertEqual(audit["unexposed_negative_control_changed_ids"], ["dev:1"])
        self.assertEqual(audit["interpretation_gate"], "needs_numerical_batching_review")

    def test_coverage_and_duplicate_ids_are_not_silently_subsetted(self):
        normal = self.rollout()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.validate_diagnostic_predictions([normal, normal], [normal], [self.question], executor=self.executor)
        other = self.rollout()
        other["id"] = "outside"
        with self.assertRaisesRegex(ValueError, "entire"):
            analysis.validate_diagnostic_predictions([normal], [other], [self.question], executor=self.executor)

    def test_direct_scope_bound_and_new_error_exposure_keep_full_denominator(self):
        normal = [self.rollout(error=True), self.rollout()]
        normal[1]["id"] = "dev:2"
        masked = deepcopy(normal)
        masked[0]["prediction"] = "wrong"
        masked[1]["events"].append({"observation": {"ok": False}})
        gold = [{"id": "dev:1", "answer": "1"}, {"id": "dev:2", "answer": "1"}]
        with patch.object(analysis, "compare_answers", side_effect=lambda a, b: a == b):
            report = analysis.exposure_analysis(normal, masked, gold)
        self.assertEqual(report["questions"], 2)
        self.assertEqual(report["normal_correct_and_error_exposed_questions"], 1)
        self.assertEqual(report["newly_error_exposed_questions"], 1)
        self.assertEqual(report["direct_intervention_scope_before_any_numerical_drift"][
                         "maximum_accuracy_loss_pp_from_original_correct_exposed_questions"], 50)
        self.assertEqual(report["fixed_normal_exposure_cohorts"]["normal_error_unexposed"]["questions"], 1)


if __name__ == "__main__":
    unittest.main()
