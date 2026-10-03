"""Execution invariants that affect validity of the agent experiment."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from experiments.agent_feedback.environment import Episode, call_message
from experiments.agent_feedback.inference import (
    execute_response, generation_groups, response_tokens, score, validate_questions,
)


class Engine:
    def __init__(self):
        self.entities = {"a": {"name": "Alpha", "attributes": [], "relations": []}}
        self.concepts = {}

    def Find(self, deps, inputs):
        return (["a"] if inputs == ["Alpha"] else [], None)

    def Count(self, deps, inputs):
        return len(deps[0][0])

    def What(self, deps, inputs):
        return self.entities[deps[0][0][0]]["name"]

    def And(self, deps, inputs):
        return (sorted(set(deps[0][0]) & set(deps[1][0])), None)


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(engine=Engine())
        self.ep = Episode(self.executor)

    def test_error_does_not_consume_handle_or_destroy_prefix(self):
        self.ep.step("Find", ["Alpha"], [])
        bad = self.ep.step("Count", [], [99])
        self.assertFalse(bad["ok"])
        self.assertEqual(len(self.ep.handles), 1)
        good = self.ep.step("Count", [], [0])
        self.assertEqual(good["handle"], 1)
        self.assertEqual(self.ep.finish(1)["answer"], "1")
        self.assertEqual(self.ep.calls, 4)

    def test_empty_result_and_zero_are_valid(self):
        empty = self.ep.step("Find", ["Missing"], [])
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["count"], 0)
        self.ep.step("Count", [], [0])
        self.assertEqual(self.ep.finish(1)["answer"], "0")

    def test_failed_execution_retains_no_partial_handle(self):
        self.ep.step("Find", ["Missing"], [])
        bad = self.ep.step("What", [], [0])
        self.assertFalse(bad["ok"])
        self.assertEqual(len(self.ep.handles), 1)

    def test_failed_observation_retains_no_partial_handle(self):
        with patch.object(self.ep, "_observation", side_effect=ValueError("bad observation")):
            self.assertFalse(self.ep.step("Find", ["Alpha"], [])["ok"])
        self.assertEqual(self.ep.handles, [])
        self.assertEqual(self.ep.step("Find", ["Alpha"], [])["handle"], 0)

    def test_branch_dependencies_are_explicit(self):
        self.ep.step("Find", ["Alpha"], [])
        self.ep.step("Find", ["Missing"], [])
        self.ep.step("Find", ["Alpha"], [])
        self.ep.step("And", [], [0, 2])
        self.ep.step("Count", [], [3])
        self.assertEqual(self.ep.finish(4)["answer"], "1")

    def test_no_arbitrary_method_dispatch(self):
        self.assertFalse(self.ep.step("__class__", [], [])["ok"])
        self.assertEqual(self.ep.handles, [])

    def test_boolean_is_not_handle(self):
        self.ep.step("Find", ["Alpha"], [])
        self.assertFalse(self.ep.step("Count", [], [False])["ok"])
        self.assertFalse(self.ep.finish(False)["ok"])

    def test_type_checks_and_finish(self):
        self.ep.step("Find", ["Alpha"], [])
        self.assertFalse(self.ep.finish(0)["ok"])
        self.ep.step("Count", [], [0])
        self.assertFalse(self.ep.step("Count", [], [1])["ok"])
        self.assertFalse(self.ep.step("QFilterStr", ["x", "y"], [0])["ok"])
        self.assertTrue(self.ep.finish(1)["ok"])
        self.assertFalse(self.ep.finish(1)["ok"])
        self.assertEqual(self.ep.prediction, "1")

    def test_budget_includes_failed_calls(self):
        ep = Episode(self.executor, max_calls=2)
        ep.step("Find", ["Alpha"], [])
        ep.step("Count", [], [99])
        self.assertFalse(ep.step("Count", [], [0])["ok"])
        self.assertIsNone(ep.prediction)
        self.assertTrue(ep.done)

    def test_shared_engine_does_not_share_episode_handles(self):
        other = Episode(self.executor)
        self.ep.step("Find", ["Alpha"], [])
        self.assertFalse(other.step("Count", [], [0])["ok"])
        self.assertEqual(other.handles, [])


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.ep = Episode(SimpleNamespace(engine=Engine()))

    def call(self, name, arguments, episode=None):
        return execute_response(episode or self.ep, call_message(name, arguments)["content"])

    def test_malformed_calls_are_observed_and_consume_budget(self):
        texts = [None, "not a call", "<tool_call>null</tool_call>",
                 "<tool_call>[]</tool_call>", "<tool_call>{}</tool_call>",
                 '<tool_call>{"name":[],"arguments":{}}</tool_call>',
                 '<tool_call>{"name":{},"arguments":{}}</tool_call>',
                 '<tool_call>{"name":"step","arguments":[]}</tool_call>',
                 '<tool_call>{"name":"step","arguments":{}}</tool_call>',
                 "<tool_call>{</tool_call>"]
        for text in texts:
            with self.subTest(text=text):
                before = self.ep.calls
                observation = execute_response(self.ep, text)
                self.assertFalse(observation["ok"])
                self.assertEqual(observation["error"], "ActionFormatError")
                self.assertEqual(self.ep.calls, before + 1)
                self.assertEqual(self.ep.handles, [])
                self.assertEqual(self.ep.events[-1]["observation"], observation)

    def test_two_calls_are_rejected_without_execution(self):
        text = call_message("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []})["content"]
        observation = execute_response(self.ep, text + text)
        self.assertFalse(observation["ok"])
        self.assertEqual(self.ep.calls, 1)
        self.assertEqual(self.ep.handles, [])

    def test_format_error_does_not_prevent_recovery(self):
        execute_response(self.ep, "bad json")
        found = self.call("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []})
        counted = self.call("step", {"function": "Count", "inputs": [], "dependencies": [found["handle"]]})
        finished = self.call("finish", {"answer_handle": counted["handle"]})
        self.assertTrue(finished["ok"])
        self.assertEqual(self.ep.prediction, "1")
        self.assertEqual(self.ep.calls, 4)

    def test_last_call_exhausts_budget_without_extra_model_turn(self):
        ep = Episode(SimpleNamespace(engine=Engine()), max_calls=1)
        execute_response(ep, "bad json")
        self.assertTrue(ep.done)
        self.assertEqual(ep.calls, 1)
        self.assertIsNone(ep.prediction)
        execute_response(ep, "bad json")
        self.assertEqual(ep.calls, 1)

    def test_successful_finish_on_last_call_is_retained(self):
        ep = Episode(SimpleNamespace(engine=Engine()), max_calls=3)
        self.call("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []}, ep)
        self.call("step", {"function": "Count", "inputs": [], "dependencies": [0]}, ep)
        finished = self.call("finish", {"answer_handle": 1}, ep)
        self.assertTrue(finished["ok"])
        self.assertEqual(ep.prediction, "1")
        self.assertTrue(ep.done)

    def test_nonterminal_last_call_cannot_be_implicitly_scored(self):
        ep = Episode(SimpleNamespace(engine=Engine()), max_calls=2)
        self.call("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []}, ep)
        self.call("step", {"function": "Count", "inputs": [], "dependencies": [0]}, ep)
        self.assertTrue(ep.done)
        self.assertIsNone(ep.prediction)

    def test_question_file_rejects_gold_and_duplicates(self):
        validate_questions([{"id": "q1", "question": "How many?"}])
        for rows in ([{"id": "q1", "question": "?", "answer": "1"}],
                     [{"id": "q1", "question": "?", "program": []}],
                     [{"id": "q1", "question": "?"}] * 2,
                     [{"id": [], "question": "?"}],
                     [{"id": "q1", "question": None}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_questions(rows)

    def test_low_budget_row_does_not_truncate_other_rows(self):
        args = SimpleNamespace(program=False, max_generated=2048, max_context=8192)
        states = [{"generated_tokens": 2047}, {"generated_tokens": 0}, {"generated_tokens": 0}]
        groups = generation_groups(states, [[1], [1], [1] * 8190], args)
        self.assertEqual(set(groups), {1, 192, 2})
        self.assertIs(groups[192][0][0], states[1])

    def test_program_baseline_receives_full_generation_budget(self):
        args = SimpleNamespace(program=True, max_generated=2048, max_context=8192)
        groups = generation_groups([{"generated_tokens": 0}], [[1]], args)
        self.assertEqual(set(groups), {2048})

    def test_response_token_accounting_retains_eos_but_not_batch_padding(self):
        # A tool tag can itself be a special token. Only EOS/pad are removed
        # from content; tokenizer-wide skip_special_tokens is not appropriate.
        self.assertEqual(response_tokens([10, 42, 11, 99, 0, 0], 99, 0),
                         ([10, 42, 11, 99], [10, 42, 11]))
        self.assertEqual(response_tokens([10, 42, 11, 99, 99], 99, 99),
                         ([10, 42, 11, 99], [10, 42, 11]))
        self.assertEqual(response_tokens([10, 42, 11, 0, 0], 99, 0),
                         ([10, 42, 11], [10, 42, 11]))


class ScoringTest(unittest.TestCase):
    def run_score(self, predictions, gold):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_path, gold_path = Path(tmp) / "predictions.jsonl", Path(tmp) / "gold.jsonl"
            for path, rows in ((prediction_path, predictions), (gold_path, gold)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(predictions=str(prediction_path), gold=str(gold_path))
            with patch("experiments.agent_feedback.inference.compare_answers",
                       side_effect=lambda gold, answer: answer is not None and gold == answer), contextlib.redirect_stdout(io.StringIO()):
                score(args)
            return json.loads(prediction_path.with_suffix(".metrics.json").read_text())

    def prediction(self, id="q1", answer="1"):
        return {"id": id, "prediction": answer, "calls": 4, "input_tokens": 100,
                "generated_tokens": 20, "invalid_calls": 1,
                "stop_reason": "finished" if answer is not None else "call_budget"}

    def test_subset_accuracy_cannot_be_reported_as_full_gold_accuracy(self):
        metrics = self.run_score([self.prediction()],
                                 [{"id": "q1", "answer": "1"}, {"id": "q2", "answer": "2"}])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["evaluation_scope"], "subset")
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["missing_predictions"], 1)
        self.assertEqual(metrics["total"], 1)
        self.assertEqual(metrics["means"]["total_tokens"], 120)

    def test_failed_episodes_remain_in_accuracy_denominator(self):
        metrics = self.run_score([self.prediction(), self.prediction("q2", None)],
                                 [{"id": "q1", "answer": "1"}, {"id": "q2", "answer": "2"}])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["evaluation_scope"], "complete")
        self.assertEqual(metrics["failed_to_finish"], 1)
        self.assertEqual(metrics["stop_reasons"]["call_budget"], 1)

    def test_ambiguous_or_empty_scoring_inputs_are_rejected(self):
        good = [{"id": "q1", "answer": "1"}]
        for predictions, gold in (([], good), ([self.prediction()], []),
                                  ([self.prediction()] * 2, good),
                                  ([self.prediction()], good * 2),
                                  ([self.prediction("unknown")], good)):
            with self.subTest(predictions=predictions, gold=gold), self.assertRaises(ValueError):
                self.run_score(predictions, gold)


if __name__ == "__main__":
    unittest.main()
