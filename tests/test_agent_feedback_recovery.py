"""Recovery-pair causality, condition preservation and handle remapping tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback.environment import Episode, call_message, compact, prompt_messages
from experiments.agent_feedback.inference import execute_response
from experiments.agent_feedback.recovery import make_pair, sha256, validate_training_sources


class Engine:
    def __init__(self):
        self.entities = {id: {"name": name, "attributes": [{"key": "power"}], "relations": []}
                         for id, name in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"))}
        self.concepts = {}

    def Find(self, deps, inputs):
        return ([id for id, entity in self.entities.items() if entity["name"] == inputs[0]], None)

    def FindAll(self, deps, inputs):
        return (list(self.entities), None)

    def Count(self, deps, inputs):
        return len(deps[0][0])

    def Or(self, deps, inputs):
        return (sorted(set(deps[0][0]) | set(deps[1][0])), None)

    def FilterNum(self, deps, inputs):
        if inputs != ["power", "5 watt", ">"]:
            raise ValueError("Unexpected numeric condition")
        ids = [id for id in deps[0][0] if id in ("a", "b")]
        return ids, [{"qualifiers": {"mode": ["tracking" if id == "a" else "search"]}} for id in ids]

    def QFilterStr(self, deps, inputs):
        if inputs != ["mode", "tracking"]:
            raise ValueError("Unexpected qualifier condition")
        pairs = [(id, fact) for id, fact in zip(*deps[0])
                 if fact["qualifiers"]["mode"] == ["tracking"]]
        return [id for id, _ in pairs], [fact for _, fact in pairs]


def step(function, inputs=None, dependencies=None):
    return call_message("step", {"function": function, "inputs": inputs or [],
                                 "dependencies": dependencies or []})["content"]


def finish(handle):
    return call_message("finish", {"answer_handle": handle})["content"]


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(engine=Engine())
        self.compare = patch("experiments.agent_feedback.recovery.compare_answers",
                             side_effect=lambda gold, prediction: prediction is not None and str(gold) == prediction)
        self.compare.start()
        self.addCleanup(self.compare.stop)
        self.gold = {"id": "train:1", "question": "How many Alpha or Beta entities?", "answer": "2",
                     "program": [{"function": "Find", "inputs": ["Alpha"], "dependencies": []},
                                 {"function": "Find", "inputs": ["Beta"], "dependencies": []},
                                 {"function": "Or", "inputs": [], "dependencies": [0, 1]},
                                 {"function": "Count", "inputs": [], "dependencies": [2]}]}

    def rollout(self, actions, gold=None, max_calls=24):
        gold = gold or self.gold
        episode = Episode(self.executor, max_calls=max_calls)
        messages = prompt_messages(gold["question"])
        for text in actions:
            observation = execute_response(episode, text)
            messages.extend([{"role": "assistant", "content": text},
                             {"role": "tool", "content": compact(observation)}])
        return {"id": gold["id"], "messages": messages, "events": episode.events,
                "calls": episode.calls, "prediction": episode.prediction}

    def assert_pair_replays(self, pair):
        self.assertEqual(pair["audit"]["status"], "accepted")
        for name in ("clean", "recovery"):
            row = pair[name]
            episode = Episode(self.executor)
            for index in range(2, len(row["messages"]) - 1, 2):
                observed = execute_response(episode, row["messages"][index]["content"])
                self.assertEqual(observed, json.loads(row["messages"][index + 1]["content"]))
            self.assertEqual(episode.prediction, row["prediction"])
            self.assertTrue(episode.done)
            self.assertEqual(len(row["messages"]), len(row["supervise"]))
            self.assertTrue(row["supervise"][-1])
            self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "Done."})
            for message, supervised in zip(row["messages"], row["supervise"]):
                if message["role"] != "assistant":
                    self.assertFalse(supervised)

    def test_executable_extra_handle_remaps_branched_suffix(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Find", ["Beta"]),
                               step("Find", ["Gamma"]), step("Count", dependencies=[2]), finish(3)])
        self.assertEqual(source["prediction"], "1")
        pair = make_pair(self.executor, source, self.gold)
        self.assert_pair_replays(pair)
        self.assertEqual(pair["audit"]["shared_prefix_calls"], 2)
        self.assertTrue(pair["audit"]["divergence_executable"])
        self.assertEqual(pair["audit"]["semantic_error_at_divergence"], "not_established")
        recovery, clean = pair["recovery"], pair["clean"]
        self.assertEqual(recovery["reference_handle_map"], {"0": 0, "1": 1, "2": 3, "3": 4})
        self.assertEqual(recovery["events"][3]["arguments"]["dependencies"], [0, 1])
        self.assertEqual(recovery["events"][4]["arguments"]["dependencies"], [3])
        self.assertEqual(recovery["events"][5]["arguments"], {"answer_handle": 4})
        self.assertEqual(recovery["messages"][:6], clean["messages"][:6])
        self.assertFalse(recovery["supervise"][6])  # real divergent action
        self.assertTrue(recovery["supervise"][8])   # first corrected suffix action
        self.assertEqual(sum(recovery["supervise"]), sum(clean["supervise"]))

    def test_rejected_action_has_no_extra_handle(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Count", dependencies=[99])])
        pair = make_pair(self.executor, source, self.gold)
        self.assert_pair_replays(pair)
        self.assertFalse(pair["audit"]["divergence_executable"])
        self.assertEqual(pair["recovery"]["reference_handle_map"], {"0": 0, "1": 1, "2": 2, "3": 3})
        self.assertEqual(pair["recovery"]["calls"], pair["clean"]["calls"] + 1)

    def test_raw_format_error_is_retained_as_masked_context(self):
        malformed = '<tool_call>{"name":[],"arguments":{}}</tool_call>'
        source = self.rollout([step("Find", ["Alpha"]), malformed])
        pair = make_pair(self.executor, source, self.gold)
        self.assert_pair_replays(pair)
        self.assertEqual(pair["audit"]["divergence_error"], "ActionFormatError")
        self.assertEqual(pair["recovery"]["messages"][4]["content"], malformed)
        self.assertFalse(pair["recovery"]["supervise"][4])

    def test_empty_result_is_valid_divergence_without_semantic_error_label(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Find", ["Missing"])])
        pair = make_pair(self.executor, source, self.gold)
        self.assert_pair_replays(pair)
        self.assertTrue(pair["audit"]["divergence_executable"])
        self.assertTrue(pair["audit"]["divergence_empty_entities"])
        self.assertEqual(pair["audit"]["semantic_error_at_divergence"], "not_established")

    def test_numeric_units_and_qualifier_conditions_are_preserved(self):
        gold = {"id": "train:2", "question": "How many entities exceed 5 watt in tracking mode?", "answer": "1",
                "program": [{"function": "FindAll", "inputs": [], "dependencies": []},
                            {"function": "FilterNum", "inputs": ["power", "5 watt", ">"], "dependencies": [0]},
                            {"function": "QFilterStr", "inputs": ["mode", "tracking"], "dependencies": [1]},
                            {"function": "Count", "inputs": [], "dependencies": [2]}]}
        source = self.rollout([step("FindAll"), step("Find", ["Missing"])], gold=gold)
        pair = make_pair(self.executor, source, gold)
        self.assert_pair_replays(pair)
        suffix = pair["recovery"]["events"][2:]
        self.assertEqual(suffix[0]["arguments"], {"function": "FilterNum", "inputs": ["power", "5 watt", ">"], "dependencies": [0]})
        self.assertEqual(suffix[1]["arguments"], {"function": "QFilterStr", "inputs": ["mode", "tracking"], "dependencies": [2]})
        self.assertEqual(suffix[2]["arguments"]["dependencies"], [3])
        self.assertEqual(suffix[3]["arguments"]["answer_handle"], 4)

    def test_successful_terminal_finish_is_never_reopened(self):
        gold = {"id": "train:3", "question": "How many entities?", "answer": "3",
                "program": [{"function": "Find", "inputs": ["Alpha"], "dependencies": []},
                            {"function": "Count", "inputs": [], "dependencies": [0]},
                            {"function": "FindAll", "inputs": [], "dependencies": []},
                            {"function": "Count", "inputs": [], "dependencies": [2]}]}
        source = self.rollout([step("Find", ["Alpha"]), step("Count", dependencies=[0]), finish(1)], gold=gold)
        pair = make_pair(self.executor, source, gold)
        self.assertEqual(pair["audit"]["reason"], "divergence_terminated_episode")
        self.assertNotIn("recovery", pair)

    def test_correct_free_run_is_not_selected_even_with_other_program(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Find", ["Beta"]),
                               step("Find", ["Gamma"]), step("Or", dependencies=[0, 1]),
                               step("Count", dependencies=[3]), finish(4)])
        pair = make_pair(self.executor, source, self.gold)
        self.assertEqual(pair["audit"]["reason"], "free_run_answer_correct")
        self.assertFalse(pair["audit"]["free_run_failed"])

    def test_divergence_requires_nonempty_shared_successful_prefix(self):
        source = self.rollout([step("Find", ["Gamma"])])
        pair = make_pair(self.executor, source, self.gold)
        self.assertEqual(pair["audit"]["reason"], "no_shared_successful_prefix")

    def test_stopping_mid_reference_is_not_a_fabricated_error_action(self):
        source = self.rollout([step("Find", ["Alpha"]), step("Find", ["Beta"])])
        pair = make_pair(self.executor, source, self.gold)
        self.assertEqual(pair["audit"]["reason"], "no_reference_divergence_before_stop")

    def test_call_budget_is_not_silently_expanded(self):
        source = self.rollout([step("Find", ["Alpha"]), "bad action"], max_calls=5)
        pair = make_pair(self.executor, source, self.gold, max_calls=5)
        self.assertEqual(pair["audit"]["reason"], "insufficient_call_budget")

    def test_recorded_observation_and_prediction_must_replay(self):
        source = self.rollout([step("Find", ["Alpha"]), "bad action"])
        for field in ("prediction", "observation", "event"):
            changed = deepcopy(source)
            if field == "prediction":
                changed["prediction"] = "made up"
            elif field == "observation":
                changed["messages"][3]["content"] = '{"ok":true,"handle":99}'
            else:
                changed["events"][0]["arguments"]["inputs"] = ["Gamma"]
            with self.subTest(field=field):
                pair = make_pair(self.executor, changed, self.gold)
                self.assertEqual(pair["audit"]["status"], "excluded")
                self.assertIn(pair["audit"]["reason"], ("rollout_prediction_mismatch", "rollout_replay_mismatch"))

    def test_contradictory_reference_is_not_used_for_training(self):
        source = self.rollout([step("Find", ["Alpha"]), "bad action"])
        gold = {**self.gold, "answer": "99"}
        pair = make_pair(self.executor, source, gold)
        self.assertEqual(pair["audit"]["reason"], "reference_answer_mismatch")


class TrainingSourceTest(unittest.TestCase):
    def test_frozen_training_ids_and_file_hash_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.gold.jsonl"
            gold = [{"id": "train:1", "question": "?", "answer": "1", "program": []}]
            path.write_text(json.dumps(gold[0]) + "\n")
            manifest = {"splits": {"train": {"ids": ["train:1"], "gold": {"sha256": sha256(path)}},
                                     "dev": {"ids": ["train:2"]}, "holdout": {"ids": ["train:3"]}}}
            self.assertEqual(set(validate_training_sources(gold, [{"id": "train:1"}], manifest, path)), {"train:1"})
            for id in ("train:2", "train:3", "unknown"):
                with self.subTest(id=id), self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_training_sources(gold, [{"id": id}], manifest, path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_training_sources(gold, [{"id": "train:1"}] * 2, manifest, path)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                validate_training_sources(gold, [{"id": "train:1"}], manifest, path)


if __name__ == "__main__":
    unittest.main()
