from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback.construction_diagnostic import diagnose_pair
from experiments.agent_feedback.environment import Episode, call_message, compact, prompt_messages
from experiments.agent_feedback.inference import execute_response
from experiments.agent_feedback.recovery import make_pair


class Engine:
    def __init__(self):
        self.entities = {"a": {"name": "Alpha", "attributes": [], "relations": []},
                         "b": {"name": "Beta", "attributes": [], "relations": []},
                         "c": {"name": "Gamma", "attributes": [], "relations": []}}
        self.concepts = {}

    def Find(self, dependencies, inputs):
        return [k for k, v in self.entities.items() if v["name"] == inputs[0]], None

    def Count(self, dependencies, inputs):
        return len(dependencies[0][0])

    def What(self, dependencies, inputs):
        return self.entities[dependencies[0][0][0]]["name"]

    def Or(self, dependencies, inputs):
        return sorted(set(dependencies[0][0]) | set(dependencies[1][0])), None


def step(fn, values=(), deps=()):
    return call_message("step", {"function": fn, "inputs": list(values), "dependencies": list(deps)})["content"]


class ConstructionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(engine=Engine())
        comparator = lambda a, p: p is not None and str(a) == p
        for module in ("experiments.agent_feedback.recovery", "experiments.agent_feedback.construction_diagnostic"):
            p = patch(module + ".compare_answers", side_effect=comparator); p.start(); self.addCleanup(p.stop)
        self.gold = {"id": "fixture", "question": "How many Alpha or Beta?", "answer": "2", "program": [
            {"function": "Find", "inputs": ["Alpha"], "dependencies": []},
            {"function": "Find", "inputs": ["Beta"], "dependencies": []},
            {"function": "Or", "inputs": [], "dependencies": [0, 1]},
            {"function": "Count", "inputs": [], "dependencies": [2]}]}

    def pair(self, actions):
        episode = Episode(self.executor)
        messages = prompt_messages(self.gold["question"])
        for text in actions:
            observation = execute_response(episode, text)
            messages.extend([{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}])
        source = {"id": "fixture", "messages": messages, "events": episode.events,
                  "calls": episode.calls, "prediction": episode.prediction}
        pair = make_pair(self.executor, source, self.gold)
        self.assertEqual(pair["audit"]["status"], "accepted")
        return pair

    def diagnose(self, pair):
        return diagnose_pair(self.executor, pair["recovery"], pair["clean"], self.gold)

    def test_extra_handle_exposes_direct_suffix_dependency_shift(self):
        pair = self.pair([step("Find", ["Alpha"]), step("Find", ["Beta"]), step("Find", ["Gamma"])])
        row = self.diagnose(pair)
        self.assertEqual(row["status"], "compared")
        self.assertTrue(row["divergence_added_handle"])
        self.assertFalse(row["direct"]["final_answer_correct"])
        self.assertTrue(row["direct"]["finish_dependency_mismatch"])
        self.assertGreater(row["direct"]["step_dependency_mismatch_count"], 0)
        self.assertTrue(row["mapped_unfiltered"]["full_gate_pass"])
        self.assertTrue(row["mapped_verified"]["same_sequence_as_unfiltered"])
        self.assertTrue(row["mapped_verified"]["same_sequence_as_saved_recovery"])

    def test_rejected_divergence_allocates_no_handle_and_both_paths_agree(self):
        row = self.diagnose(self.pair([step("Find", ["Alpha"]), step("Count", deps=[99])]))
        self.assertFalse(row["divergence_added_handle"])
        self.assertTrue(row["direct"]["full_gate_pass"])
        self.assertEqual(row["direct"]["reference_dependency_mismatch_count"], 0)
        self.assertEqual(row["direct"]["sequence_sha256"], row["mapped_unfiltered"]["sequence_sha256"])

    def test_equal_final_answer_does_not_hide_wrong_result_origin(self):
        # What returns the label "1"; the target Count returns integer 1.
        # Bare finish selects the irrelevant label and matches the answer text.
        self.executor.engine.entities["a"]["name"] = "1"
        self.gold = {"id": "fixture", "question": "How many entities named 1?", "answer": "1", "program": [
            {"function": "Find", "inputs": ["1"], "dependencies": []},
            {"function": "Count", "inputs": [], "dependencies": [0]}]}
        pair = self.pair([step("Find", ["1"]), step("What", deps=[0]),
                          call_message("finish", {"answer_handle": 0})["content"]])
        row = self.diagnose(pair)
        self.assertEqual(row["status"], "compared")
        self.assertTrue(row["direct"]["final_answer_correct"])
        self.assertTrue(row["direct"]["reference_observations_match"])
        self.assertTrue(row["direct"]["finish_dependency_mismatch"])
        self.assertEqual(row["direct"]["full_internal_result_mismatch_count"], 1)
        self.assertFalse(row["direct"]["full_gate_pass"])
        self.assertTrue(row["mapped_verified"]["accepted"])

    def test_tampered_retained_observation_is_preserved_as_unavailable(self):
        pair = self.pair([step("Find", ["Alpha"]), step("Find", ["Gamma"])])
        pair["recovery"]["messages"][3]["content"] = '{"ok":false}'
        row = self.diagnose(pair)
        self.assertEqual(row["status"], "unavailable")
        self.assertEqual(row["reason"], "retained_history_replay_mismatch")

    def test_historical_later_observation_mismatch_is_reported_not_filtered(self):
        pair = self.pair([step("Find", ["Alpha"]), step("Find", ["Gamma"])])
        pair["recovery"]["events"][-1]["observation"]["answer"] = "changed"
        row = self.diagnose(pair)
        self.assertEqual(row["status"], "compared")
        self.assertFalse(row["historical_recovery_replay"]["matches"])
        self.assertFalse(row["mapped_verified"]["same_sequence_as_saved_recovery"])
        self.assertTrue(row["mapped_unfiltered"]["final_answer_correct"])


if __name__ == "__main__":
    unittest.main()
