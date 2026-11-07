from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.radar_domain.table_program import TableExecutor, TableEpisode, execute_response, prompt_messages
from experiments.radar_domain.table_recovery_data import (
    action_text, make_pair, success, answers_equal, admit_model_training_rollouts)


def fixture(executor, reference, kind):
    episode = TableEpisode(executor)
    messages = prompt_messages(reference["question"])
    prefix = reference["actions"][0]
    divergent = deepcopy(prefix) if kind == "extra_handle" else {
        "name": "step", "arguments": {"function": "__unsupported__", "inputs": ["{}"], "dependencies": [0]}}
    actions = [prefix, divergent, {"name": "finish", "arguments": {"answer_handle": -1}}]
    for item in actions:
        text = action_text(item)
        result = execute_response(episode, text)
        messages.extend([{"role": "assistant", "content": text},
                         {"role": "tool", "content": json.dumps(result, ensure_ascii=False, sort_keys=True)}])
    return {"id": reference["id"], "messages": messages, "events": episode.events,
            "calls": episode.calls, "answer": episode.answer,
            "provenance": {"kind": "hand_constructed_cpu_fixture"}, "training_eligible": False}


class TableRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path("data/radar_domain_recovery_v2/frozen_20261107/dev")
        cls.executor = TableExecutor(base / "catalog.json")
        cls.references = [json.loads(s) for s in (base / "references.jsonl").read_text().splitlines()
                          if json.loads(s)["id"].startswith("RDR-DEV-")]

    def test_twelve_complete_tasks_recover_with_both_handle_cases(self):
        for ref in self.references:
            for kind in ("extra_handle", "rejected_action"):
                pair = make_pair(self.executor, ref, fixture(self.executor, ref, kind))
                self.assertEqual(pair["audit"]["status"], "accepted", (ref["id"], kind, pair["audit"]))
                self.assertTrue(answers_equal(pair["clean"]["answer"], pair["recovery"]["answer"]))
                self.assertEqual(pair["recovery"]["calls"], pair["clean"]["calls"] + 1)
                self.assertFalse(pair["audit"]["training_eligible"])
                self.assertFalse(pair["recovery"]["supervise"][2])
                self.assertFalse(pair["recovery"]["supervise"][4])
                for m, selected in zip(pair["recovery"]["messages"], pair["recovery"]["supervise"]):
                    if selected:
                        self.assertEqual(m["role"], "assistant")

    def test_json_object_formatting_does_not_remove_valid_prefix(self):
        ref = self.references[0]
        row = fixture(self.executor, ref, "extra_handle")
        call = deepcopy(ref["actions"][0])
        call["arguments"]["inputs"] = [json.dumps(json.loads(call["arguments"]["inputs"][0]), indent=2)]
        row["messages"][2]["content"] = action_text(call)
        row["events"][0]["arguments"] = call["arguments"]
        pair = make_pair(self.executor, ref, row)
        self.assertEqual(pair["audit"]["status"], "accepted")
        self.assertEqual(pair["audit"]["shared_prefix"], 1)

    def test_changed_observation_error_or_answer_rejected(self):
        ref = self.references[0]
        for kind in ("extra_handle", "rejected_action"):
            row = fixture(self.executor, ref, kind)
            bad = deepcopy(row); bad["events"][1]["observation"]["ok"] = not bad["events"][1]["observation"]["ok"]
            self.assertEqual(make_pair(self.executor, ref, bad)["audit"]["reason"], "rollout_replay_mismatch")
            bad = deepcopy(row); bad["answer"] = ref["answer"]
            self.assertEqual(make_pair(self.executor, ref, bad)["audit"]["reason"], "rollout_answer_mismatch")

    def test_budget_limit_and_successful_rollout_excluded(self):
        ref = self.references[0]
        row = fixture(self.executor, ref, "extra_handle")
        pair = make_pair(self.executor, ref, row, max_calls=len(ref["actions"]))
        self.assertEqual(pair["audit"]["reason"], "insufficient_call_budget")
        row = success(self.executor, ref)
        pair = make_pair(self.executor, ref, row)
        self.assertEqual(pair["audit"]["reason"], "already_correct")

    def test_training_gate_rejects_constructed_origin(self):
        split = Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            path.write_text(json.dumps({"schema": "radar_table_rollout_manifest_v1",
                "status": "completed", "origin": "hand_constructed_cpu_fixture", "split": "train"}))
            with self.assertRaises(ValueError):
                admit_model_training_rollouts(split, path)


if __name__ == "__main__":
    unittest.main()
