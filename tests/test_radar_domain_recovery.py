from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.agent_feedback.recovery import sha256
from experiments.radar_domain.record_set_interface import RecordSetEpisode, load_bridge, prompt_messages, execute_response
from experiments.radar_domain.kopl_record_bridge import compact, prototype_queries
from experiments.radar_domain.recovery_data import (
    action_text, answers_equal, make_pair, successful_trajectory, admit_training_inputs)


def fixture_rollout(executor, store, reference, actions, max_calls=24):
    """Deliberately constructed CPU trace, never a model rollout."""
    episode = RecordSetEpisode(executor, store, max_calls)
    messages = prompt_messages(reference["question"])
    for action in actions:
        text = action if isinstance(action, str) else action_text(action)
        observation = execute_response(episode, text)
        messages.extend([{"role": "assistant", "content": text},
                         {"role": "tool", "content": compact(observation)}])
    return {"id": reference["id"], "messages": messages, "events": deepcopy(episode.events),
            "calls": episode.calls, "answer": deepcopy(episode.answer),
            "provenance": {"kind": "hand_constructed_cpu_fixture"}, "training_eligible": False}


def extra_action():
    return {"name": "step", "arguments": {"function": "Find", "inputs": ["WRS300"], "dependencies": []}}


def rejected_action():
    return {"name": "step", "arguments": {"function": "FilterStr", "inputs": ["attribute", "current_rating"], "dependencies": [999]}}


class DomainRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor, cls.store = load_bridge()

    def reference(self, scalar=False, empty=False):
        program = deepcopy(prototype_queries()[1]["program"])
        if empty:
            program[-2]["inputs"][1] = "99 A"
        if not scalar:
            program = program[:-1]
        actions = [{"name": "step", "arguments": p} for p in program]
        actions.append({"name": "finish" if scalar else "finish_records",
                        "arguments": {"answer_handle" if scalar else "record_handle": len(program)-1}})
        reference = {"id": "cpu-reference", "question": "CPU接口测试：查询Cyclone Pro电流读法。", "actions": actions, "answer": None}
        reference["answer"] = fixture_rollout(self.executor, self.store, reference, actions)["answer"]
        return reference

    def failure(self, reference, rejected=False, max_calls=24):
        actions = reference["actions"][:2] + [rejected_action() if rejected else extra_action(),
            {"name": "finish_records", "arguments": {"record_handle": 1}}]
        return fixture_rollout(self.executor, self.store, reference, actions, max_calls)

    def test_success_trajectory_masks_tools_and_keeps_complete_answer(self):
        reference = self.reference()
        row = successful_trajectory(self.executor, self.store, reference)
        self.assertTrue(answers_equal(row["answer"], reference["answer"]))
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "Done."})
        self.assertEqual(row["supervise"], [m["role"] == "assistant" for m in row["messages"]])
        self.assertFalse(row["training_eligible"])

    def test_extra_handle_recovery_and_prefix_loss_masks(self):
        reference = self.reference()
        pair = make_pair(self.executor, self.store, self.failure(reference), reference)
        self.assertEqual(pair["audit"]["status"], "accepted")
        clean, recovery = pair["clean"], pair["recovery"]
        self.assertEqual(clean["reference_handle_map"], {"0": 0, "1": 1, "2": 2, "3": 3})
        self.assertEqual(recovery["reference_handle_map"], {"0": 0, "1": 1, "2": 3, "3": 4})
        self.assertEqual(recovery["events"][-1]["arguments"], {"record_handle": 4})
        self.assertEqual(clean["calls"] + 1, recovery["calls"])
        self.assertTrue(answers_equal(clean["answer"], recovery["answer"]))
        self.assertTrue(answers_equal(recovery["answer"], reference["answer"]))
        for name, prefix_calls in (("clean", 2), ("recovery", 3)):
            row = pair[name]
            self.assertFalse(any(row["supervise"][:2 + 2*prefix_calls]))
            self.assertTrue(all(not flag for m, flag in zip(row["messages"], row["supervise"]) if m["role"] == "tool"))
            self.assertEqual(sum(row["supervise"]), len(reference["actions"]) - 2 + 1)
            self.assertFalse(row["training_eligible"])

    def test_rejected_divergence_retains_original_handle_mapping(self):
        reference = self.reference()
        pair = make_pair(self.executor, self.store, self.failure(reference, rejected=True), reference)
        self.assertEqual(pair["audit"]["status"], "accepted")
        self.assertFalse(pair["audit"]["divergence_executable"])
        self.assertEqual(pair["clean"]["reference_handle_map"], pair["recovery"]["reference_handle_map"])
        self.assertFalse(pair["recovery"]["events"][2]["observation"]["ok"])

    def test_empty_set_zero_and_positive_scalar_recover(self):
        for scalar, empty in ((False, True), (True, True), (True, False)):
            with self.subTest(scalar=scalar, empty=empty):
                reference = self.reference(scalar, empty)
                pair = make_pair(self.executor, self.store, self.failure(reference), reference)
                self.assertEqual(pair["audit"]["status"], "accepted")
                result = pair["recovery"]["answer"]
                self.assertTrue(answers_equal(result, reference["answer"]))
                if scalar:
                    self.assertEqual(result["value"]["value"], 0 if empty else 2)
                    self.assertEqual(pair["recovery"]["events"][-1]["tool"], "finish")
                else:
                    self.assertEqual(result["records"], [])
                self.assertFalse(answers_equal(None, result))

    def test_successful_and_equivalent_alternative_trajectory_excluded(self):
        reference = self.reference()
        for actions in (reference["actions"], reference["actions"][:-1] + [
            {"name": "step", "arguments": {"function": "Or", "inputs": [], "dependencies": [3, 3]}},
            {"name": "finish_records", "arguments": {"record_handle": 4}}]):
            rollout = fixture_rollout(self.executor, self.store, reference, actions)
            pair = make_pair(self.executor, self.store, rollout, reference)
            self.assertEqual(pair["audit"]["reason"], "free_run_answer_correct")

    def test_tool_message_event_and_answer_tampering_rejected(self):
        reference = self.reference()
        base = self.failure(reference, rejected=True)
        for field in ("tool", "event", "answer"):
            with self.subTest(field=field):
                bad = deepcopy(base)
                if field == "tool":
                    obj = json.loads(bad["messages"][7]["content"])
                    obj["detail"] = "changed error message"
                    bad["messages"][7]["content"] = compact(obj)
                elif field == "event":
                    bad["events"][0]["observation"]["count"] = 999
                else:
                    bad["answer"]["records"][0]["value"]["text"] = "wrong"
                pair = make_pair(self.executor, self.store, bad, reference)
                self.assertEqual(pair["audit"]["status"], "excluded")
                self.assertIn(pair["audit"]["reason"], {"rollout_replay_mismatch", "rollout_answer_or_call_mismatch"})

    def test_insufficient_budget_and_terminal_divergence_excluded(self):
        reference = self.reference()
        pair = make_pair(self.executor, self.store, self.failure(reference, max_calls=5), reference, max_calls=5)
        self.assertEqual(pair["audit"]["reason"], "insufficient_call_budget")
        actions = reference["actions"][:2] + [{"name": "finish_records", "arguments": {"record_handle": 1}}]
        pair = make_pair(self.executor, self.store, fixture_rollout(self.executor, self.store, reference, actions), reference)
        self.assertEqual(pair["audit"]["reason"], "divergence_terminated_episode")

    def test_structured_comparison_keeps_all_records_units_conditions_and_sources(self):
        answer = self.reference()["answer"]
        reordered = deepcopy(answer); reordered["records"].reverse()
        self.assertTrue(answers_equal(answer, reordered))
        for field in ("missing_record", "unit", "condition", "source"):
            bad = deepcopy(answer)
            if field == "missing_record":
                bad["records"].pop(); bad["unique_record_count"] -= 1
            elif field == "unit":
                bad["records"][1]["value"]["unit_raw"] = "mA"
            elif field == "condition":
                bad["records"][1]["conditions_all"][0]["value"] = "24 V"
            else:
                next(iter(bad["sources"].values()))["source_uri"] = "https://wrong.example"
            self.assertFalse(answers_equal(answer, bad), field)
        scalar = self.reference(scalar=True)["answer"]
        changed = deepcopy(scalar); changed["value"]["value"] = 3
        self.assertFalse(answers_equal(scalar, changed))

    def test_subject_and_conjunction_order_can_change_without_losing_content(self):
        records, sources = self.store.materialize(["reading:WX-0051", "reading:marine-0001"])
        answer = {"kind": "records", "records": records, "sources": sources,
                  "input_entity_count": 2, "unique_record_count": 2,
                  "scope": "selected_readings_in_current_graph"}
        reordered = deepcopy(answer)
        reordered["records"].reverse()
        for row in reordered["records"]:
            row["subjects"].reverse()
            row["conditions_all"].reverse()
            row["citation"]["spans"].reverse()
        self.assertTrue(answers_equal(answer, reordered))
        # Execution diagnostics may differ while the selected set is identical.
        reordered["input_entity_count"] = 3
        self.assertTrue(answers_equal(answer, reordered))

    def test_development_and_cpu_origin_rejected_by_production_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); split_path = root/"split.json"; gen_path = root/"generation.json"
            split_path.write_text(json.dumps({"schema": "radar_domain_split_manifest_v1"}))
            for origin, split in (("hand_constructed_cpu_fixture", "train"), ("model_generated", "dev")):
                generation = {"schema": "radar_domain_rollout_manifest_v1", "status": "completed",
                              "origin": origin, "split": split, "max_calls": 24,
                              "split_manifest_sha256": sha256(split_path)}
                gen_path.write_text(json.dumps(generation))
                with self.assertRaisesRegex(ValueError, "model-generated training inventory"):
                    admit_training_inputs(root/"refs.jsonl", root/"rollouts.jsonl", split_path, gen_path)

    def test_train_development_id_and_source_group_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); split_path = root/"split.json"; gen_path = root/"generation.json"
            for overlap in ("id", "group"):
                split = {"schema": "radar_domain_split_manifest_v1", "splits": {
                    name: {"ids": [name], "group_by_id": {name: name}} for name in ("train", "dev", "holdout")}}
                if overlap == "id":
                    split["splits"]["dev"] = {"ids": ["train"], "group_by_id": {"train": "dev"}}
                else:
                    split["splits"]["dev"]["group_by_id"]["dev"] = "train"
                split_path.write_text(json.dumps(split))
                gen_path.write_text(json.dumps({"schema": "radar_domain_rollout_manifest_v1",
                    "status": "completed", "origin": "model_generated", "split": "train",
                    "max_calls": 24, "split_manifest_sha256": sha256(split_path)}))
                with self.assertRaisesRegex(ValueError, "overlap"):
                    admit_training_inputs(root/"refs.jsonl", root/"rollouts.jsonl", split_path, gen_path)


if __name__ == "__main__":
    unittest.main()
