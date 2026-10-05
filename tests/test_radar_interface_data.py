"""Contract, separation and counterfactual checks for fictitious interface data."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.radar_domain import interface_data as data
from experiments.radar_domain.development_environment import DevelopmentKB


class RadarInterfaceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = data.build()
        cls.manifest = json.loads(cls.outputs[data.PUBLIC / "manifest.json"])
        cls.groups = json.loads(cls.outputs[data.PUBLIC / "split_assignments.json"])["groups"]

    def rows(self, split, kind):
        return [json.loads(line) for line in self.outputs[data.RAW / f"{split}.{kind}.jsonl"].splitlines()]

    def test_bilingual_source_groups_and_entities_do_not_cross_splits(self):
        sets = {}
        for split, count in (("train", 60), ("dev", 20), ("holdout", 20)):
            groups = [group for group in self.groups if group["split"] == split]
            self.assertEqual(len(groups), count)
            self.assertEqual({group["family"] for group in groups}, set(data.FAMILIES))
            questions = self.rows(split, "questions")
            self.assertEqual(len({row["id"] for row in questions}), count * 2)
            self.assertEqual(set(Counter(row["knowledge_id"] for row in questions).values()), {2})
            sets[split] = ({group["knowledge_id"] for group in groups},
                           {alias for group in groups for alias in group["entity_aliases"]},
                           {group["template_id"] for group in groups})
            for question in questions:
                self.assertEqual(set(question), {"id", "question", "knowledge_id"})
                self.assertNotIn(question["id"], question["question"])
                for marker in ("synfact-", "<func>", "<tool_call>", "expected_fact_ids", "QueryAttr"):
                    self.assertNotIn(marker, question["question"])
        for left, right in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
            for left_set, right_set in zip(sets[left], sets[right]):
                self.assertFalse(left_set & right_set)
            self.assertFalse(set(data.FRAMES[left]) & set(data.FRAMES[right]))

    def test_complete_knowledge_routes_have_real_distractors(self):
        for group in self.groups:
            document = json.loads(self.outputs[data.RAW / "kb" / f"{group['knowledge_id']}.json"])
            records = document["records"]
            self.assertEqual(len({row["entity_name"] for row in records}), 3)
            self.assertEqual(len(records), 32)
            for entity in group["entity_aliases"]:
                subset = [row for row in records if row["entity_name"] == entity]
                self.assertEqual(len({row["attribute"] for row in subset}), 7)
                self.assertEqual({row["condition_raw"] for row in subset if row["attribute"] == "prf"}, {"surface", "air"})
                self.assertEqual({row["event_type"] for row in subset if row["attribute"] == "service_entry"}, {"in_service", "initial_operational_capacity"})
            self.assertEqual(Counter(row["value_kind"] for row in records if row["attribute"] == "power"), {"unknown": 1, "scalar": 2})
        for split in data.SPLIT_COUNTS:
            self.assertEqual({group["target_entity_schema_position"] for group in self.groups if group["split"] == split}, {0, 1, 2})
            kinds = {ref["declarative_intent"]["expected_value_kind"] for ref in self.rows(split, "references") if ref["family"] == "plain"}
            self.assertEqual(kinds, {"text", "scalar"})

    def test_supervision_contains_only_three_successful_actions_and_no_done(self):
        for split in data.SPLIT_COUNTS:
            questions = self.rows(split, "questions")
            agents, programs = self.rows(split, "agent_trajectories"), self.rows(split, "program_trajectories")
            self.assertEqual([row["id"] for row in agents], [row["id"] for row in programs])
            for question, agent, program in zip(questions, agents, programs):
                self.assertEqual(agent["id"], question["id"])
                self.assertEqual(agent["messages"][1]["content"], question["question"])
                self.assertEqual(program["messages"][1]["content"], question["question"])
                self.assertEqual(agent["supervise"], [False, False, True, False, True, False, True, False])
                self.assertEqual(program["supervise"], [False, False, True])
                for trajectory in (agent, program):
                    self.assertEqual(len(trajectory["messages"]), len(trajectory["supervise"]))
                    for message, supervised in zip(trajectory["messages"], trajectory["supervise"]):
                        self.assertEqual(supervised, message["role"] == "assistant")
                        self.assertNotEqual(message["content"], "Done")
                self.assertTrue(all(json.loads(message["content"])["ok"] for message in agent["messages"] if message["role"] == "tool"))

    def test_reference_and_program_match_same_bilingual_declarative_intent(self):
        for split in data.SPLIT_COUNTS:
            refs = self.rows(split, "references")
            for first, second in zip(refs[::2], refs[1::2]):
                self.assertEqual((first["language"], second["language"]), ("zh", "en"))
                for key in ("knowledge_id", "declarative_intent", "expected_fact_ids", "canonical_program"):
                    self.assertEqual(first[key], second[key])
                self.assertEqual(first["canonical_program"], data.canonical_program(first["declarative_intent"]))
                document = json.loads(self.outputs[data.RAW / "kb" / f"{first['knowledge_id']}.json"])
                selected = data.semantic_select(document["records"], first["declarative_intent"])
                self.assertEqual([row["fact_id"] for row in selected], first["expected_fact_ids"])

    def test_original_answer_exclusion_rejects_equal_decimal_variants(self):
        original = json.loads(data.ORIGINAL.read_bytes())["records"]
        record = deepcopy(json.loads(self.outputs[next(key for key in self.outputs if key.parent == data.RAW / "kb")])["records"][0])
        record.update(value="0.3000", min_value=None, max_value=None)
        with self.assertRaisesRegex(ValueError, "numeric answer literal"):
            data.audit_exclusion(original, [record], [])
        record["value"] = "43210"
        for forbidden in (original[0]["entity_name"], original[0]["citation"]["source_uri"], original[0]["value"]):
            with self.assertRaisesRegex(ValueError, "leaked"):
                data.audit_exclusion(original, [record], [forbidden])

    def test_independent_intent_and_source_checks_reject_corruption(self):
        spec = data.split_plan()[0]
        document, source, intent, _ = data.make_instance(spec)
        with tempfile.TemporaryDirectory() as temp:
            kb_file = Path(temp) / "kb.json"
            kb_file.write_bytes(data.encoded(document))
            kb = DevelopmentKB(kb_file)
            wrong = {**intent, "expected_value_kind": "not_a_value_kind"}
            with self.assertRaisesRegex(ValueError, "wrong value kind"):
                data.validate_instance(kb, source, wrong)
            broken = deepcopy(document)
            broken["records"][0]["citation"]["evidence_text"] = "fabricated trace text"
            kb_file.write_bytes(data.encoded(broken))
            with self.assertRaisesRegex(ValueError, "actual source entry"):
                data.validate_instance(DevelopmentKB(kb_file), source, intent)
            altered = deepcopy(document)
            next(row for row in altered["records"] if row["value_kind"] == "scalar")["value"] = "43121"
            kb_file.write_bytes(data.encoded(altered))
            with self.assertRaisesRegex(ValueError, "query fields differ"):
                data.validate_instance(DevelopmentKB(kb_file), source, intent)

    def test_unknown_record_and_empty_query_remain_distinct(self):
        spec = next(item for item in data.split_plan() if item["family"] == "unknown")
        document, source, intent, _ = data.make_instance(spec)
        with tempfile.TemporaryDirectory() as temp:
            kb_file = Path(temp) / "kb.json"
            kb_file.write_bytes(data.encoded(document))
            selected = data.validate_instance(DevelopmentKB(kb_file), source, intent)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["unknown_scope"], "bound_source_field")
        self.assertTrue(all(selected[0][key] is None for key in ("value", "min_value", "max_value", "unit_std")))


if __name__ == "__main__":
    unittest.main()
