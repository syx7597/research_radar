"""Source semantics and state-machine checks for the new development interface."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.agent_feedback.environment import call_message
from experiments.agent_feedback.inference import execute_response
from experiments.radar_domain.development_environment import (
    DevelopmentEpisode, DevelopmentKB, execute_program, prompt_messages, render_records,
)

ROOT = Path(__file__).resolve().parents[1]
READINGS = ROOT / "artifacts/thesis_direction_review/radar_development_v2/readings.json"


class RadarDevelopmentEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = DevelopmentKB(READINGS)

    def query(self, entity, attribute, qualifier=None, value=None):
        episode = DevelopmentEpisode(self.kb)
        self.assertTrue(episode.step("Find", [entity], [])["ok"])
        fn = "QueryAttr" if qualifier is None else "QueryAttrUnderCondition"
        args = [attribute] if qualifier is None else [attribute, qualifier, value]
        result = episode.step(fn, args, [0])
        self.assertTrue(result["ok"])
        self.assertTrue(episode.finish(1)["ok"])
        return episode.prediction

    def test_range_and_exact_condition_do_not_collapse(self):
        both = self.query("AN/SPG-51", "prf")
        self.assertEqual({r["condition_raw"] for r in both}, {"surface", "air"})
        air = self.query("AN/SPG-51", "prf", "condition_raw", "air")
        self.assertEqual(len(air), 1)
        self.assertEqual((air[0]["min_value"], air[0]["max_value"], air[0]["value"]), ("9600", "16700", None))
        self.assertEqual(self.query("AN/SPG-51", "prf", "condition_raw", "Air"), [])
        self.assertEqual(self.query("AN/SPG-51", "prf", "condition_raw", "not_stated"), [])

    def test_system_subject_and_quarantine_survive_query_and_render(self):
        rows = self.query("AN/MPQ-65", "type_description")
        self.assertEqual(rows[0]["source_subject"], "MIM-104 Patriot system")
        self.assertFalse(rows[0]["original_entity_attribution_usable"])
        rendered = render_records(rows)
        self.assertIn("原始设备归属断言已隔离", rendered)
        self.assertIn("非模型生成回答", rendered)
        self.assertIn("MIM-104 Patriot system", rendered)
        self.assertIn(rows[0]["citation"]["source_sha256"], rendered)

    def test_cancellation_is_not_service_and_events_remain_separate(self):
        self.assertEqual(self.query("AN/SPG-59", "service_entry"), [])
        cancel = self.query("AN/SPG-59", "lifecycle_event", "event_type", "cancellation")
        self.assertEqual([(r["event_type"], r["value"]) for r in cancel], [("cancellation", "1963")])
        self.assertEqual(self.query("AN/SPG-59", "lifecycle_event", "event_type", "in_service"), [])
        first = self.query("AN/MPQ-65", "service_entry", "event_type", "in_service")
        second = self.query("AN/MPQ-65", "service_entry", "event_type", "initial_operational_capacity")
        self.assertEqual([r["value"] for r in first], ["1981"])
        self.assertEqual([r["value"] for r in second], ["1984"])
        self.assertFalse(first[0]["original_entity_attribution_usable"])

    def test_source_scoped_unknown_and_empty_are_distinct(self):
        rows = self.query("EL/M-2080", "power")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value_kind"], "unknown")
        self.assertEqual(rows[0]["unknown_scope"], "bound_source_field")
        self.assertTrue(all(rows[0][key] is None for key in ("value", "min_value", "max_value")))
        self.assertIn("未知", render_records(rows))
        self.assertNotEqual(rows, [])
        self.assertIn("不表示数值为零", render_records([]))

    def test_failed_steps_preserve_handles_and_can_recover_through_native_parser(self):
        episode = DevelopmentEpisode(self.kb)
        def call(name, arguments):
            return execute_response(episode, call_message(name, arguments)["content"])
        self.assertTrue(call("step", dict(function="Find", inputs=["AN/SPG-51"], dependencies=[]))["ok"])
        self.assertFalse(call("finish", dict(answer_handle=0))["ok"])
        self.assertFalse(call("step", dict(function="QueryAttrUnderCondition", inputs=["prf", "mode", "air"], dependencies=[0]))["ok"])
        self.assertFalse(call("step", dict(function="QueryAttr", inputs=["prf"], dependencies=[True]))["ok"])
        self.assertEqual(len(episode.handles), 1)
        self.assertFalse(episode.done)
        self.assertTrue(call("step", dict(function="QueryAttrUnderCondition", inputs=["prf", "condition_raw", "air"], dependencies=[0]))["ok"])
        self.assertTrue(call("finish", dict(answer_handle=1))["ok"])
        self.assertEqual(episode.prediction[0]["fact_id"], "radar-review-001-08")
        self.assertEqual(episode.calls, 6)
        self.assertTrue(episode.done)

    def test_invalid_response_and_last_call_budget(self):
        episode = DevelopmentEpisode(self.kb, max_calls=2)
        self.assertFalse(execute_response(episode, "not a tool call")["ok"])
        self.assertTrue(episode.step("Find", ["AN/SPG-51"], [])["ok"])
        self.assertTrue(episode.done)
        self.assertFalse(episode.finish(0)["ok"])
        self.assertEqual(episode.calls, 2)
        self.assertIsNone(episode.prediction)
        final = DevelopmentEpisode(self.kb, max_calls=3)
        final.step("Find", ["no exact anchor"], [])
        final.step("QueryAttr", ["power"], [0])
        self.assertTrue(final.finish(1)["ok"])
        self.assertEqual(final.prediction, [])
        self.assertTrue(final.done)

    def test_program_and_agent_use_identical_records_and_semantics(self):
        text = "Find<arg>AN/SPG-51<func>QueryAttrUnderCondition<arg>frequency<arg>condition_raw<arg>Illuminator"
        p = execute_program(self.kb, text)
        self.assertEqual(p.prediction, self.query("AN/SPG-51", "frequency", "condition_raw", "Illuminator"))
        self.assertEqual(p.calls, 3)
        self.assertTrue(p.done)
        self.assertEqual([e["tool"] for e in p.events], ["step", "step", "finish"])

    def test_program_errors_are_not_silently_repaired(self):
        for text in ("Find<arg>AN/SPG-51", "```\nFind<arg>AN/SPG-51<func>QueryAttr<arg>prf\n```",
                     "Find<arg>AN/SPG-51<func>QueryAttr<arg>prf<func>QueryAttr<arg>prf",
                     "Find<arg>AN/SPG-51<func>QueryAttrUnderCondition<arg>prf<arg>mode<arg>air"):
            with self.subTest(text=text):
                episode = execute_program(self.kb, text)
                self.assertTrue(episode.done)
                self.assertIsNone(episode.prediction)
                self.assertTrue(any(not e["observation"]["ok"] for e in episode.events))

    def test_query_results_cannot_mutate_knowledge(self):
        rows = self.query("AN/SPG-51", "prf", "condition_raw", "air")
        rows[0]["min_value"] = "0"
        rows[0]["citation"]["source_uri"] = "changed"
        again = self.query("AN/SPG-51", "prf", "condition_raw", "air")
        self.assertEqual(again[0]["min_value"], "9600")
        self.assertNotEqual(again[0]["citation"]["source_uri"], "changed")

    def test_loader_rejects_gold_promotion_and_loss_of_scope(self):
        document = json.loads(READINGS.read_text())
        variants = []
        promoted = deepcopy(document)
        promoted["human_verified"] = True
        variants.append(promoted)
        missing_scope = deepcopy(document)
        missing_scope["records"][0]["scope_notes_zh"] = ""
        variants.append(missing_scope)
        collapsed = deepcopy(document)
        ranged = next(r for r in collapsed["records"] if r["value_kind"] == "range")
        ranged["value"] = ranged["min_value"]
        variants.append(collapsed)
        invented_unknown = deepcopy(document)
        unknown = next(r for r in invented_unknown["records"] if r["value_kind"] == "unknown")
        unknown["value"] = "0"
        variants.append(invented_unknown)
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "readings.json"
            for i, variant in enumerate(variants):
                with self.subTest(variant=i):
                    candidate.write_text(json.dumps(variant))
                    with self.assertRaises(ValueError):
                        DevelopmentKB(candidate)

    def test_shared_prompt_catalog_has_schema_not_reference_answer(self):
        question = "测试问题文本"
        agent = prompt_messages(question, self.kb)
        program = prompt_messages(question, self.kb, program=True)
        self.assertEqual(agent[1], {"role": "user", "content": question})
        self.assertEqual(agent[1], program[1])
        self.assertTrue(agent[0]["content"].endswith(self.kb.schema_prompt()))
        self.assertTrue(program[0]["content"].endswith(self.kb.schema_prompt()))
        self.assertNotIn("radar-review-001-", agent[0]["content"])
        self.assertNotIn("1981", agent[0]["content"])


if __name__ == "__main__":
    unittest.main()
