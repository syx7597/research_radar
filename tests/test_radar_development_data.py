"""Boundary checks for reviewed source readings and separated inference inputs."""
import csv
import io
import json
import unittest

from experiments.radar_domain import development_data as data
from scripts import prepare_radar_review_packet as packet


class RadarDevelopmentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = data.build()
        cls.readings = json.loads(cls.artifacts["readings.json"])
        cls.records = {row["fact_id"]: row for row in cls.readings["records"]}
        cls.questions = [json.loads(line) for line in cls.artifacts["questions.jsonl"].splitlines()]
        cls.references = json.loads(cls.artifacts["references.json"])["records"]

    def row(self, number):
        return self.records[f"radar-review-001-{number:02}"]

    def test_system_source_is_not_promoted_to_radar_attribution(self):
        for number in (1, 2, 3):
            row = self.row(number)
            self.assertEqual(row["entity_name"], "AN/MPQ-65")
            self.assertEqual(row["source_subject"], "MIM-104 Patriot system")
            self.assertFalse(row["original_entity_attribution_usable"])
            self.assertFalse(row["equipment_truth_verified"])
        self.assertEqual(self.row(2)["event_type"], "in_service")
        self.assertEqual(self.row(3)["event_type"], "initial_operational_capacity")
        self.assertEqual((self.row(2)["value"], self.row(3)["value"]), ("1981", "1984"))

    def test_ranges_keep_endpoints_and_conditions_separate(self):
        expected = {
            4: ("0.3", "3.0", None, "GHz"),
            5: ("10.25", "10.5", "Illuminator", "GHz"),
            6: ("5.45", "5.825", "Tracking Radar", "GHz"),
            8: ("9600", "16700", "air", "pps"),
            9: ("9.0", "9.2", None, "GHz"),
        }
        for number, values in expected.items():
            row = self.row(number)
            self.assertEqual(tuple(row[key] for key in ("min_value", "max_value", "condition_raw", "unit_std")), values)
            self.assertIsNone(row["value"])
            self.assertEqual(row["value_kind"], "range")
        self.assertEqual(self.row(4)["value_status"], "ambiguous")
        self.assertEqual((self.row(7)["value"], self.row(7)["condition_raw"]), ("4100", "surface"))

    def test_snapshot_scalar_is_not_replaced_by_external_options(self):
        row = self.row(10)
        self.assertEqual((row["value_kind"], row["value"], row["unit_std"]), ("scalar", "0.2", "us"))
        self.assertIsNone(row["min_value"])
        self.assertIsNone(row["max_value"])
        self.assertEqual(row["citation"]["source_uri"], "https://en.wikipedia.org/wiki/AN/SPN-35")
        self.assertIn("离散选项", row["scope_notes_zh"])
        self.assertIn("尚未核对原扫描", row["scope_notes_zh"])

    def test_cancellation_changes_attribute_without_rewriting_original(self):
        row = self.row(11)
        self.assertEqual((row["attribute"], row["event_type"], row["value"]),
                         ("lifecycle_event", "cancellation", "1963"))
        self.assertEqual(row["original_candidate_fields"]["attribute"], "service_entry")

    def test_missing_power_is_not_zero_or_global_unknown(self):
        row = self.row(12)
        self.assertEqual((row["value_kind"], row["value_status"]), ("unknown", "not_recorded"))
        for key in ("value", "min_value", "max_value", "unit_raw", "unit_std"):
            self.assertIsNone(row[key])
        self.assertEqual(row["unknown_scope"], "bound_source_field")
        self.assertEqual(row["unknown_reason"], "no_numeric_value_in_bound_field")
        self.assertEqual(row["source_claimed_status"], "Classified")

    def test_citations_bind_exact_original_spans(self):
        for row in self.records.values():
            citation = row["citation"]
            source = (data.ROOT / citation["source_file"]).read_bytes()
            self.assertEqual(data.digest(source), citation["source_sha256"])
            locator = citation["locator"]
            text = packet.resolve_pointer(json.loads(source), locator["json_pointer"])
            self.assertEqual(text[locator["char_start"]:locator["char_end"]], citation["evidence_text"])

    def test_questions_preserve_original_and_exclude_reference_selectors(self):
        original = list(csv.DictReader(io.StringIO((data.workflow.DEFAULT_WORKFLOW / "development_questions.csv").read_text())))
        self.assertEqual(len(self.questions), 12)
        for current, old, reference in zip(self.questions, original, self.references):
            self.assertEqual(set(current), {"id", "question"})
            self.assertEqual(current["id"], old["qid"])
            original_text, context = current["question"].split("\n", 1)
            self.assertEqual(original_text, old["question"])
            self.assertIn("source_context_id=", context)
            self.assertIn("source_uri=", context)
            for selector in ("radar-review-", "expected_fact_ids", "event_type", "condition_raw", "attribute=", "0.3", "1981", "1984", "10.25"):
                self.assertNotIn(selector, context)
            self.assertEqual(reference["expected_fact_ids"], json.loads(old["support_fact_ids"]))
            self.assertEqual(old["answer_json"], "")
            self.assertEqual(old["answer_status"], "")
        self.assertEqual(self.references[1]["acceptable_context_fact_ids"], ["radar-review-001-03"])
        self.assertEqual(self.references[2]["acceptable_context_fact_ids"], ["radar-review-001-02"])
        manifest = json.loads(self.artifacts["manifest.json"])
        self.assertNotIn("references.json", manifest["inference_input_files"])
        self.assertIn("references.json", manifest["excluded_from_inference"])

    def test_ai_review_does_not_establish_human_or_independent_gold(self):
        for row in [self.readings, *self.records.values(), *self.references]:
            self.assertFalse(row["human_verified"])
            self.assertFalse(row["independent_gold"])
        for row in self.records.values():
            self.assertEqual(row["review_status"], "AI_only_development")
            self.assertEqual(row["usage_scope"], "bound_source_snapshot_interpretation_only")
            self.assertIsNone(row["variant"])
            self.assertEqual(row["original_candidate_fields"]["variant"], "")


if __name__ == "__main__":
    unittest.main()
