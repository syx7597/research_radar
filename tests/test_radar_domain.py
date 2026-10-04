"""Evidence and scope admission tests; no models or benchmark examples."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.radar_domain.adapter import RadarRecordAdapter
from scripts import audit_radar_review_workflow as workflow
from scripts import prepare_radar_review_packet as packet


class RadarDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = RadarRecordAdapter()

    def preview(self, **query):
        return self.adapter.query(query, "source_preview")["snapshot_readings"]

    def copy_workflow(self, directory):
        target = Path(directory) / "workflow"
        target.mkdir()
        for name in ("review_decisions.csv", "development_questions.csv", "workflow_manifest.json"):
            (target / name).write_bytes((workflow.DEFAULT_WORKFLOW / name).read_bytes())
        return target

    def write_decisions(self, target, rows):
        (target / "review_decisions.csv").write_bytes(workflow.csv_bytes(workflow.DECISION_HEADERS, rows))

    def human_fixture(self, row):
        row.update(decision="accepted", decision_scope="source_snapshot_reading", reviewer_kind="human",
                   reviewer="synthetic-test-reviewer", reviewed_at="2026-10-04T10:00:00+08:00",
                   reason="Temporary synthetic test fixture, never a real acceptance")
        row.update(dict.fromkeys(workflow.CHECK_FIELDS, "confirmed"))

    def test_default_cannot_answer_with_pending_candidates(self):
        result = self.adapter.query(dict(entity="AN/SPG-51", attribute="prf", condition_raw="air"))
        self.assertEqual(result["status"], "no_accepted_independent_data")
        self.assertEqual(result["dataset_independent_fact_eligible"], 0)
        self.assertEqual(result["matching_candidate_records"], 1)
        self.assertEqual(result["accepted_facts"], [])
        self.assertEqual(result["snapshot_readings"], [])

    def test_conditions_are_exact_and_do_not_merge_ranges(self):
        rows = self.preview(entity="AN/SPG-51", attribute="prf")
        self.assertEqual({row["condition"]["raw"] for row in rows}, {"air", "surface"})
        air = self.preview(entity="AN/SPG-51", attribute="prf", condition_raw="air")
        self.assertEqual(len(air), 1)
        self.assertEqual(air[0]["value"]["min_value"], "9600")
        self.assertEqual(air[0]["value"]["max_value"], "16700")
        self.assertIsNone(air[0]["value"]["value"])
        self.assertEqual(self.preview(entity="AN/SPG-51", attribute="prf", condition_raw="Air"), [])
        self.assertEqual(self.preview(entity="AN/SPG-51", attribute="prf", condition_raw=None), [])

    def test_event_selectors_keep_cancellation_and_service_separate(self):
        self.assertEqual(self.preview(entity="AN/SPG-59", attribute="service_entry", event_type="in_service"), [])
        cancel = self.preview(entity="AN/SPG-59", attribute="service_entry", event_type="cancellation")
        self.assertEqual(cancel[0]["value"]["value"], "1963")
        first = self.preview(entity="AN/MPQ-65", attribute="service_entry", event_type="in_service")
        second = self.preview(entity="AN/MPQ-65", attribute="service_entry", event_type="initial_operational_capacity")
        self.assertEqual([row["value"]["value"] for row in first], ["1981"])
        self.assertEqual([row["value"]["value"] for row in second], ["1984"])
        self.assertEqual(first[0]["value"]["value_status"], "ambiguous")

    def test_unspecified_variant_is_not_a_known_variant(self):
        rows = self.preview(entity="AN/APY-9", attribute="frequency", variant=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant_status"], "not_recorded")
        self.assertIsNone(rows[0]["variant"])
        self.assertEqual(self.preview(entity="AN/APY-9", attribute="frequency", variant="fictional-variant"), [])

    def test_unknown_is_scoped_missing_value_not_zero(self):
        row = self.preview(entity="EL/M-2080", attribute="power")[0]
        self.assertEqual(row["value"]["value_kind"], "unknown")
        self.assertEqual(row["value"]["value_status"], "not_recorded")
        for key in ("value", "min_value", "max_value", "unit_raw", "unit_std"):
            self.assertIsNone(row["value"][key])
        self.assertEqual(row["citation"]["evidence_text"], "Classified")

    def test_preview_preserves_ambiguity_review_and_exact_citation(self):
        row = self.preview(entity="kg_v3:AN/APY-9", attribute="frequency")[0]
        self.assertEqual(row["value"]["value_status"], "ambiguous")
        self.assertEqual((row["value"]["min_value"], row["value"]["max_value"]), ("0.3", "3.0"))
        self.assertEqual(row["review"]["decision"], "pending")
        self.assertFalse(row["review"]["human_review_declared"])
        self.assertFalse(row["review"]["independent_fact_admitted"])
        source = (workflow.ROOT / row["citation"]["source_file"]).read_bytes()
        self.assertEqual(workflow.digest(source), row["citation"]["source_sha256"])
        locator = row["citation"]["locator"]
        text = packet.resolve_pointer(json.loads(source), locator["json_pointer"])
        self.assertEqual(text[locator["char_start"]:locator["char_end"]], row["citation"]["evidence_text"])

    def test_legacy_value_is_preserved_separately_from_snapshot_evidence(self):
        facts, _ = workflow.load_packet()
        fact = next(item for item in facts if item["fact_id"] == "radar-review-001-09")
        row = self.preview(entity="AN/SPN-35", attribute="frequency")[0]
        self.assertEqual(row["legacy_value_raw"], fact["value_raw"])
        self.assertEqual(row["citation"]["evidence_text"], fact["evidence_text"])
        self.assertEqual(row["value"]["unit_std"], "GHz")
        self.assertIn("Not the normalized value", row["legacy_value_scope"])
        # The original raw string and snapshot excerpt even retain different
        # whitespace; neither is overwritten by the candidate normalized value.
        self.assertNotEqual(row["legacy_value_raw"], row["citation"]["evidence_text"])

    def test_valid_csv_quoting_and_newlines_preserve_raw_byte_binding(self):
        for quoting, newline in ((csv.QUOTE_ALL, "\n"), (csv.QUOTE_ALL, "\r\n"),
                                 (csv.QUOTE_MINIMAL, "\r\n")):
            with self.subTest(quoting=quoting, newline=repr(newline)), tempfile.TemporaryDirectory() as temp:
                target = self.copy_workflow(temp)
                rows = workflow.read_csv(target / "review_decisions.csv", workflow.DECISION_HEADERS)
                stream = io.StringIO(newline="")
                writer = csv.DictWriter(stream, fieldnames=workflow.DECISION_HEADERS,
                                        quoting=quoting, lineterminator=newline)
                writer.writeheader()
                writer.writerows(rows)
                data = stream.getvalue().encode("utf-8")
                (target / "review_decisions.csv").write_bytes(data)
                adapter = RadarRecordAdapter(target)
                self.assertEqual(adapter.audit_summary()["checked_files"]["review_decisions.csv"], workflow.digest(data))
                result = adapter.query(dict(entity="AN/SPG-51", attribute="prf", condition_raw="air"))
                self.assertEqual(result["status"], "no_accepted_independent_data")
                self.assertEqual(result["matching_candidate_records"], 1)

    def test_raw_byte_change_after_audit_is_rejected(self):
        original_audit = workflow.audit
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)

            def change_after_audit(directory):
                result = original_audit(directory)
                source = directory / "review_decisions.csv"
                # Even a semantically neutral byte change must invalidate this
                # in-flight audit binding; a fresh audit can admit the new file.
                source.write_bytes(source.read_bytes() + b"\n")
                return result

            with patch.object(workflow, "audit", side_effect=change_after_audit):
                with self.assertRaisesRegex(ValueError, "Workflow changed"):
                    RadarRecordAdapter(target)

    def test_unknown_fields_and_nonexact_query_types_are_rejected(self):
        for query in (dict(entity="AN/SPG-51", attribute="prf", condition="air"),
                      dict(entity="AN/SPG-51", attribute="prf", condition_raw=[]),
                      dict(entity="AN/SPG-51", attribute="prf", variant=""),
                      dict(attribute="prf")):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.adapter.query(query)

    def test_literal_accepted_string_cannot_bypass_review(self):
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)
            rows = workflow.read_csv(target / "review_decisions.csv", workflow.DECISION_HEADERS)
            rows[4]["decision"] = "accepted"
            self.write_decisions(target, rows)
            with self.assertRaisesRegex(ValueError, "review requires"):
                RadarRecordAdapter(target)

    def test_unrelated_invalid_decision_blocks_whole_view(self):
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)
            rows = workflow.read_csv(target / "review_decisions.csv", workflow.DECISION_HEADERS)
            rows[-1]["source_sha256"] = "tampered-unqueried-row"
            self.write_decisions(target, rows)
            with self.assertRaisesRegex(ValueError, "exact fact"):
                RadarRecordAdapter(target)

    def test_snapshot_acceptance_does_not_admit_independent_fact(self):
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)
            rows = workflow.read_csv(target / "review_decisions.csv", workflow.DECISION_HEADERS)
            self.human_fixture(rows[4])
            self.write_decisions(target, rows)
            adapter = RadarRecordAdapter(target)
            result = adapter.query(dict(entity="AN/SPG-51", attribute="frequency", condition_raw="Illuminator"))
            self.assertEqual(result["status"], "no_accepted_independent_data")
            self.assertEqual(result["accepted_facts"], [])

    def test_independent_admission_requires_bound_primary_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)
            rows = workflow.read_csv(target / "review_decisions.csv", workflow.DECISION_HEADERS)
            row = rows[4]
            self.human_fixture(row)
            primary = Path(temp) / "synthetic-primary.txt"
            data = b"Synthetic test fixture: 10.25-10.5 GHz"
            primary.write_bytes(data)
            row.update(decision_scope="independent_fact", independent_source_class="primary",
                       independent_source_uri="https://example.test/synthetic-primary",
                       independent_source_file="tests/synthetic-primary.txt",
                       independent_source_sha256=workflow.digest(data),
                       independent_source_locator=json.dumps(dict(char_start=0, char_end=len(data),
                                                                  offset_unit="unicode_codepoint", end_exclusive=True)),
                       independent_evidence_text=data.decode())
            self.write_decisions(target, rows)
            with patch.object(workflow, "local_path", return_value=primary):
                adapter = RadarRecordAdapter(target)
                result = adapter.query(dict(entity="AN/SPG-51", attribute="frequency", condition_raw="Illuminator"))
                self.assertEqual(result["dataset_independent_fact_eligible"], 1)
                self.assertEqual(result["snapshot_readings"], [])
                self.assertEqual(result["accepted_facts"][0]["independent_citation"]["source_sha256"], workflow.digest(data))
                primary.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    RadarRecordAdapter(target)

    def test_source_tampering_is_detected_before_any_query(self):
        original = packet.file_info

        def changed_source(relative):
            info = original(relative)
            if relative == "radar_corpus/raw/an_spg-51.json":
                return dict(info, sha256="synthetic-tamper")
            return info

        with patch.object(packet, "file_info", side_effect=changed_source):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                RadarRecordAdapter()

    def test_invalid_question_revision_blocks_view(self):
        with tempfile.TemporaryDirectory() as temp:
            target = self.copy_workflow(temp)
            rows = workflow.read_csv(target / "development_questions.csv", workflow.QUESTION_HEADERS)
            rows[0]["answer_json"] = '"synthetic unwanted answer"'
            (target / "development_questions.csv").write_bytes(workflow.csv_bytes(workflow.QUESTION_HEADERS, rows))
            with self.assertRaisesRegex(ValueError, "answer-free"):
                RadarRecordAdapter(target)


if __name__ == "__main__":
    unittest.main()
