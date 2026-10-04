"""Review admission checks use synthetic facts, never benchmark questions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_radar_review_workflow as workflow


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.fact = {"fact_id": "synthetic-fact", "source_sha256": "snapshot-hash",
                     "source_file": "synthetic/secondary.json", "source_uri": "https://example.test/secondary",
                     "value_status": "stated"}
        self.decision = dict.fromkeys(workflow.DECISION_HEADERS, "")
        self.decision.update(fact_id=self.fact["fact_id"], fact_record_sha256=workflow.fact_digest(self.fact),
                             source_sha256=self.fact["source_sha256"], decision="pending")

    def accepted_reading(self):
        self.decision.update(decision="accepted", decision_scope="source_snapshot_reading",
                             reviewer_kind="human", reviewer="synthetic-reviewer",
                             reviewed_at="2026-10-04T10:00:00+08:00", reason="Synthetic test fixture only")
        self.decision.update(dict.fromkeys(workflow.CHECK_FIELDS, "confirmed"))

    def audit_decision(self):
        return workflow.validate_decisions([self.fact], [self.decision])

    def test_pending_is_not_accepted(self):
        result = self.audit_decision()
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["independent_fact_eligible"], 0)

    def test_fact_mutation_invalidates_existing_review(self):
        self.fact["value_status"] = "ambiguous"
        with self.assertRaisesRegex(ValueError, "exact fact"):
            self.audit_decision()

    def test_ai_review_cannot_be_accepted(self):
        self.accepted_reading()
        self.decision["reviewer_kind"] = "AI"
        with self.assertRaisesRegex(ValueError, "human reviewer"):
            self.audit_decision()

    def test_relabelled_ai_name_does_not_pass_known_marker_check(self):
        self.accepted_reading()
        self.decision["reviewer"] = "AI_initial_screening_not_human_verified"
        with self.assertRaisesRegex(ValueError, "human reviewer"):
            self.audit_decision()

    def test_missing_subject_check_blocks_acceptance(self):
        self.accepted_reading()
        self.decision["subject_check"] = ""
        with self.assertRaisesRegex(ValueError, "each subject"):
            self.audit_decision()

    def test_snapshot_acceptance_is_not_independent_fact_acceptance(self):
        self.accepted_reading()
        result = self.audit_decision()
        self.assertEqual(result["accepted_snapshot_readings"], 1)
        self.assertEqual(result["independent_fact_eligible"], 0)

    def test_known_secondary_source_cannot_be_relabelled_primary(self):
        self.accepted_reading()
        self.decision.update(decision_scope="independent_fact", independent_source_class="primary",
                             independent_source_file=self.fact["source_file"])
        with self.assertRaisesRegex(ValueError, "known secondary snapshot"):
            self.audit_decision()

    def test_independent_evidence_hash_and_excerpt_are_checked(self):
        data = json.dumps({"text": "Synthetic subject; synthetic value."}).encode()
        locator = {"json_pointer": "/text", "char_start": 0, "char_end": 17,
                   "offset_unit": "unicode_codepoint", "end_exclusive": True}
        self.accepted_reading()
        self.decision.update(decision_scope="independent_fact", independent_source_class="primary",
                             independent_source_uri="https://example.test/primary",
                             independent_source_file="synthetic/primary.json",
                             independent_source_sha256=workflow.digest(data),
                             independent_source_locator=json.dumps(locator),
                             independent_evidence_text="Synthetic subject")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "primary.json"
            source.write_bytes(data)
            with patch.object(workflow, "local_path", return_value=source):
                self.assertEqual(self.audit_decision()["independent_fact_eligible"], 1)
                self.decision["independent_evidence_text"] = "Invented evidence"
                with self.assertRaisesRegex(ValueError, "exact short"):
                    self.audit_decision()
                self.decision["independent_source_sha256"] = "wrong-hash"
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.audit_decision()


if __name__ == "__main__":
    unittest.main()
