"""Synthetic metadata-only tests: no research questions, PDFs or models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import source_split_audit as audit


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def registry(**changes):
    return {**{field: [] for field in audit.REGISTRY_FIELDS}, **changes}


def packet(index=1, split="evaluation_candidate"):
    source_id, family = f"source-{index}", f"family-{index}"
    return {"sources": [{"source_id": source_id, "canonical_document_id": f"document-{index}",
                         "family_ids": [family], "source_uri": f"https://example.test/doc/{index}?v=1",
                         "source_sha256": digest(f"source-{index}"), "split": split}],
            "questions": [{"question_id": f"question-{index}", "question_sha256": digest(f"question-{index}"),
                           "source_ids": [source_id], "family_ids": [family], "equivalence_id": None,
                           "primary_type": "range", "split": split, "annotation_origin": "ai",
                           "review_status": "ai_reviewed", "model_outputs_seen": False}]}


def combine(first, second):
    return {name: first[name] + second[name] for name in ("sources", "questions")}


def codes(report, field="blockers"):
    return {row["code"] for row in report[field]}


class SourceSplitAuditTests(unittest.TestCase):
    def test_disjoint_components_coverage_and_input_not_mutated(self):
        data = combine(packet(1, "development"), packet(2))
        data["questions"][0]["primary_type"] = "options"
        before = deepcopy(data)
        # The API must not inspect declared source paths or any research files.
        with patch.object(Path, "read_text", side_effect=AssertionError("No file reads in API")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("No file reads in API")):
            result = audit.audit_packet(data, registry())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["coverage"], {"source_count": 2, "question_count": 2, "family_count": 2,
                         "component_count": 2, "question_counts_by_primary_type": {"options": 1, "range": 1},
                         "evaluation_candidate_question_count": 1, "evaluation_candidate_source_count": 1})
        self.assertEqual(data, before)
        self.assertIn("metadata_only", codes(result, "disclosures"))
        self.assertIn("no_equivalence_id", codes(result, "disclosures"))

    def test_duplicate_source_and_question_ids_rejected(self):
        for kind in ("sources", "questions"):
            data = packet()
            data[kind].append(deepcopy(data[kind][0]))
            with self.subTest(kind=kind):
                self.assertIn("duplicate_id", codes(audit.audit_packet(data, registry())))

    def test_required_nonempty_typed_collections_and_rows(self):
        for value in ({}, None, [], {"sources": [], "questions": []},
                      {"sources": {}, "questions": []}, {"sources": [], "questions": "not a list"}):
            with self.subTest(value=value):
                self.assertEqual(audit.audit_packet(value, registry())["status"], "blocked")
        for kind in ("sources", "questions"):
            for row in (None, [], {}, {"source_id": ""}):
                data = packet()
                data[kind] = [row]
                with self.subTest(kind=kind, row=row):
                    self.assertIn("invalid_row_schema", codes(audit.audit_packet(data, registry())))

    def test_document_hash_or_family_identity_joins_split_components(self):
        for field in ("canonical_document_id", "source_sha256", "family_ids"):
            first, second = packet(1, "development"), packet(2)
            second["sources"][0][field] = first["sources"][0][field]
            result = audit.audit_packet(combine(first, second), registry())
            with self.subTest(field=field):
                self.assertIn("cross_split_component", codes(result))
                self.assertEqual(result["coverage"]["component_count"], 1)
                self.assertEqual(result["components"][0]["source_ids"], ["source-1", "source-2"])

    def test_uri_case_and_fragment_match_but_queries_are_preserved(self):
        first, second = packet(1, "development"), packet(2)
        second["sources"][0]["source_uri"] = "HTTPS://EXAMPLE.TEST/doc/1?v=1#page=5"
        self.assertIn("cross_split_component", codes(audit.audit_packet(combine(first, second), registry())))
        second["sources"][0]["source_uri"] = "https://example.test/doc/1?v=2"
        self.assertEqual(audit.audit_packet(combine(first, second), registry())["status"], "passed")
        self.assertEqual(audit.normalize_uri("HTTPS://User:PaSS@EXAMPLE.TEST/Path?Version=A#x"),
                         "https://User:PaSS@example.test/Path?Version=A")

    def test_question_family_hash_and_equivalence_join_transitively(self):
        for field, value in (("family_ids", ["shared-question-family"]),
                             ("question_sha256", digest("same-question")), ("equivalence_id", "same-intent")):
            first, second = packet(1, "development"), packet(2)
            first["questions"][0][field] = second["questions"][0][field] = value
            with self.subTest(field=field):
                result = audit.audit_packet(combine(first, second), registry())
                self.assertIn("cross_split_component", codes(result))
                self.assertEqual(result["coverage"]["component_count"], 1)

    def test_source_and_question_split_must_match_connected_component(self):
        data = packet()
        data["sources"][0]["split"] = "development"
        self.assertIn("cross_split_component", codes(audit.audit_packet(data, registry())))

    def test_all_registry_identity_types_block_candidates(self):
        data = packet()
        source, question = data["sources"][0], data["questions"][0]
        identities = {"canonical_document_ids": source["canonical_document_id"],
                      "source_sha256": source["source_sha256"].upper(),
                      "source_uris": source["source_uri"] + "#different-anchor", "family_ids": source["family_ids"][0],
                      "question_ids": question["question_id"], "question_sha256": question["question_sha256"].upper()}
        for field, identity in identities.items():
            with self.subTest(field=field):
                result = audit.audit_packet(data, registry(**{field: [identity]}))
                self.assertIn("candidate_exposure", codes(result))
                self.assertTrue(result["components"][0]["exposure_matches"])

    def test_registry_match_propagates_through_declared_equivalence(self):
        first, second = packet(1), packet(2)
        first["questions"][0]["equivalence_id"] = second["questions"][0]["equivalence_id"] = "paired"
        result = audit.audit_packet(combine(first, second), registry(question_ids=["question-1"]))
        self.assertIn("candidate_exposure", codes(result))
        self.assertEqual(result["components"][0]["question_ids"], ["question-1", "question-2"])

    def test_development_reuse_unresolved_and_seen_are_disclosures(self):
        data = packet(split="development")
        data["questions"][0].update(review_status="unresolved", model_outputs_seen=True)
        result = audit.audit_packet(data, registry(family_ids=["family-1"]))
        self.assertEqual(result["status"], "passed")
        self.assertTrue({"development_reuse", "unresolved_review", "model_outputs_seen"} <= codes(result, "disclosures"))

    def test_unknown_questions_still_need_nonempty_existing_source_references(self):
        for source_ids in ([], ["missing-source"]):
            data = packet()
            data["questions"][0].update(primary_type="unknown", source_ids=source_ids)
            with self.subTest(source_ids=source_ids):
                self.assertEqual(audit.audit_packet(data, registry())["status"], "blocked")

    def test_candidate_family_ids_required_in_both_collections(self):
        for kind in ("sources", "questions"):
            data = packet()
            data[kind][0]["family_ids"] = []
            self.assertIn("missing_family_ids", codes(audit.audit_packet(data, registry())))
            for row in [*data["sources"], *data["questions"]]:
                row["split"] = "development"
            result = audit.audit_packet(data, registry())
            self.assertEqual(result["status"], "passed")
            self.assertIn("missing_family_ids", codes(result, "disclosures"))

    def test_ai_authorship_cannot_be_promoted_to_human_gold(self):
        for origin in ("ai", "mixed"):
            data = packet()
            data["questions"][0].update(annotation_origin=origin, review_status="human_gold")
            self.assertIn("ai_origin_human_gold", codes(audit.audit_packet(data, registry())))
        data["questions"][0]["annotation_origin"] = "human"
        result = audit.audit_packet(data, registry())
        self.assertEqual(result["status"], "passed")
        self.assertIn("review_declarations_only", codes(result, "disclosures"))

    def test_unresolved_or_seen_candidates_block_and_boolean_is_strict(self):
        for changes, expected in (({"review_status": "unresolved"}, "unresolved_review"),
                                  ({"model_outputs_seen": True}, "model_outputs_seen")):
            data = packet()
            data["questions"][0].update(changes)
            self.assertIn(expected, codes(audit.audit_packet(data, registry())))
        for seen in (0, 1, "false", None, []):
            data = packet()
            data["questions"][0]["model_outputs_seen"] = seen
            with self.subTest(seen=seen):
                self.assertIn("invalid_row_metadata", codes(audit.audit_packet(data, registry())))

    def test_invalid_ids_hashes_splits_and_registry_are_blocked(self):
        for field, value in (("question_id", ""), ("question_sha256", "not-a-hash"),
                             ("family_ids", "family-1"), ("split", "test"), ("split", []),
                             ("annotation_origin", "unknown"), ("review_status", "approved")):
            data = packet()
            data["questions"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(audit.audit_packet(data, registry())["status"], "blocked")
        for exposure in ({}, None, registry(family_ids="family-1"), registry(source_sha256=["bad"]),
                         registry(source_uris=["relative/path"])):
            self.assertIn("invalid_exposure_registry", codes(audit.audit_packet(packet(), exposure)))

    def test_undeclared_full_text_is_rejected_and_not_echoed(self):
        data = packet()
        data["questions"][0]["question_text"] = "PRIVATE_QA_SENTINEL"
        result = audit.audit_packet(data, registry())
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("PRIVATE_QA_SENTINEL", json.dumps(result))

    def test_cli_exit_status_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            p, e, out = base / "packet.json", base / "exposure.json", base / "audit.json"
            p.write_text(json.dumps(packet()))
            e.write_text(json.dumps(registry()))
            command = [sys.executable, "-m", "experiments.radar_domain.source_split_audit",
                       "--packet", str(p), "--exposure", str(e), "--output", str(out)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            before = out.read_bytes()
            self.assertEqual(json.loads(before)["inputs_sha256"], {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest(),
                str(e.resolve()): hashlib.sha256(e.read_bytes()).hexdigest(),
                str(Path(audit.__file__).resolve()): hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
            })
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), before)
            e.write_text(json.dumps(registry(family_ids=["family-1"])))
            command[-1] = str(base / "blocked.json")
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            blocked = json.loads((base / "blocked.json").read_text())
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["inputs_sha256"][str(e.resolve())], hashlib.sha256(e.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
