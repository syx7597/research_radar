"""External review package checks using only synthetic, temporary 96-item data."""
from copy import deepcopy
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from experiments.radar_domain import coverage_external_review as external


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(external.encoded(value))
    return external.digest(path.read_bytes())


def fixture(root):
    paths = {"selection": "artifacts/thesis_direction_review/radar_questions_v1/selection.json",
        "snapshot": "data/radar_sources_v2/test/snapshot.json",
        "chunks": "data/radar_sources_v2/test/chunks.json",
        "scoring_adjudication": "data/radar_sources_v2/test/rubric_clarifications.json",
        "author_rubric_response": "data/radar_sources_v2/test/author_response.json",
        "author_packets": ["data/radar_sources_v2/test/author_a.json", "data/radar_sources_v2/test/author_b.json"],
        "private_dir": "data/radar_sources_v2/external_review_v1",
        "public_manifest": "artifacts/thesis_direction_review/radar_external_review_v1/manifest.json"}
    rows, chunks = [], []
    for index, family in enumerate(external.FAMILIES):
        source_id = f"synthetic_{index}"
        html_source = index % 2 == 0
        derived = f"Synthetic\twhole page for {family}. <script>source_attack()</script> & text.\n"
        text = re.sub(r"\s+", " ", derived).strip()
        source = (b"<html><script>source_attack()</script></html>" if html_source
                  else b"%PDF-1.4\n% Synthetic archive, not a real research source\n")
        source_path = f"data/radar_sources_v2/test/sources/{source_id}.archive"
        derived_path = f"data/radar_sources_v2/test/sources/{source_id}.txt"
        for relative, value in ((source_path, source), (derived_path, derived.encode())):
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        chunk = {"chunk_id": source_id + ":page", "family_id": family, "source_id": source_id,
            "text": text, "text_sha256": external.digest(text.encode()), "source_path": source_path,
            "derived_path": derived_path, "source_sha256": external.digest(source),
            "derived_sha256": external.digest(derived.encode()), "page_one_based": None if html_source else 1,
            "source_uri": "https://example.invalid/synthetic"}
        chunks.append(chunk)
        start = text.index("whole page")
        for number in range(12):
            qid = f"fixture_{index}_{number}"
            question = f"Synthetic question {qid}? </script><script>question_attack()</script>"
            rows.append({"question_id": qid, "family_id": family,
                "primary_type": external.STRATA[number % 3], "question_zh": question,
                "question_sha256": external.digest(question.encode()), "source_version": "synthetic version",
                "reference": {"answer": "REFERENCE_SECRET_" + qid},
                "support": [{"chunk_id": chunk["chunk_id"], "text_sha256": chunk["text_sha256"],
                    "spans": [{"start": start, "end": start + len("whole page"), "quote": "whole page"}]}]})
    objects = {paths["selection"]: {"synthetic": True}, paths["snapshot"]: {"synthetic": True},
        paths["chunks"]: {"normalization": "collapse_whitespace_only", "chunks": chunks},
        paths["scoring_adjudication"]: {"note": "RUBRIC_SECRET_ONLY_SIDECAR"},
        paths["author_rubric_response"]: {"required_facts_scoring_interpretation": []},
        paths["author_packets"][0]: {"questions": rows[:48]}, paths["author_packets"][1]: {"questions": rows[48:]}}
    protocol = {"schema": "radar_coverage_external_review_protocol_v1", "status": "registered", "seed": 20261008,
        "selection_rule": "sha256_utf8_pipe_rank_v1", "per_family_per_stratum": 1, "human_gold": False,
        "purpose": "source_and_QA_reference_fact_verification", "citation_sensitivity_results_seen_at_registration": False,
        "previous_original_coverage_results_known_to_project": True,
        "families": list(external.FAMILIES), "strata": list(external.STRATA), "paths": paths,
        "limits": "Synthetic risk-stratified fixture; not a human review.",
        "input_sha256": {name: write_json(root, name, value) for name, value in objects.items()}}
    write_json(root, external.DEFAULT_PROTOCOL, protocol)
    return protocol, objects


def rebind(root, protocol, name, value):
    protocol["input_sha256"][name] = write_json(root, name, value)
    write_json(root, external.DEFAULT_PROTOCOL, protocol)


def decoded(outputs, path):
    return json.loads(outputs[Path(path)])


class MetadataSelection(unittest.TestCase):
    def test_protocol_rejects_changed_design_input_inventory_or_destination(self):
        with tempfile.TemporaryDirectory() as d:
            protocol, _ = fixture(Path(d))
            external.validate_protocol(protocol)
            cases = []
            for key, value in (("seed", 20261009), ("human_gold", True),
                               ("families", list(reversed(external.FAMILIES))),
                               ("strata", ["simple_attribute", *external.STRATA])):
                changed = deepcopy(protocol); changed[key] = value; cases.append(changed)
            changed = deepcopy(protocol); changed["input_sha256"]["data/radar_sources_v2/extra.json"] = "unexpected"
            cases.append(changed)
            changed = deepcopy(protocol); changed["paths"]["private_dir"] = "results/answers"
            cases.append(changed)
            changed = deepcopy(protocol)
            old_path = changed["paths"]["snapshot"]
            changed["paths"]["snapshot"] = "results/model_scores.json"
            changed["input_sha256"]["results/model_scores.json"] = changed["input_sha256"].pop(old_path)
            cases.append(changed)
            for changed in cases:
                with self.assertRaises(ValueError):
                    external.validate_protocol(changed)

    def test_fixed_hash_ranking_and_order_are_independent_of_input_order(self):
        with tempfile.TemporaryDirectory() as d:
            protocol, objects = fixture(Path(d))
            authors = [r for name in protocol["paths"]["author_packets"] for r in objects[name]["questions"]]
            metadata = [{key: row[key] for key in external.METADATA} for row in authors]
            selected = external.select_metadata(metadata, protocol)
            self.assertEqual(selected, external.select_metadata(list(reversed(metadata)), protocol))
            self.assertEqual(len(selected), 24)
            self.assertEqual(Counter(r["family_id"] for r in selected), {family: 3 for family in external.FAMILIES})
            self.assertEqual(Counter(r["primary_type"] for r in selected), {s: 8 for s in external.STRATA})
            for picked in selected:
                candidates = [r for r in metadata if (r["family_id"], r["primary_type"]) == (picked["family_id"], picked["primary_type"])]
                def rank(row):
                    material = "|".join(["20261008", row["family_id"], row["primary_type"], row["question_id"], row["question_sha256"]])
                    return hashlib.sha256(material.encode()).hexdigest(), row["question_id"]
                expected = min(candidates, key=rank)
                self.assertEqual(picked["question_id"], expected["question_id"])
                self.assertEqual(picked["rank_sha256"], rank(expected)[0])

    def test_selector_rejects_score_or_reference_fields_duplicates_and_missing_strata(self):
        with tempfile.TemporaryDirectory() as d:
            protocol, objects = fixture(Path(d))
            rows = [{key: row[key] for key in external.METADATA} for name in protocol["paths"]["author_packets"] for row in objects[name]["questions"]]
            for key in ("reference", "model_output", "answer_correct"):
                bad = deepcopy(rows); bad[0][key] = "must never enter selector"
                with self.assertRaisesRegex(ValueError, "metadata fields"):
                    external.select_metadata(bad, protocol)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                external.select_metadata(rows + [rows[0]], protocol)
            with self.assertRaisesRegex(ValueError, "no fallback"):
                external.select_metadata([r for r in rows if r["primary_type"] != external.STRATA[0]], protocol)


class ExternalPackage(unittest.TestCase):
    def test_private_reference_separation_empty_human_fields_and_public_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); protocol, _ = fixture(root)
            outputs = external.build(root=root); private = Path(protocol["paths"]["private_dir"])
            questions = decoded(outputs, private / "questions.json")["rows"]
            references = decoded(outputs, private / "references.json")
            page = outputs[private / "index.html"].decode()
            public_bytes = outputs[Path(protocol["paths"]["public_manifest"])]
            public = json.loads(public_bytes)
            self.assertEqual(len(questions), 24)
            self.assertFalse(references["human_gold"])
            self.assertEqual(len(references["rows"]), 24)
            self.assertNotIn("REFERENCE_SECRET_", page)
            self.assertNotIn("RUBRIC_SECRET_ONLY_SIDECAR", page)
            payload = json.loads(re.search(r'<script id="packet" type="application/json">(.*?)</script>', page, re.S).group(1))
            self.assertEqual(payload["questions"], questions)
            self.assertEqual(payload["reference_sha256"], external.digest(outputs[private / "references.json"]))
            self.assertNotIn("</script><script>question_attack()", page)
            for row in questions:
                self.assertNotIn("reference", row)
                self.assertNotIn("support", row)
                for source in row["sources"]:
                    self.assertNotIn("quote", source)
                    if source["page_one_based"] is None:
                        html_page = outputs[private / source["href"]].decode()
                        self.assertIn("&lt;script&gt;source_attack()&lt;/script&gt;", html_page)
                        self.assertNotIn("<script>source_attack()", html_page)
            blank = decoded(outputs, private / "blank_responses.json")
            self.assertIsNone(blank["reviewer_identity"])
            self.assertIsNone(blank["reference_file_loaded_at"])
            self.assertFalse(blank["human_gold"])
            self.assertEqual(len(blank["rows"]), 24)
            for row in blank["rows"]:
                self.assertEqual(set(row), {"question_id", *external.EMPTY_FIELDS})
                self.assertTrue(all(row[key] is None for key in external.EMPTY_FIELDS))
            csv_rows = list(csv.DictReader(io.StringIO(outputs[private / "blank_responses.csv"].decode("utf-8-sig"))))
            self.assertEqual(len(csv_rows), 24)
            self.assertTrue(all(not row[key] for row in csv_rows for key in external.EMPTY_FIELDS))
            self.assertEqual(public["question_count"], 24)
            self.assertEqual(public["human_reviews_completed"], 0)
            self.assertEqual(public["model_inference_runs"], 0)
            self.assertFalse(public["model_outputs_read_for_selection"])
            self.assertFalse(public["human_gold"])
            for marker in (b"REFERENCE_SECRET_", b"RUBRIC_SECRET_ONLY_SIDECAR", b"Synthetic question", b"whole page"):
                self.assertNotIn(marker, public_bytes)
            for path, digest in public["private_outputs_sha256"].items():
                self.assertEqual(external.digest(outputs[Path(path)]), digest)

    def test_reference_changes_cannot_change_metadata_selection_or_question_view(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); protocol, objects = fixture(root)
            before = external.build(root=root)
            private = Path(protocol["paths"]["private_dir"])
            for name in protocol["paths"]["author_packets"]:
                for row in objects[name]["questions"]:
                    row["reference"] = {"different": "not a selector dependency"}
                rebind(root, protocol, name, objects[name])
            after = external.build(root=root)
            self.assertEqual(decoded(before, private / "selection.json")["rows"], decoded(after, private / "selection.json")["rows"])
            self.assertEqual(before[private / "questions.json"], after[private / "questions.json"])
            self.assertNotEqual(before[private / "references.json"], after[private / "references.json"])

    def test_tampered_input_or_archived_source_is_rejected(self):
        for mode in ("input", "source", "derived"):
            with tempfile.TemporaryDirectory() as d, self.subTest(mode=mode):
                root = Path(d); protocol, objects = fixture(root)
                chunk = objects[protocol["paths"]["chunks"]]["chunks"][0]
                path = protocol["paths"]["author_packets"][0] if mode == "input" else chunk[mode + "_path"]
                (root / path).write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    external.build(root=root)

    def test_rebound_source_schema_or_semantic_identity_corruption_is_rejected(self):
        for mode, message in (("question", "fingerprint"), ("count", "96 questions"),
            ("family", "family/text binding"), ("span", "source span"),
            ("page", "PDF source/page"), ("complete", "complete archived"),
            ("escape", "relative"), ("normalization", "normalization"), ("duplicate_chunk", "Duplicate source chunk")):
            with tempfile.TemporaryDirectory() as d, self.subTest(mode=mode):
                root = Path(d); protocol, objects = fixture(root)
                chunk_name = protocol["paths"]["chunks"]; author_name = protocol["paths"]["author_packets"][0]
                chunk = objects[chunk_name]["chunks"][0]
                if mode == "question":
                    objects[author_name]["questions"][0]["question_zh"] = "changed"
                elif mode == "count": objects[author_name]["questions"].pop()
                elif mode == "family": chunk["family_id"] = "another_family"
                elif mode == "span":
                    for row in objects[author_name]["questions"][:12]: row["support"][0]["spans"][0]["quote"] = "different"
                elif mode == "page": chunk["page_one_based"] = 1
                elif mode == "complete":
                    extra = (root / chunk["derived_path"]).read_bytes() + b" extra omitted from chunk"
                    (root / chunk["derived_path"]).write_bytes(extra); chunk["derived_sha256"] = external.digest(extra)
                elif mode == "escape": chunk["source_path"] = "data/radar_sources_v2/../../outside.archive"
                elif mode == "normalization": objects[chunk_name]["normalization"] = "modified_text"
                elif mode == "duplicate_chunk": objects[chunk_name]["chunks"].append(deepcopy(chunk))
                rebind(root, protocol, author_name, objects[author_name])
                rebind(root, protocol, chunk_name, objects[chunk_name])
                with self.assertRaisesRegex(ValueError, message):
                    external.build(root=root)

    def test_publish_never_overwrites_and_check_detects_changed_response(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); protocol, _ = fixture(root)
            outputs = external.build(root=root)
            external.publish(outputs, root=root)
            external.publish(external.build(root=root), root=root, check=True)
            response = root / protocol["paths"]["private_dir"] / "blank_responses.json"
            response.write_text("human edits must survive")
            with self.assertRaises(FileExistsError): external.publish(outputs, root=root)
            self.assertEqual(response.read_text(), "human edits must survive")
            with self.assertRaisesRegex(ValueError, "Reproduction mismatch"):
                external.publish(outputs, root=root, check=True)

    def test_path_resolver_rejects_absolute_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            root = Path(d)
            (root / "escape").symlink_to(Path(outside), target_is_directory=True)
            for path in ("../outside", str(Path(outside) / "file"), "escape/file"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    external.local(root, path)


if __name__ == "__main__":
    unittest.main()
