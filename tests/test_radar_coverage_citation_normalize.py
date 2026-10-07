"""Synthetic-only tests. No original 288 outputs, references, scores, or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import coverage_citation_normalize as normalize


def record(rid="record:1", cid="source:p1"):
    return {"subjects": ["synthetic"], "component": None, "attribute": "fixture",
            "conditions_all": [], "record_id": rid, "citation": {"chunk_id": cid},
            "value": {"text": "unused source value"}}


class PureCitationNormalization(unittest.TestCase):
    def test_only_direct_citation_ids_change_preserving_types_order_duplicates(self):
        obj = {"answer_text": "literal record:1 must stay", "flag": True, "n": 1, "f": 1.0,
               "zero": -0.0, "null": None,
               "citations": ["record:1", {"chunk_id": "record:1", "extra": "record:1", "nested": {"chunk_id": "record:1"}},
                             "source:p1", "record:1", "unknown"]}
        text = "  " + json.dumps(obj, ensure_ascii=False) + "\n"
        evidence = [record()]; original_evidence = deepcopy(evidence)
        result = normalize.normalize_citations(text, "flat", evidence)
        expected = deepcopy(obj)
        expected["citations"][0] = expected["citations"][1]["chunk_id"] = expected["citations"][3] = "source:p1"
        self.assertEqual(result["original_text"], text)
        self.assertIsNone(normalize.representation._difference(result["parsed_before"], obj))
        self.assertIsNone(normalize.representation._difference(result["parsed_after"], expected))
        self.assertEqual(result["changed_reference_count"], 3)
        self.assertEqual(result["rejected_reference_count"], 1)
        self.assertEqual(evidence, original_evidence)
        result["parsed_after"]["citations"][1]["nested"]["chunk_id"] = "changed after return"
        self.assertEqual(result["parsed_before"]["citations"][1]["nested"]["chunk_id"], "record:1")

    def test_bound_and_flat_use_the_same_exact_supplied_mapping(self):
        records = [record(), record("record:2", "source:p2")]
        text = '{"citations":["record:2","record:1"]}'
        bound = json.loads(normalize.representation.encode_bound(records))
        self.assertEqual(normalize.normalize_citations(text, "flat", records),
                         normalize.normalize_citations(text, "bound", bound))

    def test_no_prefix_whitespace_joining_or_unsupplied_record_repair(self):
        citations = ["record:", "record:1-extra", " record:1", "record:1 ", "record:1/source:p1", "record:2"]
        text = json.dumps({"citations": citations})
        result = normalize.normalize_citations(text, "flat", [record()])
        self.assertEqual(result["parsed_after"]["citations"], citations)
        self.assertEqual(result["changed_reference_count"], 0)
        self.assertEqual(result["rejected_reference_count"], len(citations))

    def test_duplicate_records_same_target_allowed_multiple_targets_refused(self):
        text = '{"citations":["record:1","record:ok"]}'
        evidence = [record(), record(), record("record:ok", "source:ok")]
        result = normalize.normalize_citations(text, "flat", evidence)
        self.assertEqual(result["changed_reference_count"], 2)
        evidence.append(record("record:1", "source:p2"))
        result = normalize.normalize_citations(text, "flat", evidence)
        self.assertEqual(result["parsed_after"]["citations"], ["record:1", "source:ok"])
        self.assertEqual(result["citation_diagnostics"][0]["status"], "ambiguous_record_id")

    def test_chunk_record_namespace_collision_is_refused_unless_same_target(self):
        evidence = [record("both", "source:p1"), record("other", "both")]
        result = normalize.normalize_citations('{"citations":["both"]}', "flat", evidence)
        self.assertEqual(result["parsed_after"]["citations"], ["both"])
        self.assertEqual(result["citation_diagnostics"][0]["status"], "ambiguous_chunk_record_id")
        result = normalize.normalize_citations('{"citations":["both"]}', "flat", [record("both", "both")])
        self.assertEqual(result["citation_diagnostics"][0]["status"], "already_chunk_id")

    def test_raw_uses_only_actual_chunk_ids(self):
        evidence = {"chunks": [{"chunk_id": "source:p1", "record_id": "record:1", "text": "record:other"}]}
        result = normalize.normalize_citations('{"citations":["source:p1","record:1","record:other"]}', "raw", evidence)
        self.assertEqual(result["changed_reference_count"], 0)
        self.assertEqual(result["rejected_reference_count"], 2)

    def test_entire_json_or_single_json_fence_only(self):
        obj = '{"answer_text":"literal ``` stays","citations":["record:1"]}'
        for text, wrapper in ((" \n" + obj + "\t", "json"), ("```json\n" + obj + "\n```", "json_fence"),
                              (" \r\n```json\r\n" + obj + "\r\n```\r\n", "json_fence")):
            result = normalize.normalize_citations(text, "flat", [record()])
            self.assertEqual(result["wrapper"], wrapper)
            self.assertEqual(result["original_text"], text)
            self.assertEqual(result["changed_reference_count"], 1)

    def test_truncation_explanation_multiple_objects_and_nested_fences_not_repaired(self):
        candidates = ['explanation {"citations":["record:1"]}', '{"citations":["record:1"]} trailing',
                      '{"citations":["record:1"]', '{} {}', '[]',
                      '```\n{}\n```', '```JSON\n{}\n```', '````json\n{}\n````',
                      '```json\n```json\n{}\n```\n```', 'prefix\n```json\n{}\n```']
        for text in candidates:
            result = normalize.normalize_citations(text, "flat", [record()])
            self.assertEqual(result["parse_status"], "rejected", text)
            self.assertEqual(result["original_text"], text)
            self.assertIsNone(result["parsed_before"])
            self.assertIsNone(result["parsed_after"])
            self.assertFalse(result["changes"])

    def test_duplicate_keys_and_nonfinite_numbers_are_not_silently_coerced(self):
        for text in ('{"x":1,"x":2}', '{"citations":[{"chunk_id":"a","chunk_id":"record:1"}]}',
                     '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            self.assertEqual(normalize.normalize_citations(text, "flat", [record()])["parse_status"], "rejected")

    def test_empty_missing_invalid_and_unsupported_citations_stay_exact(self):
        for obj, status in (({"citations": []}, "list"), ({"answer_text": "fixture"}, "missing"),
                            ({"citations": None}, "invalid_type"), ({"citations": "record:1"}, "invalid_type"),
                            ({"citations": [None, True, 7, {"record_id": "record:1"}, {"chunk_id": ["record:1"]}]}, "list")):
            result = normalize.normalize_citations(json.dumps(obj), "flat", [record()])
            self.assertEqual(result["citation_list_status"], status)
            self.assertIsNone(normalize.representation._difference(obj, result["parsed_after"]))
            self.assertEqual(result["changed_reference_count"], 0)

    def test_invariant_guard_rejects_body_type_order_deletion_and_unlogged_changes(self):
        before = {"answer_text": "fixture", "flag": True, "citations": ["a", "b", "a", "invalid"]}
        for after in ({**before, "answer_text": "changed"}, {**before, "flag": 1},
                      {**before, "citations": ["b", "a", "a", "invalid"]},
                      {**before, "citations": ["a", "b", "a"]}, {**before, "citations": ["a", "b", "invalid"]}):
            with self.assertRaises(ValueError):
                normalize.validate_only_id_changes(before, after, [])


def build_fixture(root):
    def save(relative, obj):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj)); return {"path": relative, "sha256": normalize.sha(path)}
    outputs, rows = {}, []
    qids = [f"fixture-{i}" for i in range(96)]
    semantic = [record()]
    evidence = {"raw": {"chunks": [{"chunk_id": "source:p1", "text": "PRIVATE_SOURCE"}]},
                "flat": semantic, "bound": json.loads(normalize.representation.encode_bound(semantic))}
    for qid in qids:
        for arm in normalize.ARMS:
            rows.append({"question_id": qid, "arm": arm, "messages": [{"role": "system", "content": "fixture"},
                {"role": "user", "content": json.dumps({"question": "PRIVATE_QUESTION", "evidence": evidence[arm]})}],
                "input_tokens": 3, "truncated": False})
            binding = save(f"data/execution/{arm}/{qid}.json", {"question_id": qid, "arm": arm, "status": "generated",
                "generated_text": '{"answer_text":"PRIVATE_ANSWER","citations":["record:1","unknown"]}'})
            outputs[binding["path"]] = binding["sha256"]
    input_path = root / normalize.INPUT_PATH; input_path.parent.mkdir(parents=True)
    input_path.write_text("\n".join(json.dumps(r) for r in rows))
    input_binding = {"path": normalize.INPUT_PATH, "sha256": normalize.sha(input_path)}
    execution = save("results/execution_protocol.json", {"status": "frozen", "inputs": input_binding,
        "question_ids": qids, "execution": {"output_root": "data/execution"}})
    private_audit = save("data/mechanical.json", {"passed": True, "protocol_sha256": execution["sha256"],
        "artifact_sha256": {str((root / p).resolve()): h for p, h in outputs.items()}})
    audit = save("results/execution_audit.json", {"passed": True, "protocol_sha256": execution["sha256"],
        "private_audit": private_audit})
    code = {}
    for path in normalize.CODE_PATHS:
        p = root / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# synthetic bound code\n")
        code[path] = normalize.sha(p)
    protocol = save("results/analysis_protocol.json", {"schema": "radar_citation_normalization_protocol_v1", "status": "frozen",
        "rules_version": normalize.RULES_VERSION, "execution_protocol": execution, "execution_audit": audit,
        "inputs": input_binding, "code_sha256": code, "model_outputs_sha256": outputs})
    return protocol, input_binding, outputs


class NormalizationBuild(unittest.TestCase):
    def test_synthetic_288_build_binds_artifacts_preserves_originals_and_private_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); protocol, inputs, originals = build_fixture(root)
            private, public = root / "data/analysis/normalized.json", root / "results/analysis/counts.json"
            with patch.object(normalize, "ROOT", root), patch.object(normalize, "INPUT_SHA", inputs["sha256"]):
                summary = normalize.build(root / protocol["path"], protocol["sha256"], private, public)
                self.assertEqual(summary["arms"]["raw"]["changed_outputs"], 0)
                self.assertEqual(summary["arms"]["flat"]["changed_outputs"], 96)
                self.assertEqual(summary["arms"]["bound"]["changed_outputs"], 96)
                self.assertNotIn("PRIVATE_", json.dumps(summary))
                self.assertEqual(len(json.loads(private.read_text())["rows"]), 288)
                self.assertTrue(private.with_suffix(".manifest.json").exists())
                self.assertTrue(all(normalize.sha(root / p) == h for p, h in originals.items()))
                with self.assertRaisesRegex(ValueError, "new private JSON"):
                    normalize.build(root / protocol["path"], protocol["sha256"], private, public)

    def test_tampered_input_output_attestation_and_code_fail_before_writing(self):
        for target in ("input", "output", "attestation", "code"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve(); protocol, inputs, originals = build_fixture(root)
                path = {"input": inputs["path"], "output": next(iter(originals)),
                        "attestation": "data/mechanical.json", "code": normalize.CODE_PATHS[0]}[target]
                (root / path).write_text("changed")
                with patch.object(normalize, "ROOT", root), patch.object(normalize, "INPUT_SHA", inputs["sha256"]):
                    with self.assertRaises(ValueError):
                        normalize.build(root / protocol["path"], protocol["sha256"], root / "data/new/normalized.json", root / "results/new.json")
                self.assertFalse((root / "data/new").exists())

    def test_public_private_path_inversion_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); protocol, inputs, _ = build_fixture(root)
            with patch.object(normalize, "ROOT", root), patch.object(normalize, "INPUT_SHA", inputs["sha256"]):
                with self.assertRaisesRegex(ValueError, "private JSON"):
                    normalize.build(root / protocol["path"], protocol["sha256"], root / "results/leak.json", root / "results/counts.json")
                with self.assertRaisesRegex(ValueError, "public count"):
                    normalize.build(root / protocol["path"], protocol["sha256"], root / "data/new.json", root / "data/counts.json")


if __name__ == "__main__":
    unittest.main()
