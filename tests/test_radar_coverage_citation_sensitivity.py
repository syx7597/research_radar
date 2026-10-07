"""Synthetic-only citation sensitivity guards; no experiment outputs or gold."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import coverage_citation_sensitivity as sensitivity


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sensitivity.sha(path)}


def support(rid, qid, value=True):
    return {"review_id": rid, "question_id": qid, "citations_support": value,
            "reason_zh": "synthetic rationale", "evidence_basis": []}


class CitationReviewRoster(unittest.TestCase):
    def test_exact_roster_real_booleans_identity_and_packet_binding(self):
        mapping = {"N1": {"question_id": "q1"}, "N2": {"question_id": "q2"}}
        manifest = {"packet_sha256": {"packets/g.json": "pinned"}}
        original = {"reviewer": "synthetic A", "exposure": "synthetic only",
                    "packet_sha256": manifest["packet_sha256"],
                    "rows": [support("N1", "q1"), support("N2", "q2", False)]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "review.json"
            save(path, original)
            self.assertEqual(set(sensitivity.review_rows(path, mapping, manifest)), set(mapping))
            bad_cases = []
            for rows in ([original["rows"][0]], [original["rows"][0]] * 2,
                         original["rows"] + [support("N3", "q3")]):
                bad_cases.append({**original, "rows": rows})
            for value in (0, 1, "true", "false", None):
                changed = deepcopy(original); changed["rows"][0]["citations_support"] = value
                bad_cases.append(changed)
            changed = deepcopy(original); changed["rows"][0]["question_id"] = "wrong"
            bad_cases.append(changed)
            bad_cases.append({**original, "packet_sha256": {"packets/g.json": "changed"}})
            bad_cases.append({**original, "exposure": ""})
            for case in bad_cases:
                save(path, case)
                with self.assertRaises(ValueError):
                    sensitivity.review_rows(path, mapping, manifest)


def input_fixture(root):
    base, private = root / "results/analysis", root / "data/analysis"
    actual = [{"question_id": f"q{i}", "arm": arm, "messages": ["synthetic source only"]}
              for i in range(96) for arm in ("raw", "flat", "bound")]
    input_path = root / "data/inputs.jsonl"; input_path.parent.mkdir(parents=True)
    input_path.write_text("\n".join(json.dumps(row) for row in actual))
    normalized = [{"question_id": r["question_id"], "arm": r["arm"],
                   "input_row_sha256": hashlib.sha256(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
                  for r in actual]
    scoring = {}
    for key in ("original_summary", "original_verdicts", "original_mapping", "evaluation_policy"):
        path = root / f"data/synthetic_{key}.json"; save(path, {"synthetic": True})
        scoring[key] = {"path": str(path.relative_to(root)), "sha256": sensitivity.sha(path)}
    protocol = {"inputs": {"path": "data/inputs.jsonl", "sha256": sensitivity.sha(input_path)}, "scoring_inputs": scoring}
    save(base / "protocol.json", protocol)
    code_path = root / "experiments/frozen.py"; code_path.parent.mkdir()
    code_path.write_text("# synthetic code binding\n")
    save(base / "scoring_code_freeze.json", {"protocol_sha256": sensitivity.sha(base / "protocol.json"),
        "code_sha256": {"experiments/frozen.py": sensitivity.sha(code_path)}})
    refresh_normalized(root, base, private, normalized)
    return base, private, normalized, actual, protocol


def refresh_normalized(root, base, private, rows):
    save(private / "normalized.json", {"rows": rows})
    save(private / "normalized.manifest.json", {"normalized_json_sha256": sensitivity.sha(private / "normalized.json")})
    save(base / "normalization_summary.json", {"protocol_sha256": sensitivity.sha(base / "protocol.json"),
        "private_manifest": {"path": str((private / "normalized.manifest.json").relative_to(root)),
                             "sha256": sensitivity.sha(private / "normalized.manifest.json")}})


class CitationInputBindings(unittest.TestCase):
    def test_amendment_binds_original_freeze_protocol_and_same_module_inventory(self):
        for mode in ("valid", "freeze", "protocol", "inventory", "code"):
            with tempfile.TemporaryDirectory() as d, self.subTest(mode=mode):
                root = Path(d).resolve(); base, private, _, _, _ = input_fixture(root)
                code_path = root / "experiments/frozen.py"
                code_path.write_text("# synthetic registered amendment\n")
                amendment = {"original_code_freeze_sha256": sensitivity.sha(base / "scoring_code_freeze.json"),
                    "protocol_sha256": sensitivity.sha(base / "protocol.json"),
                    "code_sha256": {"experiments/frozen.py": sensitivity.sha(code_path)}}
                if mode == "freeze": amendment["original_code_freeze_sha256"] = "wrong"
                if mode == "protocol": amendment["protocol_sha256"] = "wrong"
                if mode == "inventory": amendment["code_sha256"] = {}
                if mode == "code": amendment["code_sha256"]["experiments/frozen.py"] = "wrong"
                save(base / "scoring_code_amendment_v1.json", amendment)
                with patch.multiple(sensitivity, ROOT=root, BASE=base, PRIVATE=private):
                    if mode == "valid":
                        self.assertEqual(len(sensitivity.inputs()[1]), 288)
                    else:
                        with self.assertRaises(ValueError):
                            sensitivity.inputs()

    def test_bound_inputs_and_per_row_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); base, private, normalized, actual, _ = input_fixture(root)
            with patch.multiple(sensitivity, ROOT=root, BASE=base, PRIVATE=private):
                _, got, sources = sensitivity.inputs()
                self.assertEqual(got, normalized)
                self.assertEqual(sources, actual)

    def test_original_file_and_rebound_wrong_row_hash_fail(self):
        for mode, message in (("original", "Original supplied inputs"), ("row_hash", "different evidence"),
                              ("row_identity", "row identity"), ("missing", "288-row roster"),
                              ("score_input", "Frozen analysis input"), ("code", "scoring code changed")):
            with tempfile.TemporaryDirectory() as d, self.subTest(mode=mode):
                root = Path(d).resolve(); base, private, normalized, _, protocol = input_fixture(root)
                if mode == "original":
                    (root / "data/inputs.jsonl").write_text("{}\n")
                elif mode == "score_input":
                    (root / protocol["scoring_inputs"]["original_mapping"]["path"]).write_text("changed")
                elif mode == "code":
                    (root / "experiments/frozen.py").write_text("changed")
                else:
                    if mode == "row_hash": normalized[0]["input_row_sha256"] = "incorrect"
                    if mode == "row_identity": normalized[0]["question_id"] = "other"
                    if mode == "missing": normalized.pop()
                    refresh_normalized(root, base, private, normalized)
                with patch.multiple(sensitivity, ROOT=root, BASE=base, PRIVATE=private):
                    with self.assertRaisesRegex(ValueError, message):
                        sensitivity.inputs()


def aggregate_fixture(root, disagreement=False):
    base, private = root / "results/analysis", root / "data/analysis"
    package = private / "review"
    qids = [f"q{i}" for i in range(96)]
    policy = {"denominator_each_arm": 96, "question_groups": {q: f"g{i // 12}" for i, q in enumerate(qids)},
              "registered_competing_question_ids": qids[:50], "registered_complex_question_ids": qids[:31],
              "missing_reference_page_question_ids": qids[-4:], "continuation_gate": {
                  "net_joint_correct_bound_minus_flat_at_least": 5, "source_groups_with_positive_net_gain_at_least": 3}}
    changed_pairs = {("q0", "flat"), ("q1", "bound"), ("q2", "flat")}
    oldmap, oldrows, normalized = [], [], []
    for i, qid in enumerate(qids):
        for arm in ("raw", "flat", "bound"):
            rid = f"old_{arm}_{qid}"
            oldmap.append({"review_id": rid, "question_id": qid, "arm": arm})
            oldrows.append({"review_id": rid, "question_id": qid, "answer_correct": qid != "q0",
                "supported_by_supplied_evidence": qid != "q2", "citations_support": (qid, arm) not in changed_pairs,
                "binding_error_any": i % 5 == 0, "binding_error_types": ["condition"] if i % 5 == 0 else [],
                "refusal": False, "redundancy": i % 7 == 0, "reason_zh": "unchanged synthetic body rationale", "evidence_basis": []})
            normalized.append({"question_id": qid, "arm": arm, "changes": [{}] if (qid, arm) in changed_pairs else []})
    oldmap_dict = {r["review_id"]: r for r in oldmap}; oldrow_dict = {r["review_id"]: r for r in oldrows}
    original = sensitivity.summarize(oldrow_dict, oldmap_dict, policy)
    bound_objects = {"original_mapping": {"rows": oldmap}, "original_verdicts": {"rows": oldrows},
                     "original_summary": {"adjudicated": original}, "evaluation_policy": policy}
    protocol = {"scoring_inputs": {key: {"path": key, "sha256": "synthetic-pinned-" + key} for key in bound_objects}}
    digest = lambda path: hashlib.sha256(str(path).encode()).hexdigest()
    manifest = {"normalized_sha256": digest(private / "normalized.json"), "protocol_sha256": digest(base / "protocol.json"),
        "mapping_sha256": digest(package / "mapping_for_adjudicator_only.json"), "rubric_sha256": digest(package / "rubric.json"),
        "packet_sha256": {"packets/g.json": digest(package / "packets/g.json")}}
    newmap = [{"review_id": f"N{i}", "question_id": q, "arm": a} for i, (q, a) in enumerate(sorted(changed_pairs))]
    arows = [support(r["review_id"], r["question_id"], True) for r in newmap]
    brows = deepcopy(arows)
    if disagreement: brows[1]["citations_support"] = False
    objects = {str(package / "manifest.json"): manifest,
        str(package / "mapping_for_adjudicator_only.json"): {"rows": newmap},
        str(package / "packets/g.json"): {"items": [{"supplied_evidence_file": "evidence/e.json", "supplied_evidence_sha256": digest(package / "evidence/e.json")}]},
        str(package / "adjudication.json"): {
            "reviewer_sha256": {f"reviewer_{x}.json": digest(package / f"reviewer_{x}.json") for x in "AB"},
            "packet_sha256": manifest["packet_sha256"],
            "rows": [deepcopy(arows[1])] if disagreement else []}}
    for letter, rows in (("A", arows), ("B", brows)):
        objects[str(package / f"reviewer_{letter}.json")] = {"reviewer": f"synthetic {letter}", "exposure": "fixtures only",
            "packet_sha256": manifest["packet_sha256"], "rows": rows}
    return base, private, protocol, normalized, bound_objects, objects, digest


class CitationAggregation(unittest.TestCase):
    def run_fixture(self, fixture):
        base, private, protocol, normalized, bound_objects, objects, digest = fixture
        root = base.parents[1]; writes = {}
        def fake_read(path):
            return deepcopy(objects[str(Path(path))])  # Unexpected real-file access fails.
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(sensitivity, ROOT=root, BASE=base, PRIVATE=private))
            stack.enter_context(patch.object(sensitivity, "inputs", return_value=(protocol, normalized, [])))
            stack.enter_context(patch.object(sensitivity, "read", side_effect=fake_read))
            stack.enter_context(patch.object(sensitivity, "bound", side_effect=lambda b: deepcopy(bound_objects[b["path"]])))
            stack.enter_context(patch.object(sensitivity, "sha", side_effect=digest))
            stack.enter_context(patch.object(sensitivity, "write_new", side_effect=lambda p, v: writes.update({str(p): deepcopy(v)})))
            result = sensitivity.aggregate()
        return result, writes

    def test_only_citation_support_changes_with_all_288_and_96_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = aggregate_fixture(Path(d).resolve())
            result, writes = self.run_fixture(fixture)
            base, private, _, _, originals, *_ = fixture
            final = {r["review_id"]: r for r in writes[str(private / "citation_support_verdicts.json")]["rows"]}
            old = {r["review_id"]: r for r in originals["original_verdicts"]["rows"]}
            self.assertEqual(len(final), 288)
            changed = 0
            for rid in old:
                self.assertEqual({k:v for k,v in old[rid].items() if k != "citations_support"},
                                 {k:v for k,v in final[rid].items() if k != "citations_support"})
                changed += old[rid]["citations_support"] != final[rid]["citations_support"]
            self.assertEqual(changed, 3)
            for arm in ("raw", "flat", "bound"):
                self.assertEqual(result["original"][arm]["denominator"], 96)
                self.assertEqual(result["uniform_resolver"][arm]["denominator"], 96)
                for field in ("answer_correct", "binding_error_any", "supported_by_supplied_evidence", "refusal", "redundancy"):
                    self.assertEqual(result["original"][arm][field], result["uniform_resolver"][arm][field])
            self.assertEqual(result["uniform_resolver"]["bound"]["joint_correct"] - result["original"]["bound"]["joint_correct"], 1)
            self.assertEqual(result["uniform_resolver"]["flat"]["joint_correct"], result["original"]["flat"]["joint_correct"])
            public = writes[str(base / "semantic_summary.json")]
            self.assertNotIn("continuation_gate", public["uniform_resolver"])
            self.assertNotIn("continuation_gate", public["original"])
            self.assertFalse(public["body_verdicts_changed"])
            self.assertEqual(public["new_model_calls"], 0)

    def test_missing_disagreement_or_extra_override_is_rejected(self):
        for extra in (False, True):
            with tempfile.TemporaryDirectory() as d:
                fixture = aggregate_fixture(Path(d).resolve(), disagreement=True)
                _, private, _, _, _, objects, _ = fixture
                obj = objects[str(private / "review/adjudication.json")]
                obj["rows"] = obj["rows"] + [support("N0", "q0")] if extra else []
                with self.assertRaisesRegex(ValueError, "Every citation disagreement"):
                    self.run_fixture(fixture)

    def test_inherited_fixed_denominator_cannot_drop_an_original_output(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = aggregate_fixture(Path(d).resolve())
            fixture[4]["original_verdicts"]["rows"].pop()
            with self.assertRaisesRegex(ValueError, "Do not remove failures"):
                self.run_fixture(fixture)

    def test_duplicate_adjudication_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = aggregate_fixture(Path(d).resolve(), disagreement=True)
            _, private, _, _, _, objects, _ = fixture
            row = objects[str(private / "review/adjudication.json")]["rows"][0]
            objects[str(private / "review/adjudication.json")]["rows"].append({**row, "citations_support": False})
            with self.assertRaisesRegex(ValueError, "duplicate|Duplicate"):
                self.run_fixture(fixture)

    def test_adjudication_requires_current_review_and_packet_hashes(self):
        for field in ("reviewer_sha256", "packet_sha256"):
            with tempfile.TemporaryDirectory() as d, self.subTest(field=field):
                fixture = aggregate_fixture(Path(d).resolve(), disagreement=True)
                _, private, _, _, _, objects, _ = fixture
                objects[str(private / "review/adjudication.json")][field] = {"wrong": "wrong"}
                with self.assertRaisesRegex(ValueError, "Adjudication not bound"):
                    self.run_fixture(fixture)

    def test_valid_disagreement_adjudication_preserves_body_and_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = aggregate_fixture(Path(d).resolve(), disagreement=True)
            result, writes = self.run_fixture(fixture)
            private, originals = fixture[1], fixture[4]
            final = writes[str(private / "citation_support_verdicts.json")]["rows"]
            self.assertEqual(len(final), 288)
            for before, after in zip(originals["original_verdicts"]["rows"], final):
                self.assertEqual({k: v for k, v in before.items() if k != "citations_support"},
                                 {k: v for k, v in after.items() if k != "citations_support"})
            self.assertTrue(next(r for r in final if r["review_id"] == "old_bound_q1")["citations_support"])
            self.assertEqual(result["uniform_resolver"]["bound"]["denominator"], 96)


if __name__ == "__main__":
    unittest.main()
