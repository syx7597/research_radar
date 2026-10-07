"""Separate, posthoc citation-support review; never rescore answer bodies."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .coverage_citation_normalize import citation_index, validate_only_id_changes
from .coverage_review_package import canonical_evidence
from .coverage_review_summary import summarize, require
from .source_readings import ROOT, sha, write_new

BASE = ROOT / "results/radar_domain/coverage_v2_citation_sensitivity_v1"
PRIVATE = ROOT / "data/radar_sources_v2/citation_sensitivity_v1"


def read(path):
    return json.loads(Path(path).read_text())


def bound(binding):
    path = ROOT / binding["path"]
    require(sha(path) == binding["sha256"], "Frozen analysis input changed")
    return read(path)


def inputs():
    protocol = read(BASE / "protocol.json")
    code = read(BASE / "scoring_code_freeze.json")
    require(code["protocol_sha256"] == sha(BASE / "protocol.json"), "Wrong scoring code registration")
    amendment_path = BASE / "scoring_code_amendment_v1.json"
    if amendment_path.exists():
        amendment = read(amendment_path)
        require(amendment["original_code_freeze_sha256"] == sha(BASE / "scoring_code_freeze.json")
                and amendment["protocol_sha256"] == sha(BASE / "protocol.json"), "Wrong scoring code amendment")
        require(set(amendment["code_sha256"]) == set(code["code_sha256"]), "Code amendment changed module inventory")
        code = amendment
    for name, digest in code["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Frozen scoring code changed")
    norm = read(BASE / "normalization_summary.json")
    require(norm["protocol_sha256"] == sha(BASE / "protocol.json"), "Wrong normalization protocol")
    manifest = bound(norm["private_manifest"])
    normalized_path = PRIVATE / "normalized.json"
    # The normalizer manifest binds the full original/normalized paired objects.
    require(sha(normalized_path) == manifest["normalized_json_sha256"], "Normalized objects changed")
    normalized = read(normalized_path)["rows"]
    input_path = ROOT / protocol["inputs"]["path"]
    require(sha(input_path) == protocol["inputs"]["sha256"], "Original supplied inputs changed")
    actual = [json.loads(s) for s in input_path.read_text().splitlines() if s.strip()]
    require(len(actual) == len(normalized) == 288, "Fixed 288-row roster required")
    for source, row in zip(actual, normalized):
        require((source["question_id"], source["arm"]) == (row["question_id"], row["arm"]), "Normalization/input row identity changed")
        encoded = json.dumps(source, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        require(hashlib.sha256(encoded.encode()).hexdigest() == row["input_row_sha256"], "Normalized row used different evidence")
    for key in ("original_summary", "original_verdicts", "original_mapping", "evaluation_policy"):
        bound(protocol["scoring_inputs"][key])
    return protocol, normalized, actual


def prepare():
    protocol, normalized, input_rows = inputs()
    out = PRIVATE / "review"
    require(not out.exists(), "Use a new review package")
    out.mkdir()
    actual = {(r["question_id"], r["arm"]): r for r in input_rows}
    policy = bound(protocol["scoring_inputs"]["evaluation_policy"])
    changed = [r for r in normalized if r["changes"]]
    random.Random(20261008).shuffle(changed)
    packets, mapping = {}, []
    for index, row in enumerate(changed, 1):
        qid, arm = row["question_id"], row["arm"]
        validate_only_id_changes(row["parsed_before"], row["parsed_after"], row["changes"])
        payload = json.loads(actual[qid, arm]["messages"][1]["content"])
        evidence = canonical_evidence(arm, payload["evidence"])
        chunks, _ = citation_index(arm, payload["evidence"])
        encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        evidence_path = out / "evidence" / (digest + ".json")
        evidence_path.parent.mkdir(exist_ok=True)
        if not evidence_path.exists():
            evidence_path.write_text(encoded)
        rid = f"N{index:03d}"
        item = {"review_id": rid, "question_id": qid, "question": payload["question"],
                "normalized_output": row["parsed_after"], "supplied_chunk_ids": sorted(chunks),
                "supplied_evidence_file": str(evidence_path.relative_to(out)),
                "supplied_evidence_sha256": sha(evidence_path)}
        packets.setdefault(policy["question_groups"][qid], []).append(item)
        mapping.append({"review_id": rid, "question_id": qid, "arm": arm})
    paths = {}
    for group, items in sorted(packets.items()):
        path = out / "packets" / (group + ".json")
        write_new(path, {"schema": "normalized_citation_review_packet_v1", "items": items})
        paths[str(path.relative_to(out))] = sha(path)
    write_new(out / "mapping_for_adjudicator_only.json", {"rows": mapping})
    rubric = {"task": "Judge citations_support only, against the actual supplied evidence. Do not rescore answer correctness, refusal, binding errors or compare methods.",
              "rules": ["Every listed citation ID must be an actual supplied chunk ID. Unknown, absent, malformed, ambiguous, or empty citations fail.",
                        "The union of cited supplied chunks must entail every substantive assertion in the answer. A valid identifier alone is insufficient.",
                        "Use only the supplied evidence file; a full source page or reference answer not in this file cannot repair missing evidence.",
                        "Question context can supply explicitly stated conditions. Weakened but logically entailed statements may pass support, even if original answer correctness is false.",
                        "An extra supplied valid but irrelevant citation does not alone fail; no new per-citation precision threshold.",
                        "Do not infer an answer from citations alone or fill in omitted answer text."],
              "review_schema": {"reviewer": "agent identity", "exposure": "disclose previous experiment exposure; AI, not human", "packet_sha256": paths,
                                "rows": [{"review_id": "N...", "question_id": "RQ2-...", "citations_support": False, "reason_zh": "rationale", "evidence_basis": []}]}}
    write_new(out / "rubric.json", rubric)
    write_new(out / "manifest.json", {"schema": "normalized_citation_review_manifest_v1", "rows": len(changed),
              "protocol_sha256": sha(BASE / "protocol.json"), "normalized_sha256": sha(PRIVATE / "normalized.json"),
              "mapping_sha256": sha(out / "mapping_for_adjudicator_only.json"), "packet_sha256": paths,
              "rubric_sha256": sha(out / "rubric.json"), "builder_sha256": sha(Path(__file__)),
              "limits": "Changed citations only, no reference answers. Prior reviewers may remember outputs; not fresh or guaranteed blind review. Method labels and original verdicts omitted."})
    return {"review_items": len(changed), "packet_groups": len(packets)}


def review_rows(path, mapping, manifest):
    obj = read(path)
    require(bool(obj.get("reviewer")) and bool(obj.get("exposure")), "Missing review provenance")
    require(obj["packet_sha256"] == manifest["packet_sha256"], "Wrong review packets")
    rows = obj["rows"]
    require(len(rows) == len(mapping) and {r["review_id"] for r in rows} == set(mapping), "Review roster incomplete or duplicated")
    for row in rows:
        require(row["question_id"] == mapping[row["review_id"]]["question_id"], "Wrong review identity")
        require(type(row["citations_support"]) is bool and bool(row["reason_zh"]) and type(row["evidence_basis"]) is list, "Malformed support judgment")
    return {r["review_id"]: r for r in rows}


def aggregate():
    protocol, normalized, _ = inputs()
    package = PRIVATE / "review"
    manifest = read(package / "manifest.json")
    require(manifest["normalized_sha256"] == sha(PRIVATE / "normalized.json") and manifest["protocol_sha256"] == sha(BASE / "protocol.json"), "Review input changed")
    require(manifest["mapping_sha256"] == sha(package / "mapping_for_adjudicator_only.json") and manifest["rubric_sha256"] == sha(package / "rubric.json"), "Review mapping/rubric changed")
    for name, digest in manifest["packet_sha256"].items():
        require(sha(package / name) == digest, "Review packet changed")
        for item in read(package / name)["items"]:
            require(sha(package / item["supplied_evidence_file"]) == item["supplied_evidence_sha256"], "Reviewed evidence changed")
    newmap = {r["review_id"]: r for r in read(package / "mapping_for_adjudicator_only.json")["rows"]}
    require({(r["question_id"], r["arm"]) for r in newmap.values()} == {(r["question_id"], r["arm"]) for r in normalized if r["changes"]}, "Changed-output review selection changed")
    reviews = [package / f"reviewer_{x}.json" for x in "AB"]
    require(read(reviews[0])["reviewer"] != read(reviews[1])["reviewer"], "Reviewers must be distinct")
    a, b = [review_rows(p, newmap, manifest) for p in reviews]
    disagreements = {rid for rid in a if a[rid]["citations_support"] != b[rid]["citations_support"]}
    overrides_path = package / "adjudication.json"
    adj = read(overrides_path)
    require(adj["reviewer_sha256"] == {p.name: sha(p) for p in reviews}
            and adj["packet_sha256"] == manifest["packet_sha256"], "Adjudication not bound to these original reviews/packets")
    overrides = {r["review_id"]: r for r in adj["rows"]}
    require(len(overrides) == len(adj["rows"]), "Duplicate adjudication review_id")
    require(set(overrides) == disagreements, "Every citation disagreement needs explicit adjudication, no extra overrides")
    for rid, row in overrides.items():
        require(row["question_id"] == newmap[rid]["question_id"] and type(row["citations_support"]) is bool and bool(row["reason_zh"]), "Invalid adjudication")
    mapping = {r["review_id"]: r for r in bound(protocol["scoring_inputs"]["original_mapping"])["rows"]}
    pair_to_id = {(r["question_id"], r["arm"]): rid for rid, r in mapping.items()}
    oldrows = {r["review_id"]: r for r in bound(protocol["scoring_inputs"]["original_verdicts"])["rows"]}
    final = deepcopy(oldrows)
    for rid in a:
        identity = newmap[rid]
        verdict = overrides.get(rid, a[rid])
        final[pair_to_id[identity["question_id"], identity["arm"]]]["citations_support"] = verdict["citations_support"]
    for rid in oldrows:
        require({k: v for k, v in oldrows[rid].items() if k != "citations_support"} == {k: v for k, v in final[rid].items() if k != "citations_support"}, "A body/evidence/binding verdict changed")
    policy = bound(protocol["scoring_inputs"]["evaluation_policy"])
    original, resolved = summarize(oldrows, mapping, policy), summarize(final, mapping, policy)
    require(original == bound(protocol["scoring_inputs"]["original_summary"])["adjudicated"], "Original counts no longer reproduce")
    # No posthoc continuation/significance gate. Retain inherited fields only as frozen annotations.
    for summary in (original, resolved):
        summary.pop("continuation_gate")
    private_path = PRIVATE / "citation_support_verdicts.json"
    write_new(private_path, {"rows": list(final.values()), "only_mutable_verdict": "citations_support", "citation_disagreements": sorted(disagreements),
                            "normalization_sha256": sha(PRIVATE / "normalized.json"), "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [*reviews, overrides_path]}})
    public = {"schema": "radar_uniform_citation_sensitivity_summary_v1", "posthoc": True,
              "protocol_sha256": sha(BASE / "protocol.json"), "code_sha256": sha(Path(__file__)),
              "normalization_summary_sha256": sha(BASE / "normalization_summary.json"),
              "original_summary_sha256": protocol["scoring_inputs"]["original_summary"]["sha256"],
              "evaluated_outputs": 288, "changed_outputs_rereviewed": len(newmap), "citation_review_disagreements": len(disagreements),
              "original": original, "uniform_resolver": resolved,
              "private_verdicts": {"path": str(private_path.relative_to(ROOT)), "sha256": sha(private_path)},
              "answer_body_changed": False, "body_verdicts_changed": False, "new_model_calls": 0,
              "review_provenance": [{k: read(p)[k] for k in ("reviewer", "exposure")} for p in reviews],
              "limits": "Posthoc exact-ID engineering baseline, not a new model prediction, original primary result, human gold, or mechanistic proof. Changed citations re-reviewed by two AI agents; unchanged support scores inherited. All body, supplied-evidence, binding, refusal and redundancy annotations inherited; only citation support and resulting joint success re-evaluated. No new training or significance gate."}
    write_new(BASE / "semantic_summary.json", public)
    return {"changed_reviews": len(newmap), "disagreements": len(disagreements), "original": original["overall"], "uniform_resolver": resolved["overall"]}


def audit_prepared():
    """Verify the existing review view equals the exact hashed normalized input."""
    protocol, normalized, actual = inputs()
    source = {(r["question_id"], r["arm"]): r for r in actual}
    norms = {(r["question_id"], r["arm"]): r for r in normalized}
    package = PRIVATE / "review"
    manifest = read(package / "manifest.json")
    mapping = {r["review_id"]: r for r in read(package / "mapping_for_adjudicator_only.json")["rows"]}
    require(sha(package / "mapping_for_adjudicator_only.json") == manifest["mapping_sha256"], "Review map changed")
    seen = set()
    for name, digest in manifest["packet_sha256"].items():
        require(sha(package / name) == digest, "Packet bytes changed")
        for item in read(package / name)["items"]:
            rid = item["review_id"]
            require(rid not in seen, "Duplicate review item")
            seen.add(rid)
            m = mapping[rid]
            key = m["question_id"], m["arm"]
            payload = json.loads(source[key]["messages"][1]["content"])
            require(item["question_id"] == key[0] and item["question"] == payload["question"] and item["normalized_output"] == norms[key]["parsed_after"], "Review output/question changed")
            epath = package / item["supplied_evidence_file"]
            require(sha(epath) == item["supplied_evidence_sha256"] and read(epath) == canonical_evidence(key[1], payload["evidence"]), "Review did not preserve supplied evidence")
            require(set(item["supplied_chunk_ids"]) == citation_index(key[1], payload["evidence"])[0], "Supplied chunk roster changed")
    require(seen == set(mapping) and {(r["question_id"], r["arm"]) for r in mapping.values()} == {k for k, r in norms.items() if r["changes"]}, "Wrong changed-output roster")
    return {"passed": True, "reviewed_output_rows": len(seen), "input_sha256": protocol["inputs"]["sha256"],
            "code_freeze_sha256": sha(BASE / "scoring_code_freeze.json"), "review_manifest_sha256": sha(package / "manifest.json"),
            "note": "Initial package creation preceded addition of per-row input/hash guards; this independent validation rechecked all existing packet question, normalized output and actual evidence bytes before citation reviewers accessed them. Original package not rewritten."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "aggregate", "audit"))
    args = parser.parse_args()
    if args.phase == "audit":
        result = audit_prepared()
        write_new(BASE / "pre_review_audit.json", result)
    else:
        result = prepare() if args.phase == "prepare" else aggregate()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
