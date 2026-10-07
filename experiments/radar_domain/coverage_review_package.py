"""Prepare method-anonymous review copies of one frozen evaluation, without scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .coverage_representation import decode_bound, decode_flat, validate_equal_information
from .source_readings import ROOT, sha, write_new


def canonical_evidence(arm, evidence):
    if arm == "raw":
        return evidence
    encoded = json.dumps(evidence, ensure_ascii=False)
    return decode_flat(encoded) if arm == "flat" else decode_bound(encoded)


def build(output):
    output = Path(output).resolve()
    private_root = ROOT / "data/radar_sources_v2"
    if output.exists() or not output.is_relative_to(private_root):
        raise ValueError("Use a new private review directory")
    protocol_path = ROOT / "results/radar_domain/coverage_v2/protocol.json"
    policy_path = ROOT / "results/radar_domain/coverage_v2/evaluation_policy.json"
    protocol, policy = [json.loads(p.read_text()) for p in (protocol_path, policy_path)]
    if sha(policy_path) != protocol["evaluation_policy"]["sha256"]:
        raise ValueError("Scoring policy differs from execution protocol")
    for arm in ("raw", "flat", "bound"):
        receipt = json.loads((private_root / "evaluation_v1" / arm / "runtime.json").read_text())
        if receipt["protocol_sha256"] != sha(protocol_path) or receipt["status"] != "completed":
            raise ValueError("Only the completed frozen round is accepted by this packager")
    for path, digest in policy["reference_inputs_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen reference changed: {path}")
    inputs_path = private_root / "representation_v1/semantic_payload_v1/inputs.jsonl"
    inputs = [json.loads(line) for line in inputs_path.read_text().splitlines()]
    # The protocol binds this file; do not infer validity from the number of rows.
    if sha(inputs_path) != protocol["inputs"]["sha256"]:
        raise ValueError("Frozen model inputs changed")
    references = {}
    for domain in ("weather", "marine"):
        packet = json.loads((private_root / f"qa_v1/{domain}.author.json").read_text())
        references.update({row["question_id"]: row for row in packet["questions"]})
    rubric = json.loads((private_root / "qa_v1/author_rubric_response.json").read_text())
    interpretations = {r["question_id"]: r for r in rubric["required_facts_scoring_interpretation"]}
    expected = {(q, a) for q in protocol["question_ids"] for a in ("raw", "flat", "bound")}
    if len(inputs) != 288 or {(r["question_id"], r["arm"]) for r in inputs} != expected:
        raise ValueError("Incomplete or duplicate input roster")
    records, canonical_by_question = [], {}
    for row in inputs:
        qid, arm = row["question_id"], row["arm"]
        supplied = json.loads(row["messages"][1]["content"])
        if supplied["question"] != references[qid]["question_zh"]:
            raise ValueError("Question differs from frozen reference")
        evidence = canonical_evidence(arm, supplied["evidence"])
        if arm != "raw":
            if qid in canonical_by_question:
                validate_equal_information(canonical_by_question[qid], flat=json.dumps(evidence, ensure_ascii=False))
                # Use one serialization too, hiding object-field order from the codec.
                evidence = canonical_by_question[qid]
            else:
                canonical_by_question[qid] = evidence
            if any(set(r["citation"]) != {"chunk_id"} for r in evidence):
                raise ValueError("Do not add removed archival citation information")
        path = private_root / "evaluation_v1" / arm / f"{qid}.json"
        prediction = json.loads(path.read_text())
        if (prediction["question_id"], prediction["arm"], prediction["status"]) != (qid, arm, "generated"):
            raise ValueError("Output identity mismatch")
        records.append((qid, arm, supplied["question"], evidence, prediction["generated_text"], sha(path)))
    random.Random(policy["review"]["review_package_shuffle_seed"]).shuffle(records)
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    (output / "packets").mkdir()
    mapping, packets, counts = [], {}, Counter()
    for index, (qid, arm, question, evidence, text, digest) in enumerate(records, 1):
        review_id = f"V{index:03d}"
        encoded = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        evidence_digest = hashlib.sha256(encoded.encode()).hexdigest()
        evidence_name = f"evidence/{evidence_digest}.json"
        ep = output / evidence_name
        if not ep.exists():
            ep.write_text(encoded, encoding="utf-8")
        ref = references[qid]
        item = {"review_id": review_id, "question_id": qid, "question": question,
                "model_output_verbatim": text, "supplied_evidence_file": evidence_name,
                "supplied_evidence_sha256": evidence_digest,
                "reference_truth_only_not_supplied_evidence": {"reference": ref["reference"], "support": ref["support"]},
                "frozen_scoring_interpretation": interpretations.get(qid)}
        family = ref["family_id"]
        packets.setdefault(family, []).append(item)
        mapping.append({"review_id": review_id, "question_id": qid, "arm": arm, "output_sha256": digest})
        counts[arm] += 1
    for family, rows in packets.items():
        write_new(output / "packets" / f"{family}.json", {"items": rows})
    write_new(output / "mapping_for_adjudicator_only.json", {"rows": mapping})
    manifest = {"schema": "radar_coverage_anonymous_review_package_v1", "rows": len(records),
                "counts": dict(counts), "deduplicated_outputs": 0,
                "shared_evidence_storage_only": True, "evidence_files": len(list((output / "evidence").glob("*.json"))),
                "canonical_equal_information_pairs_verified": len(canonical_by_question),
                "protocol_sha256": sha(protocol_path), "policy_sha256": sha(policy_path),
                "inputs_sha256": sha(inputs_path), "builder_sha256": sha(Path(__file__)),
                "mapping_sha256": sha(output / "mapping_for_adjudicator_only.json"),
                "packet_files": {str(p.relative_to(output)): sha(p) for p in sorted((output / "packets").glob("*.json"))},
                "limits": "Method labels hidden, but evidence style may reveal raw. Reference source quotes are truth references only, not additional supplied evidence. No judgement or accuracy computed."}
    write_new(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))


if __name__ == "__main__":
    main()
