"""Validate original semantic reviews and aggregate the frozen paired comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .source_readings import ROOT, sha, write_new

BOOL_FIELDS = ("answer_correct", "supported_by_supplied_evidence", "citations_support",
               "binding_error_any", "refusal", "redundancy")
FIELDS = BOOL_FIELDS + ("binding_error_types",)
ARMS = ("raw", "flat", "bound")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_verdict(row):
    require(all(type(row.get(k)) is bool for k in BOOL_FIELDS), "Review fields must be actual booleans")
    types = row.get("binding_error_types")
    require(type(types) is list and len(types) == len(set(types))
            and set(types) <= {"subject", "component", "quantity_form", "unit", "condition"}, "Invalid binding error types")
    require(row["binding_error_any"] == bool(types), "Binding flag/types disagree")
    require(type(row.get("reason_zh")) is str and bool(row["reason_zh"].strip()), "Missing semantic rationale")
    require(type(row.get("evidence_basis")) is list, "Missing evidence basis list")


def load_review(path, mapping, packet_hashes):
    obj = json.loads(Path(path).read_text())
    require(obj.get("packet_sha256") == packet_hashes, "Review not bound to every current packet")
    require(bool(obj.get("reviewer")) and bool(obj.get("exposure")), "Missing reviewer identity/exposure")
    rows = obj["rows"]
    require(len(rows) == len(mapping) and {r["review_id"] for r in rows} == set(mapping), "Incomplete/duplicate review roster")
    for row in rows:
        require(row["question_id"] == mapping[row["review_id"]]["question_id"], "Review question identity changed")
        validate_verdict(row)
    return {r["review_id"]: r for r in rows}


def differing_fields(left, right):
    return [k for k in FIELDS if (set(left[k]) != set(right[k]) if k == "binding_error_types" else left[k] != right[k])]


def merge_reviews(first, second, overrides):
    require(set(first) == set(second), "Review rosters differ")
    edits = {r["review_id"]: r for r in overrides}
    require(len(edits) == len(overrides) and set(edits) <= set(first), "Invalid duplicate/extra adjudication")
    disagreements = {rid: differing_fields(first[rid], second[rid]) for rid in first}
    disagreements = {rid: fields for rid, fields in disagreements.items() if fields}
    require(set(disagreements) <= set(edits), "Every disputed verdict needs explicit adjudication")
    merged = {}
    for rid, a in first.items():
        row = edits.get(rid, a)
        validate_verdict(row)
        require(row["question_id"] == a["question_id"], "Adjudication changed question")
        merged[rid] = row
    return merged, disagreements


def joint(row):
    return all(row[k] for k in BOOL_FIELDS[:3])


def summarize(rows, mapping, policy):
    by_arm = {arm: {} for arm in ARMS}
    for rid, row in rows.items():
        identity = mapping[rid]
        arm, qid = identity["arm"], identity["question_id"]
        require(qid not in by_arm[arm], "Duplicate question/arm score")
        by_arm[arm][qid] = row
    questions = set(policy["question_groups"])
    denominator = policy["denominator_each_arm"]
    require(len(questions) == denominator and all(set(v) == questions for v in by_arm.values()), "Do not remove failures/missing-evidence questions")

    def counts(ids):
        result = {}
        for arm, arm_rows in by_arm.items():
            result[arm] = {"denominator": len(ids), "joint_correct": sum(joint(arm_rows[q]) for q in ids),
                           **{key: sum(arm_rows[q][key] for q in ids) for key in BOOL_FIELDS}}
        wins = sum(joint(by_arm["bound"][q]) and not joint(by_arm["flat"][q]) for q in ids)
        losses = sum(joint(by_arm["flat"][q]) and not joint(by_arm["bound"][q]) for q in ids)
        result["paired_bound_minus_flat"] = {"wins": wins, "losses": losses, "net": wins - losses,
                                               "percentage_points": 100 * (wins - losses) / len(ids) if ids else None}
        return result

    overall = counts(questions)
    groups = {g: counts({q for q in questions if policy["question_groups"][q] == g})
              for g in sorted(set(policy["question_groups"].values()))}
    gate = policy["continuation_gate"]
    positives = sum(g["paired_bound_minus_flat"]["net"] > 0 for g in groups.values())
    conditions = {"net_gain": overall["paired_bound_minus_flat"]["net"] >= gate["net_joint_correct_bound_minus_flat_at_least"],
                  "source_groups": positives >= gate["source_groups_with_positive_net_gain_at_least"],
                  "binding_errors": overall["bound"]["binding_error_any"] <= overall["flat"]["binding_error_any"]}
    return {"overall": overall, "source_groups": groups,
            "registered_competing_subset": counts(set(policy["registered_competing_question_ids"])),
            "registered_complex_subset": counts(set(policy["registered_complex_question_ids"])),
            "missing_reference_pages_subset": counts(set(policy["missing_reference_page_question_ids"])),
            "continuation_gate": {"conditions": conditions, "positive_source_groups": positives,
                                  "passed": all(conditions.values()), "statistical_significance_claim": False}}


def build(package, adjudication, private_output, public_output):
    package = Path(package).resolve()
    private_output, public_output = Path(private_output).resolve(), Path(public_output).resolve()
    require(package.is_relative_to(ROOT / "data"), "Review package must stay private")
    require(private_output.is_relative_to(ROOT / "data") and not private_output.exists(), "Use a new private verdict file under data")
    require(public_output.is_relative_to(ROOT / "results") and not public_output.exists(), "Use a new public aggregate file under results")
    manifest = json.loads((package / "manifest.json").read_text())
    execution_path = ROOT / "results/radar_domain/coverage_v2/execution_audit.json"
    execution = json.loads(execution_path.read_text())
    require(execution["passed"] is True and execution["protocol_sha256"] == manifest["protocol_sha256"], "A passed matching mechanical audit is required")
    audit_path = ROOT / execution["private_audit"]["path"]
    require(audit_path.resolve().is_relative_to(ROOT / "data") and sha(audit_path) == execution["private_audit"]["sha256"], "Private mechanical attestation changed")
    audit = json.loads(audit_path.read_text())
    require(audit["passed"] is True and audit["protocol_sha256"] == manifest["protocol_sha256"], "Private mechanical attestation must also pass and match")
    policy_path = ROOT / "results/radar_domain/coverage_v2/evaluation_policy.json"
    require(sha(policy_path) == manifest["policy_sha256"], "Policy changed")
    policy = json.loads(policy_path.read_text())
    packet_items = []
    for path, digest in manifest["packet_files"].items():
        require(sha(package / path) == digest, "Anonymous review packet changed")
        packet = json.loads((package / path).read_text())
        packet_items.extend(packet["items"])
        for item in packet["items"]:
            evidence = (package / item["supplied_evidence_file"]).resolve()
            require(evidence.is_relative_to(package.resolve() / "evidence")
                    and sha(evidence) == item["supplied_evidence_sha256"], "Supplied review evidence changed")
    mapping_path = package / "mapping_for_adjudicator_only.json"
    require(sha(mapping_path) == manifest["mapping_sha256"], "Mapping changed")
    mapping_rows = json.loads(mapping_path.read_text())["rows"]
    mapping = {r["review_id"]: r for r in mapping_rows}
    require(len(mapping) == 288 and len(mapping_rows) == 288, "Incomplete mapping")
    items_by_id = {r["review_id"]: r for r in packet_items}
    require(len(packet_items) == 288 and set(items_by_id) == set(mapping), "Anonymous packet roster is incomplete or duplicated")
    for row in mapping_rows:
        prediction = ROOT / "data/radar_sources_v2/evaluation_v1" / row["arm"] / f'{row["question_id"]}.json'
        require(sha(prediction) == row["output_sha256"] == audit["artifact_sha256"].get(str(prediction.resolve())),
                "Reviewed output does not match the mechanically audited bytes")
        saved, item = json.loads(prediction.read_text()), items_by_id[row["review_id"]]
        require(item["question_id"] == saved["question_id"] == row["question_id"]
                and item["model_output_verbatim"] == saved["generated_text"], "Anonymous item is mapped to a different output")
    reviews = [package / f"reviewer_{letter}.json" for letter in "AB"]
    declarations = [json.loads(p.read_text()) for p in reviews]
    require(sha(reviews[0]) != sha(reviews[1]) and declarations[0]["reviewer"] != declarations[1]["reviewer"], "Two distinct original reviewers are required")
    first, second = [load_review(path, mapping, manifest["packet_files"]) for path in reviews]
    adj = json.loads(Path(adjudication).read_text())
    require(adj["original_review_sha256"] == {p.name: sha(p) for p in reviews}, "Adjudication not bound to originals")
    require(type(adj.get("adjudicator_exposed_to_arm_mapping")) is bool, "Adjudicator exposure must be disclosed")
    final, disagreements = merge_reviews(first, second, adj["rows"])
    detail = {"schema": "radar_coverage_semantic_verdicts_v1", "rows": list(final.values()),
              "disagreements": disagreements, "overridden_review_ids": [r["review_id"] for r in adj["rows"]],
              "inputs_sha256": {str(p): sha(p) for p in [*reviews, Path(adjudication), mapping_path]}}
    write_new(Path(private_output), detail)
    result = {"schema": "radar_coverage_semantic_summary_v1", "reference_grade": "AI source reference, not human gold",
              "evaluated_outputs": 288, "independent_original_review_rows": [288, 288], "output_deduplication": False,
              "disagreement_rows": len(disagreements),
              "disagreement_fields": dict(Counter(k for fields in disagreements.values() for k in fields)),
              "adjudicated_rows": len(adj["rows"]),
              "adjudicator_exposed_to_arm_mapping": adj["adjudicator_exposed_to_arm_mapping"],
              "original_reviewer_A": summarize(first, mapping, policy),
              "original_reviewer_B": summarize(second, mapping, policy),
              "adjudicated": summarize(final, mapping, policy),
              "protocol_sha256": manifest["protocol_sha256"], "policy_sha256": manifest["policy_sha256"],
              "execution_audit_sha256": sha(execution_path),
              "review_sha256": {p.name: sha(p) for p in reviews}, "adjudication_sha256": sha(Path(adjudication)),
              "reviewer_identities": [d["reviewer"] for d in declarations],
              "review_process": "Two separate non-QA-author agents instructed to retain independent original verdicts and avoid mapping/other judgments. Historical design/source/rubric exposure disclosed in review_preparation.json and private declarations; separate files alone do not prove statistical independence.",
              "private_verdicts_sha256": sha(Path(private_output)), "code_sha256": sha(Path(__file__)),
              "limits": "Same-family AI reviewers are not independent human experts. Eight correlated source groups, one base-model run, no significance/generalization claim. Raw is a system reference; flat/bound have equal declared E_sem, excluding archival quote sidecars. No new recovery-SFT transfer or training evidence."}
    write_new(Path(public_output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "adjudication", "private-output", "public-output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.package, args.adjudication, args.private_output, args.public_output)
    print(json.dumps({"disagreement_rows": result["disagreement_rows"], **result["adjudicated"]["overall"],
                      "continuation_gate": result["adjudicated"]["continuation_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
