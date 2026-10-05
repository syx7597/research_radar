"""Aggregate two masked AI reviews without replacing the frozen diagnostics.

This post-run analysis adds no training, inference, alias matching or rescoring
of structured claims. Private reviews and their unblinding map stay in data/.
All reviewer disagreements require an explicit, source-grounded adjudication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import source_analysis as analysis
from . import source_probe as probe
from .development_analysis import require


FIELDS = ("answer_correct", "answer_supported", "citations_support_answer",
          "required_conditions_preserved")
REVIEW_NAMES = ("blind_answer_reviews.json", "blind_answer_reviews_second.json")
ADJUDICATION = "blind_answer_adjudication.json"


def validate_scores(row):
    require(all(f in row for f in FIELDS), "Missing review field")
    for field in FIELDS[:3]:
        require(type(row.get(field)) is bool, f"Invalid boolean: {field}")
    require(row.get(FIELDS[3]) is None or type(row[FIELDS[3]]) is bool,
            "Invalid condition verdict")


def validate_review(review, blind_ids):
    require(review["human_review"] is False and review["arm_labels_seen"] is False,
            "Expected a masked AI review, not human gold")
    require(isinstance(review.get("reviewer"), str) and review["reviewer"], "Missing reviewer identity")
    rows = review["records"]
    require(len(rows) == len(blind_ids) and {r["blind_id"] for r in rows} == set(blind_ids),
            "Review has missing, extra or duplicate IDs")
    for row in rows:
        validate_scores(row)
        require(bool(row.get("notes_zh")), "Review reason missing")
    return {r["blind_id"]: r for r in rows}


def resolve(first, second, adjudication):
    require(set(first) == set(second), "Reviewer coverage differs")
    disagreements = {bid: [f for f in FIELDS if first[bid][f] != second[bid][f]]
                     for bid in first if any(first[bid][f] != second[bid][f] for f in FIELDS)}
    decisions = adjudication["records"]
    require(len(decisions) == len(disagreements) and
            {r["blind_id"] for r in decisions} == set(disagreements),
            "Adjudication must cover exactly all score disagreements")
    final = {bid: {f: row[f] for f in FIELDS} for bid, row in first.items()}
    for decision in decisions:
        validate_scores(decision)
        require(bool(decision.get("reason_zh")), "Adjudication reason missing")
        bid = decision["blind_id"]
        for field in FIELDS:
            if field not in disagreements[bid]:
                require(decision[field] == first[bid][field], "Adjudication altered an agreed score")
        final[bid] = {f: decision[f] for f in FIELDS}
    return final, disagreements


def tally(records, verdicts, references):
    counts = Counter({field: 0 for field in FIELDS[:3]})
    counts.update({"correct_and_supported": 0, "correct_supported_and_cited": 0,
                   "qualified_questions": 0, "qualified_conditions_preserved": 0,
                   "qualified_condition_not_applicable": 0,
                   "posthoc_reference_condition_questions": 0,
                   "posthoc_reference_conditions_preserved": 0})
    for row in records:
        verdict = verdicts[row["blind_id"]]
        counts.update({f: int(verdict[f]) for f in FIELDS[:3]})
        counts["correct_and_supported"] += int(verdict["answer_correct"] and verdict["answer_supported"])
        counts["correct_supported_and_cited"] += int(all(verdict[f] for f in FIELDS[:3]))
        if references[row["id"]]["intent"].get("condition_raw") is not None:
            counts["qualified_questions"] += 1
            counts["qualified_conditions_preserved"] += int(verdict[FIELDS[3]] is True)
            counts["qualified_condition_not_applicable"] += int(verdict[FIELDS[3]] is None)
        if any(c.get("condition") is not None for c in references[row["id"]]["semantic_claims"]):
            counts["posthoc_reference_condition_questions"] += 1
            counts["posthoc_reference_conditions_preserved"] += int(verdict[FIELDS[3]] is True)
    return {"questions": len(records), **dict(counts)}


def build():
    mechanical, detail, questions, references, generated = analysis.build()
    packet, key = analysis.prepare_blind(questions, references, generated)
    # Verify the reviewer saw precisely the packet produced from completed runs.
    for path, expected in ((probe.DATA / "blind_answer_review.json", packet),
                           (probe.DATA / "blind_answer_key.json", key),
                           (probe.OUT / "mechanical_summary.json", mechanical),
                           (probe.OUT / "per_question_diagnosis.json", detail)):
        require(json.loads(path.read_text()) == expected, f"Frozen analysis drift: {path}")
    ids = {r["blind_id"] for r in packet["records"]}
    documents = [json.loads((probe.DATA / name).read_text()) for name in REVIEW_NAMES]
    require(documents[0]["reviewer"] != documents[1]["reviewer"], "Reviewers must be distinct agents")
    first, second = [validate_review(doc, ids) for doc in documents]
    adjudication = json.loads((probe.DATA / ADJUDICATION).read_text())
    require(adjudication["human_review"] is False, "AI adjudication cannot claim human review")
    final, disagreements = resolve(first, second, adjudication)
    index = {(r["arm"], r["id"]): r for r in detail["records"]}
    arms = {}
    for arm in probe.ARMS:
        assignments = [r for r in key["records"] if r["arm"] == arm]
        all_scores = {"reviewer_1": tally(assignments, first, references),
                      "reviewer_2": tally(assignments, second, references),
                      "adjudicated": tally(assignments, final, references)}
        by_source = {}
        for source_id in sorted({r["source_id"] for r in references.values()}):
            subset = [r for r in assignments if references[r["id"]]["source_id"] == source_id]
            by_source[source_id] = tally(subset, final, references)
        error_partition = Counter({"missing_required_evidence_page": 0,
                                   "required_pages_present_but_answer_incorrect": 0})
        for row in assignments:
            if not final[row["blind_id"]]["answer_correct"]:
                available = index[arm, row["id"]]["metrics"]["all_required_chunks_available"]
                error_partition["required_pages_present_but_answer_incorrect" if available
                                else "missing_required_evidence_page"] += 1
        runtime = mechanical["runs"][arm]
        selector = runtime["selection_runtime"]
        answer_tokens = runtime["answer_runtime"]["totals"]["total_tokens"]
        selector_tokens = selector["totals"]["total_tokens"] if selector else 0
        arms[arm] = {**all_scores, "mechanical": runtime["counts"], "by_source": by_source,
                     "incorrect_answer_partition": dict(error_partition),
                     "tokens": {"selection": selector_tokens, "answer": answer_tokens,
                                "total": selector_tokens + answer_tokens}}
    pairs = {}
    for a, c in (("A1", "C1"), ("A2", "C2")):
        mapping = {(r["arm"], r["id"]): final[r["blind_id"]]["answer_correct"] for r in key["records"]}
        wins = [q["id"] for q in questions if mapping[c, q["id"]] and not mapping[a, q["id"]]]
        losses = [q["id"] for q in questions if mapping[a, q["id"]] and not mapping[c, q["id"]]]
        pairs[f"{c}_minus_{a}"] = {"wins": wins, "losses": losses, "net_questions": len(wins) - len(losses),
                                  "statistical_significance_tested": False}
    bound_paths = [probe.DATA / n for n in (*REVIEW_NAMES, ADJUDICATION,
                                          "blind_answer_review.json", "blind_answer_key.json")]
    bound_paths += [probe.OUT / "mechanical_summary.json", probe.OUT / "per_question_diagnosis.json",
                    Path(__file__).relative_to(Path.cwd()), probe.PROTOCOL]
    summary = {
        "version": "radar_sources_ai_semantic_review_v1", "human_gold": False,
        "status": "completed", "unique_answers": len(ids), "arm_question_outputs": len(key["records"]),
        "reviewers": [doc["reviewer"] for doc in documents],
        "reviewer_arm_labels_seen": [doc["arm_labels_seen"] for doc in documents],
        "adjudicator_arm_results_seen": adjudication["arm_results_seen"],
        "disagreement_unique_answers": len(disagreements),
        "disagreements_by_field": {f: sum(f in fields for fields in disagreements.values()) for f in FIELDS},
        "arms": arms, "paired_natural_answer_differences": pairs,
        "qualified_subset_definition": "Frozen selector intent.condition_raw is non-null (8 questions); this is the explicit query-condition subset, not every answer qualification",
        "posthoc_reference_condition_subset_definition": "Additional post-run diagnostic: any frozen semantic_claim has non-null condition (9 questions); includes source_zh_15 beam qualification without silently replacing the 8-query denominator",
        "condition_context_rule": "An answer can inherit a condition supplied in the question; requested or necessary unstated source qualifications must be preserved",
        "limits": [
            "Two masked AI-agent reviews are not independent human gold or different-model consensus",
            "Adjudication by the root agent is source-grounded but not blind to aggregate arm results",
            "24 questions share 4 source/family groups; no statistical significance or generalization claim",
            "Curated KB and raw-page RAG have different preparation costs; this is a system-level route comparison",
            "Strict structured-field matching remains unchanged and separate from natural-answer judgments",
            "Original reviews and all disagreements are retained; adjudicated counts alone are insufficient",
            "Condition preservation alone does not imply a correct or supported answer",
        ],
        "inputs_sha256": {str(p): probe.sha(p) for p in bound_paths},
    }
    public_records = [{**row, "reviewer_1": {f: first[row["blind_id"]][f] for f in FIELDS},
                       "reviewer_2": {f: second[row["blind_id"]][f] for f in FIELDS},
                       "adjudicated": final[row["blind_id"]],
                       "disputed_fields": disagreements.get(row["blind_id"], [])}
                      for row in key["records"]]
    return summary, {"version": summary["version"], "human_gold": False, "records": public_records,
                     "adjudication_reasons": [{"blind_id": r["blind_id"], "reason_zh": r["reason_zh"]}
                                              for r in adjudication["records"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, detail = build()
    for path, value in ((probe.OUT / "ai_semantic_summary.json", summary),
                        (probe.OUT / "ai_semantic_verdicts.json", detail)):
        if args.check:
            require(json.loads(path.read_text()) == value, f"AI review aggregation drift: {path}")
        else:
            probe.write_new(path, value)
    print(json.dumps({"status": "verified" if args.check else "written",
                      "unique_answers": summary["unique_answers"],
                      "disagreements": summary["disagreement_unique_answers"]}))


if __name__ == "__main__":
    main()
