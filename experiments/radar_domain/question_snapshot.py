"""Rebuild a source-authored, AI-reviewed QA snapshot from a public selection.

This binds recorded semantic decisions; it does not replace them with an
executor, certify human gold, or declare the model experiment ready.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.radar_domain.coverage_questions import LOCAL, audit
from experiments.radar_domain.coverage_design import question_sha as normalized_question_sha
from experiments.radar_domain.source_readings import ROOT, sha, write_new

PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_questions_v1"
SELECTION = PUBLIC / "selection.json"
OUTPUT = PUBLIC / "snapshot_audit.json"


def verify(rel, expected):
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT) or p.is_relative_to(ROOT / ".git") or sha(p) != expected:
        raise ValueError(f"Changed or unsafe snapshot input: {rel}")
    return p


def review_outcomes(checks, closed_question_ids):
    """Only explicitly closed rubric clarifications may accompany acceptance."""
    pending = set()
    for row in checks:
        status = row.get("status", row.get("verdict"))
        if status == "accepted":
            if row.get("issues") or row.get("open_issues"):
                raise ValueError("Accepted review row still contains issues")
            continue
        if status not in {"scoring_clarification_requested", "issues"}:
            raise ValueError("Rejected, pending or unknown semantic review outcome")
        supported = (row.get("source_semantics") == "supported" or
                     row.get("reference_values_units_binding") == "accepted")
        if not supported or row["question_id"] not in closed_question_ids:
            raise ValueError("Review issue lacks supported values and explicit closure")
        pending.add(row["question_id"])
    return pending


def validate_history_binding(history, packets, questions):
    if any(history["inputs_sha256"].get(e["path"]) != e["sha256"] for e in packets):
        raise ValueError("Historical review does not bind current question packets")
    if len(history["rows"]) != len(questions) or {r["question_id"] for r in history["rows"]} != set(questions):
        raise ValueError("Missing per-question historical review")
    for row in history["rows"]:
        q = questions[row["question_id"]]
        # History screening uses normalized fingerprints, while author packets
        # bind literal UTF-8 bytes. Full packet hashes above bind both forms.
        if row["question_sha256"] != normalized_question_sha(q["question_zh"]) or row["family_id"] != q["family_id"]:
            raise ValueError("Historical review refers to different question text or family")
        if row["same_target_or_derivative_identified"] or row["unresolved_question_derivative_identified"]:
            raise ValueError("A derivative or unresolved candidate cannot be admitted")


def validate_response_binding(adjudication, closure_bindings, selected_bindings):
    response = adjudication.get("author_response")
    expected = selected_bindings.get(response)
    if (not response or not expected or adjudication["inputs_sha256"].get(response) != expected
            or closure_bindings.get(response) != expected):
        raise ValueError("Adjudication and independent closure must bind the same current author response")


def build():
    selected = json.loads(SELECTION.read_text())
    if selected["human_gold"] or selected["evaluation_frozen"] or selected["model_run_ready"]:
        raise ValueError("Question snapshot is not a human gold or execution protocol")
    for rel, expected in selected["inputs_sha256"].items():
        verify(rel, expected)
    adjudication = json.loads((ROOT / selected["scoring_clarifications"]["decision_file"]).read_text())
    closure = json.loads((ROOT / selected["scoring_clarifications"]["second_closure_file"]).read_text())
    closure_bindings = closure.get("inputs_sha256", closure.get("input_sha256", {}))
    validate_response_binding(adjudication, closure_bindings, selected["inputs_sha256"])
    for doc in (adjudication, closure):
        for rel, expected in doc.get("inputs_sha256", doc.get("input_sha256", {})).items():
            verify(rel, expected)
    if (adjudication["unresolved_issues"] or closure["remaining_review_issues"] or
            adjudication["decision"] != "adopt_author_rubric_response_without_changing_questions_or_values"):
        raise ValueError("Unclosed rubric adjudication")
    closed = set(adjudication["affected_questions"])
    if ({r["question_id"] for r in closure["decisions"]} != closed or
            any(r["verdict"] != "closed" for r in closure["decisions"])):
        raise ValueError("Second closure does not cover adjudicated questions")
    packets, summaries, all_ids, pending, all_questions = [], [], set(), set(), {}
    for entry in selected["accepted_packets"]:
        p = verify(entry["path"], entry["sha256"])
        if not p.is_relative_to(LOCAL):
            raise ValueError("Question packet outside local data directory")
        d = json.loads(p.read_text()); ids = {q["question_id"] for q in d["questions"]}
        if d["author"] != entry["author"] or ids & all_ids:
            raise ValueError("Author mismatch or repeated question identity")
        all_ids |= ids
        all_questions.update({q["question_id"]: q for q in d["questions"]})
        if closure_bindings.get(entry["path"]) != entry["sha256"]:
            raise ValueError("Rubric closure does not bind current question packet")
        reviewers = set()
        for item in entry["reviews"]:
            review = json.loads(verify(item["path"], item["sha256"]).read_text())
            if review["reviewer"] == d["author"] or review["human_gold"]:
                raise ValueError("Missing non-author AI review")
            reviewers.add(review["reviewer"])
            bindings = dict(review.get("inputs_sha256", {}))
            if "author_input" in review:
                a = review["author_input"]; bindings[a["path"]] = a["sha256"]
            if bindings.get(entry["path"]) != entry["sha256"]:
                raise ValueError("Review does not bind this question packet")
            for rel, expected in bindings.items():
                verify(rel, expected)
            checks = review.get("record_checks", review.get("questions", []))
            if len(checks) != len(ids) or {r["question_id"] for r in checks} != ids:
                raise ValueError("Review does not cover every question")
            pending |= review_outcomes(checks, closed)
            if adjudication["inputs_sha256"].get(item["path"]) != item["sha256"]:
                raise ValueError("Adjudication does not bind this original review")
            if review["reviewer"] == closure["reviewer"] and closure_bindings.get(item["path"]) != item["sha256"]:
                raise ValueError("Independent closure does not bind its current original review")
        if len(reviewers) != 2 or entry["open_semantic_issues"]:
            raise ValueError("Two recorded closed non-author reviews required")
        if closure["reviewer"] not in reviewers or closure["qa_author"] or closure["human_gold"]:
            raise ValueError("Rubric closure requires the recorded non-author reviewer")
        packets.append(p)
        summaries.append({"packet": entry["path"], "questions": len(ids),
                          "author": d["author"], "reviewers": sorted(reviewers)})
    if pending != closed:
        raise ValueError("Rubric closure differs from actual review issues")
    history = json.loads((ROOT / selected["history_semantic_review"]).read_text())
    validate_history_binding(history, selected["accepted_packets"], all_questions)
    for rel, expected in history["inputs_sha256"].items():
        verify(rel, expected)
    result = audit(packets)
    result.update({"schema": "radar_question_snapshot_audit_v1", "recorded_on": "2026-10-07",
                   "selection_sha256": sha(SELECTION), "status": "source_authored_two_AI_reviews",
                   "reviews": summaries, "history_counts": history["counts"],
                   "history_limits_zh": history["limits_zh"],
                   "scoring_clarifications": selected["scoring_clarifications"],
                   "semantic_limits": selected["semantic_limits"],
                   "new_model_runs": 0, "new_training_runs": 0, "new_GPU_hours": 0})
    return result


def minimal_questions():
    """Verify the source/review snapshot before exporting reference-free input."""
    build()
    selected = json.loads(SELECTION.read_text())
    return [{"question_id": q["question_id"], "question": q["question_zh"]}
            for e in selected["accepted_packets"]
            for q in json.loads((ROOT / e["path"]).read_text())["questions"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    p.add_argument("--export-questions", type=Path)
    a = p.parse_args(); d = build()
    if a.write:
        write_new(OUTPUT, d)
    elif json.loads(OUTPUT.read_text()) != d:
        raise ValueError("Saved question snapshot does not reproduce")
    if a.export_questions:
        if not a.export_questions.resolve().is_relative_to(LOCAL):
            raise ValueError("Raw questions must remain local")
        with a.export_questions.open("x") as f:
            for q in minimal_questions():
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(json.dumps({"verified_questions": d["questions"], "model_run_ready": False}))


if __name__ == "__main__":
    main()
