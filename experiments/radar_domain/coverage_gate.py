"""Verify and count recorded semantic witnesses in actual fixed evidence.

Membership/count checks cannot decide whether two readings genuinely compete.
That decision remains in the preserved independent AI reviews/adjudication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json

from experiments.radar_domain.coverage_evidence import retrieve_questions
from experiments.radar_domain.question_snapshot import PUBLIC, build as question_snapshot, minimal_questions, verify
from experiments.radar_domain.source_readings import ROOT, sha, write_new

LOCAL = ROOT / "data/radar_sources_v2/qa_v1"
OUTPUT = PUBLIC / "necessity_gate.json"


def validate_labels(rows, observations, records, questions):
    if len(rows) != len(questions) or {r["question_id"] for r in rows} != set(questions):
        raise ValueError("Gate must cover all questions exactly once")
    for r in rows:
        qid = r["question_id"]; q = questions[qid]; e = observations[qid]
        if r["question_sha256"] != q["question_sha256"] or r["family_id"] != q["family_id"]:
            raise ValueError("Gate question identity changed")
        if type(r["competing"]) is not bool or type(r["complex_subset"]) is not bool:
            raise ValueError("Gate labels must be booleans")
        witnesses = r["witness_record_ids"]
        if len(set(witnesses)) != len(witnesses) or any(x not in e["all_reading_ids"] for x in witnesses):
            raise ValueError("Witness missing from actual E(q), or repeated")
        if any(records[x]["family_id"] != q["family_id"] for x in witnesses):
            raise ValueError("Witness outside declared within-family scope")
        if r["complex_subset"] and not r["competing"]:
            raise ValueError("Complex subset must be inside competition subset")
        hit = {s["chunk_id"] for s in q["support"]} <= set(e["retrieved_chunks"])
        if r["all_reference_chunks_retrieved"] != hit:
            raise ValueError("Reference-page coverage mismatch")
        if r["competing"] and (len(witnesses) < 2 or not hit or not r["reason_zh"]):
            raise ValueError("Competition lacks complete reference coverage or distinct witnesses")
        if not r["competing"] and witnesses:
            raise ValueError("Non-competing row cannot contain accepted witnesses")


def build():
    snapshot = question_snapshot()
    decision_path = LOCAL / "gate_adjudication.json"
    decision = json.loads(decision_path.read_text())
    if decision["model_outputs_seen"] or decision["open_disagreements"]:
        raise ValueError("Gate requires a closed pre-model decision")
    for rel, expected in decision["inputs_sha256"].items():
        verify(rel, expected)
    observation = json.loads((LOCAL / "retrieval_observation.json").read_text())
    minimal = minimal_questions()
    replay = retrieve_questions(minimal)
    replay["questions_sha256"] = sha(LOCAL / "questions.minimal.jsonl")
    saved_minimal = [json.loads(s) for s in (LOCAL / "questions.minimal.jsonl").read_text().splitlines()]
    if replay != observation or saved_minimal != minimal:
        raise ValueError("Question-only retrieval does not reproduce")
    selected = json.loads((PUBLIC / "selection.json").read_text())
    questions = {q["question_id"]: q for e in selected["accepted_packets"]
                 for q in json.loads((ROOT / e["path"]).read_text())["questions"]}
    reading_selection = ROOT / "artifacts/thesis_direction_review/radar_readings_v1/selection.json"
    records = {r["record_id"]: r for e in json.loads(reading_selection.read_text())["accepted_packets"]
               for r in json.loads((ROOT / e["path"]).read_text())["records"]}
    obs = {r["question_id"]: r for r in observation["rows"]}
    rows = decision["rows"]; validate_labels(rows, obs, records, questions)
    competing = [r for r in rows if r["competing"]]
    counts = {"questions": len(rows), "all_reference_chunks_retrieved": sum(r["all_reference_chunks_retrieved"] for r in rows),
              "competing_questions": len(competing), "competing_source_groups": len({r["family_id"] for r in competing}),
              "complex_questions_in_competing_subset": sum(r["complex_subset"] for r in competing)}
    design_path = ROOT / "artifacts/thesis_direction_review/radar_coverage_v2/design.json"
    gate = json.loads(design_path.read_text())["necessity_gate_before_model_outputs"]
    thresholds = {"competing_questions": gate["minimum_questions_with_multiple_competing_readings"],
                  "competing_source_groups": gate["minimum_components_with_competing_readings"],
                  "complex_questions_in_competing_subset": gate["minimum_questions_with_two_qualifier_dimensions_or_value_correspondence"]}
    passed = all(counts[k] >= v for k, v in thresholds.items())
    by_family = []
    for f, n in sorted(snapshot["by_family"].items()):
        group = [r for r in rows if r["family_id"] == f]
        by_family.append({"family_id": f, "questions": n,
                          "all_reference_chunks_retrieved": sum(r["all_reference_chunks_retrieved"] for r in group),
                          "competing_questions": sum(r["competing"] for r in group),
                          "complex_in_competing": sum(r["complex_subset"] for r in group)})
    sizes = [len(r["all_reading_ids"]) for r in obs.values()]
    return {"schema": "radar_actual_evidence_necessity_gate_v1", "recorded_on": "2026-10-07",
            "inputs_sha256": {**decision["inputs_sha256"], str(decision_path.relative_to(ROOT)): sha(decision_path),
                              str(design_path.relative_to(ROOT)): sha(design_path)},
            "counts": counts, "thresholds": thresholds, "passed": passed, "by_family": by_family,
            "retrieval_replay": "identical_96_questions_all_page_records_no_reference_filter",
            "retrieved_records_per_question": {"min": min(sizes), "max": max(sizes)},
            "review_resolution": decision["review_resolution"], "limits_zh": decision["limits_zh"],
            "action": "prepare_equal_information_and_context_preflight" if passed else "cancel_bound_vs_flat_under_registered_stop_rule",
            "human_gold": False, "evaluation_frozen": False, "model_run_ready": False,
            "new_model_runs": 0, "new_training_runs": 0, "new_GPU_hours": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--write", action="store_true")
    a = p.parse_args(); d = build()
    if a.write:
        write_new(OUTPUT, d)
    elif json.loads(OUTPUT.read_text()) != d:
        raise ValueError("Saved gate does not reproduce")
    print(json.dumps({"counts": d["counts"], "passed": d["passed"], "action": d["action"]}))


if __name__ == "__main__":
    main()
