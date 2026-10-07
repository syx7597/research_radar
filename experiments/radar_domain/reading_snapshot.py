"""Recheck an AI-reviewed reading snapshot; no QA or model readiness implied.

The public selection records the human-readable curation decision. This checker
binds that decision to its original source bytes, author packets and AI reviews;
it cannot independently certify the correctness of the reviewer's judgments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.radar_domain.source_readings import ROOT, LOCAL, audit, load_chunks, sha, write_new

PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_readings_v1"
SELECTION = PUBLIC / "selection.json"
OUTPUT = PUBLIC / "snapshot_audit.json"


def build():
    selection = json.loads(SELECTION.read_text())
    if (selection["evaluation_frozen"] or selection["model_run_ready"] or selection["human_gold"]
            or selection["new_questions"] != 0):
        raise ValueError("This snapshot must not claim QA/evaluation readiness")
    inputs = {str(SELECTION.relative_to(ROOT)): sha(SELECTION)}
    verified, archived_code_versions = set(), []

    def verify(rel, expected):
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / ".git"):
            raise ValueError("Unsafe declared input path")
        if not path.is_file() or sha(path) != expected:
            alternatives = [v for v in selection["historical_code_versions"]
                            if v["original_path"] == rel and v["sha256"] == expected]
            if len(alternatives) != 1:
                raise ValueError(f"Changed input {rel}")
            old = alternatives[0]
            saved = (ROOT / old["archived_path"]).resolve()
            if not saved.is_relative_to(LOCAL) or sha(saved) != expected:
                raise ValueError("Missing historical reviewed code bytes")
            if old not in archived_code_versions:
                archived_code_versions.append(old)
            verified.add(old["archived_path"])
        else:
            verified.add(rel)

    for rel, expected in selection["inputs_sha256"].items():
        verify(rel, expected)
        inputs[rel] = expected
    packets, records, ledger, review_groups = [], [], [], []
    for entry in selection["accepted_packets"]:
        packet_path = ROOT / entry["path"]
        verify(entry["path"], entry["sha256"])
        if not packet_path.resolve().is_relative_to(LOCAL):
            raise ValueError("Packet outside private curation directory")
        packet = json.loads(packet_path.read_text())
        if packet["author"] != entry["author"] or entry["author"] == entry["reviewer"]:
            raise ValueError("Missing author/reviewer separation")
        matched_current_packet = False
        for item in entry["reviews"]:
            verify(item["path"], item["sha256"])
            review = json.loads((ROOT / item["path"]).read_text())
            if review["reviewer"] != entry["reviewer"] or review["human_gold"]:
                raise ValueError("Reviewer identity or reference-grade mismatch")
            bindings = dict(review.get("inputs_sha256", {}))
            if "input_path" in review:
                bindings[review["input_path"]] = review["input_sha256"]
            for rel, expected in bindings.items():
                verify(rel, expected)
            matched_current_packet |= bindings.get(entry["path"]) == entry["sha256"]
        if not matched_current_packet or entry["open_annotation_corrections"]:
            raise ValueError("Accepted packet lacks a bound closed review")
        packets.append(packet_path)
        records += packet["records"]
        ledger += packet["ledger"]
        review_groups.append({"packet": entry["path"], "author": entry["author"],
                              "reviewer": entry["reviewer"], "readings": len(packet["records"]),
                              "ledger_entries": len(packet["ledger"]), "chunks": len(packet["coverage"]),
                              "review_status": entry["review_status"]})
    result = audit(packets)
    if result["uncovered_chunks"]:
        raise ValueError("Not all predeclared pages/sections have annotation coverage")
    chunks = load_chunks()
    result.update({"schema": "radar_reading_snapshot_audit_v1", "recorded_on": "2026-10-07",
                   "status": "AI_cross_reviewed_readings_before_independent_question_authoring",
                   "inputs_sha256": inputs, "packet_review_summary": review_groups,
                   "verified_input_files": len(verified), "historical_reviewed_code": archived_code_versions,
                   "unresolved_source_topics": [{"entry_id": r["entry_id"], "family_id": r["family_id"],
                                                  "topic": r["topic"]} for r in ledger if r["status"] == "unresolved"],
                   "by_chunk": [{"chunk_id": c, "family_id": chunks[c]["family_id"],
                                 "readings": sum(r["citation"]["chunk_id"] == c for r in records),
                                 "ledger_entries": sum(r["citation"]["chunk_id"] == c for r in ledger)} for c in chunks],
                   "distinct_subject_labels": len({s for r in records for s in r["subjects"]}),
                   "subject_count_limit": "Model labels include variants; these are not independent evidence groups.",
                   "necessary_next_steps": selection["necessary_next_steps"],
                   "new_GPU_hours": 0})
    result["counts"]["citation_spans"] = sum(len(r["citation"]["spans"]) for r in records)
    result["counts"]["shared_subject_readings"] = sum(len(r["subjects"]) > 1 for r in records)
    result["counts"]["PDF_pages"] = sum(c["page_one_based"] is not None for c in chunks.values())
    result["counts"]["HTML_content_sections"] = sum(c["page_one_based"] is None for c in chunks.values())
    result["semantic_review_limits"] = selection["semantic_review_limits"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="First write only; never overwrite a snapshot")
    args = parser.parse_args()
    result = build()
    if args.write:
        write_new(OUTPUT, result)
    elif json.loads(OUTPUT.read_text()) != result:
        raise ValueError("Saved snapshot does not reproduce")
    print(json.dumps({"status": "verified", "counts": result["counts"],
                      "model_run_ready": False, "new_GPU_hours": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
