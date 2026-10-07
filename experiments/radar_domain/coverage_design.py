"""Register exposed development material for the next source coverage design.

Only existing questions are read, to fingerprint their text and retain their
metadata. No new question is authored, no model runs, and no answer is exported.
The registry is a conservative known-exposure list, not a complete historical
or pretraining contamination audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/thesis_direction_review/radar_coverage_v2"
OLD = "artifacts/thesis_direction_review/radar_review_batch/source_manifest.json"
NEW = "artifacts/thesis_direction_review/radar_sources_v1/source_index.json"
QUESTION_FILES = (
    "artifacts/thesis_direction_review/radar_review_workflow/development_questions.csv",
    "artifacts/thesis_direction_review/radar_development_v2/questions.jsonl",
    "artifacts/thesis_direction_review/radar_lookup_calibration_v1/questions.zh.jsonl",
    "artifacts/thesis_direction_review/radar_lookup_calibration_v1/questions.en.jsonl",
    "data/radar_sources_v1/questions.jsonl",
)
REFERENCES = "data/radar_sources_v1/references.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_sha(text):
    # Deliberately not a semantic near-duplicate detector.
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build():
    design_path = OUT / "design.json"
    candidates_path = OUT / "source_candidates.json"
    design = json.loads(design_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    coverage = design["coverage"]
    if sum(s["target"] for s in coverage["primary_question_strata"]) != coverage["target_questions"]:
        raise ValueError("Planned question strata do not sum to the planned denominator")
    if design["current_state"]["model_run_ready"] or design["current_state"]["evaluation_frozen"]:
        raise ValueError("This design is not a runnable frozen evaluation")
    if {a["id"] for a in design["arms"]} != {"raw", "flat", "bound", "template"}:
        raise ValueError("Design comparison arms changed")
    old = json.loads((ROOT / OLD).read_text())
    new = json.loads((ROOT / NEW).read_text())
    documents = {}
    for row in old["records"]:
        uri = row["source_uri"]
        doc = documents.setdefault(uri, {"canonical_document_id": uri, "source_uri": uri,
                                        "source_sha256": row["source_sha256"], "family_ids": []})
        assert doc["source_sha256"] == row["source_sha256"]
        doc["family_ids"] = sorted(set(doc["family_ids"]) | {row["extracted_entity_name"]})
    for source in new["sources"]:
        documents[source["url"]] = {"canonical_document_id": source["url"], "source_uri": source["url"],
                                     "source_sha256": source["pdf_sha256"], "family_ids": [source["family_id"]]}
    fingerprints, ids, per_file = set(), set(), {}
    for name in QUESTION_FILES:
        path = ROOT / name
        if path.suffix == ".csv":
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = read_lines(path)
        per_file[name] = len(rows)
        for row in rows:
            ids.add(row.get("id") or row["qid"])
            fingerprints.add(question_sha(row["question"]))
    families = {f for doc in documents.values() for f in doc["family_ids"]}
    # Known alternative identity labels, not an exhaustive family taxonomy.
    aliases = {"AN/MPQ-65": ["Patriot", "MIM-104 Patriot", "MIM-104 Patriot system"],
               "WRM200": ["WRM 200"], "METEOR_700C_735C": ["METEOR 735C", "METEOR 700C"],
               "JMA1030": ["JMA-1030", "JMA-1032", "JMA-1034"]}
    known_names = families | {alias for values in aliases.values() for alias in values}
    bindings = {name: sha(ROOT / name) for name in (OLD, NEW, *QUESTION_FILES, REFERENCES)}
    registry = {
        "version": "radar_known_exposure_v2", "status": "known_exposure_registry_not_complete_lineage",
        "canonical_document_ids": sorted(documents),
        "source_sha256": sorted({d["source_sha256"] for d in documents.values()}),
        "source_uris": sorted(documents), "family_ids": sorted(known_names),
        "question_ids": sorted(ids), "question_sha256": sorted(fingerprints),
        "documents": list(documents.values()), "known_family_aliases": aliases,
        "source_document_count": len(documents), "declared_family_group_count": len(families),
        "question_rows_by_input_file": per_file,
        "question_hash_normalization": "NFKC, casefold, whitespace collapsed; no semantic paraphrase detection",
        "limits": ["Known old 12-target lineages plus exposed source_eval_v1 material; not a whole-workspace audit",
                   "Family aliases and canonical document mappings require source review, cannot be inferred from absent matches",
                   "Old 5401 automatic QA and all paraphrases remain ineligible as new independent questions",
                   "Historical corpus overlap and base pretraining exposure remain unknown or only partly audited",
                   "Question versions and translations are not independent targets; fingerprint counts are not dataset size"],
        "inputs_sha256": bindings,
    }
    references = read_lines(ROOT / REFERENCES)
    questions = {q["id"]: q for q in read_lines(ROOT / QUESTION_FILES[-1])}
    # Retrospective admission regression: these exposed questions are lawful
    # development material, but must fail if relabelled as an unseen candidate.
    def packet(split):
        return {
            "version": "exposed_sources_v1_admission_regression", "purpose": "CPU exposure regression only",
            "sources": [{"source_id": s["source_id"], "canonical_document_id": s["url"],
                         "family_ids": [s["family_id"]], "source_uri": s["url"],
                         "source_sha256": s["pdf_sha256"], "split": split} for s in new["sources"]],
            "questions": [{"question_id": r["id"], "question_sha256": question_sha(questions[r["id"]]["question"]),
                           "source_ids": [r["source_id"]], "family_ids": [r["family_id"]],
                           "equivalence_id": r["id"], "primary_type": r["semantic_claims"][0]["value_kind"],
                           "split": split, "annotation_origin": "ai", "review_status": "ai_reviewed",
                           "model_outputs_seen": True} for r in references],
        }
    from .source_split_audit import audit_packet
    development = audit_packet(packet("development"), registry)
    relabelled = audit_packet(packet("evaluation_candidate"), registry)
    shadow = packet("evaluation_candidate")
    shadow["purpose"] = "Synthetic metadata renaming regression, not newly authored QA"
    for row in shadow["questions"]:
        row["question_id"] = "simulated-renaming-" + row["question_id"]
        row["question_sha256"] = hashlib.sha256(("synthetic-metadata:" + row["question_sha256"]).encode()).hexdigest()
        row["equivalence_id"] = None
        row["model_outputs_seen"] = False
    source_relabelling = audit_packet(shadow, registry)
    closure_blocks = [b for b in source_relabelling["blockers"] if b["code"] == "candidate_exposure"]
    if development["blockers"] or not relabelled["blockers"] or len(closure_blocks) != len(new["sources"]):
        raise ValueError("Exposure admission regression failed")
    report = {"version": "radar_coverage_design_check_v2", "new_sources_collected": 0,
              "new_questions_authored": 0, "new_model_runs": 0, "new_training_runs": 0,
              "evaluation_frozen": False, "model_run_ready": False,
              "registry_document_count": len(documents), "registry_family_group_count": len(families),
              "planned_questions": coverage["target_questions"],
              "official_source_candidates_not_admitted": len(candidates["candidates"]),
              "development_admission": development, "false_holdout_relabelling": relabelled,
              "renamed_metadata_source_closure_regression": source_relabelling,
              "interpretation": "Existing 24 questions can remain development, and cannot be promoted to an unseen evaluation by renaming the split",
              "inputs_sha256": {**bindings, "experiments/radar_domain/coverage_design.py": sha(Path(__file__)),
                                str(design_path.relative_to(ROOT)): sha(design_path),
                                str(candidates_path.relative_to(ROOT)): sha(candidates_path),
                                "experiments/radar_domain/source_split_audit.py": sha(ROOT / "experiments/radar_domain/source_split_audit.py")}}
    return {"exposure_registry.json": registry, "design_check.json": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, value in build().items():
        path = OUT / name
        raw = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != raw:
                raise ValueError(f"Coverage design artifact changed: {name}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != raw:
                raise ValueError(f"Refusing to overwrite prior design artifact: {name}")
            path.write_text(raw)
    print(json.dumps({"status": "checked" if args.check else "written", "model_run_ready": False}))


if __name__ == "__main__":
    main()
