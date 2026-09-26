"""Freeze the entire official validation split for query-repair round two.

No gold execution, answer analysis, difficulty filtering, or model scoring is
performed. Gold programs are read only for exact overlap checks; all 11,797
official validation rows remain in their original order. Answers are copied to
a separate offline-only file and never enter question-only input records.

Run: python3 -m experiments.condition_consistency.prepare_repair_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_data import EXPECTED_SHA256, program_key, question_key, sha256
from .prepare_repair_splits import frozen_write, jsonl, require


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/kqa_pro"))
    parser.add_argument("--pilot-manifest", type=Path,
                        default=Path("results/condition_consistency/data_check/split_manifest.json"))
    parser.add_argument("--repair-manifest", type=Path,
                        default=Path("results/condition_consistency/query_repair/split_manifest.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/condition_consistency/repair_splits"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/condition_consistency/query_repair/v2/official_val_manifest.json"))
    args = parser.parse_args()

    source_hashes = {}
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(args.data_dir / name)
        require(actual == expected, f"Unexpected source bytes: {name}")
        source_hashes[name] = actual
    val = json.loads((args.data_dir / "val.json").read_text(encoding="utf-8"))
    require(len(val) == 11797, "Must preserve all 11,797 official validation records")
    val_questions = {question_key(row) for row in val}
    val_programs = {program_key(row) for row in val}
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    repair = json.loads(args.repair_manifest.read_text(encoding="utf-8"))
    require(pilot["source_train_sha256"] == source_hashes["train.json"], "Pilot source changed")
    require(repair["source_sha256"] == source_hashes, "Repair source changed")
    require(repair["pilot_manifest"]["sha256"] == sha256(args.pilot_manifest), "Pilot manifest changed")

    references = {}
    generator = pilot["splits"]["generator_train"]
    references["generator_train"] = {
        "path": generator["path"], "sha256": generator["sha256"], "count": generator["count"],
        "ids": [f"train:{i}" for i in generator["indices"]],
    }
    for name, entry in repair["splits"].items():
        label = "seen_v1_holdout_diagnostic" if name == "holdout" else name
        references[label] = {**entry["gold"], "count": entry["count"], "ids": entry["ids"]}
    overlap_checks = {}
    for name, entry in references.items():
        path = Path(entry["path"])
        require(sha256(path) == entry["sha256"], f"Reference split bytes changed: {name}")
        rows = read_jsonl(path)
        require(len(rows) == entry["count"], f"Reference split size changed: {name}")
        require([row["id"] for row in rows] == entry["ids"], f"Reference split IDs/order changed: {name}")
        question_overlap = {question_key(row) for row in rows} & val_questions
        program_overlap = {program_key(row) for row in rows} & val_programs
        require(not question_overlap, f"Official-val normalized question overlap: {name}")
        require(not program_overlap, f"Official-val complete program overlap: {name}")
        overlap_checks[name] = {
            "records": len(rows), "reference_sha256": entry["sha256"],
            "normalized_question_overlap": 0, "complete_program_overlap": 0,
        }

    questions = [{"id": f"val:{i}", "question": row["question"]} for i, row in enumerate(val)]
    gold = [{"id": f"val:{i}", "source_index": i, "question": row["question"],
             "program": row["program"], "answer": row["answer"]} for i, row in enumerate(val)]
    require(all(set(row) == {"id", "question"} for row in questions), "Gold leaked into model input")
    question_path = args.split_dir / "official_val.questions.jsonl"
    gold_path = args.split_dir / "official_val.gold.jsonl"
    frozen_write(question_path, jsonl(questions))
    frozen_write(gold_path, jsonl(gold))
    manifest = {
        "protocol": "QUERY_REPAIR_V2_PROTOCOL", "dataset": "KQA Pro",
        "dataset_partition": "official validation, not hidden test",
        "source_sha256": source_hashes,
        "preparation_script_sha256": sha256(Path(__file__)),
        "pilot_manifest": {"path": str(args.pilot_manifest), "sha256": sha256(args.pilot_manifest)},
        "repair_manifest": {"path": str(args.repair_manifest), "sha256": sha256(args.repair_manifest)},
        "selection": "All 11,797 official validation records, original source order, no filtering or deduplication of evaluation rows",
        "program_usage_during_preparation": "exact question/program overlap checks only; no execution or semantic error analysis",
        "answer_usage_during_preparation": "copied to separate offline-only gold file; no execution, analysis, statistics, selection or model input",
        "history_disclosure": "This official validation set is public and was used by earlier project work; do not call it hidden test or claim never-before-seen project data. The 2,000-question v1 train holdout is now seen diagnostic data and cannot validate v2 independently.",
        "checks": {"all_passed": True, "evaluation_rows_removed": 0, "reference_split_overlap": overlap_checks},
        "splits": {"official_val": {
            "count": len(val), "ids": [row["id"] for row in questions],
            "question_only": {"path": str(question_path), "sha256": sha256(question_path), "fields": ["id", "question"]},
            "gold": {"path": str(gold_path), "sha256": sha256(gold_path),
                     "usage": "one frozen full-validation evaluation only; never generation, repair or threshold selection"},
        }},
    }
    frozen_write(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "sha256": sha256(args.manifest),
                      "questions": len(val), "evaluation_rows_removed": 0,
                      "all_overlap_checks_passed": True, "official_val_gold_scored_or_analyzed": False},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
