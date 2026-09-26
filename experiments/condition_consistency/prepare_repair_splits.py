"""Freeze train-only query-repair design/calibration/holdout partitions.

This command never executes gold programs, computes answer statistics, or scores
predictions. Gold is copied to a separate file solely for later offline scoring.
Run: python3 -m experiments.condition_consistency.prepare_repair_splits
Existing output bytes must match; a freeze cannot silently be replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .prepare_data import EXPECTED_SHA256, make_splits, program_key, question_key, sha256

DESIGN_SEED = 20260927
HOLDOUT_SEED = 20260928


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_write(path: Path, value: str):
    encoded = value.encode("utf-8")
    if path.exists():
        require(path.read_bytes() == encoded, f"Frozen output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)


def jsonl(records):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/kqa_pro"))
    parser.add_argument("--pilot-manifest", type=Path,
                        default=Path("results/condition_consistency/data_check/split_manifest.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/condition_consistency/repair_splits"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/condition_consistency/query_repair/split_manifest.json"))
    args = parser.parse_args()

    source_hashes = {}
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(args.data_dir / name)
        require(actual == expected, f"Unexpected source bytes: {name}")
        source_hashes[name] = actual
    rows = json.loads((args.data_dir / "train.json").read_text(encoding="utf-8"))
    val = json.loads((args.data_dir / "val.json").read_text(encoding="utf-8"))
    require(len(rows) == 94376 and len(val) == 11797, "Unexpected dataset sizes")
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    require(pilot["source_train_sha256"] == source_hashes["train.json"], "Pilot source differs")
    pilot_sizes = {name: entry["count"] for name, entry in pilot["splits"].items()}
    require(pilot_sizes == {"generator_train": 5000, "reranker_train": 1000, "dev": 500},
            "Expected the original 5000/1000/500 pilot")

    # Re-run the ORIGINAL function: exact question OR complete-program connected
    # components, minimum-index representative, remove components touching val.
    replayed, replay_stats = make_splits(rows, val, pilot["seed"], pilot_sizes)
    for name, ids in replayed.items():
        entry = pilot["splits"][name]
        require(ids == entry["indices"], f"Original split cannot be reproduced: {name}")
        require(sha256(entry["path"]) == entry["sha256"], f"Original split changed: {name}")
    all_components, stats = make_splits(
        rows, val, pilot["seed"], {"all": replay_stats["eligible_component_representatives"]})
    eligible = all_components["all"]
    require(len(set(eligible)) == len(eligible), "Duplicate component representatives")
    used = set().union(*(set(ids) for ids in replayed.values()))
    require(len(used) == 6500 and used <= set(eligible), "Pilot components are not disjoint eligible representatives")
    # Because each selected index is a verified minimum-index representative,
    # excluding it excludes its entire original question/program component.
    remaining = sorted(set(eligible) - used)
    random.Random(HOLDOUT_SEED).shuffle(remaining)
    require(len(remaining) >= 2000, "Insufficient untouched components")
    rr = sorted(replayed["reranker_train"])
    random.Random(DESIGN_SEED).shuffle(rr)
    splits = {"design": rr[:500], "calibration": rr[500:],
              "dev_diagnostic": replayed["dev"], "holdout": remaining[:2000]}

    checked = {"generator_train": replayed["generator_train"], **splits}
    question_sets = {name: {question_key(rows[i]) for i in ids} for name, ids in checked.items()}
    program_sets = {name: {program_key(rows[i]) for i in ids} for name, ids in checked.items()}
    val_questions, val_programs = {question_key(row) for row in val}, {program_key(row) for row in val}
    pair_checks = []
    for name, ids in checked.items():
        require(len(question_sets[name]) == len(ids) == len(program_sets[name]), f"Duplicates within {name}")
        require(not question_sets[name] & val_questions, f"Val question overlap: {name}")
        require(not program_sets[name] & val_programs, f"Val program overlap: {name}")
        for other, other_ids in checked.items():
            if name >= other:
                continue
            require(not set(ids) & set(other_ids), f"ID overlap: {name}/{other}")
            require(not question_sets[name] & question_sets[other], f"Question overlap: {name}/{other}")
            require(not program_sets[name] & program_sets[other], f"Program overlap: {name}/{other}")
            pair_checks.append({"left": name, "right": other, "id_overlap": 0,
                                "normalized_question_overlap": 0, "complete_program_overlap": 0,
                                "connected_component_overlap": 0})

    manifest = {
        "protocol": "QUERY_REPAIR_PROTOCOL_v1", "dataset": "KQA Pro",
        "source_sha256": source_hashes,
        "pilot_manifest": {"path": str(args.pilot_manifest), "sha256": sha256(args.pilot_manifest)},
        "preparation_script_sha256": sha256(Path(__file__)),
        "design_calibration_seed": DESIGN_SEED, "holdout_seed": HOLDOUT_SEED,
        "selection": "Original exact normalized-question OR complete-program connected components; one minimum-index representative; exclude official-val-overlapping components; exclude all 6500 pilot components; sort and seeded-shuffle remaining representatives; take first 2000",
        "design_calibration_selection": "Sort original reranker_train IDs, seeded-shuffle, first 500 design, remaining 500 calibration",
        "deduplication": stats, "excluded_pilot_components": len(used),
        "untouched_eligible_components_before_holdout": len(remaining),
        "holdout_answer_usage_during_preparation": "none; no execution, answer statistics, difficulty selection or scoring; gold copied separately for later frozen evaluation",
        "holdout_program_usage_during_preparation": "exact-duplicate component exclusion only; no model input, candidate construction or correctness analysis",
        "official_val_usage": "exact question/program component exclusion only",
        "data_history": "design/calibration were previously used by pilot reranker experiments; dev_diagnostic was previously inspected; holdout excludes all these and generator training components",
        "checks": {"all_passed": True, "official_val_exact_overlap": 0, "pairs": pair_checks},
        "splits": {},
    }
    for name, ids in splits.items():
        questions = [{"id": f"train:{i}", "question": rows[i]["question"]} for i in ids]
        gold = [{"id": f"train:{i}", "source_index": i, "question": rows[i]["question"],
                 "program": rows[i]["program"], "answer": rows[i]["answer"]} for i in ids]
        question_path, gold_path = args.split_dir / f"{name}.questions.jsonl", args.split_dir / f"{name}.gold.jsonl"
        frozen_write(question_path, jsonl(questions))
        frozen_write(gold_path, jsonl(gold))
        require(all(set(row) == {"id", "question"} for row in questions), "Gold leaked into inference input")
        manifest["splits"][name] = {
            "count": len(ids), "ids": [f"train:{i}" for i in ids],
            "question_only": {"path": str(question_path), "sha256": sha256(question_path), "fields": ["id", "question"]},
            "gold": {"path": str(gold_path), "sha256": sha256(gold_path), "usage": "offline evaluation/calibration only; never passed to generation or repair"},
        }
    frozen_write(args.manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "sha256": sha256(args.manifest),
                      "sizes": {name: len(ids) for name, ids in splits.items()},
                      "all_overlap_checks_passed": True,
                      "holdout_gold_scored_or_analyzed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
