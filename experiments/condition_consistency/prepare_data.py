"""Verify fixed KQA Pro files; freeze disjoint train-only pilot splits.

Example: python3 -m experiments.condition_consistency.prepare_data --replay
Raw data/split rows remain local; only manifests and aggregate checks are public.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from .executor import BASELINES_COMMIT, KOPL_COMMIT, KoPLExecutor, compare_answers, serialize_program

MIRROR_REVISION = "0b26da66cec9a4d1e42bde3560aeae9f89f6433b"
EXPECTED_SHA256 = {
    "train.json": "e9fbe4c1cdf207aac83ae0d5e4a1a53a9965a2b13b403de699ca6d5dae6e4510",
    "val.json": "b4aed6ab3d7ad071722064fe3bb02bc028cfbeb15da5f7115d57a1e2d198f3bb",
    "kb.json": "04da7408320c5cb7023c44372cce32846d56d369d8865d2e61a18c3956661a7c",
}
QUALIFIERS = {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate", "QueryAttrUnderCondition", "QueryAttrQualifier", "QueryRelationQualifier"}


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_key(row):
    return " ".join(row["question"].casefold().split())


def program_key(row):
    return serialize_program(row["program"])


def make_splits(rows, val_rows, seed, sizes):
    # Components join records sharing either an exact normalized question OR
    # exact executable program. One representative per component is used.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen_q, seen_p, keys = {}, {}, []
    for i, row in enumerate(rows):
        q, p = question_key(row), program_key(row)
        keys.append((q, p))
        if q in seen_q:
            union(i, seen_q[q])
        if p in seen_p:
            union(i, seen_p[p])
        seen_q[q], seen_p[p] = i, i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    val_q = {question_key(row) for row in val_rows}
    val_p = {program_key(row) for row in val_rows}
    eligible, excluded = [], []
    for group in groups.values():
        if any(keys[i][0] in val_q or keys[i][1] in val_p for i in group):
            excluded.append(group)
        else:
            eligible.append(min(group))
    eligible.sort()
    random.Random(seed).shuffle(eligible)
    if len(eligible) < sum(sizes.values()):
        raise ValueError("Insufficient distinct training components for requested sizes")
    splits, offset = {}, 0
    for name, size in sizes.items():
        splits[name] = eligible[offset:offset + size]
        offset += size
    # Assertions protect against later changes to representative allocation.
    for name, ids in splits.items():
        for other, other_ids in splits.items():
            if name >= other:
                continue
            assert not {keys[i][0] for i in ids} & {keys[i][0] for i in other_ids}
            assert not {keys[i][1] for i in ids} & {keys[i][1] for i in other_ids}
    stats = {
        "train_records": len(rows), "question_program_components": len(groups),
        "records_removed_as_duplicate_component_members": len(rows) - len(groups),
        "components_excluded_for_exact_val_question_or_program_overlap": len(excluded),
        "records_in_excluded_components": sum(map(len, excluded)),
        "eligible_component_representatives": len(eligible),
        "cross_split_exact_question_overlap": 0, "cross_split_exact_program_overlap": 0,
        "val_used_for": "exact question/program duplicate exclusion only; no answers, tuning, or selection by result",
    }
    return splits, stats


def check_checkout(path, expected):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Unexpected source revision at {path}: {actual}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/kqa_pro"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/condition_consistency/data_check"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/condition_consistency/splits"))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--generator-size", type=int, default=5000)
    parser.add_argument("--reranker-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--backend", choices=["baseline", "modern"], default="baseline")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in EXPECTED_SHA256.items():
        path = args.data_dir / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {path}: {actual}")
        files[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    provenance = {
        "dataset": "KQA Pro", "original_repository": "https://github.com/shijx12/KQAPro_Baselines",
        "original_download": "https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1",
        "original_download_status": "Link does not exist when checked on 2026-09-26",
        "mirror": "https://huggingface.co/datasets/drt/kqa_pro", "mirror_revision": MIRROR_REVISION,
        "train_download": f"https://huggingface.co/datasets/drt/kqa_pro/resolve/{MIRROR_REVISION}/train.json",
        "mirror_status": "third-party mirror, not verified as author-owned",
        "verification": "KB and val match existing local files byte-for-byte; train SHA256 matches mirror LFS object. No accessible original train archive was available for independent byte comparison.",
        "dataset_license": "CC-BY-SA-4.0, per original authors' README; mirror card MIT label is not used",
        "code_license": "MIT (official Baselines and KoPL checkouts)",
        "baselines_commit": check_checkout("external/kqa_pro_baselines", BASELINES_COMMIT),
        "kopl_commit": check_checkout("external/kopl", KOPL_COMMIT), "files": files,
    }
    source_files = {
        "kqa_pro_baselines": ["Program/executor_rule.py", "utils/value_class.py", "evaluate.py", "LICENSE"],
        "kopl": ["src/kopl/kopl.py", "src/kopl/data.py", "src/kopl/util.py", "LICENSE"],
    }
    provenance["source_file_sha256"] = {
        checkout: {path: sha256(Path("external") / checkout / path) for path in paths}
        for checkout, paths in source_files.items()
    }
    write_json(args.output_dir / "provenance.json", provenance)
    rows = json.loads((args.data_dir / "train.json").read_text())
    val_rows = json.loads((args.data_dir / "val.json").read_text())
    assert len(rows) == 94376 and len(val_rows) == 11797
    sizes = {"generator_train": args.generator_size, "reranker_train": args.reranker_size, "dev": args.dev_size}
    splits, stats = make_splits(rows, val_rows, args.seed, sizes)
    manifest = {"seed": args.seed, "source_train_sha256": files["train.json"]["sha256"],
                "selection": "seeded shuffle of one minimum-index representative per exact question-or-program connected component; exclude any component touching official val", "deduplication": stats, "splits": {}}
    for name, ids in splits.items():
        path = args.split_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in ids:
                row = rows[i]
                # Deliberately omit choice options/SPARQL from the model input data.
                record = {"id": f"train:{i}", "source_index": i, "question": row["question"],
                          "program": row["program"], "answer": row["answer"]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["splits"][name] = {"count": len(ids), "indices": ids, "path": str(path),
                                    "sha256": sha256(path),
                                    "qualifier_questions": sum(bool(QUALIFIERS & {s["function"] for s in rows[i]["program"]}) for i in ids)}
    write_json(args.output_dir / "split_manifest.json", manifest)
    print(json.dumps({"split_sizes": sizes, "deduplication": stats}, ensure_ascii=False), flush=True)
    if args.replay:
        executor = KoPLExecutor(args.data_dir / "kb.json", backend=args.backend)
        report = {"purpose": "gold program infrastructure replay, not model accuracy",
                  "backend": args.backend, "source_commit": BASELINES_COMMIT if args.backend == "baseline" else KOPL_COMMIT,
                  "kb_sha256": files["kb.json"]["sha256"],
                  "split_manifest_sha256": sha256(args.output_dir / "split_manifest.json"), "splits": {}}
        start = time.monotonic()
        for name, ids in splits.items():
            counts, mismatches = Counter(), []
            for offset, i in enumerate(ids):
                row = rows[i]
                result = executor.execute(row["program"])
                counts["total"] += 1
                counts["valid"] += result["valid"]
                correct = result["valid"] and compare_answers(row["answer"], result["prediction"])
                counts["correct"] += correct
                roundtrip = executor.execute(serialize_program(row["program"]))
                counts["serialization_execution_agreement"] += result == roundtrip
                if not correct:
                    mismatches.append({"id": f"train:{i}", "expected": row["answer"], **result})
                if (offset + 1) % 1000 == 0:
                    print(name, offset + 1, dict(counts), flush=True)
            report["splits"][name] = {**counts, "accuracy": counts["correct"] / counts["total"], "mismatches": mismatches}
        report["seconds"] = time.monotonic() - start
        output_name = "gold_replay.json" if args.backend == "baseline" else "gold_replay_modern.json"
        write_json(args.output_dir / output_name, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
