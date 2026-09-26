"""Offline inventory of local research inputs; never downloads or modifies data.

python3 -B scripts/audit_workspace.py --write
python3 -B scripts/audit_workspace.py --verify
The public manifest contains relative paths, sizes and hashes, not source text.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "artifacts/data_manifest.json"
LOCAL_ROOTS = ["manuals", "data", "radar_corpus", "kg_v3", "graphrag_index",
               "ca_agraphrag/data", "extraction_results", "datasets/kqa_pro",
               "datasets/webqsp", "datasets/mintaka", "datasets/lc_quad",
               "pipeline/v3/work", "pipeline/v3/cache"]
EXCLUDED_PARTS = {"__pycache__", "naval_appendix_pages", "naval_manual",
                  "globalsecurity_cache", "radartutorial_cache", "retrieval_index"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    files = []
    for folder in LOCAL_ROOTS:
        for p in sorted((ROOT / folder).rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            if EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts):
                continue
            if p.suffix in {".faiss", ".pkl", ".index", ".log", ".err", ".pyc"}:
                continue
            files.append({"path": p.relative_to(ROOT).as_posix(),
                          "bytes": p.stat().st_size, "sha256": sha256(p)})
    return files


def data_summary():
    from ca_agraphrag.kg_tools import KGTools
    raw = json.loads((ROOT / "kg_v3/edges.json").read_text())
    entities = json.loads((ROOT / "kg_v3/entities.json").read_text())
    splits, summary = {}, {}
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in
                (ROOT / f"ca_agraphrag/data/{split}.jsonl").read_text().splitlines()
                if line.strip()]
        splits[split] = rows
        summary[split] = {"count": len(rows), "types": dict(Counter(r["type"] for r in rows))}

    def anchors(rows):
        return {json.dumps(r["anchor"], ensure_ascii=False, sort_keys=True) for r in rows}

    def query_entities(rows):
        return {s[k] for r in rows for s in r["gold_support"]
                for k in ("head", "a", "b") if isinstance(s.get(k), str)}

    test_entities = query_entities(splits["test"])
    return {"raw_edges": len(raw), "entities": len(entities),
            "relations": len({e["relation"] for e in raw}),
            "rl_filtered_edge_records": KGTools(filtered=True).n_edges,
            "splits": summary,
            "sft_trajectories": sum(bool(l.strip()) for l in
                                    (ROOT / "ca_agraphrag/data/sft_train.jsonl").read_text().splitlines()),
            "train_test_anchor_overlap": len(anchors(splits["train"]) & anchors(splits["test"])),
            "test_explicit_query_entities": len(test_entities),
            "train_test_explicit_query_entity_overlap": len(query_entities(splits["train"]) & test_entities),
            "query_entity_definition": "gold_support fields head/a/b; excludes answer entities and implicit references",
            "filter_code_sha256": sha256(ROOT / "ca_agraphrag/kg_tools.py")}


def verify():
    manifest = json.loads(MANIFEST.read_text())
    missing, changed = [], []
    for item in manifest["files"]:
        p = ROOT / item["path"]
        if not p.is_file():
            missing.append(item["path"])
        elif p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            changed.append(item["path"])
    print(json.dumps({"checked": len(manifest["files"]), "missing": missing,
                      "changed": changed}, ensure_ascii=False, indent=2))
    return 1 if missing or changed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    summary = data_summary()
    if args.write:
        MANIFEST.parent.mkdir(exist_ok=True)
        manifest = {"schema_version": 1, "snapshot_date": "2026-09-26",
                    "distribution": "local-only; restore from authorized local backup, no public download promised",
                    "scope": "research inputs and expensive extraction outputs; excludes regenerable binary indexes and render caches",
                    "summary": summary, "files": inventory()}
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote {MANIFEST.relative_to(ROOT)} ({len(manifest['files'])} files)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
