"""Replay a frozen table dataset and audit isolation without changing its files."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.radar_domain.table_program import TableExecutor, execute_actions, program_answer_spec
from experiments.radar_domain.table_scoring import CatalogIndex, score_answer
from experiments.radar_domain.freeze_offline_dataset import formatted_question


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def audit(manifest_path):
    manifest_path = Path(manifest_path)
    m = load(manifest_path)
    assert m["schema"] == "radar_domain_table_split_v1" and m["status"] == "frozen"
    hashes_checked = 0
    revision_path = Path("results/radar_domain/frozen_20261107/scoring_revision.json")
    revision = load(revision_path)
    assert revision["schema"] == "radar_table_scoring_revision_v1"
    assert revision["path"] == "experiments/radar_domain/table_scoring.py"
    assert not revision["model_outputs_seen"] and not revision["model_results_used_for_change"]
    assert revision["frozen_manifest_sha256"] == sha(manifest_path)
    assert sha(revision["archived_original"]["path"]) == revision["archived_original"]["sha256"] == revision["old_sha256"]
    for path, expected in m["inputs_sha256"].items():
        if path == revision["path"]:
            assert expected == revision["old_sha256"] and sha(path) == revision["new_sha256"]
        else:
            assert sha(path) == expected, f"Frozen input changed: {path}"
        hashes_checked += 1
    assert sha(m["task_contract"]["path"]) == m["task_contract"]["sha256"]
    sets, counts = {}, {}
    for split, desc in m["splits"].items():
        for name in ("questions", "references", "metadata", "catalog", "reference_provenance", "success_trajectories"):
            entry = desc[name]
            assert sha(entry["path"]) == entry["sha256"], (split, name)
            hashes_checked += 1
        qs, refs, metadata, traces = [rows(desc[n]["path"]) for n in (
            "questions", "references", "metadata", "success_trajectories")]
        ids = desc["ids"]
        assert all([r["id"] for r in data] == ids for data in (qs, refs, metadata, traces))
        assert len(ids) == len(set(ids))
        catalog = load(desc["catalog"]["path"])
        records = catalog["records"]
        index, executor = CatalogIndex(catalog), TableExecutor(catalog)
        families = {r["family_id"] for r in records}
        assert families == set(desc["source_families"])
        assert len(records) == desc["source_record_count"]
        sets[split] = {"families": families, "records": {r["record_id"] for r in records},
                       "subjects": {s for r in records for s in r["subjects"]}, "questions": set(ids)}
        calls = 0
        for q, ref, meta, trace in zip(qs, refs, metadata, traces):
            assert set(q) == {"id", "question"}
            assert q["question"] == ref["question"] == formatted_question(meta["original_question"], meta["answer_spec"])
            assert hashlib.sha256(meta["original_question"].encode()).hexdigest() == meta["original_question_sha256"]
            assert meta["source_group"] in families
            assert meta["training_allowed"] == trace["training_eligible"] == (split == "train")
            assert not trace["model_failure"]
            if split == "train":
                assert meta["origin"] == "synthetic"
            schema = program_answer_spec(ref["actions"])
            for spec in (schema, meta["answer_spec"]):
                assert set(spec) == {"outputs"}
                for output in spec["outputs"]:
                    assert set(output) == ({"name", "kind", "columns"} if output["kind"] == "table" else {"name", "kind"})
            episode = execute_actions(executor, q["question"], ref["actions"])
            assert score_answer(episode.answer, ref["answer"], index, meta["answer_spec"])["correct"]
            assert trace["messages"] == episode.messages + [{"role": "assistant", "content": "Done."}]
            assert trace["supervise"] == [v["role"] == "assistant" for v in trace["messages"]]
            calls += episode.calls
        counts[split] = {"references": len(refs), "source_records": len(records),
                         "source_families": len(families), "successful_calls": calls}
    for a, b in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
        for key in sets[a]:
            assert not sets[a][key] & sets[b][key], (a, b, key)
    assert counts["train"]["references"] == 576 and counts["dev"]["references"] == 108 and counts["holdout"]["references"] == 24
    return {"schema": "radar_frozen_table_cpu_audit_v1", "status": "passed", "date": "2026-11-07",
            "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
            "auditor_code_sha256": sha(__file__), "verified_input_hashes": hashes_checked + 1,
            "scoring_revision": {"path": str(revision_path), "sha256": sha(revision_path)},
            "split_results": counts, "references_replayed": sum(c["references"] for c in counts.values()),
            "source_group_overlap": 0, "source_record_overlap": 0, "source_subject_overlap": 0,
            "question_overlap": 0, "all_training_sources_synthetic": True,
            "format_specs_contain_only_output_names_kinds_columns": True,
            "reference_success_masks_verified": True, "server_accessed": False,
            "model_runs": 0, "human_review": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json"))
    p.add_argument("--output", type=Path, default=Path("results/radar_domain/frozen_20261107/cpu_audit.json"))
    args = p.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "replayed": result["references_replayed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
