"""Freeze source-isolated train/development/holdout tables before model runs."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.agent_feedback.recovery import sha256
from .table_program import TableExecutor, execute_actions, program_answer_spec, digest, compact
from .table_scoring import CatalogIndex, score_answer, validate_spec

BASE = Path("data/radar_domain_recovery_v2")
OUT = BASE / "frozen_20261107"
PUBLIC = Path("results/radar_domain/frozen_20261107")


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, obj, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        if jsonl:
            for row in obj:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(obj, stream, ensure_ascii=False, indent=2); stream.write("\n")


def binding(path, rows=None):
    result = {"path": str(path), "sha256": sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def specs(path):
    obj = read(path)
    if "answer_specs" in obj:
        return obj["answer_specs"]
    return {r["id"]: r["answer_spec"] for r in obj["rows"]}


def formatted_question(question, spec):
    # The supplied specification contains only output names, kinds and columns.
    for output in spec["outputs"]:
        expected = {"name", "kind", "columns"} if output["kind"] == "table" else {"name", "kind"}
        if set(output) != expected:
            raise ValueError("Answer-bearing metadata must never enter model input")
    return question + "\n\n请按以下输出结构提交查询结果：" + compact(spec)


def task_contract():
    return {
        "schema": "radar_table_task_contract_v1", "date": "2026-11-07",
        "input": "Natural question plus value-free output names/kinds/columns; complete assigned source catalog accessed through tools",
        "actions": {"step": ["function", "inputs", "dependencies"], "finish": ["answer_handle"]},
        "operations": ["Scan", "Expand", "Project", "Join", "Filter", "Derive", "Assess", "Assemble"],
        "answer": "radar_table_answer_v1",
        "types": ["number", "bound", "range", "options", "correspondence", "tolerance", "approximate", "text"],
        "empty_answer": "Empty table with declared columns and actual executed search scope/predicates/evidence",
        "unknown_answer": "Assessment status not_established with actual source domain/restriction evidence",
        "execution_failure": "No successfully submitted answer; retained in denominator",
        "primary_metric": "Complete structured answer including values, units, bounds, conditions, pairings and provenance",
        "ordering": "Table row order, column presentation order and explicit option/correspondence row order are immaterial; value associations and other sequence semantics are retained",
        "source_support": "Every answer cell/assessment and empty search scope binds to the actual immutable catalog; fabricated or altered source records fail",
        "units": "Declared compatible unit scales and aliases normalize; incompatible dimensions fail; approximation and tolerance remain distinct types",
        "secondary_metrics": ["completion", "well_formed_answer", "tool_calls", "invalid_calls", "input_tokens", "generated_tokens", "per_source_family", "per_task_type"],
        "statistics": {"paired_question_bootstrap_replicates": 5000, "seed": 20261107,
                       "source_family_results": "report all four real holdout families",
                       "primary_population": "24 real new-source questions; synthetic development reported separately"},
        "selection": "Development may guide method choices; holdout is fixed before training and not used for tuning",
        "model_runs": 0, "human_gold": False,
    }


def main():
    if OUT.exists() or PUBLIC.exists():
        raise FileExistsError("Dataset already frozen or partially written; inspect instead of replacing")
    syn = BASE / "synthetic_20261107"
    real = BASE / "programs_dev_20261107"
    hold = BASE / "holdout_programs_20261107"
    syn_records = read(syn / "typed_records.json")["records"]
    real_records = read(BASE / "typed_sources_20261107/typed_records.json")["records"]
    hold_records = read(hold / "typed_records.json")["records"]
    group_doc = read(syn / "source_group_by_id.json")
    qgroups = group_doc["source_group_by_id"]
    split_groups = {k: set(v) for k, v in group_doc["split_groups"].items()}
    candidates = {r["task_id"]: r for r in read(
        "data/radar_domain_recovery_v1/development_task_candidates.json")["tasks"]}
    hold_q = {r["question_id"]: r for r in read(
        BASE / "new_sources_20261107/questions.json")["rows"]}
    packages = {
        "train": [("synthetic", read_rows(syn / "train.references.jsonl"), specs(syn / "answer_specs.json"))],
        "dev": [("synthetic", read_rows(syn / "synthetic_dev.references.jsonl"), specs(syn / "answer_specs.json")),
                ("real_development", read_rows(real / "references.jsonl"), specs(real / "answer_specs.json"))],
        "holdout": [("real_holdout", read_rows(hold / "references.jsonl"), specs(hold / "answer_specs.json"))],
    }
    catalogs = {
        "train": [r for r in syn_records if r["source_group_id"] in split_groups["train"]],
        "dev": [r for r in syn_records if r["source_group_id"] in split_groups["synthetic_dev"]] + real_records,
        "holdout": hold_records,
    }
    all_ids, family_sets, record_sets, subject_sets = set(), {}, {}, {}
    split_manifest, all_traces, before_after = {}, {}, []
    for split, rows in catalogs.items():
        family_sets[split] = {r["family_id"] for r in rows}
        record_sets[split] = {r["record_id"] for r in rows}
        subject_sets[split] = {s for r in rows for s in r["subjects"]}
        assert len(record_sets[split]) == len(rows)
    for a, b in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
        if family_sets[a] & family_sets[b] or record_sets[a] & record_sets[b] or subject_sets[a] & subject_sets[b]:
            raise ValueError("Family, fact identity or subject overlaps across source-isolated catalogs")
    source_paths = [
        syn / "full_manifest.json", syn / "typed_records.json", syn / "source_group_by_id.json",
        syn / "answer_specs.json", syn / "train.references.jsonl", syn / "synthetic_dev.references.jsonl",
        BASE / "typed_sources_20261107/typed_records.json", real / "references.jsonl", real / "answer_specs.json",
        hold / "typed_records.json", hold / "references.jsonl", hold / "answer_specs.json", hold / "rubric.json",
        BASE / "new_sources_20261107/manifest.json",
        Path("artifacts/thesis_direction_review/radar_new_sources_review_20261107/review_summary.safe.json"),
        Path("experiments/radar_domain/table_program.py"), Path("experiments/radar_domain/table_scoring.py"),
        Path(__file__),
    ]
    original_hashes = {str(p): sha256(p) for p in source_paths}
    contract_path = OUT / "task_contract.json"
    write(contract_path, task_contract())
    for split, package_list in packages.items():
        folder = OUT / split
        catalog = {"schema": "radar_typed_catalog_v2", "split": split, "records": catalogs[split],
                   "source_families": sorted(family_sets[split]), "human_gold": False,
                   "complete_source_groups": True, "question_specific_gold_pruning": False}
        catalog_path = folder / "catalog.json"
        write(catalog_path, catalog)
        executor, index = TableExecutor(catalog), CatalogIndex(catalog)
        questions, references, metadata, traces, success = [], [], [], [], []
        for origin, refs, answer_specs in package_list:
            for old in refs:
                ident = old["id"]
                if ident in all_ids:
                    raise ValueError("Question ID overlap")
                all_ids.add(ident)
                spec = answer_specs[ident]
                question = formatted_question(old["question"], spec)
                episode = execute_actions(executor, question, old["actions"])
                # Catalog isolation must not change any answer content or evidence.
                adjusted = deepcopy(old["answer"])
                adjusted["scope"] = deepcopy(episode.answer["scope"])
                check = score_answer(episode.answer, adjusted, index, spec)
                if not check["correct"]:
                    raise ValueError(f"Source partition changed a reference answer: {ident}: {check}")
                reference = {"id": ident, "question": question, "actions": old["actions"], "answer": episode.answer}
                questions.append({"id": ident, "question": question})
                references.append(reference)
                if origin == "synthetic":
                    group = "synthetic:" + qgroups[ident]
                    # The source generation manifest retains per-template categories.
                    task_type = ident.rsplit(":", 1)[-1]
                elif origin == "real_development":
                    group = candidates[ident]["family_id"]
                    task_type = "source_composition"
                else:
                    group = hold_q[ident]["family_id"]
                    task_type = hold_q[ident]["primary_type"]
                assert group in family_sets[split]
                metadata.append({"id": ident, "origin": origin, "source_group": group,
                    "task_type": task_type, "original_question": old["question"],
                    "original_question_sha256": hashlib.sha256(old["question"].encode()).hexdigest(), "answer_spec": spec,
                    "training_allowed": split == "train"})
                traces.append({"id": ident, "messages": episode.messages, "events": episode.events,
                               "calls": episode.calls, "model_generated": False})
                messages = deepcopy(episode.messages) + [{"role": "assistant", "content": "Done."}]
                success.append({"id": ident, "messages": messages,
                    "supervise": [m["role"] == "assistant" for m in messages],
                    "source": "executed_reference_success", "origin": origin,
                    "training_eligible": split == "train", "model_failure": False})
                before_after.append({"id": ident, "split": split, "answer_unchanged_after_catalog_isolation": True})
        for name, data in (("questions.jsonl", questions), ("references.jsonl", references),
                           ("metadata.jsonl", metadata), ("reference_replays.jsonl", traces),
                           ("success_trajectories.jsonl", success)):
            write(folder / name, data, True)
        provenance = {"schema": "radar_table_reference_provenance_v1", "split": split,
            "origin": "synthetic_generated_non_equipment_fact" if split == "train" else "development_or_holdout_reference",
            "model_generated": False, "human_gold": False, "references_verified_by_execution": len(references),
            "input_origins": dict(Counter(x["origin"] for x in metadata)), "catalog": binding(catalog_path),
            "references": binding(folder / "references.jsonl", len(references))}
        write(folder / "reference_provenance.json", provenance)
        split_manifest[split] = {
            "ids": [q["id"] for q in questions],
            "group_by_id": {m["id"]: m["source_group"] for m in metadata},
            "source_families": sorted(family_sets[split]), "source_record_count": len(catalogs[split]),
            "questions": binding(folder / "questions.jsonl", len(questions)),
            "references": binding(folder / "references.jsonl", len(references)),
            "metadata": binding(folder / "metadata.jsonl", len(metadata)),
            "catalog": binding(catalog_path), "reference_provenance": binding(folder / "reference_provenance.json"),
            "success_trajectories": binding(folder / "success_trajectories.jsonl", len(success)),
            "origin_counts": dict(Counter(m["origin"] for m in metadata)),
            "max_reference_calls": max(t["calls"] for t in traces),
        }
        all_traces[split] = {"references": len(references), "successful_calls": sum(t["calls"] for t in traces)}
    expected_counts = {"train": 576, "dev": 108, "holdout": 24}
    assert {k: len(v["ids"]) for k, v in split_manifest.items()} == expected_counts
    for name, expected in original_hashes.items():
        assert sha256(name) == expected, "An input changed during freeze"
    write(OUT / "catalog_isolation_check.json", before_after)
    manifest = {
        "schema": "radar_domain_table_split_v1", "status": "frozen", "date": "2026-11-07",
        "task_contract": binding(contract_path), "splits": split_manifest,
        "model_results_seen": False, "model_runs": 0, "training_started": False,
        "server_used": False, "formal_model_failure_pairs": 0,
        "split_unit": "Complete synthetic source instance or real source family; shared facts and subjects stay in one partition",
        "family_overlap": 0, "record_identity_overlap": 0, "subject_overlap": 0,
        "catalog_isolation": "Separate complete catalogs; train tools cannot access dev/holdout facts",
        "success_reference_role": "Train references may supervise I_domain; recovery pairs await real model-generated train failures",
        "evaluation_policy": "Keep all 24 real new-source questions; never tune on their results",
        "inputs_sha256": original_hashes,
    }
    write(OUT / "split_manifest.json", manifest)
    report = {
        "schema": "radar_offline_dataset_frozen_summary_v1", "date": "2026-11-07",
        "status": "frozen_before_model_runs", "question_counts": expected_counts,
        "source_record_counts": {k: len(v) for k, v in catalogs.items()},
        "source_family_counts": {k: len(v) for k, v in family_sets.items()},
        "reference_execution": all_traces, "all_original_reference_answers_preserved": True,
        "family_overlap": 0, "record_identity_overlap": 0, "subject_overlap": 0,
        "question_inputs_include": ["original question", "output names/kinds/columns only"],
        "training_origin": "Explicit synthetic source instances, not equipment facts",
        "real_development_questions": 12, "synthetic_development_questions": 96,
        "real_holdout_questions": 24, "real_holdout_source_families": 4,
        "AI_references": True, "human_gold": False, "model_runs": 0, "training_started": False,
        "freeze_manifest": binding(OUT / "split_manifest.json"),
        "task_contract": binding(contract_path),
    }
    write(PUBLIC / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
