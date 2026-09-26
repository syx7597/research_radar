#!/usr/bin/env python3
"""Mechanical, gold-aware coverage audit; this is not a repair algorithm.

The audit never changes predictions, ranks, labels or model parameters. The
single-field diagnostic requires exact function/dependency/argument agreement
except one schema argument. Gold programs are replayed only for offline audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.condition_consistency.executor import (
    BASELINES_COMMIT, KoPLExecutor, compare_answers, validate_program,
)
from experiments.condition_consistency.execute_predictions import candidate_deadline


FIELD_ROLES = {
    "FilterConcept": {0: "concept"},
    **{f"Filter{x}": {0: "attribute"} for x in ("Str", "Num", "Year", "Date")},
    **{f"QFilter{x}": {0: "qualifier"} for x in ("Str", "Num", "Year", "Date")},
    "Relate": {0: "relation"},
    "SelectBetween": {0: "attribute"}, "SelectAmong": {0: "attribute"},
    "QueryAttr": {0: "attribute"},
    "QueryAttrUnderCondition": {0: "attribute", 1: "qualifier"},
    "QueryAttrQualifier": {0: "attribute", 2: "qualifier"},
    "QueryRelationQualifier": {0: "relation", 1: "qualifier"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def build_schema(kb):
    roles = {role: set() for role in ("attribute", "relation", "qualifier", "concept")}
    roles["concept"].update(c["name"] for c in kb["concepts"].values())
    types = {role: defaultdict(set) for role in ("attribute", "qualifier")}
    for entity in kb["entities"].values():
        for fact in entity["attributes"]:
            roles["attribute"].add(fact["key"])
            types["attribute"][fact["key"]].add(fact["value"]["type"])
        for fact in entity["relations"]:
            roles["relation"].add(fact["predicate"])
        for fact in entity["attributes"] + entity["relations"]:
            for key, values in fact.get("qualifiers", {}).items():
                roles["qualifier"].add(key)
                types["qualifier"][key].update(value["type"] for value in values)
    return roles, types


def field_violations(program, schema):
    """Inference-visible global-role check, without gold or execution answers.

    Local absence is deliberately NOT a violation. Empty/zero answers are not
    consulted. A violation may coexist with an accidentally correct answer.
    """
    if not isinstance(program, list):
        return []
    result = []
    for index, step in enumerate(program):
        for arg, role in FIELD_ROLES.get(step["function"], {}).items():
            value = step["inputs"][arg]
            if value not in schema[role]:
                result.append({"step": index, "argument": arg,
                               "function": step["function"], "role": role, "value": value})
    return result


def argument_role(function, index, value, target):
    if index in FIELD_ROLES.get(function, {}):
        return FIELD_ROLES[function][index]
    if function == "Find":
        return "entity"
    if function == "Relate" and index == 1:
        return "direction"
    if (function in {"SelectBetween", "SelectAmong"} and index == 1
            or function.endswith(("Num", "Year", "Date"))
            and value in {"<", ">", "=", "!="} and target in {"<", ">", "=", "!="}):
        return "comparison_operator"
    if function.endswith(("Num", "Year", "Date")) or re.search(r"\d", value + target):
        return "numeric_date_or_identifier_literal"
    return "other_literal"


def difference(program, gold):
    if not isinstance(program, list):
        return {"category": "unparseable_program", "differences": []}
    try:
        program, gold = validate_program(program), validate_program(gold)
    except ValueError as error:
        return {"category": "unparseable_program", "validation_error": str(error), "differences": []}
    if [s["function"] for s in program] != [s["function"] for s in gold]:
        return {"category": "structure_function_sequence", "differences": [],
                "predicted_functions": [s["function"] for s in program],
                "gold_functions": [s["function"] for s in gold]}
    if [s["dependencies"] for s in program] != [s["dependencies"] for s in gold]:
        return {"category": "structure_dependency", "differences": []}
    changes = []
    for index, (pred, target) in enumerate(zip(program, gold)):
        for arg, (value, expected) in enumerate(zip(pred["inputs"], target["inputs"])):
            if value != expected:
                changes.append({"step": index, "argument": arg, "function": pred["function"],
                                "role": argument_role(pred["function"], arg, value, expected),
                                "predicted": value, "gold": expected})
    roles = {c["role"] for c in changes}
    if not changes:
        category = "identical_gold_program"
    elif len(changes) == 1:
        role = changes[0]["role"]
        category = "single_field" if role in {"attribute", "relation", "qualifier", "concept"} else "single_" + role
    elif roles <= {"attribute", "relation", "qualifier", "concept"}:
        category = "multiple_fields"
    elif roles == {"entity"}:
        category = "multiple_entities_or_entity_order"
    else:
        category = "mixed_arguments"
    return {"category": category, "differences": changes}


def aggregate(details):
    return {
        "questions": len(details),
        "categories": dict(sorted(Counter(d["reference_diff"]["category"] for d in details).items())),
        "reference_is_first_valid": sum(d["reference_kind"] == "first_valid" for d in details),
        "reference_is_top1_when_no_valid_candidate": sum(d["reference_kind"] == "top1_no_valid_candidate" for d in details),
        "gold_replay_correct": sum(d["gold_replay_correct"] for d in details),
        "single_field_gold_replay_correct": sum(d["single_field_gold_replay_correct"] for d in details),
        "single_core_field_gold_replay_correct": sum(d["single_core_field_gold_replay_correct"] for d in details),
        "single_field_and_changed_field_globally_invalid": sum(d["single_field_gold_replay_correct"] and d["changed_field_globally_invalid"] for d in details),
        "single_core_field_and_changed_field_globally_invalid": sum(d["single_core_field_gold_replay_correct"] and d["changed_field_globally_invalid"] for d in details),
        "reference_has_any_global_field_violation": sum(bool(d["reference_field_violations"]) for d in details),
        "any_wrong_candidate_single_field_gold_match": sum(bool(d["candidate_single_field_oracle_indices"]) for d in details),
        "any_wrong_candidate_single_core_field_gold_match": sum(bool(d["candidate_single_core_field_oracle_indices"]) for d in details),
        "any_wrong_candidate_single_field_gold_match_and_changed_field_invalid": sum(bool(d["candidate_single_field_invalid_oracle_indices"]) for d in details),
        "any_wrong_candidate_single_core_field_gold_match_and_changed_field_invalid": sum(bool(d["candidate_single_core_field_invalid_oracle_indices"]) for d in details),
        "ids": [d["id"] for d in details],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/condition_consistency/pilot_bart5k_seed20260926"))
    parser.add_argument("--kb", type=Path, default=Path("datasets/kqa_pro/kb.json"))
    parser.add_argument("--output", type=Path, default=Path("results/condition_consistency/query_repair_audit"))
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("Gold replay timeout must be positive")
    rows_path = args.pilot / "dev_executed.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    dev_metrics = json.loads((args.pilot / "dev_metrics.json").read_text())
    metric_rows = {r["id"]: r for r in dev_metrics["per_question"]}
    ranker_metrics = {seed: json.loads((args.pilot / f"targeted_metrics_seed{seed}.json").read_text()) for seed in (17, 29, 43)}
    ranker_rows = {seed: {r["id"]: r for r in metric["per_question"]} for seed, metric in ranker_metrics.items()}
    cache_hash = digest(rows_path)
    cache_meta = json.loads(rows_path.with_suffix(".jsonl.meta.json").read_text())
    if cache_meta["kb_sha256"] != digest(args.kb) or cache_meta["executor_backend"] != "baseline":
        raise ValueError("Audit KB/backend differ from executed candidate provenance")
    for label, metric in [("dev", dev_metrics), *ranker_metrics.items()]:
        if metric["candidate_cache_sha256"] != cache_hash:
            raise ValueError(f"Cache provenance mismatch: {label}")
        if set(r["id"] for r in metric["per_question"]) != set(r["id"] for r in rows):
            raise ValueError(f"ID mismatch: {label}")
    kb = json.loads(args.kb.read_text())
    schema, types = build_schema(kb)
    del kb
    executor = KoPLExecutor(args.kb, backend="baseline")
    details, trigger_rows, regression_details = [], [], []
    for row in rows:
        qid, candidates = row["id"], row["candidates"]
        outcomes = [c["valid"] and compare_answers(row["answer"], c.get("prediction")) for c in candidates]
        selected = next((i for i, c in enumerate(candidates) if c["valid"]), None)
        first_correct = selected is not None and outcomes[selected]
        mr = metric_rows[qid]
        if selected != mr["selected_indices"]["first_valid"] or first_correct != mr["correct"]["first_valid"] or any(outcomes) != mr["correct"]["oracle_at_k"]:
            raise ValueError(f"Recomputed outcome differs from frozen metrics: {qid}")
        reference_index = selected if selected is not None else (0 if candidates else None)
        reference = candidates[reference_index] if reference_index is not None else {"program": None}
        violations = field_violations(reference.get("program"), schema)
        trigger_rows.append({"id": qid, "first_valid_correct": first_correct,
                             "first_valid_index": selected, "reference_index": reference_index,
                             "reference_has_global_field_violation": bool(violations),
                             "reference_field_violations": violations})
        regressed_seeds = []
        for seed, seed_rows in ranker_rows.items():
            rm = seed_rows[qid]
            if first_correct and not rm["correct"]["linear_ranker"]:
                regressed_seeds.append(seed)
        if first_correct and not regressed_seeds:
            continue
        with candidate_deadline(args.timeout_seconds):
            gold_replay = executor.execute(row["program"])
        gold_correct = gold_replay["valid"] and compare_answers(row["answer"], gold_replay["prediction"])
        diffs = [difference(c.get("program"), row["program"]) for c in candidates]
        ref_diff = diffs[reference_index] if reference_index is not None else difference(None, row["program"])
        single = ref_diff["category"] == "single_field" and gold_correct and not first_correct
        core = single and ref_diff["differences"][0]["role"] != "concept"
        invalid_changed = single and ref_diff["differences"][0]["predicted"] not in schema[ref_diff["differences"][0]["role"]]
        oracle_indices = [i for i, d in enumerate(diffs) if not outcomes[i] and gold_correct and d["category"] == "single_field"]
        invalid_oracle_indices = [i for i in oracle_indices if diffs[i]["differences"][0]["predicted"] not in schema[diffs[i]["differences"][0]["role"]]]
        record = {
            "id": qid, "first_valid_correct": first_correct, "first_valid_index": selected,
            "reference_index": reference_index,
            "reference_kind": "first_valid" if selected is not None else "top1_no_valid_candidate",
            "oracle_at_4_correct": any(outcomes), "correct_candidate_indices": [i for i, yes in enumerate(outcomes) if yes],
            "reference_diff": ref_diff, "reference_field_violations": violations,
            "gold_replay_correct": gold_correct, "gold_replay": gold_replay,
            "single_field_gold_replay_correct": single, "single_core_field_gold_replay_correct": core,
            "changed_field_globally_invalid": bool(invalid_changed),
            "candidate_single_field_oracle_indices": oracle_indices,
            "candidate_single_core_field_oracle_indices": [i for i in oracle_indices if diffs[i]["differences"][0]["role"] != "concept"],
            "candidate_single_field_invalid_oracle_indices": invalid_oracle_indices,
            "candidate_single_core_field_invalid_oracle_indices": [i for i in invalid_oracle_indices if diffs[i]["differences"][0]["role"] != "concept"],
            "candidate_differences": diffs,
        }
        if not first_correct:
            details.append(record)
        if regressed_seeds:
            selected_by_seed = {str(seed): ranker_rows[seed][qid]["selected_indices"]["linear_ranker"] for seed in regressed_seeds}
            indices = sorted(set(selected_by_seed.values()))
            regression_details.append({
                "id": qid, "seeds": regressed_seeds, "first_valid_index": selected,
                "ranker_indices": selected_by_seed, "first_valid_diff_to_gold": ref_diff,
                "ranker_diff_to_first_valid": {str(i): difference(candidates[i].get("program"), reference["program"]) for i in indices},
                "ranker_diff_to_gold": {str(i): diffs[i] for i in indices},
                "ranker_global_field_violations": {str(i): field_violations(candidates[i].get("program"), schema) for i in indices},
            })
    triggered = [r for r in trigger_rows if r["reference_has_global_field_violation"]]
    summary = {
        "purpose": "Mechanical offline coverage audit, not a new model or repair accuracy result",
        "scope": "All 500 frozen development questions; selected first-valid candidate, or top1 diagnostic only if none is valid",
        "single_field_definition": "Exactly one attribute/relation/qualifier/concept argument differs from gold, all functions/dependencies/other inputs equal, candidate answer wrong and gold replay answer correct",
        "core_field_definition": "The single-field definition excluding FilterConcept concept edits",
        "oracle_warning": "Gold equality is a restricted reference-program coverage diagnostic, not a global semantic repair upper bound and not achievable accuracy; alternate correct programs may exist",
        "trigger_definition": "Exact field string absent from its global KB role schema; no gold, local-absence, empty-result or zero-answer trigger",
        "source_sha256": {"dev_executed.jsonl": cache_hash, "kb.json": digest(args.kb),
                          "dev_metrics.json": digest(args.pilot / "dev_metrics.json"),
                          **{f"targeted_metrics_seed{s}.json": digest(args.pilot / f"targeted_metrics_seed{s}.json") for s in ranker_metrics},
                          "audit_script": digest(Path(__file__))},
        "executor_backend": "baseline", "executor_source_commit": BASELINES_COMMIT,
        "executor_source_file_sha256": digest(ROOT / "external/kqa_pro_baselines/Program/executor_rule.py"),
        "per_gold_replay_timeout_seconds": args.timeout_seconds,
        "questions": len(rows), "first_valid_correct": sum(r["first_valid_correct"] for r in trigger_rows),
        "remaining_errors": aggregate(details),
        "remaining_errors_with_first_valid_output": aggregate([d for d in details if d["first_valid_index"] is not None]),
        "remaining_errors_without_any_valid_candidate": aggregate([d for d in details if d["first_valid_index"] is None]),
        "remaining_errors_with_correct_cached_candidate": aggregate([d for d in details if d["oracle_at_4_correct"]]),
        "remaining_errors_without_correct_cached_candidate": aggregate([d for d in details if not d["oracle_at_4_correct"]]),
        "global_role_trigger_reference": {
            "triggered_questions": len(triggered),
            "baseline_wrong": sum(not r["first_valid_correct"] for r in triggered),
            "baseline_correct": sum(r["first_valid_correct"] for r in triggered),
            "ids_of_baseline_correct": [r["id"] for r in triggered if r["first_valid_correct"]],
            "warning": "An invalid schema field does not prove that the final answer is wrong; the baseline may be accidentally correct",
        },
        "targeted_ranker_regression_ids_by_seed": {str(s): [d["id"] for d in regression_details if s in d["seeds"]] for s in ranker_metrics},
        "schema_role_counts": {role: len(values) for role, values in schema.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "schema.json", {
        "source_kb_sha256": digest(args.kb),
        "field_argument_roles": FIELD_ROLES,
        "names_by_role": {role: sorted(values) for role, values in schema.items()},
        "observed_value_types_by_role": {role: {name: sorted(v) for name, v in sorted(values.items())} for role, values in types.items()},
        "warning": "KB-derived schema only. Observed local absence is not a trigger. No question/gold-derived aliases.",
    })
    write_json(args.output / "regressions.json", regression_details)
    # Full diagnostics stay local under the existing JSONL ignore rule.
    with (args.output / "details.jsonl").open("w") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output / "triggers.jsonl").open("w") as handle:
        for record in trigger_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k in {"questions", "first_valid_correct", "remaining_errors", "global_role_trigger_reference", "schema_role_counts"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
