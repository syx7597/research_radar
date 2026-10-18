"""Post-hoc, read-only decomposition of the already completed project holdout.

Never changes scores, selects checkpoints, or supplies examples for training.
Full example messages stay in an ignored private JSONL; public output is aggregate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.condition_consistency.executor import compare_answers
from .compare import indexed
from .recovery import sha256
from .training_data import read_jsonl

STATES = ("unfinished", "finished_wrong", "correct")
RUNS = {
    "20261003": ("clean_holdout_v1", "recovery_holdout_v1"),
    "20261004": ("clean_holdout_seed20261004", "recovery_holdout_seed20261004"),
}
CASE_SEED = 20261018


def state(row, reference):
    if "prediction" not in row:
        raise ValueError("Missing prediction is not an unfinished prediction")
    finished = row["prediction"] is not None
    if (row.get("stop_reason") == "finished") != finished:
        raise ValueError("Stop reason and final prediction disagree")
    if not finished:
        return "unfinished"
    return "correct" if compare_answers(reference["answer"], row["prediction"]) else "finished_wrong"


def analyze(a_rows, c_rows, gold, seed):
    a, c, refs = indexed(a_rows, "A"), indexed(c_rows, "C"), indexed(gold, "gold")
    if not refs or set(a) != set(refs) or set(c) != set(refs):
        raise ValueError("Complete identical question coverage is required")
    matrix = {x: {y: 0 for y in STATES} for x in STATES}
    cells = {x + "->" + y: [] for x in STATES for y in STATES}
    stops = {arm: Counter() for arm in ("A", "C")}
    for ident, ref in refs.items():
        x, y = state(a[ident], ref), state(c[ident], ref)
        matrix[x][y] += 1
        cells[x + "->" + y].append(ident)
        for arm, row, status in (("A", a[ident], x), ("C", c[ident], y)):
            if status == "unfinished":
                stops[arm][row["stop_reason"]] += 1
    n = len(refs)
    margins = {"A": {x: sum(matrix[x].values()) for x in STATES},
               "C": {y: sum(matrix[x][y] for x in STATES) for y in STATES}}
    decomposition = {
        "unfinished_to_correct": matrix["unfinished"]["correct"],
        "wrong_to_correct": matrix["finished_wrong"]["correct"],
        "correct_to_unfinished": matrix["correct"]["unfinished"],
        "correct_to_wrong": matrix["correct"]["finished_wrong"],
        "completion_boundary_net": matrix["unfinished"]["correct"] - matrix["correct"]["unfinished"],
        "both_finished_net": matrix["finished_wrong"]["correct"] - matrix["correct"]["finished_wrong"],
        "total_correct_net": margins["C"]["correct"] - margins["A"]["correct"],
    }
    if decomposition["completion_boundary_net"] + decomposition["both_finished_net"] != decomposition["total_correct_net"]:
        raise AssertionError("Transition decomposition does not reconstruct accuracy gain")
    cases = []
    for cell, ids in cells.items():
        if not ids or cell == "correct->correct":
            continue
        ident = min(ids, key=lambda key: (hashlib.sha256(
            f"{CASE_SEED}|{seed}|{cell}|{key}".encode()).hexdigest(), str(key)))
        cases.append({"seed": seed, "cell": cell, "id": ident,
                      "reference": refs[ident], "A": a[ident], "C": c[ident]})
    return {"questions": n, "row_A_column_C": matrix, "margins": margins,
            "decomposition": decomposition,
            "unfinished_stop_reasons": {key: dict(val) for key, val in stops.items()},
            "selected_cases": [{k: row[k] for k in ("seed", "cell", "id")} for row in cases]}, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/agent_feedback/state_transitions_v1"))
    args = parser.parse_args()
    root, output = args.root, args.root / args.output
    if output.exists():
        raise FileExistsError("Do not overwrite a completed post-hoc analysis")
    r = root / "results/agent_feedback"
    complete_path = r / "holdout_v1_generation_complete.json"
    summary_path = r / "holdout_v1_summary.json"
    audit_path = r / "holdout_v1_execution_audit.json"
    complete, summary, audit = [json.loads(p.read_text()) for p in (complete_path, summary_path, audit_path)]
    original_analysis_path = root / summary["analysis"]
    if sha256(original_analysis_path) != summary["analysis_sha256"]:
        raise ValueError("Original analysis binding differs")
    original_analysis = json.loads(original_analysis_path.read_text())
    if (summary["generation_complete_sha256"] != sha256(complete_path)
            or summary["execution_audit_sha256"] != sha256(audit_path)
            or audit["audit_passed"] is not True or audit["mismatch_count"] != 0):
        raise ValueError("Original completed-run or execution-audit binding differs")
    gold_path = root / "data/agent_feedback/holdout.gold.jsonl"
    if sha256(gold_path) != summary["gold_sha256"]:
        raise ValueError("Original gold binding differs")
    paths = [r / (name + ".jsonl") for pair in RUNS.values() for name in pair]
    for path in paths:
        if sha256(path) != complete["outputs_sha256"][str(path.relative_to(root))]:
            raise ValueError(f"Original prediction binding differs: {path}")
    gold = read_jsonl(gold_path)
    if len(gold) != 2000:
        raise ValueError("Expected all 2000 frozen holdout questions")
    report, cases = {}, []
    for seed, names in RUNS.items():
        report[seed], selected = analyze(*(read_jsonl(r / (name + ".jsonl")) for name in names), gold, seed)
        cases.extend(selected)
        for arm, label in (("A", f"A_{seed}"), ("C", f"C_{seed}")):
            original = original_analysis["arms"][label]
            if report[seed]["margins"][arm]["correct"] != original["correct"]:
                raise ValueError("Recomputed correct count differs from original report")
    result = {"schema": "posthoc_holdout_state_transitions_v1", "date": "2026-10-18",
              "analysis_status": "completed", "original_results_modified": False,
              "new_model_calls": 0, "training": False, "posthoc": True,
              "state_order": list(STATES), "seeds": report,
              "case_selection": {"seed": CASE_SEED, "per_nonempty_cell_per_seed": 1,
                  "exclude": "correct->correct", "rule": "minimum sha256(seed|continuation_seed|cell|id)",
                  "selection_blinding": "outcome categories used; hash chosen before reading full selected case contents"},
              "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in
                  [complete_path, summary_path, audit_path, original_analysis_path, gold_path, *paths]},
              "code_sha256": sha256(__file__),
              "limitations": ["Descriptive decomposition, not causal mediation.",
                  "The two continuation seeds share their initialization and training corpus.",
                  "Cases illustrate categories and are not independent validation or training data.",
                  "No semantic error labels inferred solely from reference divergence or an empty result."]}
    output.mkdir(parents=True)
    private = output / "selected_cases.private.jsonl"
    private.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases))
    result["private_cases_sha256"] = sha256(private)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["decomposition"] for key, value in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
