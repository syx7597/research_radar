"""Second bounded development round: keep a schema-clean first-valid query.

The guard was motivated after inspecting the v1 2,000-question holdout. That
set is now a seen diagnostic set. This module never modifies v1 code/caches;
thresholds use the same training-internal 500-question calibration and 73-grid.
The complete official validation split is evaluated only after policy freeze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import sha256, write_json
from .evaluate import paired_comparison
from . import evaluate_repairs as v1
from .executor import compare_answers

GUARD = "keep_first_valid_if_schema_clean_is_true"
SPLIT_ROLES = {
    "dev_diagnostic": "previously inspected development diagnostic; not independent validation",
    "holdout_seen_diagnostic": "v1 2000-question holdout inspected before v2 guard design; seen diagnostic, not independent validation",
    "official_val": "complete official public validation split; v2 policy frozen before this evaluation; earlier project work accessed official val",
}


def guard_applies(row: dict) -> bool:
    index = v1.baseline_index(row)
    return index is not None and row["candidates"][index].get("schema_clean") is True


def choose_repair(row: dict, policy: dict) -> int | None:
    """Gold-free guard; otherwise delegate unchanged to the frozen v1 selector."""
    if guard_applies(row):
        return v1.baseline_index(row)
    return v1.choose_repair(row, policy)


def calibration_grid() -> list[dict]:
    return [dict(disabled=True, alpha=0.0, min_match=0.0, margin=0.0)] + [
        dict(disabled=False, alpha=alpha, min_match=minimum, margin=margin)
        for alpha in (0.0, 0.25, 0.5)
        for minimum in (0.0, 0.4, 0.6, 0.8)
        for margin in (-0.25, 0.0, 0.1, 0.25, 0.5, 1.0)]


def code_identity() -> dict:
    here = Path(__file__)
    return {"selector_sha256": sha256(here),
            "v1_selector_sha256": sha256(here.with_name("evaluate_repairs.py")),
            "repair_sha256": sha256(here.with_name("repair.py")),
            "scorer_sha256": sha256(here.with_name("score_repairs.py")),
            "adapter_sha256": sha256(here.with_name("executor.py"))}


def verify_originals(caches: dict) -> None:
    fields = ("program_text", "program", "prediction", "valid", "answers", "empty_result",
              "error", "schema_clean", "schema_issues")
    for qid, row in caches["global"].items():
        other = caches["local"][qid]
        left = row["candidates"][:4]
        right = other["candidates"][:4]
        if [[candidate.get(key) for key in fields] for candidate in left] != [
                [candidate.get(key) for key in fields] for candidate in right]:
            raise ValueError(f"Global/local originals disagree: {qid}")
        if any(type(candidate.get("schema_clean")) is not bool for candidate in left + right):
            raise ValueError("Guard requires explicit boolean schema checks on every original")


def verify_val_manifest(args) -> dict:
    manifest = json.loads(args.val_manifest.read_text())
    original = json.loads(args.manifest.read_text())
    expected = manifest["splits"]["official_val"]
    if (manifest["source_sha256"] != original["source_sha256"]
            or expected["count"] != 11797
            or len(expected["ids"]) != 11797
            or set(expected["ids"]) != {f"val:{index}" for index in range(11797)}):
        raise ValueError("Official validation manifest must declare the complete original 11797 questions")
    return manifest


def verify_v1_policy(args, identity: dict) -> dict:
    policy = json.loads(args.v1_policy.read_text())
    sources = code_identity()
    if (policy["selector_sha256"] != sources["v1_selector_sha256"]
            or policy["repair_sha256"] != sources["repair_sha256"]
            or policy["scorer_sha256"] != sources["scorer_sha256"]
            or policy["cache_identity"] != identity
            or policy["checkpoint_manifest_sha256"] != sha256(args.checkpoint_manifest)
            or policy["split_manifest_sha256"] != sha256(args.manifest)):
        raise ValueError("Frozen v1 policy/code/data/model identity changed")
    return policy


def calibrate(args) -> dict:
    gold, caches = v1.load_aligned({"global": args.global_cache, "local": args.local_cache}, args.gold)
    v1.verify_split(args, gold, "calibration")
    identity = v1.verify_cache_provenance(args)
    old_policy = verify_v1_policy(args, identity)
    verify_val_manifest(args)
    verify_originals(caches)
    if set(gold) != set(old_policy["calibration_ids"]):
        raise ValueError("v2 must reuse exactly the v1 calibration questions")
    config = {
        "version": "query_repair_v2", "stage": "frozen_on_training_internal_calibration",
        "guard": GUARD, "code_identity": code_identity(),
        "v1_policy_sha256": sha256(args.v1_policy), "v1_policies": old_policy["policies"],
        "split_manifest_sha256": sha256(args.manifest),
        "val_manifest_sha256": sha256(args.val_manifest),
        "checkpoint_manifest_sha256": sha256(args.checkpoint_manifest),
        "cache_identity": identity, "calibration_ids": sorted(gold),
        "calibration_gold_sha256": sha256(args.gold),
        "development_history": {
            "guard_design": "After inspecting v1 holdout errors; this is the second bounded development round",
            "threshold_selection": "Only original training-internal calibration500; unchanged 73-policy grid",
            "holdout_status": SPLIT_ROLES["holdout_seen_diagnostic"],
            "official_val_status": SPLIT_ROLES["official_val"],
            "candidate_generation": "Unchanged v1 candidates, repair algorithms, executor and teacher-forced scores"},
        "tie_break": old_policy["tie_break"], "policies": {}, "calibration": {}}
    for mode, rows in caches.items():
        base = {qid: v1.outcome(row, v1.baseline_index(row), gold[qid]["answer"])
                for qid, row in rows.items()}
        trials = []
        for policy in calibration_grid():
            correct = corrected = regressed = switches = 0
            for qid, row in rows.items():
                pick = choose_repair(row, policy)
                ok = v1.outcome(row, pick, gold[qid]["answer"])
                correct += ok
                corrected += ok and not base[qid]
                regressed += base[qid] and not ok
                switches += pick != v1.baseline_index(row)
            trials.append(dict(policy=policy, correct=correct, corrected=corrected,
                               regressed=regressed, switches=switches))
        best = max(trials, key=lambda trial: (trial["correct"], -trial["regressed"],
                                             -trial["switches"], -trial["policy"]["alpha"]))
        config["policies"][mode] = best["policy"]
        config["calibration"][mode] = dict(
            questions=len(rows), baseline_correct=sum(base.values()),
            guard_retained=sum(guard_applies(row) for row in rows.values()),
            selected=best, trials=trials, input_sha256=sha256(getattr(args, mode + "_cache")))
    write_json(args.output, config)
    print(json.dumps({mode: item["selected"] for mode, item in config["calibration"].items()}, indent=2))
    return config


def verify_evaluation_split(args, gold: dict, policy: dict) -> None:
    if args.split not in SPLIT_ROLES:
        raise ValueError("v2 does not expose the already inspected holdout as an independent split")
    if args.split == "official_val":
        manifest = verify_val_manifest(args)
        expected = manifest["splits"]["official_val"]
        if (sha256(args.gold) != expected["gold"]["sha256"]
                or set(gold) != set(expected["ids"]) or len(gold) != 11797):
            raise ValueError("Official validation gold does not match the complete frozen manifest")
    else:
        v1.verify_split(args, gold, "holdout" if args.split == "holdout_seen_diagnostic" else args.split)
    if set(gold).intersection(policy["calibration_ids"]):
        raise ValueError("Evaluation overlaps calibration IDs")


def evaluate(args) -> dict:
    paths = {"global": args.global_cache, "local": args.local_cache, "beam8": args.beam8_cache}
    if args.beam8_cache is None:
        raise ValueError("v2 evaluation requires the beam8 control")
    gold, caches = v1.load_aligned(paths, args.gold)
    identity = v1.verify_cache_provenance(args)
    old_policy = verify_v1_policy(args, identity)
    config = json.loads(args.policy.read_text())
    if (config["version"] != "query_repair_v2" or config["guard"] != GUARD
            or config["code_identity"] != code_identity()
            or config["cache_identity"] != identity
            or config["v1_policy_sha256"] != sha256(args.v1_policy)
            or config["v1_policies"] != old_policy["policies"]
            or config["split_manifest_sha256"] != sha256(args.manifest)
            or config["val_manifest_sha256"] != sha256(args.val_manifest)
            or config["checkpoint_manifest_sha256"] != sha256(args.checkpoint_manifest)):
        raise ValueError("v2 method/policy/model/data changed after freeze")
    verify_evaluation_split(args, gold, config)
    verify_originals(caches)
    beam_meta = json.loads(args.beam8_cache.with_suffix(".jsonl.meta.json").read_text())
    if (beam_meta["output_sha256"] != sha256(args.beam8_cache)
            or beam_meta.get("python_hash_seed") != "20260926"
            or beam_meta.get("adapter_source_sha256") != code_identity()["adapter_sha256"]
            or beam_meta["kb_sha256"] != json.loads(args.manifest.read_text())["source_sha256"]["kb.json"]):
        raise ValueError("Beam8 execution provenance/hash seed mismatch")
    correctness, details = {}, []
    for qid, target in gold.items():
        row = caches["global"][qid]
        picks = {"top1": (row, 0), "first_valid": (row, v1.baseline_index(row)),
                 "beam8_first_valid": (caches["beam8"][qid], v1.baseline_index(caches["beam8"][qid]))}
        for mode in ("global", "local"):
            r = caches[mode][qid]
            picks["v1_" + mode + "_repair"] = (r, v1.choose_repair(r, old_policy["policies"][mode]))
            picks["v2_" + mode + "_repair"] = (r, choose_repair(r, config["policies"][mode]))
        correct = {method: v1.outcome(r, index, target["answer"])
                   for method, (r, index) in picks.items()}
        for mode in ("global", "local"):
            correct[mode + "_oracle"] = any(candidate["valid"] and compare_answers(target["answer"], candidate["prediction"])
                                              for candidate in caches[mode][qid]["candidates"])
        for method, ok in correct.items():
            correctness.setdefault(method, []).append(ok)
        details.append({"id": qid, "correct": correct,
                        "selected": {method: index for method, (_, index) in picks.items()},
                        "guard_retained": guard_applies(row),
                        "repairs": {mode: sum(c.get("origin") == "repair" for c in caches[mode][qid]["candidates"])
                                    for mode in ("global", "local")}})
    n = len(gold)
    report = {
        "version": "query_repair_v2", "split": args.split, "split_role": SPLIT_ROLES[args.split],
        "questions": n, "gold_sha256": sha256(args.gold), "policy_sha256": sha256(args.policy),
        "v1_policy_sha256": sha256(args.v1_policy), "code_identity": code_identity(),
        "input_sha256": {mode: sha256(path) for mode, path in paths.items()},
        "metrics": {method: dict(correct=sum(values), questions=n, accuracy=sum(values) / n)
                    for method, values in correctness.items()},
        "comparisons": {}, "per_question": details,
        "guard": {"rule": GUARD, "retained_questions": sum(item["guard_retained"] for item in details),
                  "blocked_questions_may_include_incorrect_originals": True},
        "coverage": {mode: dict(questions_with_repairs=sum(item["repairs"][mode] > 0 for item in details),
                                new_candidates=sum(item["repairs"][mode] for item in details))
                     for mode in ("global", "local")}}
    comparisons = [(method, "first_valid") for method in (
        "beam8_first_valid", "v1_global_repair", "v1_local_repair", "v2_global_repair", "v2_local_repair")]
    comparisons += [("v2_" + mode + "_repair", "v1_" + mode + "_repair") for mode in ("global", "local")]
    comparisons += [("v2_local_repair", "v2_global_repair"), ("v2_local_repair", "beam8_first_valid"),
                    ("v2_global_repair", "beam8_first_valid"), ("v1_local_repair", "beam8_first_valid"),
                    ("v1_local_repair", "v1_global_repair")]
    for method, baseline in comparisons:
        label = method + "_vs_" + baseline
        report["comparisons"][label] = paired_comparison(correctness[baseline], correctness[method], seed=20260926)
        print(f"Computed paired comparison: {label}", flush=True)
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "evaluate"))
    for name in ("global-cache", "local-cache", "gold", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--beam8-cache", type=Path)
    root = Path("results/condition_consistency/query_repair")
    parser.add_argument("--v1-policy", type=Path, default=root / "frozen_policy.json")
    parser.add_argument("--manifest", type=Path, default=root / "split_manifest.json")
    parser.add_argument("--val-manifest", type=Path, default=root / "v2/official_val_manifest.json")
    parser.add_argument("--checkpoint-manifest", type=Path,
                        default=Path("results/condition_consistency/pilot_bart5k_seed20260926/checkpoint_manifest.json"))
    parser.add_argument("--split", choices=tuple(SPLIT_ROLES), default="holdout_seen_diagnostic")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen policy or report")
    if args.command == "evaluate" and (args.policy is None or args.beam8_cache is None):
        parser.error("evaluate requires --policy and --beam8-cache")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (calibrate if args.command == "calibrate" else evaluate)(args)


if __name__ == "__main__":
    main()
