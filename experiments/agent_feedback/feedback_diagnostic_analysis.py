"""Full-dev CPU audit and descriptive error-diagnostic masking comparison.

Original events are replayed against the real KB; model-visible messages are
checked against the declared renderer separately. Never label empty output an
error, pool seeds, open holdout or decide whether to train another method.
"""
from copy import deepcopy
from collections import Counter
import argparse
import datetime
import json
import os
from pathlib import Path

from .compare import (compare_predictions, digest, indexed, validate_dev_path,
                      validate_frozen_dev, validate_prediction)
from .environment import MAX_CALLS, compact
from .feedback_mask import MODE, SEED, visible_observation
from .inference import validate_questions
from .recovery import Excluded, replay_rollout
from .training_data import read_jsonl
from experiments.condition_consistency.executor import KoPLExecutor, compare_answers


def first_error_index(row):
    return next((i for i, e in enumerate(row["events"]) if e["observation"]["ok"] is False), None)


def audit_visible_rollout(executor, row, question, mode):
    """Reconstruct only the audit copy of raw tool messages, then replay fully."""
    validate_prediction(row, mode)
    messages, events = row["messages"], row["events"]
    if len(messages) != 2 + 2 * len(events):
        raise ValueError("Visible message/event alignment differs")
    raw = deepcopy(row)
    for index, event in enumerate(events):
        message = messages[3 + 2 * index]
        expected = compact(visible_observation(event["observation"], mode))
        if message != {"role": "tool", "content": expected}:
            raise ValueError("Model-visible observation differs from the declared rendering intervention")
        raw["messages"][3 + 2 * index] = {"role": "tool", "content": compact(event["observation"])}
    _, episode = replay_rollout(executor, raw, question, MAX_CALLS)
    if (episode.selected != row["selected_handle"] or episode.calls != row["calls"]
            or sum(not event["observation"]["ok"] for event in events) != row["invalid_calls"]):
        raise ValueError("Real selected handle or call accounting differs")


def prefix_before_intervention_equal(normal, masked):
    """The first failure action/result is common; only its rendered feedback may differ."""
    first = first_error_index(normal)
    if first is None:
        return (normal["messages"] == masked["messages"] and normal["events"] == masked["events"]
                and normal["prediction"] == masked["prediction"] and normal["stop_reason"] == masked["stop_reason"])
    if len(masked["events"]) <= first:
        return False
    # Slice stops after the assistant action and before the first failure observation.
    return (normal["messages"][:3 + 2 * first] == masked["messages"][:3 + 2 * first]
            and normal["events"][:first + 1] == masked["events"][:first + 1])


def validate_diagnostic_predictions(normal, masked, questions, *, executor):
    """Gold-free full supplied-ID audit; non-exposed numerical drift is reported separately."""
    validate_questions(questions)
    by_question = indexed(questions, "questions")
    arms = {"normal": indexed(normal, "normal"), "masked": indexed(masked, "masked")}
    if any(set(rows) != set(by_question) for rows in arms.values()):
        raise ValueError("Both diagnostic arms must cover the entire supplied question ID set")
    mismatches = []
    for arm, rows in arms.items():
        for key, row in rows.items():
            try:
                audit_visible_rollout(executor, row, by_question[key], "original" if arm == "normal" else MODE)
            except (Excluded, ValueError, KeyError, TypeError, IndexError) as exc:
                mismatches.append({"arm": arm, "id": key, "reason": getattr(exc, "reason", str(exc)),
                                   "detail": getattr(exc, "detail", None)})
    exposed = {arm: [key for key, row in rows.items() if first_error_index(row) is not None]
               for arm, rows in arms.items()}
    normal_exposed = set(exposed["normal"])
    prediv = [key for key in by_question if not prefix_before_intervention_equal(arms["normal"][key], arms["masked"][key])]
    negative = [key for key in prediv if key not in normal_exposed]
    return {"audit_passed": not mismatches, "mismatches": mismatches,
            "coverage": {"questions": len(by_question), "normal": len(normal), "masked": len(masked),
                         "trajectories_replay_verified": len(normal) + len(masked) - len(mismatches)},
            "normal_error_exposed_ids": exposed["normal"], "masked_error_exposed_ids": exposed["masked"],
            "newly_error_exposed_ids": [key for key in exposed["masked"] if key not in normal_exposed],
            "pre_intervention_divergence_ids": prediv, "unexposed_negative_control_changed_ids": negative,
            "pre_intervention_equivalence_passed": not prediv,
            "interpretation_gate": "needs_numerical_batching_review" if prediv else "conditional_on_this_fixed_checkpoint_and_renderer",
            "negative_control_note": "Original no-error trajectories receive no intended intervention. Changes there or before the first masked observation require a batching/numerical reproducibility explanation, not an error-feedback mechanism claim."}


def exposure_analysis(normal, masked, gold):
    left, right, refs = indexed(normal, "normal"), indexed(masked, "masked"), indexed(gold, "gold")
    if set(left) != set(right) or set(left) != set(refs):
        raise ValueError("Exposure analysis requires complete matching question IDs")
    counts = Counter()
    cohorts = {"normal_error_exposed": Counter(), "normal_error_unexposed": Counter()}
    for key, ref in refs.items():
        before, after = first_error_index(left[key]) is not None, first_error_index(right[key]) is not None
        nc = bool(compare_answers(ref["answer"], left[key]["prediction"]))
        mc = bool(compare_answers(ref["answer"], right[key]["prediction"]))
        counts["normal_error_exposed_questions"] += before
        counts["normal_correct_and_error_exposed_questions"] += before and nc
        counts["masked_error_exposed_questions"] += after
        counts["masked_correct_and_error_exposed_questions"] += after and mc
        counts["newly_error_exposed_questions"] += after and not before
        group = cohorts["normal_error_exposed" if before else "normal_error_unexposed"]
        group.update(questions=1, normal_correct=int(nc), masked_correct=int(mc),
                     corrected=int(mc and not nc), regressed=int(nc and not mc), net_corrected=int(mc) - int(nc))
    n = len(refs)
    return {**dict(counts), "questions": n, "fixed_normal_exposure_cohorts": {k: dict(v) for k, v in cohorts.items()},
            "direct_intervention_scope_before_any_numerical_drift": {
                "maximum_questions_exposed": counts["normal_error_exposed_questions"],
                "fraction_of_complete_dev": counts["normal_error_exposed_questions"] / n,
                "maximum_accuracy_loss_pp_from_original_correct_exposed_questions":
                    100 * counts["normal_correct_and_error_exposed_questions"] / n,
                "note": "Logical support bound if the no-error runs and pre-intervention prefixes reproduce; not a bound after unexplained numerical/batching drift."},
            "interpretation": "Exposure cohorts are fixed by the original trajectory, not selected by masked outcomes. New error exposure is separate. Correct answers can omit conditions; these are benchmark labels only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--masked", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--questions", default="data/agent_feedback/dev.questions.jsonl")
    parser.add_argument("--gold", default="data/agent_feedback/dev.gold.jsonl")
    parser.add_argument("--split-manifest", default="results/agent_feedback/split_manifest.json")
    parser.add_argument("--protocol", default="results/agent_feedback/protocol.json")
    parser.add_argument("--kb", default="datasets/kqa_pro/kb.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostic analysis exists; never overwrite")
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError("Replay requires PYTHONHASHSEED=20261003")
    if any("holdout" in value.lower() or "held_out" in value.lower() for value in
           (args.normal, args.masked, args.questions, args.gold)):
        raise ValueError("Holdout files are outside this diagnostic")
    split = json.loads(Path(args.split_manifest).read_text())
    validate_dev_path(args.gold, split)
    spec = split["splits"]["dev"]
    if (Path(args.questions).resolve() != Path(spec["questions"]["path"]).resolve()
            or digest(args.questions) != spec["questions"]["sha256"]):
        raise ValueError("Questions differ from the frozen development split")
    gold, questions = read_jsonl(args.gold), read_jsonl(args.questions)
    validate_frozen_dev(gold, args.gold, split)
    if len(gold) != 500 or len(questions) != 500:
        raise ValueError("Complete 500-question diagnostic required")
    normal, masked = read_jsonl(args.normal), read_jsonl(args.masked)
    audit = validate_diagnostic_predictions(normal, masked, questions, executor=KoPLExecutor(args.kb))
    if not audit["audit_passed"]:
        raise ValueError(f"Real/visible trajectory audit failed: {audit['mismatches'][:5]}")
    protocol = json.loads(Path(args.protocol).read_text())
    comparison = compare_predictions(normal, masked, gold, go_rule=protocol["go_rule"])
    comparison.pop("pilot_investment_decision", None)
    comparison["interpretation"] = (
        "Descriptive same-checkpoint feedback diagnostic only; neither a score drop nor no drop "
        "decides main-method success, further investment or additional training."
    )
    pair_path = Path("results/agent_feedback/recovery_pairs.json")
    pair_audit_path = Path("results/agent_feedback/recovery_pair_audit.json")
    pairs, pair_audit = json.loads(pair_path.read_text()), json.loads(pair_audit_path.read_text())
    source_paths = [Path(args.normal), Path(args.masked), Path(args.questions), Path(args.gold),
                    Path(args.split_manifest), Path(args.protocol), Path(args.kb), pair_path, pair_audit_path,
                    Path("experiments/agent_feedback/feedback_mask.py"), Path(__file__),
                    Path(__file__).with_name("recovery.py"), Path(__file__).with_name("environment.py"),
                    Path(__file__).with_name("inference.py"), Path(__file__).with_name("compare.py")]
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Full fixed development, same-checkpoint generic error-diagnostic rendering intervention; no training or holdout",
              "feedback_mode": MODE, "comparison": comparison, "audit": audit,
              "coverage": exposure_analysis(normal, masked, gold),
              "recovery_training_error_coverage": {
                  "unique_pairs": pairs["counts"]["accepted_pairs"],
                  "rejected_divergence_actions": pairs["accepted_divergence_execution"]["rejected"],
                  "executable_divergence_actions": pairs["accepted_divergence_execution"]["executable"],
                  "valid_empty_divergence_results": pair_audit["observed_pair_composition"]["valid_empty_entity_results_at_divergence"],
                  "interpretation": "Only 4 of 418 recovery pairs contain a rejected divergence action; 414 are executable and 198 empty results are valid, not error labels."},
              "decision": {"method_success_or_failure_gate": False, "allow_automatic_training": False,
                           "allow_automatic_model_revision": False, "allow_holdout_evaluation": False,
                           "interpretation": "A limited mechanism probe; neither a score drop nor no drop is a gate for the main recovery-SFT method."},
              "limitations": ["Only failure error/detail strings are masked; success flags and all successful values/schema remain visible.",
                              "Effects concern this fixed renderer and checkpoint, not all feedback use or the cause of the SFT gain.",
                              "Generic strings change prompt length and introduce a distribution shift; no token-length-matched control was run.",
                              "Greedy outputs can depend on floating-point batching; pre-intervention and no-error controls are reported.",
                              "The same 500 questions across two continuation seeds are not 1000 independent test questions.",
                              "Question bootstrap does not measure seed uncertainty; answer equality does not prove semantic correctness."],
              "inputs_sha256": {str(p): digest(p) for p in source_paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "audit_passed": audit["audit_passed"],
                      "pre_intervention_equivalence_passed": audit["pre_intervention_equivalence_passed"],
                      "accuracy_delta_pp": comparison["accuracy_delta_pp"], "coverage": report["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
