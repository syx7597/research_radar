"""Analysis fixed before opening the project holdout; no model selection.

Pure functions only. The controller must validate all frozen inputs and complete
predictions before loading any answers. The two continuation seeds share the
initial checkpoint and questions: bootstrap QUESTIONS, never seed-question rows.
"""
from collections import Counter
import random

from experiments.condition_consistency.executor import compare_answers
from .compare import compare_predictions, indexed, percentile

SEEDS = (20261003, 20261004)
LABELS = ("A_20261003", "C_20261003", "A_20261004", "C_20261004", "P")
GROUP_NAMES = ("reference_steps_le_5", "reference_steps_gt_5", "qualifier",
               "set_operation", "comparison_numeric_or_count")


def question_groups(reference):
    """Same overlapping descriptive definitions used in the development audit."""
    functions = [step["function"] for step in reference["program"]]
    groups = ["reference_steps_le_5" if len(functions) <= 5 else "reference_steps_gt_5"]
    if any(fn.startswith("QFilter") or fn in {
            "QueryAttrUnderCondition", "QueryAttrQualifier", "QueryRelationQualifier"}
           for fn in functions):
        groups.append("qualifier")
    if any(fn in {"And", "Or"} for fn in functions):
        groups.append("set_operation")
    if any(fn in {"SelectAmong", "SelectBetween", "FilterNum", "FilterYear", "FilterDate",
                  "VerifyNum", "VerifyYear", "VerifyDate", "Count"} for fn in functions):
        groups.append("comparison_numeric_or_count")
    return groups


def seed_average_bootstrap(differences, replicates=5000, seed=20261003):
    """One value per question: mean C-minus-A correctness across BOTH seeds."""
    if (not differences or any(type(x) not in (int, float) or x not in (-1, -.5, 0, .5, 1)
                               for x in differences)):
        raise ValueError("Expected one two-seed average correctness difference per question")
    if type(replicates) is not int or replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates required")
    counts = Counter(differences)
    values = [-1, -.5, 0, .5, 1]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(100 * sum(rng.choices(values, weights=[counts[v] for v in values], k=n)) / n
                     for _ in range(replicates))
    low, high = percentile(samples, .025), percentile(samples, .975)
    return {"accuracy_delta_pp": 100 * sum(differences) / n,
            "accuracy_delta_pp_ci": [low, high], "confidence_level": .95,
            "questions": n, "continuation_seeds": list(SEEDS),
            "replicates": replicates, "bootstrap_seed": seed,
            "method": "paired percentile bootstrap of questions, retaining both seed outcomes together",
            "scope": "conditional on these two continuation checkpoints; shared initial model; no training-seed uncertainty",
            "interval_assessment": "entirely_above_zero" if low > 0 else
                                   "entirely_below_zero" if high < 0 else "includes_zero"}


def compare_fixed(baseline, candidate, gold, replicates):
    report = compare_predictions(baseline, candidate, gold, bootstrap_replicates=replicates)
    # Development investment decisions cannot trigger any new holdout tuning.
    report.pop("pilot_investment_decision")
    report["interpretation"] = (
        "Complete fixed project holdout; estimation only, no training or model selection follows. "
        "Failures remain in the denominator. Question bootstrap conditions on checkpoints.")
    return report


def analyze_predictions(predictions, gold, bootstrap_replicates=5000):
    if set(predictions) != set(LABELS):
        raise ValueError("Both matched continuation seeds and the fixed program baseline are required")
    by_gold = indexed(gold, "gold")
    by_label = {label: indexed(rows, label) for label, rows in predictions.items()}
    if any(set(rows) != set(by_gold) for rows in by_label.values()):
        raise ValueError("Every frozen model must cover all identical holdout questions")
    comparisons = {}
    for seed in SEEDS:
        a, c = f"A_{seed}", f"C_{seed}"
        comparisons[f"C_vs_A_{seed}"] = compare_fixed(predictions[a], predictions[c], gold, bootstrap_replicates)
        comparisons[f"C_vs_P_{seed}"] = compare_fixed(predictions["P"], predictions[c], gold, bootstrap_replicates)
    summaries = {"P": comparisons[f"C_vs_P_{SEEDS[0]}"]["arms"]["baseline"]}
    for seed in SEEDS:
        pair = comparisons[f"C_vs_A_{seed}"]["arms"]
        summaries[f"A_{seed}"] = pair["baseline"]
        summaries[f"C_{seed}"] = pair["candidate"]
    groups = {name: {"questions": 0, "correct": {label: 0 for label in LABELS}}
              for name in GROUP_NAMES}
    differences = []
    for ident, reference in by_gold.items():
        correct = {label: int(compare_answers(reference["answer"], rows[ident]["prediction"]))
                   for label, rows in by_label.items()}
        differences.append(sum(correct[f"C_{seed}"] - correct[f"A_{seed}"] for seed in SEEDS) / len(SEEDS))
        for name in question_groups(reference):
            groups[name]["questions"] += 1
            for label in LABELS:
                groups[name]["correct"][label] += correct[label]
    for group in groups.values():
        n = group["questions"]
        group["accuracy"] = {label: count / n if n else None for label, count in group["correct"].items()}
        group["C_minus_A_pp_by_seed"] = {
            str(seed): 100 * (group["correct"][f"C_{seed}"] - group["correct"][f"A_{seed}"]) / n if n else None
            for seed in SEEDS}
    primary = seed_average_bootstrap(differences, bootstrap_replicates)
    primary["A_mean_accuracy"] = sum(summaries[f"A_{seed}"]["accuracy"] for seed in SEEDS) / len(SEEDS)
    primary["C_mean_accuracy"] = sum(summaries[f"C_{seed}"]["accuracy"] for seed in SEEDS) / len(SEEDS)
    primary["both_seed_point_estimates_positive"] = all(
        comparisons[f"C_vs_A_{seed}"]["accuracy_delta_pp"] > 0 for seed in SEEDS)
    primary["practical_gain_at_least_2pp"] = primary["accuracy_delta_pp"] >= 2 - 1e-12
    tails = {}
    for label, rows in predictions.items():
        tails[label] = {}
        for key in ("calls", "input_tokens", "generated_tokens", "total_tokens", "invalid_calls"):
            values = sorted(row["input_tokens"] + row["generated_tokens"] if key == "total_tokens" else row[key]
                            for row in rows)
            tails[label][key] = {"p50": percentile(values, .5), "p95": percentile(values, .95), "max": values[-1]}
    return {"evaluation_split": "project_holdout_not_official_hidden_test", "questions": len(gold),
            "primary": primary, "arms": summaries, "paired_comparisons": comparisons,
            "descriptive_overlapping_groups": groups, "cost_distribution": tails,
            "group_inference": "descriptive only; no subgroup winner selection or multiplicity-adjusted claims",
            "program_comparison": "same backbone but different representation/training/inference cost; not the causal recovery contrast",
            "post_evaluation_policy": "report all fixed models and outcomes; no further tuning or selection on this split",
            "limitations": ["Both continuation seeds share the initial model and corpus.",
                            "2000 questions evaluated twice remain 2000 question units, not 4000 independent samples.",
                            "Project split excludes historical exact question/program components; not strict compositional generalization.",
                            "Public benchmark pretraining contamination cannot be excluded.",
                            "Answer matching does not prove semantic condition preservation.",
                            "Full-prefill plus generated token counts are not GPU FLOPs or wall-clock speed."]}
