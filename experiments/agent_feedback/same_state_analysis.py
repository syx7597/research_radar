"""Analyze all four fixed resumes; never score a partial or selected cohort.

The CLI requires the frozen execution binding and terminal runtime records,
replays every continuation on the original Linux environment, then opens gold.
Token costs are validated recorded counters, not independently retokenized costs.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

from experiments.condition_consistency.executor import KoPLExecutor, compare_answers
from .compare import indexed, percentile, validate_prediction
from .environment import compact
from .inference import execute_response
from .recovery import sha256
from .same_state import CONDITIONS, restore, write_new
from .same_state_inference import ARMS, validate_inventory
from .training_data import read_jsonl

SEEDS = (20261003, 20261004)
COSTS = ("continuation_calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens")


def validate_resumes(executor, states, predictions, *, probe=False):
    """Check complete ordered coverage and actual execution, including failures."""
    validate_inventory(states)
    required = {"A_20261003"} if probe else set(ARMS)
    if set(predictions) != required:
        raise ValueError("All registered checkpoints for this evaluation or explicit old-dev probe are required")
    keys = [(r["id"], r["condition"]) for r in states]
    for arm, rows in predictions.items():
        if [(r["id"], r["condition"]) for r in rows] != keys:
            raise ValueError("Missing, duplicate, extra or reordered resume states")
        for state, row in zip(states, rows):
            validate_prediction(row, arm)
            if (row.get("arm") != arm or row.get("initial_state_sha256") != state["state_sha256"]
                    or row.get("prefix_calls") != state["prefix_calls"]
                    or row.get("remaining_calls") != state["remaining_calls"]):
                raise ValueError("Resume does not bind to the registered starting state")
            n = row.get("continuation_calls")
            if (type(n) is not int or not 0 <= n <= state["remaining_calls"]
                    or row["calls"] != state["prefix_calls"] + n
                    or row["generated_tokens"] > state["remaining_generated_tokens"]):
                raise ValueError("Continuation exceeds its frozen budget or call accounting differs")
            start = len(state["messages"])
            if (row["messages"][:start] != state["messages"]
                    or len(row["messages"]) != start + 2 * n
                    or row["events"][:state["prefix_calls"]] != state["events"]
                    or len(row["events"]) != row["calls"]):
                raise ValueError("Recorded history differs from the frozen prefix or turn count")
            episode = restore(executor, state)
            for i in range(n):
                assistant, tool = row["messages"][start + 2*i:start + 2*i + 2]
                if episode.done or set(assistant) != {"role", "content"} or assistant["role"] != "assistant":
                    raise ValueError("Malformed continuation or action after termination")
                actual = execute_response(episode, assistant["content"])
                if tool != {"role": "tool", "content": compact(actual)}:
                    raise ValueError("Continuation tool observation differs on replay")
            if episode.events != row["events"] or episode.prediction != row["prediction"]:
                raise ValueError("Continuation events or final prediction differ on replay")
            invalid = sum(not e["observation"]["ok"] for e in episode.events[state["prefix_calls"]:])
            if invalid != row["invalid_calls"]:
                raise ValueError("Invalid-call cost must count continuation only")
            reason = row["stop_reason"]
            if reason not in {"finished", "call_budget", "context_budget", "generation_budget"}:
                raise ValueError("Unregistered termination reason")
            if reason == "call_budget" and n != state["remaining_calls"]:
                raise ValueError("Call-budget failure has unused calls")
            if reason == "generation_budget" and row["generated_tokens"] != 2048:
                raise ValueError("Generation-budget failure has unused generation tokens")


def contrasts(question_rows):
    """Jointly bootstrap question vectors, retaining seeds AND conditions."""
    if not question_rows:
        raise ValueError("No eligible states; recovery contrast is not estimable")
    vectors = []
    for row in question_rows:
        clean, divergent = [sum(row["correct"][f"C_{s}"][c] - row["correct"][f"A_{s}"][c]
                                  for s in SEEDS) / 2 for c in CONDITIONS]
        vectors.append((divergent, clean, divergent - clean))
    counts = Counter(vectors)
    values = sorted(counts)
    rng = random.Random(20261018)
    samples = [[], [], []]
    n = len(vectors)
    for _ in range(5000):
        draw = rng.choices(values, weights=[counts[v] for v in values], k=n)
        for j in range(3):
            samples[j].append(100 * sum(v[j] for v in draw) / n)
    result = {}
    for j, name in enumerate(("divergent_C_minus_A", "clean_C_minus_A", "difference_of_differences")):
        low, high = [percentile(sorted(samples[j]), q) for q in (.025, .975)]
        result[name] = {"delta_pp": 100 * sum(v[j] for v in vectors) / n,
                        "ci95_pp": [low, high], "questions": n,
                        "interval_assessment": "entirely_above_zero" if low > 0 else
                        "entirely_below_zero" if high < 0 else "includes_zero"}
    return result


def summarize(predictions, gold, states):
    """Only call after validate_resumes; returns aggregates plus private rows."""
    refs = indexed(gold, "gold")
    ids = [s["id"] for s in states[::2]]
    if not ids or not set(ids) <= set(refs):
        raise ValueError("Frozen cohort must be nonempty and covered by reference answers")
    by_arm = {a: {(r["id"], r["condition"]): r for r in rows} for a, rows in predictions.items()}
    questions = []
    for clean, divergent in zip(states[::2], states[1::2]):
        ident = clean["id"]
        observation = divergent["events"][-1]["observation"]
        questions.append({"id": ident,
            "strata": ["executable" if observation["ok"] else "rejected"] +
                      (["empty_entity"] if observation["ok"] and observation.get("count") == 0 else []),
            "correct": {a: {c: int(compare_answers(refs[ident]["answer"], rows[ident, c]["prediction"]))
                            for c in CONDITIONS} for a, rows in by_arm.items()}})
    arms = {}
    for arm, rows in by_arm.items():
        arms[arm] = {}
        for condition in CONDITIONS:
            data = [rows[i, condition] for i in ids]
            arms[arm][condition] = {
                "questions": len(ids), "correct": sum(q["correct"][arm][condition] for q in questions),
                "finished": sum(r["prediction"] is not None for r in data),
                "stop_reasons": dict(Counter(r["stop_reason"] for r in data)),
                "costs": {k: {"total": sum(r[k] for r in data), "mean": sum(r[k] for r in data)/len(ids),
                              "p95": percentile(sorted(r[k] for r in data), .95)} for k in COSTS}}
    paired = {}
    for seed in SEEDS:
        paired[str(seed)] = {}
        for condition in CONDITIONS:
            pairs = Counter((q["correct"][f"A_{seed}"][condition], q["correct"][f"C_{seed}"][condition])
                            for q in questions)
            paired[str(seed)][condition] = {"both_correct": pairs[1, 1], "both_wrong": pairs[0, 0],
                "corrected": pairs[0, 1], "regressed": pairs[1, 0],
                "delta_pp": 100 * (pairs[0, 1] - pairs[1, 0]) / len(ids)}
    strata = {}
    for name in ("executable", "rejected", "empty_entity"):
        subset = [q for q in questions if name in q["strata"]]
        strata[name] = {"questions": len(subset), "correct_counts": {
            a: {c: sum(q["correct"][a][c] for q in subset) for c in CONDITIONS} for a in ARMS}}
    return {"schema": "same_state_analysis_v1", "questions": len(ids), "contrasts": contrasts(questions),
        "arms": arms, "per_seed_paired_outcomes": paired, "descriptive_overlapping_strata": strata,
        "bootstrap": {"replicates": 5000, "seed": 20261018, "unit": "question",
                      "retains": "both seeds and both conditions jointly", "confidence_level": .95},
        "limitations": ["Conditional on fixed checkpoints and constructible failed-I states, not all errors.",
                        "Intervals do not estimate training-seed uncertainty or correct multiple comparisons.",
                        "Empty entities and first reference divergence are not semantic error annotations.",
                        "Recorded token counters are not FLOPs or speed; continuation replay verifies outcomes.",
                        "Neither contrast alone establishes semantic error recognition."],
        "new_training_authorized_by_analysis": False}, questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10) or os.environ.get("PYTHONHASHSEED") != "20261003":
        raise ValueError("Replay on original Linux Python3.10 with PYTHONHASHSEED=20261003")
    if args.output_dir.exists():
        raise FileExistsError("Never overwrite or silently rescore existing analysis")
    binding = json.loads(args.binding.read_text())
    if (binding.get("schema") != "same_state_execution_binding_v1" or binding.get("ready") is not True
            or binding.get("states_sha256") != sha256(args.states)
            or binding.get("protocol_sha256") != sha256(binding["protocol_path"])
            or binding.get("state_replay_passed") is not True or binding.get("token_budget_passed") is not True):
        raise ValueError("Frozen execution binding is missing or differs")
    # This analysis must itself be frozen BEFORE inference, not selected after results.
    code_path = "experiments/agent_feedback/same_state_analysis.py"
    if binding["inputs_sha256"].get(code_path) != sha256(__file__):
        raise ValueError("Analysis implementation was not frozen in the execution binding")
    for path, expected in binding["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    protocol = json.loads(Path(binding["protocol_path"]).read_text())
    manifest_path = Path("results/agent_feedback/same_state_v1/pool_manifest.json")
    if sha256(manifest_path) != protocol["pool"]["manifest_sha256"]:
        raise ValueError("Question pool manifest changed")
    pool = json.loads(manifest_path.read_text())
    paths = [args.binding, args.states, Path(binding["protocol_path"]), manifest_path]
    predictions = {}
    for arm in ARMS:
        path = args.predictions_dir / f"{arm}.jsonl"
        runtime_path = path.with_suffix(".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        if (runtime.get("status") != "completed" or runtime.get("arm") != arm
                or runtime.get("binding_sha256") != sha256(args.binding)
                or runtime.get("predictions_sha256") != sha256(path)
                or runtime.get("versions") != binding["versions"]):
            raise ValueError("Partial, unbound or changed prediction run")
        predictions[arm] = read_jsonl(path)
        if runtime["states"] != len(predictions[arm]):
            raise ValueError("Runtime count differs from actual predictions")
        paths.extend([path, runtime_path])
    states = read_jsonl(args.states)
    validate_resumes(KoPLExecutor("datasets/kqa_pro/kb.json"), states, predictions)
    gold_path = Path("data/agent_feedback/same_state_v1/pool.gold.jsonl")
    if pool["artifacts"].get(str(gold_path)) != sha256(gold_path):
        raise ValueError("Reference answers differ from the frozen pool")
    gold = read_jsonl(gold_path)
    if [r["id"] for r in gold] != pool["ids"]:
        raise ValueError("Reference coverage differs from the frozen pool")
    report, rows = summarize(predictions, gold, states)
    report["inputs_sha256"] = {str(p): sha256(p) for p in paths + [gold_path, Path(__file__)]}
    report["continuation_replay_passed"] = True
    write_new(args.output_dir / "question_results.private.jsonl", rows, True)
    write_new(args.output_dir / "summary.json", report)
    print(json.dumps(report["contrasts"], indent=2))


if __name__ == "__main__":
    main()
