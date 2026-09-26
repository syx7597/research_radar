"""Gold-free, deliberately small condition-consistency ranking baselines.

No feature reads the reference answer/program, question category, or candidate
correctness. Rules are fixed before observing the held-out diagnosis. A trained
linear model is an optional probe, not a replacement for a strong neural baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re


QUALIFIERS = {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate",
              "QueryAttrQualifier", "QueryRelationQualifier", "QueryAttrUnderCondition"}
COMPARISONS = {"FilterNum", "FilterYear", "FilterDate", "QFilterNum", "QFilterYear",
               "QFilterDate", "VerifyNum", "VerifyYear", "VerifyDate"}
STOP_WORDS = set("a an the what which who whose is are was were of to in on at by for with and or has have had does do did that it its".split())
FEATURES = ["bias", "number_coverage", "number_precision", "missing_number_fraction",
            "extra_number_fraction", "lexical_coverage", "lexical_precision",
            "comparison_match", "comparison_mismatch", "year_with_condition",
            "year_without_condition", "condition_key_overlap", "condition_count",
            "program_length", "valid", "empty_prediction"]


def empty_prediction(candidate: dict) -> bool:
    # Official execution serializes [] as the string "None". Prefer the raw
    # answer-list observation so an actual entity named "None" is not dropped.
    if "empty_result" in candidate:
        return bool(candidate["empty_result"])
    if "answers" in candidate:
        return candidate["answers"] == []
    return candidate.get("prediction") in (None, "", [], "None")


def output_kind(candidate: dict) -> str:
    if empty_prediction(candidate):
        return "empty"
    value = str(candidate.get("prediction", ""))
    if value.lower() in {"yes", "no"}:
        return "boolean"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return "number"
    if re.match(r"^-?\d{1,4}-\d{2}-\d{2}$", value):
        return "date"
    return "text_or_quantity"


def _tokens(text):
    return set(re.findall(r"[a-z]+", str(text).lower())) - STOP_WORDS


def _numbers(text):
    # Commas inside numeric values are formatting; minus signs are meaningful.
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text))
    result = set()
    for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text):
        try:
            result.add(format(float(value), ".12g"))
        except ValueError:
            pass
    return result


def _expected_comparison(question):
    """Conservative global cue: abstain when several directions are mentioned."""
    q = question.lower()
    directions = set()
    if re.search(r"\b(not equal|other than|different from)\b", q):
        directions.add("!=")
    if re.search(r"\b(more than|greater than|larger than|higher than|longer than|after)\b", q):
        directions.add(">")
    if re.search(r"\b(less than|fewer than|smaller than|lower than|shorter than|before)\b", q):
        directions.add("<")
    # An equality is not inferred merely from the absence of a comparative.
    return next(iter(directions)) if len(directions) == 1 else None


def _get_program(candidate):
    program = candidate.get("program", [])
    if isinstance(program, str):
        try:
            from .executor import parse_program
            program = parse_program(program)
        except (ValueError, TypeError, ImportError):
            return []
    return program if isinstance(program, list) else []


def features(question: str, candidate: dict) -> list[float]:
    """Accept only question + candidate; never pass the dataset row here."""
    program = _get_program(candidate)
    inputs, numeric_inputs, qualifier_keys, operators = [], [], [], []
    n_conditions = 0
    for step in program:
        if not isinstance(step, dict):
            continue
        fn, args = step.get("function", ""), step.get("inputs", [])
        if not isinstance(args, list):
            continue
        inputs.extend(str(arg) for arg in args)
        # Entity names such as Apollo 11 also carry necessary question numbers.
        numeric_inputs.extend(str(arg) for arg in args)
        if fn in QUALIFIERS:
            n_conditions += 1
            pos = {"QueryAttrUnderCondition": 1, "QueryAttrQualifier": 2,
                   "QueryRelationQualifier": 1}.get(fn, 0)
            if pos < len(args):
                qualifier_keys.append(str(args[pos]))
        if fn in COMPARISONS and args:
            operators.append(str(args[-1]))
    question_numbers = _numbers(question)
    program_numbers = _numbers(" ".join(numeric_inputs))
    matched = question_numbers & program_numbers
    number_coverage = len(matched) / len(question_numbers) if question_numbers else 0.0
    number_precision = len(matched) / len(program_numbers) if program_numbers else 0.0
    missing = len(question_numbers - program_numbers) / max(1, len(question_numbers))
    extra = len(program_numbers - question_numbers) / max(1, len(program_numbers))
    q_tokens, p_tokens = _tokens(question), _tokens(" ".join(inputs))
    common = q_tokens & p_tokens
    expected = _expected_comparison(question)
    single = operators[0] if len(operators) == 1 else None
    has_year = bool(re.search(r"\b(?:1[5-9]|20)\d{2}\b", question))
    key_tokens = _tokens(" ".join(qualifier_keys))
    return [1.0, number_coverage, number_precision, missing, extra,
            len(common) / max(1, len(q_tokens)), len(common) / max(1, len(p_tokens)),
            float(expected is not None and expected == single),
            float(expected is not None and single is not None and expected != single),
            float(has_year and n_conditions > 0), float(has_year and n_conditions == 0),
            len(key_tokens & q_tokens) / max(1, len(key_tokens)),
            min(n_conditions, 5) / 5.0, min(len(program), 30) / 30.0,
            float(candidate.get("valid", False)),
            float(empty_prediction(candidate)),
            ]


def rule_score(question: str, candidate: dict) -> float:
    values = dict(zip(FEATURES, features(question, candidate)))
    # Empty answers are legitimate and are not penalized. Year presence alone
    # also does not warrant assuming a qualifier (it may be an entity name).
    return (2.0 * values["number_coverage"] - 2.0 * values["missing_number_fraction"]
            - values["extra_number_fraction"] + values["lexical_coverage"]
            + values["comparison_match"] - values["comparison_mismatch"])


def choose(question: str, candidates: list[dict], model: dict | None = None) -> int | None:
    valid = [i for i, item in enumerate(candidates) if item.get("valid", False)]
    if not valid:
        return None
    if model is not None and model.get("features") != FEATURES:
        raise ValueError("Ranker feature schema mismatch")
    def score(index):
        if model is None:
            return rule_score(question, candidates[index])
        return sum(w * x for w, x in zip(model["weights"], features(question, candidates[index])))
    # Equal scores retain generator order, including all duplicate candidates.
    return max(valid, key=lambda i: (score(i), -i))


def train_pairs(rows: list[dict], seed=17, epochs=20, learning_rate=0.05, l2=0.001) -> dict:
    """Pairwise logistic probe with exactly one positive/negative per row.

    Labels must be externally verified by the official executor. Rows carry
    question, positive, negative, and split='train'. Sources are not features.
    """
    if not rows:
        raise ValueError("No training pairs")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("All pairs must explicitly declare split='train'")
    diffs = [[a - b for a, b in zip(features(r["question"], r["positive"]),
                                  features(r["question"], r["negative"]))] for r in rows]
    weights = [0.0] * len(FEATURES)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            diff = diffs[idx]
            margin = sum(w * x for w, x in zip(weights, diff))
            gradient = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, margin))))
            weights = [w + learning_rate * (gradient * x - l2 * w)
                       for w, x in zip(weights, diff)]
    return {"method": "linear_pairwise_logistic_probe", "features": FEATURES,
            "weights": weights, "seed": seed, "epochs": epochs,
            "learning_rate": learning_rate, "l2": l2, "pairs": len(rows),
            "pair_ids": sorted({str(r.get("id", "")) for r in rows}),
            "warning": "Diagnostic low-capacity probe; not a neural reranker replication"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    raw = args.train_pairs.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    model = train_pairs(rows, args.seed, args.epochs, args.learning_rate)
    model["training_file_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in model.items() if k not in {"weights", "pair_ids"}}))


if __name__ == "__main__":
    main()
