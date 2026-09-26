"""Join question-only predictions to gold offline, then execute every candidate.

python3 -m experiments.condition_consistency.execute_predictions \
    --predictions runs/dev_candidates.jsonl \
    --gold data/condition_consistency/splits/dev.jsonl \
    --kb datasets/kqa_pro/kb.json --output runs/dev_executed.jsonl

Gold fields never influence generation, candidate ordering, parsing or execution.
Errors and empty candidate lists remain in the evaluation denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.condition_consistency.baseline import read_records, sha256, write_json
from experiments.condition_consistency.executor import BASELINES_COMMIT, KOPL_COMMIT, KoPLExecutor, parse_program


class CandidateTimeout(TimeoutError):
    """A uniform per-candidate time limit was exceeded."""


@contextmanager
def candidate_deadline(seconds: float):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("Candidate deadlines require POSIX signal.setitimer")

    def timed_out(_signum, _frame):
        raise CandidateTimeout(f"Candidate exceeded {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def indexed(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"Missing id in {label}")
        identifier = str(row["id"])
        if identifier in result:
            raise ValueError(f"Duplicate id in {label}: {identifier}")
        result[identifier] = row
    return result


def validate_join(predictions: list[dict], gold: list[dict]) -> dict[str, dict]:
    predicted = indexed(predictions, "predictions")
    expected = indexed(gold, "gold")
    missing = sorted(expected.keys() - predicted.keys())
    unexpected = sorted(predicted.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(f"ID sets differ: {len(missing)} missing, {len(unexpected)} unexpected; "
                         f"examples missing={missing[:3]}, unexpected={unexpected[:3]}")
    if not predicted:
        raise ValueError("Prediction/gold inputs are empty")
    for identifier, row in predicted.items():
        target = expected[identifier]
        if row["question"] != target["question"]:
            raise ValueError(f"Question mismatch for id {identifier}")
        if "answer" not in target or not isinstance(target.get("program"), list):
            raise ValueError(f"Missing gold answer/program for id {identifier}")
        if not isinstance(row.get("candidates"), list):
            raise ValueError(f"Candidates must be a list for id {identifier}")
        if not all(isinstance(candidate, dict) for candidate in row["candidates"]):
            raise ValueError(f"Every candidate must be an object for id {identifier}")
    return expected


def execute_candidate(candidate: dict, executor: KoPLExecutor, timeout_seconds: float) -> dict:
    # Keep generator scores and original text. Always overwrite stale execution
    # fields, so untrusted cached validity cannot bypass the official executor.
    result = dict(candidate)
    result.update(program=None, valid=False, prediction=None, error=None, answers=[], empty_result=False)
    started = time.perf_counter()
    try:
        with candidate_deadline(timeout_seconds):
            result["program"] = parse_program(candidate.get("program_text"))
            result.update(executor.execute(result["program"]))
    except Exception as error:
        result.update(valid=False, prediction=None,
                      error=f"{type(error).__name__}: {error}", answers=[], empty_result=False)
    result["execution_seconds"] = time.perf_counter() - started
    return result


def run(args: argparse.Namespace) -> dict:
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.output.exists() and not args.overwrite:
        raise ValueError("Output exists; use a new path or explicitly --overwrite")
    if args.output.resolve() in {args.predictions.resolve(), args.gold.resolve(), args.kb.resolve()}:
        raise ValueError("Output cannot overwrite an input")
    predictions = read_records(args.predictions)
    gold = validate_join(predictions, read_records(args.gold))
    if args.expected_candidates is not None:
        for row in predictions:
            if len(row["candidates"]) != args.expected_candidates:
                raise ValueError(f"Candidate count differs from the declared K for id {row['id']}")
    started = time.perf_counter()
    source = args.executor_source or ROOT / (
        "external/kqa_pro_baselines" if args.backend == "baseline" else "external/kopl")
    source_file = source / ("Program/executor_rule.py" if args.backend == "baseline" else "src/kopl/kopl.py")
    metadata = {
        "predictions_path": str(args.predictions), "predictions_sha256": sha256(args.predictions),
        "gold_path": str(args.gold), "gold_sha256": sha256(args.gold),
        "kb_path": str(args.kb), "kb_sha256": sha256(args.kb),
        "executor_backend": args.backend,
        "executor_source_commit": BASELINES_COMMIT if args.backend == "baseline" else KOPL_COMMIT,
        "executor_source_path": str(source),
        "executor_source_file_sha256": sha256(source_file),
        "gold_usage": "offline join for evaluation only; no candidate insertion or reordering",
        "limits": {"per_candidate_timeout_seconds": args.timeout_seconds,
                   "program_max_steps": 64, "program_max_characters": 65536,
                   "argument_max_characters": 2048},
        "questions": len(predictions),
    }
    initialized = time.perf_counter()
    executor = KoPLExecutor(args.kb, source_root=source, backend=args.backend)
    metadata["executor_initialization_seconds"] = time.perf_counter() - initialized
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    valid = total = valid_questions = 0
    errors, counts = Counter(), Counter()
    execution_started = time.perf_counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, prediction in enumerate(predictions):
            identifier = str(prediction["id"])
            target = gold[identifier]
            candidates = [execute_candidate(candidate, executor, args.timeout_seconds)
                          for candidate in prediction["candidates"]]
            record = {"id": identifier, "question": prediction["question"],
                      "answer": target["answer"], "program": target["program"],
                      "candidates": candidates}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[len(candidates)] += 1
            total += len(candidates)
            valid += sum(candidate["valid"] for candidate in candidates)
            valid_questions += any(candidate["valid"] for candidate in candidates)
            for candidate in candidates:
                if not candidate["valid"]:
                    errors[str(candidate["error"]).split(":", 1)[0]] += 1
            if (index + 1) % 100 == 0 or index + 1 == len(predictions):
                handle.flush()
                print(f"Executed {index + 1}/{len(predictions)} questions; "
                      f"valid candidates {valid}/{total}", flush=True)
    temporary.replace(args.output)
    metadata.update(
        output_path=str(args.output), output_sha256=sha256(args.output),
        candidate_count_distribution=dict(sorted(counts.items())),
        total_candidates=total, valid_candidates=valid,
        candidate_valid_ratio=valid / total if total else None,
        questions_with_valid_candidates=valid_questions,
        question_any_valid_ratio=valid_questions / len(predictions),
        error_types=dict(errors),
        candidate_execution_seconds=time.perf_counter() - execution_started,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output.with_suffix(args.output.suffix + ".meta.json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("baseline", "modern"), default="baseline",
                        help="Dataset-release RuleExecutor by default; modern KoPL only for explicit comparison")
    parser.add_argument("--executor-source", "--kopl-source", dest="executor_source", type=Path,
                        help="Source checkout matching --backend; --kopl-source is a compatibility alias")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
