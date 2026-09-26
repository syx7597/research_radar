"""Create execution-verified single-condition counterexamples on train/dev.

Reference programs are used ONLY for training/diagnosis. These mutations must
never be inserted into the natural candidate pools used to report QA accuracy.
An unchanged answer is excluded, not declared a negative or an equivalent form.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .execute_predictions import CandidateTimeout, candidate_deadline, indexed
from .rerank import COMPARISONS, empty_prediction, output_kind


def qualifier_schema(kb: dict) -> dict[str, set[str]]:
    result = defaultdict(set)
    for entity in kb["entities"].values():
        for fact in entity.get("attributes", []) + entity.get("relations", []):
            for key, values in fact.get("qualifiers", {}).items():
                for value in values:
                    result[key].add(value.get("type", "unknown"))
    return dict(result)


def _remove_unary(program, index):
    """Delete a qualifier filter and reconnect every downstream dependency."""
    deps = program[index].get("dependencies", [])
    if len(deps) != 1:
        return None
    replacement = deps[0]
    if replacement >= index:
        return None
    result = []
    for old_index, step in enumerate(deepcopy(program)):
        if old_index == index:
            continue
        new_deps = []
        for dependency in step.get("dependencies", []):
            dependency = replacement if dependency == index else dependency
            new_deps.append(dependency - int(dependency > index))
        step["dependencies"] = new_deps
        result.append(step)
    return result


def propose(program: list[dict], schema: dict[str, set[str]]):
    """Yield one semantic edit per candidate, never arbitrary multi-edit noise."""
    for index, step in enumerate(program):
        fn, inputs = step["function"], step.get("inputs", [])
        if fn in COMPARISONS and inputs and inputs[-1] in {"=", "!=", ">", "<"}:
            for operator in ("=", "!=", ">", "<"):
                if operator != inputs[-1]:
                    new = deepcopy(program)
                    new[index]["inputs"][-1] = operator
                    yield {"kind": "comparison_operator", "step": index,
                           "from": inputs[-1], "to": operator}, new
        if fn in {"SelectAmong", "SelectBetween"} and inputs:
            opposite = {"largest": "smallest", "smallest": "largest",
                        "greater": "less", "less": "greater"}.get(inputs[-1])
            if opposite:
                new = deepcopy(program)
                new[index]["inputs"][-1] = opposite
                yield {"kind": "comparison_operator", "step": index,
                       "from": inputs[-1], "to": opposite}, new
        if fn.startswith("QFilter"):
            new = _remove_unary(program, index)
            if new:
                yield {"kind": "omit_qualifier_filter", "step": index}, new
        if fn == "QueryAttrUnderCondition":
            new = deepcopy(program)
            new[index]["function"] = "QueryAttr"
            new[index]["inputs"] = [inputs[0]]
            yield {"kind": "omit_attribute_condition", "step": index}, new
        key_pos = {"QFilterStr": 0, "QFilterNum": 0, "QFilterYear": 0,
                   "QFilterDate": 0, "QueryAttrUnderCondition": 1,
                   "QueryAttrQualifier": 2, "QueryRelationQualifier": 1}.get(fn)
        if key_pos is not None and key_pos < len(inputs):
            original = inputs[key_pos]
            types = schema.get(original, set())
            for key in sorted(schema):
                if key == original or not types & schema[key]:
                    continue
                # For typed filters use their required literal type, not merely
                # another incidental type that the original key may also have.
                required = {"QFilterStr": "string", "QFilterNum": "quantity",
                            "QFilterYear": "year", "QFilterDate": "date"}.get(fn)
                if required and required not in schema[key]:
                    continue
                new = deepcopy(program)
                new[index]["inputs"][key_pos] = key
                yield {"kind": "qualifier_key", "step": index,
                       "from": original, "to": key}, new


def bounded_execute(executor, program, timeout_seconds=5.0):
    """Apply the same deadline as candidate-cache construction to every replay."""
    if timeout_seconds <= 0:
        raise ValueError("Execution timeout must be positive")
    try:
        with candidate_deadline(timeout_seconds):
            return executor.execute(program)
    except CandidateTimeout as error:
        return {"valid": False, "prediction": None, "answers": [], "empty_result": False,
                "error": f"CandidateTimeout: {error}"}


def _timed_out(execution):
    # KoPLExecutor may catch the signal exception and return it as an error.
    return str(execution.get("error", "")).startswith("CandidateTimeout:")


def validated_natural_cache(rows, cache_rows):
    expected = indexed(rows, "mutation source")
    cache = indexed(cache_rows, "natural execution cache")
    for qid, row in expected.items():
        if qid not in cache:
            raise ValueError(f"Missing natural-cache question: {qid}")
        if cache[qid].get("question") != row["question"]:
            raise ValueError(f"Natural-cache question mismatch: {qid}")
        if not isinstance(cache[qid].get("candidates"), list):
            raise ValueError(f"Missing natural-cache candidates: {qid}")
    return cache


def cached_natural_negative(row, cache_row, compare_answers):
    """Read previously verified observations; never resurrect invalid programs."""
    for candidate in cache_row["candidates"]:
        if candidate.get("valid") is not True:
            continue
        if not isinstance(candidate.get("program"), list) or candidate.get("prediction") is None:
            raise ValueError(f"Malformed valid execution in natural cache: {row['id']}")
        if not compare_answers(row["answer"], candidate["prediction"]):
            keys = ("program", "valid", "prediction", "error", "answers", "empty_result", "execution_seconds")
            return {key: candidate[key] for key in keys if key in candidate}
    return None


def verified_negatives(row, executor, compare_answers, schema, *, seed=17, max_attempts=24,
                       timeout_seconds=5.0):
    baseline = bounded_execute(executor, row["program"], timeout_seconds)
    stats = Counter()
    if _timed_out(baseline):
        stats["timeout_count"] += 1
        stats["gold_execution_timeout"] += 1
        return [], stats
    if not baseline["valid"] or not compare_answers(row["answer"], baseline["prediction"]):
        stats["gold_execution_mismatch"] += 1
        return [], stats
    # Stable per-question randomness makes a prefix independent of batch order.
    key = str(row.get("id", row["question"]))
    rng = random.Random(f"{seed}:{key}")
    # Large qualifier vocabularies must not crowd all comparison/deletion edits
    # out of the bounded execution budget. Interleave edit families randomly.
    by_kind = defaultdict(list)
    for edit, program in propose(row["program"], schema):
        by_kind[edit["kind"]].append((edit, program))
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    for values in by_kind.values():
        rng.shuffle(values)
    proposals = []
    while len(proposals) < max_attempts and any(by_kind.values()):
        for kind in kinds:
            if by_kind[kind]:
                proposals.append(by_kind[kind].pop())
                if len(proposals) == max_attempts:
                    break
    negatives, seen = [], set()
    for edit, program in proposals[:max_attempts]:
        signature = json.dumps(program, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        stats["attempted"] += 1
        execution = bounded_execute(executor, program, timeout_seconds)
        if _timed_out(execution):
            stats["timeout_count"] += 1
            stats["mutation_execution_timeout"] += 1
        if not execution["valid"]:
            stats["invalid_execution"] += 1
            continue
        if compare_answers(row["answer"], execution["prediction"]):
            stats["same_answer_excluded"] += 1
            continue
        negative = {"program": program, **execution, "edit": edit}
        negatives.append(negative)
        stats["accepted"] += 1
        stats["accepted_" + edit["kind"]] += 1
        if empty_prediction(execution):
            stats["accepted_empty_prediction"] += 1
    return negatives, stats


def _read_rows(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--natural-cache", type=Path)
    parser.add_argument("--natural-output", type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-negative-length-difference", type=int, default=2,
                        help="When comparing sources, match output kind and program length within this tolerance")
    args = parser.parse_args()
    if bool(args.natural_cache) != bool(args.natural_output):
        parser.error("--natural-cache and --natural-output must be supplied together")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    from .executor import KoPLExecutor, compare_answers
    executor = KoPLExecutor(args.kb, source_root=args.source_root)
    schema = qualifier_schema(json.loads(args.kb.read_text()))
    rows = _read_rows(args.data)[:args.limit]
    natural = validated_natural_cache(rows, _read_rows(args.natural_cache)) if args.natural_cache else {}
    natural_metadata = None
    if args.natural_cache:
        metadata_path = args.natural_cache.with_suffix(args.natural_cache.suffix + ".meta.json")
        if not metadata_path.is_file():
            raise ValueError("Natural cache needs execution metadata from execute_predictions.py")
        natural_metadata = json.loads(metadata_path.read_text())
        if natural_metadata.get("output_sha256") != hashlib.sha256(args.natural_cache.read_bytes()).hexdigest():
            raise ValueError("Natural cache does not match its execution metadata hash")
        if natural_metadata.get("executor_backend") != "baseline":
            raise ValueError("Natural and targeted executions must use the same baseline executor")
        if natural_metadata.get("limits", {}).get("per_candidate_timeout_seconds") != args.timeout_seconds:
            raise ValueError("Natural and targeted executions must use the same timeout")
        if natural_metadata.get("kb_sha256") != hashlib.sha256(args.kb.read_bytes()).hexdigest():
            raise ValueError("Natural and targeted executions must use the same KB")
    targeted_pairs, natural_pairs, stats = [], [], Counter()
    for row in rows:
        stats["questions_considered"] += 1
        natural_negative = None
        if args.natural_cache:
            natural_negative = cached_natural_negative(row, natural[str(row["id"])], compare_answers)
            if natural_negative is None:
                stats["no_natural_negative"] += 1
                continue
        negatives, counts = verified_negatives(row, executor, compare_answers, schema,
                                               seed=args.seed, max_attempts=args.max_attempts,
                                               timeout_seconds=args.timeout_seconds)
        stats.update(counts)
        if not negatives:
            continue
        if natural_negative:
            negatives = [negative for negative in negatives
                         if output_kind(negative) == output_kind(natural_negative)
                         and abs(len(negative["program"]) - len(natural_negative["program"]))
                         <= args.max_negative_length_difference]
            if not negatives:
                stats["no_output_length_matched_targeted_negative"] += 1
                continue
        # One pair per question in both groups. If supplied, natural/targeted
        # output files consequently have identical question IDs and pair counts.
        positive_execution = bounded_execute(executor, row["program"], args.timeout_seconds)
        if _timed_out(positive_execution):
            stats["timeout_count"] += 1
            stats["positive_replay_timeout"] += 1
            continue
        if not positive_execution["valid"] or not compare_answers(row["answer"], positive_execution["prediction"]):
            stats["positive_replay_mismatch"] += 1
            continue
        positive = {"program": row["program"], **positive_execution}
        common = {"id": row["id"], "question": row["question"],
                  "positive": positive, "split": args.split}
        targeted_pairs.append({**common, "negative": negatives[0], "negative_source": "single_condition_edit"})
        if natural_negative:
            natural_pairs.append({**common, "negative": natural_negative, "negative_source": "natural_generator_error"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in targeted_pairs))
    if args.natural_output:
        args.natural_output.parent.mkdir(parents=True, exist_ok=True)
        args.natural_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in natural_pairs))
    report = {"seed": args.seed, "split": args.split, "targeted_pairs": len(targeted_pairs),
              "natural_pairs": len(natural_pairs), "counts": dict(stats),
              "max_attempts_per_question": args.max_attempts,
              "per_candidate_timeout_seconds": args.timeout_seconds,
              "timeout_count": stats["timeout_count"],
              "natural_cache_sha256": natural_metadata["output_sha256"] if natural_metadata else None,
              "natural_execution_policy": "Reuse only valid=True observations from hash-verified execution cache; no re-execution",
              "proposal_sampling": "seeded round-robin over single-edit families",
              "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "matched_output_kind": bool(args.natural_cache),
              "max_negative_length_difference": args.max_negative_length_difference,
              "negative_distributions": {
                  name: {"output_kind": dict(Counter(output_kind(pair["negative"]) for pair in pairs)),
                         "program_length": dict(Counter(len(pair["negative"]["program"]) for pair in pairs)),
                         "answer_count": dict(Counter(len(pair["negative"].get("answers", [])) for pair in pairs)),
                         "edit_kind": dict(Counter(pair["negative"].get("edit", {}).get("kind", "natural") for pair in pairs))}
                  for name, pairs in [("targeted", targeted_pairs), ("natural", natural_pairs)]},
              "warning": "Synthetic diagnostic/training pairs, NOT natural QA candidate evaluation. Output kind is matched when natural cache is supplied; length matching has a bounded tolerance and does not remove all source shortcuts."}
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
