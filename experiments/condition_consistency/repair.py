"""Gold-free, bounded single-field repairs for the released KQA Pro executor.

Example: python3 -m experiments.condition_consistency.repair --predictions
    runs/dev_candidates.jsonl --kb datasets/kqa_pro/kb.json --mode local
    --output runs/dev_local_repairs.jsonl

This proposes candidates; it does not accept a repair or consult reference
answers. Empty results and locally absent legal fields never trigger repair.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .baseline import read_records, sha256, write_json
from .execute_predictions import candidate_deadline, execute_candidate
from .executor import BASELINES_COMMIT, KoPLExecutor, parse_program, serialize_program

VERSION = "single_field_v1"
GENERATOR_FIELDS = {"rank", "sequence_logprob", "normalized_logprob", "beam_score",
                    "generated_tokens", "ended_with_eos", "hit_generation_limit"}
FILTER_TYPES = {"Str": {"string"}, "Num": {"quantity"}, "Year": {"year", "date"},
                "Date": {"year", "date"}}
ATTRIBUTE_FUNCTIONS = {"FilterStr", "FilterNum", "FilterYear", "FilterDate",
                       "QueryAttr", "QueryAttrUnderCondition", "QueryAttrQualifier",
                       "SelectAmong", "SelectBetween"}


def field_slots(program: list[dict], index: int) -> list[dict]:
    """Only schema fields are editable; literals and entity/concept names are not."""
    function = program[index]["function"]
    slots = []
    if function in ATTRIBUTE_FUNCTIONS:
        slots.append({"input": 0, "role": "attribute"})
    if function.startswith("QFilter"):
        slots.append({"input": 0, "role": "qualifier"})
    if function in {"Relate", "QueryRelationQualifier"}:
        slots.append({"input": 0, "role": "relation"})
    qualifier_position = {"QueryAttrUnderCondition": 1, "QueryAttrQualifier": 2,
                          "QueryRelationQualifier": 1}.get(function)
    if qualifier_position is not None:
        slots.append({"input": qualifier_position, "role": "qualifier"})
    for slot in slots:
        expected = None
        if function.startswith(("Filter", "QFilter")):
            expected = FILTER_TYPES.get(function.removeprefix("Q").removeprefix("Filter"))
        if function in {"SelectAmong", "SelectBetween"}:
            expected = {"quantity", "year", "date"}
        # A Verify dependency supplies an explicit output type, unlike a guessed
        # question category or a numeric-looking string literal.
        is_output = (function == "QueryAttr" or
                     function == "QueryAttrUnderCondition" and slot["input"] == 0 or
                     function in {"QueryAttrQualifier", "QueryRelationQualifier"}
                     and slot["role"] == "qualifier")
        if is_output:
            for later in program[index + 1:]:
                if index in later["dependencies"] and later["function"].startswith("Verify"):
                    expected = FILTER_TYPES[later["function"].removeprefix("Verify")]
        slot["expected_types"] = sorted(expected) if expected else []
    return slots


def field_score(question: str, old: str, new: str) -> float:
    normalize = lambda text: " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    old_words, new_words = normalize(old), normalize(new)
    lexical = SequenceMatcher(None, old_words, new_words).ratio()
    tokens = set(new_words.split())
    question_overlap = len(tokens & set(normalize(question).split())) / max(1, len(tokens))
    return 0.85 * lexical + 0.15 * question_overlap


def _value_type(value) -> str:
    return value["type"] if isinstance(value, dict) else value.type


def _fact_types(fact: dict, role: str, key: str) -> set[str]:
    if role == "relation":
        return {"entity"}
    if role == "attribute":
        return {_value_type(fact["value"])}
    return {_value_type(value) for value in fact.get("qualifiers", {}).get(key, [])}


class Repairer:
    """Use the fixed official engine both for prefixes and full execution.

    `repair_row` explicitly reads only id, question, and candidate program_text
    plus a generator-score whitelist. Cached program/validity/answer/correctness
    fields cannot influence a proposal. Output is intentionally gold-free.
    """
    def __init__(self, executor: KoPLExecutor, mode: str = "local", max_original: int = 4,
                 max_repairs: int = 4, timeout_seconds: float = 5.0,
                 min_similarity: float = 0.0):
        if mode not in {"global", "local"}:
            raise ValueError("mode must be global or local")
        if not 1 <= max_original <= 4 or not 0 <= max_repairs <= 4:
            raise ValueError("At most four originals and four repairs are allowed")
        if timeout_seconds <= 0 or not 0 <= min_similarity <= 1:
            raise ValueError("Invalid timeout or similarity threshold")
        if executor.backend != "baseline":
            raise ValueError("Repair prefix semantics require the dataset-release baseline executor")
        self.executor, self.engine, self.mode = executor, executor.engine, mode
        self.max_original, self.max_repairs = max_original, max_repairs
        self.timeout_seconds, self.min_similarity = timeout_seconds, min_similarity
        self.schema = {role: defaultdict(set) for role in ("attribute", "relation", "qualifier")}
        for entity in self.engine.entities.values():
            for role, facts in (("attribute", entity.get("attributes", [])),
                                ("relation", entity.get("relations", []))):
                for fact in facts:
                    key = fact["key"] if role == "attribute" else fact["predicate"]
                    self.schema[role][key].update(_fact_types(fact, role, key))
                    for qualifier in fact.get("qualifiers", {}):
                        self.schema["qualifier"][qualifier].update(_fact_types(fact, "qualifier", qualifier))
        self.config = {"version": VERSION, "mode": mode, "max_original": max_original,
                       "max_repairs": max_repairs, "timeout_seconds": timeout_seconds,
                       "min_similarity": min_similarity,
                       "score": "0.85*old_new_sequence_ratio+0.15*question_new_token_recall",
                       "allocation": "round_robin_parent_rank_then_match_score",
                       "trigger": "global_role_illegal_or_explicit_function_type_mismatch",
                       "empty_result_triggers": False, "local_missing_field_triggers": False,
                       "gold_usage": "none", "rounds": 1}
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def _trigger(self, step: dict, slot: dict) -> str | None:
        key = step["inputs"][slot["input"]]
        role_schema = self.schema[slot["role"]]
        if key not in role_schema:
            return "globally_illegal_role_field"
        if slot["expected_types"] and not role_schema[key].intersection(slot["expected_types"]):
            return "explicit_function_type_mismatch"
        return None

    def _annotate_schema(self, candidate: dict) -> dict:
        program = candidate.get("program")
        issues = []
        if program is None:
            issues.append({"reason": "unparseable_program"})
        else:
            for index, step in enumerate(program):
                for slot in field_slots(program, index):
                    trigger = self._trigger(step, slot)
                    if trigger:
                        issues.append({"step": index, "input": slot["input"],
                                       "role": slot["role"], "reason": trigger})
        candidate["schema_clean"] = not issues
        candidate["schema_issues"] = issues
        return candidate

    def _prefix(self, program: list[dict], target: int) -> tuple[list, dict]:
        memory, visited = {}, []
        self.row_cost["prefix_calls"] += 1

        def visit(index):
            if index in memory:
                return memory[index]
            step = program[index]
            for slot in field_slots(program, index):
                if self._trigger(step, slot):
                    raise ValueError(f"upstream_schema_conflict_at_{index}")
            dependencies = [visit(dep) for dep in step["dependencies"]]
            function, inputs = step["function"], step["inputs"]
            if function == "Find":
                ids = list(self.engine.entity_name_to_ids.get(inputs[0], []))
                ids += list(self.engine.concept_name_to_ids.get(inputs[0], []))
                if len(set(ids)) != 1:
                    raise ValueError(f"upstream_entity_not_unique_at_{index}")
            if function == "FilterConcept" and not self.engine.concept_name_to_ids.get(inputs[0]):
                raise ValueError(f"upstream_unknown_concept_at_{index}")
            self.row_cost["prefix_executed_steps"] += 1
            result = getattr(self.engine, function)(dependencies, inputs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(f"upstream_not_entity_context_at_{index}")
            if not result[0]:
                raise ValueError(f"upstream_empty_context_at_{index}")
            memory[index] = result
            visited.append(index)
            return result

        started = time.perf_counter()
        try:
            with candidate_deadline(self.timeout_seconds):
                dependencies = [visit(dep) for dep in program[target]["dependencies"]]
        finally:
            self.row_cost["prefix_seconds"] += time.perf_counter() - started
        context = {"trusted": True, "prefix_steps": sorted(visited),
                   "trust_criteria": ["valid_structure", "ancestor_schema_types_compatible",
                                      "named_entity_unique", "prefix_execution_success",
                                      "nonempty_entity_context"],
                   "semantic_correctness_guaranteed": False,
                   "entity_counts": [len(dep[0]) for dep in dependencies],
                   "entity_ids_sample": [list(dep[0])[:20] for dep in dependencies],
                   "attached_fact_counts": [len(dep[1]) if dep[1] is not None else 0
                                            for dep in dependencies]}
        return dependencies, context

    def _entity(self, identifier: str) -> dict:
        return self.engine.entities.get(identifier, self.engine.concepts.get(identifier, {}))

    def _matches(self, value, key: str, literal: str) -> bool:
        """Use official value parsing/time containment, not string equality."""
        try:
            parsed = self.engine._parse_key_value(key, literal)
            if not value.can_compare(parsed):
                return False
            return bool(parsed.contains(value) if parsed.isTime() else value == parsed)
        except (ValueError, TypeError, KeyError, AssertionError, AttributeError):
            return False

    def _local_fields(self, program: list[dict], index: int, slot: dict,
                      dependencies: list) -> dict[str, set[str]]:
        step, role = program[index], slot["role"]
        function, args = step["function"], step["inputs"]
        fields = defaultdict(set)
        first_ids, attached = dependencies[0]
        if function.startswith("QFilter"):
            if attached is None or len(attached) != len(first_ids):
                raise ValueError("qualifier_context_has_no_aligned_facts")
            facts = attached
        elif function == "QueryRelationQualifier":
            first, second = first_ids[0], dependencies[1][0][0]
            facts = [fact for fact in self._entity(first).get("relations", [])
                     if fact["object"] == second and fact["direction"] == "forward"
                     and (role == "relation" or fact["predicate"] == args[0])]
            if role == "relation":
                facts = [fact for fact in facts if args[1] in fact.get("qualifiers", {})]
        elif function == "Relate":
            facts = [fact for fact in self._entity(first_ids[0]).get("relations", [])
                     if fact["direction"] == args[1]]
        else:
            ids = first_ids if function.startswith("Filter") or function == "SelectAmong" else first_ids[:1]
            if function == "SelectBetween":
                ids = [first_ids[0], dependencies[1][0][0]]
            facts = [fact for identifier in ids for fact in self._entity(identifier).get("attributes", [])]
            if role == "qualifier":
                facts = [fact for fact in facts if fact["key"] == args[0]]
            if function == "QueryAttrQualifier":
                # A qualifier is retrieved only from the original attribute/value
                # fact. Replacing the attribute still preserves its value and qkey.
                facts = [fact for fact in facts if self._matches(fact["value"], fact["key"], args[1])]
                if role == "attribute":
                    facts = [fact for fact in facts if args[2] in fact.get("qualifiers", {})]
            if function == "QueryAttrUnderCondition" and role == "attribute":
                facts = [fact for fact in facts if any(
                    self._matches(value, args[1], args[2])
                    for value in fact.get("qualifiers", {}).get(args[1], []))]
            if function in {"SelectAmong", "SelectBetween"}:
                # Official selection expects the attribute on every involved
                # entity. The original comparison direction is left unchanged.
                keys = [set(f["key"] for f in self._entity(identifier).get("attributes", [])) for identifier in ids]
                shared = set.intersection(*keys) if keys else set()
                facts = [fact for fact in facts if fact["key"] in shared]
        for fact in facts:
            keys = fact.get("qualifiers", {}) if role == "qualifier" else [
                fact["key"] if role == "attribute" else fact["predicate"]]
            for key in keys:
                fields[key].update(_fact_types(fact, role, key))
        return fields

    def repair_row(self, row: dict) -> dict:
        row_started = time.perf_counter()
        self.row_cost = {"prefix_calls": 0, "prefix_executed_steps": 0, "prefix_seconds": 0.0,
                         "field_score_calls": 0, "field_scoring_seconds": 0.0}
        if not isinstance(row.get("candidates"), list):
            raise ValueError("Input needs a candidate list")
        question = row["question"]
        if not isinstance(question, str):
            raise ValueError("Question must be a string")
        originals, queues, audits = [], [], []
        for parent_index, candidate in enumerate(row["candidates"][:self.max_original]):
            clean = {key: candidate[key] for key in GENERATOR_FIELDS if key in candidate}
            clean.update(program_text=candidate.get("program_text"), origin="original",
                         original_index=parent_index)
            executed = self._annotate_schema(execute_candidate(clean, self.executor, self.timeout_seconds))
            originals.append(executed)
            proposals = []
            audit = {"parent_index": parent_index, "triggers": [], "rejections": []}
            program = executed["program"]
            if program is None:
                audit["rejections"].append({"reason": "unparseable_program", "detail": executed["error"]})
            else:
                for index, step in enumerate(program):
                    for slot in field_slots(program, index):
                        trigger = self._trigger(step, slot)
                        if trigger is None:
                            continue
                        old = step["inputs"][slot["input"]]
                        marker = {"step": index, "input": slot["input"], "role": slot["role"],
                                  "from": old, "reason": trigger,
                                  "expected_types": slot["expected_types"]}
                        audit["triggers"].append(marker)
                        context = {"scope": "global", "trusted": None}
                        candidates = self.schema[slot["role"]]
                        if self.mode == "local":
                            try:
                                dependencies, context = self._prefix(program, index)
                                candidates = self._local_fields(program, index, slot, dependencies)
                                context["scope"] = "local"
                                context["qualifier_fact_binding"] = slot["role"] == "qualifier"
                            except Exception as error:
                                audit["rejections"].append({**marker, "reason": "untrusted_local_context",
                                                            "detail": f"{type(error).__name__}: {error}"})
                                continue
                        matching_started = time.perf_counter()
                        allowed = [(key, field_score(question, old, key)) for key, types in candidates.items()
                                   if key != old and (not slot["expected_types"] or types.intersection(slot["expected_types"]))]
                        self.row_cost["field_score_calls"] += len(allowed)
                        allowed = sorted((key, score) for key, score in allowed if score >= self.min_similarity)
                        allowed.sort(key=lambda pair: -pair[1])
                        self.row_cost["field_scoring_seconds"] += time.perf_counter() - matching_started
                        if not allowed:
                            audit["rejections"].append({**marker, "reason": "no_compatible_replacement", "context": context})
                        for key, score in allowed[:self.max_repairs]:
                            changed = copy.deepcopy(program)
                            changed[index]["inputs"][slot["input"]] = key
                            proposals.append({"program_text": serialize_program(changed),
                                              "origin": "repair", "parent_index": parent_index,
                                              "repair_mode": self.mode,
                                              "edit": {"step": index, "input": slot["input"], "role": slot["role"], "from": old, "to": key},
                                              "trigger": trigger, "match_score": score,
                                              "context": {**context, "compatible_field_count": len(allowed)},
                                              "preserved": ["entities", "concepts", "literal_values", "directions", "functions", "dependencies", "other_fields"]})
            proposals.sort(key=lambda proposal: (-proposal["match_score"], proposal["edit"]["step"],
                                                 proposal["edit"]["input"], proposal["edit"]["to"]))
            audit["proposed_before_budget"] = len(proposals)
            queues.append(proposals)
            audits.append(audit)
        # Fair bounded allocation across the fixed generator ranks. Duplicate
        # programs consume no new execution; validity/answers do not select them.
        seen = {candidate["program_text"] for candidate in originals}
        repairs = []
        while len(repairs) < self.max_repairs and any(queues):
            for queue in queues:
                while queue and queue[0]["program_text"] in seen:
                    queue.pop(0)
                if queue and len(repairs) < self.max_repairs:
                    proposal = queue.pop(0)
                    seen.add(proposal["program_text"])
                    repairs.append(self._annotate_schema(execute_candidate(proposal, self.executor, self.timeout_seconds)))
        self.row_cost.update(original_execution_calls=len(originals), repair_execution_calls=len(repairs),
                             original_execution_seconds=sum(c["execution_seconds"] for c in originals),
                             repair_execution_seconds=sum(c["execution_seconds"] for c in repairs),
                             row_seconds=time.perf_counter() - row_started)
        return {"id": str(row["id"]), "question": question, "candidates": originals + repairs,
                "repair_audit": {"config_sha256": self.config_hash, "mode": self.mode,
                                 "input_candidates": len(row["candidates"]),
                                 "original_candidates": len(originals), "repair_candidates": len(repairs),
                                 "cost": self.row_cost,
                                 "parents": audits}}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.overwrite:
        raise ValueError("Output exists; use --overwrite or a new path")
    if args.output.resolve() in {args.predictions.resolve(), args.kb.resolve()}:
        raise ValueError("Output cannot overwrite an input")
    started = time.perf_counter()
    executor = KoPLExecutor(args.kb, source_root=args.executor_source, backend="baseline")
    repairer = Repairer(executor, args.mode, args.max_original, args.max_repairs,
                        args.timeout_seconds, args.min_similarity)
    rows = read_records(args.predictions)
    identifiers = [str(row["id"]) for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise ValueError("Input must be nonempty with unique IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    counts, triggers, rejections, cost = Counter(), Counter(), Counter(), Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            result = repairer.repair_row(row)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts["questions"] += 1
            counts["original_candidates"] += result["repair_audit"]["original_candidates"]
            counts["repair_candidates"] += result["repair_audit"]["repair_candidates"]
            counts["questions_with_repairs"] += bool(result["repair_audit"]["repair_candidates"])
            counts["questions_triggered"] += any(parent["triggers"] for parent in result["repair_audit"]["parents"])
            cost.update(result["repair_audit"]["cost"])
            for parent in result["repair_audit"]["parents"]:
                triggers.update(marker["reason"] for marker in parent["triggers"])
                rejections.update(marker["reason"] for marker in parent["rejections"])
            if (index + 1) % 100 == 0:
                print(f"Repaired {index + 1}/{len(rows)} questions", flush=True)
    temporary.replace(args.output)
    source = args.executor_source or Path(__file__).resolve().parents[2] / "external/kqa_pro_baselines"
    metadata = {"config": repairer.config, "config_sha256": repairer.config_hash,
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unfixed"),
                "predictions_sha256": sha256(args.predictions), "kb_sha256": sha256(args.kb),
                "output_sha256": sha256(args.output), "repair_source_sha256": sha256(Path(__file__)),
                "adapter_source_sha256": sha256(Path(__file__).with_name("executor.py")),
                "executor_source_commit": BASELINES_COMMIT,
                "executor_source_sha256": sha256(source / "Program/executor_rule.py"),
                "counts": dict(counts), "triggers": dict(triggers), "rejections": dict(rejections),
                "cost": dict(cost),
                "elapsed_seconds": time.perf_counter() - started,
                "gold_usage": "none; input gold metadata is discarded",
                "acceptance": "not performed; output contains all budgeted proposals, even failed ones"}
    write_json(args.output.with_suffix(args.output.suffix + ".meta.json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "local"), required=True)
    parser.add_argument("--executor-source", type=Path)
    parser.add_argument("--max-original", type=int, default=4)
    parser.add_argument("--max-repairs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
