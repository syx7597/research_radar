#!/usr/bin/env python3
"""CPU-only tokenizer replay audit; never generate, train, or execute queries.

The fixed scope is 5,000 training gold programs, 500 development gold programs,
and the existing 4/8 development candidates. Candidate text is retokenized: this
does not recover its original generation token trajectory or measure accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILES = (
    "added_tokens.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
ROLES = ("attribute", "relation", "qualifier")
INPUT_PATHS = {
    "checkpoint_manifest": "results/condition_consistency/pilot_bart5k_seed20260926/checkpoint_manifest.json",
    "schema": "results/condition_consistency/query_repair_audit/schema.json",
    "generator_train": "data/condition_consistency/splits/generator_train.jsonl",
    "dev_gold": "data/condition_consistency/splits/dev.jsonl",
    "dev_beam4": "results/condition_consistency/query_repair/dev_global_scored.jsonl",
    "dev_beam8": "results/condition_consistency/query_repair/dev_beam8_executed.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def read_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"{path.name}: expected {expected} rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != expected or any(not value.startswith("train:") for value in ids):
        raise ValueError(f"{path.name}: duplicate or non-training-source IDs")
    return rows


def text_of_gold(row: dict[str, Any]) -> str:
    # Keep the frozen serializer's exact spacing without loading model code.
    chunks = []
    for step in row["program"]:
        function, inputs = step["function"], step["inputs"]
        if not isinstance(function, str) or not all(isinstance(value, str) for value in inputs):
            raise ValueError(f"Non-string gold program component: {row['id']}")
        if any(marker in value for value in [function, *inputs] for marker in ("<func>", "<arg>")):
            raise ValueError(f"Reserved delimiter in gold program: {row['id']}")
        chunks.append(function + "".join(" <arg> " + value for value in inputs))
    return " <func> ".join(chunks)


def lexical_inventory(text: str, roles: dict[str, dict[str, str]],
                      signatures: dict[str, Any], names: dict[str, set[str]]) -> dict[str, Any]:
    fields: Counter[str] = Counter()
    out_of_vocab: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for chunk in text.strip().split("<func>"):
        parts = [part.strip() for part in chunk.split("<arg>")]
        function, arguments = parts[0], parts[1:]
        if function not in signatures:
            unknown[function] += 1
        for index, role in roles.get(function, {}).items():
            if role in ROLES and int(index) < len(arguments):
                fields[role] += 1
                if arguments[int(index)] not in names[role]:
                    out_of_vocab[role] += 1
    return {"fields": fields, "unknown": unknown, "out_of_vocab_fields": out_of_vocab}


def replay(text: str, constraint: Any, tokenizer: Any, decoder_start: int) -> dict[str, Any]:
    target_ids = tokenizer.encode(text, add_special_tokens=True)
    if target_ids[0] != tokenizer.bos_token_id or target_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Checkpoint tokenizer must encode targets as BOS + text + EOS")
    prefix = [decoder_start]
    inspected: Counter[str] = Counter()
    first_rejection = None
    for position, next_token in enumerate(target_ids):
        state = constraint.inspect(prefix)
        mode = state["mode"]
        if mode not in {"free", "field", "ended"}:
            raise ValueError(f"Unexpected inspect mode: {mode}")
        inspected[mode] += 1
        allowed = state["allowed"]
        if mode == "field" and allowed is None:
            raise ValueError("Field state cannot have unconstrained allowed=None")
        if allowed is not None and next_token not in allowed:
            first_rejection = {
                "target_token_index_zero_based": position,
                "prefix_length_including_decoder_start": len(prefix),
                "next_token_id": next_token,
                "next_token": tokenizer.convert_ids_to_tokens(next_token),
                "mode": mode, "function": state.get("function"),
                "argument": state.get("argument"), "role": state.get("role"),
                "reason": state.get("reason"), "allowed_token_count": len(allowed),
            }
            break
        prefix.append(next_token)
    return {"accepted": first_rejection is None, "first_rejection": first_rejection,
            "target_tokens": len(target_ids), "inspected_modes": inspected}


def audit_group(items: list[dict[str, Any]], constraint: Any, tokenizer: Any,
                decoder_start: int, roles: dict, signatures: dict,
                parse_program: Any, names: dict[str, set[str]], *, is_gold: bool) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    out_of_vocab_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    failures = []
    parse_failures = []
    for item in items:
        text = item["text"]
        inventory = lexical_inventory(text, roles, signatures, names)
        field_counts.update(inventory["fields"])
        out_of_vocab_counts.update(inventory["out_of_vocab_fields"])
        unknown_counts.update(inventory["unknown"])
        totals["programs"] += 1
        totals["programs_with_fields"] += bool(inventory["fields"])
        has_oov = bool(inventory["out_of_vocab_fields"])
        totals["programs_with_out_of_vocab_fields"] += has_oov
        totals["programs_with_unknown_functions"] += bool(inventory["unknown"])
        totals["cached_hit_generation_limit"] += bool(item.get("hit_generation_limit"))
        identity = {key: item[key] for key in ("id", "rank") if key in item}
        try:
            parse_program(text)  # Syntax/dependency validation only; no executor instance.
        except (ValueError, TypeError, KeyError, IndexError) as error:
            failure = {**identity, "error": f"{type(error).__name__}: {error}"}
            if is_gold:
                failure["program_text"] = text
            parse_failures.append(failure)
        result = replay(text, constraint, tokenizer, decoder_start)
        totals["retokenized_target_tokens"] += result["target_tokens"]
        mode_counts.update(result["inspected_modes"])
        totals["accepted" if result["accepted"] else "blocked"] += 1
        if not result["accepted"]:
            totals["blocked_with_out_of_vocab_fields" if has_oov
                   else "blocked_without_out_of_vocab_fields"] += 1
            rejection = result["first_rejection"]
            rejection_reasons[str(rejection["reason"])] += 1
            failure = {**identity, "first_rejection": rejection,
                       "out_of_vocab_fields_by_role": dict(inventory["out_of_vocab_fields"])}
            if is_gold:
                failure["program_text"] = text
            failures.append(failure)
    return {
        **{name: totals[name] for name in (
            "programs", "accepted", "blocked", "programs_with_fields",
            "programs_with_out_of_vocab_fields", "blocked_with_out_of_vocab_fields",
            "blocked_without_out_of_vocab_fields",
            "programs_with_unknown_functions", "cached_hit_generation_limit",
            "retokenized_target_tokens")},
        "field_occurrences_by_role": dict(sorted(field_counts.items())),
        "out_of_vocab_field_occurrences_by_role": dict(sorted(out_of_vocab_counts.items())),
        "unknown_function_occurrences": dict(sorted(unknown_counts.items())),
        "syntax_parse_failures": len(parse_failures), "parse_failures": parse_failures,
        "inspected_modes_until_first_block": dict(sorted(mode_counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "blocked_programs": failures,
    }


def audit_adversarial_token_paths(constraint: Any, tokenizer: Any,
                                  decoder_start: int) -> dict[str, Any]:
    """Three bounded token-level checks that canonical text replay cannot cover."""
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = ([decoder_start, tokenizer.bos_token_id] + encode("Query")
              + [tokenizer.bos_token_id] + encode("Attr <arg>"))
    state = constraint.inspect(prefix)
    restored = (state["mode"] == "field" and state["function"] == "QueryAttr"
                and state["argument"] == 0 and state["role"] == "attribute"
                and state["allowed"] is not None)
    first_bad_position = None
    bad_prefix = list(prefix)
    # Check the whole bad field + boundary: its first BPE token could legitimately
    # be shared with a real KB name and should not be arbitrarily disallowed.
    for position, next_token in enumerate(encode(" nonsense") + [tokenizer.eos_token_id]):
        allowed = constraint.inspect(bad_prefix)["allowed"]
        if allowed is not None and next_token not in allowed:
            first_bad_position = position
            break
        bad_prefix.append(next_token)
    probes = [{"name": "special_token_inside_function", "prefix_token_ids": prefix,
               "mode": state["mode"], "function": state["function"], "role": state["role"],
               "reason": state["reason"], "restored_attribute_slot": restored,
               "illegal_field_probe": "nonsense",
               "first_rejected_bad_field_token_index": first_bad_position,
               "passed": restored and first_bad_position is not None}]
    for label in ("arg", "func"):
        marker = "<" + label + ">"
        fragments = encode("<") + encode(label) + encode(">")
        atomic = encode(marker)
        decoded = tokenizer.decode(fragments, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        is_non_atomic = len(atomic) == 1 and atomic[0] not in fragments and decoded == marker
        prefix = [decoder_start, tokenizer.bos_token_id] + encode("QueryAttr ") + fragments
        state = constraint.inspect(prefix)
        passed = is_non_atomic and state["reason"] == "non_atomic_delimiter" and state["allowed"] == []
        probes.append({"name": f"fragmented_{label}_delimiter", "fragment_token_ids": fragments,
                       "atomic_token_ids": atomic, "decoded_fragments": decoded,
                       "genuinely_non_atomic": is_non_atomic, "reason": state["reason"],
                       "allowed_token_count": None if state["allowed"] is None else len(state["allowed"]),
                       "passed": passed})
    return {"description": "Three actual-checkpoint tokenizer probes; no model execution",
            "all_passed": all(probe["passed"] for probe in probes), "probes": probes}


def audit_lexicon(names: dict[str, list[str]], constraint: Any, tokenizer: Any,
                   decoder_start: int) -> dict[str, Any]:
    functions = {"attribute": "QueryAttr", "relation": "Relate", "qualifier": "QFilterStr"}
    per_role = {}
    failures = []
    for role in ROLES:
        blocked = 0
        for field in names[role]:
            for leading in ("", " "):
                for trailing in ("", " "):
                    # Deliberately a single-field lexical probe, not an executable query.
                    text = functions[role] + " <arg>" + leading + field + trailing
                    result = replay(text, constraint, tokenizer, decoder_start)
                    if not result["accepted"]:
                        blocked += 1
                        failures.append({"role": role, "field": field,
                                         "leading_spaces": len(leading),
                                         "trailing_spaces": len(trailing),
                                         "first_rejection": result["first_rejection"]})
        per_role[role] = {"fields": len(names[role]), "probes": 4 * len(names[role]),
                          "blocked": blocked, "accepted": 4 * len(names[role]) - blocked}
    return {"description": "All KB field names, with zero/one leading and trailing ASCII spaces; synthetic EOS boundary; not executable programs",
            "fields": sum(len(names[role]) for role in ROLES),
            "probes": sum(value["probes"] for value in per_role.values()),
            "blocked": len(failures), "by_role": per_role, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/thesis_direction_review/schema_constraint_coverage.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tokenizer_path = args.tokenizer.resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError("--tokenizer must name an existing local directory")
    started = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(root))
    from transformers import AutoTokenizer
    from experiments.condition_consistency.executor import SIGNATURES, parse_program
    from experiments.condition_consistency.schema_constraints import SchemaFieldConstraint

    paths = {name: root / relative for name, relative in INPUT_PATHS.items()}
    identities = {name: fingerprint(path, root) for name, path in paths.items()}
    manifest = json.loads(paths["checkpoint_manifest"].read_text(encoding="utf-8"))
    verified_tokenizer = {}
    for name in (*TOKENIZER_FILES, "config.json"):
        path = tokenizer_path / name
        expected = manifest["files"][name]
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise ValueError(f"Checkpoint identity mismatch: {name}")
        verified_tokenizer[name] = actual
    config = json.loads((tokenizer_path / "config.json").read_text(encoding="utf-8"))
    decoder_start = config["decoder_start_token_id"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                              trust_remote_code=False)
    if config["bos_token_id"] != tokenizer.bos_token_id or config["eos_token_id"] != tokenizer.eos_token_id:
        raise ValueError("Checkpoint model/tokenizer BOS/EOS mismatch")
    for delimiter in ("<arg>", "<func>"):
        encoded = tokenizer.encode(delimiter, add_special_tokens=False)
        if len(encoded) != 1 or delimiter in tokenizer.all_special_tokens:
            raise ValueError(f"Expected ordinary, atomic delimiter: {delimiter}")
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    names = {role: schema["names_by_role"][role] for role in ROLES}
    if sum(map(len, names.values())) != 1267 or any(len(set(value)) != len(value) for value in names.values()):
        raise ValueError("Expected the fixed 1,267 role-specific KB fields without duplicates")
    constraint = SchemaFieldConstraint(tokenizer, names, decoder_start_token_id=decoder_start)

    train = read_rows(paths["generator_train"], 5000)
    dev = read_rows(paths["dev_gold"], 500)
    train_ids, dev_ids = {row["id"] for row in train}, {row["id"] for row in dev}
    if train_ids & dev_ids:
        raise ValueError("Training/development IDs overlap")
    groups = {
        "generator_train_gold": ([{"id": row["id"], "text": text_of_gold(row)} for row in train], True),
        "development_gold": ([{"id": row["id"], "text": text_of_gold(row)} for row in dev], True),
    }
    for name, width in (("dev_beam4", 4), ("dev_beam8", 8)):
        rows = read_rows(paths[name], 500)
        if {row["id"] for row in rows} != dev_ids:
            raise ValueError(f"{name}: candidate IDs differ from development gold")
        items = []
        for row in rows:
            candidates = [candidate for candidate in row["candidates"]
                          if name != "dev_beam4" or candidate.get("origin") == "original"]
            if len(candidates) != width or sorted(candidate["rank"] for candidate in candidates) != list(range(width)):
                raise ValueError(f"{name}: unexpected candidate count/ranks for {row['id']}")
            for candidate in candidates:
                items.append({"id": row["id"], "rank": candidate["rank"],
                              "text": candidate["program_text"],
                              "hit_generation_limit": candidate.get("hit_generation_limit", False)})
        groups[name] = (items, False)
    name_sets = {role: set(values) for role, values in names.items()}
    coverage = {name: audit_group(items, constraint, tokenizer, decoder_start,
                                 schema["field_argument_roles"], SIGNATURES, parse_program,
                                 name_sets,
                                 is_gold=is_gold)
                for name, (items, is_gold) in groups.items()}
    lexicon = audit_lexicon(names, constraint, tokenizer, decoder_start)
    token_path_probes = audit_adversarial_token_paths(constraint, tokenizer, decoder_start)
    code_paths = [Path(__file__).resolve(),
                  root / "experiments/condition_consistency/schema_constraints.py",
                  root / "experiments/condition_consistency/executor.py"]
    passed = token_path_probes["all_passed"] and lexicon["blocked"] == 0 and all(
        coverage[name]["blocked"] == 0 and coverage[name]["syntax_parse_failures"] == 0
        for name in ("generator_train_gold", "development_gold"))
    result = {
        "audit": "schema_field_constraint_tokenizer_replay_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU-only fixed train/development coverage; no official validation data, model loading, generation, training, execution or answer scoring",
        "limitations": [
            "Coverage and syntactic admissibility only; not accuracy, efficacy, novelty, or permission to start training.",
            "Existing candidate text is retokenized; original model-generated token IDs are unavailable here.",
            "BOS/text/EOS targets are replayed after checkpoint decoder-start; EOS is synthetic for truncated cached candidates.",
            "Forced-EOS, max-length and beam-search runtime interactions are not exercised by this audit.",
            "Only recognized relation/attribute/qualifier slots are constrained; unknown functions and other slots may remain unrestricted.",
            "All rows remain in the report; no field is added to the KB lexicon from a gold program.",
            "Programs rejected by the field constraint may also fail syntax; counts are reported separately and overlap.",
            "Out-of-vocabulary counts use the fixed KB role lexicons and lexical slots, not execution or answer labels; empty/malformed field arguments can also count as out of vocabulary.",
        ],
        "future_generation_requirements": {
            "forced_eos": "SchemaFieldConstraint.generation_kwargs rejects a non-None forced_eos_token_id; disable it explicitly in BOTH comparison arms and record the protocol change before any future generation.",
            "truncation": "Unfinished maximum-length outputs must remain failures; this replay does not exercise generation stopping.",
            "token_paths": "Canonical tokenizer encodings with zero/one ASCII boundary spaces only; not arbitrary BPE segmentation or whitespace.",
        },
        "inputs": identities, "code": [fingerprint(path, root) for path in code_paths],
        "tokenizer": {"manifest_verified_files": verified_tokenizer,
                      "tokenizer_class": type(tokenizer).__name__, "vocabulary_size": len(tokenizer),
                      "decoder_start_token_id": decoder_start, "bos_token_id": tokenizer.bos_token_id,
                      "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id},
        "schema": {"source_kb_sha256": schema["source_kb_sha256"],
                   "field_counts_by_role": {role: len(names[role]) for role in ROLES},
                   "excluded_slots": ["concept", "entity", "value", "direction", "comparator"],
                   "function_grammar_constrained": False},
        "coverage": coverage, "all_lexicon_space_variants": lexicon,
        "adversarial_token_path_probes": token_path_probes,
        "gold_and_lexicon_coverage_passed": passed,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "transformers": importlib.metadata.version("transformers"),
                    "tokenizers": importlib.metadata.version("tokenizers"),
                    "device": "cpu", "model_loaded": False, "gpu_used": False,
                    "seconds": time.perf_counter() - started},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "coverage_passed": passed,
                      "blocked": {name: stats["blocked"] for name, stats in coverage.items()},
                      "lexicon_blocked": lexicon["blocked"],
                      "adversarial_token_paths_passed": token_path_probes["all_passed"],
                      "seconds": result["runtime"]["seconds"]}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
