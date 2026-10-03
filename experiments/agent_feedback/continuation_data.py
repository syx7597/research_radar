"""Build paired A/C continuation data with exactly matched supervised-token budgets.

A receives clean suffixes; C receives replay-verified recovery suffixes. Half of
the one-success-epoch token budget goes to these pairs, and the rest to identical
ordinary successes. Only labels are masked to meet a budget; input context is
never truncated. Repeated records are explicitly counted, not independent data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

from .environment import prompt_messages, tool_schemas
from .training_data import read_jsonl, tokenize_trajectory

SEED = 20261003
MAX_LENGTH = 8192


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_trajectory(row):
    messages, mask = row.get("messages"), row.get("supervise")
    if not isinstance(messages, list) or not isinstance(mask, list) or len(messages) != len(mask):
        raise ValueError("Trajectory requires aligned messages and explicit supervision")
    if len(messages) < 3 or messages[:2] != prompt_messages(messages[1].get("content")):
        raise ValueError("Trajectory requires the canonical question-only prompt")
    if any(type(flag) is not bool for flag in mask) or not any(mask):
        raise ValueError("Supervision mask must contain booleans and at least one target")
    if any(flag and message.get("role") != "assistant" for message, flag in zip(messages, mask)):
        raise ValueError("Only assistant messages may be supervised")
    if row.get("correct") is not True:
        raise ValueError("Only replay-verified correct target trajectories are accepted")


def match_pairs(success, clean, recovery, allowed_ids):
    """Validate TRAIN-only identity/lineage before selecting or tokenizing anything."""
    success_by_id = {}
    for row in success:
        validate_trajectory(row)
        if row["id"] in success_by_id or row["id"] not in allowed_ids:
            raise ValueError("Success IDs must be unique members of the formal training split")
        if row.get("source") != "gold_success":
            raise ValueError("Ordinary data must be verified gold_success trajectories")
        success_by_id[row["id"]] = row
    if not success_by_id:
        raise ValueError("No successful training trajectories")

    by_arm = []
    for rows, expected_source in [(clean, "paired_clean_suffix"), (recovery, "real_failure_recovery")]:
        by_pair = {}
        for row in rows:
            validate_trajectory(row)
            pair_id = row.get("pair_id")
            if pair_id != row["id"] or pair_id in by_pair or pair_id not in success_by_id:
                raise ValueError("Pair IDs must be unique verified-success training question IDs")
            if row.get("source") != expected_source or row.get("selection", {}).get("status") != "accepted":
                raise ValueError("Pairs must come from the replay-verified real-failure builder")
            if row["messages"][:2] != success_by_id[pair_id]["messages"][:2]:
                raise ValueError("Paired trajectory question differs from the training question")
            by_pair[pair_id] = row
        by_arm.append(by_pair)
    if not by_arm[0] or set(by_arm[0]) != set(by_arm[1]):
        raise ValueError("Clean and recovery files require identical nonempty pair ID sets")
    # Sort by ID before seeded sampling, so incidental input-file ordering is not
    # a hidden source of differences between the two arms.
    return [success_by_id[k] for k in sorted(success_by_id)], [
        {"source_id": key, "clean": by_arm[0][key], "recovery": by_arm[1][key]}
        for key in sorted(by_arm[0])]


def cap_supervision(row, cap):
    """Retain the chronologically first cap targets without modifying input IDs."""
    ids, labels = row["input_ids"], row["labels"]
    if len(ids) != len(labels) or any(label != -100 and label != token for token, label in zip(ids, labels)):
        raise ValueError("Causal labels must align with input tokens")
    available = sum(label != -100 for label in labels)
    if row.get("supervised_tokens") != available or type(cap) is not int or not 0 < cap <= available:
        raise ValueError("Invalid supervision cap or source token count")
    kept, last = 0, -1
    new_labels = []
    for index, label in enumerate(labels):
        if label != -100 and kept < cap:
            new_labels.append(label)
            kept += 1
            last = index
        else:
            new_labels.append(-100)
    result = {"input_ids": list(ids), "labels": new_labels, "supervised_tokens": cap,
              "supervised_tokens_before_cap": available, "masked_by_budget": available - cap,
              "spans": [[turn, start, min(end, last + 1)] for turn, start, end in row.get("spans", [])
                        if start < min(end, last + 1)]}
    if sum(x != -100 for x in result["labels"]) != cap or result["input_ids"] != ids:
        raise AssertionError("Supervision cap changed the input or failed its exact budget")
    return result


def select_budget(pool, budget, kind, rng):
    """Shuffle deterministically on each pass; label every repeated occurrence."""
    if not pool or budget < 1:
        raise ValueError("Each half-budget requires a nonempty pool and positive target")
    remaining, cycle, selected = budget, 0, []
    while remaining:
        order = list(range(len(pool)))
        rng.shuffle(order)
        for index in order:
            item = pool[index]
            available = min(item["A"]["supervised_tokens"], item["C"]["supervised_tokens"])
            if available < 1:
                raise ValueError("Selected trajectory has no supervised tokens")
            cap = min(available, remaining)
            selected.append({"kind": kind, "pool_index": index, "cycle": cycle,
                             "source_id": item["source_id"], "supervised_token_cap": cap})
            remaining -= cap
            if not remaining:
                break
        cycle += 1
    return selected


def build_schedule(ordinary, pairs):
    total = sum(row["A"]["supervised_tokens"] for row in ordinary)
    if total < 2:
        raise ValueError("One success epoch must provide at least two supervised tokens")
    paired_budget = total // 2
    common_budget = total - paired_budget
    rng = random.Random(SEED)
    selected = select_budget(ordinary, common_budget, "ordinary", rng)
    selected += select_budget(pairs, paired_budget, "paired_suffix", rng)
    rng.shuffle(selected)
    return selected, {"total": total, "ordinary": common_budget, "paired_suffix": paired_budget,
                      "target_paired_fraction": 0.5, "actual_paired_fraction": paired_budget / total,
                      "odd_token_policy": "if total is odd, ordinary receives the single extra token"}


def materialize_pair(entry, position, ordinary, pairs):
    source = (ordinary if entry["kind"] == "ordinary" else pairs)[entry["pool_index"]]
    metadata = {"id": f"continuation:{position:07d}", "source_id": entry["source_id"],
                "kind": entry["kind"], "pool_cycle": entry["cycle"],
                "supervised_token_cap": entry["supervised_token_cap"]}
    result = tuple({**metadata, **cap_supervision(source[arm], entry["supervised_token_cap"])}
                   for arm in ("A", "C"))
    if result[0]["supervised_tokens"] != result[1]["supervised_tokens"]:
        raise AssertionError("Pair supervision differs")
    if entry["kind"] == "ordinary" and result[0] != result[1]:
        raise AssertionError("Ordinary examples must be identical in both arms")
    return result


def reuse_summary(schedule, kind, pool, formal_count):
    counts = Counter(row["source_id"] for row in schedule if row["kind"] == kind)
    available = len({row["source_id"] for row in pool})
    return {"available_unique_questions": available, "used_unique_questions": len(counts),
            "available_fraction_of_formal_train": available / formal_count,
            "used_fraction_of_formal_train": len(counts) / formal_count,
            "selected_records": sum(counts.values()),
            "records_per_used_question": sum(counts.values()) / len(counts),
            "max_records_per_question": max(counts.values()),
            "reuse_histogram": dict(sorted(Counter(counts.values()).items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--success", default="data/agent_feedback/train.success.jsonl")
    parser.add_argument("--clean", default="data/agent_feedback/recovery.clean.jsonl")
    parser.add_argument("--recovery", default="data/agent_feedback/recovery.recovery.jsonl")
    parser.add_argument("--pair-manifest", default="results/agent_feedback/recovery_pairs.json")
    parser.add_argument("--split-manifest", default="results/agent_feedback/split_manifest.json")
    parser.add_argument("--output-dir", default="data/agent_feedback/continuation")
    parser.add_argument("--manifest", default="results/agent_feedback/continuation_data.json")
    args = parser.parse_args()
    outputs = {arm: Path(args.output_dir) / f"{arm}.jsonl" for arm in ("A", "C")}
    manifest_path = Path(args.manifest)
    if any(path.exists() for path in [*outputs.values(), manifest_path]):
        raise FileExistsError("Continuation artifacts already exist; do not overwrite")
    split = json.loads(Path(args.split_manifest).read_text())
    lineage = json.loads(Path(args.pair_manifest).read_text())
    if lineage["sources"]["split_manifest_sha256"] != digest(args.split_manifest):
        raise ValueError("Recovery pairs were built against a different split manifest")
    for arm, path in (("clean", args.clean), ("recovery", args.recovery)):
        if lineage["artifacts"][arm]["sha256"] != digest(path):
            raise ValueError("Recovery pair file differs from its verified manifest")
    if lineage["sources"]["gold_sha256"] != split["splits"]["train"]["gold"]["sha256"]:
        raise ValueError("Recovery pair gold is not the formal training gold")
    allowed_ids = set(split["splits"]["train"]["ids"])
    success, raw_pairs = match_pairs(read_jsonl(args.success), read_jsonl(args.clean),
                                     read_jsonl(args.recovery), allowed_ids)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ordinary = []
    for index, row in enumerate(success):
        encoded = tokenize_trajectory(tokenizer, row, MAX_LENGTH)
        ordinary.append({"source_id": row["id"], "A": encoded, "C": encoded})
        if (index + 1) % 500 == 0:
            print(json.dumps({"success_tokenized": index + 1}), flush=True)
    pairs, excluded = [], []
    for row in raw_pairs:
        lengths = {name: len(tokenizer.apply_chat_template(row[name]["messages"], tools=tool_schemas(),
                    tokenize=True, add_generation_prompt=False, return_dict=False))
                   for name in ("clean", "recovery")}
        if max(lengths.values()) > MAX_LENGTH:
            excluded.append({"id": row["source_id"], "reason": "context_exceeds_8192", "lengths": lengths})
            continue
        pairs.append({"source_id": row["source_id"],
                      "A": tokenize_trajectory(tokenizer, row["clean"], MAX_LENGTH),
                      "C": tokenize_trajectory(tokenizer, row["recovery"], MAX_LENGTH)})
    if not pairs:
        raise ValueError("No eligible real recovery pairs; cannot construct a recovery arm")
    schedule, budget = build_schedule(ordinary, pairs)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    totals = {arm: Counter() for arm in ("A", "C")}
    max_lengths = {arm: 0 for arm in ("A", "C")}
    # Manifest is written last and serves as the completion marker. An incomplete
    # output without that manifest must never enter training.
    with outputs["A"].open("x") as a_handle, outputs["C"].open("x") as c_handle:
        for position, entry in enumerate(schedule):
            pair = materialize_pair(entry, position, ordinary, pairs)
            for arm, row, handle in zip(("A", "C"), pair, (a_handle, c_handle)):
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                totals[arm].update(records=1, input_tokens=len(row["input_ids"]),
                    supervised_tokens=row["supervised_tokens"], masked_by_budget=row["masked_by_budget"],
                    **{entry["kind"] + "_supervised_tokens": row["supervised_tokens"]})
                max_lengths[arm] = max(max_lengths[arm], len(row["input_ids"]))
    for arm in ("A", "C"):
        if totals[arm]["supervised_tokens"] != budget["total"]:
            raise AssertionError("Written dataset differs from the prescribed token budget")
    manifest = {
        "purpose": "paired continuation training data; not evidence of model improvement",
        "seed": SEED, "max_context": MAX_LENGTH, "budget": budget,
        "supervision": "retain first cap supervised tokens in time order; cap=min(clean,recovery,pool budget remaining); keep entire input context",
        "pairing": "same source questions, record order, repeat counts and per-record supervision in A/C; ordinary records identical",
        "training_instruction": "one epoch over each pretokenized file; same seed, batch size and accumulation; do not independently resample or retokenize",
        "scope": "formal training only; no development or holdout data accessed",
        "synthetic_pairs": 0, "formal_training_questions": len(allowed_ids),
        "verified_success_questions": len(success), "raw_real_pairs": len(raw_pairs),
        "excluded_context_pairs": excluded,
        "real_recovery_coverage": reuse_summary(schedule, "paired_suffix", pairs, len(allowed_ids)),
        "ordinary_coverage": reuse_summary(schedule, "ordinary", ordinary, len(allowed_ids)),
        "combined_unique_training_questions": len({row["source_id"] for row in schedule}),
        "arms": {arm: {**dict(totals[arm]), "max_input_length": max_lengths[arm],
                        "path": str(outputs[arm]), "sha256": digest(outputs[arm])} for arm in ("A", "C")},
        "sources": {name: {"path": str(path), "sha256": digest(path)} for name, path in {
            "success": args.success, "clean": args.clean, "recovery": args.recovery,
            "pair_manifest": args.pair_manifest, "split_manifest": args.split_manifest,
            "builder": __file__, "tokenization": Path(__file__).with_name("training_data.py"),
            "environment": Path(__file__).with_name("environment.py")}.items()},
        "tokenizer": {"model_path": args.model, "transformers": importlib.metadata.version("transformers"),
            "files_sha256": {name: digest(Path(args.model) / name) for name in
                ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json") if (Path(args.model) / name).exists()}},
        "limitations": "Recovery records are reused when scarce. Repetition is not additional independent coverage; input tokens and compute are not matched.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
