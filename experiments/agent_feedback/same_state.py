"""Same-state diagnostic primitives, separate from the frozen training pipeline.

Offline construction may read references. Resume accepts only actual history,
state digests and budgets: never a reference program, answer or target suffix.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import random

from experiments.condition_consistency.executor import KoPLExecutor
from experiments.condition_consistency.prepare_data import EXPECTED_SHA256, make_splits
from .environment import Episode, MAX_CALLS, compact, prompt_messages, tool_schemas
from .inference import execute_response, response_tokens
from .recovery import make_pair, sha256
from .training_data import read_jsonl

SEED = 20261018
HASH_SEED = "20261003"
POOL_SIZE = 2400
FIELDS = {"id", "condition", "messages", "events", "prefix_calls", "remaining_calls",
          "remaining_generated_tokens", "state_sha256"}
CONDITIONS = ("clean", "divergent")


def canonical(value):
    """Preserve full internal values, ordering and types, not bounded display text."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [canonical(x) for x in value]}
    if isinstance(value, list):
        return [canonical(x) for x in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Unsupported non-string state key")
        return {k: canonical(v) for k, v in sorted(value.items())}
    if type(value).__name__ == "ValueClass" and set(vars(value)) == {"type", "value", "unit"}:
        return {"kind": "ValueClass", "fields": canonical(vars(value))}
    raise ValueError(f"Unsupported state value: {type(value).__name__}")


def object_sha(value):
    return hashlib.sha256(json.dumps(canonical(value), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_sha(episode):
    return object_sha({"calls": episode.calls, "done": episode.done,
                       "prediction": episode.prediction, "selected": episode.selected,
                       "handles": [vars(handle) for handle in episode.handles],
                       "events": episode.events})


def replay_prefix(executor, messages, events, max_calls):
    if (len(messages) != 2 + 2 * len(events) or len(messages) < 4
            or messages[:2] != prompt_messages(messages[1].get("content"))):
        raise ValueError("Prefix prompt/turn alignment differs")
    episode = Episode(executor, max_calls=max_calls)
    for index, expected in enumerate(events):
        assistant, tool = messages[2 + 2 * index:4 + 2 * index]
        if episode.done or assistant.get("role") != "assistant" or tool.get("role") != "tool":
            raise ValueError("Cannot resume a terminal or malformed prefix")
        actual = execute_response(episode, assistant["content"])
        if actual != json.loads(tool["content"]) or episode.events[-1] != expected:
            raise ValueError("Actual replay differs from saved observation/event")
    if episode.done:
        raise ValueError("Cannot resume a terminal prefix")
    return episode


def restore(executor, row):
    if set(row) != FIELDS or row["condition"] not in CONDITIONS:
        raise ValueError("Resume accepts only the registered state schema; no reference targets")
    if (type(row["prefix_calls"]) is not int or row["prefix_calls"] != len(row["events"])
            or type(row["remaining_calls"]) is not int or row["remaining_calls"] < 1
            or row["prefix_calls"] + row["remaining_calls"] > MAX_CALLS
            or row["remaining_generated_tokens"] != 2048):
        raise ValueError("Invalid remaining budget")
    episode = replay_prefix(executor, row["messages"], row["events"],
                            row["prefix_calls"] + row["remaining_calls"])
    if state_sha(episode) != row["state_sha256"]:
        raise ValueError("Full internal handle state differs, even if displayed observations match")
    return episode


def fixture_pair(executor, pair):
    """Trim verified target suffix BEFORE writing inference inputs."""
    if pair["audit"]["status"] != "accepted":
        raise ValueError("Only replay-verified pairs can provide diagnostic states")
    k = pair["audit"]["shared_prefix_calls"]
    remaining = MAX_CALLS - k - 1
    states = []
    for condition, name, calls in (("clean", "clean", k), ("divergent", "recovery", k + 1)):
        target = pair[name]
        messages, events = deepcopy(target["messages"][:2 + 2 * calls]), deepcopy(target["events"][:calls])
        episode = replay_prefix(executor, messages, events, calls + remaining)
        row = {"id": target["id"], "condition": condition, "messages": messages,
               "events": events, "prefix_calls": calls, "remaining_calls": remaining,
               "remaining_generated_tokens": 2048, "state_sha256": state_sha(episode)}
        restored = restore(executor, row)
        if restored.max_calls - restored.calls != remaining:
            raise AssertionError("Remaining calls changed during restore")
        # Execute the registered reference suffix only as an offline budget probe.
        for event in target["events"][calls:]:
            from .environment import call_message
            observation = execute_response(restored, call_message(event["tool"], event["arguments"])["content"])
            if restored.events[-1] != event or observation != event["observation"]:
                raise ValueError("Reference suffix cannot replay under the matched remaining budget")
        if restored.prediction != target["prediction"]:
            raise ValueError("Reference suffix final answer changed")
        states.append(row)
    if states[1]["messages"][:-2] != states[0]["messages"]:
        raise ValueError("Conditions do not share the identical clean history")
    return states


def build_states(executor, rollouts, gold):
    if [r["id"] for r in rollouts] != [r["id"] for r in gold] or len({r["id"] for r in gold}) != len(gold):
        raise ValueError("Full ordered source coverage required; never skip missing rollouts")
    states, audits, reasons = [], [], Counter()
    for rollout, reference in zip(rollouts, gold):
        pair = make_pair(executor, rollout, reference)
        audit = pair["audit"]
        audits.append(audit)
        reasons[audit.get("reason", "accepted")] += 1
        if audit["status"] == "accepted":
            states.extend(fixture_pair(executor, pair))
    return states, audits, dict(reasons)


def write_new(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        if jsonl:
            handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value))
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def freeze_pool(root):
    output = root / "results/agent_feedback/same_state_v1/pool_manifest.json"
    if output.exists():
        raise FileExistsError("Question pool is already frozen")
    for name, expected in EXPECTED_SHA256.items():
        if sha256(root / "datasets/kqa_pro" / name) != expected:
            raise ValueError("Source dataset differs")
    paths = [root / p for p in ("results/condition_consistency/data_check/split_manifest.json",
             "results/condition_consistency/query_repair/split_manifest.json", "results/agent_feedback/split_manifest.json")]
    pilot, historical, agent = [json.loads(p.read_text()) for p in paths]
    rows = json.loads((root / "datasets/kqa_pro/train.json").read_text())
    val = json.loads((root / "datasets/kqa_pro/val.json").read_text())
    pool, dedup = make_splits(rows, val, pilot["seed"],
                             {"all": pilot["deduplication"]["eligible_component_representatives"]})
    used = {i for split in pilot["splits"].values() for i in split["indices"]}
    for manifest in (historical, agent):
        used.update(int(ident.split(":")[1]) for split in manifest["splits"].values() for ident in split["ids"])
    if not used <= set(pool["all"]):
        raise ValueError("Historical IDs no longer match component representatives")
    candidates = sorted(set(pool["all"]) - used)
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:POOL_SIZE]
    if len(selected) != POOL_SIZE:
        raise ValueError("Insufficient independent question components")
    gold = [{"id": f"train:{i}", "question": rows[i]["question"],
             "program": rows[i]["program"], "answer": rows[i]["answer"]} for i in selected]
    private = root / "data/agent_feedback/same_state_v1"
    question_path, gold_path = private / "pool.questions.jsonl", private / "pool.gold.jsonl"
    if question_path.exists() or gold_path.exists():
        raise FileExistsError("Partial pool artifacts exist; inspect, do not overwrite")
    write_new(question_path, [{k: row[k] for k in ("id", "question")} for row in gold], True)
    write_new(gold_path, gold, True)
    report = {"schema": "same_state_pool_v1", "date": "2026-10-18", "seed": SEED,
              "questions": POOL_SIZE, "ids": [row["id"] for row in gold], "deduplication": dedup,
              "excluded_historical_components": len(used), "historical_overlap": 0,
              "selection": "fixed shuffle of unused exact-question-or-program component representatives",
              "no_adaptive_expansion": True, "new_pool_model_results_seen": False,
              "gold_use": "copied only at freeze; later offline selection/scoring, never resume input",
              "source_sha256": EXPECTED_SHA256,
              "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in paths},
              "artifacts": {str(p.relative_to(root)): sha256(p) for p in (question_path, gold_path)},
              "builder_sha256": sha256(__file__)}
    write_new(output, report)
    print(json.dumps({k: report[k] for k in ("questions", "excluded_historical_components", "historical_overlap")}))


def preflight(root):
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        raise ValueError("Use the original executor hash seed 20261003")
    predictions = root / "results/agent_feedback/initial_dev_v1.jsonl"
    gold = root / "data/agent_feedback/dev.gold.jsonl"
    out = root / "results/agent_feedback/same_state_v1/dev_replay_preflight.json"
    if out.exists():
        raise FileExistsError("Preflight already exists")
    executor = KoPLExecutor(root / "datasets/kqa_pro/kb.json")
    states, audits, counts = build_states(executor, read_jsonl(predictions), read_jsonl(gold))
    path = root / "data/agent_feedback/same_state_v1/dev.states.jsonl"
    write_new(path, states, True)
    write_new(out, {"schema": "same_state_development_replay_v1", "source_questions": len(audits),
        "accepted_pairs": len(states) // 2, "counts": counts, "state_restore_and_suffix_call_budget_passed": True,
        "token_budget_checked": False, "gpu_inference_performed": False, "development_only": True,
        "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in (predictions, gold)},
        "state_file": str(path.relative_to(root)), "state_file_sha256": sha256(path), "code_sha256": sha256(__file__)})
    print(json.dumps(counts))


def token_check(executor, tokenizer, states, gold, rollouts, max_context=8192):
    """Check prompt space and exact target action token budgets on OLD dev states."""
    refs, sources = {r["id"]: r for r in gold}, {r["id"]: r for r in rollouts}
    checks = []
    for row in states:
        restore(executor, row)
        pair = make_pair(executor, sources[row["id"]], refs[row["id"]])
        target = pair["clean" if row["condition"] == "clean" else "recovery"]
        prefix_len = len(row["messages"])
        prompts, targets = [], []
        for i in range(prefix_len, len(target["messages"]) - 1, 2):
            before = tokenizer.apply_chat_template(target["messages"][:i], tools=tool_schemas(),
                tokenize=True, return_dict=False, add_generation_prompt=True)
            after = tokenizer.apply_chat_template(target["messages"][:i+1], tools=tool_schemas(),
                tokenize=True, return_dict=False, add_generation_prompt=False)
            if after[:len(before)] != before:
                raise ValueError("Chat template changes previous token boundaries")
            prompts.append(len(before)); targets.append(len(after) - len(before))
        checks.append({"id": row["id"], "condition": row["condition"],
            "initial_prompt_tokens": prompts[0], "max_reference_prompt_tokens": max(prompts),
            "reference_generated_tokens": sum(targets), "max_reference_turn_tokens": max(targets),
            "within_budgets": max(p+t for p,t in zip(prompts, targets)) <= max_context
                and sum(targets) <= 2048 and max(targets) <= 192,
            "initial_prompt_ids_sha256": object_sha(tokenizer.apply_chat_template(row["messages"],
                tools=tool_schemas(), tokenize=True, return_dict=False, add_generation_prompt=True))})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-pool", "preflight"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    (freeze_pool if args.command == "freeze-pool" else preflight)(args.root)


if __name__ == "__main__":
    main()
