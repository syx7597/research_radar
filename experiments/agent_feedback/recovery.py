"""Build replay-verified recovery/clean pairs from TRAIN-only free-running failures.

Reference divergence is a selection marker, not a semantic-error annotation.
Gold programs are used offline to construct training targets, never by inference.
No synthetic actions are inserted and no development/holdout rows are sampled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.condition_consistency.executor import KoPLExecutor, compare_answers, validate_program
from .environment import Episode, MAX_CALLS, call_message, compact, prompt_messages
from .inference import execute_response
from .training_data import read_jsonl


class Excluded(Exception):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail
        super().__init__(reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_rollout(executor, rollout, gold, max_calls):
    """Verify the original free-run outcome, observations and event alignment."""
    messages, events = rollout.get("messages"), rollout.get("events")
    if not isinstance(messages, list) or not isinstance(events, list):
        raise Excluded("missing_rollout_messages_or_events")
    if messages[:2] != prompt_messages(gold["question"]):
        raise Excluded("rollout_prompt_mismatch")
    tail = messages[2:]
    if len(tail) != 2 * len(events) or rollout.get("calls") != len(events):
        raise Excluded("rollout_event_alignment_mismatch")
    episode, turns = Episode(executor, max_calls=max_calls), []
    for index, event in enumerate(events):
        assistant, tool = tail[index * 2:index * 2 + 2]
        if (not isinstance(assistant, dict) or not isinstance(tool, dict) or
                assistant.get("role") != "assistant" or tool.get("role") != "tool" or
                not isinstance(assistant.get("content"), str) or
                not isinstance(tool.get("content"), str)):
            raise Excluded("rollout_message_format_mismatch")
        if episode.done:
            raise Excluded("rollout_continued_after_termination")
        observation = execute_response(episode, assistant["content"])
        try:
            recorded_observation = json.loads(tool["content"])
        except (ValueError, TypeError):
            raise Excluded("rollout_observation_not_json") from None
        if recorded_observation != observation or episode.events[-1] != event:
            raise Excluded("rollout_replay_mismatch", {"call_index": index})
        turns.append({"message": deepcopy(assistant), "event": deepcopy(event)})
    if episode.prediction != rollout.get("prediction"):
        raise Excluded("rollout_prediction_mismatch")
    return turns, episode


def reference_calls(executor, gold, max_calls):
    try:
        program = validate_program(gold["program"])
    except (ValueError, TypeError, KeyError) as exc:
        raise Excluded("invalid_reference_program", type(exc).__name__) from None
    calls = [{"tool": "step", "arguments": action} for action in program]
    calls.append({"tool": "finish", "arguments": {"answer_handle": len(program) - 1}})
    episode = Episode(executor, max_calls=max_calls)
    for index, call in enumerate(calls):
        observation = execute_response(episode, call_message(call["tool"], call["arguments"])["content"])
        if not observation["ok"]:
            raise Excluded("reference_execution_failed", {"call_index": index})
    if not compare_answers(gold["answer"], episode.prediction):
        raise Excluded("reference_answer_mismatch")
    return calls, episode.events


def append_action(messages, supervise, episode, text, train):
    observation = execute_response(episode, text)
    messages.extend([{"role": "assistant", "content": text},
                     {"role": "tool", "content": compact(observation)}])
    supervise.extend([train, False])
    return observation


def continuation(executor, gold, turns, calls, reference_events, divergence, insert_divergence, max_calls):
    episode = Episode(executor, max_calls=max_calls)
    messages, supervise = prompt_messages(gold["question"]), [False, False]
    mapping = {}
    for index in range(divergence):
        observation = append_action(messages, supervise, episode,
                                    turns[index]["message"]["content"], False)
        if observation != turns[index]["event"]["observation"]:
            raise Excluded("shared_prefix_replay_mismatch")
        if calls[index]["tool"] == "step":
            mapping[index] = observation["handle"]
    if insert_divergence:
        observation = append_action(messages, supervise, episode,
                                    turns[divergence]["message"]["content"], False)
        if observation != turns[divergence]["event"]["observation"]:
            raise Excluded("divergence_replay_mismatch")
        if episode.done:
            raise Excluded("divergence_terminated_episode" if episode.prediction is not None
                           else "insufficient_call_budget")
    remaining = len(calls) - divergence
    if episode.calls + remaining > max_calls:
        raise Excluded("insufficient_call_budget")
    for index in range(divergence, len(calls)):
        original = calls[index]
        arguments = deepcopy(original["arguments"])
        if original["tool"] == "step":
            arguments["dependencies"] = [mapping[x] for x in arguments["dependencies"]]
        else:
            arguments["answer_handle"] = mapping[arguments["answer_handle"]]
        observation = append_action(messages, supervise, episode,
                                    call_message(original["tool"], arguments)["content"], True)
        if not observation["ok"]:
            raise Excluded("correct_suffix_execution_failed", {"reference_call_index": index})
        expected = deepcopy(reference_events[index]["observation"])
        if original["tool"] == "step":
            mapping[index] = observation["handle"]
            expected["handle"] = observation["handle"]
        if observation != expected:
            raise Excluded("correct_suffix_observation_mismatch", {"reference_call_index": index})
    if not episode.done or not compare_answers(gold["answer"], episode.prediction):
        raise Excluded("correct_suffix_answer_mismatch")
    messages.append({"role": "assistant", "content": "Done."})
    supervise.append(True)
    return {"id": gold["id"], "pair_id": gold["id"], "messages": messages, "supervise": supervise,
            "prediction": episode.prediction, "correct": True, "calls": episode.calls,
            "source": "real_failure_recovery" if insert_divergence else "paired_clean_suffix",
            "reference_handle_map": {str(k): v for k, v in mapping.items()},
            "events": episode.events}


def make_pair(executor, rollout, gold, max_calls=MAX_CALLS, min_shared_prefix=1):
    """Return a verified pair or an explicit exclusion; never infer error from emptiness."""
    if rollout.get("id") != gold.get("id"):
        raise ValueError("Rollout and reference IDs differ")
    if min_shared_prefix < 1:
        raise ValueError("At least one shared successful reference call is required")
    audit = {"id": gold["id"], "selection_label": "first_reference_divergence_in_failed_rollout",
             "semantic_error_at_divergence": "not_established"}
    try:
        turns, source_episode = replay_rollout(executor, rollout, gold, max_calls)
        failed = not compare_answers(gold["answer"], source_episode.prediction)
        audit["free_run_failed"] = failed
        if not failed:
            raise Excluded("free_run_answer_correct")
        calls, reference_events = reference_calls(executor, gold, max_calls)
        divergence = 0
        for turn, reference in zip(turns, calls):
            event = turn["event"]
            if (not event["observation"]["ok"] or event["tool"] != reference["tool"] or
                    event["arguments"] != reference["arguments"]):
                break
            divergence += 1
        audit["shared_prefix_calls"] = divergence
        if divergence < min_shared_prefix:
            raise Excluded("no_shared_successful_prefix")
        if divergence == len(turns):
            raise Excluded("no_reference_divergence_before_stop")
        if divergence >= len(calls):
            raise Excluded("reference_already_completed")
        event = turns[divergence]["event"]
        audit.update(divergence_call_index=divergence,
                     divergence_executable=bool(event["observation"]["ok"]),
                     divergence_tool=event["tool"],
                     divergence_empty_entities=bool(event["observation"]["ok"] and
                                                    event["observation"].get("count") == 0),
                     divergence_error=event["observation"].get("error"),
                     source_rollout_calls=source_episode.calls)
        recovery = continuation(executor, gold, turns, calls, reference_events, divergence, True, max_calls)
        clean = continuation(executor, gold, turns, calls, reference_events, divergence, False, max_calls)
        # Target functions, literals and dependency graph match the reference.
        # Handle numerals may differ; exact token balancing is a later step.
        audit.update(status="accepted", supervised_reference_calls=len(calls) - divergence,
                     recovery_calls=recovery["calls"], clean_calls=clean["calls"])
        for row in (recovery, clean):
            row["selection"] = deepcopy(audit)
        return {"recovery": recovery, "clean": clean, "audit": audit}
    except Excluded as exc:
        audit.update(status="excluded", reason=exc.reason)
        if exc.detail is not None:
            audit["detail"] = exc.detail
        return {"audit": audit}


def validate_training_sources(gold_rows, rollouts, split_manifest, gold_path):
    train = split_manifest["splits"]["train"]
    allowed = set(train["ids"])
    if sha256(gold_path) != train["gold"]["sha256"]:
        raise ValueError("Recovery requires the exact frozen formal-training gold file")
    by_id = {row["id"]: row for row in gold_rows}
    if len(by_id) != len(gold_rows) or set(by_id) != allowed:
        raise ValueError("Training gold IDs differ from frozen formal-training IDs")
    ids = [row["id"] for row in rollouts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate free-run rollout IDs")
    if not set(ids) <= allowed:
        raise ValueError("Development/holdout or unknown IDs are forbidden in recovery construction")
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", default="data/agent_feedback/train.gold.jsonl")
    parser.add_argument("--split-manifest", default="results/agent_feedback/split_manifest.json")
    parser.add_argument("--kb", default="datasets/kqa_pro/kb.json")
    parser.add_argument("--output-prefix", default="data/agent_feedback/recovery")
    parser.add_argument("--manifest", default="results/agent_feedback/recovery_pairs.json")
    args = parser.parse_args()
    paths = {name: Path(args.output_prefix + suffix) for name, suffix in
             (("recovery", ".recovery.jsonl"), ("clean", ".clean.jsonl"), ("audit", ".audit.jsonl"))}
    manifest_path = Path(args.manifest)
    if any(path.exists() for path in [*paths.values(), manifest_path]):
        raise FileExistsError("Recovery artifacts already exist; do not overwrite a frozen pair set")
    rollouts, gold_rows = read_jsonl(args.predictions), read_jsonl(args.gold)
    if not rollouts:
        raise ValueError("No free-run training rollouts supplied")
    split_manifest = json.loads(Path(args.split_manifest).read_text())
    gold = validate_training_sources(gold_rows, rollouts, split_manifest, args.gold)
    executor = KoPLExecutor(args.kb)
    counts = Counter({key: 0 for key in ("rollouts_supplied", "rollouts_replay_verified",
                                       "free_run_failed", "free_run_correct", "accepted_pairs")})
    exclusions = Counter()
    legality, accepted_legality, prefix_lengths = Counter(), Counter(), Counter()
    rows = {name: [] for name in paths}
    for rollout in rollouts:
        pair = make_pair(executor, rollout, gold[rollout["id"]])
        audit = pair["audit"]
        rows["audit"].append(audit)
        counts["rollouts_supplied"] += 1
        if "free_run_failed" in audit:
            counts["rollouts_replay_verified"] += 1
            counts["free_run_failed"] += audit["free_run_failed"]
            counts["free_run_correct"] += not audit["free_run_failed"]
        if "divergence_executable" in audit:
            legality["executable" if audit["divergence_executable"] else "rejected"] += 1
        if audit["status"] == "accepted":
            for name in ("recovery", "clean"):
                rows[name].append(pair[name])
            counts["accepted_pairs"] += 1
            accepted_legality["executable" if audit["divergence_executable"] else "rejected"] += 1
            prefix_lengths[str(audit["shared_prefix_calls"])] += 1
        else:
            exclusions[audit["reason"]] += 1
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            for row in rows[name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "offline training-pair construction; not model improvement or recovery accuracy",
        "selection_label": "first_reference_divergence_in_failed_rollout",
        "semantic_error_at_divergence": "not_established",
        "scope": "frozen formal-training IDs only; no synthetic samples; no dev/holdout sampling",
        "counts": dict(counts), "exclusions": dict(exclusions),
        "candidate_divergence_execution": dict(legality),
        "accepted_divergence_execution": dict(accepted_legality),
        "accepted_shared_prefix_call_histogram": dict(prefix_lengths),
        "coverage": {"formal_training_questions": len(gold),
                     "free_run_training_question_fraction": len(rollouts) / len(gold),
                     "accepted_pair_fraction_of_all_supplied_rollouts": counts["accepted_pairs"] / len(rollouts),
                     "accepted_pair_fraction_of_replay_verified_failed_rollouts":
                         counts["accepted_pairs"] / counts["free_run_failed"] if counts["free_run_failed"] else None},
        "supervision": "same reference suffix functions, literals and dependency graph; preceding shared actions, real divergent action and all tools masked; Done. supervised in both. Handle IDs may differ. Token budgets are not yet matched.",
        "sources": {"predictions_sha256": sha256(args.predictions), "gold_sha256": sha256(args.gold),
                    "split_manifest_sha256": sha256(args.split_manifest), "kb_sha256": sha256(args.kb),
                    "builder_sha256": sha256(__file__),
                    "environment_sha256": sha256(Path(__file__).with_name("environment.py")),
                    "inference_sha256": sha256(Path(__file__).with_name("inference.py"))},
        "artifacts": {name: {"path": str(path), "count": len(rows[name]), "sha256": sha256(path)}
                      for name, path in paths.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
