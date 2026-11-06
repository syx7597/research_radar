"""Domain success/recovery trajectories with structured answers and train gates.

CPU fixtures are explicitly marked and never admitted as model failures. The
production CLI requires a training split, completed generation inventory, model
and code hashes, full question coverage and per-rollout model provenance.
"""
import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path

from experiments.agent_feedback.recovery import Excluded, sha256
from .kopl_record_bridge import compact, prototype_queries, write_new
from .record_set_interface import RecordSetEpisode, execute_response, load_bridge, prompt_messages

REFERENCE_FIELDS = {"id", "question", "actions", "answer"}
ACTION_FIELDS = {"step": {"function", "inputs", "dependencies"},
                 "finish": {"answer_handle"}, "finish_records": {"record_handle"}}
ROOT = Path(__file__).resolve().parents[2]


def typed_json(value):
    """Exact typed JSON equality; mapping order is immaterial."""
    if value is None:
        return ("null",)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ("float", value)
    if type(value) is list:
        return ("list", tuple(typed_json(v) for v in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, typed_json(value[k])) for k in sorted(value)))
    raise ValueError("Expected finite typed JSON")


def answer_signature(answer):
    """Normalize only semantic set order, retaining complete records and sources."""
    if not isinstance(answer, dict) or answer.get("kind") not in {"records", "scalar"}:
        raise ValueError("A computed structured answer is required")
    field = "records" if answer["kind"] == "records" else "source_records"
    fields = {"kind", field, "sources", "input_entity_count", "unique_record_count", "scope"}
    if answer["kind"] == "scalar":
        fields.add("value")
    if set(answer) != fields:
        raise ValueError("Unexpected structured answer fields")
    data = deepcopy(answer)
    records = data[field]
    if (not isinstance(records, list) or not isinstance(data["sources"], dict)
            or type(data["input_entity_count"]) is not int
            or type(data["unique_record_count"]) is not int
            or data["unique_record_count"] != len(records)
            or data["input_entity_count"] < len(records)):
        raise ValueError("Record collection/count schema differs")
    ids = []
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
            raise ValueError("Complete identified source records required")
        ids.append(row["record_id"])
        if not {"subjects", "conditions_all", "value", "citation", "component", "attribute"} <= set(row):
            raise ValueError("Incomplete source record")
        for key in ("subjects", "conditions_all"):
            if not isinstance(row[key], list):
                raise ValueError("Invalid source set field")
            row[key] = sorted(row[key], key=compact)
        citation = row["citation"]
        if not isinstance(citation, dict) or not isinstance(citation.get("spans"), list):
            raise ValueError("Complete source citation required")
        citation["spans"] = sorted(citation["spans"], key=compact)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source-record identity")
    if set(data["sources"]) != {r["citation"]["chunk_id"] for r in records}:
        raise ValueError("Source mapping differs from selected records")
    data[field] = sorted(records, key=lambda r: r["record_id"])
    # This records how an execution reached the same set, not answer content.
    data.pop("input_entity_count")
    if data["kind"] == "scalar":
        scalar = data["value"]
        if not isinstance(scalar, dict) or "type" not in scalar or "value" not in scalar:
            raise ValueError("Typed scalar required")
        if scalar["type"] == "quantity":
            if set(scalar) != {"type", "value", "unit"} or type(scalar["value"]) not in (int, float):
                raise ValueError("Quantity requires numeric value and exact unit")
            number = Decimal(str(scalar["value"]))
            if not number.is_finite():
                raise ValueError("Non-finite quantity")
            scalar["value"] = {"decimal": str(number.normalize())}
    return typed_json(data)


def answers_equal(left, right):
    if left is None or right is None:
        return False
    try:
        return answer_signature(left) == answer_signature(right)
    except (ValueError, TypeError, KeyError):
        return False


def action_text(action):
    return "<tool_call>\n" + compact(action) + "\n</tool_call>"


def validate_actions(reference):
    if (set(reference) != REFERENCE_FIELDS or type(reference["id"]) is not str
            or not reference["id"] or not isinstance(reference["question"], str)):
        raise ValueError("Reference requires exactly id/question/actions/answer")
    actions = reference["actions"]
    if (not isinstance(actions, list) or not actions or not isinstance(actions[-1], dict)
            or actions[-1].get("name") not in {"finish", "finish_records"}):
        raise ValueError("Reference must end in an explicit scalar/set finish")
    handles = 0
    for i, action in enumerate(actions):
        if (not isinstance(action, dict) or set(action) != {"name", "arguments"}
                or action.get("name") not in ACTION_FIELDS
                or not isinstance(action["arguments"], dict)
                or set(action["arguments"]) != ACTION_FIELDS[action["name"]]):
            raise ValueError("Malformed reference action")
        name, args = action["name"], action["arguments"]
        if i < len(actions)-1 and name != "step":
            raise ValueError("Only final action may terminate")
        if name == "step":
            deps = args["dependencies"]
            if (not isinstance(args["function"], str) or not isinstance(args["inputs"], list)
                    or any(type(x) is not str for x in args["inputs"]) or not isinstance(deps, list)
                    or any(type(x) is not int or not 0 <= x < handles for x in deps)):
                raise ValueError("Reference dependency or literal schema differs")
            handles += 1
        else:
            handle = args["answer_handle" if name == "finish" else "record_handle"]
            if type(handle) is not int or not 0 <= handle < handles:
                raise ValueError("Final handle is not an existing reference result")
    answer_signature(reference["answer"])
    return actions


def append_action(episode, messages, supervise, text, train):
    observation = execute_response(episode, text)
    messages.extend([{"role": "assistant", "content": text},
                     {"role": "tool", "content": compact(observation)}])
    supervise.extend([train, False])
    return observation


def successful_trajectory(executor, store, reference, max_calls=24):
    actions = validate_actions(reference)
    if len(actions) > max_calls:
        raise Excluded("reference_exceeds_call_budget")
    episode = RecordSetEpisode(executor, store, max_calls)
    messages, supervise = prompt_messages(reference["question"]), [False, False]
    for action in actions:
        if episode.done:
            raise Excluded("reference_terminated_early")
        observation = append_action(episode, messages, supervise, action_text(action), True)
        if not observation["ok"]:
            raise Excluded("reference_execution_failed")
    if not episode.done or not answers_equal(episode.answer, reference["answer"]):
        raise Excluded("reference_answer_mismatch")
    messages.append({"role": "assistant", "content": "Done."})
    supervise.append(True)
    return {"id": reference["id"], "messages": messages, "events": episode.events,
            "supervise": supervise, "calls": episode.calls, "answer": episode.answer,
            "source": "executed_reference_success", "training_eligible": False}


def replay_rollout(executor, store, rollout, reference, max_calls):
    if rollout.get("id") != reference["id"]:
        raise Excluded("rollout_reference_id_mismatch")
    messages, events = rollout.get("messages"), rollout.get("events")
    if (not isinstance(messages, list) or not isinstance(events, list)
            or type(rollout.get("calls")) is not int or rollout["calls"] != len(events)
            or "answer" not in rollout or messages[:2] != prompt_messages(reference["question"])):
        raise Excluded("rollout_schema_or_prompt_mismatch")
    tail = messages[2:]
    has_done = bool(tail and tail[-1] == {"role": "assistant", "content": "Done."})
    if has_done:
        tail = tail[:-1]
    if len(tail) != 2 * len(events):
        raise Excluded("rollout_event_alignment_mismatch")
    episode, turns = RecordSetEpisode(executor, store, max_calls), []
    for i, event in enumerate(events):
        assistant, tool = tail[2*i:2*i+2]
        if (not isinstance(assistant, dict) or set(assistant) != {"role", "content"}
                or assistant["role"] != "assistant" or not isinstance(assistant["content"], str)
                or not isinstance(tool, dict) or set(tool) != {"role", "content"}
                or tool["role"] != "tool" or not isinstance(tool["content"], str)):
            raise Excluded("rollout_message_schema_mismatch")
        if episode.done:
            raise Excluded("rollout_continued_after_termination")
        observation = execute_response(episode, assistant["content"])
        try:
            recorded = json.loads(tool["content"])
            equal = (typed_json(recorded) == typed_json(observation)
                     and typed_json(event) == typed_json(episode.events[-1]))
        except (ValueError, TypeError):
            equal = False
        if not equal:
            raise Excluded("rollout_replay_mismatch", {"call_index": i})
        turns.append({"message": deepcopy(assistant), "event": deepcopy(event)})
    if (episode.calls != rollout["calls"]
            or typed_json(episode.answer) != typed_json(rollout["answer"])):
        raise Excluded("rollout_answer_or_call_mismatch")
    if has_done and episode.answer is None:
        raise Excluded("done_without_computed_answer")
    return episode, turns


def continuation(executor, store, reference, target, turns, k, recovery, max_calls):
    episode = RecordSetEpisode(executor, store, max_calls)
    messages, supervise = prompt_messages(reference["question"]), [False, False]
    mapping = {}
    for i in range(k):
        observation = append_action(episode, messages, supervise, turns[i]["message"]["content"], False)
        if typed_json(observation) != typed_json(turns[i]["event"]["observation"]):
            raise Excluded("shared_prefix_replay_mismatch")
        mapping[target["events"][i]["observation"]["handle"]] = observation["handle"]
    if recovery:
        observation = append_action(episode, messages, supervise, turns[k]["message"]["content"], False)
        if typed_json(observation) != typed_json(turns[k]["event"]["observation"]):
            raise Excluded("divergence_replay_mismatch")
        if episode.done:
            raise Excluded("divergence_terminated_episode" if episode.answer is not None else "insufficient_call_budget")
    actions = reference["actions"]
    if episode.calls + len(actions) - k > max_calls:
        raise Excluded("insufficient_call_budget")
    for i in range(k, len(actions)):
        action = deepcopy(actions[i])
        args, name = action["arguments"], action["name"]
        if name == "step":
            args["dependencies"] = [mapping[h] for h in args["dependencies"]]
        else:
            key = "answer_handle" if name == "finish" else "record_handle"
            args[key] = mapping[args[key]]
        observation = append_action(episode, messages, supervise, action_text(action), True)
        if not observation["ok"]:
            raise Excluded("correct_suffix_execution_failed", {"reference_action_index": i})
        expected = deepcopy(target["events"][i]["observation"])
        if name == "step":
            mapping[expected["handle"]] = observation["handle"]
            expected["handle"] = observation["handle"]
        if typed_json(observation) != typed_json(expected):
            raise Excluded("correct_suffix_observation_mismatch", {"reference_action_index": i})
    if not episode.done or not answers_equal(episode.answer, reference["answer"]):
        raise Excluded("correct_suffix_answer_mismatch")
    messages.append({"role": "assistant", "content": "Done."})
    supervise.append(True)
    return {"id": reference["id"], "pair_id": reference["id"], "messages": messages,
            "events": episode.events, "supervise": supervise, "calls": episode.calls,
            "answer": episode.answer, "reference_handle_map": {str(k): v for k, v in mapping.items()},
            "source": "domain_recovery_suffix" if recovery else "domain_clean_suffix",
            "training_eligible": False}


def make_pair(executor, store, rollout, reference, max_calls=24, min_shared_prefix=1):
    if min_shared_prefix < 1:
        raise ValueError("A shared successful prefix is required")
    audit = {"id": reference.get("id"), "status": "excluded",
             "selection": "first_reference_divergence_in_failed_rollout",
             "semantic_error_at_divergence": "not_established",
             "rollout_provenance": deepcopy(rollout.get("provenance", {"kind": "unverified"})),
             "training_eligible": False, "token_budget_matched": False}
    try:
        try:
            target = successful_trajectory(executor, store, reference, max_calls)
        except (ValueError, TypeError, KeyError) as exc:
            raise Excluded("invalid_reference", type(exc).__name__) from exc
        source, turns = replay_rollout(executor, store, rollout, reference, max_calls)
        if answers_equal(source.answer, reference["answer"]):
            raise Excluded("free_run_answer_correct")
        k = 0
        for turn, action in zip(turns, reference["actions"]):
            actual = turn["event"]
            if (not actual["observation"]["ok"] or actual["tool"] != action["name"]
                    or typed_json(actual["arguments"]) != typed_json(action["arguments"])):
                break
            k += 1
        audit["shared_prefix_calls"] = k
        if k < min_shared_prefix:
            raise Excluded("no_shared_successful_prefix")
        if k == len(turns):
            raise Excluded("no_divergence_before_stop")
        if k >= len(reference["actions"]):
            raise Excluded("reference_already_completed")
        audit.update(divergence_executable=turns[k]["event"]["observation"]["ok"],
                     divergence_tool=turns[k]["event"]["tool"])
        recovery = continuation(executor, store, reference, target, turns, k, True, max_calls)
        clean = continuation(executor, store, reference, target, turns, k, False, max_calls)
        audit.update(status="accepted", supervised_reference_actions=len(reference["actions"])-k,
                     clean_calls=clean["calls"], recovery_calls=recovery["calls"],
                     matched_target_actions=True)
        for row in (clean, recovery):
            row["selection"] = deepcopy(audit)
        return {"audit": audit, "clean": clean, "recovery": recovery}
    except Excluded as exc:
        audit.update(reason=exc.reason)
        if exc.detail is not None:
            audit["detail"] = exc.detail
        return {"audit": audit}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bound_path(spec):
    path = Path(spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or sha256(path) != spec["sha256"]:
        raise ValueError("Declared workspace artifact changed")
    return path


def admit_training_inputs(reference_path, rollout_path, split_path, generation_path):
    split, generation = [json.loads(Path(p).read_text()) for p in (split_path, generation_path)]
    if (split.get("schema") != "radar_domain_split_manifest_v1"
            or generation.get("schema") != "radar_domain_rollout_manifest_v1"
            or generation.get("status") != "completed" or generation.get("origin") != "model_generated"
            or generation.get("split") != "train"
            or generation.get("max_calls") != 24
            or generation.get("split_manifest_sha256") != sha256(split_path)):
        raise ValueError("Completed model-generated training inventory and split required")
    partitions = split["splits"]
    if set(partitions) != {"train", "dev", "holdout"}:
        raise ValueError("Train/dev/holdout partitions required")
    all_ids, all_groups = set(), set()
    for part in partitions.values():
        ids = part["ids"]
        groups = set(part["group_by_id"].values())
        if (len(ids) != len(set(ids)) or set(ids) & all_ids or set(part["group_by_id"]) != set(ids)
                or any(type(g) is not str or not g for g in groups) or groups & all_groups):
            raise ValueError("Split ID or source-group overlap")
        all_ids.update(ids); all_groups.update(groups)
    train = partitions["train"]
    if (not train["ids"] or bound_path(train["references"]) != Path(reference_path).resolve()
            or bound_path(generation["rollouts"]) != Path(rollout_path).resolve()
            or generation.get("question_input") != train["questions"]):
        raise ValueError("Only bound train references/rollouts are admitted")
    questions = read_jsonl(bound_path(train["questions"]))
    if ([q.get("id") for q in questions] != train["ids"]
            or any(set(q) != {"id", "question"} for q in questions)):
        raise ValueError("Generation inputs must be complete question-only train rows")
    for field in ("model_files_sha256", "implementation_files_sha256"):
        if not generation.get(field):
            raise ValueError("Model and generator implementation identities required")
        for name, expected in generation[field].items():
            bound_path({"path": name, "sha256": expected})
    interface = "experiments/radar_domain/record_set_interface.py"
    if generation["implementation_files_sha256"].get(interface) != sha256(interface):
        raise ValueError("Domain execution interface identity missing")
    references, rollouts = read_jsonl(reference_path), read_jsonl(rollout_path)
    if ([r.get("id") for r in references] != train["ids"]
            or [r.get("id") for r in rollouts] != train["ids"]
            or generation["rollouts"].get("rows") != len(rollouts)):
        raise ValueError("Full ordered train coverage required")
    q_by_id = {q["id"]: q["question"] for q in questions}
    for ref, row in zip(references, rollouts):
        if (ref["question"] != q_by_id[ref["id"]]
                or row.get("provenance") != {"kind": "model_generated",
                                           "generation_id": generation.get("generation_id")}
                or not isinstance(generation.get("generation_id"), str) or not generation["generation_id"]):
            raise ValueError("Real model-generation provenance required; CPU fixtures are excluded")
    return references, rollouts


def production_build(args):
    if args.output.exists():
        raise FileExistsError("Do not overwrite recovery data")
    references, rollouts = admit_training_inputs(
        args.references, args.rollouts, args.split_manifest, args.generation_manifest)
    generation = json.loads(args.generation_manifest.read_text())
    if (bound_path(generation["kb"]) != args.kb.resolve()
            or bound_path(generation["source_chunks"]) != args.chunks.resolve()):
        raise ValueError("Generation and construction graph identities differ")
    executor, store = load_bridge(args.kb, args.chunks)
    pairs, audits, success = [], [], []
    for reference, rollout in zip(references, rollouts):
        result = make_pair(executor, store, rollout, reference)
        audits.append(result["audit"])
        if result["audit"]["status"] == "accepted":
            result["audit"]["training_eligible"] = True
            for key in ("clean", "recovery"):
                result[key]["training_eligible"] = True
                result[key]["data_grade"] = "manifest_verified_model_training_rollout"
                result[key]["selection"] = deepcopy(result["audit"])
            pairs.append(result)
        row = successful_trajectory(executor, store, reference)
        row.update(training_eligible=True, data_grade="bound_train_reference")
        success.append(row)
    write_new(args.output / "success.json", success)
    write_new(args.output / "pairs.json", pairs)
    write_new(args.output / "audits.json", audits)
    write_new(args.output / "manifest.json", {
        "schema": "radar_domain_recovery_data_v1", "actual_model_rollouts": len(rollouts),
        "actual_model_recovery_pairs": len(pairs), "training_started": False,
        "counts": dict(Counter(a.get("reason", "accepted") for a in audits)),
        "token_budget_matched": False,
        "inputs_sha256": {str(p): sha256(p) for p in [
            args.references, args.rollouts, args.split_manifest, args.generation_manifest,
            args.kb, args.chunks, Path(__file__)]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    production_build(parser.parse_args())


if __name__ == "__main__":
    main()
