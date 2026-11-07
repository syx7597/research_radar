"""Recovery suffix construction for the shared typed-table tool interface."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.agent_feedback.recovery import Excluded, sha256
from experiments.radar_domain.recovery_data import typed_json
from .table_program import TableEpisode, TableExecutor, compact, execute_response, prompt_messages
from .table_scoring import answers_equal


def action_text(action):
    return "<tool_call>\n" + compact(action) + "\n</tool_call>"


def action_signature(name, arguments):
    data = deepcopy(arguments)
    if name == "step":
        if not isinstance(data.get("inputs"), list) or len(data["inputs"]) != 1:
            raise ValueError("One JSON config input required")
        # JSON object formatting is not an execution divergence. Literal value
        # types and list order remain part of the reference matching rule.
        data["inputs"] = [json.loads(data["inputs"][0])]
    return typed_json({"name": name, "arguments": data})


def append(episode, messages, masks, text, train):
    result = execute_response(episode, text)
    messages.extend([{"role": "assistant", "content": text},
                     {"role": "tool", "content": compact(result)}])
    masks.extend([train, False])
    return result


def success(executor, reference, max_calls=24):
    if set(reference) != {"id", "question", "actions", "answer"}:
        raise ValueError("Reference schema differs")
    actions = reference["actions"]
    if (not actions or len(actions) > max_calls or actions[-1]["name"] != "finish"
            or any(a["name"] != "step" for a in actions[:-1])):
        raise ValueError("Reference must end with one explicit finish")
    episode = TableEpisode(executor, max_calls)
    messages, masks = prompt_messages(reference["question"]), [False, False]
    for action in actions:
        result = append(episode, messages, masks, action_text(action), True)
        if not result["ok"]:
            raise Excluded("reference_execution_failed")
    if not episode.done or not answers_equal(episode.answer, reference["answer"]):
        raise Excluded("reference_answer_mismatch")
    messages.append({"role": "assistant", "content": "Done."}); masks.append(True)
    return {"id": reference["id"], "messages": messages, "supervise": masks,
            "events": episode.events, "calls": episode.calls, "answer": episode.answer,
            "training_eligible": False, "source": "executed_reference_success"}


def replay(executor, reference, rollout, max_calls):
    if rollout.get("id") != reference["id"]:
        raise Excluded("rollout_id_mismatch")
    messages, events = rollout.get("messages"), rollout.get("events")
    if (not isinstance(messages, list) or not isinstance(events, list)
            or messages[:2] != prompt_messages(reference["question"])
            or rollout.get("calls") != len(events) or "answer" not in rollout):
        raise Excluded("rollout_schema_mismatch")
    tail = messages[2:]
    if tail and tail[-1] == {"role": "assistant", "content": "Done."}:
        tail = tail[:-1]
    if len(tail) != 2 * len(events):
        raise Excluded("rollout_alignment_mismatch")
    episode = TableEpisode(executor, max_calls)
    for i, event in enumerate(events):
        assistant, tool = tail[2*i:2*i+2]
        if episode.done or assistant.get("role") != "assistant" or tool.get("role") != "tool":
            raise Excluded("rollout_turn_or_termination_mismatch")
        actual = execute_response(episode, assistant["content"])
        if (typed_json(actual) != typed_json(json.loads(tool["content"]))
                or typed_json(episode.events[-1]) != typed_json(event)):
            raise Excluded("rollout_replay_mismatch")
    if typed_json(episode.answer) != typed_json(rollout["answer"]):
        raise Excluded("rollout_answer_mismatch")
    return episode


def continuation(executor, reference, target, rollout, prefix, inserted, max_calls):
    episode = TableEpisode(executor, max_calls)
    messages, masks, mapping = prompt_messages(reference["question"]), [False, False], {}
    for i in range(prefix):
        text = rollout["messages"][2 + 2*i]["content"]
        observation = append(episode, messages, masks, text, False)
        if typed_json(observation) != typed_json(rollout["events"][i]["observation"]):
            raise Excluded("shared_prefix_replay_mismatch")
        mapping[target["events"][i]["observation"]["handle"]] = observation["handle"]
    if inserted:
        text = rollout["messages"][2 + 2*prefix]["content"]
        observation = append(episode, messages, masks, text, False)
        if typed_json(observation) != typed_json(rollout["events"][prefix]["observation"]):
            raise Excluded("divergence_replay_mismatch")
        if episode.done:
            raise Excluded("divergence_terminated_episode")
    if episode.calls + len(reference["actions"]) - prefix > max_calls:
        raise Excluded("insufficient_call_budget")
    for i in range(prefix, len(reference["actions"])):
        action = deepcopy(reference["actions"][i])
        args = action["arguments"]
        if action["name"] == "step":
            args["dependencies"] = [mapping[h] for h in args["dependencies"]]
        else:
            args["answer_handle"] = mapping[args["answer_handle"]]
        observation = append(episode, messages, masks, action_text(action), True)
        expected = deepcopy(target["events"][i]["observation"])
        if not observation["ok"]:
            raise Excluded("suffix_execution_failed")
        if action["name"] == "step":
            mapping[expected["handle"]] = observation["handle"]
            expected["handle"] = observation["handle"]
        if typed_json(observation) != typed_json(expected):
            raise Excluded("suffix_observation_mismatch")
    if not episode.done or not answers_equal(episode.answer, reference["answer"]):
        raise Excluded("suffix_answer_mismatch")
    messages.append({"role": "assistant", "content": "Done."}); masks.append(True)
    return {"id": reference["id"], "messages": messages, "supervise": masks,
        "events": episode.events, "calls": episode.calls, "answer": episode.answer,
        "handle_map": {str(k): v for k, v in mapping.items()}, "training_eligible": False,
        "source": "table_recovery_suffix" if inserted else "table_clean_suffix"}


def make_pair(executor, reference, rollout, max_calls=24):
    audit = {"id": reference["id"], "status": "excluded", "training_eligible": False,
             "selection": "first_reference_divergence_of_failed_rollout",
             "token_budget_matched": False, "provenance": rollout.get("provenance")}
    try:
        target = success(executor, reference, max_calls)
        actual = replay(executor, reference, rollout, max_calls)
        if answers_equal(actual.answer, reference["answer"]):
            raise Excluded("already_correct")
        prefix = 0
        for event, action in zip(rollout["events"], reference["actions"]):
            if (not event["observation"]["ok"] or
                    action_signature(event["tool"], event["arguments"]) !=
                    action_signature(action["name"], action["arguments"])):
                break
            prefix += 1
        if prefix < 1:
            raise Excluded("no_shared_successful_prefix")
        if prefix >= len(rollout["events"]):
            raise Excluded("no_divergence_before_stop")
        clean = continuation(executor, reference, target, rollout, prefix, False, max_calls)
        recovery = continuation(executor, reference, target, rollout, prefix, True, max_calls)
        audit.update(status="accepted", shared_prefix=prefix,
                     divergence_executable=rollout["events"][prefix]["observation"]["ok"])
        return {"audit": audit, "clean": clean, "recovery": recovery}
    except (Excluded, ValueError, KeyError, TypeError) as exc:
        audit["reason"] = exc.reason if isinstance(exc, Excluded) else type(exc).__name__
        return {"audit": audit}


def admit_model_training_rollouts(split_path, generation_path):
    """Only actual model-generated train data can enter formal recovery training."""
    split = json.loads(Path(split_path).read_text())
    generation = json.loads(Path(generation_path).read_text())
    if (split.get("schema") != "radar_domain_table_split_v1" or split.get("status") != "frozen"
            or generation.get("schema") != "radar_table_rollout_manifest_v1"
            or generation.get("status") != "completed" or generation.get("origin") != "model_generated"
            or generation.get("split") != "train" or generation.get("split_sha256") != sha256(split_path)):
        raise ValueError("Frozen split and completed model-generated training evidence required")
    train = split["splits"]["train"]
    for key in ("questions", "references", "catalog"):
        bound = train[key]
        if generation.get(key) != bound or sha256(bound["path"]) != bound["sha256"]:
            raise ValueError("Generation and frozen training inputs differ")
    if not generation.get("model_files_sha256") or not generation.get("implementation_sha256"):
        raise ValueError("Actual model and generator identities required")
    for group in ("model_files_sha256", "implementation_sha256"):
        for path, expected in generation[group].items():
            if sha256(path) != expected:
                raise ValueError("Bound model/generator changed")
    implementation = "experiments/radar_domain/table_program.py"
    if generation["implementation_sha256"].get(implementation) != sha256(implementation):
        raise ValueError("Executed tool implementation identity missing")
    path = generation["rollouts"]["path"]
    if sha256(path) != generation["rollouts"]["sha256"]:
        raise ValueError("Rollout file changed")
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    if ([r["id"] for r in rows] != train["ids"]
            or len(rows) != generation["rollouts"].get("rows")
            or not generation.get("generation_id")):
        raise ValueError("Incomplete ordered training rollout coverage")
    for row in rows:
        if row.get("provenance") != {"kind": "model_generated", "generation_id": generation["generation_id"]}:
            raise ValueError("Constructed CPU fixtures cannot enter model-failure training")
    return rows
