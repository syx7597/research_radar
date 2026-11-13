"""CPU diagnostic of suffix construction on the frozen 418 training pairs.

This executes fixed reference operations, not a language model. The historical
sample, source artifacts and original training/evaluation code remain unchanged.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from experiments.condition_consistency.executor import KoPLExecutor, compare_answers
from .environment import Episode, MAX_CALLS, call_message, compact, prompt_messages
from .inference import execute_response
from .recovery import Excluded, continuation, reference_calls, sha256
from .same_state import canonical, state_sha
from .training_data import read_jsonl

PUBLIC = Path("results/agent_feedback/construction_diagnostic_20261113")
PRIVATE = Path("data/agent_feedback/construction_diagnostic_20261113")
PAIR_MANIFEST = Path("results/agent_feedback/recovery_pairs.json")
GOLD = Path("data/agent_feedback/train.gold.jsonl")
KB = Path("datasets/kqa_pro/kb.json")
CODE = Path("experiments/agent_feedback/construction_diagnostic.py")
TEST = Path("tests/test_construction_diagnostic.py")


def encode(value, lines=False):
    if lines:
        return "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in value)
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_new(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encode(value, lines))


def identity(value):
    return hashlib.sha256(json.dumps(canonical(value), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inputs():
    manifest = json.loads(PAIR_MANIFEST.read_text())
    sources = manifest["sources"]
    paths = {str(PAIR_MANIFEST): sha256(PAIR_MANIFEST), str(GOLD): sources["gold_sha256"],
             str(KB): sources["kb_sha256"],
             "results/agent_feedback/split_manifest.json": sources["split_manifest_sha256"],
             "experiments/agent_feedback/recovery.py": sources["builder_sha256"],
             "experiments/agent_feedback/environment.py": sources["environment_sha256"],
             "experiments/agent_feedback/inference.py": sources["inference_sha256"]}
    for entry in manifest["artifacts"].values():
        paths[entry["path"]] = entry["sha256"]
    for path in (CODE, TEST, Path("experiments/agent_feedback/same_state.py"),
                 Path("experiments/condition_consistency/executor.py")):
        paths[str(path)] = sha256(path)
    for path, expected in paths.items():
        if sha256(path) != expected:
            raise ValueError("Frozen construction input changed: " + path)
    return manifest, paths


def register():
    manifest, paths = inputs()
    recovery = read_jsonl(manifest["artifacts"]["recovery"]["path"])
    clean = read_jsonl(manifest["artifacts"]["clean"]["path"])
    ids = [r["id"] for r in recovery]
    if len(ids) != 418 or len(set(ids)) != 418 or ids != [r["id"] for r in clean]:
        raise ValueError("Use all 418 historical pairs in their frozen order")
    design = {"schema": "recovery_construction_diagnostic_design_v1", "date": "2026-11-13",
        "status": "registered_before_new_comparison_execution", "population": "All 418 historically accepted TRAIN pairs",
        "fixed_denominator": 418, "ordered_ids_sha256": identity(ids), "max_calls": MAX_CALLS,
        "same_start": "Replay exactly the saved shared prefix plus its real divergent action/observation in every arm",
        "arms": {
            "direct": "Append original reference suffix with unchanged dependency and finish-handle numbers; attempt its remaining fixed calls once, with no repair/retry",
            "mapped_unfiltered": "Remap dependencies and finish handle, execute actual tools; measure reference consistency but do not use consistency to stop or filter this diagnostic candidate",
            "mapped_verified": "Use the existing frozen continuation builder with dependency remapping, step observation checks and final reference-answer check"},
        "metrics": ["prefix_and_historical_trace_replay_consistency", "suffix_calls_all_execute", "submitted_answer",
                    "reference_observations_match", "reference_dependency_origins_match", "final_answer_matches_reference",
                    "full_internal_result_exact_equality", "mapped_candidate_full_gate_pass", "candidate_sequence_identity"],
        "direct_failure_policy": "Continue the remaining reference action list after rejected calls unless the episode terminates or exhausts its existing budget; record all rejections",
        "observation_policy": "Normalize only returned handle identifiers for comparison; actual values, counts, result types and errors remain unchanged",
        "internal_result_policy": "Exact canonical typed internal values/order; this is not an arbitrary semantic equivalence oracle",
        "unavailable_policy": "Preserve all 418 rows; prefix/history/platform replay mismatches are separately counted and are not silently excluded",
        "source_limit": "Uses saved historical pair histories and references; does not regenerate or re-audit all original 5000 model rollouts",
        "case_selection": "Up to 3 cases per predeclared category, ascending SHA256('20261113|category|id'); selection is illustrative, all rows remain in aggregate",
        "case_categories": ["mapped_correct_direct_incorrect", "same_answer_with_reference_dependency_mismatch", "rejected_divergence_no_new_handle"],
        "prior_knowledge": "All historical mapped pairs previously passed the construction gate; their prior success is known. Direct-concatenation outcomes have not been executed in this round.",
        "interpretation": "Construction validity conditional on this historical training-pair sample; not model accuracy, learning benefit or a new held-out experiment",
        "new_model_runs": 0, "training_allowed": False, "server_accessed": False,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "required_pythonhashseed": "20261003"},
        "inputs_sha256": paths}
    write_new(PUBLIC / "design.json", design)
    print(json.dumps({"status": "registered", "pairs": 418, "design": str(PUBLIC / "design.json")}))


def replay_saved(executor, saved, question):
    messages = saved["messages"]
    if messages[:2] != prompt_messages(question):
        raise ValueError("saved_prompt_mismatch")
    tail = messages[2:]
    if tail and tail[-1] == {"role": "assistant", "content": "Done."}:
        tail = tail[:-1]
    if len(tail) != 2 * len(saved["events"]) or saved["calls"] != len(saved["events"]):
        raise ValueError("saved_event_alignment_mismatch")
    episode = Episode(executor, max_calls=MAX_CALLS)
    mismatches = []
    for i, event in enumerate(saved["events"]):
        if episode.done:
            mismatches.append({"index": i, "reason": "saved_continues_after_termination"})
            break
        assistant, tool = tail[2*i:2*i+2]
        if assistant["role"] != "assistant" or tool["role"] != "tool":
            raise ValueError("saved_turn_roles_differ")
        actual = execute_response(episode, assistant["content"])
        if actual != json.loads(tool["content"]) or episode.events[-1] != event:
            mismatches.append({"index": i, "reason": "saved_observation_or_event_differs"})
    if episode.prediction != saved["prediction"]:
        mismatches.append({"reason": "saved_prediction_differs"})
    return {"matches": not mismatches, "mismatches": mismatches}, episode


def restore_start(executor, saved, question, prefix_length, calls):
    if saved["messages"][:2] != prompt_messages(question) or prefix_length < 1:
        raise ValueError("Invalid saved common prefix")
    episode, mapping = Episode(executor, max_calls=MAX_CALLS), {}
    for i in range(prefix_length + 1):
        event = saved["events"][i]
        assistant, tool = saved["messages"][2+2*i:4+2*i]
        if episode.done or assistant["role"] != "assistant" or tool["role"] != "tool":
            raise ValueError("Invalid or terminal retained history")
        actual = execute_response(episode, assistant["content"])
        if actual != json.loads(tool["content"]) or episode.events[-1] != event:
            raise ValueError("retained_history_replay_mismatch")
        if i < prefix_length:
            if (not actual["ok"] or event["tool"] != calls[i]["tool"]
                    or event["arguments"] != calls[i]["arguments"]):
                raise ValueError("shared_prefix_is_not_successful_reference")
            mapping[i] = actual["handle"]
    if episode.done:
        raise ValueError("Retained history already terminates")
    if episode.calls + len(calls) - prefix_length > MAX_CALLS:
        raise ValueError("Reference suffix exceeds existing call budget")
    return episode, mapping


def execute_suffix(executor, saved, gold, calls, reference_events, reference_values, prefix_length, *, remap):
    episode, correspondence = restore_start(executor, saved, gold["question"], prefix_length, calls)
    start_sha = state_sha(episode)
    prefix_messages = deepcopy(saved["messages"][:2+2*(prefix_length+1)])
    messages = prefix_messages[:]
    steps, stopped = [], None
    for i in range(prefix_length, len(calls)):
        if episode.done:
            stopped = "episode_terminated_before_reference_end"
            break
        call = calls[i]
        args = deepcopy(call["arguments"])
        ref_dependencies = args["dependencies"] if call["tool"] == "step" else [args["answer_handle"]]
        intended = [correspondence.get(j) for j in ref_dependencies]
        if remap and any(h is None for h in intended):
            stopped = "missing_mapped_reference_dependency"
            break
        if remap:
            if call["tool"] == "step":
                args["dependencies"] = intended
            else:
                args["answer_handle"] = intended[0]
        actual_dependencies = args["dependencies"] if call["tool"] == "step" else [args["answer_handle"]]
        dependency_mismatch = actual_dependencies != intended
        text = call_message(call["tool"], args)["content"]
        observation = execute_response(episode, text)
        messages.extend([{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}])
        expected = deepcopy(reference_events[i]["observation"])
        exact_result = None
        if call["tool"] == "step" and observation["ok"]:
            correspondence[i] = observation["handle"]
            expected["handle"] = observation["handle"]
            exact_result = canonical(episode.handles[observation["handle"]].value) == reference_values[i]
        elif call["tool"] == "finish" and observation["ok"]:
            exact_result = canonical(episode.handles[episode.selected].value) == reference_values[ref_dependencies[0]]
        steps.append({"reference_index": i, "tool": call["tool"],
            "function": args.get("function", "finish"), "arguments": args,
            "intended_reference_dependencies": intended, "actual_dependencies": actual_dependencies,
            "reference_dependency_mismatch": dependency_mismatch,
            "observation": observation, "reference_observation_matches": observation == expected,
            "full_internal_result_exact_equal": exact_result})
    submitted = episode.selected is not None and episode.done
    all_calls = len(steps) == len(calls) - prefix_length
    correct = submitted and compare_answers(gold["answer"], episode.prediction)
    observation_ok = all_calls and all(s["reference_observation_matches"] for s in steps)
    interface_ok = all_calls and all(s["observation"]["ok"] for s in steps)
    if submitted:
        messages.append({"role": "assistant", "content": "Done."})
    trace = {"messages": messages, "events": episode.events, "prediction": episode.prediction,
             "calls": episode.calls, "reference_handle_map": {str(k): v for k, v in correspondence.items()}}
    return {"start_state_sha256": start_sha, "all_reference_calls_attempted": all_calls,
        "suffix_all_calls_execute": interface_ok, "submitted_answer": submitted,
        "reference_observations_match": observation_ok, "final_answer_correct": correct,
        "reference_dependency_mismatch_count": sum(s["reference_dependency_mismatch"] for s in steps),
        "step_dependency_mismatch_count": sum(s["reference_dependency_mismatch"] for s in steps if s["tool"] == "step"),
        "finish_dependency_mismatch": any(s["reference_dependency_mismatch"] for s in steps if s["tool"] == "finish"),
        "invalid_suffix_calls": sum(not s["observation"]["ok"] for s in steps),
        "reference_observation_mismatch_count": sum(not s["reference_observation_matches"] for s in steps),
        "full_internal_result_mismatch_count": sum(s["full_internal_result_exact_equal"] is False for s in steps),
        "full_gate_pass": interface_ok and observation_ok and correct and not any(s["reference_dependency_mismatch"] for s in steps),
        "calls": episode.calls, "stopped_reason": stopped, "prediction": episode.prediction,
        "sequence_sha256": identity(trace), "trace": trace, "steps": steps}


def diagnose_pair(executor, recovery, clean, gold):
    ident = recovery["id"]
    if ident != clean["id"] or ident != gold["id"]:
        raise ValueError("Unpaired row identity")
    k = recovery["selection"]["shared_prefix_calls"]
    result = {"id": ident, "status": "unavailable", "shared_prefix_calls": k,
              "source_divergence_executable": recovery["selection"]["divergence_executable"],
              "training_eligible": False, "model_generated": False}
    try:
        calls, events = reference_calls(executor, gold, MAX_CALLS)
        reference = Episode(executor, max_calls=MAX_CALLS)
        reference_values = {}
        for i, call in enumerate(calls):
            observed = execute_response(reference, call_message(call["tool"], call["arguments"])["content"])
            if call["tool"] == "step":
                reference_values[i] = canonical(reference.handles[observed["handle"]].value)
        result["historical_recovery_replay"], _ = replay_saved(executor, recovery, gold["question"])
        result["historical_clean_replay"], _ = replay_saved(executor, clean, gold["question"])
        # Freeze all variants to a replay-identical retained real history.
        start, _ = restore_start(executor, recovery, gold["question"], k, calls)
        result["retained_history_verified"] = True
        result["divergence_added_handle"] = recovery["events"][k]["observation"]["ok"] and recovery["events"][k]["tool"] == "step"
        result["start_state_sha256"] = state_sha(start)
        result["direct"] = execute_suffix(executor, recovery, gold, calls, events, reference_values, k, remap=False)
        result["mapped_unfiltered"] = execute_suffix(executor, recovery, gold, calls, events, reference_values, k, remap=True)
        turns = [{"message": recovery["messages"][2+2*i], "event": recovery["events"][i]} for i in range(k+1)]
        try:
            verified = continuation(executor, gold, turns, calls, events, k, True, MAX_CALLS)
        except Excluded as exc:
            result["mapped_verified"] = {"accepted": False, "reason": exc.reason, "detail": exc.detail}
        else:
            trace = {key: verified[key] for key in ("messages", "events", "prediction", "calls", "reference_handle_map")}
            result["mapped_verified"] = {"accepted": True, "final_answer_correct": compare_answers(gold["answer"], verified["prediction"]),
                "sequence_sha256": identity(trace),
                "same_sequence_as_unfiltered": identity(trace) == result["mapped_unfiltered"]["sequence_sha256"],
                "same_sequence_as_saved_recovery": all(verified[key] == recovery[key] for key in trace), "trace": trace}
        if result["direct"]["start_state_sha256"] != result["mapped_unfiltered"]["start_state_sha256"]:
            raise ValueError("Comparison histories differ")
        if result["mapped_verified"]["accepted"] != result["mapped_unfiltered"]["full_gate_pass"]:
            raise ValueError("Independent full gate differs from frozen builder")
        result["status"] = "compared"
        return result
    except (Excluded, ValueError) as exc:
        result["reason"] = exc.reason if isinstance(exc, Excluded) else str(exc)
        return result


def arm_counts(rows_, arm):
    available = [r[arm] for r in rows_ if r["status"] == "compared"]
    keys = ("suffix_all_calls_execute", "submitted_answer", "reference_observations_match", "final_answer_correct", "full_gate_pass")
    return {"denominator": len(rows_), "compared": len(available), "unavailable": len(rows_)-len(available),
        **{k: sum(x[k] for x in available) for k in keys},
        "any_reference_dependency_mismatch": sum(x["reference_dependency_mismatch_count"] > 0 for x in available),
        "any_step_dependency_mismatch": sum(x["step_dependency_mismatch_count"] > 0 for x in available),
        "finish_dependency_mismatch": sum(x["finish_dependency_mismatch"] for x in available),
        "any_full_internal_result_mismatch": sum(x["full_internal_result_mismatch_count"] > 0 for x in available),
        "invalid_suffix_calls": sum(x["invalid_suffix_calls"] for x in available)}


def select_cases(rows_):
    predicates = {
        "mapped_correct_direct_incorrect": lambda r: r["mapped_unfiltered"]["final_answer_correct"] and not r["direct"]["final_answer_correct"],
        "same_answer_with_reference_dependency_mismatch": lambda r: r["direct"]["final_answer_correct"] and r["direct"]["reference_dependency_mismatch_count"] > 0,
        "rejected_divergence_no_new_handle": lambda r: not r["divergence_added_handle"],
    }
    selected, public = [], []
    for category, predicate in predicates.items():
        candidates = [r for r in rows_ if r["status"] == "compared" and predicate(r)]
        order = lambda r: hashlib.sha256(f"20261113|{category}|{r['id']}".encode()).hexdigest()
        for row in sorted(candidates, key=order)[:3]:
            tag = order(row)[:16]
            selected.append({"category": category, "case_tag": tag, "row": row})
            public.append({"category": category, "case_tag": tag, "shared_prefix_calls": row["shared_prefix_calls"],
                "divergence_added_handle": row["divergence_added_handle"],
                "direct_correct": row["direct"]["final_answer_correct"],
                "mapped_correct": row["mapped_unfiltered"]["final_answer_correct"],
                "direct_suffix_functions": [s["function"] for s in row["direct"]["steps"]],
                "direct_mismatched_reference_steps": [s["reference_index"] for s in row["direct"]["steps"] if s["reference_dependency_mismatch"]]})
    return selected, public


def run(check=False):
    if os.environ.get("PYTHONHASHSEED") != "20261003":
        raise ValueError("Run with the registered PYTHONHASHSEED=20261003")
    design = json.loads((PUBLIC / "design.json").read_text())
    for path, expected in design["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Registered diagnostic input changed: " + path)
    if not check and ((PUBLIC / "summary.json").exists() or PRIVATE.exists()):
        raise FileExistsError("Preserve completed or partial diagnostic")
    manifest, _ = inputs()
    recovery = read_jsonl(manifest["artifacts"]["recovery"]["path"])
    clean = read_jsonl(manifest["artifacts"]["clean"]["path"])
    gold = {r["id"]: r for r in read_jsonl(GOLD)}
    if identity([r["id"] for r in recovery]) != design["ordered_ids_sha256"]:
        raise ValueError("Registered sample changed")
    executor = KoPLExecutor(KB)
    records = []
    for i, (r, c) in enumerate(zip(recovery, clean)):
        records.append(diagnose_pair(executor, r, c, gold[r["id"]]))
        if (i+1) % 100 == 0:
            print(json.dumps({"compared": i+1, "total": len(recovery)}), flush=True)
    if len(records) != 418:
        raise ValueError("Lost rows from the fixed denominator")
    selected, public_cases = select_cases(records)
    compared = [r for r in records if r["status"] == "compared"]
    summary = {"schema": "recovery_construction_diagnostic_v1", "date": "2026-11-13", "status": "completed",
        "kind": "offline_reference_construction_diagnostic", "new_model_runs": 0, "training_allowed": False,
        "server_accessed": False, "fixed_training_pair_denominator": 418, "compared": len(compared),
        "unavailable": len(records)-len(compared),
        "unavailable_reasons": dict(Counter(r["reason"] for r in records if r["status"] != "compared")),
        "historical_replays": {"recovery_matches": sum(r.get("historical_recovery_replay", {}).get("matches", False) for r in records),
            "clean_matches": sum(r.get("historical_clean_replay", {}).get("matches", False) for r in records),
            "same_full_sequence_as_saved_recovery": sum(r.get("mapped_verified", {}).get("same_sequence_as_saved_recovery", False) for r in records)},
        "arms": {arm: arm_counts(records, arm) for arm in ("direct", "mapped_unfiltered")},
        "mapped_verified": {"denominator": 418, "accepted": sum(r["mapped_verified"]["accepted"] for r in compared),
            "candidate_gate_rejections": sum(not r["mapped_verified"]["accepted"] for r in compared),
            "same_candidate_sequence_when_accepted": sum(r["mapped_verified"].get("same_sequence_as_unfiltered", False) for r in compared)},
        "strata": {label: {"direct": arm_counts(group, "direct"), "mapped_unfiltered": arm_counts(group, "mapped_unfiltered")}
            for label, group in (("divergence_added_handle", [r for r in compared if r["divergence_added_handle"]]),
                                 ("divergence_no_handle", [r for r in compared if not r["divergence_added_handle"]]))},
        "paired_outcomes": {"mapped_correct_direct_incorrect": sum(r["mapped_unfiltered"]["final_answer_correct"] and not r["direct"]["final_answer_correct"] for r in compared),
            "both_answer_correct": sum(r["mapped_unfiltered"]["final_answer_correct"] and r["direct"]["final_answer_correct"] for r in compared),
            "direct_answer_equal_despite_reference_dependency_mismatch": sum(r["direct"]["final_answer_correct"] and r["direct"]["reference_dependency_mismatch_count"] > 0 for r in compared)},
        "cases": public_cases,
        "interpretation": ["Counts measure reference suffix construction on the historically admitted training sample, not a trained model or held-out question success rate.",
            "Mapped-unfiltered executes tools and records observations; it differs from full construction only in whether consistency gates decide admission.",
            "Equal final answers under differing dependency origins are reported separately; answer coincidence alone does not establish reference-path equivalence.",
            "Saved histories are replayed locally; exact internal state equality is checked between current diagnostic arms, not asserted against unrecorded historical Linux internal states."],
        "design": {"path": str(PUBLIC / "design.json"), "sha256": sha256(PUBLIC / "design.json")},
        "implementation_sha256": sha256(CODE),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "pythonhashseed": os.environ["PYTHONHASHSEED"]}}
    outputs = {PRIVATE / "per_pair.jsonl": encode(records, True), PRIVATE / "cases.json": encode(selected)}
    summary["private_outputs"] = {str(path): hashlib.sha256(text.encode()).hexdigest() for path, text in outputs.items()}
    outputs[PUBLIC / "summary.json"] = encode(summary)
    for path, expected in design["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Input changed during diagnostic: " + path)
    for path, text in outputs.items():
        if check:
            if not path.exists() or path.read_text() != text:
                raise ValueError("Diagnostic output changed: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(text)
    print(json.dumps({"status": "checked" if check else "completed", "compared": len(compared),
                      "direct": summary["arms"]["direct"], "mapped": summary["arms"]["mapped_unfiltered"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("register", "run", "check"))
    args = p.parse_args()
    if args.command == "register":
        register()
    else:
        run(check=args.command == "check")


if __name__ == "__main__":
    main()
