"""Verify and replay all frozen development runs before opening AI references.

Reported successes are source-record selection checks on exposed development
questions. They are not independent domain accuracy or model-authored semantic
answers. Diagnostic tags describe observable traces, never presumed causes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.agent_feedback.environment import compact
from experiments.agent_feedback.inference import execute_response, validate_questions
from . import development_probe as probe
from .development_environment import (
    DevelopmentEpisode, DevelopmentKB, execute_program, prompt_messages, render_records,
)

VERSION = "radar_development_probe_analysis_v1"
TOTAL_KEYS = ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def replay_row(kb, row, question, protocol, label):
    """Replay exactly the stored generation text, including every tool reply."""
    prefix = prompt_messages(question["question"], kb, program=label == "P")
    require(row.get("messages", [])[:2] == prefix, f"{label}/{row.get('id')}: initial prompt drift")
    messages = row["messages"][2:]
    if label == "P":
        if "program" not in row:
            require(not messages and row.get("stop_reason") in {"context_budget", "generation_budget"},
                    "P missing program without a pre-generation budget stop")
            episode = DevelopmentEpisode(kb, max_calls=protocol["inference"]["max_calls"])
        else:
            require(messages == [{"role": "assistant", "content": row["program"]}],
                    "P program does not match its sole generated message")
            episode = execute_program(kb, row["program"])
    else:
        require(len(messages) % 2 == 0, "agent transcript must contain complete assistant/tool pairs")
        episode = DevelopmentEpisode(kb, max_calls=protocol["inference"]["max_calls"])
        for index in range(0, len(messages), 2):
            assistant, tool = messages[index:index + 2]
            require(set(assistant) == {"role", "content"} and assistant["role"] == "assistant",
                    "unexpected non-assistant generation message")
            require(not episode.done, "stored generation continued after episode termination")
            observation = execute_response(episode, assistant["content"])
            require(tool == {"role": "tool", "content": compact(observation)},
                    f"{label}/{row.get('id')}: tool observation replay mismatch")
    expected = {
        "events": episode.events, "prediction": episode.prediction,
        "selected_handle": episode.selected, "calls": episode.calls,
        "invalid_calls": sum(not event["observation"].get("ok", False) for event in episode.events),
        "rendered_evidence": render_records(episode.prediction) if episode.prediction is not None else None,
    }
    for key, value in expected.items():
        require(key in row and row[key] == value, f"{label}/{row.get('id')}: {key} replay mismatch")
    for key in TOTAL_KEYS:
        require(type(row.get(key)) is int and row[key] >= 0, f"invalid token/call count: {key}")
    require(row["total_tokens"] == row["input_tokens"] + row["generated_tokens"], "token sum mismatch")
    require(row["generated_tokens"] <= protocol["inference"]["max_generated"], "generation budget exceeded")
    require(row["calls"] <= protocol["inference"]["max_calls"], "call budget exceeded")
    if episode.prediction is not None:
        require(row["stop_reason"] == ("program_completed" if label == "P" else "finished"),
                "completed prediction has inconsistent stop reason")
    else:
        allowed = {"program_error", "context_budget", "generation_budget"} if label == "P" else {
            "call_budget", "context_budget", "generation_budget"}
        require(row["stop_reason"] in allowed, "unfinished prediction has inconsistent stop reason")
        if row["stop_reason"] == "generation_budget":
            require(row["generated_tokens"] == protocol["inference"]["max_generated"],
                    "generation-budget stop before exhaustion")
        if label != "P" and row["stop_reason"] == "call_budget":
            require(episode.calls == protocol["inference"]["max_calls"], "call-budget stop before exhaustion")
    return episode


def load_runs_and_replay():
    """Do not read references anywhere in this phase, even for failed runs."""
    protocols = {label: probe.verify(label, weights=False) for label in probe.MODELS}
    protocol = protocols["P"]
    require(all(item == protocol for item in protocols.values()), "protocol changed during verification")
    require(set(protocol["models"]) == set(probe.MODELS), "the five frozen model labels are required")
    questions = read_jsonl(probe.DATA / "questions.jsonl")
    validate_questions(questions)
    require(len(questions) == protocol["question_count"], "frozen question count mismatch")
    expected_ids = [question["id"] for question in questions]
    kb = DevelopmentKB(probe.DATA / "readings.json")
    protocol_hash = probe.sha(probe.PROTOCOL)
    runs, runtimes, input_hashes = {}, {}, {str(probe.PROTOCOL): protocol_hash}
    # Establish all five completion and byte bindings before replay or scoring.
    for label in probe.MODELS:
        output = probe.OUT / f"{label}.jsonl"
        runtime_path = output.with_suffix(".runtime.json")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        require(runtime.get("label") == label and runtime.get("status") == "completed",
                f"{label}: run is not explicitly completed")
        require(runtime.get("protocol_sha256") == protocol_hash, f"{label}: runtime protocol mismatch")
        require(runtime.get("output_sha256") == probe.sha(output), f"{label}: output hash mismatch")
        require(runtime.get("config") == protocol["inference"], f"{label}: runtime config mismatch")
        require(runtime.get("model") == protocol["models"][label], f"{label}: runtime model mismatch")
        rows = read_jsonl(output)
        require([row.get("id") for row in rows] == expected_ids, f"{label}: question order/coverage mismatch")
        require(all(row.get("label") == label for row in rows), f"{label}: row label mismatch")
        runs[label], runtimes[label] = rows, runtime
        input_hashes[str(output)] = probe.sha(output)
        input_hashes[str(runtime_path)] = probe.sha(runtime_path)
    replayed = 0
    for label, rows in runs.items():
        totals = Counter(questions=len(rows))
        for row, question in zip(rows, questions):
            replay_row(kb, row, question, protocol, label)
            totals.update({key: row[key] for key in TOTAL_KEYS})
            replayed += 1
        require(dict(totals) == runtimes[label]["totals"], f"{label}: runtime aggregate mismatch")
    # Catch accidental local changes between the beginning and replay completion.
    for label in probe.MODELS:
        require(probe.verify(label, weights=False) == protocol, "protocol changed during replay")
    return protocol, kb, runs, runtimes, input_hashes, replayed


def load_references(protocol, questions):
    reference_path = probe.DATA / "references.json"
    require(protocol["reference_sha256"] == {str(reference_path): probe.sha(reference_path)},
            "frozen reference hash mismatch")
    document = json.loads(reference_path.read_text(encoding="utf-8"))
    require(document.get("human_verified") is False and document.get("independent_gold") is False,
            "AI development references must not claim independent human gold")
    records = document["records"]
    require([row["id"] for row in records] == questions, "reference order/coverage mismatch")
    for row in records:
        target, context = row["expected_fact_ids"], row["acceptable_context_fact_ids"]
        require(isinstance(target, list) and target and len(set(target)) == len(target), "invalid target set")
        require(isinstance(context, list) and len(set(context)) == len(context), "invalid context set")
        require(not set(target) & set(context), "target/context sets must be distinct")
    return {row["id"]: row for row in records}


def selection_metrics(prediction, reference):
    selected = {row["fact_id"] for row in prediction or []}
    target = set(reference["expected_fact_ids"])
    allowed = set(reference["acceptable_context_fact_ids"])
    finished = prediction is not None
    covered = finished and target <= selected
    return {
        "finished": finished,
        "exact_target_selection": finished and selected == target,
        "target_covered": covered,
        "target_with_allowed_context": covered and selected <= target | allowed,
        "allowed_context_selected": bool(selected & allowed),
        "extra_selection": bool(selected - target),
        "unrelated_extra_selection": bool(selected - target - allowed),
        "empty_selection": finished and not selected,
        "unfinished": not finished,
    }


def error_category(event):
    observation, arguments = event["observation"], event["arguments"]
    detail = observation.get("detail", "")
    if observation.get("error") in {"ActionFormatError", "ProgramFormatError"}:
        return "format"
    if "unknown_function" in detail:
        return "function"
    if "unsupported_qualifier_key" in detail:
        return "condition_or_event_selector"
    if any(token in detail for token in ("answer_handle", "finish_requires", "budget", "episode_finished")):
        return "finish_or_budget"
    if any(token in detail for token in ("dependency", "handles")):
        return "dependency"
    if "requires" in detail and "inputs" in detail:
        return "function_arguments"
    if arguments.get("function") == "Find":
        return "entity_or_source"
    return "unresolved"


def diagnose(row, reference, kb_by_id):
    selected = row["prediction"] or []
    selected_ids = {item["fact_id"] for item in selected}
    target = set(reference["expected_fact_ids"])
    context = set(reference["acceptable_context_fact_ids"])
    metrics = selection_metrics(row["prediction"], reference)
    errors, empty_queries, queries = [], [], []
    for index, event in enumerate(row["events"]):
        observation, arguments = event["observation"], event["arguments"]
        item = {"event_index": index, "tool": event["tool"], "arguments": arguments,
                "ok": observation.get("ok", False), "returned_count": observation.get("count")}
        if not item["ok"]:
            item.update(category=error_category(event), error=observation.get("error"), detail=observation.get("detail"))
            errors.append(dict(item))
        elif observation.get("count") == 0:
            function = arguments.get("function")
            if function == "Find":
                category = "entity_or_source"
            elif function == "QueryAttrUnderCondition":
                category = "event" if arguments.get("inputs", [None, None])[1] == "event_type" else "condition"
            else:
                category = "attribute"
            empty_queries.append({**item, "category": category,
                                  "interpretation": "Observed empty query result; not proof that the language or selector was wrong."})
        queries.append(item)
    tags = []
    if metrics["unfinished"]:
        tags.append("unfinished_" + row["stop_reason"])
    if metrics["empty_selection"]:
        tags.append("finished_empty_selection")
    if metrics["allowed_context_selected"]:
        tags.append("allowed_comparison_context_selected")
    for extra in sorted(selected_ids - target - context):
        found = kb_by_id[extra]
        expected = [kb_by_id[fact_id] for fact_id in target]
        same_entity = [item for item in expected if item["entity_name"] == found["entity_name"]]
        same_attribute = [item for item in same_entity if item["attribute"] == found["attribute"]]
        if not same_entity:
            tags.append("entity_or_source_selection_difference")
        elif not same_attribute:
            tags.append("attribute_selection_difference")
        else:
            if any(item["event_type"] != found["event_type"] for item in same_attribute):
                tags.append("event_selection_difference")
            if any(item["condition_raw"] != found["condition_raw"] for item in same_attribute):
                tags.append("condition_selection_difference")
    if not metrics["target_covered"] and not tags:
        tags.append("unresolved_target_not_selected")
    fields = ("fact_id", "entity_name", "source_subject", "attribute", "event_type", "condition_raw")
    return {
        "id": row["id"], "label": row["label"], "stop_reason": row["stop_reason"],
        "metrics": metrics, "expected_fact_ids": sorted(target),
        "acceptable_context_fact_ids": sorted(context), "selected_fact_ids": sorted(selected_ids),
        "missing_target_fact_ids": sorted(target - selected_ids),
        "unrelated_extra_fact_ids": sorted(selected_ids - target - context),
        "selected_fields": [{key: item[key] for key in fields} for item in selected],
        "first_error": errors[0] if errors else None,
        "error_histogram": dict(sorted(Counter(item["category"] for item in errors).items())),
        "errors": errors, "empty_query_observations": empty_queries,
        "selection_observation_tags": sorted(set(tags)), "executed_actions": queries,
        "calls": row["calls"], "invalid_calls": row["invalid_calls"],
        "tokens": {key: row[key] for key in ("input_tokens", "generated_tokens", "total_tokens")},
        "semantic_answer_scored": False,
    }


def build():
    protocol, kb, runs, runtimes, input_hashes, replayed = load_runs_and_replay()
    require(replayed == len(probe.MODELS) * protocol["question_count"], "incomplete CPU replay")
    # Reference access occurs only after all five runs completed and replayed.
    references = load_references(protocol, [row["id"] for row in runs["P"]])
    kb_by_id = {record["fact_id"]: record for record in kb.records}
    require(all(set(ref["expected_fact_ids"] + ref["acceptable_context_fact_ids"]) <= set(kb_by_id)
                for ref in references.values()), "reference contains a record outside the frozen development KB")
    diagnoses, model_summaries = [], {}
    for label, rows in runs.items():
        counts, errors, stop_reasons, empty_queries, selection_tags = Counter(), Counter(), Counter(), Counter(), Counter()
        for row in rows:
            diagnosis = diagnose(row, references[row["id"]], kb_by_id)
            diagnoses.append(diagnosis)
            counts.update({key: int(value) for key, value in diagnosis["metrics"].items()})
            errors.update(diagnosis["error_histogram"])
            empty_queries.update(item["category"] for item in diagnosis["empty_query_observations"])
            selection_tags.update(diagnosis["selection_observation_tags"])
            stop_reasons.update([row["stop_reason"]])
        model_summaries[label] = {
            "questions": len(rows), "selection_counts": dict(counts),
            "execution_totals": runtimes[label]["totals"], "seconds": runtimes[label]["seconds"],
            "error_event_counts": dict(sorted(errors.items())),
            "empty_query_event_counts": dict(sorted(empty_queries.items())),
            "selection_observation_question_counts": dict(sorted(selection_tags.items())),
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
        }
    diagnosis_document = {
        "version": VERSION, "scope": protocol["scope"], "human_gold": False,
        "independent_evaluation": False, "semantic_answer_scored": False,
        "diagnosis_policy": "Observable selections/actions only. Empty or nonexact selection is not automatically a Chinese-understanding error. Allowed event comparison context is scored separately.",
        "records": diagnoses,
    }
    diagnostic_bytes = (json.dumps(diagnosis_document, ensure_ascii=False, indent=2) + "\n").encode()
    import hashlib
    input_hashes.update(protocol["inputs_sha256"])
    input_hashes.update(protocol["reference_sha256"])
    input_hashes[str(Path(__file__).resolve().relative_to(Path.cwd().resolve()))] = probe.sha(__file__)
    summary = {
        "version": VERSION, "scope": protocol["scope"],
        "models": model_summaries,
        "cpu_replay": {"rows": replayed, "mismatches": 0, "all_models_completed_before_reference_access": True,
                       "checked": ["frozen code/data hashes", "runtime protocol/output hashes", "initial prompts",
                                   "all assistant/tool messages", "events", "prediction", "selected_handle",
                                   "calls", "invalid_calls", "deterministic renderer", "token sums", "runtime totals"]},
        "metric_definitions": {
            "finished": "A terminal result was selected; an empty list still counts as finished.",
            "exact_target_selection": "Selected record set equals the predeclared expected_fact_ids exactly.",
            "target_covered": "All target records occur in the selected result, regardless of extras.",
            "target_with_allowed_context": "All targets selected and no records outside target plus predeclared context; includes exact selections.",
            "extra_selection": "Any record outside the target set, including allowed comparison context.",
            "unrelated_extra_selection": "Any selected record outside target plus predeclared allowed context.",
            "empty_selection": "Finished with an empty selected list; excludes unfinished episodes.",
            "tokens": "Unpadded complete prefills summed over turns plus generated tokens including EOS; not FLOPs.",
        },
        "interpretation_limits": [
            "The same twelve already exposed AI-reviewed source-reading development questions; no independent radar evaluation.",
            "The checkpoints were trained on the public English KoPL task; Chinese questions and this new source-record interface change multiple factors at once.",
            "Selection differences cannot isolate a language, training-method or domain-causality effect.",
            "Scoped prose is deterministically rendered from AI-reviewed KB notes; model semantic adjudication is not measured.",
            "No new training, prompt search, retries by score or automated next round.",
            "Analysis verifies existing inference artifacts and does not rerun GPU inference or revalidate local model weight bytes.",
        ],
        "new_training_started": False, "human_gold": False, "independent_evaluation": False,
        "semantic_answer_scored": False,
        "inputs_sha256": input_hashes,
        "outputs_sha256": {"per_question_diagnosis.json": hashlib.sha256(diagnostic_bytes).hexdigest()},
    }
    return {"per_question_diagnosis.json": diagnosis_document, "summary.json": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate reproduction without overwriting")
    args = parser.parse_args()
    outputs = build()
    if args.check:
        for name, value in outputs.items():
            expected = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            require((probe.OUT / name).read_text(encoding="utf-8") == expected, f"analysis artifact drift: {name}")
    else:
        require(not any((probe.OUT / name).exists() for name in outputs), "analysis version already exists; use --check")
        for name, value in outputs.items():
            probe.write_new(probe.OUT / name, value)
    print(json.dumps({"mode": "verified" if args.check else "created", "version": VERSION,
                      "cpu_replay": outputs["summary.json"]["cpu_replay"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
