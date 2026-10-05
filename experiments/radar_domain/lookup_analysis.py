"""Replay the ten fixed bilingual development runs before opening references.

The measurements describe source-record selection on exposed, AI-rewritten
lookup questions. Neither language differences nor comparisons between the
checkpoints establish a domain method effect or independent answer accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.agent_feedback.inference import validate_questions
from . import lookup_probe as probe
from .development_analysis import (
    TOTAL_KEYS, diagnose, read_jsonl, replay_row, require,
)
from .development_environment import DevelopmentKB, execute_program

VERSION = "radar_lookup_probe_analysis_v1"


def load_runs_and_replay():
    """Check every completion and byte binding before replaying any transcript."""
    protocols = {label: probe.verify(label, weights=False) for label in probe.MODELS}
    protocol = protocols["P"]
    require(all(item == protocol for item in protocols.values()), "protocol changed during verification")
    require(set(protocol["models"]) == set(probe.MODELS), "the five frozen models are required")
    require(protocol["languages"] == list(probe.LANGUAGES), "frozen language list mismatch")
    require(protocol["kb"] == str(probe.KB_PATH), "frozen KB path mismatch")
    require(protocol["questions"] == {language: str(probe.DATA / f"questions.{language}.jsonl")
                                     for language in probe.LANGUAGES}, "frozen language question paths mismatch")
    questions = {}
    for language in probe.LANGUAGES:
        rows = read_jsonl(probe.DATA / f"questions.{language}.jsonl")
        validate_questions(rows)
        require(len(rows) == protocol["question_count"], "frozen question count mismatch")
        questions[language] = rows
    expected_ids = [row["id"] for row in questions[probe.LANGUAGES[0]]]
    require(all([row["id"] for row in rows] == expected_ids for rows in questions.values()),
            "paired language question IDs/order mismatch")
    kb = DevelopmentKB(probe.KB_PATH)
    protocol_hash = probe.sha(probe.PROTOCOL)
    runs, runtimes, input_hashes = {}, {}, {str(probe.PROTOCOL): protocol_hash}
    for label in probe.MODELS:
        for language in probe.LANGUAGES:
            name = f"{label}_{language}"
            output = probe.OUT / f"{name}.jsonl"
            runtime_path = output.with_suffix(".runtime.json")
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            require(runtime.get("label") == label and runtime.get("language") == language,
                    f"{name}: runtime label/language mismatch")
            require(runtime.get("status") == "completed", f"{name}: run is not explicitly completed")
            require(runtime.get("protocol_sha256") == protocol_hash, f"{name}: runtime protocol mismatch")
            require(runtime.get("output_sha256") == probe.sha(output), f"{name}: output hash mismatch")
            require(runtime.get("config") == protocol["inference"], f"{name}: runtime config mismatch")
            require(runtime.get("model") == protocol["models"][label], f"{name}: runtime model mismatch")
            rows = read_jsonl(output)
            require([row.get("id") for row in rows] == expected_ids, f"{name}: question order/coverage mismatch")
            require(all(row.get("label") == label and row.get("language") == language for row in rows),
                    f"{name}: row label/language mismatch")
            runs[(label, language)], runtimes[(label, language)] = rows, runtime
            input_hashes[str(output)] = probe.sha(output)
            input_hashes[str(runtime_path)] = probe.sha(runtime_path)
    replayed = 0
    for (label, language), rows in runs.items():
        totals = Counter(questions=len(rows))
        for row, question in zip(rows, questions[language]):
            replay_row(kb, row, question, protocol, label)
            totals.update({key: row[key] for key in TOTAL_KEYS})
            replayed += 1
        require(dict(totals) == runtimes[(label, language)]["totals"],
                f"{label}_{language}: runtime aggregate mismatch")
    # Mutable input and run bytes must still be identical when replay finishes.
    for label in probe.MODELS:
        require(probe.verify(label, weights=False) == protocol, "protocol changed during replay")
    for name, expected in input_hashes.items():
        require(probe.sha(name) == expected, f"run artifact changed during replay: {name}")
    return protocol, kb, runs, runtimes, input_hashes, replayed


def load_references(protocol, expected_ids, kb):
    reference_path = probe.DATA / "references.json"
    require(protocol["reference_sha256"] == {str(reference_path): probe.sha(reference_path)},
            "frozen reference hash mismatch")
    document = json.loads(reference_path.read_text(encoding="utf-8"))
    require(document.get("human_verified") is False and document.get("independent_gold") is False,
            "AI development references must not claim independent human gold")
    rows = document["records"]
    require([row["id"] for row in rows] == expected_ids, "reference order/coverage mismatch")
    known_ids = {record["fact_id"] for record in kb.records}
    references = {}
    for row in rows:
        target = row["expected_fact_ids"]
        require(isinstance(target, list) and len(target) == 1 and set(target) <= known_ids,
                "single-record calibration reference must select exactly one known record")
        require(row.get("languages") == list(probe.LANGUAGES), "reference language mismatch")
        require(not row.get("acceptable_context_fact_ids"), "single-intent calibration has no allowed extra context")
        episode = execute_program(kb, row["canonical_program"])
        require(episode.prediction is not None and
                {item["fact_id"] for item in episode.prediction} == set(target),
                "canonical reference program does not select the declared target")
        references[row["id"]] = {**row, "acceptable_context_fact_ids": []}
    return references


def trace_diagnosis(row, reference, kb_by_id):
    result = diagnose(row, reference, kb_by_id)
    target = set(reference["expected_fact_ids"])
    covered_events, exact_events = [], []
    for index, event in enumerate(row["events"]):
        observation = event["observation"]
        if not (event["tool"] == "step" and observation.get("ok") and observation.get("type") == "value"):
            continue
        returned = {record["fact_id"] for record in observation["value"]}
        if target <= returned:
            covered_events.append(index)
        if target == returned:
            exact_events.append(index)
    result.update(
        language=row["language"],
        target_returned_event_indices=covered_events,
        exact_target_returned_event_indices=exact_events,
        trace_metrics={
            "target_ever_returned": bool(covered_events),
            "exact_target_ever_returned": bool(exact_events),
            "target_returned_but_not_finally_covered": bool(covered_events) and not result["metrics"]["target_covered"],
            "exact_target_returned_but_not_finally_exact": bool(exact_events) and not result["metrics"]["exact_target_selection"],
            "exact_target_returned_but_unfinished": bool(exact_events) and result["metrics"]["unfinished"],
            "has_invalid_call": row["invalid_calls"] > 0,
            "has_argument_error": bool(result["error_histogram"].get("function_arguments")),
            "exact_final_after_invalid_call": row["invalid_calls"] > 0 and result["metrics"]["exact_target_selection"],
        },
    )
    return result


def paired_comparisons(diagnoses):
    indexed = {(row["label"], row["language"], row["id"]): row for row in diagnoses}
    result = {}
    for label in probe.MODELS:
        ids = [row["id"] for row in diagnoses if row["label"] == label and row["language"] == "zh"]
        counts = Counter({key: 0 for key in ("both_exact", "english_only_exact", "chinese_only_exact", "neither_exact")})
        pairs = []
        for qid in ids:
            zh, en = [indexed[(label, language, qid)] for language in ("zh", "en")]
            a, b = zh["metrics"]["exact_target_selection"], en["metrics"]["exact_target_selection"]
            category = ("both_exact" if a and b else "english_only_exact" if b else
                        "chinese_only_exact" if a else "neither_exact")
            counts[category] += 1
            pairs.append({"id": qid, "category": category, "zh_stop_reason": zh["stop_reason"],
                          "en_stop_reason": en["stop_reason"],
                          "zh_selected_fact_ids": zh["selected_fact_ids"],
                          "en_selected_fact_ids": en["selected_fact_ids"]})
        result[label] = {"question_pairs": len(pairs), "counts": dict(counts), "pairs": pairs,
                         "formal_significance_tested": False}
    return result


def build():
    protocol, kb, runs, runtimes, input_hashes, replayed = load_runs_and_replay()
    require(replayed == len(probe.MODELS) * len(probe.LANGUAGES) * protocol["question_count"],
            "incomplete CPU replay")
    # No semantic reference access is allowed until all ten runs passed replay.
    references = load_references(protocol, [row["id"] for row in runs[("P", "zh")]], kb)
    kb_by_id = {record["fact_id"]: record for record in kb.records}
    diagnoses, summaries = [], {}
    for (label, language), rows in runs.items():
        counts, traces, errors, stops, empties, tags = [Counter() for _ in range(6)]
        for row in rows:
            diagnosis = trace_diagnosis(row, references[row["id"]], kb_by_id)
            diagnoses.append(diagnosis)
            counts.update({key: int(value) for key, value in diagnosis["metrics"].items()})
            traces.update({key: int(value) for key, value in diagnosis["trace_metrics"].items()})
            errors.update(diagnosis["error_histogram"])
            stops.update([row["stop_reason"]])
            empties.update(item["category"] for item in diagnosis["empty_query_observations"])
            tags.update(diagnosis["selection_observation_tags"])
        summaries[f"{label}_{language}"] = {
            "label": label, "language": language, "questions": len(rows),
            "selection_counts": dict(counts), "trace_question_counts": dict(traces),
            "execution_totals": runtimes[(label, language)]["totals"],
            "seconds": runtimes[(label, language)]["seconds"],
            "error_event_counts": dict(sorted(errors.items())),
            "empty_query_event_counts": dict(sorted(empties.items())),
            "selection_observation_question_counts": dict(sorted(tags.items())),
            "stop_reason_counts": dict(sorted(stops.items())),
        }
    diagnosis_document = {
        "version": VERSION, "scope": protocol["scope"], "human_gold": False,
        "independent_evaluation": False, "semantic_answer_scored": False,
        "diagnosis_policy": "Observable actions and selections only. A returned target followed by a failed final selection is reported separately from never returning it. Empty results and language differences are not causal diagnoses.",
        "records": diagnoses,
    }
    diagnostic_bytes = (json.dumps(diagnosis_document, ensure_ascii=False, indent=2) + "\n").encode()
    input_hashes.update(protocol["inputs_sha256"])
    input_hashes.update(protocol["reference_sha256"])
    for module in (__file__, "experiments/radar_domain/development_analysis.py"):
        relative = str(Path(module).resolve().relative_to(Path.cwd().resolve()))
        input_hashes[relative] = probe.sha(module)
    summary = {
        "version": VERSION, "scope": protocol["scope"], "runs": summaries,
        "paired_language_comparison": paired_comparisons(diagnoses),
        "cpu_replay": {"rows": replayed, "mismatches": 0, "completed_runs": len(runs),
                       "all_runs_completed_before_reference_access": True,
                       "all_runs_replayed_before_reference_access": True,
                       "checked": ["frozen code/data hashes", "runtime protocol/output/model/config/language bindings",
                                   "paired question IDs and exact language prompts", "assistant/tool messages",
                                   "events", "prediction", "selected_handle", "calls", "invalid_calls",
                                   "deterministic renderer", "token sums", "runtime totals", "canonical reference executability"]},
        "metric_definitions": {
            "finished": "A terminal result was selected; an empty list counts as finished.",
            "exact_target_selection": "Final selected fact IDs equal the fixed one-record target.",
            "target_covered": "The final result contains the target, possibly with extra records.",
            "unrelated_extra_selection": "Any final record outside the single target; no allowed comparison context.",
            "empty_selection": "Finished with an empty final list; excludes unfinished episodes.",
            "target_ever_returned": "At least one successful QueryAttr/QueryAttrUnderCondition observation contains the target, even if no final selection follows.",
            "exact_target_ever_returned": "At least one successful query observation contains exactly the target.",
            "exact_final_after_invalid_call": "An exact final selection follows at least one invalid call; this is not a counterfactual recovery effect.",
            "tokens": "Unpadded complete prefills summed over turns plus generated tokens including EOS; not FLOPs.",
        },
        "interpretation_limits": [
            "Twelve exposed AI-rewritten lookup questions in two language versions; not 24 independent items or a held-out radar benchmark.",
            "Bilingual meanings were AI reviewed; common instructions, schema and source observations remain mixed language.",
            "Within-checkpoint language pairs are descriptive compatibility diagnostics, not isolated proof of a language capability cause.",
            "Compared with the original twelve-question probe, task wording and scope also changed; gains cannot be assigned solely to language or prompt optimization.",
            "No formal significance test, domain efficacy claim, or new training decision follows automatically from these twelve items.",
            "Source-scoped prose is rendered from AI-reviewed KB notes; model semantic adjudication and equipment truth are not scored.",
            "Analysis replays existing artifacts on CPU and does not rerun GPU inference or revalidate local checkpoint weight bytes.",
        ],
        "new_training_started": False, "human_gold": False, "independent_evaluation": False,
        "semantic_answer_scored": False, "formal_significance_tested": False,
        "inputs_sha256": input_hashes,
        "outputs_sha256": {"per_question_diagnosis.json": hashlib.sha256(diagnostic_bytes).hexdigest()},
    }
    return {"per_question_diagnosis.json": diagnosis_document, "summary.json": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify reproduction without overwriting")
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
