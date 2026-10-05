"""Replay fixed interface adaptation diagnostics and report all arms without selection."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import interface_probe as probe
from . import interface_training as training
from .development_analysis import read_jsonl, require, replay_row, TOTAL_KEYS, error_category
from .development_environment import DevelopmentKB, execute_program


def load_and_replay():
    protocol = json.loads(probe.PROTOCOL.read_text())
    questions = {phase: probe.question_rows(phase) for phase in probe.PHASES}
    kbs = {row["knowledge_id"]: DevelopmentKB(probe.kb_path(row["knowledge_id"]))
           for row in questions["after"]}
    runs, runtimes, hashes = {}, {}, {str(probe.PROTOCOL): probe.sha(probe.PROTOCOL)}
    # Establish completion of all fixed runs before replaying or loading references.
    for label in probe.MODELS:
        for phase in probe.PHASES:
            current, model = probe.verify(label, phase, weights=False)
            require(current == protocol, "Protocol drift")
            path = probe.OUT / f"{label}_{phase}.jsonl"
            runtime_path = path.with_suffix(".runtime.json")
            runtime = json.loads(runtime_path.read_text())
            rows = read_jsonl(path)
            require(runtime["status"] == "completed" and runtime["label"] == label and runtime["phase"] == phase,
                    "Incomplete run or wrong arm")
            require(runtime["protocol_sha256"] == hashes[str(probe.PROTOCOL)] and
                    runtime["output_sha256"] == probe.sha(path), "Run bytes are not bound")
            require(runtime["model"] == model and runtime["config"] == protocol["inference"], "Model/config drift")
            require([row["id"] for row in rows] == [q["id"] for q in questions[phase]], "Question order/coverage drift")
            hashes.update({str(path): probe.sha(path), str(runtime_path): probe.sha(runtime_path)})
            runs[label, phase], runtimes[label, phase] = rows, runtime
    replayed = 0
    for (label, phase), rows in runs.items():
        totals = Counter(questions=len(rows))
        for row, question in zip(rows, questions[phase]):
            require(row["label"] == label and row["phase"] == phase, "Wrong row arm")
            require(all(row[key] == question[key] for key in ("language", "dataset", "knowledge_id")), "Wrong row routing")
            replay_row(kbs[question["knowledge_id"]], row, question, protocol, label)
            totals.update({key: row[key] for key in TOTAL_KEYS})
            replayed += 1
        require(dict(totals) == runtimes[label, phase]["totals"], "Runtime totals mismatch")
    require(replayed == protocol["total_new_rows"], "Incomplete replay")
    training.hash_mapping(hashes)
    training.hash_mapping(protocol["inputs_sha256"])
    return protocol, questions, kbs, runs, runtimes, hashes, replayed


def load_references(protocol, questions, kbs):
    training.hash_mapping(protocol["reference_sha256"])
    synthetic = read_jsonl(training.DATA / "holdout.references.jsonl")
    require([row["id"] for row in synthetic] == [row["id"] for row in questions["before"]], "Synthetic reference coverage")
    references = {row["id"]: row for row in synthetic}
    original = json.loads((probe.old.DATA / "references.json").read_text())
    require(original["human_verified"] is False and original["independent_gold"] is False, "Wrong radar gold scope")
    radar = {row["id"]: row for row in original["records"]}
    for question in questions["after"]:
        if question["dataset"] == "radar_seen":
            references[question["id"]] = {**radar[question["original_id"]], "family": "radar_seen"}
        reference = references[question["id"]]
        expected = reference["expected_fact_ids"]
        require(len(expected) == 1, "Expected exactly one source-record target")
        episode = execute_program(kbs[question["knowledge_id"]], reference["canonical_program"])
        require(episode.prediction is not None and {r["fact_id"] for r in episode.prediction} == set(expected),
                "Reference program failed exact selection")
    return references


def verify_training():
    protocol = json.loads(training.PROTOCOL.read_text())
    records = {}
    for label in probe.MODELS:
        path = training.OUT / label / "run_result.json"
        row = json.loads(path.read_text())
        kind = "program" if label == "P" else "agent"
        require(row["status"] == "completed" and row["label"] == label, "Incomplete training")
        require(row["protocol_sha256"] == probe.sha(training.PROTOCOL), "Training protocol drift")
        require(row["actual_supervised_tokens"] == protocol["caches"][kind]["supervised_tokens"] * protocol["config"]["epochs"],
                "Training target budget mismatch")
        require(row["cache_sha256"] == protocol["caches"][kind]["sha256"], "Training cache mismatch")
        require(row["seed"] == protocol["seeds"][label], "Training seed mismatch")
        records[label] = row
    for a, c in (("A1", "C1"), ("A2", "C2")):
        for key in ("actual_supervised_tokens", "actual_input_tokens", "microbatches", "sampled_order_sha256",
                    "sampled_examples", "global_step", "seed", "cache_sha256"):
            require(records[a][key] == records[c][key], f"A/C mismatch: {key}")
    return records


def build():
    protocol, questions, kbs, runs, runtimes, hashes, replayed = load_and_replay()
    # Semantic references are opened only after every fixed run is complete and replayed.
    training_results = verify_training()
    references = load_references(protocol, questions, kbs)
    diagnoses, summaries, indexed = [], {}, {}
    for (label, phase), rows in runs.items():
        for dataset in ("synthetic", "radar_seen"):
            for language in probe.old.LANGUAGES:
                group = [row for row in rows if row["dataset"] == dataset and row["language"] == language]
                if not group:
                    continue
                counts, totals, errors, families = Counter(), Counter(), Counter(), {}
                for row in group:
                    reference = references[row["id"]]
                    selected = {item["fact_id"] for item in row["prediction"] or []}
                    target = set(reference["expected_fact_ids"])
                    exact = row["prediction"] is not None and selected == target
                    metrics = {"exact": exact, "finished": row["prediction"] is not None,
                               "empty": row["prediction"] == [], "has_invalid_call": row["invalid_calls"] > 0}
                    counts.update({key: int(value) for key, value in metrics.items()})
                    totals.update({key: row[key] for key in TOTAL_KEYS})
                    errors.update(error_category(e) for e in row["events"] if not e["observation"].get("ok"))
                    family = families.setdefault(reference["family"], {"questions": 0, "exact": 0})
                    family["questions"] += 1
                    family["exact"] += int(exact)
                    diagnosis = {"label": label, "phase": phase, "dataset": dataset, "language": language,
                                 "id": row["id"], "knowledge_id": row["knowledge_id"], "family": reference["family"],
                                 "metrics": metrics, "selected_fact_ids": sorted(selected), "expected_fact_ids": sorted(target),
                                 "stop_reason": row["stop_reason"]}
                    diagnoses.append(diagnosis)
                    indexed[label, phase, row["id"]] = diagnosis
                summaries[f"{label}_{phase}_{dataset}_{language}"] = {
                    "questions": len(group), "counts": dict(counts), "execution_totals": dict(totals),
                    "error_events": dict(errors), "families": families}
    paired = {}
    for label in probe.MODELS:
        transitions = Counter()
        for question in questions["before"]:
            before, after = [indexed[label, phase, question["id"]]["metrics"]["exact"] for phase in probe.PHASES]
            transitions["both_exact" if before and after else "after_only" if after else "before_only" if before else "neither"] += 1
        paired[label] = dict(transitions)
    old_summary = json.loads((probe.old.OUT / "summary.json").read_text())
    radar_before = {name: {"questions": item["questions"], "counts": item["selection_counts"]}
                    for name, item in old_summary["runs"].items()}
    hashes.update(protocol["inputs_sha256"])
    hashes.update(protocol["reference_sha256"])
    for label in probe.MODELS:
        path = training.OUT / label / "run_result.json"
        hashes[str(path)] = probe.sha(path)
    summary = {"version": "radar_interface_analysis_v1", "scope": protocol["scope"], "runs": summaries,
               "synthetic_before_after_paired": paired, "radar_before_reused": radar_before,
               "cpu_replay": {"rows": replayed, "mismatches": 0, "all_ten_completed_and_replayed_before_reference_access": True},
               "training": {label: {key: row[key] for key in ("actual_supervised_tokens", "actual_input_tokens", "global_step",
                             "sampled_order_sha256", "seconds", "starting_adapter_sha256", "final_adapter_sha256")}
                            for label, row in training_results.items()},
               "A_C_matched_samples_order_tokens_verified": True, "formal_significance_tested": False,
               "limits": ["20 synthetic groups paired across languages; not 40 independent samples",
                          "Only instance/wording isolation, shared ontology and reasoning structures",
                          "Twelve radar targets are exposed AI development derivatives, not human gold",
                          "P sees same question exposures but fewer target tokens; not token-matched to agents",
                          "No public benchmark rerun after adaptation; original public results apply to original adapters"],
               "inputs_sha256": hashes}
    return summary, {"version": summary["version"], "records": diagnoses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, detail = build()
    for name, data in (("summary.json", summary), ("per_question_diagnosis.json", detail)):
        path = probe.OUT / name
        if args.check:
            require(json.loads(path.read_text()) == data, f"Analysis artifact drift: {name}")
        else:
            probe.write_new(path, data)
    print(json.dumps({"status": "verified" if args.check else "written", "cpu_replay": summary["cpu_replay"]}))


if __name__ == "__main__":
    main()
