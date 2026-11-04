"""Read-only structural diagnosis of the completed radar selector/answer study.

Uses original hashed outputs and references, replays selections, and reports
mechanical recoverability of their first divergence. No labels or data change.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from experiments.agent_feedback.environment import compact
from experiments.agent_feedback.inference import execute_response
from experiments.agent_feedback.recovery import sha256
from .development_environment import DevelopmentKB, DevelopmentEpisode, execute_program, SIGNATURES
from .development_analysis import read_jsonl, replay_row
from .source_rag import BM25Index, evidence_for_records, answer_messages

OUT = Path("results/radar_domain/method_alignment_review_20261104")
OLD = Path("results/radar_domain/source_eval_v1")
DATA = Path("data/radar_sources_v1")
ARMS = ("A1", "C1", "A2", "C2")


def fact_ids(prediction):
    return sorted(r["fact_id"] for r in prediction or [])


def first_divergence_probe(kb, source, reference):
    """Offline construction feasibility; no model continues these states."""
    target = execute_program(kb, reference["canonical_program"])
    calls = target.events
    k = 0
    for actual, expected in zip(source["events"], calls):
        if (not actual["observation"]["ok"] or actual["tool"] != expected["tool"]
                or actual["arguments"] != expected["arguments"]):
            break
        k += 1
    audit = {"id": reference["id"], "shared_prefix_calls": k}
    if fact_ids(source["prediction"]) == sorted(reference["expected_fact_ids"]):
        return dict(audit, reason="selection_already_correct")
    if k < 1:
        return dict(audit, reason="no_common_successful_prefix")
    if k == len(source["events"]):
        return dict(audit, reason="no_divergence_before_stop")
    if k >= len(calls):
        return dict(audit, reason="reference_completed")
    episode = DevelopmentEpisode(kb)
    mapping = {}
    for i in range(k + 1):
        message = source["messages"][2 + 2 * i]["content"]
        observation = execute_response(episode, message)
        assert episode.events[-1] == source["events"][i]
        if i < k and calls[i]["tool"] == "step":
            mapping[i] = observation["handle"]
    audit.update(divergence_executable=episode.events[-1]["observation"]["ok"],
                 divergence_tool=episode.events[-1]["tool"])
    if episode.done:
        return dict(audit, reason="divergence_terminal")
    if episode.calls + len(calls) - k > 24:
        return dict(audit, reason="insufficient_calls")
    for i in range(k, len(calls)):
        call = calls[i]
        args = deepcopy(call["arguments"])
        if call["tool"] == "step":
            args["dependencies"] = [mapping[d] for d in args["dependencies"]]
        else:
            args["answer_handle"] = mapping[args["answer_handle"]]
        text = "<tool_call>" + compact({"name": call["tool"], "arguments": args}) + "</tool_call>"
        observation = execute_response(episode, text)
        if not observation["ok"]:
            return dict(audit, reason="suffix_execution_failed")
        expected = deepcopy(call["observation"])
        if call["tool"] == "step":
            mapping[i] = observation["handle"]
            expected["handle"] = observation["handle"]
        assert observation == expected
    assert fact_ids(episode.prediction) == sorted(reference["expected_fact_ids"])
    return dict(audit, reason="executable_recovery_pair", suffix_calls=len(calls) - k)


def main():
    if OUT.exists():
        raise FileExistsError("Do not overwrite a completed descriptive review")
    protocol_path = OLD / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    paths = [protocol_path, DATA / "references.jsonl", DATA / "questions.jsonl", DATA / "kb.json",
             DATA / "chunks.jsonl", OLD / "ai_semantic_verdicts.json",
             Path("experiments/radar_domain/development_environment.py"),
             Path("experiments/radar_domain/interface_data.py"),
             Path("experiments/radar_domain/interface_training.py"),
             Path("experiments/radar_domain/source_answers.py"), Path(__file__)]
    for name, expected in {**protocol["inputs_sha256"], **protocol["reference_sha256"]}.items():
        assert sha256(name) == expected, name
    questions = read_jsonl(DATA / "questions.jsonl")
    references = read_jsonl(DATA / "references.jsonl")
    assert [q["id"] for q in questions] == [r["id"] for r in references]
    refs = {r["id"]: r for r in references}
    kb = DevelopmentKB(DATA / "kb.json")
    index = BM25Index.from_path(DATA / "chunks.jsonl")
    canonical = {r["id"]: execute_program(kb, r["canonical_program"]) for r in references}
    for ident, target in canonical.items():
        assert fact_ids(target.prediction) == sorted(refs[ident]["expected_fact_ids"])
    selections, answers = {}, {}
    for arm in ("P", *ARMS):
        path = OLD / f"{arm}.selection.jsonl"
        runtime_path = path.with_suffix(".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        assert runtime["status"] == "completed" and runtime["protocol_sha256"] == sha256(protocol_path)
        assert runtime["output_sha256"] == sha256(path)
        rows = read_jsonl(path)
        assert [r["id"] for r in rows] == [q["id"] for q in questions]
        for row, question in zip(rows, questions):
            replay_row(kb, row, question, protocol, arm)
        selections[arm] = {r["id"]: r for r in rows}
        paths.extend((path, runtime_path))
    for arm in ("RAG", "P", *ARMS):
        path = OLD / f"{arm}.answers.jsonl"
        runtime_path = path.with_suffix(".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        assert runtime["status"] == "completed" and runtime["protocol_sha256"] == sha256(protocol_path)
        assert runtime["output_sha256"] == sha256(path)
        assert runtime["adapter"] is None and runtime["model"] == protocol["model"]
        assert runtime["config"] == protocol["answer_inference"]
        rows = read_jsonl(path)
        assert [r["id"] for r in rows] == [q["id"] for q in questions]
        answers[arm] = {r["id"]: r for r in rows}
        paths.extend((path, runtime_path))
    verdicts = {(r["arm"], r["id"]): r["adjudicated"]
                for r in json.loads((OLD / "ai_semantic_verdicts.json").read_text())["records"]}
    arm_stats = {}
    recovery = {}
    for arm, rows in selections.items():
        counts = Counter()
        bad_ids = []
        oracle_prompt_diff = []
        for ident, row in rows.items():
            correct = fact_ids(row["prediction"]) == sorted(refs[ident]["expected_fact_ids"])
            counts["exact_selection"] += correct
            counts["finished"] += row["prediction"] is not None
            counts["calls"] += row["calls"]
            counts["invalid_calls"] += row["invalid_calls"]
            counts["answer_correct"] += verdicts[arm, ident]["answer_correct"]
            counts["selection_correct_answer_wrong"] += correct and not verdicts[arm, ident]["answer_correct"]
            counts["selection_wrong_answer_correct"] += not correct and verdicts[arm, ident]["answer_correct"]
            if not correct:
                bad_ids.append(ident)
            expected = evidence_for_records(row["prediction"] or [], index)
            assert expected == answers[arm][ident]["evidence"]
            assert answer_messages(refs[ident]["question"], expected) == answers[arm][ident]["messages"]
            oracle = evidence_for_records(canonical[ident].prediction, index)
            if answer_messages(refs[ident]["question"], oracle) != answers[arm][ident]["messages"]:
                oracle_prompt_diff.append(ident)
        arm_stats[arm] = {"counts": dict(counts), "calls_histogram": dict(Counter(r["calls"] for r in rows.values())),
                          "failed_selection_ids": bad_ids, "oracle_correct_selection_changes_answer_prompt_ids": oracle_prompt_diff}
        if arm in ARMS:
            audits = [first_divergence_probe(kb, rows[r["id"]], r) for r in references]
            recovery[arm] = {"counts": dict(Counter(a["reason"] for a in audits)),
                             "eligible": [a for a in audits if a["reason"] == "executable_recovery_pair"],
                             "not_eligible_failures": [a for a in audits if a["reason"] not in {"selection_already_correct", "executable_recovery_pair"}]}
    paired = {}
    for a, c in (("A1", "C1"), ("A2", "C2")):
        counts = Counter()
        rows = []
        for ident in refs:
            same_selected = fact_ids(selections[a][ident]["prediction"]) == fact_ids(selections[c][ident]["prediction"])
            same_prompt = answers[a][ident]["messages"] == answers[c][ident]["messages"]
            same_text = answers[a][ident]["generated_text"] == answers[c][ident]["generated_text"]
            flags = {"same_selected_records": same_selected, "same_answer_prompt": same_prompt,
                     "same_generated_answer": same_text, "different_selection_same_prompt": not same_selected and same_prompt,
                     "same_prompt_different_generated_answer": same_prompt and not same_text}
            counts.update(flags)
            rows.append({"id": ident, **flags})
        paired[f"{c}_vs_{a}"] = {"counts": dict(counts), "question_flags": rows}
    public_gold = Path("data/agent_feedback/train.gold.jsonl")
    public_lengths = None
    if public_gold.exists():
        data = read_jsonl(public_gold)
        lengths = sorted(len(r["program"]) for r in data)
        public_lengths = {"questions": len(data), "operation_histogram": dict(Counter(lengths)),
                          "median": lengths[len(lengths)//2], "mean": sum(lengths)/len(lengths)}
        paths.append(public_gold)
    report = {"schema": "radar_method_alignment_review_v1", "date": "2026-11-04",
        "new_model_calls": 0, "old_labels_changed": False, "human_gold": False,
        "source_questions": len(refs), "replayed_selector_outputs": 120,
        "reference_operations_histogram": dict(Counter(len(r["canonical_program"].split("<func>")) for r in references)),
        "reference_total_calls_histogram": dict(Counter(ep.calls for ep in canonical.values())),
        "reference_function_histogram": dict(Counter(e["arguments"]["function"] for ep in canonical.values() for e in ep.events if e["tool"] == "step")),
        "reference_explicit_condition_questions": sum(r["intent"].get("condition_raw") is not None for r in references),
        "available_functions": SIGNATURES, "query_dependency_type": "Find only",
        "public_training_program_lengths": public_lengths,
        "answer_generator": {"model": protocol["model"], "adapter": None, "same_config_all_arms": True,
                            "selector_values_and_histories_in_answer_prompt": False},
        "arms": arm_stats, "paired": paired, "offline_first_divergence_recovery": recovery,
        "four_agent_same_answer_prompt_questions": sum(len({compact(answers[a][i]["messages"]) for a in ARMS}) == 1 for i in refs),
        "interpretation": "Existing exposed diagnostic only. Feasible suffix replay is not model recovery success; answer scores reuse original AI verdicts. Oracle prompt changes are not oracle model accuracy.",
        "inputs_sha256": {str(p): sha256(p) for p in paths}}
    OUT.mkdir(parents=True)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("reference_operations_histogram","reference_function_histogram","arms","paired","offline_first_divergence_recovery")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
