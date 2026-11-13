"""Independently recount the saved construction diagnostic without rerunning it."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.condition_consistency.executor import compare_answers

PUBLIC = Path("results/agent_feedback/construction_diagnostic_20261113")
PRIVATE = Path("data/agent_feedback/construction_diagnostic_20261113")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    summary = json.loads((PUBLIC / "summary.json").read_text())
    design = json.loads((PUBLIC / "design.json").read_text())
    for path, expected in design["inputs_sha256"].items():
        assert sha(path) == expected, path
    for path, expected in summary["private_outputs"].items():
        assert sha(path) == expected, path
    data = rows(PRIVATE / "per_pair.jsonl")
    gold = {r["id"]: r for r in rows("data/agent_feedback/train.gold.jsonl")}
    clean = {r["id"]: r for r in rows("data/agent_feedback/recovery.clean.jsonl")}
    assert len(data) == len({r["id"] for r in data}) == 418
    compared = [r for r in data if r["status"] == "compared"]
    unavailable = [r for r in data if r["status"] != "compared"]
    assert len(compared) == summary["compared"] and len(unavailable) == summary["unavailable"]
    recounted = {}
    for arm in ("direct", "mapped_unfiltered"):
        count = Counter()
        for row in compared:
            result = row[arm]; trace = result["trace"]; k = row["shared_prefix_calls"]
            expected_suffix = clean[row["id"]]["events"][k:]
            actual_suffix = trace["events"][k+1:]
            assert len(actual_suffix) == len(result["steps"])
            all_attempted = len(actual_suffix) == len(expected_suffix)
            all_ok = all_attempted and all(s["observation"]["ok"] for s in actual_suffix)
            submitted = bool(actual_suffix and actual_suffix[-1]["tool"] == "finish"
                             and actual_suffix[-1]["observation"].get("ok")
                             and actual_suffix[-1]["observation"].get("done"))
            correct = submitted and compare_answers(gold[row["id"]]["answer"], trace["prediction"])
            correspondence = {i: clean[row["id"]]["events"][i]["observation"]["handle"] for i in range(k)}
            mismatch, step_mismatch, finish_mismatch, observation_equal = 0, 0, False, all_attempted
            for offset, (actual, expected) in enumerate(zip(actual_suffix, expected_suffix)):
                ref_index = k + offset
                observed, target = actual["observation"], deepcopy(expected["observation"])
                old_args, new_args = expected["arguments"], actual["arguments"]
                old_deps = old_args["dependencies"] if expected["tool"] == "step" else [old_args["answer_handle"]]
                new_deps = new_args["dependencies"] if actual["tool"] == "step" else [new_args["answer_handle"]]
                bad = new_deps != [correspondence.get(h) for h in old_deps]
                mismatch += bad
                if actual["tool"] == "step":
                    step_mismatch += bad
                    if observed["ok"]:
                        correspondence[ref_index] = observed["handle"]
                        target["handle"] = observed["handle"]
                else:
                    finish_mismatch |= bad
                observation_equal &= observed == target
                assert bad == result["steps"][offset]["reference_dependency_mismatch"]
                assert (observed == target) == result["steps"][offset]["reference_observation_matches"]
            derived = {"suffix_all_calls_execute": all_ok, "submitted_answer": submitted,
                       "reference_observations_match": observation_equal, "final_answer_correct": correct,
                       "full_gate_pass": all_ok and observation_equal and correct and mismatch == 0}
            for key, value in derived.items():
                assert value == result[key], (row["id"], arm, key)
                count[key] += value
            count["any_reference_dependency_mismatch"] += mismatch > 0
            count["any_step_dependency_mismatch"] += step_mismatch > 0
            count["finish_dependency_mismatch"] += finish_mismatch
            count["invalid_suffix_calls"] += sum(not e["observation"]["ok"] for e in actual_suffix)
        for key, value in count.items():
            assert value == summary["arms"][arm][key], (arm, key)
        recounted[arm] = dict(count)
    same = sum(r["mapped_unfiltered"]["trace"] == r["mapped_verified"].get("trace") for r in compared)
    assert same == summary["mapped_verified"]["same_candidate_sequence_when_accepted"]
    detail_path = PRIVATE / "replay_detail/differences.json"
    details = json.loads(detail_path.read_text())["rows"]
    assert {r["id"] for r in details} == {r["id"] for r in unavailable}
    assert all(r["differing_fields"] == ["detail"] and r["arguments_equal"] and r["tool_equal"]
               and r["new_handles"] == 0 and r["saved_observation"]["error"] == r["local_observation"]["error"] == "UnboundLocalError" for r in details)
    coincidences = [r["direct"] for r in compared if r["direct"]["final_answer_correct"]]
    report = {"schema": "construction_diagnostic_count_audit_v1", "date": "2026-11-13", "status": "passed",
        "reviewer": "AI author using separately implemented read-only recount; not independent human review",
        "denominator": len(data), "compared": len(compared), "strict_replay_unavailable": len(unavailable),
        "recounted_from_saved_actions_and_observations": recounted,
        "mapped_unfiltered_and_full_identical_traces": same,
        "all_four_unavailable_only_exception_detail_wording_differs": True,
        "posthoc_description_of_all_direct_answer_matches": {
            "answers_matching": len(coincidences), "all_calls_execute": sum(r["suffix_all_calls_execute"] for r in coincidences),
            "observations_all_match": sum(r["reference_observations_match"] for r in coincidences),
            "any_reference_dependency_mismatch": sum(r["reference_dependency_mismatch_count"] > 0 for r in coincidences),
            "any_full_internal_result_mismatch": sum(r["full_internal_result_mismatch_count"] > 0 for r in coincidences)},
        "code_sha256": sha(__file__), "summary_sha256": sha(PUBLIC / "summary.json"),
        "per_pair_sha256": sha(PRIVATE / "per_pair.jsonl"), "replay_detail_sha256": sha(detail_path),
        "new_model_runs": 0, "original_data_changed": False}
    path = PUBLIC / "count_audit.json"
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == text, "Preserve the completed audit"
    else:
        path.write_text(text)
    print(json.dumps({"status": "passed", "recounted": len(data), "compared": len(compared), "unavailable": len(unavailable)}))


if __name__ == "__main__":
    main()
