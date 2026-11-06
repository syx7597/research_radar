"""Create explicitly constructed CPU fixtures, never model-training rollouts."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from experiments.agent_feedback.recovery import sha256
from .kopl_record_bridge import compact, write_new
from .record_set_interface import RecordSetEpisode, execute_response, load_bridge, prompt_messages
from .recovery_data import action_text, make_pair, read_jsonl, successful_trajectory

REFERENCES = Path("data/radar_domain_recovery_v1/reference_dev_20261106/references.jsonl")
PRIVATE = Path("data/radar_domain_recovery_v1/recovery_cpu_fixtures_20261106")
PUBLIC = Path("results/radar_domain/recovery_pipeline_20261106")


def write_lines(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def constructed_rollout(executor, store, reference, kind):
    if kind == "extra_handle":
        divergence = {"name": "step", "arguments": {
            "function": "Find", "inputs": ["WRS300"], "dependencies": []}}
    elif kind == "rejected_action":
        divergence = {"name": "step", "arguments": {
            "function": "FilterStr", "inputs": ["attribute", "current_rating"], "dependencies": [999]}}
    else:
        raise ValueError("Unknown CPU fixture kind")
    # End with a deliberately broad record selection to exercise failure input.
    actions = reference["actions"][:2] + [divergence, {
        "name": "finish_records", "arguments": {"record_handle": 1}}]
    episode = RecordSetEpisode(executor, store)
    messages = prompt_messages(reference["question"])
    for action in actions:
        text = action_text(action)
        observation = execute_response(episode, text)
        messages.extend([{"role": "assistant", "content": text},
                         {"role": "tool", "content": compact(observation)}])
    ident = f"CPU-FIXTURE:{reference['id']}:{kind}"
    return {"id": reference["id"], "fixture_id": ident, "messages": messages,
            "events": episode.events, "calls": episode.calls, "answer": episode.answer,
            "provenance": {"kind": "hand_constructed_cpu_fixture", "fixture_id": ident},
            "training_eligible": False}


def main():
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("Preserve CPU validation evidence")
    references = read_jsonl(REFERENCES)
    if len(references) != 7:
        raise ValueError("This bounded check uses the seven prepared development references")
    executor, store = load_bridge()
    success, rollouts, clean, recovery, audits = [], [], [], [], []
    for reference in references:
        row = successful_trajectory(executor, store, reference)
        row["data_grade"] = "development_reference_cpu_not_training_split"
        success.append(row)
        for kind in ("extra_handle", "rejected_action"):
            source = constructed_rollout(executor, store, reference, kind)
            pair = make_pair(executor, store, source, reference)
            if pair["audit"]["status"] != "accepted":
                raise ValueError(pair["audit"])
            rollouts.append(source)
            audit = {**pair["audit"], "fixture_id": source["fixture_id"], "fixture_kind": kind}
            audits.append(audit)
            for name, target in (("clean", clean), ("recovery", recovery)):
                row = pair[name]
                row.update(id=source["fixture_id"], pair_id=source["fixture_id"],
                           reference_id=reference["id"], data_grade="hand_constructed_cpu_fixture",
                           fixture_kind=kind, training_eligible=False)
                target.append(row)
    PRIVATE.mkdir(parents=True)
    for name, rows in (("success.reference.jsonl", success), ("fixture.rollouts.jsonl", rollouts),
                       ("fixture.clean.jsonl", clean), ("fixture.recovery.jsonl", recovery),
                       ("fixture.audits.jsonl", audits)):
        write_lines(PRIVATE / name, rows)
    inputs = [REFERENCES, Path(__file__).relative_to(Path.cwd()),
              Path("experiments/radar_domain/recovery_data.py"),
              Path("experiments/radar_domain/record_set_interface.py"),
              Path("experiments/radar_domain/kopl_record_bridge.py"),
              Path("data/radar_domain_recovery_v1/kopl_bridge_20261106/kb.json"),
              Path("data/radar_domain_recovery_v1/kopl_bridge_20261106/source_chunks.json")]
    report = {"schema": "radar_domain_recovery_cpu_validation_v1", "date": "2026-11-06",
              "development_only": True, "human_gold": False, "training_started": False,
              "actual_model_rollouts": 0, "actual_model_recovery_pairs": 0,
              "successful_development_references": len(success),
              "hand_constructed_cpu_fixture_rollouts": len(rollouts),
              "cpu_fixture_pairs_constructed": len(clean),
              "fixture_kinds": dict(Counter(row["fixture_kind"] for row in audits)),
              "answer_kinds": dict(Counter(row["answer"]["kind"] for row in success)),
              "all_reference_and_fixture_replays_passed": True,
              "all_pair_targets_semantically_equal": True,
              "loss_mask": "Question/prefix/divergence/tool observations masked; only reference suffix assistant actions and Done supervised",
              "tokenization_checked": False, "supervised_token_budget_matched": False,
              "gpu_model_calls": 0,
              "production_gate": "Train split, question-only generation inputs, completed model-generated origin inventory, per-row provenance and exact model/code/KB hashes required",
              "scope": "Seven source-derived reference traces and explicitly constructed CPU fixtures. This is not a trained domain model, real failure collection or a complete twelve-task evaluation.",
              "inputs_sha256": {str(p): sha256(p) for p in inputs},
              "outputs_sha256": {str(p): sha256(p) for p in PRIVATE.glob("*.jsonl")}}
    write_new(PUBLIC / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
