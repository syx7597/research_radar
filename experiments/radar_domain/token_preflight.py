"""CPU chat-template and supervision checks for domain development trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform

from experiments.agent_feedback.recovery import sha256
from experiments.agent_feedback.training_data import read_jsonl
from .record_set_interface import load_bridge, tool_schemas


def tokenize(tokenizer, trajectory, max_length=8192):
    messages, supervision = trajectory["messages"], trajectory["supervise"]
    if len(messages) != len(supervision) or any(type(s) is not bool for s in supervision):
        raise ValueError("An explicit per-message boolean supervision mask is required")
    schema = tool_schemas()
    ids = tokenizer.apply_chat_template(messages, tools=schema, tokenize=True,
                                        return_dict=False, add_generation_prompt=False)
    if len(ids) > max_length:
        raise ValueError(f"Full trajectory exceeds context budget: {len(ids)} > {max_length}")
    labels = [-100] * len(ids)
    spans = []
    for i, (message, selected) in enumerate(zip(messages, supervision)):
        if not selected:
            continue
        if message["role"] != "assistant":
            raise ValueError("Only assistant actions/Done may receive supervision")
        before = tokenizer.apply_chat_template(messages[:i], tools=schema, tokenize=True,
            return_dict=False, add_generation_prompt=True)
        after = tokenizer.apply_chat_template(messages[:i+1], tools=schema, tokenize=True,
            return_dict=False, add_generation_prompt=False)
        if ids[:len(before)] != before or ids[:len(after)] != after or len(after) <= len(before):
            raise ValueError("The chat template changes preceding token boundaries")
        labels[len(before):len(after)] = ids[len(before):len(after)]
        spans.append({"message_index": i, "start": len(before), "end": len(after),
                      "tokens": len(after)-len(before),
                      "is_done_text": message["content"] == "Done."})
    if not spans:
        raise ValueError("No supervised assistant targets")
    return {"input_ids": ids, "labels": labels, "spans": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path,
        default=Path("data/radar_domain_recovery_v1/reference_dev_20261106/references.jsonl"))
    parser.add_argument("--trajectories", type=Path, help="Optional CPU test-fixture trajectories only")
    parser.add_argument("--tokenizer", type=Path, default=Path("cache/qwen2.5-3b-tokenizer"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep completed preflight evidence")
    original = Path("results/agent_feedback/clean_sft_v1/input_manifest.json")
    expected = json.loads(original.read_text())["continuation"]["tokenizer_files_sha256"]
    tokenizer_files = [args.tokenizer / name for name in expected]
    for path in tokenizer_files:
        if sha256(path) != expected[path.name]:
            raise ValueError("Tokenizer identity differs from the original Qwen revision")
    import transformers
    import tokenizers
    import jinja2
    from transformers import AutoTokenizer
    if transformers.__version__ != "5.18.0" or tokenizers.__version__ != "0.23.2":
        raise ValueError("Use the pinned tokenizer implementation")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    from .recovery_data import successful_trajectory
    executor, store = load_bridge()
    references = read_jsonl(args.references)
    trajectories = [successful_trajectory(executor, store, ref) for ref in references]
    labelled = [("development_reference", row) for row in trajectories]
    if args.trajectories:
        rows = read_jsonl(args.trajectories)
        if any(row.get("training_eligible") is not False for row in rows):
            raise ValueError("Optional input is limited to explicitly non-training CPU fixtures")
        labelled.extend(("hand_constructed_cpu_fixture", row) for row in rows)
    checks = []
    for kind, row in labelled:
        result = tokenize(tokenizer, row)
        actions = [s for s in result["spans"] if not s["is_done_text"]]
        checks.append({
            "id": row["id"], "input_kind": kind,
            "full_context_tokens": len(result["input_ids"]),
            "supervised_tokens": sum(x != -100 for x in result["labels"]),
            "supervised_action_messages": len(actions),
            "maximum_supervised_action_tokens": max((s["tokens"] for s in actions), default=0),
            "supervised_continuation_tokens": sum(s["tokens"] for s in actions),
            "within_8192_context": len(result["input_ids"]) <= 8192,
            "within_192_per_turn": all(s["tokens"] <= 192 for s in actions),
            "within_2048_supervised_continuation": sum(s["tokens"] for s in actions) <= 2048,
            "only_assistant_supervised": all(row["messages"][s["message_index"]]["role"] == "assistant"
                                            for s in result["spans"]),
        })
    # Exercise rejection using an actual tokenizer, not a token-count mock.
    first = trajectories[0]
    try:
        tokenize(tokenizer, first, max_length=1)
    except ValueError:
        overflow_rejected = True
    else:
        raise AssertionError("Overflow was silently accepted")
    from copy import deepcopy
    invalid = deepcopy(first)
    tool_index = next(i for i, m in enumerate(invalid["messages"]) if m["role"] == "tool")
    invalid["supervise"][tool_index] = True
    try:
        tokenize(tokenizer, invalid)
    except ValueError:
        tool_supervision_rejected = True
    else:
        raise AssertionError("Tool output was accepted as a target")
    paths = [args.references, original, *tokenizer_files, Path(__file__),
             Path("experiments/radar_domain/record_set_interface.py"),
             Path("experiments/radar_domain/recovery_data.py"),
             Path("experiments/radar_domain/kopl_record_bridge.py")]
    if args.trajectories:
        paths.append(args.trajectories)
    report = {
        "schema": "radar_domain_cpu_token_preflight_v1", "date": "2026-11-06",
        "development_only": True, "formal_training_ready": False, "model_weights_loaded": False,
        "model_generation_performed": False, "training_started": False,
        "tools": [x["function"]["name"] for x in tool_schemas()],
        "versions": {"python": platform.python_version(), "platform": platform.system(),
                     "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                     "jinja2": jinja2.__version__},
        "counts": dict(Counter(kind for kind, _ in labelled)), "checks": checks,
        "all_budget_checks_passed": all(c["within_8192_context"] and c["within_192_per_turn"]
            and c["within_2048_supervised_continuation"] for c in checks),
        "overflow_rejected": overflow_rejected, "tool_supervision_rejected": tool_supervision_rejected,
        "scope": "Actual tokenizer checks on developer references and explicit test fixtures. Full tool answers stay in context and never in supervised labels. Generated-budget estimates include chat-template end markers; no measured model generation.",
        "inputs_sha256": {str(p): sha256(p) for p in paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, ensure_ascii=False, indent=2); out.write("\n")
    print(json.dumps({"counts": report["counts"],
        "maximum_context_tokens": max(c["full_context_tokens"] for c in checks),
        "all_budget_checks_passed": report["all_budget_checks_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
