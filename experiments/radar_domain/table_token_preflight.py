"""Measure every frozen reference with the actual pinned tokenizer, on CPU."""
from collections import Counter
import json
from pathlib import Path

from experiments.agent_feedback.recovery import sha256
from .table_program import tool_schemas

ROOT = Path("data/radar_domain_recovery_v2/frozen_20261107")
OUTPUT = Path("results/radar_domain/frozen_20261107/token_preflight.json")


def tokenize(tokenizer, row, max_context=32768):
    messages, masks = row["messages"], row["supervise"]
    if len(messages) != len(masks) or any(type(m) is not bool for m in masks):
        raise ValueError("Explicit boolean message mask required")
    kwargs = {"tools": tool_schemas(), "tokenize": True, "return_dict": False}
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if len(ids) > max_context:
        raise ValueError("Reference exceeds the maximum supported context")
    spans = []
    for i, (message, selected) in enumerate(zip(messages, masks)):
        if not selected:
            continue
        if message["role"] != "assistant":
            raise ValueError("Tool output cannot be a supervised target")
        before = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **kwargs)
        after = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, **kwargs)
        if ids[:len(before)] != before or ids[:len(after)] != after or len(after) <= len(before):
            raise ValueError("Chat template prefix boundary changed")
        spans.append({"message": i, "tokens": len(after)-len(before),
                      "start": len(before), "end": len(after), "done": message["content"] == "Done."})
    if not spans:
        raise ValueError("No supervised targets")
    return ids, spans


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep completed budget evidence")
    import transformers, tokenizers
    from transformers import AutoTokenizer
    tokenizer_path = Path("cache/qwen2.5-3b-tokenizer")
    original = Path("results/agent_feedback/clean_sft_v1/input_manifest.json")
    hashes = json.loads(original.read_text())["continuation"]["tokenizer_files_sha256"]
    for name, digest in hashes.items():
        if sha256(tokenizer_path/name) != digest:
            raise ValueError("Original tokenizer identity differs")
    if transformers.__version__ != "5.18.0" or tokenizers.__version__ != "0.23.2":
        raise ValueError("Use pinned tokenizer implementation")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    manifest_path = ROOT / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output, input_paths = {}, [manifest_path, original, Path(__file__),
        Path("experiments/radar_domain/table_program.py"), *[tokenizer_path/n for n in hashes]]
    checks = []
    for split, entry in manifest["splits"].items():
        path = Path(entry["success_trajectories"]["path"])
        if sha256(path) != entry["success_trajectories"]["sha256"]:
            raise ValueError("Frozen trajectories changed")
        input_paths.append(path)
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        counts = []
        for i, row in enumerate(rows):
            ids, spans = tokenize(tokenizer, row)
            action_spans = [s for s in spans if not s["done"]]
            counts.append({"id": row["id"], "origin": row["origin"], "context_tokens": len(ids),
                "supervised_tokens": sum(s["tokens"] for s in spans),
                "max_action_tokens": max(s["tokens"] for s in action_spans),
                "total_action_tokens": sum(s["tokens"] for s in action_spans),
                "supervised_assistant_messages": len(spans)})
            if (i+1) % 100 == 0:
                print(json.dumps({"split": split, "tokenized": i+1}), flush=True)
        checks.extend(dict(x, split=split) for x in counts)
        output[split] = {"questions": len(rows), "max_context_tokens": max(x["context_tokens"] for x in counts),
            "max_action_tokens": max(x["max_action_tokens"] for x in counts),
            "max_total_action_tokens": max(x["total_action_tokens"] for x in counts),
            "total_supervised_tokens": sum(x["supervised_tokens"] for x in counts),
            "origin_counts": dict(Counter(x["origin"] for x in counts))}
    all_context = max(x["context_tokens"] for x in checks)
    all_action = max(x["max_action_tokens"] for x in checks)
    all_generated = max(x["total_action_tokens"] for x in checks)
    chosen_context = next((n for n in (8192, 16384, 32768) if all_context <= n), None)
    chosen_turn = next((n for n in (192, 256, 384, 512) if all_action <= n), None)
    if chosen_context is None or chosen_turn is None or all_generated > 2048:
        raise ValueError("Reference actions require protocol revision before any model run")
    report = {"schema": "radar_table_frozen_token_preflight_v1", "date": "2026-11-07",
        "reference_trajectories": len(checks), "splits": output,
        "all_prefix_boundaries_checked": True, "only_assistant_actions_supervised": True,
        "maximum_context_tokens": all_context, "maximum_single_action_tokens": all_action,
        "maximum_total_action_tokens": all_generated,
        "registered_runtime_budget": {"context_tokens": chosen_context, "per_turn_generated_tokens": chosen_turn,
                                     "total_generated_tokens": 2048, "max_calls": 24},
        "budget_selection_rule": "Smallest registered context in 8192/16384/32768 and turn cap in 192/256/384/512 that fits all source-derived references, before any model run",
        "tokenizer_files_sha256": hashes, "versions": {"transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__}, "model_weights_loaded": False,
        "model_runs": 0, "training_started": False,
        "inputs_sha256": {str(p): sha256(p) for p in input_paths}}
    private = ROOT / "token_checks.jsonl"
    with private.open("x") as stream:
        for row in checks:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    report["private_checks"] = {"path": str(private), "sha256": sha256(private)}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"questions": len(checks), "max_context": all_context,
                     "max_action": all_action, "budget": report["registered_runtime_budget"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
