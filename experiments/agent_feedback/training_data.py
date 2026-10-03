"""Exact chat-template tokenization with explicit action-only supervision."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.condition_consistency.executor import serialize_program
from .environment import FUNCTION_HELP, tool_schemas


def tokenize_trajectory(tokenizer, row: dict, max_length: int = 8192, tools=True):
    messages = row["messages"]
    if "supervise" not in row:
        raise ValueError("An explicit supervision mask is required, including for recovery traces")
    supervision = row["supervise"]
    if len(messages) != len(supervision):
        raise ValueError("Message supervision mask length differs")
    kwargs = {"tools": tool_schemas()} if tools else {}
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False, **kwargs)
    if len(ids) > max_length:
        raise ValueError(f"Context overflow for {row['id']}: {len(ids)} > {max_length}; do not silently truncate")
    labels = [-100] * len(ids)
    spans = []
    for i, (message, supervise) in enumerate(zip(messages, supervision)):
        if not supervise:
            continue
        if message["role"] != "assistant":
            raise ValueError("Only assistant actions may receive supervision")
        before = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_dict=False, add_generation_prompt=True, **kwargs)
        after = tokenizer.apply_chat_template(messages[:i + 1], tokenize=True, return_dict=False, add_generation_prompt=False, **kwargs)
        if ids[:len(before)] != before or ids[:len(after)] != after:
            raise ValueError(f"Chat template changed previous token boundaries for {row['id']} turn {i}")
        if not len(before) < len(after):
            raise ValueError("Empty supervised action")
        labels[len(before):len(after)] = ids[len(before):len(after)]
        spans.append([i, len(before), len(after)])
    if not any(x != -100 for x in labels):
        raise ValueError("No supervised actions")
    return {"id": row["id"], "input_ids": ids, "labels": labels,
            "supervised_tokens": sum(x != -100 for x in labels), "spans": spans}


def program_trajectory(row):
    messages = [{"role": "system", "content":
                 "Translate the question into a complete executable KoPL program. "
                 "Output only function and literal inputs separated by <arg>, with steps separated by <func>. "
                 "Dependencies follow the standard branch-stack order.\n" + FUNCTION_HELP},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": serialize_program(row["program"])}]
    return {"id": row["id"], "messages": messages, "supervise": [False, False, True]}


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(x) for x in handle if x.strip()]
