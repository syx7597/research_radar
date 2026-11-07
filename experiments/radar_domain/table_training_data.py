"""Exact table-tool tokenization and equal supervised-token pairing."""
from copy import deepcopy

from .table_token_preflight import tokenize


def pack(tokenizer, trajectory, max_context=16384):
    ids, spans = tokenize(tokenizer, trajectory, max_context)
    labels = [-100] * len(ids)
    for span in spans:
        labels[span["start"]:span["end"]] = ids[span["start"]:span["end"]]
    return {"id": trajectory["id"], "input_ids": ids, "labels": labels,
            "supervised_tokens": sum(x != -100 for x in labels), "spans": spans,
            "training_eligible": trajectory.get("training_eligible", False)}


def balanced_pair(tokenizer, clean, recovery, max_context=16384):
    if clean["id"] != recovery["id"]:
        raise ValueError("Paired trajectories must refer to the same question")
    a, c = pack(tokenizer, clean, max_context), pack(tokenizer, recovery, max_context)
    target = min(a["supervised_tokens"], c["supervised_tokens"])
    if target <= 0:
        raise ValueError("Both trajectories need supervised assistant targets")
    for row in (a, c):
        remaining = target
        masked = 0
        for i, value in enumerate(row["labels"]):
            if value == -100:
                continue
            if remaining:
                remaining -= 1
            else:
                row["labels"][i] = -100
                masked += 1
        row["supervised_tokens"] = target
        row["budget_masked_tokens"] = masked
    return a, c
