# -*- coding: utf-8 -*-
import os, json, statistics
from transformers import AutoTokenizer
from rog_common import get_planner_system, build_llama2, PLANNER_USER_TPL

base = os.path.expanduser("~/rog_ft/llama2-7b-chat")
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
system = get_planner_system("lexicon")
print("LLaMA-2 system prompt tokens:", len(tok(system, add_special_tokens=False)["input_ids"]))
rows = [json.loads(l) for l in open("train_data.jsonl", encoding="utf-8")]
lens = []
for r in rows:
    user = PLANNER_USER_TPL.format(question=r["question"])
    full = build_llama2(system, user, assistant=r["plan"])
    lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
print("full example tokens: min=%d max=%d mean=%.0f p95=%d" % (
    min(lens), max(lens), statistics.mean(lens), sorted(lens)[int(0.95 * len(lens))]))
