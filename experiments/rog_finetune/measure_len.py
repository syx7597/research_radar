# -*- coding: utf-8 -*-
import json, glob, os, statistics
from transformers import AutoTokenizer
from rog_common import get_planner_system, build_chatml, PLANNER_USER_TPL

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/*/"))[0]
tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
system = get_planner_system("lexicon")
print("system prompt tokens:", len(tok(system, add_special_tokens=False)["input_ids"]))
rows = [json.loads(l) for l in open("train_data.jsonl", encoding="utf-8")]
lens, plan_lens = [], []
for r in rows:
    user = PLANNER_USER_TPL.format(question=r["question"])
    full = build_chatml(system, user, assistant=r["plan"])
    lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
    plan_lens.append(len(tok(r["plan"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]))
print("full example tokens: min=%d max=%d mean=%.0f p95=%d" % (
    min(lens), max(lens), statistics.mean(lens), sorted(lens)[int(0.95 * len(lens))]))
print("plan(target) tokens: min=%d max=%d mean=%.0f" % (min(plan_lens), max(plan_lens), statistics.mean(plan_lens)))
