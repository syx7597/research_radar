# -*- coding: utf-8 -*-
"""Run the fine-tuned Qwen planner on the 499 eval questions, emit rog_ft_plans.json.
Runs in mixrag env on GPU 0. Parsing mirrors qa_rog_baseline.plan_paths exactly so
the downstream walker/reasoner are identical to the zero-shot RoG condition.

  CUDA_VISIBLE_DEVICES=0 python infer_plans.py --adapter adapter --out rog_ft_plans.json
"""
import os, re, json, glob, argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rog_common import get_planner_system, build_prompt, PLANNER_USER_TPL

MODEL_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/*/")


def parse_plan(raw):
    start_m = re.search(r"START\s*[:：]\s*(.+?)(?:\n|$)", raw)
    start = start_m.group(1).strip() if start_m else ""
    paths = []
    for pm in re.finditer(r"<PATH>(.*?)</PATH>", raw, flags=re.DOTALL):
        body = pm.group(1).strip()
        rels = []
        for tok in body.split("<SEP>"):
            tok = tok.strip()
            if tok:
                rels.append(tok)  # keep ^-1 marker inline; walker parses it
        if rels:
            paths.append(rels)
    return start, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapter")
    ap.add_argument("--questions", default="results/qa500_3way_full.json")
    ap.add_argument("--lexicon", default="lexicon")
    ap.add_argument("--out", default="rog_ft_plans.json")
    ap.add_argument("--base", default="", help="base model path; empty = cached Qwen glob")
    ap.add_argument("--fmt", default="qwen", choices=["qwen", "llama2"])
    args = ap.parse_args()

    snap = args.base if args.base else glob.glob(MODEL_GLOB)[0]
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = getattr(tok, "eod_id", None) or tok.eos_token_id
    if args.fmt == "qwen":
        base = AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True, bf16=True, device_map="cuda:0").eval()
    else:
        base = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    if args.adapter.lower() in ("none", "base", ""):
        model = base
        print(f"[*] base model loaded (NO adapter — zero-shot control) fmt={args.fmt}")
    else:
        model = PeftModel.from_pretrained(base, args.adapter).eval()
        print(f"[*] model + adapter loaded fmt={args.fmt}")

    system = get_planner_system(args.lexicon)
    data = json.load(open(args.questions, encoding="utf-8"))["per_question"]
    if args.fmt == "qwen":
        stop_id = tok("<|im_end|>", add_special_tokens=False)["input_ids"][-1]
    else:
        stop_id = tok.eos_token_id

    out = []
    t0 = time.time()
    for i, q in enumerate(data):
        user = PLANNER_USER_TPL.format(question=q["question"])
        prompt = build_prompt(args.fmt, system, user, assistant=None)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                 eos_token_id=stop_id, pad_token_id=tok.pad_token_id)
        raw = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        start, paths = parse_plan(raw)
        out.append({"id": q["id"], "type": q["type"], "question": q["question"],
                    "start": start, "paths": paths, "raw": raw})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(data)}  ({time.time()-t0:.0f}s)  last: start={start!r} paths={paths}")

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_start = sum(1 for o in out if o["start"])
    n_paths = sum(1 for o in out if o["paths"])
    print(f"\n[+] wrote {len(out)} plans to {args.out}")
    print(f"[+] non-empty START: {n_start}/{len(out)}  non-empty paths: {n_paths}/{len(out)}")


if __name__ == "__main__":
    main()
