# -*- coding: utf-8 -*-
"""LoRA fine-tune Qwen-7B-Chat as a RoG-style path planner on KG-mined data.
Runs in the `mixrag` conda env (transformers 4.49 + peft 0.7.1) on GPU 0.

  CUDA_VISIBLE_DEVICES=0 python train_lora.py --data train_data.jsonl --out adapter
"""
import os, json, glob, argparse, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model
from rog_common import get_planner_system, build_prompt, PLANNER_USER_TPL

MODEL_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/*/")


class PlanDataset(Dataset):
    def __init__(self, rows, tok, system, max_len=1536, fmt="qwen"):
        self.ex = []
        n_trunc = 0
        for r in rows:
            user = PLANNER_USER_TPL.format(question=r["question"])
            prefix = build_prompt(fmt, system, user, assistant=None)
            full = build_prompt(fmt, system, user, assistant=r["plan"])
            pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            full_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_len:
                n_trunc += 1
                full_ids = full_ids[:max_len]
            labels = list(full_ids)
            # mask the prompt prefix
            for i in range(min(len(pre_ids), len(labels))):
                labels[i] = -100
            self.ex.append({"input_ids": full_ids, "labels": labels,
                            "attention_mask": [1] * len(full_ids)})
        if n_trunc:
            print(f"[!] {n_trunc} examples truncated at max_len={max_len}")

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_data.jsonl")
    ap.add_argument("--lexicon", default="lexicon")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="", help="base model path; empty = cached Qwen glob")
    ap.add_argument("--fmt", default="qwen", choices=["qwen", "llama2"])
    ap.add_argument("--targets", default="c_attn", help="comma-sep LoRA target modules")
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--grad_ckpt", type=int, default=0, help="1=enable gradient checkpointing")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    snap = args.base if args.base else glob.glob(MODEL_GLOB)[0]
    print(f"[*] base model: {snap}  fmt={args.fmt}  targets={args.targets}  max_len={args.max_len}")
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = getattr(tok, "eod_id", None) or tok.eos_token_id
    pad_id = tok.pad_token_id

    system = get_planner_system(args.lexicon)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.shuffle(rows)
    n_dev = max(12, len(rows) // 10)
    dev_rows, train_rows = rows[:n_dev], rows[n_dev:]
    print(f"[*] train={len(train_rows)} dev={len(dev_rows)}")

    if args.fmt == "qwen":
        model = AutoModelForCausalLM.from_pretrained(
            snap, trust_remote_code=True, bf16=True, device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            snap, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=[t.strip() for t in args.targets.split(",")],
                    task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    train_ds = PlanDataset(train_rows, tok, system, max_len=args.max_len, fmt=args.fmt)
    dev_ds = PlanDataset(dev_rows, tok, system, max_len=args.max_len, fmt=args.fmt)

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[+] adapter saved to {args.out}")


if __name__ == "__main__":
    main()
