# -*- coding: utf-8 -*-
"""SFT 冷启动训练(在 4×4090 上跑) —— Qwen2.5-7B + LoRA,吃 sft_train.jsonl 多轮轨迹。
只对 assistant 轮(工具调用/finish)算 loss,tool 观察作为输入不算 loss。

冒烟(CPU,验证数据/分词/掩码,不训练):
    python ca_agraphrag/train_sft.py --dry
单卡:
    python ca_agraphrag/train_sft.py
4×4090:
    accelerate launch --multi_gpu --num_processes 4 ca_agraphrag/train_sft.py
依赖:transformers>=4.44, peft, accelerate, datasets, torch(bf16)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ca_agraphrag" / "data" / "sft_train.jsonl"
OUT = ROOT / "ca_agraphrag" / "ckpt_sft"

# ---------------- 配置 ----------------
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 2048
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(per_device_train_batch_size=2, gradient_accumulation_steps=8,
             learning_rate=1e-4, num_train_epochs=2, warmup_ratio=0.03,
             lr_scheduler_type="cosine", logging_steps=10, save_steps=200,
             bf16=True, gradient_checkpointing=True, report_to="none")

# tool 观察作为一条 user 消息喂回(便携,不依赖各模型的 tool-role 模板)
ROLE_MAP = {"tool": "user"}


def to_chat(msgs):
    return [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m["content"]} for m in msgs]


def build_example(tok, msgs):
    """对多轮对话分词,labels 只保留 assistant 轮的 token(其余 -100)。"""
    chat = to_chat(msgs)
    input_ids, labels = [], []
    for i, m in enumerate(chat):
        # 逐轮增量套用 chat template,取本轮新增的 token
        prev = tok.apply_chat_template(chat[:i], tokenize=True,
                                       add_generation_prompt=(m["role"] == "assistant")) if i else []
        cur = tok.apply_chat_template(chat[:i + 1], tokenize=True, add_generation_prompt=False)
        seg = cur[len(prev):]
        input_ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
    input_ids, labels = input_ids[:MAX_LEN], labels[:MAX_LEN]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def load_data(tok):
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [build_example(tok, r["messages"]) for r in rows]


def dry():
    """CPU 冒烟:验证数据加载/分词/掩码,不下载大模型权重(只要 tokenizer)。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()[:3] if l.strip()]
    for r in rows:
        ex = build_example(tok, r["messages"])
        n_lab = sum(1 for x in ex["labels"] if x != -100)
        print(f"[{r['type']}] tokens={len(ex['input_ids'])} 计loss的assistant token={n_lab} "
              f"({n_lab/len(ex['input_ids']):.0%})")
        # 还原 assistant 部分,肉眼核对掩码对不对
        asst = tok.decode([x for x in ex["labels"] if x != -100])
        print(f"   assistant(计loss)片段: {asst[:120]}...")
    print(f"\n总轨迹数: {sum(1 for _ in DATA.open(encoding='utf-8'))}")
    print("冒烟通过:数据/分词/掩码管线 OK。上 GPU 前把 MODEL 权重下好即可。")


def train():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list(load_data(tok))
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()
    if TRAIN["gradient_checkpointing"]:
        model.enable_input_require_grads()

    args = TrainingArguments(output_dir=str(OUT), **TRAIN)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(OUT / "adapter")
    tok.save_pretrained(OUT / "adapter")
    print(f"[SFT] LoRA 适配器已保存 -> {OUT/'adapter'}")


if __name__ == "__main__":
    dry() if "--dry" in sys.argv else train()
