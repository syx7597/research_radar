"""LoRA action-only SFT with Transformers Trainer; no home-made optimizer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .training_data import program_trajectory, read_jsonl, tokenize_trajectory


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_continuation_options(args):
    if not args.pretokenized:
        return
    if not args.adapter or args.epochs != 1 or args.max_steps != -1 or args.limit != 0:
        raise ValueError("Pretokenized continuation requires --adapter, --epochs 1, --max-steps -1 and --limit 0")
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Continuation requires the existing SFT adapter config and safetensors weights")


def validate_continuation_manifest(args, examples, tokenizer_identity, transformers_version):
    """The builder writes this manifest last; both completed artifacts must match."""
    path = Path(args.continuation_manifest)
    if not path.is_file():
        raise FileNotFoundError("Continuation completion manifest is missing; do not train partial artifacts")
    manifest = json.loads(path.read_text())
    arms, budget = manifest["arms"], manifest["budget"]
    if set(arms) != {"A", "C"} or not examples:
        raise ValueError("Continuation manifest requires two nonempty A/C arms")
    if (type(budget["total"]) is not int or budget["total"] < 2 or
            budget["paired_suffix"] != budget["total"] // 2 or
            budget["ordinary"] != budget["total"] - budget["paired_suffix"]):
        raise ValueError("Continuation manifest has an invalid matched-token budget")
    if args.max_length != manifest["max_context"]:
        raise ValueError("Continuation context setting differs from the completed corpus")
    selected = []
    for arm, spec in arms.items():
        artifact = Path(spec["path"])
        if not artifact.is_file() or digest(artifact) != spec["sha256"]:
            raise ValueError(f"Completed continuation artifact {arm} is missing or changed")
        if (spec["records"] < 1 or spec["supervised_tokens"] != budget["total"] or
                spec["ordinary_supervised_tokens"] != budget["ordinary"] or
                spec["paired_suffix_supervised_tokens"] != budget["paired_suffix"]):
            raise ValueError(f"Continuation arm {arm} does not satisfy the frozen budget")
        if artifact.resolve() == Path(args.data).resolve():
            selected.append(arm)
    if len(selected) != 1 or arms["A"]["records"] != arms["C"]["records"]:
        raise ValueError("Input must be exactly one completed A/C artifact with matched record counts")
    arm = selected[0]
    spec = arms[arm]
    expected_files = manifest["tokenizer"]["files_sha256"]
    tokenizer_files = {name: value for name, value in tokenizer_identity.items()
                       if name in {"tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"}}
    if not {"tokenizer_config.json", "tokenizer.json"} <= set(expected_files) or tokenizer_files != expected_files:
        raise ValueError("Tokenizer files differ from those used to construct continuation targets")
    if manifest["tokenizer"]["transformers"] != transformers_version:
        raise ValueError("Transformers version differs from the continuation tokenization environment")
    # The present builder records an inline Qwen template in tokenizer_config.
    # An unrecorded external template could silently override it at load time.
    if "chat_template.jinja" in tokenizer_identity and "chat_template.jinja" not in expected_files:
        raise ValueError("External chat template is not covered by the continuation tokenizer identity")
    if any(row["kind"] not in {"ordinary", "paired_suffix"} for row in examples):
        raise ValueError("Unexpected continuation record kind")
    if len({row["id"] for row in examples}) != len(examples):
        raise ValueError("Continuation record IDs must be unique")
    actual = {"records": len(examples),
              "supervised_tokens": sum(row["supervised_tokens"] for row in examples),
              "input_tokens": sum(len(row["input_ids"]) for row in examples),
              "max_input_length": max(len(row["input_ids"]) for row in examples),
              "ordinary_supervised_tokens": sum(row["supervised_tokens"] for row in examples if row["kind"] == "ordinary"),
              "paired_suffix_supervised_tokens": sum(row["supervised_tokens"] for row in examples if row["kind"] == "paired_suffix")}
    if any(value != spec[key] for key, value in actual.items()):
        raise ValueError("Actual continuation records differ from the completed manifest")
    adapter = Path(args.adapter)
    return {"arm": arm, "manifest_sha256": digest(path), "artifact_sha256": spec["sha256"],
            "expected_supervised_tokens": budget["total"],
            "adapter_config_sha256": digest(adapter / "adapter_config.json"),
            "adapter_weights_sha256": digest(adapter / "adapter_model.safetensors"),
            "tokenizer_files_sha256": tokenizer_files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/agent_feedback/train.success.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--program", action="store_true")
    parser.add_argument("--pretokenized", action="store_true",
                        help="Use the frozen, matched-target-token continuation corpus")
    parser.add_argument("--continuation-manifest", default="results/agent_feedback/continuation_data.json",
                        help="Completion manifest written after both frozen A/C artifacts")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tokenize-only", action="store_true")
    args = parser.parse_args()
    if args.program and args.pretokenized:
        raise ValueError("Program conversion and pretokenized input are mutually exclusive")
    validate_continuation_options(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_result.json").exists():
        raise FileExistsError("Run already completed; do not overwrite results")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = read_jsonl(args.data)
    if args.program:
        valid_ids = {x["id"] for x in read_jsonl("data/agent_feedback/train.success.jsonl")}
        raw = [program_trajectory(x) for x in raw if x["id"] in valid_ids]
    if args.limit:
        raw = raw[:args.limit]
    tokenizer_identity = {p.name: digest(p) for p in Path(args.model).iterdir()
                          if p.name in {"tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                                        "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"}}
    key = hashlib.sha256(json.dumps({"data": digest(args.data), "program": args.program,
        "limit": args.limit, "max_length": args.max_length,
        "tokenizer_files": tokenizer_identity,
        "tokenizer_libraries": {p: importlib.metadata.version(p) for p in ["transformers", "tokenizers", "jinja2"]},
        "program_eligibility": digest("data/agent_feedback/train.success.jsonl") if args.program else None,
        "tokenizer_code": digest(Path(__file__).with_name("training_data.py")),
        "environment": digest(Path(__file__).with_name("environment.py"))}, sort_keys=True).encode()).hexdigest()
    cache = Path("data/agent_feedback/tokenized") / (key + ".jsonl")
    continuation = None
    if args.pretokenized:
        examples = raw
        cache = Path(args.data)
        for row in examples:
            ids, labels = row["input_ids"], row["labels"]
            if not ids or len(ids) != len(labels) or len(ids) > args.max_length:
                raise ValueError("Invalid pretokenized lengths")
            if any(type(x) is not int or x < 0 for x in ids):
                raise ValueError("Invalid pretokenized token IDs")
            if any(type(y) is not int or (y != -100 and y != x) for x, y in zip(ids, labels)):
                raise ValueError("Pretokenized labels are not aligned with input tokens")
            actual = sum(y != -100 for y in labels)
            if actual < 1 or actual != row["supervised_tokens"]:
                raise ValueError("Pretokenized supervised-token budget differs")
        continuation = validate_continuation_manifest(args, examples, tokenizer_identity,
                                                      importlib.metadata.version("transformers"))
    elif cache.exists():
        examples = read_jsonl(cache)
    else:
        examples = []
        for i, row in enumerate(raw):
            examples.append(tokenize_trajectory(tokenizer, row, args.max_length, tools=not args.program))
            if (i + 1) % 500 == 0:
                print(json.dumps({"tokenized": i + 1}), flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in examples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(cache)
    summary = {"examples": len(examples), "input_tokens": sum(len(x["input_ids"]) for x in examples),
               "supervised_tokens": sum(x["supervised_tokens"] for x in examples),
               "max_length": max(len(x["input_ids"]) for x in examples),
               "token_cache_sha256": digest(cache), "config": vars(args),
               "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "accelerate", "trl"]}}
    if continuation is not None:
        summary["continuation"] = continuation
    (output / "input_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if args.tokenize_only:
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return {k: examples[i][k] for k in ("input_ids", "labels")}

    def collate(rows):
        length = (max(len(x["input_ids"]) for x in rows) + 7) // 8 * 8
        return {"input_ids": torch.tensor([x["input_ids"] + [tokenizer.pad_token_id] * (length - len(x["input_ids"])) for x in rows]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (length - len(x["input_ids"])) for x in rows]),
                "labels": torch.tensor([x["labels"] + [-100] * (length - len(x["labels"])) for x in rows])}

    training_args = TrainingArguments(output_dir=str(output),
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
            num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.learning_rate,
            warmup_ratio=0.03, weight_decay=0.0, bf16=True, tf32=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", logging_steps=10, save_strategy="epoch", save_total_limit=2,
            report_to="none", seed=args.seed, data_seed=args.seed, dataloader_num_workers=0,
            remove_unused_columns=False, disable_tqdm=True)
    class AccountedTrainer(Trainer):
        supervised_tokens_seen = 0
        input_tokens_seen = 0
        microbatches_seen = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if model.training:
                self.supervised_tokens_seen += int(inputs["labels"].ne(-100).sum().item())
                self.input_tokens_seen += int(inputs["attention_mask"].sum().item())
                self.microbatches_seen += 1
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    trainer = AccountedTrainer(model=model, args=training_args, train_dataset=Dataset(),
                      data_collator=collate, processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    record = {"status": "completed", "seconds": time.monotonic() - start,
              "gpu_count": torch.cuda.device_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "max_memory_allocated": torch.cuda.max_memory_allocated(), "metrics": result.metrics,
              "actual_supervised_tokens": trainer.supervised_tokens_seen,
              "actual_input_tokens": trainer.input_tokens_seen, "microbatches": trainer.microbatches_seen,
              "seed": args.seed, "data_sha256": digest(args.data), "script_sha256": digest(__file__)}
    (output / "run_result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
