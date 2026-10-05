"""Bounded interface adaptation on admitted synthetic training trajectories.

Preparation tokenizes on CPU with the original action-only tokenizer. All four
agent adapters receive one identical cache; P receives the same question
instances as complete programs. This module never opens dev/holdout questions,
reference files, or the public benchmark. Training continues existing LoRA
weights with stock Transformers Trainer and writes a new final adapter only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from experiments.agent_feedback.training_data import read_jsonl, tokenize_trajectory
from .development_probe import MODELS, PROTOCOL as ORIGINAL_PROTOCOL, sha, write_new

DATA = Path("data/radar_interface_v1")
ADMISSION = Path("artifacts/thesis_direction_review/radar_interface_synthetic_v1/manifest.json")
AUDIT = Path("artifacts/thesis_direction_review/radar_interface_synthetic_v1/independent_ai_audit.json")
OUT = Path("results/radar_domain/interface_adapt_v1")
PREPARATION = OUT / "preparation.json"
PROTOCOL = OUT / "protocol.json"
TRAIN_PATHS = {kind: DATA / f"train.{kind}_trajectories.jsonl" for kind in ("agent", "program")}
QUESTION_PATH = DATA / "train.questions.jsonl"
CACHE_PATHS = {kind: DATA / "tokenized" / f"{kind}.jsonl" for kind in TRAIN_PATHS}
SEEDS = {"P": 20261003, "A1": 20261003, "C1": 20261003, "A2": 20261004, "C2": 20261004}
CODE = ["experiments/radar_domain/interface_training.py",
        "experiments/radar_domain/development_probe.py",
        "experiments/agent_feedback/training_data.py",
        "experiments/agent_feedback/environment.py",
        "experiments/agent_feedback/run_job.py"]
TOKENIZER_FILES = {"tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"}


def versions(names):
    return {name: importlib.metadata.version(name) for name in names}


def hash_mapping(mapping):
    for name, expected in mapping.items():
        if sha(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def tokenizer_identity(model):
    return {path.name: sha(path) for path in sorted(Path(model).iterdir()) if path.name in TOKENIZER_FILES}


def load_admitted_training():
    """Open only the three admitted training files; never inspect held-out data."""
    admission = json.loads(ADMISSION.read_text())
    if admission.get("status") != "cpu_validated":
        raise ValueError("Synthetic corpus requires its completed CPU admission manifest")
    paths = [QUESTION_PATH, *TRAIN_PATHS.values()]
    bound = {str(path): admission["files_sha256"][str(path)] for path in paths}
    hash_mapping(bound)
    questions = read_jsonl(QUESTION_PATH)
    ids = [row["id"] for row in questions]
    if not questions or len(ids) != len(set(ids)):
        raise ValueError("Training question IDs must be nonempty and unique")
    question_text = {row["id"]: row["question"] for row in questions}
    trajectories = {kind: read_jsonl(path) for kind, path in TRAIN_PATHS.items()}
    for kind, rows in trajectories.items():
        if [row["id"] for row in rows] != ids:
            raise ValueError("Agent and program trajectories must share the exact question order")
        for row in rows:
            messages, supervise = row["messages"], row["supervise"]
            if (len(messages) != len(supervise) or any(type(flag) is not bool for flag in supervise)
                    or any(flag != (message["role"] == "assistant") for message, flag in zip(messages, supervise))):
                raise ValueError("Only and all canonical assistant messages receive supervision")
            if [message["content"] for message in messages if message["role"] == "user"] != [question_text[row["id"]]]:
                raise ValueError("Training trajectories changed or added question instances")
            if sum(supervise) != (3 if kind == "agent" else 1):
                raise ValueError("Expected three canonical agent actions or one complete program")
    bound[str(ADMISSION)] = sha(ADMISSION)
    return trajectories, bound


def validate_examples(examples, max_length):
    if not examples or len({row["id"] for row in examples}) != len(examples):
        raise ValueError("Token cache IDs must be nonempty and unique")
    for row in examples:
        ids, labels = row["input_ids"], row["labels"]
        if not ids or len(ids) != len(labels) or len(ids) > max_length:
            raise ValueError("Invalid tokenized lengths; truncation is forbidden")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid input token IDs")
        if any(type(label) is not int or (label != -100 and label != token) for token, label in zip(ids, labels)):
            raise ValueError("Action-only labels are not aligned to input tokens")
        if sum(label != -100 for label in labels) != row["supervised_tokens"] or row["supervised_tokens"] < 1:
            raise ValueError("Invalid supervised-token accounting")
    return {"records": len(examples), "input_tokens": sum(len(row["input_ids"]) for row in examples),
            "supervised_tokens": sum(row["supervised_tokens"] for row in examples),
            "max_input_length": max(len(row["input_ids"]) for row in examples)}


def prepare(model="models/qwen2.5-3b-instruct", max_length=8192):
    if PREPARATION.exists() or any(path.exists() for path in CACHE_PATHS.values()):
        raise FileExistsError("Preparation already exists or is partial; never overwrite token caches")
    original = json.loads(ORIGINAL_PROTOCOL.read_text())
    if model != original["model"] or max_length != original["inference"]["max_context"]:
        raise ValueError("Use the pinned base tokenizer and original context length")
    identity = tokenizer_identity(model)
    if any(identity.get(name) != expected for name, expected in original["tokenizer_sha256"].items()):
        raise ValueError("Pinned tokenizer changed")
    if "chat_template.jinja" in identity and "chat_template.jinja" not in original["tokenizer_sha256"]:
        raise ValueError("Unbound external chat template could override the pinned tokenizer")
    trajectories, raw_hashes = load_admitted_training()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    examples = {kind: [tokenize_trajectory(tokenizer, row, max_length, tools=kind == "agent") for row in rows]
                for kind, rows in trajectories.items()}
    summaries = {kind: validate_examples(rows, max_length) for kind, rows in examples.items()}
    # All validation completes before writing either cache. Manifest is written last.
    for kind, rows in examples.items():
        path = CACHE_PATHS[kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[kind].update(path=str(path), sha256=sha(path), tools=kind == "agent")
    hashes = dict(raw_hashes)
    hashes.update({name: sha(name) for name in CODE + [str(ORIGINAL_PROTOCOL)]})
    hash_mapping(raw_hashes)
    if tokenizer_identity(model) != identity:
        raise ValueError("Tokenizer changed during preparation")
    result = {"version": "radar_interface_tokenization_v1", "status": "completed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
              "training_started": False, "truncation": False, "max_length": max_length,
              "model": model, "inputs_sha256": hashes,
              "tokenizer": {"files_sha256": identity, "libraries": versions(["transformers", "tokenizers", "jinja2"])},
              "caches": summaries, "agent_cache_reused_by": ["A1", "C1", "A2", "C2"],
              "program_comparison": "Same question instances; program/action supervision lengths differ and are not token-matched."}
    write_new(PREPARATION, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def verify_preparation():
    prepared = json.loads(PREPARATION.read_text())
    if prepared["status"] != "completed" or not prepared["cpu_only"] or prepared["truncation"]:
        raise ValueError("Preparation must be complete, CPU-only and untruncated")
    hash_mapping(prepared["inputs_sha256"])
    hash_mapping({spec["path"]: spec["sha256"] for spec in prepared["caches"].values()})
    return prepared


def freeze(*, epochs, learning_rate, batch_size=2, accumulation=8, maximum_hours_per_run=0.5):
    if type(epochs) is not int or epochs < 1 or epochs > 5:
        raise ValueError("Choose an explicit bounded integer epoch count (1 to 5)")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-4:
        raise ValueError("Invalid bounded continuation learning rate")
    if batch_size < 1 or accumulation < 1 or not 0 < maximum_hours_per_run <= 1:
        raise ValueError("Invalid training batch or supervisor time budget")
    prepared = verify_preparation()
    audit = json.loads(AUDIT.read_text())
    if audit.get("status") != "passed":
        raise ValueError("An explicit passed independent AI audit is required before training freeze")
    original = json.loads(ORIGINAL_PROTOCOL.read_text())
    hashes = dict(prepared["inputs_sha256"])
    hashes.update({spec["path"]: spec["sha256"] for spec in prepared["caches"].values()})
    hashes[str(PREPARATION)] = sha(PREPARATION)
    hashes[str(AUDIT)] = sha(AUDIT)
    config = {"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
              "accumulation": accumulation, "max_length": prepared["max_length"],
              "warmup_steps": 0, "weight_decay": 0.0, "lr_scheduler_type": "linear",
              "optimizer": "adamw_torch", "bf16": True, "tf32": True,
              "gradient_checkpointing": True, "max_grad_norm": 1.0}
    result = {"version": "radar_interface_adapt_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "One admitted synthetic interface adaptation; not a new recovery algorithm or radar factual evaluation",
              "independent_radar_gold": False, "automatic_next_round": False,
              "inputs_sha256": hashes, "model": original["model"], "models": original["models"],
              "base_weights": original["base_weights"], "tokenizer": prepared["tokenizer"],
              "seeds": SEEDS, "config": config, "caches": prepared["caches"],
              "maximum_hours_per_run": maximum_hours_per_run,
              "maximum_additional_gpu_hours": maximum_hours_per_run * len(MODELS),
              "order_policy": "All A/C use the same full agent cache; within each A/C pair use the same training and data seed. Record the actual sampled-order digest.",
              "loss_policy": "Stock Trainer causal language-model loss, original assistant-only labels. No recovery-specific data, additional adapter, optimizer replacement or held-out access.",
              "program_comparison": prepared["program_comparison"],
              "stop_rule": "Train each of the five starting adapters once for the fixed full epochs, retain all results, then run the separately frozen evaluation. No score-based restart or automatic extra epochs."}
    write_new(PROTOCOL, result)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL), "config": config}), flush=True)
    return result


def verify(label, *, weights=True):
    if label not in MODELS:
        raise ValueError("Unknown fixed adapter label")
    protocol = json.loads(PROTOCOL.read_text())
    hash_mapping(protocol["inputs_sha256"])
    if weights:
        hash_mapping(protocol["models"][label]["files_sha256"])
        identity = tokenizer_identity(protocol["model"])
        if identity != protocol["tokenizer"]["files_sha256"]:
            raise ValueError("Tokenizer changed since CPU preparation")
        if versions(list(protocol["tokenizer"]["libraries"])) != protocol["tokenizer"]["libraries"]:
            raise ValueError("Tokenization libraries changed since CPU preparation")
    return protocol


def validate_loss_history(metrics, history):
    """A nominal Trainer completion cannot admit non-finite training evidence."""
    if "train_loss" not in metrics or not math.isfinite(float(metrics["train_loss"])):
        raise ValueError("Training loss is missing or non-finite")
    counts = Counter()
    for entry in history:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in entry:
                if not math.isfinite(float(entry[key])):
                    raise ValueError(f"Non-finite logged training evidence: {key}")
                counts[key] += 1
    return {"finite_training_loss": float(metrics["train_loss"]),
            "finite_logged_loss_entries": counts["loss"],
            "finite_logged_gradient_entries": counts["grad_norm"]}


def train(label):
    protocol = verify(label)
    config = protocol["config"]
    kind = "program" if label == "P" else "agent"
    spec = protocol["caches"][kind]
    examples = read_jsonl(spec["path"])
    observed = validate_examples(examples, config["max_length"])
    if any(observed[key] != spec[key] for key in observed):
        raise ValueError("Token cache content disagrees with admitted statistics")
    output = OUT / label
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import PeftModel
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This bounded continuation requires exactly one visible GPU and one process")
    seed = protocol["seeds"][label]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], local_files_only=True,
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, protocol["models"][label]["adapter"], is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters < 1 or any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise ValueError("Continue only the existing LoRA adapter parameters")
    manifest = {"label": label, "seed": seed, "kind": kind, "protocol_sha256": sha(PROTOCOL),
                "config": config, "cache": spec, "starting_model": protocol["models"][label],
                "trainable_parameters": parameters, "tokenizer": protocol["tokenizer"],
                "versions": versions(["torch", "transformers", "peft", "accelerate"])}
    write_new(output / "input_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return {key: examples[index][key] for key in ("id", "input_ids", "labels")}

    def collate(rows):
        length = (max(len(row["input_ids"]) for row in rows) + 7) // 8 * 8
        return {"input_ids": torch.tensor([row["input_ids"] + [tokenizer.pad_token_id] * (length - len(row["input_ids"])) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (length - len(row["input_ids"])) for row in rows]),
                "labels": torch.tensor([row["labels"] + [-100] * (length - len(row["labels"])) for row in rows]),
                "sample_ids": [row["id"] for row in rows]}

    arguments = TrainingArguments(output_dir=str(output), per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["accumulation"], num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"], warmup_steps=config["warmup_steps"],
            weight_decay=config["weight_decay"], lr_scheduler_type=config["lr_scheduler_type"],
            max_grad_norm=config["max_grad_norm"], bf16=config["bf16"], tf32=config["tf32"],
            gradient_checkpointing=config["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=config["optimizer"], logging_steps=1, save_strategy="no", report_to="none",
            seed=seed, data_seed=seed, dataloader_num_workers=0, dataloader_drop_last=False,
            remove_unused_columns=False, disable_tqdm=True)

    class AccountedTrainer(Trainer):
        supervised_tokens_seen = 0
        input_tokens_seen = 0
        microbatches_seen = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            sample_ids = inputs.pop("sample_ids")
            if model.training:
                self.supervised_tokens_seen += int(inputs["labels"].ne(-100).sum().item())
                self.input_tokens_seen += int(inputs["attention_mask"].sum().item())
                self.microbatches_seen += 1
                sampled_order.update((json.dumps(sample_ids, ensure_ascii=False) + "\n").encode())
                sampled_counts.update(sample_ids)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    sampled_order, sampled_counts = hashlib.sha256(), Counter()
    trainer = AccountedTrainer(model=model, args=arguments, train_dataset=Dataset(),
                               data_collator=collate, processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train()
    finite_evidence = validate_loss_history(result.metrics, trainer.state.log_history)
    checked_tensors = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            checked_tensors += 1
            if not bool(torch.isfinite(parameter).all().item()):
                raise ValueError(f"Non-finite trained LoRA parameter: {name}")
    finite_evidence["finite_trainable_parameter_tensors"] = checked_tensors
    if sampled_counts != Counter({row["id"]: config["epochs"] for row in examples}):
        raise ValueError("Actual sampled examples did not equal the fixed full epochs")
    if trainer.supervised_tokens_seen != spec["supervised_tokens"] * config["epochs"]:
        raise ValueError("Actual supervised tokens did not equal the frozen epoch budget")
    verify(label)
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    old_weights = protocol["models"][label]["files_sha256"][protocol["models"][label]["adapter"] + "/adapter_model.safetensors"]
    new_weights = sha(output / "model" / "adapter_model.safetensors")
    if new_weights == old_weights or trainer.state.global_step < 1:
        raise ValueError("No completed adapter update was observed")
    record = {"status": "completed", "label": label, "seconds": time.monotonic() - started,
              "gpu_count": torch.cuda.device_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "max_memory_allocated": torch.cuda.max_memory_allocated(), "seed": seed,
              "protocol_sha256": sha(PROTOCOL), "cache_sha256": spec["sha256"], "metrics": result.metrics,
              "actual_supervised_tokens": trainer.supervised_tokens_seen,
              "actual_input_tokens": trainer.input_tokens_seen, "microbatches": trainer.microbatches_seen,
              "sampled_order_sha256": sampled_order.hexdigest(), "sampled_examples": sum(sampled_counts.values()),
              "sample_count_histogram": dict(Counter(sampled_counts.values())),
              "global_step": trainer.state.global_step, "loss_and_gradient_history": trainer.state.log_history,
              "finite_evidence": finite_evidence,
              "starting_adapter_sha256": old_weights, "final_adapter_sha256": new_weights,
              "final_adapter_config_sha256": sha(output / "model" / "adapter_config.json"),
              "trainable_parameters": parameters, "script_sha256": sha(__file__)}
    write_new(output / "run_result.json", record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "freeze", "verify", "train"])
    parser.add_argument("--label", choices=MODELS, default="P")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--maximum-hours-per-run", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "freeze":
        if args.epochs is None or args.learning_rate is None:
            parser.error("freeze requires explicit --epochs and --learning-rate after token-count review")
        freeze(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
               accumulation=args.accumulation, maximum_hours_per_run=args.maximum_hours_per_run)
    elif args.command == "verify":
        verify(args.label)
        print("Frozen training inputs and starting adapter verified")
    else:
        train(args.label)


if __name__ == "__main__":
    main()
