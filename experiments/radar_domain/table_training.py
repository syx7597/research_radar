"""Admitted synthetic SFT and matched clean/recovery continuation for table tools.

Preparation is CPU-only and never opens evaluation references. Model loading is
local-only and deferred to the explicit train command after runtime admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time

from .table_runtime_executor import TableRuntimeExecutor as TableExecutor
from .table_recovery_data import admit_model_training_rollouts, make_pair, success
from .table_training_data import balanced_pair, pack

SPLIT = Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")
SPLIT_SUMMARY = Path("results/radar_domain/frozen_20261107/summary.json")
TOKEN_REPORT = Path("results/radar_domain/frozen_20261107/token_preflight.json")
MAX_CONTEXT = 16384
CODE = ["experiments/radar_domain/table_training.py", "experiments/radar_domain/table_training_data.py",
        "experiments/radar_domain/table_token_preflight.py", "experiments/radar_domain/table_recovery_data.py",
        "experiments/radar_domain/table_program.py", "experiments/radar_domain/table_runtime_executor.py",
        "experiments/radar_domain/table_scoring.py",
        "experiments/radar_domain/recovery_data.py", "experiments/agent_feedback/recovery.py"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def verify_hashes(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Bound training input changed: " + str(path))


def write_new(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def tokenizer_identity(directory):
    directory = Path(directory)
    report = read(TOKEN_REPORT)
    expected = report["tokenizer_files_sha256"]
    actual = {n: sha(directory/n) for n in expected}
    if actual != expected:
        raise ValueError("Use the original pinned Qwen tokenizer")
    if any((directory/n).exists() and n not in expected for n in
           ("chat_template.jinja", "chat_template.json", "added_tokens.json", "special_tokens_map.json")):
        raise ValueError("Unbound tokenizer/template overrides are forbidden")
    versions = {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers")}
    if versions != {"transformers": "5.18.0", "tokenizers": "0.23.2"}:
        raise ValueError("Use the registered tokenizer-only environment")
    return {"directory": str(directory), "files_sha256": actual, "libraries": versions}


def load_tokenizer(directory):
    identity = tokenizer_identity(directory)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True), identity


def load_training_partition(split_path=SPLIT):
    if sha(split_path) != read(SPLIT_SUMMARY)["freeze_manifest"]["sha256"]:
        raise ValueError("Use the originally frozen split identity; relabelled partitions are not admitted")
    split = read(split_path)
    if split.get("schema") != "radar_domain_table_split_v1" or split.get("status") != "frozen":
        raise ValueError("Only the frozen train/dev/holdout split is admitted")
    if set(split["splits"]) != {"train", "dev", "holdout"}:
        raise ValueError("Expected the original three-part split")
    for a, b in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
        left, right = split["splits"][a], split["splits"][b]
        if set(left["ids"]) & set(right["ids"]) or set(left["source_families"]) & set(right["source_families"]):
            raise ValueError("Training and evaluation identities overlap")
    train = split["splits"]["train"]
    if len(train["ids"]) != 576 or train["origin_counts"] != {"synthetic": 576}:
        raise ValueError("Use the complete admitted 576-example synthetic training partition")
    bound = {str(split_path): sha(split_path), str(SPLIT_SUMMARY): sha(SPLIT_SUMMARY)}
    for name in ("questions", "references", "metadata", "catalog", "reference_provenance", "success_trajectories"):
        entry = train[name]
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen training binding changed: " + name)
        bound[entry["path"]] = entry["sha256"]
    # Only training files are opened. Evaluation names/IDs above are metadata.
    questions, refs, metadata, trajectories = [read_rows(train[n]["path"]) for n in
        ("questions", "references", "metadata", "success_trajectories")]
    if any([r["id"] for r in data] != train["ids"] for data in (questions, refs, metadata, trajectories)):
        raise ValueError("Incomplete or reordered frozen training material")
    catalog = read(train["catalog"]["path"])
    if len(catalog["records"]) != 1728 or any(not r["family_id"].startswith("synthetic:") for r in catalog["records"]):
        raise ValueError("Training catalog must contain only complete synthetic source instances")
    for q, ref, meta, trace in zip(questions, refs, metadata, trajectories):
        if (set(q) != {"id", "question"} or q["question"] != ref["question"]
                or meta["origin"] != "synthetic" or meta["training_allowed"] is not True
                or trace["training_eligible"] is not True or trace["model_failure"] is not False
                or trace["source"] != "executed_reference_success"):
            raise ValueError("Evaluation or unadmitted material cannot become training targets")
    return split, train, refs, trajectories, catalog, bound


def validate_examples(examples, max_context=MAX_CONTEXT):
    if not examples or len({r["id"] for r in examples}) != len(examples):
        raise ValueError("Nonempty uniquely identified packed training rows required")
    for row in examples:
        ids, labels = row["input_ids"], row["labels"]
        if row.get("training_eligible") is not True:
            raise ValueError("Packed evaluation/fixture row is not training-eligible")
        if not ids or len(ids) != len(labels) or len(ids) > max_context:
            raise ValueError("Invalid packed lengths; no truncation is allowed")
        if any(type(v) is not int or v < 0 for v in ids):
            raise ValueError("Token IDs must be nonnegative integers")
        if any(type(y) is not int or y != -100 and y != x for x, y in zip(ids, labels)):
            raise ValueError("Supervised labels must equal their actual input token")
        if row["supervised_tokens"] != sum(y != -100 for y in labels) or row["supervised_tokens"] < 1:
            raise ValueError("Supervision count differs from actual labels")
    return {"records": len(examples), "supervised_tokens": sum(r["supervised_tokens"] for r in examples),
            "input_tokens": sum(len(r["input_ids"]) for r in examples),
            "max_context_tokens": max(len(r["input_ids"]) for r in examples)}


def code_bindings():
    return {p: sha(p) for p in CODE}


def prepare_initial(output, *, split_path=SPLIT, tokenizer_dir="cache/qwen2.5-3b-tokenizer", seed=20261108):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve complete or partial preparation directories")
    _, train, refs, trajectories, catalog, bound = load_training_partition(split_path)
    tokenizer, identity = load_tokenizer(tokenizer_dir)
    executor = TableExecutor(catalog)
    examples = []
    for ref, stored in zip(refs, trajectories):
        actual = success(executor, ref)
        if actual["messages"] != stored["messages"] or actual["supervise"] != stored["supervise"]:
            raise ValueError("Training reference replay differs from frozen success trace")
        packed = pack(tokenizer, stored, MAX_CONTEXT)
        packed.update(source_id=ref["id"], kind="ordinary", origin="synthetic_train_success")
        examples.append(packed)
    random.Random(seed).shuffle(examples)
    stats = validate_examples(examples)
    bound.update(code_bindings()); bound[str(TOKEN_REPORT)] = sha(TOKEN_REPORT)
    bound.update({str(Path(tokenizer_dir)/n): h for n, h in identity["files_sha256"].items()})
    verify_hashes(bound)
    data = output / "initial.jsonl"
    write_new(data, examples, lines=True)
    result = {"schema": "radar_table_training_preparation_v1", "status": "completed", "stage": "initial",
        "source_split": "train", "training_eligible": True, "model_runs": 0, "training_started": False,
        "split_manifest": binding(split_path), "tokenizer": identity, "max_context": MAX_CONTEXT,
        "sequence_seed": seed, "one_pass_supervised_tokens": stats["supervised_tokens"],
        "arms": {"I": {**binding(data), **stats}}, "inputs_sha256": bound,
        "source_ids": train["ids"], "training_origin": "Frozen synthetic successful references only"}
    write_new(output / "preparation.json", result)
    return result


def verify_preparation(path):
    result = read(path)
    if (result.get("schema") != "radar_table_training_preparation_v1" or result.get("status") != "completed"
            or result.get("source_split") != "train" or result.get("training_eligible") is not True
            or result.get("stage") not in {"initial", "continuation"} or result["max_context"] != MAX_CONTEXT):
        raise ValueError("An admitted completed table training preparation is required")
    verify_hashes(result["inputs_sha256"])
    if sha(result["split_manifest"]["path"]) != result["split_manifest"]["sha256"]:
        raise ValueError("Parent split changed")
    expected_arms = {"I"} if result["stage"] == "initial" else {"A", "C"}
    if set(result["arms"]) != expected_arms:
        raise ValueError("Preparation has unexpected training arms")
    examples = {}
    for arm, spec in result["arms"].items():
        if sha(spec["path"]) != spec["sha256"]:
            raise ValueError("Packed training artifact changed")
        examples[arm] = read_rows(spec["path"])
        stats = validate_examples(examples[arm], result["max_context"])
        if any(stats[k] != spec[k] for k in stats):
            raise ValueError("Packed training counts changed")
    if result["stage"] == "continuation":
        validate_matched_corpus(examples["A"], examples["C"], result["budget"])
    return result, examples


def mask_to_count(example, count):
    """Keep the full causal history; mask only excess target positions at its tail."""
    if type(count) is not int or not 0 < count <= example["supervised_tokens"]:
        raise ValueError("Invalid supervision cap")
    row = deepcopy(example)
    remaining = count
    for i, label in enumerate(row["labels"]):
        if label != -100:
            if remaining:
                remaining -= 1
            else:
                row["labels"][i] = -100
    if remaining:
        raise ValueError("Incorrect original supervision count")
    row["supervised_tokens"] = count
    row["budget_masked_tokens"] = example.get("budget_masked_tokens", 0) + example["supervised_tokens"]-count
    return row


def sample_exact_budget(pool, budget, seed, kind):
    if not pool or type(budget) is not int or budget < 1:
        raise ValueError("A nonempty admitted pool and positive token budget are required")
    rng, result, remaining, cycle = random.Random(seed), [], budget, 0
    while remaining:
        order = list(range(len(pool))); rng.shuffle(order)
        for index in order:
            source = pool[index]
            target = min(source["supervised_tokens"], remaining)
            row = mask_to_count(source, target)
            row.update(id=f"{kind}:{len(result):06d}", source_id=source.get("source_id", source["id"]),
                       kind=kind, sampling_cycle=cycle)
            result.append(row); remaining -= target
            if not remaining:
                break
        cycle += 1
    return result


def make_matched_corpus(ordinary, pairs, total_budget, seed):
    if not pairs or type(total_budget) is not int or total_budget < 2:
        raise ValueError("Real eligible pairs and positive total token budget required")
    paired_budget = total_budget // 2
    budget = {"total": total_budget, "ordinary": total_budget-paired_budget, "paired_suffix": paired_budget}
    common = sample_exact_budget(ordinary, budget["ordinary"], seed, "ordinary")
    a_pool, c_pool = [p[0] for p in pairs], [p[1] for p in pairs]
    for a, c in pairs:
        if a["id"] != c["id"] or a["supervised_tokens"] != c["supervised_tokens"]:
            raise ValueError("Pairs must match identity and actual supervised-token count")
    a_suffix = sample_exact_budget(a_pool, paired_budget, seed+1, "paired_suffix")
    c_suffix = sample_exact_budget(c_pool, paired_budget, seed+1, "paired_suffix")
    a, c = deepcopy(common)+a_suffix, deepcopy(common)+c_suffix
    order = list(range(len(a))); random.Random(seed+2).shuffle(order)
    a, c = [a[i] for i in order], [c[i] for i in order]
    validate_matched_corpus(a, c, budget)
    return a, c, budget


def validate_matched_corpus(a, c, budget):
    if (type(budget["total"]) is not int or budget["total"] < 2
            or budget["paired_suffix"] != budget["total"]//2
            or budget["ordinary"] != budget["total"]-budget["paired_suffix"]):
        raise ValueError("A/C must use the registered half-success, half-suffix target budget")
    validate_examples(a); validate_examples(c)
    if len(a) != len(c):
        raise ValueError("Paired corpus row counts differ")
    for left, right in zip(a, c):
        if any(left[k] != right[k] for k in ("id", "source_id", "kind", "supervised_tokens")):
            raise ValueError("A/C sample identity, order or target budget differs")
        if left["kind"] not in {"ordinary", "paired_suffix"}:
            raise ValueError("Unregistered training record kind")
        if left["kind"] == "ordinary" and left != right:
            raise ValueError("Ordinary material must be byte-equivalent across A/C")
    for rows_ in (a, c):
        for kind in ("ordinary", "paired_suffix"):
            if sum(r["supervised_tokens"] for r in rows_ if r["kind"] == kind) != budget[kind]:
                raise ValueError("Supervised token budget differs by corpus component")


def prepare_continuation(initial_preparation, generation_path, output, *, tokenizer_dir="cache/qwen2.5-3b-tokenizer", seed=20261108):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve completed or partial continuation preparation")
    prepared, initial = verify_preparation(initial_preparation)
    if prepared["stage"] != "initial":
        raise ValueError("Continuation needs the admitted initial success corpus")
    split_path = prepared["split_manifest"]["path"]
    rollouts = admit_model_training_rollouts(split_path, generation_path)
    generation = read(generation_path)
    if generation.get("collector_role") != "I_domain" or not generation.get("adapter_dir") or not generation.get("base_model_dir"):
        raise ValueError("Collect failures using the explicitly identified shared I_domain adapter")
    training_proof = validate_collector_training(generation, initial_preparation)
    from .table_runtime import model_files
    collector_files = model_files(generation["base_model_dir"], generation["adapter_dir"])
    if collector_files != generation["model_files_sha256"]:
        raise ValueError("I_domain model inventory differs from actual generation")
    _, _, refs, _, catalog, bound = load_training_partition(split_path)
    tokenizer, identity = load_tokenizer(tokenizer_dir)
    if identity["files_sha256"] != prepared["tokenizer"]["files_sha256"]:
        raise ValueError("Initial and paired targets use different tokenizers")
    executor, pairs, audits = TableExecutor(catalog), [], []
    for ref, rollout in zip(refs, rollouts):
        result = make_pair(executor, ref, rollout)
        entry = result["audit"]
        if entry["status"] == "accepted":
            try:
                a, c = balanced_pair(tokenizer, result["clean"], result["recovery"], MAX_CONTEXT)
            except ValueError as exc:
                entry.update(status="excluded", reason="token_budget_or_template_failure", detail=str(exc))
            else:
                for row in (a, c):
                    row.update(training_eligible=True, source_id=ref["id"], kind="paired_suffix",
                               origin="admitted_model_generated_train_failure")
                pairs.append((a, c))
                entry.update(training_eligible=True, token_budget_matched=True,
                             supervised_tokens=a["supervised_tokens"])
        audits.append(entry)
    if not pairs:
        write_new(output/"exclusions.jsonl", audits, lines=True)
        write_new(output/"audit.json", {"status": "no_eligible_recovery_pairs", "training_ready": False,
            "generation": binding(generation_path), "exclusion_counts": dict(Counter(a.get("reason") for a in audits))})
        raise ValueError("No eligible real failures; exclusion ledger saved, no training corpus created")
    a, c, budget = make_matched_corpus(initial["I"], pairs, prepared["one_pass_supervised_tokens"], seed)
    bound.update(prepared["inputs_sha256"])
    bound.update(code_bindings()); bound[str(initial_preparation)] = sha(initial_preparation)
    bound[str(generation_path)] = sha(generation_path)
    bound[training_proof["path"]] = training_proof["sha256"]
    bound[generation["rollouts"]["path"]] = generation["rollouts"]["sha256"]
    bound.update(collector_files); bound.update(generation["implementation_sha256"])
    verify_hashes(bound)
    arms = {}
    for name, data in (("A", a), ("C", c)):
        path = output/(name+".jsonl"); write_new(path, data, lines=True)
        arms[name] = {**binding(path), **validate_examples(data)}
    write_new(output/"selection.jsonl", audits, lines=True)
    result = {"schema": "radar_table_training_preparation_v1", "status": "completed", "stage": "continuation",
        "source_split": "train", "training_eligible": True, "training_started": False,
        "split_manifest": prepared["split_manifest"], "max_context": MAX_CONTEXT, "sequence_seed": seed,
        "tokenizer": identity, "one_pass_supervised_tokens": prepared["one_pass_supervised_tokens"],
        "budget": budget, "arms": arms, "inputs_sha256": bound, "generation": binding(generation_path),
        "collector": {"role": "I_domain", "base_model_dir": generation["base_model_dir"],
            "adapter_dir": generation["adapter_dir"], "model_files_sha256": collector_files,
            "training_result": training_proof},
        "selection": {"rollouts": len(rollouts), "accepted_pairs": len(pairs), "ledger": binding(output/"selection.jsonl"),
            "excluded_by_reason": dict(Counter(a["reason"] for a in audits if a["status"] == "excluded"))},
        "training_origin": "Actually generated I_domain train failures with replayed and token-matched suffixes"}
    write_new(output/"preparation.json", result)
    return result


def validate_collector_training(generation, initial_preparation):
    proof = generation.get("collector_training_result")
    if not isinstance(proof, dict) or set(proof) != {"path", "sha256"} or sha(proof["path"]) != proof["sha256"]:
        raise ValueError("Actual initial training completion proof is required for failure collection")
    result = read(proof["path"])
    if (result.get("schema") != "radar_table_training_run_result_v1" or result.get("status") != "completed"
            or result.get("stage") != "initial" or result.get("arm") != "I"
            or result.get("preparation") != binding(initial_preparation) or result.get("global_steps", 0) < 1):
        raise ValueError("Collector is not the completed I_domain trained on this admitted success corpus")
    expected = result.get("final_adapter_files_sha256", {})
    adapter = Path(generation["adapter_dir"]).resolve()
    if (not expected or any(not Path(path).resolve().is_relative_to(adapter) for path in expected)
            or not {"adapter_config.json", "adapter_model.safetensors"} <= {Path(p).name for p in expected}):
        raise ValueError("Initial completion proof does not identify this I_domain adapter")
    verify_hashes(expected)
    if any(generation["model_files_sha256"].get(p) != h for p, h in expected.items()):
        raise ValueError("Failure collector weights differ from the actually trained I_domain")
    return proof


def config_defaults(stage="initial"):
    if stage not in {"initial", "continuation"}:
        raise ValueError("Unknown training stage")
    return {"schema": "radar_table_train_config_v1", "stage": stage, "seed": 20261003,
            "learning_rate": 1e-4 if stage == "initial" else 5e-5,
            "epochs": 2 if stage == "initial" else 1, "batch_size": 1, "accumulation": 8,
            "max_context": MAX_CONTEXT, "max_gpu_hours": 1.0, "runtime_user": "syx",
            "ledger_path": "results/radar_domain/table_runtime_v1/gpu_budget.json", "job_id": "REPLACE_WITH_UNIQUE_RUN_ID",
            "base_model_dir": "models/qwen2.5-3b-instruct", "adapter_dir": None,
            "lora": {"r": 16, "alpha": 32, "dropout": .05, "target_modules": "all-linear"},
            "configuration_status": "proposal_requires_actual_runtime_binding"}


def validate_config(config, prepared, arm):
    if config.get("schema") != "radar_table_train_config_v1" or config.get("stage") != prepared["stage"]:
        raise ValueError("Explicit matching training configuration required")
    if config["max_context"] != prepared["max_context"] or arm not in prepared["arms"]:
        raise ValueError("Training arm/context differs from completed preparation")
    if (type(config["epochs"]) is not int or not 1 <= config["epochs"] <= 3
            or prepared["stage"] == "continuation" and config["epochs"] != 1):
        raise ValueError("Initial SFT uses full bounded epochs; continuation uses its exact one-pass token budget")
    if (not math.isfinite(config["learning_rate"]) or not 0 < config["learning_rate"] <= 1e-4
            or not math.isfinite(config["max_gpu_hours"]) or not 0 < config["max_gpu_hours"] <= 8):
        raise ValueError("Invalid learning rate or bounded GPU allowance")
    if (any(type(config[k]) is not int or config[k] < 1 for k in ("batch_size", "accumulation"))
            or type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32):
        raise ValueError("Explicit seed and positive batch settings required")
    if config.get("runtime_user") in {None, "root", "ubuntu"}:
        raise ValueError("Training requires a normal experiment account")
    if (not isinstance(config.get("ledger_path"), str) or not config["ledger_path"]
            or not isinstance(config.get("job_id"), str) or not config["job_id"].strip()
            or config["job_id"] == "REPLACE_WITH_UNIQUE_RUN_ID"):
        raise ValueError("Declare the shared budget ledger and a unique real job ID")
    from .table_job_budget import require_round_ledger
    require_round_ledger(config["ledger_path"])
    if prepared["stage"] == "initial" and config.get("adapter_dir"):
        raise ValueError("Initial domain SFT starts from the declared base model")
    if prepared["stage"] == "continuation":
        collector = prepared["collector"]
        if any(Path(config.get(k) or "").resolve() != Path(collector[k]).resolve() for k in ("base_model_dir", "adapter_dir")):
            raise ValueError("Both continuation arms must start from their shared collecting I_domain")
    if config["lora"] != {"r": 16, "alpha": 32, "dropout": .05, "target_modules": "all-linear"}:
        raise ValueError("Use the declared LoRA recipe; register any change separately")


def runtime_bindings(preparation, config):
    implementations = code_bindings()
    implementations["experiments/radar_domain/table_runtime.py"] = sha("experiments/radar_domain/table_runtime.py")
    implementations["experiments/radar_domain/table_job_budget.py"] = sha("experiments/radar_domain/table_job_budget.py")
    return implementations, {str(preparation): sha(preparation), str(config): sha(config)}


def verify_training_run(preparation, arm, config_path, runtime_path):
    prepared, examples = verify_preparation(preparation)
    config = read(config_path); validate_config(config, prepared, arm)
    from .table_runtime import model_files, validate_runtime_evidence
    files = model_files(config["base_model_dir"], config.get("adapter_dir"))
    if prepared["stage"] == "continuation" and files != prepared["collector"]["model_files_sha256"]:
        raise ValueError("Continuation starting model differs from failed-rollout collector")
    implementations, inputs = runtime_bindings(preparation, config_path)
    evidence = validate_runtime_evidence(runtime_path, purpose="training", model_files_sha256=files,
        implementation_sha256=implementations, input_files_sha256=inputs, expected_user=config["runtime_user"])
    identity = tokenizer_identity(prepared["tokenizer"]["directory"])
    if identity != prepared["tokenizer"]:
        raise ValueError("Packed tokenizer/libraries changed")
    return prepared, examples[arm], config, files, evidence


def prepare_training_runtime(preparation, arm, config_path, output, gpu_index=0):
    prepared, _ = verify_preparation(preparation)
    config = read(config_path); validate_config(config, prepared, arm)
    from .table_runtime import preflight_inputs
    implementations, inputs = runtime_bindings(preparation, config_path)
    return preflight_inputs(purpose="training", base_model_dir=config["base_model_dir"],
        adapter_dir=config.get("adapter_dir"), input_files_sha256=inputs, implementation_sha256=implementations,
        output=output, expected_user=config["runtime_user"], gpu_index=gpu_index,
        ledger_path=config["ledger_path"], job_id=config["job_id"]+"_cuda_preflight")


def training_environment():
    import torch
    return {"libraries": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "tokenizers", "peft", "accelerate", "safetensors", "jinja2")},
            "torch_runtime_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
            "python_version": platform.python_version()}


def train_worker(preparation, arm, config_path, runtime_path, output, admitted, budget, environment):
    """Run in the job-budget process so its hard watchdog also terminates CUDA."""
    prepared, examples, config, files, evidence = admitted
    budget.check()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one explicitly admitted GPU per run; pair arms can run on separately admitted devices")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(prepared["tokenizer"]["directory"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(config["base_model_dir"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if config.get("adapter_dir"):
        model = PeftModel.from_pretrained(model, config["adapter_dir"], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))
    model.config.use_cache = False; model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    if not any(p.requires_grad for p in model.parameters()) or any(p.requires_grad and "lora_" not in n for n, p in model.named_parameters()):
        raise ValueError("Only declared LoRA parameters may be trained")

    def collate(batch):
        length = (max(len(r["input_ids"]) for r in batch)+7)//8*8
        return {"input_ids": torch.tensor([r["input_ids"]+[tokenizer.pad_token_id]*(length-len(r["input_ids"])) for r in batch]),
                "attention_mask": torch.tensor([[1]*len(r["input_ids"])+[0]*(length-len(r["input_ids"])) for r in batch]),
                "labels": torch.tensor([r["labels"]+[-100]*(length-len(r["labels"])) for r in batch]),
                "sample_ids": [r["id"] for r in batch]}

    # Same training material, longest input, actual differentiable loss, no update.
    probe = collate([max(examples, key=lambda r: len(r["input_ids"]))]); probe.pop("sample_ids")
    model.train(); model.zero_grad(set_to_none=True)
    loss = model(**{k: v.to("cuda") for k, v in probe.items()}).loss
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Training-sample forward probe produced nonfinite loss")
    loss.backward()
    finite_grads = all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None)
    if not finite_grads or not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise ValueError("Training-sample backward probe failed")
    probe_loss = float(loss.item()); model.zero_grad(set_to_none=True); del loss, probe
    set_seed(config["seed"])
    budget.check()
    write_new(Path(output)/"model_probe.json", {"status": "passed", "source": "longest_actual_training_example",
        "finite_loss": probe_loss, "finite_gradients": True, "optimizer_steps": 0,
        "cleared_gradients": True, "preparation": binding(preparation)})

    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(examples)
        def __getitem__(self, index): return examples[index]

    class AccountedTrainer(Trainer):
        supervised_seen, input_seen = 0, 0
        sampled_ids = []
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            budget.check()
            ids = inputs.pop("sample_ids")
            if model.training:
                self.supervised_seen += int(inputs["labels"].ne(-100).sum().item())
                self.input_seen += int(inputs["attention_mask"].sum().item())
                self.sampled_ids.extend(ids)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["accumulation"], num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"], warmup_steps=0, weight_decay=0., bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", save_strategy="no", report_to="none", logging_steps=1,
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, disable_tqdm=True)
    trainer = AccountedTrainer(model=model, args=args, train_dataset=Dataset(), data_collator=collate, processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    trained = trainer.train()
    expected_ids = [r["id"] for r in examples]*config["epochs"]
    if trainer.sampled_ids != expected_ids or trainer.supervised_seen != prepared["arms"][arm]["supervised_tokens"]*config["epochs"]:
        raise ValueError("Actual training order or supervised-token budget differs from preparation")
    if not math.isfinite(float(trained.metrics.get("train_loss", math.nan))):
        raise ValueError("Training loss is missing or nonfinite")
    for entry in trainer.state.log_history:
        for key in ("loss", "grad_norm", "train_loss"):
            if key in entry and not math.isfinite(float(entry[key])):
                raise ValueError("Nonfinite training history")
    if any(not bool(torch.isfinite(p).all().item()) for p in model.parameters() if p.requires_grad):
        raise ValueError("Nonfinite trained adapter tensor")
    verify_hashes(files); verify_preparation(preparation)
    budget.check()
    trainer.save_model(str(Path(output)/"model")); tokenizer.save_pretrained(Path(output)/"model")
    if trainer.state.global_step < 1:
        raise ValueError("No optimizer step completed")
    final_weights = Path(output)/"model/adapter_model.safetensors"
    if config.get("adapter_dir") and sha(final_weights) == sha(Path(config["adapter_dir"])/"adapter_model.safetensors"):
        raise ValueError("Continuation weights did not change")
    write_new(Path(output)/"worker_result.json", {"schema": "radar_table_training_run_result_v1",
        "status": "completed", "arm": arm, "stage": prepared["stage"], "seed": config["seed"],
        "preparation": binding(preparation), "config": binding(config_path),
        "training_environment": environment,
        "global_steps": trainer.state.global_step, "actual_supervised_tokens": trainer.supervised_seen,
        "actual_input_tokens": trainer.input_seen, "actual_sampled_order_sha256": hashlib.sha256(json.dumps(trainer.sampled_ids).encode()).hexdigest(),
        "metrics": trained.metrics, "loss_history": trainer.state.log_history,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "base_and_starting_model_files_sha256": files,
        "final_adapter_files_sha256": {str(p): sha(p) for p in (Path(output)/"model").iterdir() if p.is_file()},
        "model_probe": binding(Path(output)/"model_probe.json"), "runtime_evidence": binding(runtime_path)})


def claim_paired_settings(preparation, config, environment):
    """First admitted arm fixes shared settings; only the unique job ID may vary."""
    import fcntl
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("An explicit valid continuation seed is required")
    path = Path(preparation).parent / f"pair_training_settings.seed{config['seed']}.json"
    common = {k: v for k, v in config.items() if k != "job_id"}
    expected = {"schema": "radar_table_paired_training_settings_v1", "preparation": binding(preparation),
                "common_config": common, "training_environment": environment}
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists():
                if read(path) != expected:
                    raise ValueError("A/C continuation settings differ; preserve the matched comparison")
            else:
                write_new(path, expected)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return binding(path)


def train(preparation, arm, config_path, runtime_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite a completed or failed training run")
    admitted = verify_training_run(preparation, arm, config_path, runtime_path)
    prepared, _, config, files, _ = admitted
    environment = training_environment()
    probe = admitted[-1]["probe"]
    if (probe["torch_version"] != environment["torch_runtime_version"]
            or probe["cuda_version"] != environment["cuda_runtime_version"]):
        raise ValueError("Torch/CUDA version changed since the admitted runtime probe")
    pairing = claim_paired_settings(preparation, config, environment) if prepared["stage"] == "continuation" else None
    output.mkdir(parents=True)
    write_new(output/"input_manifest.json", {"schema": "radar_table_training_run_v1", "status": "starting",
        "arm": arm, "preparation": binding(preparation), "config": binding(config_path),
        "runtime_evidence": binding(runtime_path), "model_files_sha256": files,
        "expected_supervised_tokens": prepared["arms"][arm]["supervised_tokens"]*config["epochs"],
        "implementation_sha256": runtime_bindings(preparation, config_path)[0], "gpu_count": 1,
        "paired_settings": pairing, "training_environment": environment})
    started = time.monotonic()
    from .table_job_budget import JobBudget
    try:
        with JobBudget(ledger_path=config["ledger_path"], job_id=config["job_id"], phase="training",
                       max_gpu_hours=config["max_gpu_hours"], gpu_count=1) as budget:
            train_worker(preparation, arm, config_path, runtime_path, output, admitted, budget, environment)
            result = read(output/"worker_result.json")
            if pairing and sha(pairing["path"]) != pairing["sha256"]:
                raise ValueError("Paired settings changed during training")
            result.update(wall_seconds=time.monotonic()-started, gpu_count=1,
                gpu_hours=(time.monotonic()-started)/3600, max_gpu_hours=config["max_gpu_hours"],
                budget_ledger=config["ledger_path"], job_id=config["job_id"], paired_settings=pairing,
                input_manifest=binding(output/"input_manifest.json"))
            budget.check()
        write_new(output/"run_result.json", result)
    except Exception as exc:
        if not (output/"run_result.json").exists():
            write_new(output/"run_result.json", {"schema": "radar_table_training_run_result_v1", "status": "failed",
                "arm": arm, "stage": prepared["stage"], "error_type": type(exc).__name__, "detail": str(exc),
                "wall_seconds": time.monotonic()-started, "gpu_count": 1,
                "gpu_hours": (time.monotonic()-started)/3600, "budget_ledger": config["ledger_path"],
                "input_manifest": binding(output/"input_manifest.json")})
        raise
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("prepare-initial", "prepare-continuation"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True); sub.add_argument("--tokenizer", default="cache/qwen2.5-3b-tokenizer")
        sub.add_argument("--seed", type=int, default=20261108)
        if name == "prepare-initial": sub.add_argument("--split-manifest", default=str(SPLIT))
        else:
            sub.add_argument("--initial-preparation", required=True); sub.add_argument("--generation", required=True)
    sub = commands.add_parser("verify"); sub.add_argument("--preparation", required=True)
    sub = commands.add_parser("config-template"); sub.add_argument("--stage", choices=("initial", "continuation"), default="initial")
    for name in ("preflight", "train"):
        sub = commands.add_parser(name)
        for flag in ("preparation", "config", "output"): sub.add_argument("--"+flag, required=True)
        sub.add_argument("--arm", choices=("I", "A", "C"), required=True)
        if name == "train": sub.add_argument("--runtime-evidence", required=True)
        else: sub.add_argument("--gpu-index", type=int, required=True)
    args = p.parse_args()
    if args.command == "prepare-initial":
        result = prepare_initial(args.output, split_path=args.split_manifest, tokenizer_dir=args.tokenizer, seed=args.seed)
        print(json.dumps({"status": result["status"], "stage": result["stage"], "arms": result["arms"]}))
    elif args.command == "prepare-continuation":
        result = prepare_continuation(args.initial_preparation, args.generation, args.output, tokenizer_dir=args.tokenizer, seed=args.seed)
        print(json.dumps({"status": result["status"], "budget": result["budget"], "selection": result["selection"]}))
    elif args.command == "verify":
        result, _ = verify_preparation(args.preparation)
        print(json.dumps({"status": "passed", "stage": result["stage"], "arms": result["arms"]}))
    elif args.command == "config-template":
        print(json.dumps(config_defaults(args.stage), indent=2))
    elif args.command == "preflight":
        result = prepare_training_runtime(args.preparation, args.arm, args.config, args.output, args.gpu_index)
        print(json.dumps({"status": result["status"], "scope": result["scope"], "output": args.output}))
    else:
        print(json.dumps(train(args.preparation, args.arm, args.config, args.runtime_evidence, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
