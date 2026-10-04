"""Observe the frozen native GRPO without modifying its loss, rollout or rewards.

This separate entry point preserves historical train_grpo.py byte identity.
Short smoke weights are never saved or eligible as formal B/D results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time

from experiments.condition_consistency.executor import KoPLExecutor
from .environment import MAX_CALLS, compact
from .training_data import read_jsonl
from . import train_grpo as frozen_grpo_entry
from .train_grpo import (
    EXPECTED_TRL, EXPECTED_TRAINER_SHA256, NATIVE_CONTRACT,
    TOTAL_COMPLETION_TOKEN_CAP, PER_TURN_TOKEN_CAP,
    RewardEnvironment, digest, make_rows, protocol_check, run_settings, tokenizer_preflight,
)

EXPECTED_BASE_ENTRY_SHA256 = "0adeef4b717ea1c5b7daab4f59162d838051f99d97847bbf363512950de13d75"


def select_smoke_question_ids(eligible_ids, seed=20261003, count=16):
    """Outcome-blind selection for the external smoke controller, never a prompt field."""
    ids = list(eligible_ids)
    if len(ids) != len(set(ids)) or type(count) is not int or count < 1 or len(ids) < count:
        raise ValueError("Smoke selection needs enough unique eligible training IDs")
    return sorted(ids, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
                                         compact(value)))[:count]


def reward_group_records(ids, rewards, advantages, loss_tokens, native_lengths, group_size=4):
    """Inspect native, pre-shuffle groups; do not reconstruct or change advantages."""
    if (not ids or len(ids) % group_size or
            any(len(values) != len(ids) for values in (rewards, advantages, loss_tokens, native_lengths))):
        raise ValueError("Signal audit received incomplete native reward groups")
    records = []
    for offset in range(0, len(ids), group_size):
        end = offset + group_size
        if len(set(ids[offset:end])) != 1:
            raise ValueError("Native reward group mixes different training questions")
        rs, adv, tokens = rewards[offset:end], advantages[offset:end], loss_tokens[offset:end]
        if any(type(n) is not int or n < 0 for n in tokens + native_lengths[offset:end]):
            raise ValueError("Invalid retained token counts in signal audit")
        finite_reward = all(math.isfinite(x) for x in rs)
        finite_advantage = all(math.isfinite(x) for x in adv)
        effective = [math.isfinite(a) and a != 0 and n > 0 for a, n in zip(adv, tokens)]
        records.append({"example_id": ids[offset],
            "rewards": [x if math.isfinite(x) else None for x in rs],
            "reward_spread": max(rs) - min(rs) if finite_reward else None,
            "all_rewards_finite": finite_reward,
            "all_rewards_binary": finite_reward and all(x in (0.0, 1.0) for x in rs),
            "advantages": [x if math.isfinite(x) else None for x in adv],
            "all_advantages_finite": finite_advantage,
            "loss_tokens_after_completion_and_tool_masks": tokens,
            "native_retained_completion_tokens_including_observations": native_lengths[offset:end],
            "zero_loss_token_rollouts": sum(n == 0 for n in tokens),
            "nonzero_advantage_rollouts": sum(math.isfinite(a) and a != 0 for a in adv),
            "effective_signal_rollouts": sum(effective),
            "effective_signal_tokens": sum(n for n, valid in zip(tokens, effective) if valid)})
    return records


def signal_summary(groups, steps, parameters, expected_ids):
    """Report execution evidence, not method effectiveness or a positive-loss gate."""
    observed_ids = [row["example_id"] for row in groups]
    finite_groups = all(row["all_rewards_finite"] and row["all_advantages_finite"] for row in groups)
    binary_rewards = bool(groups) and all(row["all_rewards_binary"] for row in groups)
    finite_gradients = bool(steps) and all(row["all_gradients_finite"] for row in steps)
    mixed = sum(row["reward_spread"] is not None and row["reward_spread"] > 0 for row in groups)
    nonzero = sum(row["all_gradients_finite"] and row["gradient_l2_after_clip"] is not None
                  and row["gradient_l2_after_clip"] > 0 for row in steps)
    positive_lr_grad = sum(row["all_gradients_finite"] and row["gradient_l2_after_clip"] is not None
        and row["gradient_l2_after_clip"] > 0 and any(lr is not None and lr > 0 for lr in row["learning_rates"])
        for row in steps)
    updated = bool(parameters and parameters["all_before_finite"] and parameters["all_after_finite"]
                   and parameters["changed_elements"] > 0 and parameters["l2_delta"] > 0
                   and parameters["sha256_before"] != parameters["sha256_after"])
    effective = sum(row["effective_signal_rollouts"] > 0 for row in groups)
    return {"interpretation": "mechanical RL learning-signal evidence only; not a quality gain",
        "group_size": 4, "question_groups": len(groups), "rollouts": len(groups) * 4,
        "group_question_ids_in_generation_order": observed_ids,
        "unique_questions_observed": len(set(observed_ids)), "eligible_questions": len(expected_ids),
        "all_eligible_questions_observed": set(observed_ids) == set(expected_ids),
        "unobserved_eligible_ids": [value for value in expected_ids if value not in set(observed_ids)],
        "mixed_reward_groups": mixed,
        "all_zero_reward_groups": sum(row["rewards"] == [0.0] * 4 for row in groups),
        "all_one_reward_groups": sum(row["rewards"] == [1.0] * 4 for row in groups),
        "all_rewards_and_advantages_finite": bool(groups) and finite_groups,
        "all_rewards_binary": binary_rewards,
        "groups_with_unmasked_nonzero_advantage": effective,
        "zero_loss_token_rollouts": sum(row["zero_loss_token_rollouts"] for row in groups),
        "effective_signal_tokens": sum(row["effective_signal_tokens"] for row in groups),
        "native_retained_completion_tokens_including_observations": sum(sum(
            row["native_retained_completion_tokens_including_observations"]) for row in groups),
        "optimizer_steps_observed": len(steps), "all_observed_gradients_finite": finite_gradients,
        "finite_nonzero_gradient_steps": nonzero, "positive_lr_finite_nonzero_gradient_steps": positive_lr_grad,
        "optimizer_observations": steps, "trainable_parameter_update": parameters,
        "verified_nonzero_learning_signal": bool(groups) and finite_groups and binary_rewards and mixed > 0 and effective > 0
            and finite_gradients and positive_lr_grad > 0 and updated,
        "loss_note": "A group-averaged GRPO loss may equal zero while its gradient is nonzero.",
        "coverage_note": "Input size is not rollout coverage; full small-smoke coverage must be checked separately."}


def capture_trainable_parameters(model):
    """CPU copies only of actual trainable LoRA tensors, retaining their native dtype."""
    snapshots = {name: parameter.detach().cpu().contiguous().clone()
                 for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not snapshots or any("lora_" not in name for name in snapshots):
        raise ValueError("Signal audit expects nonempty LoRA-only trainable parameters")
    return snapshots


def parameter_update_summary(before, model):
    import torch
    after = capture_trainable_parameters(model)
    if before.keys() != after.keys():
        raise ValueError("Trainable parameter names changed during GRPO")
    hashes = [hashlib.sha256(), hashlib.sha256()]
    details, dtypes = [], {}
    squared_delta, maximum, changed = 0.0, 0.0, 0
    all_before, all_after = True, True
    for name in sorted(before):
        old, new = before[name], after[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError("Trainable parameter shape/dtype changed during GRPO")
        dtype = str(old.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + old.numel()
        metadata = compact({"name": name, "shape": list(old.shape), "dtype": dtype}).encode()
        for h, tensor in zip(hashes, (old, new)):
            h.update(metadata)
            # uint8 viewing supports BF16 too; numpy itself cannot encode BF16.
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        finite_before, finite_after = bool(torch.isfinite(old).all()), bool(torch.isfinite(new).all())
        all_before &= finite_before
        all_after &= finite_after
        count = int(torch.count_nonzero(old != new))
        delta = new.double() - old.double()
        l2_squared = float(delta.square().sum()) if finite_before and finite_after else None
        max_delta = float(delta.abs().max()) if finite_before and finite_after else None
        changed += count
        if l2_squared is not None:
            squared_delta += l2_squared
            maximum = max(maximum, max_delta)
        details.append({"name": name, "dtype": dtype, "elements": old.numel(), "changed_elements": count,
                        "all_before_finite": finite_before, "all_after_finite": finite_after,
                        "l2_delta": math.sqrt(l2_squared) if l2_squared is not None else None,
                        "max_abs_delta": max_delta})
    return {"tensor_count": len(before), "elements": sum(x.numel() for x in before.values()),
            "elements_by_dtype": dtypes, "sha256_before": hashes[0].hexdigest(), "sha256_after": hashes[1].hexdigest(),
            "all_before_finite": all_before, "all_after_finite": all_after,
            "changed_tensors": sum(row["changed_elements"] > 0 for row in details), "changed_elements": changed,
            "l2_delta": math.sqrt(squared_delta) if all_before and all_after else None,
            "max_abs_delta": maximum if all_before and all_after else None, "tensors": details}


def gradient_observation(model, optimizer, step):
    import torch
    gradients = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
    norms = torch.stack([gradient.float().norm() for gradient in gradients]) if gradients else None
    finite = norms is not None and bool(torch.isfinite(norms).all())
    norm = float(norms.double().norm()) if finite else None
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    return {"optimizer_step": step, "gradient_scope": "rank-local, after native clipping, before optimizer.step",
            "parameters_with_gradient": len(gradients), "all_gradients_finite": finite,
            "gradient_l2_after_clip": norm, "learning_rates": [lr if math.isfinite(lr) else None for lr in rates]}


def make_signal_callback(base_callback, output, rank):
    """Use the installed Trainer's events; never call backward/step or mutate control."""
    class SignalCallback(base_callback):
        def __init__(self):
            self.steps, self.parameters, self.before = [], None, None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.before = capture_trainable_parameters(model)
            layout = {}
            for tensor in self.before.values():
                key = str(tensor.dtype)
                layout[key] = layout.get(key, 0) + tensor.numel()
            (output / f"trainable_layout.rank{rank}.json").write_text(json.dumps({
                "tensor_count": len(self.before), "elements_by_dtype": layout,
                "measurement": "actual trainable LoRA tensors after trainer preparation; no dtype cast"}, indent=2) + "\n")

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            row = gradient_observation(model, optimizer, state.global_step + 1)
            self.steps.append(row)
            with (output / f"optimizer_signal.rank{rank}.jsonl").open("a") as handle:
                handle.write(compact(row) + "\n")

        def on_train_end(self, args, state, control, model=None, **kwargs):
            full = parameter_update_summary(self.before, model)
            path = output / f"trainable_parameter_signal.rank{rank}.json"
            path.write_text(json.dumps(full, indent=2, allow_nan=False) + "\n")
            self.parameters = {key: value for key, value in full.items() if key != "tensors"}
            self.parameters.update(audit_path=str(path), audit_sha256=digest(path))
            self.before = None

    return SignalCallback()


def make_observed_trainer(base_trainer, output, groups):
    """Read native rewards and loss masks after super(); return the same objects."""
    class ObservedGRPOTrainer(base_trainer):
        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            from accelerate.utils import gather_object
            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            metadata = gather_object([{"example_id": row["example_id"], "native_length": len(tokens)}
                                      for row, tokens in zip(inputs, completion_ids_list, strict=True)])
            if rewards.ndim != 2 or rewards.shape[1] != 1 or self.reward_weights.tolist() != [1.0]:
                raise ValueError("Signal audit requires the unchanged single outcome reward")
            self._signal_rewards = (metadata, rewards[:, 0].detach().cpu().tolist())
            return rewards

        def _generate_and_score_completions(self, inputs):
            from accelerate.utils import gather_object
            self._signal_rewards = None
            result = super()._generate_and_score_completions(inputs)
            metadata, rewards = self._signal_rewards
            mask = result["completion_mask"]
            if "tool_mask" in result:
                mask = mask * result["tool_mask"]
            tokens = mask.sum(dim=1).detach().cpu().tolist()
            advantages = result["advantages"].detach().cpu().tolist()
            actual = gather_object([{"example_id": row["example_id"], "tokens": int(count), "advantage": advantage}
                for row, count, advantage in zip(inputs, tokens, advantages, strict=True)])
            ids = [row["example_id"] for row in metadata]
            if ids != [row["example_id"] for row in actual]:
                raise ValueError("Reward and mask audit ordering differs")
            batch = reward_group_records(ids, rewards, [row["advantage"] for row in actual],
                [row["tokens"] for row in actual], [row["native_length"] for row in metadata], self.num_generations)
            if self.is_world_process_zero():
                with (output / "reward_group_signal.jsonl").open("a") as handle:
                    for row in batch:
                        row.update(group_index=len(groups), optimizer_steps_completed=self.state.global_step)
                        groups.append(row)
                        handle.write(compact(row) + "\n")
            return result

    return ObservedGRPOTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--questions", default="data/agent_feedback/train.questions.jsonl")
    parser.add_argument("--gold", default="data/agent_feedback/train.gold.jsonl")
    parser.add_argument("--eligible-ids", default="data/agent_feedback/train.success.jsonl")
    parser.add_argument("--protocol", default="results/agent_feedback/protocol_rl_native.json")
    parser.add_argument("--kb", default="datasets/kqa_pro/kb.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--preflight", action="store_true", help="Check without loading weights or training")
    parser.add_argument("--smoke-steps", type=int, choices=(1, 2),
                        help="Compatibility check only; output must contain smoke and no adapter is saved")
    args = parser.parse_args()
    run = run_settings(args.smoke_steps, args.output)
    if digest(frozen_grpo_entry.__file__) != EXPECTED_BASE_ENTRY_SHA256:
        raise RuntimeError("Historical GRPO entry changed; observation-only identity is no longer established")
    rank = int(os.environ.get("RANK", "0"))
    if args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("Batch size and accumulation must be positive")
    protocol = json.loads(Path(args.protocol).read_text())
    native_recorded = protocol_check(protocol)
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    if importlib.metadata.version("trl") != EXPECTED_TRL or digest(inspect.getfile(GRPOTrainer)) != EXPECTED_TRAINER_SHA256:
        raise RuntimeError("TRL version/source differs from the audited native-loop implementation")
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_result.json").exists() or (output / "run_started.json").exists():
        raise FileExistsError("Do not overwrite a started or completed RL run")
    rows = make_rows(read_jsonl(args.questions), read_jsonl(args.gold),
                     {x["id"] for x in read_jsonl(args.eligible_ids)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    preflight = tokenizer_preflight(tokenizer, rows)
    manifest = {"status": "preflight", "training_questions": len(rows),
                "smoke": run["smoke"], "formal_result_eligible": run["formal_result_eligible"],
                "planned_optimizer_steps": run["max_steps"],
                "native_contract_recorded_in_protocol": native_recorded, "native_contract": NATIVE_CONTRACT,
                "tokenizer_checks": preflight, "config": vars(args),
                "sha256": {name: digest(path) for name, path in {
                    "questions": args.questions, "gold": args.gold, "eligible_ids": args.eligible_ids,
                    "protocol": args.protocol, "script": __file__,
                    "frozen_grpo_entry": frozen_grpo_entry.__file__,
                    "environment": Path(__file__).with_name("environment.py"),
                    "trl_trainer": inspect.getfile(GRPOTrainer),
                    "trl_config": inspect.getfile(GRPOConfig),
                    "transformers_trainer": inspect.getfile(Trainer),
                    "transformers_trainer_callback": inspect.getfile(TrainerCallback)}.items()},
                "versions": {name: importlib.metadata.version(name) for name in
                             ["torch", "transformers", "peft", "accelerate", "trl"]}}
    if rank == 0:
        (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    if args.preflight:
        return
    if not native_recorded:
        raise ValueError("Record rl.native_trl_contract in the run protocol before native RL; see input_manifest.json")
    if not args.adapter:
        raise ValueError("--adapter must point to the completed SFT adapter; base-only RL is forbidden")
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("SFT adapter config and safetensors weights are required")
    manifest["sha256"].update({"adapter_config": digest(adapter / "adapter_config.json"),
                              "adapter_weights": digest(adapter / "adapter_model.safetensors")})
    # Each rank owns one KB engine, shared only by its independent environments.
    executor = KoPLExecutor(args.kb)
    lock = threading.RLock()
    audit_path = output / f"rollout_audit.rank{rank}.jsonl"
    if any(path.exists() for path in (audit_path, output / "reward_group_signal.jsonl",
            output / f"optimizer_signal.rank{rank}.jsonl", output / f"trainable_parameter_signal.rank{rank}.json",
            output / f"trainable_layout.rank{rank}.json")):
        raise FileExistsError("Do not append to an existing rollout or signal audit")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    config = GRPOConfig(output_dir=str(output), max_steps=run["max_steps"], learning_rate=5e-6,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation,
        num_generations=4, beta=0.0, num_iterations=1, loss_type="grpo", scale_rewards="group",
        epsilon=0.2, disable_dropout=True, temperature=1.0, top_p=1.0, top_k=0,
        max_completion_length=TOTAL_COMPLETION_TOKEN_CAP, max_tool_calling_iterations=MAX_CALLS,
        generation_kwargs={"max_new_tokens": PER_TURN_TOKEN_CAP, "use_cache": True}, use_vllm=False,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
        weight_decay=0.0, warmup_steps=0.03, logging_steps=1, save_strategy="no" if run["smoke"] else "steps",
        save_steps=50, save_total_limit=2, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True)
    signal_groups = []
    signal_callback = make_signal_callback(TrainerCallback, output, rank)
    observed_trainer = make_observed_trainer(GRPOTrainer, output, signal_groups)
    trainer = observed_trainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer, environment_factory=lambda: RewardEnvironment(executor, lock, audit_path),
        callbacks=[signal_callback])
    if (trainer.generation_config.max_new_tokens != PER_TURN_TOKEN_CAP or
            trainer.max_completion_length != TOTAL_COMPLETION_TOKEN_CAP or
            trainer.args.max_steps != run["max_steps"]):
        raise RuntimeError("Trainer did not retain the audited generation budgets")
    if trainer.is_world_process_zero():
        manifest["status"] = "started"
        manifest["grpo_config"] = config.to_dict()
        manifest["signal_audit"] = {
            "changes_training_semantics": False,
            "reward_groups": "native global pre-shuffle ordering; no answers/programs in audit or prompts",
            "effective_tokens": "native completion_mask multiplied by tool_mask, without modifying either",
            "gradient": "rank-local after native clipping, before optimizer.step",
            "parameters": "actual LoRA dtype, SHA256 and numerical difference before/after training",
            "eligible_question_ids": [row["example_id"] for row in rows],
            "generation_batch_size": config.generation_batch_size,
            "steps_per_generation": config.steps_per_generation,
            "planned_question_groups": run["max_steps"] * args.batch_size * args.accumulation
                * trainer.accelerator.num_processes // config.num_generations,
            "mask_truncated_completions": config.mask_truncated_completions,
            "loss_note": "Zero group-mean GRPO loss alone does not establish zero gradient."}
        (output / "run_started.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = trainer.train()
    # A compatibility smoke must not produce an adapter that could be mistaken
    # for a completed B/D result. Checkpoints are disabled above as well.
    if run["save_model"]:
        trainer.save_model(str(output / "model"))
    if trainer.is_world_process_zero():
        if run["save_model"]:
            tokenizer.save_pretrained(output / "model")
        seconds = time.monotonic() - start
        signals = signal_summary(signal_groups, signal_callback.steps, signal_callback.parameters,
                                 [row["example_id"] for row in rows])
        signals.update(gradient_parameter_scope=f"rank{rank}",
            optimizer_callbacks_match_trainer_steps=len(signal_callback.steps) == trainer.state.global_step,
            reward_groups_audit_path=str(output / "reward_group_signal.jsonl"),
            reward_groups_audit_sha256=digest(output / "reward_group_signal.jsonl"),
            elapsed_seconds_per_question_group=seconds / len(signal_groups) if signal_groups else None,
            elapsed_seconds_per_rollout=seconds / (len(signal_groups) * 4) if signal_groups else None,
            timing_note="Whole trainer runtime including gradient/parameter audit; retained token totals are not GPU FLOPs.")
        record = {"status": "completed", "seconds": seconds,
                  "smoke": run["smoke"], "formal_result_eligible": run["formal_result_eligible"],
                  "model_artifact": str(output / "model") if run["save_model"] else None,
                  "world_size": trainer.accelerator.num_processes,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "rank0_max_memory_allocated": torch.cuda.max_memory_allocated(),
                  "metrics": result.metrics, "optimizer_steps": trainer.state.global_step,
                  "seed": args.seed, "sha256": manifest["sha256"], "native_contract": NATIVE_CONTRACT,
                  "learning_signal": signals}
        (output / "run_result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
