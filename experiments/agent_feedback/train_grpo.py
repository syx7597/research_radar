"""Outcome-only TRL GRPO, continuing an SFT LoRA adapter.

Run --preflight before loading model weights. Native TRL 1.14.1 counts tool
observations in its completion budget and accepts multiple calls per turn. Those
semantics must be recorded in the run protocol before training; they are not the
strict single-call evaluation loop. No optimizer or GRPO loss is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import threading
import time

from experiments.condition_consistency.executor import KoPLExecutor, compare_answers
from .environment import Episode, MAX_CALLS, call_message, compact, prompt_messages, tool_schemas
from .training_data import read_jsonl


EXPECTED_TRL = "1.14.1"
EXPECTED_TRAINER_SHA256 = "ec6ff9aa69de59237588aafe3de861342b78c3779a595d08a763eed120a3c406"
TOTAL_COMPLETION_TOKEN_CAP = 4096
PER_TURN_TOKEN_CAP = 192
MAX_CONTEXT = 8192
FORMAL_UPDATES = 200
NATIVE_CONTRACT = {
    "completion_budget_includes_tool_observations": True,
    "total_completion_token_cap": TOTAL_COMPLETION_TOKEN_CAP,
    "per_turn_max_new_tokens": PER_TURN_TOKEN_CAP,
    "max_tool_calling_iterations": 24,
    "episode_max_calls": 24,
    "multiple_tool_calls_per_turn": "native_trl_sequential",
    "unknown_tools_and_parse_errors": "native_trl_not_episode_calls",
    "post_finish_generation": "native_trl_until_no_tool_call_or_budget",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class JSONObservation(dict):
    """Keep a dictionary result while TRL's str(result) emits canonical JSON."""

    def __str__(self):
        return compact(self)


class RewardEnvironment:
    """Only step/finish are model-visible; reference answers stay in private state."""

    def __init__(self, executor, lock=None, audit_path=None):
        self._executor = executor
        self._lock = lock if lock is not None else threading.RLock()
        self._audit_path = Path(audit_path) if audit_path is not None else None
        self._episode = None
        self._answer = None
        self._id = None
        self._cached_reward = None

    def reset(self, prompt, example_id, reference_answer, **kwargs):
        # TRL appends a non-None reset return to the user's prompt. Return None:
        # neither the answer nor any gold program becomes an observation.
        if kwargs:
            raise ValueError(f"Unexpected dataset columns: {sorted(kwargs)}")
        if prompt != prompt_messages(prompt[-1]["content"]):
            raise ValueError("RL prompt must contain only canonical instructions and question")
        self._episode = Episode(self._executor, max_calls=MAX_CALLS)
        self._answer = reference_answer
        self._id = example_id
        self._cached_reward = None
        return None

    def step(self, function: str, inputs: list[str], dependencies: list[int]) -> dict:
        """Execute one KoPL function and retain its result as an integer handle.

        Args:
            function: KoPL function name from the instruction.
            inputs: Literal string arguments, excluding dependency handles.
            dependencies: Handles returned by previous successful calls.

        Returns:
            Result type, handle, bounded observation, or an execution error.
        """
        with self._lock:
            if self._episode is None:
                raise RuntimeError("Environment has not been reset")
            return JSONObservation(self._episode.step(function, inputs, dependencies))

    def finish(self, answer_handle: int) -> dict:
        """Finish the episode using a previously computed answer result.

        Args:
            answer_handle: Integer handle of an answer-producing KoPL function.

        Returns:
            The computed answer and termination status, or an error.
        """
        with self._lock:
            if self._episode is None:
                raise RuntimeError("Environment has not been reset")
            return JSONObservation(self._episode.finish(answer_handle))

    def get_reward(self):
        if self._episode is None:
            raise RuntimeError("Environment has not been reset")
        if self._cached_reward is None:
            self._cached_reward = float(compare_answers(self._answer, self._episode.prediction))
            if self._audit_path is not None:
                record = {"id": self._id, "reward": self._cached_reward,
                          "calls": self._episode.calls, "finished": self._episode.prediction is not None,
                          "invalid_calls": sum(not x["observation"]["ok"] for x in self._episode.events)}
                with self._lock, self._audit_path.open("a") as handle:
                    handle.write(compact(record) + "\n")
        return self._cached_reward


def make_rows(questions, gold, eligible_ids=None):
    """Join IDs privately; the Dataset contains no gold program or gold trajectory."""
    if not questions or not gold:
        raise ValueError("Empty training data")
    if any(set(row) != {"id", "question"} for row in questions):
        raise ValueError("Questions input accepts only id and question")
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    by_id = {row["id"]: row for row in gold}
    if len(by_id) != len(gold) or set(by_id) != {r["id"] for r in questions}:
        raise ValueError("Gold and questions must have identical unique IDs")
    if eligible_ids is not None and not set(eligible_ids) <= set(by_id):
        raise ValueError("Eligible IDs contain questions outside the training split")
    result = []
    for row in questions:
        reference = by_id[row["id"]]
        if row["question"] != reference["question"]:
            raise ValueError("Gold/question text mismatch")
        if eligible_ids is None or row["id"] in eligible_ids:
            result.append({"prompt": prompt_messages(row["question"]), "example_id": row["id"],
                           "reference_answer": str(reference["answer"])})
    if not result:
        raise ValueError("No eligible training questions")
    return result


def protocol_check(protocol):
    if "total_generated_token_cap" in protocol["rl"]:
        raise ValueError("Native RL uses total_completion_token_cap including observations, not a generated-only cap")
    expected = {"group_size": 4, "max_updates": FORMAL_UPDATES, "learning_rate": 5e-6,
                "beta": 0.0, "num_iterations": 1, "max_calls": MAX_CALLS,
                "per_turn_max_new_tokens": PER_TURN_TOKEN_CAP,
                "total_completion_token_cap": TOTAL_COMPLETION_TOKEN_CAP}
    for name, value in expected.items():
        if protocol["rl"].get(name) != value:
            raise ValueError(f"Unexpected frozen RL setting: {name}")
    return protocol["rl"].get("native_trl_contract") == NATIVE_CONTRACT


def run_settings(smoke_steps, output):
    smoke = smoke_steps is not None
    if smoke and (type(smoke_steps) is not int or smoke_steps not in (1, 2)):
        raise ValueError("Smoke runs allow only one or two optimizer updates")
    if smoke and "smoke" not in str(output).lower():
        raise ValueError("Smoke output path must contain 'smoke'")
    return {"smoke": smoke, "max_steps": smoke_steps if smoke else FORMAL_UPDATES,
            "formal_result_eligible": not smoke, "save_model": not smoke}


def context_bound(prompt_lengths):
    if not prompt_lengths or any(type(length) is not int or length < 1 for length in prompt_lengths):
        raise ValueError("Preflight requires nonempty positive prompt lengths")
    bound = max(prompt_lengths) + TOTAL_COMPLETION_TOKEN_CAP + PER_TURN_TOKEN_CAP
    if bound > MAX_CONTEXT:
        raise ValueError("Native completion budget cannot guarantee the 8192 context cap")
    return bound


def tokenizer_preflight(tokenizer, rows):
    from transformers.utils.chat_template_utils import get_json_schema
    from trl.chat_template_utils import (
        add_response_schema, get_training_chat_template, is_chat_template_prefix_preserving,
        parse_response, supports_tool_calling,
    )
    if not supports_tool_calling(tokenizer):
        raise ValueError("Tokenizer does not support tool calling")
    env = RewardEnvironment(None)
    methods = [method for name, method in inspect.getmembers(env, inspect.ismethod)
               if not name.startswith("_") and name not in {"reset", "get_reward"}]
    actual_schemas = [get_json_schema(method) for method in methods]
    if actual_schemas != tool_schemas():
        raise ValueError("TRL-generated schemas/order differ from canonical SFT tool_schemas()")
    # Dict equality alone ignores key insertion order; Jinja's JSON rendering
    # can expose that order to the model. Compare actual rendered prompt tokens.
    schema_probe = prompt_messages("Count the entities named Alpha.")
    schema_kwargs = {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
    if tokenizer.apply_chat_template(schema_probe, tools=actual_schemas, **schema_kwargs) != \
            tokenizer.apply_chat_template(schema_probe, tools=tool_schemas(), **schema_kwargs):
        raise ValueError("TRL-generated schema serialization changes SFT prompt tokens")
    template = None if is_chat_template_prefix_preserving(tokenizer) else get_training_chat_template(tokenizer)
    # Compare initial prompts and a multi-turn history under the exact selected
    # template. Qwen2.5 currently needs no replacement; don't assume this forever.
    probe = prompt_messages("Count the entities named Alpha.")
    probe += [call_message("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []}),
              {"role": "tool", "content": compact({"ok": True, "handle": 0, "count": 1})},
              call_message("step", {"function": "Count", "inputs": [], "dependencies": [0]}),
              {"role": "tool", "content": compact({"ok": True, "handle": 1, "type": "answer", "value": "1"})}]
    for size in range(2, len(probe) + 1):
        kwargs = {"tokenize": True, "tools": tool_schemas(), "add_generation_prompt": True,
                  "return_dict": False}
        old = tokenizer.apply_chat_template(probe[:size], **kwargs)
        new = tokenizer.apply_chat_template(probe[:size], chat_template=template, **kwargs)
        if old != new:
            raise ValueError("TRL-selected chat template changes SFT history tokens")
    add_response_schema(tokenizer)
    action = call_message("step", {"function": "Find", "inputs": ["Alpha"], "dependencies": []})["content"]
    ids = tokenizer.encode(action, add_special_tokens=False) + [tokenizer.eos_token_id]
    prefix = tokenizer.apply_chat_template(probe[:2], tools=actual_schemas,
                chat_template=template, tokenize=True, add_generation_prompt=True, return_dict=False)
    parsed = parse_response(tokenizer, ids, prefix=prefix)
    calls = parsed.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"] != {"name": "step", "arguments": {
            "function": "Find", "inputs": ["Alpha"], "dependencies": []}}:
        raise ValueError("TRL response parser did not recover the canonical SFT action")
    lengths = [len(tokenizer.apply_chat_template(row["prompt"], tools=actual_schemas,
                   chat_template=template, tokenize=True, add_generation_prompt=True,
                   return_dict=False)) for row in rows]
    # Native TRL generates a full turn before trimming to the total completion
    # budget. Reserve an extra turn so even temporary generated sequences fit.
    temporary_context_bound = context_bound(lengths)
    return {"tool_names": [x["function"]["name"] for x in actual_schemas],
            "schemas_equal_sft": True, "prompt_tokens_max": max(lengths),
            "chat_template_replaced": template is not None, "template_probe_tokens_equal": True,
            "canonical_action_parser_pass": True,
            "total_completion_token_cap_including_observations": TOTAL_COMPLETION_TOKEN_CAP,
            "temporary_context_bound_with_extra_turn": temporary_context_bound,
            "chat_template_sha256": hashlib.sha256((template or tokenizer.chat_template).encode()).hexdigest()}


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
    rank = int(os.environ.get("RANK", "0"))
    if args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("Batch size and accumulation must be positive")
    protocol = json.loads(Path(args.protocol).read_text())
    native_recorded = protocol_check(protocol)
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
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
                    "environment": Path(__file__).with_name("environment.py"),
                    "trl_trainer": inspect.getfile(GRPOTrainer)}.items()},
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
    if audit_path.exists():
        raise FileExistsError("Do not append to an existing rollout audit")
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
    trainer = GRPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer, environment_factory=lambda: RewardEnvironment(executor, lock, audit_path))
    if (trainer.generation_config.max_new_tokens != PER_TURN_TOKEN_CAP or
            trainer.max_completion_length != TOTAL_COMPLETION_TOKEN_CAP or
            trainer.args.max_steps != run["max_steps"]):
        raise RuntimeError("Trainer did not retain the audited generation budgets")
    if trainer.is_world_process_zero():
        manifest["status"] = "started"
        manifest["grpo_config"] = config.to_dict()
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
        record = {"status": "completed", "seconds": time.monotonic() - start,
                  "smoke": run["smoke"], "formal_result_eligible": run["formal_result_eligible"],
                  "model_artifact": str(output / "model") if run["save_model"] else None,
                  "world_size": trainer.accelerator.num_processes,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "rank0_max_memory_allocated": torch.cuda.max_memory_allocated(),
                  "metrics": result.metrics, "optimizer_steps": trainer.state.global_step,
                  "seed": args.seed, "sha256": manifest["sha256"], "native_contract": NATIVE_CONTRACT}
        (output / "run_result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
