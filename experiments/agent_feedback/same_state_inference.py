"""Resume frozen histories with existing A/C adapters; no gold input accepted.

Requires an execution binding written after Linux replay/token preflight and a
single unoccupied GPU. A design protocol alone is deliberately insufficient.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

from .download_weights import ROOT as MODEL_ROOT, FILES
from .environment import compact, tool_schemas
from .inference import execute_response, response_tokens
from .recovery import sha256
from .same_state import FIELDS, restore, object_sha
from .training_data import read_jsonl

ARMS = {"A_20261003": "clean_sft_v1", "C_20261003": "recovery_sft_v1",
        "A_20261004": "clean_sft_seed20261004", "C_20261004": "recovery_sft_seed20261004"}


def validate_inventory(states):
    if not states or len(states) % 2:
        raise ValueError("Expected a nonempty sequence of complete state pairs")
    ids = set()
    for clean, divergent in zip(states[::2], states[1::2]):
        if (set(clean) != FIELDS or set(divergent) != FIELDS
                or clean["condition"] != "clean" or divergent["condition"] != "divergent"
                or clean["id"] != divergent["id"] or clean["id"] in ids
                or clean["messages"] != divergent["messages"][:-2]
                or clean["remaining_calls"] != divergent["remaining_calls"]
                or clean["remaining_generated_tokens"] != divergent["remaining_generated_tokens"]):
            raise ValueError("State order, identity, shared prefix or remaining budgets differ")
        ids.add(clean["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (pwd.getpwuid(os.geteuid()).pw_name != "syx" or sys.platform != "linux"
            or sys.version_info[:2] != (3, 10) or os.environ.get("PYTHONHASHSEED") != "20261003"):
        raise ValueError("Use the registered syx/Linux/Python3.10 execution environment")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1", "2", "3"):
        raise ValueError("Exactly one registered GPU is required")
    active = subprocess.check_output(["/usr/bin/nvidia-smi", "-i", gpu,
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU is occupied or process ownership is unresolved; never interrupt it")
    binding = json.loads(args.binding.read_text())
    if (binding.get("schema") != "same_state_execution_binding_v1" or binding.get("ready") is not True
            or binding.get("state_replay_passed") is not True or binding.get("token_budget_passed") is not True
            or binding.get("protocol_sha256") != sha256(binding["protocol_path"])
            or binding.get("states_sha256") != sha256(args.states)
            or binding.get("inference_sha256") != sha256(__file__)):
        raise ValueError("A frozen, replay- and token-checked execution binding is required")
    if args.output.exists() or args.output.with_suffix(".runtime.json").exists():
        raise FileExistsError("Never overwrite or resume a partial prediction run")
    for path, expected in binding["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Execution input changed: {path}")
    for filename, expected in FILES.items():
        if sha256(MODEL_ROOT / filename) != expected:
            raise ValueError(f"Pinned backbone file changed: {filename}")
    adapter = Path("results/agent_feedback") / ARMS[args.arm] / "model"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / filename
        if binding["inputs_sha256"].get(str(path)) != sha256(path):
            raise ValueError("Original checkpoint identity is not bound")
    states = read_jsonl(args.states)
    validate_inventory(states)
    from experiments.condition_consistency.executor import KoPLExecutor
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    if binding["versions"] != versions:
        raise ValueError("Runtime differs from the frozen execution binding")
    set_seed(20261003)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    executor = KoPLExecutor("datasets/kqa_pro/kb.json")
    # All inputs must replay and bind to token IDs before loading a model.
    for row in states:
        restore(executor, row)
        ids = tokenizer.apply_chat_template(row["messages"], tools=tool_schemas(),
            tokenize=True, return_dict=False, add_generation_prompt=True)
        if (object_sha(ids) != binding["initial_prompt_ids_sha256"][row["id"]][row["condition"]]
                or len(ids) >= 8192):
            raise ValueError("Input identity or initial context budget differs")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, adapter).eval()
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for index, row in enumerate(states):
            episode = restore(executor, row)
            messages = deepcopy(row["messages"])
            generated, input_tokens, stop = 0, 0, None
            for _ in range(row["remaining_calls"]):
                ids = tokenizer.apply_chat_template(messages, tools=tool_schemas(),
                    tokenize=True, return_dict=False, add_generation_prompt=True)
                if len(ids) >= 8192:
                    stop = "context_budget"; break
                if generated >= 2048:
                    stop = "generation_budget"; break
                allowance = min(192, 2048 - generated, 8192 - len(ids))
                inputs = tokenizer.pad({"input_ids": [ids]}, padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=allowance, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                tokens, content = response_tokens(output[0, len(ids):].tolist(),
                    tokenizer.eos_token_id, tokenizer.pad_token_id)
                generated += len(tokens); input_tokens += len(ids)
                text = tokenizer.decode(content, skip_special_tokens=False)
                observation = execute_response(episode, text)
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "tool", "content": compact(observation)}])
                if episode.done:
                    stop = "finished" if episode.prediction is not None else "call_budget"; break
            continuation = episode.events[row["prefix_calls"]:]
            prediction = {"id": row["id"], "condition": row["condition"], "arm": args.arm,
                "initial_state_sha256": row["state_sha256"], "prefix_calls": row["prefix_calls"],
                "remaining_calls": row["remaining_calls"], "calls": episode.calls,
                "continuation_calls": len(continuation), "prediction": episode.prediction,
                "stop_reason": stop or "call_budget", "messages": messages, "events": episode.events,
                "invalid_calls": sum(not e["observation"]["ok"] for e in continuation),
                "input_tokens": input_tokens, "generated_tokens": generated, "total_tokens": input_tokens + generated}
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n"); handle.flush()
            print(json.dumps({"states_completed": index + 1, "seconds": time.monotonic() - start}), flush=True)
    args.output.with_suffix(".runtime.json").write_text(json.dumps({"status": "completed", "arm": args.arm,
        "states": len(states), "seconds": time.monotonic() - start, "binding_sha256": sha256(args.binding),
        "predictions_sha256": sha256(args.output), "versions": versions,
        "token_accounting": "New assistant generation only; input counts full unpadded prefills including fixed history.",
        "training": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
