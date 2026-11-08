"""Question-only typed-table generation; explicit CUDA run, separate offline score.

CPU imports/help never load torch or weights. Injectable fixture generators cannot
produce a model_generated manifest. Formal runs preserve the complete frozen order.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from .table_program import TableEpisode, execute_response, prompt_messages, tool_schemas, compact, digest
from .table_runtime_executor import TableRuntimeExecutor as TableExecutor
from .table_runtime import sha, verify_files, model_files, directory_identity, preflight_inputs, validate_runtime_evidence

SPLIT = Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")
SUPPLEMENT = Path("data/radar_domain_recovery_v2/compositional_supplement_20261108/manifest.json")
BUDGET = {"context_tokens": 16384, "max_calls": 24, "per_turn_generated_tokens": 192, "total_generated_tokens": 2048}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def validate_questions(questions):
    if not questions or any(not isinstance(r, dict) or set(r) != {"id", "question"}
                            or not isinstance(r["id"], str) or not isinstance(r["question"], str) for r in questions):
        raise ValueError("Generation accepts only nonempty ID/question rows, never references or answer fields")
    if len({r["id"] for r in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")


def load_partition(manifest=SPLIT, split="dev", supplement_manifest=SUPPLEMENT):
    """References/metadata are descriptors only; generation never opens them."""
    pinned = read("results/radar_domain/frozen_20261107/summary.json")["freeze_manifest"]
    if sha(manifest) != pinned["sha256"]:
        raise ValueError("Use the registered frozen manifest, not a rewritten or reduced population")
    parent = read(manifest)
    if parent.get("schema") != "radar_domain_table_split_v1" or parent.get("status") != "frozen":
        raise ValueError("Frozen domain table split required")
    if split == "supplement":
        pinned_extra = read("results/radar_domain/compositional_supplement_20261108/summary.json")["manifest"]
        if sha(supplement_manifest) != pinned_extra["sha256"]:
            raise ValueError("Use the registered frozen supplement manifest")
        extra = read(supplement_manifest)
        if (extra.get("schema") != "radar_table_evaluation_supplement_v1"
                or extra.get("status") != "frozen_before_model_runs" or extra.get("training_allowed") is not False
                or extra["parent_split"]["sha256"] != sha(manifest)):
            raise ValueError("Matching frozen evaluation supplement required")
        desc = {name: extra["outputs"][name + ".json" if name == "catalog" else name + ".jsonl"]
                for name in ("questions", "references", "catalog", "metadata")}
        desc.update(ids=extra["ids"], group_by_id=extra["group_by_id"])
        selected_manifest = Path(supplement_manifest)
    else:
        if split not in {"train", "dev", "holdout"}:
            raise ValueError("Unknown frozen split")
        desc = parent["splits"][split]
        selected_manifest = Path(manifest)
    verify_files({desc[k]["path"]: desc[k]["sha256"] for k in ("questions", "catalog")})
    questions, catalog = rows(desc["questions"]["path"]), read(desc["catalog"]["path"])
    validate_questions(questions)
    if [r["id"] for r in questions] != desc["ids"]:
        raise ValueError("Question order differs from frozen manifest")
    return {"split": split, "manifest": str(selected_manifest), "manifest_sha256": sha(selected_manifest),
            "parent_manifest": str(manifest), "parent_manifest_sha256": sha(manifest),
            "descriptor": desc, "questions": questions, "catalog": catalog}


def encode_prompt(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tools=tool_schemas(), tokenize=True,
                                        return_dict=False, add_generation_prompt=True)
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("Tokenizer must return complete integer IDs")
    return ids


def trim_tokens(tokens, tokenizer):
    tokens = list(tokens)
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Generated tokens must be integer IDs")
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos is not None and eos in tokens:
        counted = tokens[:tokens.index(eos) + 1]
        return counted, counted[:-1]
    while tokens and pad is not None and tokens[-1] == pad:
        tokens.pop()
    return tokens, tokens


class TransformersGenerator:
    """Constructed only after CUDA/input verification by generate()."""
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def generate(self, prompt_ids, max_new_tokens):
        import torch
        ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return output[0, ids.shape[1]:].tolist()


def run_episode(question, executor, tokenizer, generator, *, generation_id="cpu_fixture",
                budgets=None, budget_check=None):
    """Deterministic control flow; malformed actions stay in the episode denominator."""
    validate_questions([question])
    limits = dict(BUDGET if budgets is None else budgets)
    if set(limits) != set(BUDGET) or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("Positive explicit token/call budgets required")
    episode = TableEpisode(executor, limits["max_calls"])
    messages = prompt_messages(question["question"])
    turn_tokens, input_total, generated_total = [], 0, 0
    stop = None
    while not episode.done:
        if budget_check:
            budget_check()
        prompt = encode_prompt(tokenizer, messages)
        if len(prompt) >= limits["context_tokens"]:
            stop = "context_budget"; break
        remaining = limits["total_generated_tokens"] - generated_total
        if remaining <= 0:
            stop = "generation_budget"; break
        allowance = min(limits["per_turn_generated_tokens"], remaining, limits["context_tokens"] - len(prompt))
        generated = generator.generate(prompt, allowance)
        if len(generated) > allowance:
            raise ValueError("Generator exceeded its granted token budget")
        counted, content = trim_tokens(generated, tokenizer)
        text = tokenizer.decode(content, skip_special_tokens=False)
        # Unexpected backend exceptions abort the run and retain partial artifacts;
        # they are not silently converted to normal model-quality observations.
        observation = execute_response(episode, text)
        messages.extend([{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}])
        after = encode_prompt(tokenizer, messages)
        input_total += len(prompt); generated_total += len(counted)
        turn_tokens.append({"input_tokens": len(prompt), "generated_tokens": len(counted),
                            "granted_tokens": allowance, "generated_token_ids": counted,
                            "context_tokens_after_observation": len(after)})
        if episode.done:
            stop = "finished" if episode.answer is not None else "call_budget"
        elif len(after) >= limits["context_tokens"]:
            stop = "context_budget"; break
        elif generated_total >= limits["total_generated_tokens"]:
            stop = "generation_budget"; break
    return {"id": question["id"], "messages": messages, "events": episode.events,
            "answer": episode.answer, "calls": episode.calls, "selected_handle": episode.selected,
            "invalid_calls": sum(not e["observation"]["ok"] for e in episode.events),
            "input_tokens": input_total, "generated_tokens": generated_total,
            "total_tokens": input_total + generated_total, "turn_tokens": turn_tokens,
            "stop_reason": stop or "call_budget", "budgets": limits,
            "provenance": {"kind": "cpu_fixture", "generation_id": generation_id},
            "training_eligible": False}


def implementation_files():
    paths = ["experiments/radar_domain/" + name + ".py" for name in
             ("table_inference", "table_runtime", "table_runtime_executor", "table_program", "table_job_budget")]
    return {p: sha(p) for p in paths}


def collector_training_identity(path, *, role, adapter_dir, parent_manifest, base_model_dir=None):
    result = read(path)
    expected_stage, arm = ("initial", "I") if role == "I_domain" else ("continuation", role[0])
    if (result.get("schema") != "radar_table_training_run_result_v1" or result.get("status") != "completed"
            or result.get("stage") != expected_stage or result.get("arm") != arm):
        raise ValueError("Collector checkpoint needs its completed matching training result")
    preparation = result["preparation"]
    verify_files({preparation["path"]: preparation["sha256"]})
    prepared = read(preparation["path"])
    parent = read(parent_manifest)
    if (prepared.get("schema") != "radar_table_training_preparation_v1"
            or prepared.get("status") != "completed" or prepared.get("stage") != expected_stage
            or prepared.get("source_split") != "train" or prepared.get("training_eligible") is not True
            or prepared["split_manifest"]["sha256"] != sha(parent_manifest)):
        raise ValueError("Collector training preparation is not the frozen training split")
    if arm == "I" and prepared.get("source_ids") != parent["splits"]["train"]["ids"]:
        raise ValueError("Initial collector was not prepared from all frozen training questions")
    final = result["final_adapter_files_sha256"]
    verify_files(final)
    if any(Path(p).resolve().parent != Path(adapter_dir).resolve() for p in final):
        raise ValueError("Training result points to another adapter directory")
    if not {"adapter_config.json", "adapter_model.safetensors"} <= {Path(p).name for p in final}:
        raise ValueError("Completed adapter identities are incomplete")
    if base_model_dir is None:
        raise ValueError("Generation must identify the collector's base model")
    config_binding = result["config"]
    verify_files({config_binding["path"]: config_binding["sha256"]})
    trained_config = read(config_binding["path"])
    actual_base = directory_identity(model_files(base_model_dir), base_model_dir)
    trained_base = directory_identity(result["base_and_starting_model_files_sha256"], trained_config["base_model_dir"])
    if actual_base != trained_base:
        raise ValueError("Generation base weights/config/tokenizer differ from the completed training run")
    return {"path": str(path), "sha256": sha(path)}, result


def load_config(path):
    config = read(path)
    required = {"schema", "split_manifest", "split", "base_model_dir", "adapter_dir", "model_role",
                "runtime_user", "gpu_index", "seed", "max_gpu_hours", "ledger_path", "job_id", "collector_training_result"}
    if set(config) - (required | {"supplement_manifest"}) or not required <= set(config):
        raise ValueError("Explicit generation config fields required")
    if config["schema"] != "radar_table_generation_config_v1" or config["model_role"] not in {"I_domain", "A_domain", "C_domain"}:
        raise ValueError("Unsupported formal domain model role")
    if config["split"] == "train" and config["model_role"] != "I_domain":
        raise ValueError("Training failures are collected only from I_domain")
    if not config["adapter_dir"] or type(config["seed"]) is not int or not 0 < config["max_gpu_hours"] <= 8:
        raise ValueError("A local adapted checkpoint, seed and bounded runtime are required")
    data = load_partition(config["split_manifest"], config["split"], config.get("supplement_manifest", SUPPLEMENT))
    training, _ = collector_training_identity(config["collector_training_result"], role=config["model_role"],
                                              adapter_dir=config["adapter_dir"], parent_manifest=config["split_manifest"],
                                              base_model_dir=config["base_model_dir"])
    inputs = {str(path): sha(path), data["manifest"]: data["manifest_sha256"],
              data["parent_manifest"]: data["parent_manifest_sha256"]}
    inputs.update({data["descriptor"][k]["path"]: data["descriptor"][k]["sha256"] for k in ("questions", "catalog")})
    inputs[training["path"]] = training["sha256"]
    return config, data, inputs


def preflight(config_path, output):
    config, _, inputs = load_config(config_path)
    return preflight_inputs(purpose="generation", base_model_dir=config["base_model_dir"],
        adapter_dir=config["adapter_dir"], input_files_sha256=inputs,
        implementation_sha256=implementation_files(), output=output,
        ledger_path=config["ledger_path"], job_id=config["job_id"] + "_cuda_preflight",
        expected_user=config["runtime_user"], gpu_index=config["gpu_index"])


def generate(config_path, runtime_evidence, output_dir):
    config, data, inputs = load_config(config_path)
    from .table_job_budget import JobBudget, require_round_ledger
    require_round_ledger(config["ledger_path"])
    output = Path(output_dir)
    if not output.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Full generated traces must remain under the private data directory")
    if output.exists():
        raise FileExistsError("Never overwrite a model run")
    code = implementation_files()
    weights = model_files(config["base_model_dir"], config["adapter_dir"])
    validate_runtime_evidence(runtime_evidence, purpose="generation", model_files_sha256=weights,
        implementation_sha256=code, input_files_sha256=inputs, expected_user=config["runtime_user"])
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "rollouts.jsonl"
    generation_id = config["job_id"]
    manifest = {"schema": "radar_table_rollout_manifest_v1", "status": "running", "origin": "model_generated",
        "generation_id": generation_id, "split": data["split"], "split_sha256": data["manifest_sha256"],
        "split_manifest": data["manifest"], "parent_split_sha256": data["parent_manifest_sha256"],
        "parent_split_manifest": data["parent_manifest"],
        "collector_role": config["model_role"], "base_model_dir": config["base_model_dir"],
        "adapter_dir": config["adapter_dir"], "model_files_sha256": weights, "implementation_sha256": code,
        "base_model_identity": directory_identity(weights, config["base_model_dir"]),
        "collector_training_result": {"path": config["collector_training_result"], "sha256": sha(config["collector_training_result"])},
        "input_files_sha256": inputs, "runtime_evidence": {"path": str(runtime_evidence), "sha256": sha(runtime_evidence)},
        "questions": data["descriptor"]["questions"], "references": data["descriptor"]["references"],
        "catalog": data["descriptor"]["catalog"], "metadata": data["descriptor"]["metadata"],
        "budgets": BUDGET, "max_calls": 24, "seed": config["seed"], "decoding": "greedy", "batch_size": 1,
        "config": {"path": str(config_path), "sha256": sha(config_path)}, "model_gpu_smoke": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = output / "generation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed = 0
    try:
        with JobBudget(ledger_path=config["ledger_path"], job_id=generation_id, phase="generation",
                       max_gpu_hours=config["max_gpu_hours"], gpu_count=1) as budget:
            import torch
            import transformers
            import tokenizers
            import peft
            from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
            from peft import PeftModel
            set_seed(config["seed"])
            tokenizer = AutoTokenizer.from_pretrained(config["base_model_dir"], local_files_only=True, trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config["base_model_dir"], local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
            model = PeftModel.from_pretrained(model, config["adapter_dir"], is_trainable=False)
            model.eval(); budget.check()
            with torch.inference_mode():
                smoke = model(input_ids=torch.tensor([[tokenizer.eos_token_id]], device="cuda"), use_cache=False)
                finite = bool(torch.isfinite(smoke.logits).all().item())
            if not finite:
                raise ValueError("Actual model GPU forward returned nonfinite logits")
            del smoke
            manifest["model_gpu_smoke"] = {"finite_logits": True, "model_forward_executed": True,
                "task_questions_used": False, "torch": torch.__version__, "transformers": transformers.__version__}
            manifest["runtime_versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "peft": peft.__version__, "cuda": torch.version.cuda,
                "dtype": "bfloat16", "attention": "sdpa"}
            generator, executor = TransformersGenerator(model, tokenizer), TableExecutor(data["catalog"])
            with prediction_path.open("x") as stream:
                for question in data["questions"]:
                    budget.check()
                    row = run_episode(question, executor, tokenizer, generator, generation_id=generation_id,
                                      budget_check=budget.check)
                    # Only this verified, real-model CLI path admits generation
                    # provenance. Direct injectable-controller calls stay fixtures.
                    row["provenance"] = {"kind": "model_generated", "generation_id": generation_id}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
                    completed += 1
                    if completed % 16 == 0:
                        print(json.dumps({"completed": completed, "total": len(data["questions"])}), flush=True)
            verify_files(inputs); verify_files(code)
            manifest.update(status="outputs_written_budget_finalization_pending", completed_questions=completed,
                rollouts={"path": str(prediction_path), "sha256": sha(prediction_path), "rows": completed})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            del generator, model
            torch.cuda.empty_cache()
        ledger = read(config["ledger_path"])
        job = next(j for j in ledger["jobs"] if j["job_id"] == generation_id)
        if job["status"] != "completed":
            raise ValueError("Generation GPU budget did not reach completed terminal state")
        manifest["gpu_budget_completion"] = {"ledger_path": config["ledger_path"], "job_id": generation_id,
            "job_sha256": digest(job), "status": "completed", "gpu_hours": job["gpu_hours"]}
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException as exc:
        manifest.update(status="failed", completed_questions=completed,
                        failure={"type": type(exc).__name__, "message": str(exc)})
        if prediction_path.exists():
            manifest["rollouts"] = {"path": str(prediction_path), "sha256": sha(prediction_path), "rows": completed}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    print(json.dumps({"status": "completed", "questions": completed, "manifest": str(manifest_path)}))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preflight"); prep.add_argument("--config", required=True); prep.add_argument("--output", required=True)
    run = commands.add_parser("generate"); run.add_argument("--config", required=True)
    run.add_argument("--runtime-evidence", required=True); run.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.command == "preflight":
        result = preflight(args.config, args.output)
        print(json.dumps({"status": result["status"], "scope": result["scope"], "output": args.output}))
    else:
        generate(args.config, args.runtime_evidence, args.output_dir)


if __name__ == "__main__":
    main()
