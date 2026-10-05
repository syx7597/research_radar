"""Fixed before/after interface diagnostic; no reference access during generation.

Synthetic held groups test instance/wording retention only. Radar questions are
previously exposed AI development derivatives, not independent equipment gold.
The original twelve-pair baseline is reused, never silently replaced.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments.agent_feedback.environment import compact, tool_schemas
from experiments.agent_feedback.inference import execute_response, generation_groups, response_tokens
from .development_environment import DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages, render_records
from .development_analysis import read_jsonl, require, replay_row, TOTAL_KEYS
from . import interface_training as training
from . import lookup_probe as old

OUT = training.OUT / "evaluation"
PROTOCOL = OUT / "protocol.json"
AUDIT = Path("artifacts/thesis_direction_review/radar_interface_synthetic_v1/independent_ai_audit.json")
MODELS = old.MODELS
PHASES = ("before", "after")
sha, write_new = old.sha, old.write_new


def question_rows(phase):
    if phase not in PHASES:
        raise ValueError("Unknown phase")
    synthetic = read_jsonl(training.DATA / "holdout.questions.jsonl")
    require(all(set(row) == {"id", "question", "knowledge_id"} for row in synthetic), "Question-only synthetic schema required")
    rows = [{**row, "dataset": "synthetic", "language": row["id"].rsplit("-", 1)[-1]} for row in synthetic]
    if phase == "after":
        for language in old.LANGUAGES:
            radar = read_jsonl(old.DATA / f"questions.{language}.jsonl")
            require(all(set(row) == {"id", "question"} for row in radar), "Question-only radar schema required")
            rows += [{**row, "id": row["id"] + "-" + language, "original_id": row["id"],
                      "knowledge_id": "radar_development_v2", "dataset": "radar_seen", "language": language}
                     for row in radar]
    require(len(rows) == (40 if phase == "before" else 64), "Fixed question coverage changed")
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate question IDs")
    require(all(row["language"] in old.LANGUAGES for row in rows), "Invalid language")
    return rows


def inference_batches(questions, size):
    # Preserve the old diagnostic's independent zh and en 8+4 batches.
    groups = [[row for row in questions if row["dataset"] == "synthetic"]]
    groups += [[row for row in questions if row["dataset"] == "radar_seen" and row["language"] == lang]
               for lang in old.LANGUAGES]
    for group in groups:
        for offset in range(0, len(group), size):
            yield group[offset:offset + size]


def kb_path(knowledge_id):
    if knowledge_id == "radar_development_v2":
        return old.KB_PATH
    if not knowledge_id.startswith("ifkb-") or not all(c in "0123456789abcdef" for c in knowledge_id[5:]):
        raise ValueError("Invalid neutral knowledge routing key")
    return training.DATA / "kb" / f"{knowledge_id}.json"


def freeze():
    admitted = json.loads(training.ADMISSION.read_text())
    audit = json.loads(AUDIT.read_text())
    require(admitted["status"] == "cpu_validated" and audit.get("status") == "passed", "CPU and separate AI audit required")
    train_protocol = training.verify("P", weights=False)
    original = old.verify("P", weights=False)
    rows = question_rows("after")
    paths = [__file__, str(training.PROTOCOL), str(training.ADMISSION), str(AUDIT),
             "experiments/radar_domain/development_analysis.py", "experiments/radar_domain/development_environment.py",
             "experiments/agent_feedback/environment.py", "experiments/agent_feedback/inference.py",
             "experiments/radar_domain/interface_training.py", "experiments/radar_domain/lookup_probe.py",
             "experiments/radar_domain/interface_analysis.py",
             "experiments/radar_domain/interface_pipeline.py",
             "results/radar_domain/lookup_probe_v1/summary.json", str(old.PROTOCOL),
             str(training.DATA / "holdout.questions.jsonl")]
    paths += [str(old.DATA / f"questions.{language}.jsonl") for language in old.LANGUAGES]
    paths += sorted({str(kb_path(row["knowledge_id"])) for row in rows})
    paths = [str(Path(name).resolve().relative_to(Path.cwd().resolve())) for name in paths]
    references = [str(training.DATA / "holdout.references.jsonl"), str(old.DATA / "references.json")]
    # Freeze reference bytes here; generation's verifier explicitly excludes them.
    protocol = {"version": "radar_interface_evaluation_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Synthetic instance/wording holdout and exposed radar development diagnostic; no independent domain efficacy claim",
                "inputs_sha256": {name: sha(name) for name in paths},
                "reference_sha256": {name: sha(name) for name in references},
                "model": original["model"], "models": original["models"], "inference": original["inference"],
                "training_protocol_sha256": sha(training.PROTOCOL), "tokenizer_sha256": original["tokenizer_sha256"],
                "run_rows": {"before": 40, "after": 64}, "runs": 10, "total_new_rows": 520,
                "maximum_hours_per_run": 0.35, "maximum_additional_gpu_hours": 3.5,
                "paired_units": "20 synthetic groups, each zh/en; 12 exposed radar targets each zh/en; languages are not independent samples",
                "unchanged": "Runtime, prompts, parser, greedy generation and budgets unchanged; only adapter weights change within label",
                "training_config": train_protocol["config"],
                "training_rationale": "120 bilingual training rows, five complete passes give 40 updates at effective batch 16. LR 5e-5 is half the original continuation LR. This is one small fixed interface-learning budget, not a tuned optimum.",
                "synthetic_operational_gate": {"agent_labels": ["A1", "C1", "A2", "C2"],
                    "each_agent_min_exact_of_40": 32, "each_language_min_exact_of_20": 15,
                    "interpretation": "Practical gate for reliable simple three-action interface use, not a significance test or radar efficacy threshold. Failure blocks automatic scale-up; no retuning on this holdout."},
                "analysis_order": "Complete and CPU replay all ten runs before loading semantic references; score every fixed label/language",
                "baseline_radar": "Reuse lookup_probe_v1 frozen results for the identical 24 questions; no rerun or score selection",
                "dev_split": "Reserved; no model evaluation or hyperparameter selection in this bounded experiment",
                "stop_rule": "One fixed adaptation and evaluation only. If synthetic execution remains poor, stop scale-up. If synthetic works but radar does not, investigate schema/source mismatch without another automatic training round.",
                "human_gold": False, "independent_radar_evaluation": False, "automatic_next_round": False}
    write_new(PROTOCOL, protocol)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL)}))


def selected_model(protocol, label, phase, weights=True):
    if phase == "before":
        model = protocol["models"][label]
    else:
        result_path = training.OUT / label / "run_result.json"
        run = json.loads(result_path.read_text())
        require(run["status"] == "completed" and run["label"] == label and
                run["protocol_sha256"] == protocol["training_protocol_sha256"], "Incomplete or unbound adapted model")
        adapter = str(training.OUT / label / "model")
        expected_old = protocol["models"][label]["files_sha256"][protocol["models"][label]["adapter"] + "/adapter_model.safetensors"]
        require(run["starting_adapter_sha256"] == expected_old, "Adaptation changed its starting model")
        model = {"adapter": adapter, "program": label == "P", "training_result_sha256": sha(result_path),
                 "files_sha256": {adapter + "/adapter_model.safetensors": run["final_adapter_sha256"],
                                  adapter + "/adapter_config.json": run["final_adapter_config_sha256"]}}
    if weights:
        training.hash_mapping(model["files_sha256"])
    return model


def verify(label, phase, weights=True):
    require(label in MODELS and phase in PHASES, "Unknown fixed run")
    protocol = json.loads(PROTOCOL.read_text())
    require(not set(protocol["inputs_sha256"]) & set(protocol["reference_sha256"]), "Reference included in inference inputs")
    training.hash_mapping(protocol["inputs_sha256"])
    if weights:
        training.hash_mapping({str(Path(protocol["model"]) / name): digest for name, digest in protocol["tokenizer_sha256"].items()})
    return protocol, selected_model(protocol, label, phase, weights)


def generate(label, phase):
    protocol, model_spec = verify(label, phase)
    config = protocol["inference"]
    args = argparse.Namespace(**config, program=label == "P")
    output = OUT / f"{label}_{phase}.jsonl"
    if output.exists() or output.with_suffix(".runtime.json").exists():
        raise FileExistsError("No overwrite or score-selected retry")
    questions = question_rows(phase)
    kbs = {row["knowledge_id"]: DevelopmentKB(kb_path(row["knowledge_id"])) for row in questions}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, model_spec["adapter"])
    model.eval()
    started = time.monotonic()
    totals = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for batch in inference_batches(questions, args.batch_size):
            states = [{"id": row["id"], "messages": prompt_messages(row["question"], kbs[row["knowledge_id"]], program=args.program),
                       "episode": DevelopmentEpisode(kbs[row["knowledge_id"]], max_calls=args.max_calls),
                       "knowledge_id": row["knowledge_id"], "dataset": row["dataset"], "language": row["language"],
                       "input_tokens": 0, "generated_tokens": 0, "stop_reason": None} for row in batch]
            for _ in range(1 if args.program else args.max_calls):
                active, prompts = [], []
                for state in states:
                    if state["stop_reason"]:
                        continue
                    kw = {} if args.program else {"tools": tool_schemas()}
                    ids = tokenizer.apply_chat_template(state["messages"], tokenize=True, return_dict=False,
                                                        add_generation_prompt=True, **kw)
                    if len(ids) + 1 > args.max_context:
                        state["stop_reason"] = "context_budget"
                    elif state["generated_tokens"] >= args.max_generated:
                        state["stop_reason"] = "generation_budget"
                    else:
                        active.append(state)
                        prompts.append(ids)
                if not active:
                    break
                for allowance, group in generation_groups(active, prompts, args).items():
                    inputs = tokenizer.pad({"input_ids": [p for _, p in group]}, padding=True,
                                           return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, max_new_tokens=allowance, do_sample=False,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 eos_token_id=tokenizer.eos_token_id, use_cache=True)
                    for (state, prompt), seq in zip(group, outputs):
                        tokens, content = response_tokens(seq[inputs["input_ids"].shape[1]:].tolist(),
                                                           tokenizer.eos_token_id, tokenizer.pad_token_id)
                        text = tokenizer.decode(content, skip_special_tokens=False)
                        state["input_tokens"] += len(prompt)
                        state["generated_tokens"] += len(tokens)
                        state["messages"].append({"role": "assistant", "content": text})
                        if args.program:
                            state["program"] = text
                            state["episode"] = execute_program(kbs[state["knowledge_id"]], text)
                            state["stop_reason"] = "program_completed" if state["episode"].prediction is not None else "program_error"
                        else:
                            observation = execute_response(state["episode"], text)
                            state["messages"].append({"role": "tool", "content": compact(observation)})
                            if state["episode"].done:
                                state["stop_reason"] = "finished" if state["episode"].prediction is not None else "call_budget"
            for state in states:
                episode = state.pop("episode")
                state.update(prediction=episode.prediction, selected_handle=episode.selected,
                             calls=episode.calls, events=episode.events, label=label, phase=phase,
                             stop_reason=state["stop_reason"] or "call_budget")
                state["invalid_calls"] = sum(not e["observation"].get("ok", False) for e in episode.events)
                state["total_tokens"] = state["input_tokens"] + state["generated_tokens"]
                state["rendered_evidence"] = render_records(episode.prediction) if episode.prediction is not None else None
                stream.write(json.dumps(state, ensure_ascii=False) + "\n")
                for key in ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens"):
                    totals[key] += state[key]
                totals["questions"] += 1
            stream.flush()
            print(json.dumps({"label": label, "phase": phase, "completed": totals["questions"], "seconds": time.monotonic()-started}), flush=True)
    # Recheck mutable code/data and adapter after inference; references remain unopened.
    verify(label, phase)
    report = {"label": label, "phase": phase, "status": "completed", "seconds": time.monotonic()-started,
              "totals": dict(totals), "protocol_sha256": sha(PROTOCOL), "output_sha256": sha(output),
              "scope": protocol["scope"], "config": config, "model": model_spec,
              "token_accounting": "Unpadded complete prefills summed over turns plus generated tokens incl EOS; not FLOPs. P calls are executed program steps including automatic finish, not model tool turns."}
    write_new(output.with_suffix(".runtime.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "generate", "verify"])
    parser.add_argument("--label", choices=MODELS, default="P")
    parser.add_argument("--phase", choices=PHASES, default="before")
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    elif args.command == "generate": generate(args.label, args.phase)
    else:
        verify(args.label, args.phase)
        print("Frozen evaluation verified")


if __name__ == "__main__":
    main()
