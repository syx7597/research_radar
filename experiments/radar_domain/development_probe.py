"""Fixed, question-only inference on the exposed radar development packet.

This is an interface transfer probe, not a domain benchmark or training run.
Reference answers are not opened here. The model selects records; a deterministic
renderer displays source-scoped, AI-reviewed evidence, not model-authored prose.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from experiments.agent_feedback.environment import compact, tool_schemas
from experiments.agent_feedback.inference import (
    execute_response, generation_groups, response_tokens, validate_questions,
)
from .development_environment import (
    DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages, render_records,
)

DATA = Path("artifacts/thesis_direction_review/radar_development_v2")
OUT = Path("results/radar_domain/development_probe_v1")
PROTOCOL = OUT / "protocol.json"
MODELS = {
    "P": "program_sft_v1", "A1": "clean_sft_v1", "C1": "recovery_sft_v1",
    "A2": "clean_sft_seed20261004", "C2": "recovery_sft_seed20261004",
}
CODE = [
    "experiments/radar_domain/development_data.py",
    "experiments/radar_domain/development_environment.py",
    "experiments/radar_domain/development_probe.py",
    "experiments/agent_feedback/environment.py",
    "experiments/agent_feedback/inference.py",
    "experiments/agent_feedback/run_job.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze():
    # Reuse precisely the five adapters admitted by the completed public study.
    from .development_data import build
    for name, data in build().items():
        if (DATA / name).read_bytes() != data:
            raise ValueError(f"Development artifact drift: {name}")
    old = json.loads(Path("results/agent_feedback/protocol_holdout_v1.json").read_text())
    hashes = old["inputs_sha256"]
    models = {}
    for label, name in MODELS.items():
        adapter = f"results/agent_feedback/{name}/model"
        models[label] = {"adapter": adapter, "program": label == "P",
                         "files_sha256": {f"{adapter}/{f}": hashes[f"{adapter}/{f}"]
                                          for f in ("adapter_model.safetensors", "adapter_config.json")}}
    files = CODE + [str(DATA / f) for f in ("readings.json", "questions.jsonl", "manifest.json")]
    files.append("results/agent_feedback/weights_verified.json")
    obj = {
        "version": "radar_development_probe_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "12 exposed AI-reviewed development questions; interface compatibility only",
        "training": False, "independent_evaluation": False, "human_gold": False,
        "automatic_next_round": False, "models": models,
        "inputs_sha256": {f: sha(f) for f in files},
        "reference_sha256": {str(DATA / "references.json"): sha(DATA / "references.json")},
        "model": "models/qwen2.5-3b-instruct",
        "tokenizer_sha256": old["manifest"]["tokenizer"]["files_sha256"],
        "base_weights": json.loads(Path("results/agent_feedback/weights_verified.json").read_text()),
        "base_verification_policy": "Rehash both pinned base shards once on the syx host before any launch; each inference also checks frozen inputs, tokenizer and selected adapter before/after generation.",
        "inference": {"seed": 20261003, "batch_size": 8, "max_context": 8192,
                      "max_generated": 2048, "max_calls": 24, "agent_turn_tokens": 192,
                      "decoding": "greedy", "dtype": "bfloat16", "attention": "sdpa"},
        "question_count": 12, "maximum_additional_gpu_hours": 1.25,
        "schedule": "four idle GPUs for A1/C1/A2/C2, then P; each supervisor capped at 0.25 hours",
        "metrics": ["finished", "exact_target_selection", "target_covered", "target_with_allowed_context",
                    "unrelated_extra_selection", "empty_selection", "invalid_calls", "tokens"],
        "error_policy": "Report observable format/function, entity/source, attribute/event, condition, finish/budget issues; do not infer Chinese-language causality without a control.",
        "answer_policy": "Models select records only. Source notes are AI-reviewed KB content; deterministic evidence cards do not establish model semantic judgment or factual truth.",
        "stop_rule": "Complete all fixed runs and replay; report errors, then decide next work. No training, prompt tuning, or retry by score.",
    }
    write_new(PROTOCOL, obj)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL)}))


def verify(label, *, weights=True):
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["inputs_sha256"].items():
        if sha(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if weights:
        bound = dict(protocol["models"][label]["files_sha256"])
        bound.update({str(Path(protocol["model"]) / f): v for f, v in protocol["tokenizer_sha256"].items()})
        for name, expected in bound.items():
            if sha(name) != expected:
                raise ValueError(f"Frozen checkpoint/tokenizer changed: {name}")
    return protocol


def generate(label):
    protocol = verify(label)
    config = protocol["inference"]
    args = argparse.Namespace(**config, program=protocol["models"][label]["program"])
    output = OUT / f"{label}.jsonl"
    if output.exists():
        raise FileExistsError("Do not overwrite or select a retry by score")
    questions = [json.loads(line) for line in (DATA / "questions.jsonl").read_text().splitlines()]
    validate_questions(questions)
    if len(questions) != protocol["question_count"]:
        raise ValueError("Unexpected question count")
    kb = DevelopmentKB(DATA / "readings.json")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(model, protocol["models"][label]["adapter"])
    model.eval()
    started = time.monotonic()
    totals = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for offset in range(0, len(questions), args.batch_size):
            batch = questions[offset:offset + args.batch_size]
            states = [{"id": row["id"], "messages": prompt_messages(row["question"], kb, program=args.program),
                       "episode": DevelopmentEpisode(kb, max_calls=args.max_calls),
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
                            state["episode"] = execute_program(kb, text)
                            state["stop_reason"] = "program_completed" if state["episode"].prediction is not None else "program_error"
                        else:
                            observation = execute_response(state["episode"], text)
                            state["messages"].append({"role": "tool", "content": compact(observation)})
                            if state["episode"].done:
                                state["stop_reason"] = "finished" if state["episode"].prediction is not None else "call_budget"
            for state in states:
                episode = state.pop("episode")
                state.update(prediction=episode.prediction, selected_handle=episode.selected,
                             calls=episode.calls, events=episode.events, label=label,
                             stop_reason=state["stop_reason"] or "call_budget")
                state["invalid_calls"] = sum(not e["observation"].get("ok", False) for e in episode.events)
                state["total_tokens"] = state["input_tokens"] + state["generated_tokens"]
                state["rendered_evidence"] = render_records(episode.prediction) if episode.prediction is not None else None
                stream.write(json.dumps(state, ensure_ascii=False) + "\n")
                for key in ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens"):
                    totals[key] += state[key]
                totals["questions"] += 1
            stream.flush()
            print(json.dumps({"label": label, "completed": totals["questions"], "seconds": time.monotonic()-started}), flush=True)
    # Recheck mutable code/data and adapter after inference; references remain unopened.
    verify(label)
    report = {"label": label, "status": "completed", "seconds": time.monotonic()-started,
              "totals": dict(totals), "protocol_sha256": sha(PROTOCOL), "output_sha256": sha(output),
              "scope": protocol["scope"], "config": config, "model": protocol["models"][label],
              "token_accounting": "Unpadded complete prefills summed over turns plus generated tokens incl EOS; not FLOPs. P calls are executed program steps including automatic finish, not model tool turns."}
    write_new(output.with_suffix(".runtime.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "generate", "verify"])
    parser.add_argument("--label", choices=MODELS, default="P")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "generate":
        generate(args.label)
    else:
        verify(args.label)
        print("Frozen inputs and checkpoint verified")


if __name__ == "__main__":
    main()
