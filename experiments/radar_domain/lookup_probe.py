"""One fixed bilingual lookup diagnostic using the completed study's adapters.

Only task wording and question language change from development_probe_v1. The
prompts, KB, tools, parser, runtime semantics and generation limits are unchanged.
The two languages are paired views of twelve exposed targets, not independent
samples. Inference never opens reference programs or answer records.
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

DATA = Path("artifacts/thesis_direction_review/radar_lookup_calibration_v1")
KB_PATH = Path("artifacts/thesis_direction_review/radar_development_v2/readings.json")
OUT = Path("results/radar_domain/lookup_probe_v1")
PROTOCOL = OUT / "protocol.json"
LANGUAGES = ("zh", "en")
MODELS = {
    "P": "program_sft_v1", "A1": "clean_sft_v1", "C1": "recovery_sft_v1",
    "A2": "clean_sft_seed20261004", "C2": "recovery_sft_seed20261004",
}
CODE = [
    "experiments/radar_domain/lookup_probe.py",
    "experiments/radar_domain/task_calibration.py",
    "experiments/radar_domain/lookup_analysis.py",
    "experiments/radar_domain/development_analysis.py",
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
    # Preparation can inspect references. Generation below cannot and does not.
    from . import development_probe as previous
    from .task_calibration import build
    for name, data in build().items():
        if (DATA / name).read_bytes() != data:
            raise ValueError(f"Calibration artifact drift: {name}")
    original = previous.verify("P", weights=False)
    if set(original["models"]) != set(MODELS):
        raise ValueError("The fixed five-model roster changed")
    questions = {language: str(DATA / f"questions.{language}.jsonl") for language in LANGUAGES}
    question_rows = {}
    for language, name in questions.items():
        rows = [json.loads(line) for line in Path(name).read_text().splitlines()]
        validate_questions(rows)
        if len(rows) != original["question_count"]:
            raise ValueError("Unexpected calibration question count")
        question_rows[language] = rows
    if [row["id"] for row in question_rows["zh"]] != [row["id"] for row in question_rows["en"]]:
        raise ValueError("The two languages must have the same ordered target IDs")
    # Keep the entire original input binding, including the unchanged runtime.
    inputs = dict(original["inputs_sha256"])
    files = CODE + [str(previous.PROTOCOL), str(DATA / "manifest.json")] + list(questions.values())
    inputs.update({name: sha(name) for name in files})
    obj = {
        "version": "radar_lookup_probe_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One fixed bilingual lookup diagnostic on 12 exposed AI-rewritten development targets; interface compatibility only",
        "training": False, "independent_evaluation": False, "human_gold": False,
        "automatic_next_round": False, "models": original["models"],
        "inputs_sha256": inputs,
        "reference_sha256": {str(DATA / "references.json"): sha(DATA / "references.json")},
        "model": original["model"], "tokenizer_sha256": original["tokenizer_sha256"],
        "base_weights": original["base_weights"],
        "base_verification_policy": original["base_verification_policy"],
        "inference": original["inference"], "question_count": original["question_count"],
        "questions": questions, "languages": list(LANGUAGES), "kb": str(KB_PATH),
        "run_count": len(MODELS) * len(LANGUAGES), "total_rows": 120,
        "maximum_additional_gpu_hours": 2.5, "maximum_hours_per_run": 0.25,
        "schedule": "Ten fixed runs on idle GPUs under syx; one GPU per run and run_job supervisor capped at 0.25 hours. No score-selected retry.",
        "metrics": original["metrics"],
        "analysis_order": "Validate all ten complete runtime manifests and all 120 CPU replays before opening reference records or computing target-selection scores.",
        "paired_comparison": "For each fixed model compare zh/en on the same 12 IDs using exact-selection and finished counts plus paired concordant/discordant outcomes; descriptive only, no independence assumption across language views or checkpoints.",
        "original_probe_comparison": "Post-hoc task-rewrite diagnostic only: wording and task demands changed. Do not attribute differences from development_probe_v1 solely to language.",
        "language_scope": "Only the question and its source-context wrapper are translated; the shared bilingual system prompt, schema and Chinese KB notes are unchanged. The en condition is not an entirely English interface.",
        "error_policy": "Report the same observable format/function, entity/source, attribute/event, condition, finish/budget issues; any language association is descriptive on twelve exposed AI-translated pairs.",
        "answer_policy": original["answer_policy"],
        "stop_rule": "Complete the single fixed diagnostic and CPU replay, report errors, then make an explicit next-work decision. No training, prompt tuning, or automatic next round.",
        "next_work_policy": [
            "This exhausts the one planned zero-shot alignment retest; no further prompt or wording search by score.",
            "If signature/handle loops persist, stop direct transfer and specify matched-budget interface-adaptation data before any new training.",
            "If execution is usable but fields/conditions are wrong, specify field/condition supervision rather than silently relaxing the parser.",
            "If lookup selection is usable, expand source-separated domain development and baseline coverage before claiming domain efficacy; these twelve exposed pairs cannot establish it.",
            "Report all five checkpoints in both languages. Do not select a favorable seed or merge language variants into independent observations.",
        ],
    }
    write_new(PROTOCOL, obj)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL)}))


def verify(label, *, weights=True):
    if label not in MODELS:
        raise ValueError("Unknown fixed model label")
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


def load_questions(protocol, language):
    """Read only the selected question view; references are never read here."""
    if language not in LANGUAGES:
        raise ValueError("Unknown fixed question language")
    questions = [json.loads(line) for line in Path(protocol["questions"][language]).read_text().splitlines()]
    validate_questions(questions)
    if len(questions) != protocol["question_count"]:
        raise ValueError("Unexpected question count")
    return questions


def generate(label, language):
    if language not in LANGUAGES:
        raise ValueError("Unknown fixed question language")
    protocol = verify(label)
    config = protocol["inference"]
    args = argparse.Namespace(**config, program=protocol["models"][label]["program"])
    output = OUT / f"{label}_{language}.jsonl"
    if output.exists() or output.with_suffix(".runtime.json").exists():
        raise FileExistsError("Do not overwrite or select a retry by score")
    questions = load_questions(protocol, language)
    kb = DevelopmentKB(KB_PATH)
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
                             calls=episode.calls, events=episode.events, label=label, language=language,
                             stop_reason=state["stop_reason"] or "call_budget")
                state["invalid_calls"] = sum(not e["observation"].get("ok", False) for e in episode.events)
                state["total_tokens"] = state["input_tokens"] + state["generated_tokens"]
                state["rendered_evidence"] = render_records(episode.prediction) if episode.prediction is not None else None
                stream.write(json.dumps(state, ensure_ascii=False) + "\n")
                for key in ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens"):
                    totals[key] += state[key]
                totals["questions"] += 1
            stream.flush()
            print(json.dumps({"label": label, "language": language, "completed": totals["questions"], "seconds": time.monotonic()-started}), flush=True)
    # Recheck mutable code/data and adapter after inference; references remain unopened.
    verify(label)
    report = {"label": label, "language": language, "status": "completed", "seconds": time.monotonic()-started,
              "totals": dict(totals), "protocol_sha256": sha(PROTOCOL), "output_sha256": sha(output),
              "scope": protocol["scope"], "config": config, "model": protocol["models"][label],
              "token_accounting": "Unpadded complete prefills summed over turns plus generated tokens incl EOS; not FLOPs. P calls are executed program steps including automatic finish, not model tool turns."}
    write_new(output.with_suffix(".runtime.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "generate", "verify"])
    parser.add_argument("--label", choices=MODELS, default="P")
    parser.add_argument("--language", choices=LANGUAGES, default="zh")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "generate":
        generate(args.label, args.language)
    else:
        verify(args.label)
        print("Frozen inputs and checkpoint verified")


if __name__ == "__main__":
    main()
