"""Question-only, batched real tool interaction; scoring is a separate command."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time

from experiments.condition_consistency.executor import KoPLExecutor, compare_answers
from .environment import Episode, MAX_CALLS, compact, prompt_messages, tool_schemas
from .training_data import FUNCTION_HELP, read_jsonl


def execute_response(episode: Episode, text: str):
    try:
        if not isinstance(text, str):
            raise ValueError("tool_response_must_be_text")
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
        if len(matches) != 1:
            raise ValueError("exactly_one_tool_call_required")
        call = json.loads(matches[0])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("tool_call_requires_name_and_arguments")
        name, arguments = call["name"], call["arguments"]
        expected = {"step": {"function", "inputs", "dependencies"}, "finish": {"answer_handle"}}
        if (not isinstance(name, str) or name not in expected or
                not isinstance(arguments, dict) or set(arguments) != expected[name]):
            raise ValueError("unknown_tool_or_invalid_arguments")
    except (ValueError, TypeError) as exc:
        try:
            episode._begin()
            observation = {"ok": False, "error": "ActionFormatError", "detail": str(exc)[:240]}
        except ValueError as limit:
            observation = {"ok": False, "error": "ValueError", "detail": str(limit)}
        episode.events.append({"tool": "invalid", "arguments": {}, "observation": observation})
    else:
        observation = episode.step(**arguments) if name == "step" else episode.finish(**arguments)
    # Exhaustion takes effect immediately, without an extra rejected call. A
    # successful finish on the last allowed call retains its answer.
    if episode.calls >= episode.max_calls:
        episode.done = True
    return observation


def validate_questions(questions):
    ids = set()
    for row in questions:
        if not isinstance(row, dict) or set(row) != {"id", "question"}:
            raise ValueError("Generation accepts only ID and question; gold/program fields forbidden")
        if type(row["id"]) not in (str, int) or not isinstance(row["question"], str):
            raise ValueError("Question IDs must be strings or integers and questions must be text")
        if row["id"] in ids:
            raise ValueError("Duplicate question IDs")
        ids.add(row["id"])


def generation_groups(active, prompts, args):
    """Do not let a nearly exhausted row truncate other rows' actions."""
    groups = defaultdict(list)
    for state, prompt in zip(active, prompts):
        allowance = min(args.max_generated if args.program else 192,
                        args.max_generated - state["generated_tokens"],
                        args.max_context - len(prompt))
        if allowance <= 0:
            raise ValueError("Exhausted row reached generation")
        groups[allowance].append((state, prompt))
    return groups


def response_tokens(tokens, eos_token_id, pad_token_id):
    """Return counted tokens and text tokens, retaining structured tool tags."""
    tokens = list(tokens)
    if eos_token_id is not None and eos_token_id in tokens:
        tokens = tokens[:tokens.index(eos_token_id) + 1]
        return tokens, tokens[:-1]
    while tokens and tokens[-1] == pad_token_id:
        tokens.pop()
    return tokens, tokens


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(args.seed)
    questions = read_jsonl(args.questions)
    validate_questions(questions)
    if (args.offset < 0 or args.limit < 0 or args.batch_size < 1 or
            args.max_context < 1 or args.max_generated < 1):
        raise ValueError("Invalid question slice, batch size or token budget")
    source_questions = len(questions)
    questions = questions[args.offset:args.offset + args.limit] if args.limit else questions[args.offset:]
    if not questions:
        raise ValueError("Selected question subset is empty")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Do not overwrite a prediction run")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                 torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    executor = KoPLExecutor(args.kb)
    start = time.monotonic()
    total = Counter()
    with output.open("w") as f:
        for offset in range(0, len(questions), args.batch_size):
            batch = questions[offset:offset + args.batch_size]
            states = [{"id": r["id"], "messages": prompt_messages(r["question"]),
                       "episode": Episode(executor), "input_tokens": 0, "generated_tokens": 0,
                       "stop_reason": None} for r in batch]
            if args.program:
                for state, row in zip(states, batch):
                    state["messages"] = [{"role": "system", "content":
                        "Translate the question into a complete executable KoPL program. "
                        "Output only function and literal inputs separated by <arg>, with steps separated by <func>. "
                        "Dependencies follow the standard branch-stack order.\n" + FUNCTION_HELP},
                        {"role": "user", "content": row["question"]}]
            for turn in range(1 if args.program else MAX_CALLS):
                active, prompts = [], []
                for state in states:
                    if state["stop_reason"]:
                        continue
                    kwargs = {} if args.program else {"tools": tool_schemas()}
                    ids = tokenizer.apply_chat_template(state["messages"], tokenize=True, return_dict=False,
                                                        add_generation_prompt=True, **kwargs)
                    if len(ids) + 1 > args.max_context:
                        state["stop_reason"] = "context_budget"
                    elif state["generated_tokens"] >= args.max_generated:
                        state["stop_reason"] = "generation_budget"
                    else:
                        active.append(state)
                        prompts.append(ids)
                if not active:
                    break
                for max_new, group in generation_groups(active, prompts, args).items():
                    inputs = tokenizer.pad({"input_ids": [p for _, p in group]},
                                           padding=True, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                   use_cache=True)
                    for (state, prompt), seq in zip(group, outputs):
                        tokens, content_tokens = response_tokens(
                            seq[inputs["input_ids"].shape[1]:].tolist(),
                            tokenizer.eos_token_id, tokenizer.pad_token_id)
                        text = tokenizer.decode(content_tokens, skip_special_tokens=False)
                        state["input_tokens"] += len(prompt)
                        state["generated_tokens"] += len(tokens)
                        state["messages"].append({"role": "assistant", "content": text})
                        if args.program:
                            state["program_result"] = executor.execute(text)
                            state["stop_reason"] = "program_completed" if state["program_result"]["valid"] else "program_error"
                        else:
                            observation = execute_response(state["episode"], text)
                            state["messages"].append({"role": "tool", "content": compact(observation)})
                            if state["episode"].done:
                                state["stop_reason"] = "finished" if state["episode"].prediction is not None else "call_budget"
            for state in states:
                episode = state.pop("episode")
                state["stop_reason"] = state["stop_reason"] or "call_budget"
                state["prediction"] = state.get("program_result", {}).get("prediction") if args.program else episode.prediction
                state["calls"] = int("program_result" in state) if args.program else episode.calls
                state["events"] = episode.events
                state["selected_handle"] = episode.selected
                state["invalid_calls"] = (int(not state["program_result"]["valid"])
                                          if args.program and "program_result" in state else
                                          sum(not e["observation"]["ok"] for e in episode.events))
                state["total_tokens"] = state["input_tokens"] + state["generated_tokens"]
                f.write(json.dumps(state, ensure_ascii=False) + "\n")
                for key in ("calls", "input_tokens", "generated_tokens", "total_tokens", "invalid_calls"):
                    total[key] += state[key]
                total["questions"] += 1
            f.flush()
            print(json.dumps({"completed": total["questions"], "seconds": time.monotonic() - start}), flush=True)
    report = {"status": "completed", "seconds": time.monotonic() - start, "totals": dict(total),
              "source_questions": source_questions, "selected_questions": len(questions),
              "token_accounting": "input_tokens sums unpadded full prefills on every turn; generated_tokens includes EOS, excludes batch padding; total_tokens is their sum. Inference recomputes prior history per turn; this is not GPU FLOPs or allocated padding cost.",
              "config": vars(args)}
    output.with_suffix(".runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def score(args):
    predictions, gold = read_jsonl(args.predictions), read_jsonl(args.gold)
    if not predictions or not gold:
        raise ValueError("Predictions and gold must be nonempty")
    by_id = {r["id"]: r for r in gold}
    if len(by_id) != len(gold):
        raise ValueError("Duplicate gold IDs")
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate predictions")
    counts = Counter()
    rows = []
    for row in predictions:
        if row["id"] not in by_id:
            raise ValueError(f"Prediction ID absent from gold: {row['id']}")
        reference = by_id[row["id"]]
        correct = compare_answers(reference["answer"], row["prediction"])
        counts["correct"] += correct
        counts["total"] += 1
        counts["failed_to_finish"] += row["prediction"] is None
        rows.append({"id": row["id"], "correct": correct, "prediction": row["prediction"],
                     "stop_reason": row.get("stop_reason"), "invalid_calls": row["invalid_calls"]})
    report = {**counts, "accuracy": counts["correct"] / counts["total"],
              "evaluation_scope": "complete" if len(predictions) == len(gold) else "subset",
              "gold_questions": len(gold), "prediction_questions": len(predictions),
              "missing_predictions": len(gold) - len(predictions),
              "coverage": len(predictions) / len(gold),
              "accuracy_denominator": "prediction_questions; unpredicted gold questions are not evaluated",
              "stop_reasons": dict(Counter(r.get("stop_reason", "unknown") for r in predictions)),
              "means": {k: sum(r[k] for r in predictions) / len(predictions)
                        for k in ["calls", "input_tokens", "generated_tokens", "invalid_calls"]}}
    report["means"]["total_tokens"] = report["means"]["input_tokens"] + report["means"]["generated_tokens"]
    path = Path(args.predictions)
    path.with_suffix(".scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kb", default="datasets/kqa_pro/kb.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-generated", type=int, default=2048)
    p.add_argument("--program", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--predictions", required=True)
    p.add_argument("--gold", required=True)
    args = parser.parse_args()
    generate(args) if args.command == "generate" else score(args)


if __name__ == "__main__":
    main()
