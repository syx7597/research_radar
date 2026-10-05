"""Frozen selection on new manufacturer-source snapshots; no additional training.

The packet is independently AI-reviewed source interpretation, not human gold.
References are never accessed by generation. The five adapted checkpoints and
existing tool runtime are reused without changed prompts or parser repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments.agent_feedback.environment import compact, tool_schemas
from experiments.agent_feedback.inference import execute_response, generation_groups, response_tokens, validate_questions
from .development_environment import DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages, render_records
from .development_analysis import read_jsonl, require
from . import interface_training as previous_training
from . import interface_probe as previous

DATA = Path("data/radar_sources_v1")
ARTIFACTS = Path("artifacts/thesis_direction_review/radar_sources_v1")
OUT = Path("results/radar_domain/source_eval_v1")
PROTOCOL = OUT / "protocol.json"
MODELS = previous.MODELS
ARMS = ["RAG", *MODELS]
sha, write_new = previous.sha, previous.write_new
CODE = ["experiments/radar_domain/source_probe.py", "experiments/radar_domain/source_rag.py",
        "experiments/radar_domain/source_answers.py", "experiments/radar_domain/source_analysis.py",
        "experiments/radar_domain/source_pipeline.py", "experiments/radar_domain/source_packet.py",
        "experiments/radar_domain/source_preflight.py", "experiments/radar_domain/interface_training.py",
        "experiments/radar_domain/interface_probe.py", "experiments/radar_domain/lookup_probe.py",
        "experiments/radar_domain/development_environment.py", "experiments/radar_domain/development_analysis.py",
        "experiments/agent_feedback/environment.py", "experiments/agent_feedback/inference.py",
        "experiments/agent_feedback/run_job.py"]


def questions():
    rows = read_jsonl(DATA / "questions.jsonl")
    require(rows and all(set(row) == {"id", "question"} for row in rows), "Only question text and neutral ID may enter inference")
    validate_questions(rows)
    return rows


def freeze():
    admission_path, audit_path = ARTIFACTS / "manifest.json", ARTIFACTS / "independent_ai_audit.json"
    admission, audit = [json.loads(path.read_text()) for path in (admission_path, audit_path)]
    require(admission["status"] == "cpu_validated" and audit["status"] == "passed", "Source data and separate AI review required")
    previous_training.hash_mapping(admission["files_sha256"])
    previous_training.hash_mapping(audit["files_sha256"])
    from .interface_analysis import verify_training
    verify_training()
    old = json.loads(previous.PROTOCOL.read_text())
    base = json.loads(previous_training.PROTOCOL.read_text())["base_weights"]
    preflight = json.loads((OUT / "preflight.json").read_text())
    require(preflight["status"] == "passed" and preflight["base_weights"] == base, "CPU/tokenizer/base preflight required")
    previous_training.hash_mapping(preflight["inputs_sha256"])
    rows = questions()
    models = {label: previous.selected_model(old, label, "after", weights=False) for label in MODELS}
    files = CODE + [str(admission_path), str(audit_path), str(DATA / "questions.jsonl"), str(DATA / "kb.json"),
                    str(DATA / "chunks.jsonl"), str(previous.PROTOCOL), str(previous_training.PROTOCOL), str(OUT / "preflight.json")]
    observed = ARTIFACTS / "pre_freeze_retrieval_observation.json"
    require(observed.exists(), "Preparation-side retrieval inspection must be disclosed")
    files.append(str(observed))
    files += [str(previous_training.OUT / label / "run_result.json") for label in MODELS]
    protocol = {"version": "radar_source_evaluation_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "New manufacturer-source snapshots and newly authored AI-reviewed questions; exploratory source-grounded evaluation, not independent human equipment gold",
                "inputs_sha256": {name: sha(name) for name in files},
                "reference_sha256": {str(DATA / "references.jsonl"): sha(DATA / "references.jsonl")},
                "model": old["model"], "models": models, "tokenizer_sha256": old["tokenizer_sha256"],
                "base_weights": base,
                "base_verification": "CPU preflight hashes both base shards; each GPU run checks the attested size/mtime/ctime before and after; controller rehashes both shards at completion",
                "inference": old["inference"], "question_count": len(rows), "arms": ARMS,
                "answer_model": "Same unadapted Qwen2.5-3B-Instruct base for all six evidence routes",
                "pre_freeze_retrieval_observation": {"artifact": str(observed),
                    "disclosure": "The data preparer inspected source-page retrieval coverage before freeze (23/24). No question, chunk or retrieval configuration was changed in response. This is an exposed preparation-side retrieval diagnostic; model-generated outputs remain unseen at freeze."},
                "answer_inference": {"seed": 20261003, "batch_size": 8, "max_context": 8192, "max_generated": 768,
                                     "do_sample": False, "dtype": "bfloat16", "attention": "sdpa"},
                "retrieval": {"method": "BM25", "k1": 1.2, "b": 0.75, "top_k": 4,
                              "query_expansion": "Frozen generic Chinese/English field ontology; no source/question-specific routing",
                              "tool_evidence": "Selected records map to their raw source chunks; deduplicate by chunk ID, sort, take at most four. No structured answers or reference notes enter the common generator."},
                "metrics": ["selector exact record sets for P/A/C", "source-chunk support coverage for every arm",
                            "answer JSON validity", "citation existence", "structured claim value/unit/condition agreement",
                            "separate blinded AI review of natural answer text; never equate claim validity with prose fidelity"],
                "primary_comparison": "Describe all six arms and both C-A continuation pairs on the same questions; no favorable seed or language selection",
                "scope_limits": ["Question creation and source review are AI-only; not human reference labels",
                                 "Source snapshots/QA are new relative to this experiment; some equipment may exist in historical unused corpus or base pretraining",
                                 "Source/family components are the correlation units; this small pilot does not support stable significance claims",
                                 "KG-assisted source retrieval and raw-text RAG differ in curated representation and training history; system comparison, not single-algorithm causal attribution",
                                 "Retrieving the wrong record may still expose a source chunk containing the answer; report selection and answer metrics separately"],
                "maximum_hours_per_selector": 0.35, "maximum_hours_per_answer_arm": 0.25,
                "maximum_additional_gpu_hours": 3.25, "training": False, "human_gold": False,
                "analysis_order": "Finish all five selector outputs and all six common-generator outputs, replay every selection and verify evidence inputs, then open semantic references",
                "stop_rule": "One fixed evaluation. No training, prompt/search retuning or selective retry after scoring; retain failures and decide next work explicitly."}
    write_new(PROTOCOL, protocol)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL), "questions": len(rows)}))


def verify(label, weights=True):
    require(label in MODELS, "Unknown adapted selector")
    protocol = json.loads(PROTOCOL.read_text())
    require(not set(protocol["inputs_sha256"]) & set(protocol["reference_sha256"]), "Reference leaked into inference verification")
    previous_training.hash_mapping(protocol["inputs_sha256"])
    if weights:
        from .source_preflight import verify_live_base
        verify_live_base(protocol)
        previous_training.hash_mapping(protocol["models"][label]["files_sha256"])
        previous_training.hash_mapping({str(Path(protocol["model"]) / name): digest for name, digest in protocol["tokenizer_sha256"].items()})
    return protocol


def generate(label):
    protocol = verify(label)
    model_spec = protocol["models"][label]
    config = protocol["inference"]
    args = argparse.Namespace(**config, program=label == "P")
    output = OUT / f"{label}.selection.jsonl"
    if output.exists() or output.with_suffix(".runtime.json").exists():
        raise FileExistsError("No overwrite or score-selected retry")
    question_rows = questions()
    require(len(question_rows) == protocol["question_count"], "Frozen question count mismatch")
    kb = DevelopmentKB(DATA / "kb.json")
    questions_for_run = question_rows
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
        for offset in range(0, len(questions_for_run), args.batch_size):
            batch = questions_for_run[offset:offset + args.batch_size]
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
              "scope": protocol["scope"], "config": config, "model": model_spec,
              "token_accounting": "Unpadded complete prefills summed over turns plus generated tokens incl EOS; not FLOPs. P calls are executed program steps including automatic finish, not model tool turns."}
    write_new(output.with_suffix(".runtime.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "generate", "verify"])
    parser.add_argument("--label", choices=MODELS, default="P")
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    elif args.command == "generate": generate(args.label)
    else:
        verify(args.label)
        print("Frozen new-source selection verified")


if __name__ == "__main__": main()
