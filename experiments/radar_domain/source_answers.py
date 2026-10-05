"""Common base-model answer generation from frozen source evidence routes.

All arms expose the same raw-chunk schema to one unadapted answer generator.
The KG selectors do not contribute their extracted values or rendered notes to
this prompt. References and expected answers are never opened here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from experiments.agent_feedback.inference import response_tokens
from .development_analysis import read_jsonl, require, replay_row, TOTAL_KEYS
from .development_environment import DevelopmentKB
from . import source_probe as probe
from . import source_rag as rag


def verify_inputs():
    protocol = probe.verify("P", weights=False)
    from .source_preflight import verify_live_base
    verify_live_base(protocol)
    probe.previous_training.hash_mapping({str(Path(protocol["model"]) / name): digest
                                          for name, digest in protocol["tokenizer_sha256"].items()})
    return protocol


def evidence_rows(arm, protocol):
    require(arm in probe.ARMS, "Unknown frozen evidence arm")
    questions = probe.questions()
    require(len(questions) == protocol["question_count"], "Question count changed")
    index = rag.BM25Index.from_path(probe.DATA / "chunks.jsonl")
    hashes = {}
    if arm == "RAG":
        return [{"id": q["id"], "question": q["question"],
                 "evidence": rag.evidence_for_retrieval(index.retrieve(q["question"]))} for q in questions], hashes
    path = probe.OUT / f"{arm}.selection.jsonl"
    runtime_path = path.with_suffix(".runtime.json")
    runtime = json.loads(runtime_path.read_text())
    require(runtime["status"] == "completed" and runtime["label"] == arm, "Selector not complete")
    require(runtime["protocol_sha256"] == probe.sha(probe.PROTOCOL) and runtime["output_sha256"] == probe.sha(path),
            "Selector output binding failed")
    require(runtime["model"] == protocol["models"][arm] and runtime["config"] == protocol["inference"], "Selector model/config drift")
    selected = read_jsonl(path)
    require([row["id"] for row in selected] == [q["id"] for q in questions], "Selector question coverage changed")
    kb = DevelopmentKB(probe.DATA / "kb.json")
    rows = []
    totals = Counter(questions=len(selected))
    for row, question in zip(selected, questions):
        require(row.get("label") == arm, "Wrong selector row label")
        replay_row(kb, row, question, protocol, arm)
        totals.update({key: row[key] for key in TOTAL_KEYS})
        rows.append({"id": question["id"], "question": question["question"],
                     "evidence": rag.evidence_for_records(row["prediction"] or [], index)})
    require(dict(totals) == runtime["totals"], "Selector runtime totals mismatch")
    hashes = {str(path): probe.sha(path), str(runtime_path): probe.sha(runtime_path)}
    return rows, hashes


def generate(arm):
    protocol = verify_inputs()
    rows, evidence_hashes = evidence_rows(arm, protocol)
    output = probe.OUT / f"{arm}.answers.jsonl"
    if output.exists() or output.with_suffix(".runtime.json").exists():
        raise FileExistsError("Never overwrite answers or choose a retry by score")
    config = protocol["answer_inference"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Tokenize all questions before loading model weights; do not truncate evidence.
    for row in rows:
        row["messages"] = rag.answer_messages(row["question"], row["evidence"])
        row["prompt_ids"] = tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False,
                                                         add_generation_prompt=True)
        require(len(row["prompt_ids"]) + config["max_generated"] <= config["max_context"],
                "Common answer prompt exceeds its fixed context budget; no silent truncation")
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.eval()
    totals = Counter()
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as stream:
        for offset in range(0, len(rows), config["batch_size"]):
            batch = rows[offset:offset + config["batch_size"]]
            inputs = tokenizer.pad({"input_ids": [row["prompt_ids"] for row in batch]}, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=config["max_generated"], do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                         use_cache=True)
            for row, generated in zip(batch, outputs):
                tokens, content = response_tokens(generated[inputs["input_ids"].shape[1]:].tolist(),
                                                  tokenizer.eos_token_id, tokenizer.pad_token_id)
                text = tokenizer.decode(content, skip_special_tokens=False)
                parsed, parse_error = None, None
                try:
                    parsed = rag.parse_answer(text, row["evidence"])
                except (ValueError, TypeError, KeyError) as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"
                result = {"id": row["id"], "arm": arm, "question": row["question"], "evidence": row["evidence"],
                          "messages": row["messages"], "generated_text": text, "parsed": parsed,
                          "parse_error": parse_error, "input_tokens": len(row["prompt_ids"]),
                          "generated_tokens": len(tokens), "total_tokens": len(row["prompt_ids"]) + len(tokens),
                          "stop_reason": "eos" if tokenizer.eos_token_id in tokens else "generation_budget"}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                totals["questions"] += 1
                totals["parsed"] += int(parsed is not None)
                for key in ("input_tokens", "generated_tokens", "total_tokens"):
                    totals[key] += result[key]
            stream.flush()
            print(json.dumps({"arm": arm, "completed": totals["questions"], "seconds": time.monotonic() - started}), flush=True)
    verify_inputs()
    probe.previous_training.hash_mapping(evidence_hashes)
    runtime = {"status": "completed", "arm": arm, "protocol_sha256": probe.sha(probe.PROTOCOL),
               "output_sha256": probe.sha(output), "evidence_inputs_sha256": evidence_hashes,
               "config": config, "model": protocol["model"], "adapter": None,
               "seconds": time.monotonic() - started, "totals": dict(totals),
               "claim_limit": "Parsed claims are not automatically supported claims; no semantic reference was accessed."}
    probe.write_new(output.with_suffix(".runtime.json"), runtime)
    print(json.dumps(runtime), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=probe.ARMS, required=True)
    args = parser.parse_args()
    generate(args.arm)


if __name__ == "__main__":
    main()
