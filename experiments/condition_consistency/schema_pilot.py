"""Bounded, question-only three-arm schema baseline pilot; no training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

from .baseline import (read_records, sha256, write_json, program_generation_config,
                       token_scores, PROGRAM_GENERATION_POLICY)
from .schema_constraints import SchemaFieldConstraint, FIELD_ROLES

ROOT = Path(__file__).resolve().parents[2]
RUN = Path("results/condition_consistency/schema_pilot")
DATA = Path("data/condition_consistency/schema_pilot")
GOLD = Path("data/condition_consistency/splits/dev.jsonl")
SCHEMA = Path("results/condition_consistency/query_repair_audit/schema.json")
MODEL = Path("results/condition_consistency/pilot_bart5k_seed20260926/generator/final")
CHECKPOINT = MODEL.parent.parent / "checkpoint_manifest.json"
ARMS = {"plain4": (4, False), "constrained4": (4, True), "plain8": (8, False)}
CODE = ["experiments/condition_consistency/" + name for name in (
    "schema_pilot.py", "schema_constraints.py", "baseline.py", "executor.py",
    "execute_predictions.py", "evaluate.py")]


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_for(phase, arm, kind="generated"):
    return RUN / phase / f"{arm}_{kind}.jsonl"


def meta_for(path):
    return path.with_suffix(".jsonl.meta.json")


def freeze(_args):
    if (RUN / "protocol.json").exists():
        raise ValueError("Protocol already frozen")
    gold = read_records(GOLD)
    if len(gold) != 500 or len({row["id"] for row in gold}) != 500:
        raise ValueError("Expected the fixed 500-question development set")
    coverage = json.loads(Path("artifacts/thesis_direction_review/schema_constraint_coverage.json").read_text())
    if sha256(GOLD) != coverage["inputs"]["dev_gold"]["sha256"]:
        raise ValueError("Development data differs from previous coverage audit")
    inputs = {}
    for phase, count in (("smoke20", 20), ("dev500", 500)):
        path = DATA / f"{phase}.questions.jsonl"
        jsonl(path, [{"id": row["id"], "question": row["question"]} for row in gold[:count]])
        inputs[phase] = {"path": str(path), "sha256": sha256(path), "count": count,
                         "ids": [row["id"] for row in gold[:count]]}
    legacy = Path("results/condition_consistency/query_repair/dev_beam8_generated.jsonl.meta.json")
    old = json.loads(legacy.read_text())["protocol"]
    common = {key: old[key] for key in ("seed", "max_source_length", "max_new_tokens",
                                       "length_penalty", "precision")}
    common.update(batch_size=8, forced_eos_token_id=None)
    protocol = {
        "version": "schema_baseline_pilot_v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_status": "Previously inspected development set; no independent generalization claim",
        "inputs": inputs, "gold": {"path": str(GOLD), "sha256": sha256(GOLD)},
        "schema": {"path": str(SCHEMA), "sha256": sha256(SCHEMA)},
        "kb": {"path": "datasets/kqa_pro/kb.json",
               "sha256": json.loads(SCHEMA.read_text())["source_kb_sha256"]},
        "checkpoint_manifest": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)},
        "model": str(MODEL), "code_sha256": {name: sha256(name) for name in CODE},
        "reference_generation": {"path": str(legacy), "sha256": sha256(legacy)},
        "common_generation": common, "neutral_generation_policy": PROGRAM_GENERATION_POLICY,
        "arms": {arm: {"beams": k, "candidates": k, "field_constraint": enabled}
                 for arm, (k, enabled) in ARMS.items()},
        "selection": "First candidate in returned beam-score order that ended with EOS and passed frozen official execution; no valid candidate counts wrong",
        "schema_scope": "attribute/relation/qualifier only, global KB vocabulary; no gold at inference",
        "error_policy": "Abort on callback/trace failure; never drop a question or fall back to unconstrained generation",
        "execution": {"backend": "baseline", "hash_seed": "20260926", "timeout_seconds": 5},
        "budget": {"total_gpu_seconds_ceiling": 3600, "smoke_per_arm_wall_seconds": 180,
                   "dev_per_arm_wall_seconds": 900, "remaining_seconds_reserved_for_fault_diagnosis": 360,
                   "gpu_mapping_when_idle": {"plain4": 0, "constrained4": 1, "plain8": 2}},
        "metrics": ["first_valid_accuracy", "top1_accuracy", "oracle_at_k", "corrected/regressed",
                    "schema_violations", "syntax_failures", "no_valid_answer", "truncation", "wall_time"],
        "bootstrap": {"seed": 20260926, "samples": 2000, "role": "descriptive paired development uncertainty"},
        "gate": "All three smoke20 generations and executions must complete; zero constrained trace/field violations before dev500",
        "training": False, "official_validation_access": False,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(RUN / "protocol.json", protocol)
    print(json.dumps({"protocol": str(RUN / "protocol.json"), "sha256": sha256(RUN / "protocol.json")}, indent=2))


def verified_protocol():
    p = json.loads((RUN / "protocol.json").read_text())
    for name, expected in p["code_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen code differs: {name}")
    for item in (p["schema"], p["checkpoint_manifest"], *p["inputs"].values()):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input differs: {item['path']}")
    if os.environ.get("PYTHONHASHSEED") != p["execution"]["hash_seed"]:
        raise ValueError("Launch with PYTHONHASHSEED=20260926")
    return p


def field_issues(text, schema):
    """Lexical field check includes malformed/empty slots, never uses gold."""
    issues, fields = [], 0
    for step, chunk in enumerate(text.split("<func>")):
        parts = [value.strip() for value in chunk.split("<arg>")]
        for index, role in FIELD_ROLES.get(parts[0], {}).items():
            if index + 1 < len(parts):
                fields += 1
                if parts[index + 1] not in schema[role]:
                    issues.append(dict(step=step, argument=index, role=role, value=parts[index + 1]))
    return issues, fields


def check_trace(tokens, constraint):
    """Replay actual returned token IDs, including EOS; discard padding only after EOS."""
    for position in range(1, len(tokens)):
        state = constraint.inspect(tokens[:position])
        if state["allowed"] is not None and tokens[position] not in state["allowed"]:
            return {"position": position, "token": tokens[position], "state": state}
        if tokens[position] == constraint.eos:
            break
    return None


class PaddingSafeCallback:
    """HF's finished batch rows append PAD to an unfinished discarded beam.

    Permit only further PAD on these placeholder paths. Never open the field
    vocabulary, allow EOS or apply this exception to returned-token validation.
    An active path that generates PAD also enters this non-answering sink.
    """
    def __init__(self, constraint):
        self.constraint = constraint
        self.padding_calls = 0

    def __call__(self, batch_id, tokens):
        if int(tokens[-1]) == self.constraint.pad:
            self.padding_calls += 1
            return [self.constraint.pad]
        return self.constraint(batch_id, tokens)


def generate(args):
    protocol = verified_protocol()
    if args.phase == "dev500":
        gate = json.loads((RUN / "smoke20" / "gate.json").read_text())
        if not gate["passed"] or gate["protocol_sha256"] != sha256(RUN / "protocol.json"):
            raise ValueError("Smoke gate has not passed for this protocol")
    path = path_for(args.phase, args.arm)
    if path.exists() or meta_for(path).exists():
        raise ValueError("Never overwrite or resume a generation arm")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {"protocol_sha256": sha256(RUN / "protocol.json"), "arm": args.arm,
                "phase": args.phase, "status": "running", "completed_rows": 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_inputs": ["question"]}
    write_json(meta_for(path), metadata)
    limit = protocol["budget"]["smoke_per_arm_wall_seconds" if args.phase == "smoke20" else "dev_per_arm_wall_seconds"]

    def expired(_signal, _frame):
        raise TimeoutError(f"GPU arm wall-time budget {limit}s exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one available GPU to this arm")
        cfg = argparse.Namespace(**protocol["common_generation"], **protocol["arms"][args.arm])
        set_seed(cfg.seed)
        manifest = json.loads(CHECKPOINT.read_text())
        for name, expected in manifest["files"].items():
            if sha256(MODEL / name) != expected["sha256"]:
                raise ValueError(f"Checkpoint differs: {name}")
        rows = read_records(protocol["inputs"][args.phase]["path"])
        if any(set(row) != {"id", "question"} for row in rows):
            raise ValueError("Generation input must contain only id/question")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True,
                                                     torch_dtype=torch.bfloat16).to("cuda").eval()
        config = program_generation_config(transformers, model.generation_config, cfg)
        config.forced_eos_token_id = None
        schema = json.loads(SCHEMA.read_text())["names_by_role"]
        constraint = SchemaFieldConstraint(tokenizer, schema, config.decoder_start_token_id) if cfg.field_constraint else None
        kwargs = constraint.generation_kwargs(config) if constraint else {"generation_config": config}
        callback = PaddingSafeCallback(constraint) if constraint else None
        if callback:
            kwargs["prefix_allowed_tokens_fn"] = callback
        metadata.update(torch=torch.__version__, transformers=transformers.__version__,
                        gpu_name=torch.cuda.get_device_name(0), generation_config=config.to_dict())
        generation_seconds = validation_seconds = 0.0
        with path.open("x", encoding="utf-8") as stream, torch.inference_mode():
            for offset in range(0, len(rows), cfg.batch_size):
                batch = rows[offset:offset + cfg.batch_size]
                questions = [row["question"] for row in batch]
                if any(len(ids) > cfg.max_source_length for ids in tokenizer(questions)["input_ids"]):
                    raise ValueError("Source truncation prohibited")
                encoded = tokenizer(questions, padding=True, return_tensors="pt").to("cuda")
                torch.cuda.synchronize()
                tick = time.perf_counter()
                output = model.generate(**encoded, **kwargs)
                torch.cuda.synchronize()
                generation_seconds += time.perf_counter() - tick
                scores = model.compute_transition_scores(output.sequences, output.scores,
                                                          output.beam_indices, normalize_logits=True).cpu().tolist()
                token_rows = output.sequences.cpu().tolist()
                texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                beam_scores = output.sequences_scores.cpu().tolist()
                tick = time.perf_counter()
                for row_index, row in enumerate(batch):
                    candidates = []
                    for rank in range(cfg.candidates):
                        index = row_index * cfg.candidates + rank
                        tokens = token_rows[index]
                        total, count, ended = token_scores(tokens[1:1 + len(scores[index])], scores[index], tokenizer.eos_token_id)
                        text = texts[index].strip()
                        issues, fields = field_issues(text, schema)
                        if constraint and (check_trace(tokens, constraint) is not None or issues):
                            raise ValueError(f"Returned constrained candidate violates token/field constraint: {row['id']} rank {rank}")
                        if not math.isfinite(beam_scores[index]):
                            raise ValueError("Nonfinite returned beam score")
                        candidates.append(dict(rank=rank, program_text=text, sequence_logprob=total,
                            normalized_logprob=total / max(1, count), beam_score=beam_scores[index],
                            generated_tokens=count, ended_with_eos=ended, hit_generation_limit=count >= cfg.max_new_tokens,
                            token_ids=tokens, schema_issues=issues, field_occurrences=fields,
                            constraint_trace_checked=bool(constraint)))
                    stream.write(json.dumps(dict(id=row["id"], question=row["question"], candidates=candidates), ensure_ascii=False) + "\n")
                stream.flush()
                validation_seconds += time.perf_counter() - tick
                metadata.update(completed_rows=min(offset + cfg.batch_size, len(rows)),
                                generation_seconds=generation_seconds, returned_trace_validation_seconds=validation_seconds,
                                padding_sink_callback_calls=callback.padding_calls if callback else 0,
                                elapsed_seconds=time.perf_counter() - started,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                write_json(meta_for(path), metadata)
                print(f"{args.phase}/{args.arm} {metadata['completed_rows']}/{len(rows)} elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)
        metadata.update(status="complete", output_sha256=sha256(path))
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        metadata["elapsed_seconds"] = time.perf_counter() - started
        write_json(meta_for(path), metadata)


def execute(args):
    from .executor import KoPLExecutor, parse_program
    from .execute_predictions import execute_candidate
    protocol = verified_protocol()
    input_path, output_path = path_for(args.phase, args.arm), path_for(args.phase, args.arm, "executed")
    if output_path.exists():
        raise ValueError("Execution output exists")
    meta = json.loads(meta_for(input_path).read_text())
    if meta["status"] != "complete" or meta["output_sha256"] != sha256(input_path):
        raise ValueError("Generation did not complete or bytes changed")
    if sha256(protocol["kb"]["path"]) != protocol["kb"]["sha256"]:
        raise ValueError("KB changed")
    rows = read_records(input_path)
    if [row["id"] for row in rows] != protocol["inputs"][args.phase]["ids"]:
        raise ValueError("Incomplete/misaligned questions")
    started = time.perf_counter()
    executor = KoPLExecutor(Path(protocol["kb"]["path"]))
    with output_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            candidates = []
            for candidate in row["candidates"]:
                executed = execute_candidate(candidate, executor, protocol["execution"]["timeout_seconds"])
                try:
                    parse_program(candidate["program_text"])
                    executed["syntax_valid"] = True
                except ValueError:
                    executed["syntax_valid"] = False
                if not candidate["ended_with_eos"]:
                    executed.update(valid=False, error="GenerationTruncation: no EOS", prediction=None)
                candidates.append(executed)
            stream.write(json.dumps(dict(id=row["id"], question=row["question"], candidates=candidates), ensure_ascii=False) + "\n")
    write_json(meta_for(output_path), dict(status="complete", questions=len(rows), output_sha256=sha256(output_path),
        input_sha256=sha256(input_path), protocol_sha256=sha256(RUN / "protocol.json"),
        elapsed_seconds=time.perf_counter() - started, gold_used=False, python_hash_seed=os.environ["PYTHONHASHSEED"]))
    print(f"Executed {args.phase}/{args.arm}: {len(rows)} questions", flush=True)


def report(args):
    from .executor import compare_answers
    from .evaluate import paired_comparison
    protocol = verified_protocol()
    target = RUN / args.phase / "metrics.json"
    if target.exists():
        raise ValueError("Metrics exist")
    if sha256(GOLD) != protocol["gold"]["sha256"]:
        raise ValueError("Gold changed")
    expected = protocol["inputs"][args.phase]["ids"]
    gold = {row["id"]: row for row in read_records(GOLD)}
    summaries, correct, details, provenance = {}, {}, {}, {}
    gpu_seconds = 0.0
    for arm, (k, constrained) in ARMS.items():
        path = path_for(args.phase, arm, "executed")
        metadata = json.loads(meta_for(path).read_text())
        genmeta = json.loads(meta_for(path_for(args.phase, arm)).read_text())
        if (metadata["output_sha256"] != sha256(path) or metadata["protocol_sha256"] != sha256(RUN / "protocol.json")
                or metadata["input_sha256"] != sha256(path_for(args.phase, arm)) or genmeta["status"] != "complete"):
            raise ValueError("Execution/generation provenance mismatch")
        rows = read_records(path)
        if [row["id"] for row in rows] != expected:
            raise ValueError("All questions must remain in every arm")
        counts, arm_details, outcomes = Counter(), [], []
        for row in rows:
            if row["question"] != gold[row["id"]]["question"] or len(row["candidates"]) != k:
                raise ValueError("Question or candidate budget mismatch")
            candidates = row["candidates"]
            if constrained and any(c["schema_issues"] or not c["constraint_trace_checked"] for c in candidates):
                raise ValueError("Constrained final trace/field violation")
            pick = next((i for i, c in enumerate(candidates) if c["valid"]), None)
            oks = [bool(c["valid"] and compare_answers(gold[row["id"]]["answer"], c["prediction"])) for c in candidates]
            ok = pick is not None and oks[pick]
            outcomes.append(ok)
            counts.update(questions=1, correct=int(ok), top1_correct=int(oks[0]), oracle_correct=int(any(oks)),
                          no_valid_candidate=int(pick is None), selected_empty=int(pick is not None and candidates[pick]["empty_result"]))
            for c in candidates:
                counts.update(candidates=1, valid_candidates=int(c["valid"]), syntax_failures=int(not c["syntax_valid"]),
                    candidates_with_schema_violations=int(bool(c["schema_issues"])), schema_violation_fields=len(c["schema_issues"]),
                    field_occurrences=c["field_occurrences"], no_eos=int(not c["ended_with_eos"]),
                    hit_generation_limit=int(c["hit_generation_limit"]), empty_results=int(c["empty_result"]))
            arm_details.append(dict(id=row["id"], selected=pick, correct=ok, oracle_correct=any(oks)))
        correct[arm], details[arm] = outcomes, arm_details
        summaries[arm] = dict(counts, accuracy=counts["correct"] / len(rows),
                              generation_seconds=genmeta["generation_seconds"], gpu_process_wall_seconds=genmeta["elapsed_seconds"],
                              trace_validation_seconds=genmeta["returned_trace_validation_seconds"],
                              execution_seconds=metadata["elapsed_seconds"])
        gpu_seconds += genmeta["elapsed_seconds"]
        provenance[arm] = dict(executed_sha256=sha256(path), generation_meta=genmeta, execution_meta=metadata)
    comparisons = {f"{after}_vs_{before}": paired_comparison(correct[before], correct[after], seed=20260926, samples=2000)
                   for before, after in (("plain4", "constrained4"), ("plain8", "constrained4"), ("plain4", "plain8"))}
    result = dict(phase=args.phase, protocol_sha256=sha256(RUN / "protocol.json"), scope=protocol["dataset_status"],
                  summaries=summaries, comparisons=comparisons, per_question=details, provenance=provenance,
                  phase_gpu_process_seconds=gpu_seconds, limits=["Single checkpoint/seed on previously seen development questions",
                  "Constraint is an established baseline, not a novelty claim", "Wall times include concurrent host load; no isolated latency benchmark"])
    write_json(target, result)
    if args.phase == "smoke20":
        write_json(RUN / args.phase / "gate.json", dict(passed=True, protocol_sha256=sha256(RUN / "protocol.json"),
                   reason="All three generations/executions complete, fixed candidate counts, finite scores, zero constrained token/field violations",
                   phase_gpu_process_seconds=gpu_seconds))
    print(json.dumps(dict(summaries=summaries, comparisons=comparisons), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    for command in ("generate", "execute", "report"):
        item = sub.add_parser(command)
        item.add_argument("--phase", choices=("smoke20", "dev500"), required=True)
        if command != "report":
            item.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    {"freeze": freeze, "generate": generate, "execute": execute, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
