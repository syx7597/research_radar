"""CPU preparation for a bounded OLD-development runtime probe, never execution.

Only the frozen v1 question/chunk files are accepted. The first old question is
used with its fixed BM25 evidence, then with answer-free synthetic padding near
8192 and 31000 prompt tokens. This measures prompt sizes, not model capability.
There is deliberately no GPU, SSH, model-loading, or formal-evaluation command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

from .coverage_preflight import SYSTEM_PROMPT, messages
from .source_readings import ROOT, sha, write_new
from . import source_rag


PROTOCOL = ROOT / "results/radar_domain/source_eval_v1/protocol.json"
PROTOCOL_SHA256 = "a3ded57e4626f6d5458f81946e22b53889524fabf8542993fb28390db16a0471"
QUESTIONS = ROOT / "data/radar_sources_v1/questions.jsonl"
CHUNKS = ROOT / "data/radar_sources_v1/chunks.jsonl"
OUTPUT_ROOT = ROOT / "data/radar_sources_v2/runtime_probe_v1"
CONFIG_SHA256 = "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b"
GENERATION_CONFIG_SHA256 = "ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9"
MAX_CONTEXT, MAX_GENERATED = 32768, 768
TARGETS, TARGET_TOLERANCE = (8192, 31000), 64
PADDING = " [runtime-padding]"


def old_inputs():
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Old development protocol identity changed")
    protocol = json.loads(PROTOCOL.read_text())
    bindings = {str(PROTOCOL.relative_to(ROOT)): PROTOCOL_SHA256}
    for path in (QUESTIONS, CHUNKS, Path(source_rag.__file__)):
        relative = str(path.relative_to(ROOT))
        if sha(path) != protocol["inputs_sha256"].get(relative):
            raise ValueError(f"Changed old-development input: {relative}")
        bindings[relative] = sha(path)
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if (len(questions) != protocol["question_count"]
            or any(set(q) != {"id", "question"} for q in questions)
            or len({q["id"] for q in questions}) != len(questions)):
        raise ValueError("Only the fixed old question IDs and text may enter the probe")
    chunks = [json.loads(line) for line in CHUNKS.read_text().splitlines() if line.strip()]
    source_rag.validate_chunks(chunks, raw=True)
    return protocol, questions, chunks, bindings


def token_ids(tokenizer, msg):
    ids = tokenizer.apply_chat_template(msg, tokenize=True, return_dict=False,
                                        add_generation_prompt=True, truncation=False)
    if type(ids) is not list or any(type(t) is not int for t in ids):
        raise ValueError("Tokenizer must return a complete single prompt token list")
    if len(ids) + MAX_GENERATED > MAX_CONTEXT:
        raise ValueError("Prompt exceeds context budget; truncation is forbidden")
    return ids


def probe_cases(tokenizer, question, evidence):
    """Build exactly three CPU cases; callers cannot supply answer references."""
    if set(question) != {"id", "question"} or set(evidence) != {"chunks"}:
        raise ValueError("Old question text and source chunks only")

    def make(repeats, target):
        supplied = dict(evidence)
        if repeats:
            supplied["runtime_padding"] = PADDING * repeats
        msg = messages(question["question"], supplied)
        ids = token_ids(tokenizer, msg)
        return {"case_id": "old_base" if target is None else f"old_padded_{target}",
                "old_question_id": question["id"], "target_prompt_tokens": target,
                "padding_repetitions": repeats, "input_tokens": len(ids),
                "messages": msg, "prompt_ids": ids, "truncated": False}

    baseline = make(0, None)
    if baseline["input_tokens"] >= TARGETS[0]:
        raise ValueError("Old baseline already exceeds the first padding target")
    cases = [baseline]
    for target in TARGETS:
        # Grow source-free padding only. Overshoot probes never truncate the old
        # evidence; if an exploratory size crosses the cap, search below it.
        low, high = 0, target
        while low + 1 < high:
            middle = (low + high) // 2
            try:
                candidate = make(middle, target)
            except ValueError as exc:
                if "exceeds context budget" not in str(exc):
                    raise
                high = middle
                continue
            if candidate["input_tokens"] < target:
                low = middle
            else:
                high = middle
        candidate = make(high, target)
        if not target <= candidate["input_tokens"] <= target + TARGET_TOLERANCE:
            raise ValueError("Synthetic padding did not reach the bounded target")
        cases.append(candidate)
    return cases


def verify_tokenizer(directory, protocol):
    directory = Path(directory).resolve()
    expected = protocol["tokenizer_sha256"]
    for name, digest in expected.items():
        if sha(directory / name) != digest:
            raise ValueError("Tokenizer differs from the frozen old revision")
    if sha(directory / "config.json") != CONFIG_SHA256:
        raise ValueError("Local model configuration differs from the pinned base")
    # CPU preparation uses a tokenizer-only directory. Auxiliary token files or
    # chat_template.jinja/chat_templates can override otherwise pinned inputs.
    allowed = set(expected) | {"config.json", "download_manifest.json"}
    extras = {p.name for p in directory.iterdir()} - allowed
    if extras:
        raise ValueError(f"Unpinned tokenizer auxiliary files: {sorted(extras)}")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


def build_plan(tokenizer, protocol, questions, chunks, bindings):
    question = questions[0]  # Fixed old row, never selected by score or runtime.
    index = source_rag.BM25Index(chunks)
    evidence = source_rag.evidence_for_retrieval(index.retrieve(question["question"]))
    cases = probe_cases(tokenizer, question, evidence)
    return {"schema": "radar_old_development_runtime_probe_plan_v1",
            "status": "cpu_prepared_gpu_environment_unverified", "cpu_only": True,
            "model_run_ready": False, "formal_evaluation_permitted": False,
            "new_model_runs": 0, "new_GPU_hours": 0,
            "inputs_sha256": bindings,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "cases": cases, "evidence_scope": "old_raw_chunks_with_optional_fact_free_padding",
            "model_identity": {"model": protocol["model"], "adapter": None,
                               "expected_base_weights": protocol["base_weights"],
                               "expected_tokenizer_sha256": protocol["tokenizer_sha256"],
                               "expected_config_sha256": CONFIG_SHA256,
                               "expected_generation_config_sha256": GENERATION_CONFIG_SHA256,
                               "local_tokenizer_verified": False,
                               "remote_weights_verified": False, "remote_runtime_verified": False},
            "future_execution_boundaries": {
                "required_os_user": "syx", "forbidden_os_users": ["admin", "root"],
                "gpu_count": 1, "batch_size": 1, "max_gpu_wall_seconds": 300,
                "max_gpu_hours": 300 / 3600, "charged_to_existing_gpu_hours_budget": 72,
                "max_context_tokens": MAX_CONTEXT, "max_generated_tokens": MAX_GENERATED,
                "decoding": "greedy", "dtype": "bfloat16", "attention": "sdpa",
                "truncate": False, "automatic_retries": 0, "formal_96_question_runs": 0,
                "failure_action": "Persist timeout/OOM failure, release GPU, stop; never start formal evaluation."},
            "remote_checks_still_required": [
                "Verify effective and real OS user is syx before any CUDA initialization.",
                "Hash the pinned base shards, tokenizer, config and generation config on the actual host; reject adapters.",
                "Record actual Python/torch/transformers/tokenizers/CUDA/GPU identity and available memory.",
                "Select an idle GPU after checking current device processes; never displace other users' jobs.",
                "Re-tokenize these exact old messages and require identical prompt_ids without truncation.",
                "Implement and validate a single-GPU 300-second hard watchdog and failure persistence before executing.",
                "Check the shared JSON answer/citation format on old raw evidence only; do not infer QA accuracy or flat/bound hierarchy understanding.",
                "Measure per-case peak GPU memory and elapsed time at batch size one; semantic accuracy is not established."],
            "limits": "Preparation only, using old raw evidence. Synthetic padding tests resource use only. No flat/bound hierarchy understanding or QA accuracy is verified. No GPU execution CLI is provided."}


def prepare(output, tokenizer_directory):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT) or output == OUTPUT_ROOT:
        raise ValueError("Use a new subdirectory under data/radar_sources_v2/runtime_probe_v1")
    protocol, questions, chunks, bindings = old_inputs()
    tok = verify_tokenizer(tokenizer_directory, protocol)
    for path in (Path(__file__), Path(__file__).with_name("coverage_preflight.py")):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    result = build_plan(tok, protocol, questions, chunks, bindings)
    result["model_identity"]["local_tokenizer_verified"] = True
    import transformers
    import tokenizers
    result["cpu_environment"] = {"python": platform.python_version(), "transformers": transformers.__version__,
                                 "tokenizers": tokenizers.__version__}
    output.mkdir(parents=True)
    write_new(output / "plan.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.output, args.tokenizer)
    print(json.dumps({"status": plan["status"], "input_tokens": [c["input_tokens"] for c in plan["cases"]],
                      "model_run_ready": False, "new_model_runs": 0, "new_GPU_hours": 0}))


if __name__ == "__main__":
    main()
