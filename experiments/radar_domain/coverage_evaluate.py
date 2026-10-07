"""One frozen 96-question arm: generate and preserve outputs, never score them.

Only messages and token identities enter this runner. References, annotations and
semantic judgments are neither opened nor inferred. Three exclusive arm slots,
each capped at 2400 single-GPU seconds, reserve at most 7200 GPU seconds total.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from . import coverage_runtime_execute as runtime
from .source_readings import ROOT

ARMS = ("raw", "flat", "bound")
INPUT_PATH = "data/radar_sources_v2/representation_v1/semantic_payload_v1/inputs.jsonl"
INPUT_SHA = "f315d92f8ed6b0cba237e3496b60b44fdff4749f85f2efcceb37f789d559c1d1"
INFERENCE = {"seed": 2026, "batch_size": 1, "dtype": "bfloat16", "attention": "sdpa",
             "decoding": "greedy", "max_context_tokens": 32768,
             "max_generated_tokens": 768, "truncate": False}
REQUIRED_CODE = tuple("experiments/radar_domain/" + name for name in (
    "coverage_evaluate.py", "coverage_runtime_execute.py", "coverage_runtime_probe.py",
    "coverage_preflight.py", "source_readings.py"))
require, digest, strict_json, write_new = runtime.require, runtime.digest, runtime.strict_json, runtime.write_new


def path_in_root(relative):
    path = (ROOT / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(ROOT.resolve()), "Expected repository-relative path")
    return path


def load_protocol(path, expected_sha):
    require(digest(path) == expected_sha, "Protocol SHA mismatch")
    p = strict_json(Path(path).read_text())
    require(p["schema"] == "radar_coverage_evaluation_protocol_v1" and p["status"] == "frozen", "Protocol is not frozen")
    require(p["inputs"] == {"path": INPUT_PATH, "sha256": INPUT_SHA}, "Wrong frozen generation inputs")
    require(p["inference"] == INFERENCE and p["model_identity"]["adapter"] is None, "Inference or adapter drift")
    execution = p["execution"]
    require(execution["max_gpu_seconds_per_arm"] == 2400 and execution["max_total_gpu_seconds"] == 7200
            and execution["automatic_retries"] == 0 and execution["training_runs"] == 0
            and execution["project_gpu_hours_budget"] == 72, "Fixed one-round resource envelope changed")
    require(type(execution["project_gpu_hours_before"]) in (int, float)
            and 0 <= execution["project_gpu_hours_before"] <= 70,
            "Prior project usage plus the reserved two GPU hours exceeds 72")
    require(set(execution["gpu_by_arm"]) == set(ARMS)
            and len(set(execution["gpu_by_arm"].values())) == 3
            and all(v.startswith("GPU-") for v in execution["gpu_by_arm"].values()), "Three distinct physical GPU UUIDs required")
    require(set(REQUIRED_CODE).issubset(p["code_sha256"]), "Missing executable code binding")
    require(type(p["question_ids"]) is list and len(p["question_ids"]) == len(set(p["question_ids"])) == 96
            and all(type(q) is str and q and all(c.isalnum() or c in "_-" for c in q)
                    for q in p["question_ids"]), "Expected fixed 96-question inventory")
    for name, sha in p["code_sha256"].items():
        require(name.endswith(".py") and digest(path_in_root(name)) == sha, f"Bound code changed: {name}")
    for key in ("inputs", "token_identity", "old_dev_pass"):
        require(digest(path_in_root(p[key]["path"])) == p[key]["sha256"], f"Bound {key} file changed")
    old = strict_json(path_in_root(p["old_dev_pass"]["path"]).read_text())
    require(old["status"] == "completed" and old["format_passed_cases"] == runtime.CASE_IDS
            and old.get("process_reaped") is True and old.get("GPU_hours_is_lower_bound") is False,
            "Old-dev runtime gate has not passed")
    return p


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def select_inputs(rows, identities, arm, count=96):
    require(arm in ARMS and len(rows) == len(identities) == count * 3, "Wrong frozen row count")
    expected_keys = {"question_id", "arm", "messages", "input_tokens", "truncated"}
    identity_keys = {"question_id", "arm", "token_count", "token_ids_sha256"}
    seen, qids, selected = set(), [], []
    for position, (row, bound) in enumerate(zip(rows, identities)):
        require(set(row) == expected_keys and set(bound) == identity_keys, "Generation input contains unexpected fields")
        require(row["arm"] == ARMS[position % 3], "Frozen arm order changed")
        key = (row["question_id"], row["arm"])
        require(key not in seen and key == (bound["question_id"], bound["arm"]), "Duplicate/mismatched token identity")
        seen.add(key)
        if position % 3 == 0:
            require(type(row["question_id"]) is str and row["question_id"]
                    and all(c.isalnum() or c in "_-" for c in row["question_id"]), "Invalid question ID")
            qids.append(row["question_id"])
        require(row["question_id"] == qids[-1] and row["truncated"] is False, "Question order/truncation changed")
        require(type(row["input_tokens"]) is int and 0 < row["input_tokens"] <= 32000
                and row["input_tokens"] == bound["token_count"], "Invalid stored token count")
        require(type(row["messages"]) is list and len(row["messages"]) == 2
                and [m.get("role") for m in row["messages"]] == ["system", "user"]
                and all(set(m) == {"role", "content"} and type(m["content"]) is str for m in row["messages"]), "Wrong prompt structure")
        if row["arm"] == arm:
            selected.append({**row, "token_ids_sha256": bound["token_ids_sha256"]})
    require(len(set(qids)) == count, "Question denominator must remain fixed")
    return selected


def load_inputs(protocol, arm):
    rows = [strict_json(line) for line in path_in_root(protocol["inputs"]["path"]).read_text().splitlines() if line.strip()]
    identities = strict_json(path_in_root(protocol["token_identity"]["path"]).read_text())
    selected = select_inputs(rows, identities, arm)
    require([r["question_id"] for r in selected] == protocol["question_ids"], "Protocol question order changed")
    return selected


def tokenize_rows(tokenizer, rows):
    result = []
    for row in rows:
        ids = runtime.preparation.token_ids(tokenizer, row["messages"])
        require(len(ids) == row["input_tokens"] and token_hash(ids) == row["token_ids_sha256"],
                f"Prompt IDs/count changed: {row['question_id']}")
        result.append({**row, "prompt_ids": ids})
    return result


def environment():
    import torch
    import transformers
    import tokenizers
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "cuda": torch.version.cuda}


def format_diagnostic(text, row):
    """Syntax and source-ID membership only; never correctness or support."""
    try:
        answer = strict_json(text)
        require(type(answer) is dict and set(answer) == {"answer_text", "citations"}, "Wrong answer fields")
        require(type(answer["answer_text"]) is str and bool(answer["answer_text"].strip())
                and type(answer["citations"]) is list, "Wrong answer/citations types")
        allowed = set()
        def visit(value):
            if type(value) is dict:
                if type(value.get("chunk_id")) is str:
                    allowed.add(value["chunk_id"])
                for child in value.values():
                    visit(child)
            elif type(value) is list:
                for child in value:
                    visit(child)
        visit(strict_json(row["messages"][1]["content"])["evidence"])
        for citation in answer["citations"]:
            require(type(citation) is dict and set(citation) == {"chunk_id"}
                    and type(citation["chunk_id"]) is str and citation["chunk_id"] in allowed, "Unknown source citation")
        return {"valid": True, "error": None}
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def watch(command, env, output, seconds=2400, grace=2):
    """Leave one grace interval before the hard cap for scheduling/accounting."""
    require(0 < 3 * grace < seconds <= 2400, "Invalid single-arm watchdog")
    started, process, status, error = time.monotonic(), None, "completed", None
    with (Path(output) / "worker.log").open("x") as log:
        try:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write_new(Path(output) / "child_started.json", {"pid": process.pid, "started_at_unix": time.time(),
                      "gpu_count": 1, "max_seconds": seconds})
            try:
                process.wait(timeout=max(0, seconds - 3 * grace - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                status = "budget_stopped"
                runtime.terminate_owned_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    runtime.terminate_owned_group(process, signal.SIGKILL)
                    try:
                        process.wait(timeout=max(0, seconds - grace - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        status = "cleanup_unverified"
            if status == "completed" and process.returncode != 0:
                status = "worker_failed"
        except BaseException as exc:
            status, error = "interrupted", f"{type(exc).__name__}: {exc}"
            if process is not None:
                runtime.terminate_owned_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=min(grace, max(0, seconds - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    status = "cleanup_unverified"
        finally:
            if process is not None:
                runtime.terminate_owned_group(process, signal.SIGKILL)
    elapsed = 0 if process is None else time.monotonic() - started
    reaped = process is None or process.returncode is not None
    return {"status": status, "error": error, "returncode": None if process is None else process.returncode,
            "process_started": process is not None, "process_reaped": reaped,
            "gpu_seconds": elapsed, "gpu_hours": elapsed / 3600, "GPU_hours_is_lower_bound": not reaped,
            "within_arm_budget": elapsed <= seconds and reaped}


def inventory(output, rows):
    result = []
    for row in rows:
        qid = row["question_id"]
        path, status = Path(output) / f"{qid}.json", "not_run"
        if path.exists():
            try:
                saved = strict_json(path.read_text())
                require(saved["question_id"] == qid and saved["arm"] == row["arm"] and saved["status"] == "generated", "Output identity mismatch")
                status = "generated"
            except (ValueError, KeyError, TypeError, OSError):
                status = "incomplete_output"
        elif (Path(output) / f"{qid}.started.json").exists():
            status = "failed_or_interrupted"
        result.append({"question_id": qid, "status": status})
    return result


def worker(protocol_path, protocol_sha, arm, output):
    started, current, stage = time.monotonic(), None, "worker_precheck"
    try:
        runtime.identity()
        receipt = strict_json((output / "cpu_preflight.json").read_text())
        require(receipt["parent_pid"] == os.getppid() and os.getsid(0) == os.getpid(), "Worker requires its supervising parent/private session")
        p = load_protocol(protocol_path, protocol_sha)
        require(receipt["protocol_sha256"] == protocol_sha and receipt["arm"] == arm, "Worker receipt binding changed")
        runtime.check_fingerprints(receipt["model_files"])
        runtime.check_model_view(receipt["model_view"], receipt["model_view_files"])
        require(environment() == p["runtime_environment"], "Runtime environment changed")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == p["execution"]["gpu_by_arm"][arm], "Wrong visible GPU")
        gpu = runtime.idle_gpu(os.environ["CUDA_VISIBLE_DEVICES"])
        tokenizer, unused = runtime.tokenizer_precheck(p["tokenizer_path"], p["model_identity"])
        rows = tokenize_rows(tokenizer, load_inputs(p, arm))
        import torch
        from transformers import AutoModelForCausalLM
        require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "One bf16 GPU required")
        config = runtime.greedy_config(strict_json((Path(p["model_path"]) / "generation_config.json").read_text()), tokenizer)
        write_new(output / "environment.json", {**environment(), "gpu": gpu,
                  "effective_generation_config": config.to_dict(), "identity": runtime.identity()})
        torch.manual_seed(2026)
        stage = "model_loading"
        model = AutoModelForCausalLM.from_pretrained(receipt["model_view"], local_files_only=True,
                trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda:0")
        model.eval()
        require(model.dtype == torch.bfloat16 and model.config._attn_implementation == "sdpa"
                and model.config.max_position_embeddings >= 32768, "Model dtype/attention/context changed")
        runtime.check_fingerprints(receipt["model_files"])
        for row in rows:
            current, stage = row["question_id"], "generation"
            write_new(output / f"{current}.started.json", {"question_id": current, "arm": arm, "worker_seconds": time.monotonic() - started})
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                generated = runtime.generate_one(model, ids, mask, config)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            tokens = generated[0, ids.shape[1]:].tolist()
            require(0 < len(tokens) <= 768, "Generated budget changed")
            eos = config.eos_token_id if type(config.eos_token_id) is list else [config.eos_token_id]
            ended = tokens[-1] in eos
            text = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
            write_new(output / f"{current}.json", {"question_id": current, "arm": arm, "status": "generated",
                "input_tokens": row["input_tokens"], "prompt_ids_sha256": row["token_ids_sha256"],
                "generated_ids": tokens, "generated_text": text, "generated_tokens": len(tokens),
                "stop_reason": "eos" if ended else "generation_budget", "seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "truncated": False, "format_diagnostic": format_diagnostic(text, row)})
            del ids, mask, generated
        write_new(output / "worker_result.json", {"status": "completed", "question_count": 96})
        return 0
    except BaseException as exc:
        failure = {"status": "failed", "question_id": current, "stage": stage, "seconds": time.monotonic() - started,
                   "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        if "torch" in locals() and torch.cuda.is_initialized():
            try:
                failure["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                failure["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            except Exception:
                pass
        write_new(output / "worker_failure.json", failure)
        return 1


def execute(protocol_path, protocol_sha, arm, gpu, output):
    runtime.identity()
    p = load_protocol(protocol_path, protocol_sha)
    require(arm in ARMS, "Unknown arm")
    output = Path(output).resolve()
    require(output == path_in_root(p["execution"]["output_root"]) / arm, "Use the arm's single frozen output slot")
    output.mkdir(parents=True, exist_ok=False)  # Consumes the arm slot; never retry elsewhere.
    rows = [{"question_id": q, "arm": arm} for q in p["question_ids"]]
    report = {"schema": "radar_coverage_arm_execution_v1", "status": "cpu_precheck_failed",
        "arm": arm, "protocol_sha256": protocol_sha, "question_denominator": 96,
        "gpu_seconds": 0, "gpu_hours": 0, "new_training_runs": 0,
        "semantic_scoring_performed": False, "automatic_retries": 0,
        "accounting_basis": "Entire single-GPU child lifetime, including imports, loading, generation and cleanup; CPU preflight excluded.",
        "project_gpu_hours_budget": 72, "formal_round_reserved_gpu_seconds": 7200}
    report["project_gpu_hours_before"] = p["execution"]["project_gpu_hours_before"]
    try:
        rows = load_inputs(p, arm)
        require(environment() == p["runtime_environment"], "CPU runtime environment differs from frozen protocol")
        model_files = runtime.verify_model(p["model_path"], p["model_identity"])
        tokenizer, token_files = runtime.tokenizer_precheck(p["tokenizer_path"], p["model_identity"])
        tokenize_rows(tokenizer, rows)
        view = runtime.model_view(output, model_files)
        effective = runtime.check_generation_api(view, tokenizer)
        selected = runtime.idle_gpu(gpu)
        require(selected["uuid"] == p["execution"]["gpu_by_arm"][arm], "GPU differs from frozen arm allocation")
        view_files = {Path(name).name: name for name in model_files}
        model_files.update(token_files)
        runtime.check_fingerprints(model_files)
        write_new(output / "cpu_preflight.json", {"parent_pid": os.getpid(), "arm": arm,
            "protocol_sha256": protocol_sha, "model_files": model_files,
            "model_view": str(view), "model_view_files": view_files,
            "all_96_prompt_ids_verified": True, "effective_generation_config": effective})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="2026")
        command = [sys.executable, "-B", "-m", "experiments.radar_domain.coverage_evaluate",
            "--protocol", str(Path(protocol_path).resolve()), "--protocol-sha256", protocol_sha,
            "--arm", arm, "--gpu", selected["uuid"], "--output", str(output), "--_worker"]
        report.update(watch(command, env, output))
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        report["rows"] = inventory(output, rows)
        report["generated_count"] = sum(row["status"] == "generated" for row in report["rows"])
        report["not_run_question_ids"] = [r["question_id"] for r in report["rows"] if r["status"] == "not_run"]
        if report["status"] == "completed" and (report["generated_count"] != 96 or not report["within_arm_budget"]):
            report["status"] = "incomplete_or_budget_failure"
        write_new(output / "runtime.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        return worker(args.protocol, args.protocol_sha256, args.arm, args.output)
    result = execute(args.protocol, args.protocol_sha256, args.arm, args.gpu, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
