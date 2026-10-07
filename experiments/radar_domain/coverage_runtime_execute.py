"""Execute only the frozen three OLD-dev resource/format probes, never evaluation.

CPU hashing and retokenization precede one isolated GPU child. Its entire lifetime
is charged to the existing 72 GPU-hour budget. No retries, adapters, training,
reference answers, new questions, truncation, or readiness claims are provided.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import signal
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from . import coverage_runtime_probe as preparation
from .source_readings import ROOT

PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_representation_v1/runtime_probe_preparation.json"
PUBLIC_SHA = "da37da385ae7dee71a41430645c437557543ff5f6cbba4a70bb7af5162c0ab76"
PLAN_SHA = "90291e73b9ba8b3c007418860c111bdf0d3c4a235c3ee5f521dd436ef3ca3eda"
CASE_IDS = ["old_base", "old_padded_8192", "old_padded_31000"]
CASE_LENGTHS = [2733, 8192, 31000]
TOKENIZER = ROOT / "data/radar_sources_v2/representation_v1/tokenizer"
WALL_SECONDS, TERMINATE_GRACE = 300, 2
CLAIM_LIMIT = "Old raw-evidence resource and JSON/citation-format probe only; no QA accuracy, flat/bound comprehension or formal-evaluation readiness is established."


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "Duplicate JSON key")
            obj[key] = value
        return obj
    def bad(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity():
    users = {"real_uid": os.getuid(), "effective_uid": os.geteuid()}
    users.update(real_user=pwd.getpwuid(users["real_uid"]).pw_name,
                 effective_user=pwd.getpwuid(users["effective_uid"]).pw_name)
    require(users["real_user"] == users["effective_user"] == "syx"
            and users["real_uid"] == users["effective_uid"] != 0,
            "Both real and effective OS user must be syx; admin/root are forbidden")
    return users


def load_plan(path):
    require(digest(PUBLIC) == PUBLIC_SHA, "Frozen public preparation changed")
    public = strict_json(PUBLIC.read_text())
    path = Path(path).resolve()
    require(path == (ROOT / public["local_plan"]["path"]).resolve(), "Only the frozen v3 old-dev plan is accepted")
    require(digest(path) == PLAN_SHA == public["local_plan"]["sha256"], "Plan/public SHA binding failed")
    plan = strict_json(path.read_text())
    for key in ("schema", "inputs_sha256", "system_prompt_sha256", "future_execution_boundaries", "evidence_scope"):
        require(plan[key] == public[key], f"Public preparation mismatch: {key}")
    require([c["case_id"] for c in plan["cases"]] == CASE_IDS
            and [c["input_tokens"] for c in plan["cases"]] == CASE_LENGTHS,
            "Exactly three frozen old cases are required")
    for name, expected in plan["inputs_sha256"].items():
        require(digest(ROOT / name) == expected, f"Frozen input changed: {name}")
    require(plan["model_identity"]["adapter"] is None, "Adapters are forbidden")
    return plan


def fingerprint(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "device": s.st_dev, "inode": s.st_ino}


def check_fingerprints(files):
    for name, record in files.items():
        require(fingerprint(name) == record["fingerprint"], f"Attested model file changed: {name}")


def verify_model(model, model_identity):
    """Hash all effective inputs, and derive the unpinned shard index from tensors."""
    model = Path(model).resolve()
    weights = model_identity["expected_base_weights"]["files"]
    require(len(weights) == 2, "Exactly two pinned base shards required")
    expected = {"config.json": model_identity["expected_config_sha256"],
                "generation_config.json": model_identity["expected_generation_config_sha256"]}
    expected.update({w["file"]: w["sha256"] for w in weights})
    index_name = "model.safetensors.index.json"
    # The original checkpoint may contain inactive download remnants/token files.
    # Loading uses ONLY the attested-file view created below, never this directory.
    files = {}
    for name, sha in expected.items():
        path = model / name
        before = fingerprint(path)
        require(digest(path) == sha and before == fingerprint(path), f"Model identity changed: {name}")
        files[str(path)] = {"sha256": sha, "fingerprint": before}
    tensors = {}
    for weight in weights:
        path = model / weight["file"]
        require(path.stat().st_size == weight["bytes"], "Pinned shard size changed")
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            require(0 < size < 16 * 1024 * 1024, "Invalid safetensors header length")
            header = strict_json(stream.read(size).decode("utf-8"))
        for name in header:
            if name != "__metadata__":
                require(name not in tensors, "Tensor appears in multiple shards")
                tensors[name] = weight["file"]
    index_path = model / index_name
    before = fingerprint(index_path)
    index = strict_json(index_path.read_text())
    require(tensors and index.get("weight_map") == tensors, "Shard index must exactly match pinned tensor headers")
    files[str(index_path)] = {"sha256": digest(index_path), "fingerprint": before}
    check_fingerprints(files)
    return files


def tokenizer_precheck(directory, model_identity):
    directory = Path(directory).resolve()
    expected = dict(model_identity["expected_tokenizer_sha256"])
    expected["config.json"] = model_identity["expected_config_sha256"]
    files = {}
    for name, sha in expected.items():
        path = directory / name
        before = fingerprint(path)
        require(digest(path) == sha and before == fingerprint(path), "Tokenizer file changed")
        files[str(path)] = {"sha256": sha, "fingerprint": before}
    tokenizer = preparation.verify_tokenizer(directory, {"tokenizer_sha256": model_identity["expected_tokenizer_sha256"]})
    check_fingerprints(files)
    return tokenizer, files


def model_view(output, files):
    view = Path(output) / "verified_model"
    view.mkdir()
    for name in files:
        (view / Path(name).name).symlink_to(Path(name))
    return view


def check_model_view(view, expected):
    view = Path(view)
    require({p.name for p in view.iterdir()} == set(expected), "Restricted model view roster changed")
    for name, target in expected.items():
        require((view / name).resolve() == Path(target), "Restricted model view target changed")


def verify_prompts(tokenizer, plan):
    require([c["case_id"] for c in plan["cases"]] == CASE_IDS, "Unexpected case roster")
    for case in plan["cases"]:
        ids = preparation.token_ids(tokenizer, case["messages"])
        require(ids == case["prompt_ids"] and len(ids) == case["input_tokens"]
                and case["truncated"] is False, "Prompt IDs changed or truncation occurred")


def parse_gpu_xml(text, selector):
    root = ET.fromstring(text)
    devices = root.findall("gpu")
    selected = [g for i, g in enumerate(devices) if selector in (str(i), g.findtext("uuid"))]
    require(len(selected) == 1, "GPU must be one physical index or UUID")
    gpu = selected[0]
    display = []
    for process in gpu.findall("./processes/process_info"):
        memory = process.findtext("used_memory", "")
        require(process.findtext("type") == "G"
                and process.findtext("process_name") == "/usr/lib/xorg/Xorg"
                and memory.endswith(" MiB") and 0 <= int(memory[:-4]) <= 8,
                "Selected GPU has an active or unrecognized process")
        display.append({"pid": process.findtext("pid"), "type": "G",
                        "process_name": "/usr/lib/xorg/Xorg", "used_memory": memory})
    require(gpu.findtext("./mig_mode/current_mig") in ("Disabled", "N/A", "[N/A]"), "MIG is not supported by this single-card probe")
    def amount(path, unit):
        text = gpu.findtext(path, "")
        require(text.endswith(unit), f"Missing GPU telemetry: {path}")
        return int(text[:-len(unit)].strip())
    used = amount("./fb_memory_usage/used", "MiB")
    utilization = amount("./utilization/gpu_util", "%")
    require(used <= 256 and utilization == 0, "Selected GPU is not idle")
    return {"uuid": gpu.findtext("uuid"), "name": gpu.findtext("product_name"),
            "driver_version": root.findtext("driver_version"), "used_MiB": used,
            "total_MiB": amount("./fb_memory_usage/total", "MiB"), "gpu_utilization_percent": utilization,
            "allowed_idle_display_processes": display}


def idle_gpu(selector):
    result = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=15)
    return parse_gpu_xml(result.stdout, selector)


def greedy_config(generation_config, tokenizer):
    """Fresh config avoids inheriting model defaults such as top_p/temperature."""
    from transformers import GenerationConfig
    config = GenerationConfig(max_new_tokens=768, do_sample=False, num_beams=1,
                              num_beam_groups=1, num_return_sequences=1,
                              temperature=1.0, top_p=1.0, top_k=50,
                              repetition_penalty=1.0, length_penalty=1.0,
                              use_cache=True, bos_token_id=generation_config.get("bos_token_id"),
                              eos_token_id=generation_config["eos_token_id"],
                              pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    require(config.get_generation_mode().value == "greedy_search", "Effective generation must be greedy")
    return config


def parse_format(text, case):
    answer = strict_json(text)
    require(type(answer) is dict and set(answer) == {"answer_text", "citations"}, "Wrong answer JSON fields")
    require(type(answer["answer_text"]) is str and bool(answer["answer_text"].strip()), "Missing answer text")
    require(type(answer["citations"]) is list, "Citations must be an array")
    evidence = strict_json(case["messages"][1]["content"])["evidence"]
    allowed = {chunk["chunk_id"] for chunk in evidence["chunks"]}
    for citation in answer["citations"]:
        require(type(citation) is dict and set(citation) == {"chunk_id"}
                and type(citation["chunk_id"]) is str and citation["chunk_id"] in allowed,
                "Citation must identify supplied raw evidence")
    return answer


def generate_one(model, ids, mask, config):
    # Replace checkpoint defaults too: current Transformers fills unset fields
    # from model.generation_config and no longer accepts use_model_defaults.
    model.generation_config = config
    return model.generate(input_ids=ids, attention_mask=mask, generation_config=config)


def check_generation_api(model_path, tokenizer):
    """Validate installed generation configuration on meta tensors, without CUDA."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = greedy_config(strict_json((Path(model_path) / "generation_config.json").read_text()), tokenizer)
    architecture = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(architecture, trust_remote_code=False,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.generation_config = config
    effective, kwargs = model._prepare_generation_config(config, input_ids=torch.empty((1, 1), dtype=torch.long),
                                                        attention_mask=torch.ones((1, 1), dtype=torch.long))
    model._validate_model_kwargs(kwargs.copy())
    require(effective.get_generation_mode().value == "greedy_search"
            and effective.max_new_tokens == 768 and not torch.cuda.is_initialized(),
            "CPU generation validation changed decoding or initialized CUDA")
    return effective.to_dict()


def terminate_owned_group(process, sig):
    # start_new_session=True makes this child's pid the session/process-group id.
    # Never inspect/terminate any unrelated GPU process or shared process group.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def watch_child(command, env, output, wall_seconds=WALL_SECONDS, grace=TERMINATE_GRACE):
    require(0 < 2 * grace < wall_seconds <= WALL_SECONDS, "Invalid watchdog budget")
    started = time.monotonic()
    status, process, error = "completed", None, None
    with (Path(output) / "worker.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=ROOT)
            try:
                process.wait(timeout=max(0, wall_seconds - 2 * grace - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                status = "timeout"
                terminate_owned_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=min(grace, max(0, wall_seconds - grace - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    terminate_owned_group(process, signal.SIGKILL)
                    try:
                        process.wait(timeout=max(0, wall_seconds - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        status = "cleanup_unverified"
            if process.returncode != 0 and status == "completed":
                status = "worker_failed"
        except BaseException as exc:
            status, error = "interrupted", f"{type(exc).__name__}: {exc}"
            if process is not None:
                terminate_owned_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=min(grace, max(0, wall_seconds - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    status = "cleanup_unverified"
        finally:
            if process is not None:
                terminate_owned_group(process, signal.SIGKILL)  # Any descendants in our own session.
    reaped = process is None or process.returncode is not None
    return {"status": status, "returncode": None if process is None else process.returncode,
            "gpu_child_wall_seconds": 0 if process is None else time.monotonic() - started,
            "gpu_count": int(process is not None), "hard_watchdog_seconds": wall_seconds,
            "process_started": process is not None, "process_reaped": reaped,
            "GPU_hours_is_lower_bound": not reaped, "watchdog_error": error}


def collect_cases(output):
    result = {"written_cases": [], "format_passed_cases": [], "incomplete_case_files": []}
    for name in CASE_IDS:
        path = Path(output) / f"{name}.json"
        if path.exists():
            result["written_cases"].append(name)
            try:
                row = strict_json(path.read_text())
                if row["format_error"] is None:
                    result["format_passed_cases"].append(name)
            except (ValueError, TypeError, KeyError, OSError):
                result["incomplete_case_files"].append(path.name)
    return result


def worker(plan_path, output, model):
    """Internal child entry; all CUDA work occurs here, under the parent watchdog."""
    started = time.monotonic()
    stage, current_case = "worker_precheck", None
    try:
        identity()
        attestation = strict_json((output / "cpu_preflight.json").read_text())
        require(os.getppid() == attestation["parent_pid"] and os.getsid(0) == os.getpid(), "Worker requires its supervising parent and private session")
        require(digest(plan_path) == PLAN_SHA and str(model.resolve()) == attestation["model_path"], "Worker input binding changed")
        require(digest(Path(__file__)) == attestation["executor_sha256"], "Executor changed after CPU preflight")
        check_fingerprints(attestation["model_files"])
        check_model_view(attestation["model_view"], attestation["model_view_files"])
        gpu = idle_gpu(attestation["gpu"]["uuid"])
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpu["uuid"], "Worker must see only the selected UUID")
        plan = strict_json(plan_path.read_text())
        import torch
        import transformers
        import tokenizers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "One bf16-capable CUDA GPU required")
        tokenizer = preparation.verify_tokenizer(attestation["tokenizer_path"],
                     {"tokenizer_sha256": plan["model_identity"]["expected_tokenizer_sha256"]})
        verify_prompts(tokenizer, plan)
        config = greedy_config(strict_json((model / "generation_config.json").read_text()), tokenizer)
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": gpu,
                       "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "identity": identity(),
                       "effective_generation_config": config.to_dict()}
        write_new(output / "environment.json", environment)
        stage = "model_loading"
        load_started = time.monotonic()
        torch.manual_seed(2026)
        model_object = AutoModelForCausalLM.from_pretrained(attestation["model_view"], local_files_only=True, trust_remote_code=False,
                       use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        model_object.eval()
        torch.cuda.synchronize()
        require(model_object.config._attn_implementation == "sdpa" and model_object.dtype == torch.bfloat16, "Effective model dtype/attention changed")
        require(model_object.config.max_position_embeddings >= 32768, "Model context capacity is insufficient")
        check_fingerprints(attestation["model_files"])
        write_new(output / "model_loaded.json", {"seconds": time.monotonic() - load_started,
                  "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "attention": "sdpa", "dtype": "bfloat16"})
        for case in plan["cases"]:
            stage, current_case = "generation", case["case_id"]
            write_new(output / f"{current_case}.started.json", {"case_id": current_case, "input_tokens": case["input_tokens"], "worker_seconds": time.monotonic() - started})
            torch.cuda.reset_peak_memory_stats()
            case_started = time.monotonic()
            ids = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                generated = generate_one(model_object, ids, mask, config)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - case_started
            tokens = generated[0, ids.shape[1]:].tolist()
            require(0 < len(tokens) <= 768, "Invalid generated-token budget")
            eos = config.eos_token_id if type(config.eos_token_id) is list else [config.eos_token_id]
            ended = tokens[-1] in eos
            text = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
            result = {"case_id": current_case, "input_tokens": len(case["prompt_ids"]), "generated_tokens": len(tokens),
                      "generated_ids": tokens, "generated_text": text, "stop_reason": "eos" if ended else "generation_budget",
                      "seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "truncated": False,
                      "parsed": None, "format_error": None}
            try:
                result["parsed"] = parse_format(text, case)
            except (ValueError, TypeError, KeyError) as exc:
                result["format_error"] = f"{type(exc).__name__}: {exc}"
            write_new(output / f"{current_case}.json", result)
            require(result["format_error"] is None, "Output format failed; stop without retries")
            del generated, ids, mask
        write_new(output / "worker_result.json", {"status": "completed", "case_count": 3, "seconds": time.monotonic() - started})
        return 0
    except BaseException as exc:
        failure = {"status": "failed", "stage": stage, "case_id": current_case,
                   "error_type": type(exc).__name__, "error": str(exc), "seconds": time.monotonic() - started,
                   "oom": "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower(),
                   "traceback": traceback.format_exc()}
        if "torch" in locals() and torch.cuda.is_initialized():
            try:
                failure["gpu_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            except Exception as memory_error:
                failure["memory_telemetry_error"] = str(memory_error)
        write_new(output / "worker_failure.json", failure)
        return 1  # Process exit releases the CUDA context, including failed loads.


def execute(plan_path, output, model, gpu, tokenizer_path=TOKENIZER, wall_seconds=WALL_SECONDS):
    output, model = Path(output).resolve(), Path(model).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, child_started = time.monotonic(), None
    result = {"schema": "radar_old_dev_runtime_execution_v1", "status": "cpu_precheck_failed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
              "public_preparation_sha256": PUBLIC_SHA, "formal_evaluation_permitted": False,
              "model_run_ready": False, "new_training_runs": 0, "claim_limit": CLAIM_LIMIT,
              "charged_to_existing_gpu_hours_budget": 72, "new_GPU_hours": 0, "new_model_runs": 0}
    try:
        users = identity()
        plan = load_plan(plan_path)
        files = verify_model(model, plan["model_identity"])
        tokenizer, tokenizer_files = tokenizer_precheck(tokenizer_path, plan["model_identity"])
        verify_prompts(tokenizer, plan)
        check_fingerprints(files)
        view = model_view(output, files)
        effective_generation = check_generation_api(view, tokenizer)
        view_files = {Path(p).name: p for p in files}
        files.update(tokenizer_files)
        selected = idle_gpu(gpu)
        attestation = {"parent_pid": os.getpid(), "model_path": str(model), "model_files": files,
                       "gpu": selected, "identity": users, "plan_sha256": PLAN_SHA,
                       "executor_sha256": digest(Path(__file__)), "all_prompt_ids_verified": True,
                       "CPU_validated_effective_generation_config": effective_generation,
                       "model_view": str(view), "model_view_files": view_files,
                       "tokenizer_path": str(Path(tokenizer_path).resolve())}
        write_new(output / "cpu_preflight.json", attestation)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        command = [sys.executable, "-B", "-m", "experiments.radar_domain.coverage_runtime_execute",
                   "--plan", str(Path(plan_path).resolve()), "--output", str(output), "--model", str(model),
                   "--gpu", selected["uuid"], "--_worker"]
        child_started = time.monotonic()
        result["new_model_runs"] = 1
        result.update(watch_child(command, env, output, wall_seconds=wall_seconds))
        result["new_model_runs"] = int(result["process_started"])
        result["new_GPU_hours"] = result["gpu_child_wall_seconds"] / 3600
        result.update(collect_cases(output))
        if result["status"] == "completed":
            require(strict_json((output / "worker_result.json").read_text())["case_count"] == 3,
                    "Worker exited without completing all three cases")
            require(result["format_passed_cases"] == CASE_IDS, "Three complete format-valid case files required")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if child_started is not None and "gpu_child_wall_seconds" not in result:
            result["gpu_child_wall_seconds"] = time.monotonic() - child_started
            result["new_GPU_hours"] = result["gpu_child_wall_seconds"] / 3600
    finally:
        result["total_wall_seconds"] = time.monotonic() - started
        result["accounting_basis"] = "Conservative single-card GPU child lifetime, including imports/loading/cleanup; CPU preflight excluded."
        write_new(output / "runtime.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--max-seconds", type=float, default=WALL_SECONDS,
                        help="May lower, never raise, the 300-second GPU child limit")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Convert SIGTERM to an exception so the parent can reap only its own child.
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if args._worker:
        return worker(args.plan, args.output, args.model)
    result = execute(args.plan, args.output, args.model, args.gpu, args.tokenizer, args.max_seconds)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
