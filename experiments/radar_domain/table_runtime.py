"""Local CUDA/input binding for explicitly invoked domain training or generation.

Imports are CPU-safe. The probe does not load model weights and is not a model
forward/backward smoke test. No remote access, downloads, resets or process kills.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(bindings):
    if not bindings:
        raise ValueError("Nonempty file bindings required")
    for path, expected in bindings.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError("Missing or changed bound file: " + str(path))


def model_files(base_model_dir, adapter_dir=None):
    """Identify complete local official base weights and an optional LoRA adapter."""
    base = Path(base_model_dir)
    expected = json.loads(Path("results/agent_feedback/weights_verified.json").read_text())
    paths, known_weights = [], {}
    for item in expected["files"]:
        path = base / item["file"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Pinned official Qwen base weight missing or changed: " + str(path))
        known_weights[str(path)] = item["sha256"]
    for required in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        if not (base / required).is_file():
            raise ValueError("Base configuration/tokenizer missing: " + required)
    allowed = {".json", ".safetensors", ".txt", ".jinja"}
    paths.extend(p for p in base.iterdir() if p.is_file() and p.suffix in allowed)
    if adapter_dir:
        adapter = Path(adapter_dir)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / name).is_file():
                raise ValueError("Local LoRA adapter missing: " + str(adapter / name))
        paths.extend(p for p in adapter.iterdir() if p.is_file() and p.suffix in allowed)
    return {str(p): known_weights.get(str(p)) or sha(p) for p in sorted(set(paths))}


def directory_identity(bindings, directory):
    root = Path(directory).resolve()
    selected = {Path(p).name: h for p, h in bindings.items() if Path(p).resolve().parent == root}
    if not selected:
        raise ValueError("No bound files belong to the specified model directory")
    return selected


def _user_check(expected_user):
    euid = os.geteuid()
    user = pwd.getpwuid(euid).pw_name
    if expected_user in {"root", "ubuntu", ""} or user != expected_user or euid == 0:
        raise ValueError("Use the specified ordinary experiment account; admin training/generation is forbidden")
    if platform.system() != "Linux":
        raise ValueError("Formal model runs require the registered Linux CUDA environment")
    return {"user": user, "euid": euid, "platform": platform.platform()}


def _gpu_state(gpu_index):
    if type(gpu_index) is not int or gpu_index < 0:
        raise ValueError("Explicit nonnegative GPU index required")
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu_index),
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    uuid, memory, utilization = [s.strip() for s in raw.split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    others = []
    for line in processes.strip().splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
            others.append(int(fields[1]))
    if others or int(memory) > 1024 or int(utilization) > 5:
        raise ValueError("Selected GPU is occupied or not idle; no existing task is interrupted")
    return {"index": gpu_index, "uuid": uuid, "memory_used_mib": int(memory),
            "utilization_percent": int(utilization), "other_compute_processes": others}


def _bind_visible_device(gpu_index):
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is not None and current != str(gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES must match the selected physical GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def _cuda_probe(gpu_index):
    _bind_visible_device(gpu_index)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one selected CUDA device must be available")
    value = torch.tensor([1.0, 2.0], device="cuda")
    passed = float((value * 2).sum().item()) == 6.0
    torch.cuda.synchronize()
    del value
    torch.cuda.empty_cache()
    if not passed:
        raise ValueError("CUDA tensor probe failed")
    return {"cuda_tensor_check_passed": True, "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "model_loaded": False,
            "model_gpu_smoke_passed": False}


def preflight_inputs(*, purpose, base_model_dir, adapter_dir, input_files_sha256,
                     implementation_sha256, output, ledger_path, job_id, expected_user="syx", gpu_index=0):
    if purpose not in {"training", "generation"}:
        raise ValueError("Purpose must be training or generation")
    from .table_job_budget import JobBudget, require_round_ledger
    require_round_ledger(ledger_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Keep prior runtime evidence")
    identity = _user_check(expected_user)
    verify_files(input_files_sha256); verify_files(implementation_sha256)
    weights = model_files(base_model_dir, adapter_dir)
    gpu = _gpu_state(gpu_index)
    with JobBudget(ledger_path=ledger_path, job_id=job_id, phase="probe", max_gpu_hours=1/60, gpu_count=1) as budget:
        probe = _cuda_probe(gpu_index)
        budget.check()
    report = {"schema": "radar_table_runtime_evidence_v1", "status": "passed",
              "scope": "cuda_device_and_input_binding_only", "purpose": purpose,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), **identity,
              "cuda": gpu, "probe": probe, "base_model_dir": str(base_model_dir),
              "gpu_budget": {"ledger_path": str(ledger_path), "job_id": job_id, "max_gpu_hours": 1/60},
              "adapter_dir": str(adapter_dir) if adapter_dir else None,
              "model_files_sha256": weights, "implementation_sha256": implementation_sha256,
              "input_files_sha256": input_files_sha256}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    return report


def validate_runtime_evidence(path, *, purpose, model_files_sha256,
                              implementation_sha256, input_files_sha256,
                              max_age_seconds=900, expected_user="syx"):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get("schema") != "radar_table_runtime_evidence_v1" or evidence.get("status") != "passed"
            or evidence.get("purpose") != purpose or evidence.get("scope") != "cuda_device_and_input_binding_only"
            or not evidence.get("probe", {}).get("cuda_tensor_check_passed")):
        raise ValueError("Matching successful CUDA/input evidence required")
    _user_check(expected_user)
    if evidence["user"] != expected_user or evidence["euid"] != os.geteuid():
        raise ValueError("Runtime account identity changed")
    created = datetime.fromisoformat(evidence["created_at_utc"])
    age = (datetime.now(timezone.utc) - created).total_seconds()
    if not 0 <= age <= max_age_seconds:
        raise ValueError("Runtime evidence expired; recheck the available device")
    from .table_job_budget import require_round_ledger
    require_round_ledger(evidence["gpu_budget"]["ledger_path"])
    if evidence["model_files_sha256"] != model_files_sha256:
        raise ValueError("Model/adapter binding differs")
    for key, expected in (("implementation_sha256", implementation_sha256), ("input_files_sha256", input_files_sha256)):
        if any(evidence[key].get(p) != h for p, h in expected.items()):
            raise ValueError("Runtime " + key + " binding differs")
    verify_files(model_files_sha256); verify_files(implementation_sha256); verify_files(input_files_sha256)
    gpu = _gpu_state(evidence["cuda"]["index"])
    if gpu["uuid"] != evidence["cuda"]["uuid"]:
        raise ValueError("CUDA device identity changed")
    _bind_visible_device(gpu["index"])
    return evidence
