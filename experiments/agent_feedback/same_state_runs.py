"""Bounded, explicit stages for the frozen same-state diagnostic; no training.

Run under syx with the explicit project Python and PYTHONHASHSEED=20261003.
Stages never retry jobs or overwrite evidence. Availability failures before a
stage starts do not create a started marker. Completed artifacts are audited
separately, so an observation timeout never authorizes another generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from . import inference
from .compare import validate_prediction
from .download_weights import ROOT as MODEL_ROOT, FILES
from .recovery import sha256
from .run_job import total_gpu_seconds
from .same_state import restore, write_new
from .same_state_analysis import validate_resumes
from .same_state_cohort import construct_cohort
from .same_state_inference import ARMS, validate_inventory
from .training_data import read_jsonl

ROOT = Path("results/agent_feedback/same_state_v1")
DATA = Path("data/agent_feedback/same_state_v1")
PROTOCOL = ROOT / "design_protocol.json"
POOL = ROOT / "pool_manifest.json"
KB = Path("datasets/kqa_pro/kb.json")
QUESTIONS, GOLD = DATA / "pool.questions.jsonl", DATA / "pool.gold.jsonl"
PROBE_STATES, STATES = DATA / "probe.states.jsonl", DATA / "cohort.states.jsonl"
PROBE_BINDING = ROOT / "probe_binding.json"
COLLECTION_BINDING = ROOT / "collection_binding.json"
BINDING = ROOT / "execution_binding.json"
PREFIX = "same_state_v1_"


def read(path):
    return json.loads(Path(path).read_text())


def hashes(paths):
    return {str(p): sha256(p) for p in paths}


def check_hashes(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen file changed: {path}")


def runtime():
    if (pwd.getpwuid(os.geteuid()).pw_name != "syx" or sys.platform != "linux"
            or sys.version_info[:2] != (3, 10) or os.environ.get("PYTHONHASHSEED") != "20261003"
            or Path(sys.executable).resolve() != Path("/usr/bin/python3.10")
            or os.environ.get("LD_PRELOAD")):
        raise ValueError("Use syx, original Linux Python3.10, fixed hash seed and no LD_PRELOAD")
    # Avoid the anomalous /usr/sbin/python3 path for every child invocation.
    os.environ.update(PATH="/usr/bin:/bin", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    import torch
    import transformers
    import peft
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    original = read("results/agent_feedback/clean_sft_v1/input_manifest.json")["versions"]
    if any(str(v).split("+")[0] != original[k] for k, v in versions.items()):
        raise ValueError("Runtime package versions differ from original training")
    return versions


def inventory():
    """Full input identity, frozen before any new-pool generation."""
    protocol, pool = read(PROTOCOL), read(POOL)
    check_hashes(protocol["inputs_sha256"])
    check_hashes(protocol["checkpoints"])
    check_hashes(pool["artifacts"])
    check_hashes({str(MODEL_ROOT / p): h for p, h in FILES.items()})
    if sha256(KB) != pool["source_sha256"]["kb.json"]:
        raise ValueError("Knowledge base changed")
    tokenizer = read("results/agent_feedback/clean_sft_v1/input_manifest.json")["continuation"]["tokenizer_files_sha256"]
    check_hashes({str(MODEL_ROOT / p): h for p, h in tokenizer.items()})
    if (sha256(POOL) != protocol["pool"]["manifest_sha256"] or len(pool["ids"]) != 2400
            or len(set(pool["ids"])) != 2400 or protocol["models"] != list(ARMS)):
        raise ValueError("Frozen pool/model identities differ")
    paths = [PROTOCOL, POOL, KB, *map(Path, pool["artifacts"]),
             *map(Path, protocol["checkpoints"]), *map(Path, protocol["inputs_sha256"]),
             *Path("experiments/agent_feedback").glob("*.py"),
             *Path("experiments/condition_consistency").glob("*.py"),
             Path("external/kqa_pro_baselines/utils/value_class.py"),
             Path("results/agent_feedback/clean_sft_v1/input_manifest.json"),
             *[p for p in MODEL_ROOT.iterdir() if p.suffix in {".json", ".txt", ".safetensors"}]]
    return hashes(sorted(set(paths)))


def idle_gpu(gpu):
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError("Invalid GPU index")
    cmd = ["/usr/bin/nvidia-smi", "-i", str(gpu)]
    health = ET.fromstring(subprocess.check_output(cmd + ["-q", "-x"], text=True))
    entries = health.findall("gpu")
    if len(entries) != 1 or entries[0].findtext("gpu_recovery_action") in {
            "Reset", "Reboot", "Drain P2P", "Drain and Reset"}:
        raise RuntimeError("GPU health is unresolved")
    processes = subprocess.check_output(cmd + ["--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    info = subprocess.check_output(cmd + ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
    values = [x.strip() for x in info.strip().split(",")]
    if (processes or len(values) != 2 or not all(x.isdigit() for x in values)
            or int(values[0]) > 5 or int(values[1]) > 128):
        raise RuntimeError("GPU is occupied or ownership is unresolved; no launch or reset")


def stage_tasks(stage):
    if stage == "probe":
        return [{"name": PREFIX + "probe", "cap_seconds": 300, "output": ROOT / "probe.jsonl",
                 "args": ["experiments.agent_feedback.same_state_inference", "--binding", str(PROBE_BINDING),
                          "--states", str(PROBE_STATES), "--arm", "A_20261003", "--output", str(ROOT / "probe.jsonl")]}]
    if stage == "collect":
        tasks = []
        for i, offset in enumerate((0, 1200)):
            output = ROOT / f"collector_{i}.jsonl"
            tasks.append({"name": PREFIX + f"collect_{i}", "cap_seconds": 5400, "output": output,
                "args": ["experiments.agent_feedback.same_state_runs", "collector-child", "--shard", str(i)]})
        return tasks
    if stage == "resume":
        return [{"name": PREFIX + arm, "cap_seconds": 5400, "output": ROOT / f"{arm}.jsonl",
                 "args": ["experiments.agent_feedback.same_state_inference", "--binding", str(BINDING),
                          "--states", str(STATES), "--arm", arm, "--output", str(ROOT / f"{arm}.jsonl")]} for arm in ARMS]
    raise ValueError("Unknown GPU stage")


def collector_config(shard):
    if type(shard) is not int or shard not in (0, 1):
        raise ValueError("Exactly two registered collection shards")
    return {"command": "generate", "model": str(MODEL_ROOT),
            "adapter": "results/agent_feedback/initial_sft_v1/model",
            "questions": str(QUESTIONS), "output": str(ROOT / f"collector_{shard}.jsonl"),
            "kb": str(KB), "batch_size": 8, "limit": 1200, "offset": 1200 * shard,
            "seed": 20261003, "max_context": 8192, "max_generated": 2048, "program": False}


def check_binding(path, versions):
    binding = read(path)
    schema = "same_state_collection_binding_v1" if path == COLLECTION_BINDING else "same_state_execution_binding_v1"
    if (binding.get("schema") != schema or binding.get("ready") is not True or binding.get("versions") != versions
            or binding.get("protocol_sha256") != sha256(PROTOCOL)):
        raise ValueError("Binding is not ready or runtime/protocol identity changed")
    check_hashes(binding["inputs_sha256"])
    return binding


def probe_prepare(versions):
    if PROBE_BINDING.exists() or PROBE_STATES.exists():
        raise FileExistsError("Probe already prepared; inspect its existing binding")
    inputs = inventory()
    replay_path, tokens_path = ROOT / "linux_dev_replay_preflight.json", ROOT / "linux_dev_token_preflight.json"
    replay, tokens = read(replay_path), read(tokens_path)
    dev_states = Path(replay["state_file"])
    check_hashes(replay["inputs_sha256"])
    check_hashes(tokens["inputs_sha256"])
    if (tokens.get("all_within_budgets") is not True
            or tokens["implementation_sha256"] != sha256("experiments/agent_feedback/same_state.py")
            or replay["code_sha256"] != tokens["implementation_sha256"]):
        raise ValueError("Development checks use different construction code or failed budgets")
    if sha256(dev_states) != replay["state_file_sha256"] or tokens["inputs_sha256"][str(dev_states)] != sha256(dev_states):
        raise ValueError("Original Linux development states differ")
    states = read_jsonl(dev_states)
    validate_inventory(states)
    from experiments.condition_consistency.executor import KoPLExecutor
    executor = KoPLExecutor(KB)
    for row in states:
        restore(executor, row)
    # First pair in the already fixed original dev order, no model-result selection.
    selected = states[:2]
    checks = tokens["checks"][:2]
    if ([(r["id"], r["condition"]) for r in selected] != [(r["id"], r["condition"]) for r in checks]
            or not all(r["within_budgets"] for r in checks)):
        raise ValueError("Probe token preflight differs")
    write_new(PROBE_STATES, selected, True)
    inputs.update(hashes([replay_path, tokens_path, dev_states, PROBE_STATES]))
    write_new(PROBE_BINDING, resume_binding(inputs, versions, PROBE_STATES,
        {selected[0]["id"]: {r["condition"]: r["initial_prompt_ids_sha256"] for r in checks}}, "old_dev_probe"))


def resume_binding(inputs, versions, states_path, prompt_hashes, phase):
    return {"schema": "same_state_execution_binding_v1", "ready": True, "phase": phase,
            "protocol_path": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
            "states_sha256": sha256(states_path), "state_replay_passed": True, "token_budget_passed": True,
            "inference_sha256": sha256("experiments/agent_feedback/same_state_inference.py"),
            "inputs_sha256": inputs, "versions": versions, "initial_prompt_ids_sha256": prompt_hashes}


def validate_job(task):
    path = Path("results/agent_feedback/jobs") / f"{task['name']}.json"
    job = read(path)
    if (job.get("status") != "completed" or job.get("returncode") != 0 or job.get("finished_at") is None
            or job.get("user") != "syx" or len(job.get("gpus", [])) != 1
            or job.get("stop_reason") or job.get("gpu_seconds", 0) <= 0
            or job["gpu_seconds"] > task["cap_seconds"]):
        raise ValueError("Job is incomplete, failed, over budget or has wrong ownership")
    expected = [sys.executable, "-u", "-m", *task["args"]]
    if job.get("command") != expected:
        raise ValueError("Job command differs from the fixed stage")
    return path


def launch(stage, gpus, versions):
    tasks = stage_tasks(stage)
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in range(4) for g in gpus):
        raise ValueError("Provide distinct GPU indices from 0 through 3")
    binding_path = {"probe": PROBE_BINDING, "collect": COLLECTION_BINDING, "resume": BINDING}[stage]
    check_binding(binding_path, versions)
    marker, completed = ROOT / f"{stage}.started.json", ROOT / f"{stage}.jobs_complete.json"
    paths = [marker, completed]
    for task in tasks:
        paths.extend([task["output"], task["output"].with_suffix(".runtime.json"),
                      Path("results/agent_feedback/jobs") / f"{task['name']}.json"])
    if any(p.exists() for p in paths):
        raise FileExistsError("Stage already started or has artifacts; inspect, never restart automatically")
    for gpu in gpus:
        idle_gpu(gpu)
    reserved = sum(t["cap_seconds"] for t in tasks)
    if total_gpu_seconds(time.time()) + reserved > 72 * 3600:
        raise RuntimeError("Insufficient remaining project GPU budget")
    write_new(marker, {"stage": stage, "started_at": time.time(), "binding_sha256": sha256(binding_path),
                       "gpus": gpus, "job_names": [t["name"] for t in tasks], "no_retry": True})

    def run(task, gpu):
        idle_gpu(gpu)
        # Existing supervisor polls every 10 s and allows 20 s termination grace.
        # Reserve 35 s within each registered cap; charge actual wall time.
        command = [sys.executable, "-u", "-m", "experiments.agent_feedback.run_job", "--name", task["name"],
                   "--gpus", str(gpu), "--max-hours", str((task["cap_seconds"] - 35) / 3600),
                   "--", sys.executable, "-u", "-m", *task["args"]]
        subprocess.run(command, check=True)
        return validate_job(task)

    evidence = []
    # One to four GPUs supported without changing question order or batch size.
    for start in range(0, len(tasks), len(gpus)):
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run, t, gpu) for t, gpu in zip(tasks[start:start+len(gpus)], gpus)]
            for future in futures:
                evidence.append(future.result())
    evidence.extend(p for t in tasks for p in (t["output"], t["output"].with_suffix(".runtime.json")))
    write_new(completed, {"stage": stage, "status": "completed", "binding_sha256": sha256(binding_path),
                           "inputs_sha256": hashes(evidence), "gpu_seconds": sum(read(validate_job(t))["gpu_seconds"] for t in tasks)})


def validate_stage(stage, binding):
    path = ROOT / f"{stage}.jobs_complete.json"
    record = read(path)
    if record.get("status") != "completed" or record.get("binding_sha256") != sha256(binding):
        raise ValueError("Stage lacks terminal evidence for the current binding")
    check_hashes(record["inputs_sha256"])
    for task in stage_tasks(stage):
        validate_job(task)
    return path


def freeze_collection(versions):
    if COLLECTION_BINDING.exists():
        raise FileExistsError("Collection binding already exists")
    binding = check_binding(PROBE_BINDING, versions)
    stage = validate_stage("probe", PROBE_BINDING)
    path = ROOT / "probe.jsonl"
    meta = read(path.with_suffix(".runtime.json"))
    if (meta.get("status") != "completed" or meta.get("binding_sha256") != sha256(PROBE_BINDING)
            or meta.get("predictions_sha256") != sha256(path) or meta.get("arm") != "A_20261003"
            or meta.get("states") != 2 or meta.get("versions") != versions):
        raise ValueError("Old-dev probe runtime is incomplete or changed")
    from experiments.condition_consistency.executor import KoPLExecutor
    rows, states = read_jsonl(path), read_jsonl(PROBE_STATES)
    validate_resumes(KoPLExecutor(KB), states, {"A_20261003": rows}, probe=True)
    if any(r["continuation_calls"] < 1 or r["generated_tokens"] < 1 or r["input_tokens"] < 1 for r in rows):
        raise ValueError("GPU probe did not actually generate and execute both continuations")
    inputs = dict(binding["inputs_sha256"])
    inputs.update(hashes([PROBE_BINDING, stage, path, path.with_suffix(".runtime.json")]))
    write_new(COLLECTION_BINDING, {"schema": "same_state_collection_binding_v1", "ready": True,
        "protocol_sha256": sha256(PROTOCOL), "versions": versions, "inputs_sha256": inputs,
        "old_dev_gpu_probe_replay_passed": True, "shards": [collector_config(i) for i in (0, 1)],
        "analysis_frozen_before_new_pool_inference": True})


def collector_child(shard, versions):
    binding = check_binding(COLLECTION_BINDING, versions)
    config = collector_config(shard)
    if binding.get("shards") != [collector_config(i) for i in (0, 1)]:
        raise ValueError("Collector configuration differs from its binding")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1", "2", "3"):
        raise ValueError("Collector needs exactly one registered GPU")
    idle_gpu(int(gpu))
    from types import SimpleNamespace
    inference.generate(SimpleNamespace(**config))


def validate_collection(rows_by_shard, runtimes, pool_ids):
    if len(rows_by_shard) != 2 or len(runtimes) != 2 or len(pool_ids) != 2400 or len(set(pool_ids)) != 2400:
        raise ValueError("Complete fixed 2400-question collection is required")
    combined = []
    for shard, (rows, meta) in enumerate(zip(rows_by_shard, runtimes)):
        if (meta.get("status") != "completed" or meta.get("config") != collector_config(shard)
                or meta.get("selected_questions") != 1200 or meta.get("source_questions") != 2400
                or meta.get("totals", {}).get("questions") != 1200
                or [r["id"] for r in rows] != pool_ids[shard*1200:(shard+1)*1200]):
            raise ValueError("Missing, reordered or differently configured collection shard")
        for row in rows:
            validate_prediction(row, "fixed I")
            if row["calls"] > 24 or row["generated_tokens"] > 2048:
                raise ValueError("Collector exceeded its action/generation budget")
        combined.extend(rows)
    return combined


def cohort(versions):
    if BINDING.exists() or STATES.exists() or (ROOT / "cohort.json").exists():
        raise FileExistsError("Cohort already exists or is partial; do not resample")
    binding = check_binding(COLLECTION_BINDING, versions)
    stage = validate_stage("collect", COLLECTION_BINDING)
    paths = [ROOT / f"collector_{i}.jsonl" for i in (0, 1)]
    runtimes = [p.with_suffix(".runtime.json") for p in paths]
    pool = read(POOL)
    rows = validate_collection([read_jsonl(p) for p in paths], [read(p) for p in runtimes], pool["ids"])
    gold = read_jsonl(GOLD)
    if [r["id"] for r in gold] != pool["ids"]:
        raise ValueError("Gold order differs from the frozen pool")
    from experiments.condition_consistency.executor import KoPLExecutor
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    states, ledger, report = construct_cohort(KoPLExecutor(KB), tokenizer, rows, gold)
    write_new(STATES, states, True)
    ledger_path = DATA / "cohort.ledger.jsonl"
    write_new(ledger_path, ledger, True)
    inputs = dict(binding["inputs_sha256"])
    inputs.update(hashes([COLLECTION_BINDING, stage, *paths, *runtimes, STATES, ledger_path]))
    report["inputs_sha256"] = inputs
    report["all_source_rollouts_replayed"] = all("free_run_failed" in row["construction"] for row in ledger)
    write_new(ROOT / "cohort.json", report)
    if not report["all_source_rollouts_replayed"]:
        raise ValueError("Source rollout replay failed; ledger retained but no evaluation binding issued")
    if not report["evaluation_possible"]:
        print("NO_ELIGIBLE_STATES; no evaluation or automatic pool expansion", flush=True)
        return
    inputs.update(hashes([ROOT / "cohort.json"]))
    write_new(BINDING, resume_binding(inputs, versions, STATES, report["initial_prompt_ids_sha256"], "fixed_new_pool"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-probe", "probe", "freeze-collection", "collect",
                                         "collector-child", "cohort", "resume", "analyze"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    versions = runtime()
    if args.stage == "prepare-probe":
        probe_prepare(versions)
    elif args.stage in ("probe", "collect", "resume"):
        launch(args.stage, [int(g) for g in args.gpus.split(",")], versions)
    elif args.stage == "freeze-collection":
        freeze_collection(versions)
    elif args.stage == "collector-child":
        collector_child(args.shard, versions)
    elif args.stage == "cohort":
        cohort(versions)
    else:
        check_binding(BINDING, versions)
        validate_stage("resume", BINDING)
        subprocess.run([sys.executable, "-m", "experiments.agent_feedback.same_state_analysis",
            "--binding", str(BINDING), "--states", str(STATES), "--predictions-dir", str(ROOT),
            "--output-dir", str(ROOT / "analysis")], check=True)


if __name__ == "__main__":
    main()
