"""One fixed selection stage followed by common-answer generation on idle GPUs.

Every GPU command uses the shared 72 GPU-hour run_job supervisor. A failure
stops new work in its stage; already running jobs finish, and the next stage does
not start. No retries, training, score inspection or adaptive scheduling occur.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import threading
import time

from experiments.agent_feedback.run_job import ROOT as JOB_ROOT, total_gpu_seconds
from . import source_probe as probe
from . import source_answers as answers
from .development_analysis import read_jsonl, require

SCHEDULES = {
    "select": {0: ["A1", "P"], 1: ["C1"], 2: ["A2"], 3: ["C2"]},
    "answer": {0: ["RAG", "C2"], 1: ["P"], 2: ["A1", "C1"], 3: ["A2"]},
}


def build_jobs(protocol):
    require(protocol["maximum_hours_per_selector"] == 0.35 and
            protocol["maximum_hours_per_answer_arm"] == 0.25 and
            protocol["maximum_additional_gpu_hours"] == 3.25, "Fixed source evaluation budget changed")
    jobs = []
    for stage, lanes in SCHEDULES.items():
        flattened = [label for labels in lanes.values() for label in labels]
        expected = set(probe.MODELS) if stage == "select" else set(probe.ARMS)
        require(len(flattened) == len(expected) and set(flattened) == expected, "Incomplete fixed job schedule")
        hours = protocol["maximum_hours_per_selector"] if stage == "select" else protocol["maximum_hours_per_answer_arm"]
        for gpu, labels in lanes.items():
            for label in labels:
                name = f"radar_sources_v1_{label}_{stage}"
                command = [sys.executable, "-m", "experiments.agent_feedback.run_job", "--name", name,
                           "--gpus", str(gpu), "--max-hours", str(hours), "--", sys.executable, "-m"]
                command += (["experiments.radar_domain.source_probe", "generate", "--label", label]
                            if stage == "select" else ["experiments.radar_domain.source_answers", "--arm", label])
                jobs.append({"name": name, "gpu": gpu, "stage": stage, "label": label,
                             "maximum_gpu_hours": hours, "command": command})
    require(abs(sum(job["maximum_gpu_hours"] for job in jobs) - 3.25) < 1e-12,
            "The fixed schedule no longer sums to its reserved GPU budget")
    return jobs


def ensure_idle(gpu=None):
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    if gpu is not None:
        command += ["-i", str(gpu)]
    if subprocess.check_output(command, text=True).strip():
        raise RuntimeError("A requested GPU has compute processes; do not interfere with other work")


def verify_completion(job, protocol):
    if job["stage"] == "select":
        # Replay real tools and verify checkpoint/runtime bindings without gold.
        rows, _ = answers.evidence_rows(job["label"], protocol)
        require(len(rows) == protocol["question_count"], "Incomplete selector output")
        return
    path = probe.OUT / f"{job['label']}.answers.jsonl"
    runtime = json.loads(path.with_suffix(".runtime.json").read_text())
    require(runtime["status"] == "completed" and runtime["arm"] == job["label"], "Incomplete answer arm")
    require(runtime["protocol_sha256"] == probe.sha(probe.PROTOCOL) and runtime["output_sha256"] == probe.sha(path),
            "Answer output binding failed")
    require(runtime["config"] == protocol["answer_inference"] and runtime["model"] == protocol["model"]
            and runtime["adapter"] is None, "Answer model/config changed")
    rows = read_jsonl(path)
    require([row["id"] for row in rows] == [row["id"] for row in probe.questions()], "Incomplete answer coverage")
    require(all(row["arm"] == job["label"] for row in rows), "Wrong answer arm")
    require(runtime["totals"]["questions"] == len(rows), "Answer count mismatch")


def run_stage(stage, jobs, protocol):
    failed = threading.Event()
    stage_jobs = [job for job in jobs if job["stage"] == stage]

    def lane(gpu):
        results = []
        assigned = [job for job in stage_jobs if job["gpu"] == gpu]
        for job in assigned:
            if failed.is_set():
                break
            result = {"name": job["name"], "label": job["label"], "stage": stage}
            try:
                ensure_idle(gpu)
                if failed.is_set():
                    break
                completed = subprocess.run(job["command"], check=False)
                result["returncode"] = completed.returncode
                if completed.returncode == 0:
                    verify_completion(job, protocol)
                else:
                    failed.set()
            except Exception as exc:
                result.update(returncode=1, error=f"{type(exc).__name__}: {exc}")
                failed.set()
            results.append(result)
            if result["returncode"]:
                break
        status = "completed" if len(results) == len(assigned) and all(row["returncode"] == 0 for row in results) else "failed_or_skipped"
        probe.write_new(probe.OUT / f"pipeline_{stage}_gpu{gpu}.json", {"stage": stage, "gpu": gpu,
                        "status": status, "results": results, "unstarted_jobs": len(assigned) - len(results)})
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lane, SCHEDULES[stage]))
    complete = sum(len(lane_results) for lane_results in results)
    okay = complete == len(stage_jobs) and all(row["returncode"] == 0 for lane_results in results for row in lane_results)
    return {"stage": stage, "status": "completed" if okay else "failed", "jobs_returned": complete,
            "expected_jobs": len(stage_jobs), "results": results}


def main():
    if pwd.getpwuid(os.geteuid()).pw_name != "syx":
        raise PermissionError("Only the authorized syx account may run the GPU evaluation")
    if any(probe.OUT.glob("pipeline_*.json")):
        raise FileExistsError("Controller artifacts already exist; never overwrite or automatically restart")
    ensure_idle()
    protocol = probe.verify("P")
    for label in probe.MODELS:
        require(probe.verify(label) == protocol, "Source protocol drift")
    jobs = build_jobs(protocol)
    for job in jobs:
        if (JOB_ROOT / f"{job['name']}.json").exists():
            raise FileExistsError("A fixed supervised job already exists; no retry")
        suffix = "selection" if job["stage"] == "select" else "answers"
        output = probe.OUT / f"{job['label']}.{suffix}.jsonl"
        if output.exists() or output.with_suffix(".runtime.json").exists():
            raise FileExistsError("A fixed result already exists; inspect instead of rerunning")
    used = total_gpu_seconds(time.time()) / 3600
    if used + protocol["maximum_additional_gpu_hours"] > 72:
        raise RuntimeError("Insufficient shared budget for the complete fixed evaluation")
    probe.write_new(probe.OUT / "pipeline_started.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "user": "syx",
        "protocol_sha256": probe.sha(probe.PROTOCOL), "gpu_hours_before": used,
        "maximum_additional_gpu_hours": protocol["maximum_additional_gpu_hours"], "jobs": jobs,
        "policy": "All five selectors complete before any common answer job; failed stages prevent later stages, no score-selected retry."})
    stages = []
    for stage in SCHEDULES:
        result = run_stage(stage, jobs, protocol)
        stages.append(result)
        if result["status"] != "completed":
            break
    jobs_returned = sum(stage["jobs_returned"] for stage in stages)
    okay = len(stages) == 2 and jobs_returned == len(jobs) and all(stage["status"] == "completed" for stage in stages)
    base_files_final, final_error = None, None
    if okay:
        from .source_preflight import hash_base
        try:
            base_files_final = hash_base(protocol)
        except Exception as exc:
            okay = False
            final_error = f"{type(exc).__name__}: {exc}"
    probe.write_new(probe.OUT / "pipeline_completed.json", {
        "status": "completed" if okay else "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_hours_after": total_gpu_seconds(time.time()) / 3600, "stages": stages,
        "jobs_returned": jobs_returned, "protocol_sha256": probe.sha(probe.PROTOCOL),
        "base_files_final": base_files_final, "final_error": final_error,
        "eligible_for_analysis": okay})
    if not okay:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
