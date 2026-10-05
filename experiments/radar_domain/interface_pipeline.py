"""One fixed per-adapter diagnostic/train/diagnostic sequence on idle GPUs.

Every GPU command runs under the existing 72 GPU-hour supervisor. The controller
does not inspect scores, retry failures, or choose checkpoints. It runs as syx.
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
import time

from . import interface_training as training
from . import interface_probe as probe
from experiments.agent_feedback.run_job import total_gpu_seconds

OUT = training.OUT
SCHEDULE = {0: ["A1", "P"], 1: ["C1"], 2: ["A2"], 3: ["C2"]}


def main():
    if pwd.getpwuid(os.geteuid()).pw_name != "syx":
        raise PermissionError("GPU experiment must run only under the authorized syx account")
    if (OUT / "pipeline_started.json").exists():
        raise FileExistsError("This pipeline was already launched; inspect it, never duplicate")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if processes:
        raise RuntimeError("GPUs have compute processes; no experiment launched")
    used = total_gpu_seconds(time.time()) / 3600
    tp = training.verify("P")
    ep, _ = probe.verify("P", "before")
    reserved = tp["maximum_additional_gpu_hours"] + ep["maximum_additional_gpu_hours"]
    if used + reserved > 72:
        raise RuntimeError("Insufficient shared GPU budget for the complete bounded sequence")
    jobs = []
    for gpu, labels in SCHEDULE.items():
        for label in labels:
            training.verify(label)
            for stage in ("before", "train", "after"):
                name = f"radar_interface_v1_{label}_{stage}"
                if (Path("results/agent_feedback/jobs") / f"{name}.json").exists():
                    raise FileExistsError("An experiment job already exists")
                hours = tp["maximum_hours_per_run"] if stage == "train" else ep["maximum_hours_per_run"]
                command = [sys.executable, "-m", "experiments.agent_feedback.run_job", "--name", name,
                           "--gpus", str(gpu), "--max-hours", str(hours), "--", sys.executable, "-m"]
                command += (["experiments.radar_domain.interface_training", "train", "--label", label]
                            if stage == "train" else
                            ["experiments.radar_domain.interface_probe", "generate", "--label", label, "--phase", stage])
                jobs.append({"gpu": gpu, "label": label, "stage": stage, "name": name, "command": command})
    probe.write_new(OUT / "pipeline_started.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "user": "syx",
        "training_protocol_sha256": probe.sha(training.PROTOCOL), "evaluation_protocol_sha256": probe.sha(probe.PROTOCOL),
        "gpu_hours_before": used, "maximum_additional_gpu_hours": reserved, "jobs": jobs,
        "policy": "Fixed per-GPU order; stage failure stops that lane, no retries or score inspection. Other lanes retain their authorized work."})

    def lane(gpu):
        results = []
        for job in [item for item in jobs if item["gpu"] == gpu]:
            result = subprocess.run(job["command"], check=False)
            results.append({"name": job["name"], "returncode": result.returncode})
            if result.returncode:
                break
        probe.write_new(OUT / f"pipeline_gpu{gpu}.json", {"gpu": gpu, "results": results,
                        "status": "completed" if len(results) == len([j for j in jobs if j["gpu"] == gpu]) and
                                  all(r["returncode"] == 0 for r in results) else "failed"})
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lane, SCHEDULE))
    completed = sum(len(lane_results) for lane_results in results)
    okay = completed == len(jobs) and all(row["returncode"] == 0 for lane_results in results for row in lane_results)
    probe.write_new(OUT / "pipeline_completed.json", {"status": "completed" if okay else "failed",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(), "jobs_returned": completed,
                    "gpu_hours_after": total_gpu_seconds(time.time()) / 3600, "results": results})
    if not okay:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
