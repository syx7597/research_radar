"""Own-process GPU job accounting and a shared 72 GPU-hour execution cap.

Run as a detached supervisor, not as an untracked shell background command.
Each job records its child PID, live heartbeat, exact command and terminal code.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

ROOT = Path("results/agent_feedback/jobs")


def save(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def total_gpu_seconds(now):
    total = 0.0
    for path in ROOT.glob("*.json"):
        row = json.loads(path.read_text())
        end = row.get("finished_at")
        if end is None:
            # A missing heartbeat is not evidence that the process ended.
            # Conservatively charge unresolved jobs until explicitly reconciled.
            end = now
        total += max(0, end - row["started_at"]) * len(row["gpus"])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Job name must be alphanumeric")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Missing command")
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or any(x not in range(4) for x in gpus):
        raise ValueError("Invalid GPU assignment")
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{args.name}.json"
    with (ROOT / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise FileExistsError("Job name already exists; never duplicate an unobserved job")
        now = time.time()
        if total_gpu_seconds(now) >= 72 * 3600:
            raise RuntimeError("Pilot GPU budget exhausted")
        record = {"name": args.name, "supervisor_pid": os.getpid(), "child_pid": None,
                  "user": pwd.getpwuid(os.geteuid()).pw_name, "uid": os.geteuid(),
                  "gpus": gpus, "command": command, "started_at": now,
                  "heartbeat_at": now, "status": "starting", "finished_at": None}
        save(path, record)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PYTHONHASHSEED="20261003",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               WANDB_DISABLED="true")
    child = None
    try:
        with (ROOT / f"{args.name}.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record.update(child_pid=child.pid, status="running")
            save(path, record)
            while child.poll() is None:
                now = time.time()
                record["heartbeat_at"] = now
                save(path, record)
                if (now - record["started_at"] >= args.max_hours * 3600 or
                        total_gpu_seconds(now) >= 72 * 3600):
                    record["stop_reason"] = "budget_limit"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait()
        record.update(status="completed" if code == 0 else "failed", returncode=code)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=20)
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_at"] = time.time()
        record["gpu_seconds"] = (record["finished_at"] - record["started_at"]) * len(gpus)
        save(path, record)
        print(json.dumps(record), flush=True)
    sys.exit(record.get("returncode", 1))


if __name__ == "__main__":
    main()
