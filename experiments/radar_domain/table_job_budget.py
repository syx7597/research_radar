"""Shared GPU allocation-time budget for the first domain experiment round.

Reservations prevent parallel jobs from spending the same remaining allocation.
The watchdog records timeout and exits only its own process, including when model
code is blocked. CPU preparation never enters this context manager.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time

ROUND_CAP_GPU_HOURS = 8.0
PRIOR_PROJECT_GPU_HOURS = 24.192471
PROJECT_CAP_GPU_HOURS = 72.0
ROUND_LEDGER_PATH = Path("results/radar_domain/table_runtime_v1/gpu_budget.json")


def require_round_ledger(path):
    """Production entry points must share this round's one allocation ledger."""
    if Path(path).resolve() != ROUND_LEDGER_PATH.resolve():
        raise ValueError("Use the registered shared domain-round GPU budget ledger")
    return ROUND_LEDGER_PATH


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BudgetExceeded(RuntimeError):
    pass


class JobBudget:
    def __init__(self, *, ledger_path, job_id, phase, max_gpu_hours, gpu_count=1):
        if (not isinstance(job_id, str) or not job_id.strip() or phase not in {"training", "generation", "probe"}
                or type(gpu_count) is not int or not 1 <= gpu_count <= 4
                or type(max_gpu_hours) not in {int, float} or not math.isfinite(max_gpu_hours)
                or not 0 < max_gpu_hours <= ROUND_CAP_GPU_HOURS):
            raise ValueError("Explicit job, phase and finite GPU allocation required")
        self.path, self.job_id, self.phase = Path(ledger_path), job_id, phase
        self.max_gpu_hours, self.gpu_count = float(max_gpu_hours), gpu_count
        self.started = None
        self.deadline_monotonic = None
        self._thread = None
        self._stop = threading.Event()

    @contextmanager
    def _ledger(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if self.path.exists():
                    data = json.loads(self.path.read_text())
                    if (data.get("schema") != "radar_domain_gpu_budget_v1"
                            or data.get("round_cap_gpu_hours") != ROUND_CAP_GPU_HOURS
                            or data.get("prior_project_gpu_hours") != PRIOR_PROJECT_GPU_HOURS
                            or data.get("project_cap_gpu_hours") != PROJECT_CAP_GPU_HOURS):
                        raise ValueError("Budget ledger identity differs")
                else:
                    data = {"schema": "radar_domain_gpu_budget_v1", "round_cap_gpu_hours": ROUND_CAP_GPU_HOURS,
                            "prior_project_gpu_hours": PRIOR_PROJECT_GPU_HOURS,
                            "project_cap_gpu_hours": PROJECT_CAP_GPU_HOURS, "jobs": []}
                yield data
                tmp = self.path.with_name(self.path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                os.replace(tmp, self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _charged(data):
        total = 0.0
        for job in data["jobs"]:
            if job["status"] == "running":
                charge = job["reserved_gpu_hours"]
            elif job["status"] in {"completed", "failed", "timed_out"}:
                charge = job["gpu_hours"]
            else:
                raise ValueError("Unknown job status; inspect ledger")
            if type(charge) not in {int, float} or not math.isfinite(charge) or charge < 0:
                raise ValueError("Invalid recorded allocation")
            total += charge
        return total

    def __enter__(self):
        if self.started is not None:
            raise ValueError("JobBudget cannot be reused")
        with self._ledger() as ledger:
            if any(j["job_id"] == self.job_id for j in ledger["jobs"]):
                raise ValueError("Job ID already recorded; preserve the earlier attempt")
            charged = self._charged(ledger)
            if (charged + self.max_gpu_hours > ROUND_CAP_GPU_HOURS + 1e-12
                    or PRIOR_PROJECT_GPU_HOURS + charged + self.max_gpu_hours > PROJECT_CAP_GPU_HOURS):
                raise BudgetExceeded("Round allocation exhausted or reserved by other jobs")
            self.started = time.monotonic()
            self.deadline_monotonic = self.started + self.max_gpu_hours * 3600 / self.gpu_count
            ledger["jobs"].append({"job_id": self.job_id, "phase": self.phase, "pid": os.getpid(),
                "status": "running", "started_at_utc": utc_now(), "gpu_count": self.gpu_count,
                "reserved_gpu_hours": self.max_gpu_hours,
                "accounting": "wall-clock allocation including loading, model smoke, compute and saving"})
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    @property
    def remaining_seconds(self):
        if self.started is None:
            raise ValueError("Job has not entered its budget context")
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def check(self):
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("Job allocation-time limit reached")

    def _finish(self, status):
        elapsed = max(0.0, time.monotonic() - self.started)
        with self._ledger() as ledger:
            job = next(j for j in ledger["jobs"] if j["job_id"] == self.job_id)
            if job["status"] == "running":
                job.update(status=status, finished_at_utc=utc_now(), elapsed_seconds=elapsed,
                           gpu_hours=elapsed * self.gpu_count / 3600)

    def _watch(self):
        if not self._stop.wait(self.remaining_seconds):
            # Persist timeout before ending this process; no foreign PID is touched.
            try:
                self._finish("timed_out")
            finally:
                os._exit(124)

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        timeout = isinstance(exc_value, BudgetExceeded) or self.remaining_seconds <= 0
        self._finish("timed_out" if timeout else "failed" if exc_type else "completed")
        if timeout and exc_type is None:
            raise BudgetExceeded("Job finished after its allocation-time limit")
        return False
