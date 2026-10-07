"""Detach a single frozen three-arm run only after remote CPU checks pass."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from . import coverage_evaluate as evaluation
from .source_readings import ROOT

OUTPUT = ROOT / "data/radar_sources_v2/evaluation_launch_v1"


def launch(protocol_path, protocol_sha):
    evaluation.runtime.identity()
    protocol = evaluation.load_protocol(protocol_path, protocol_sha)
    OUTPUT.mkdir(parents=True, exist_ok=False)  # One pipeline slot, never retry silently.
    try:
        evaluation.require(evaluation.environment() == protocol["runtime_environment"], "Remote environment drift")
        with (OUTPUT / "CPU_tests.log").open("x") as log:
            subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                            "-p", "test_radar_coverage*.py"], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        tokenizer, unused = evaluation.runtime.tokenizer_precheck(protocol["tokenizer_path"], protocol["model_identity"])
        counts = {arm: len(evaluation.tokenize_rows(tokenizer, evaluation.load_inputs(protocol, arm)))
                  for arm in evaluation.ARMS}
        # No allocation yet. Each arm repeats this check immediately before its
        # own GPU child; this pipeline never evicts existing processes.
        devices = {arm: evaluation.runtime.idle_gpu(protocol["execution"]["gpu_by_arm"][arm])
                   for arm in evaluation.ARMS}
        evaluation.write_new(OUTPUT / "cpu_passed.json", {
            "protocol_sha256": protocol_sha, "token_ids_verified": counts,
            "devices": devices, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_sha256": evaluation.digest(Path(__file__)), "GPU_used": False})
        launched = []
        for arm in evaluation.ARMS:
            output = evaluation.path_in_root(protocol["execution"]["output_root"]) / arm
            evaluation.require(not output.exists(), "An arm slot is already consumed; no retry")
        for arm in evaluation.ARMS:
            output = evaluation.path_in_root(protocol["execution"]["output_root"]) / arm
            command = [sys.executable, "-B", "-m", "experiments.radar_domain.coverage_evaluate",
                       "--protocol", str(Path(protocol_path).resolve()), "--protocol-sha256", protocol_sha,
                       "--arm", arm, "--gpu", protocol["execution"]["gpu_by_arm"][arm], "--output", str(output)]
            with (OUTPUT / f"{arm}.supervisor.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row = {"arm": arm, "supervisor_pid": process.pid, "command": command,
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
            evaluation.write_new(OUTPUT / f"{arm}.launched.json", row)
            launched.append(row)
        result = {"status": "launched", "protocol_sha256": protocol_sha, "arms": launched,
                  "maximum_reserved_GPU_seconds": 7200, "automatic_retries": 0,
                  "meaning": "Supervisors started; this is not a completion or correctness result."}
        evaluation.write_new(OUTPUT / "launched.json", result)
        return result
    except BaseException as exc:
        evaluation.write_new(OUTPUT / "failed.json", {"status": "failed", "error": str(exc),
                              "traceback": traceback.format_exc(), "automatic_retries": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = launch(args.protocol, args.protocol_sha256)
    print(json.dumps({"status": result["status"], "arms": len(result["arms"])}))


if __name__ == "__main__":
    main()
