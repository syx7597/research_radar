from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback import same_state_runs as runs


class SameStateRunsTest(unittest.TestCase):
    def collection(self):
        ids = [f"question:{i}" for i in range(2400)]
        rows = [[{"id": ident, "prediction": None, "stop_reason": "call_budget", "calls": 24,
                  "invalid_calls": 0, "input_tokens": 100, "generated_tokens": 100, "total_tokens": 200}
                 for ident in ids[i*1200:(i+1)*1200]] for i in (0, 1)]
        meta = [{"status": "completed", "config": runs.collector_config(i), "selected_questions": 1200,
                 "source_questions": 2400, "totals": {"questions": 1200}} for i in (0, 1)]
        return rows, meta, ids

    def test_full_failed_collection_kept(self):
        rows, meta, ids = self.collection()
        combined = runs.validate_collection(rows, meta, ids)
        self.assertEqual([r["id"] for r in combined], ids)
        self.assertTrue(all(r["prediction"] is None for r in combined))

    def test_partial_duplicate_reordered_collection_rejected(self):
        rows, meta, ids = self.collection()
        for replacement in (rows[1][:-1], rows[1][::-1], rows[0]):
            with self.assertRaises(ValueError):
                runs.validate_collection([rows[0], replacement], meta, ids)

    def test_changed_decoding_or_nonterminal_runtime_rejected(self):
        rows, meta, ids = self.collection()
        for key, value in (("batch_size", 1), ("max_generated", 4096), ("offset", 1), ("seed", 42)):
            bad = deepcopy(meta); bad[0]["config"][key] = value
            with self.assertRaises(ValueError):
                runs.validate_collection(rows, bad, ids)
        bad = deepcopy(meta); bad[0]["status"] = "running"
        with self.assertRaises(ValueError):
            runs.validate_collection(rows, bad, ids)

    def test_fixed_jobs_have_unique_names_and_registered_total_cap(self):
        tasks = sum([runs.stage_tasks(s) for s in ("probe", "collect", "resume")], [])
        self.assertEqual(len(tasks), 7)
        self.assertEqual(len({t["name"] for t in tasks}), 7)
        self.assertEqual(sum(t["cap_seconds"] for t in tasks), 32700)
        self.assertFalse(any("train" in str(t["args"]) for t in tasks))
        self.assertEqual(runs.collector_config(1)["offset"], 1200)

    def test_unresolved_occupancy_is_rejected(self):
        health = '<nvidia_smi_log><gpu id="x"><gpu_recovery_action>None</gpu_recovery_action></gpu></nvidia_smi_log>'
        for pid, counters in (("4280\n", "100, 20020\n"), ("[N/A]\n", "0, 0\n"), ("", "99, 20020\n")):
            with patch.object(runs.subprocess, "check_output", side_effect=[health, pid, counters]):
                with self.assertRaises(RuntimeError):
                    runs.idle_gpu(0)
        with patch.object(runs.subprocess, "check_output", side_effect=[health, "", "0, 0\n"]):
            runs.idle_gpu(0)

    def test_busy_gate_leaves_no_started_marker_and_launches_nothing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runs, "ROOT", Path(folder)), \
             patch.object(runs, "check_binding"), patch.object(runs, "idle_gpu", side_effect=RuntimeError("busy")), \
             patch.object(runs.subprocess, "run") as launch:
            with self.assertRaises(RuntimeError):
                runs.launch("collect", [0, 1], {})
            launch.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_started_marker_blocks_restart_even_without_terminal_output(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runs, "ROOT", Path(folder)), \
             patch.object(runs, "check_binding"), patch.object(runs, "idle_gpu") as idle, \
             patch.object(runs.subprocess, "run") as launch:
            (Path(folder) / "collect.started.json").write_text('{"status":"running"}')
            with self.assertRaises(FileExistsError):
                runs.launch("collect", [0], {})
            idle.assert_not_called(); launch.assert_not_called()

    def test_changed_bound_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text('first')
            inventory = runs.hashes([path])
            runs.check_hashes(inventory)
            path.write_text('second')
            with self.assertRaises(ValueError):
                runs.check_hashes(inventory)

    def test_failed_or_overbudget_job_cannot_authorize_next_stage(self):
        task = runs.stage_tasks("collect")[0]
        base = {"status": "completed", "returncode": 0, "finished_at": 123, "user": "syx",
                "gpus": [0], "gpu_seconds": 100,
                "command": [runs.sys.executable, "-u", "-m", *task["args"]]}
        with patch.object(runs, "read", return_value=base):
            runs.validate_job(task)
        for key, value in (("status", "running"), ("returncode", 1), ("user", "ubuntu"),
                           ("finished_at", None), ("gpu_seconds", 5401), ("command", ["other"]),
                           ("stop_reason", "budget_limit")):
            with patch.object(runs, "read", return_value=dict(base, **{key: value})):
                with self.assertRaises(ValueError):
                    runs.validate_job(task)


if __name__ == "__main__":
    unittest.main()
