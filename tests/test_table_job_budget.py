import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain.table_job_budget import JobBudget, BudgetExceeded, require_round_ledger, ROUND_LEDGER_PATH


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "fixture_budget.json"

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, name, cap=1, **kw):
        return JobBudget(ledger_path=self.path, job_id=name, phase="training", max_gpu_hours=cap, **kw)

    def test_running_reservations_cannot_be_spent_twice(self):
        with self.job("A", 5):
            with self.assertRaises(BudgetExceeded):
                with self.job("C", 4):
                    self.fail("overlapping reservations admitted")
        jobs = json.loads(self.path.read_text())["jobs"]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["status"], "completed")

    def test_failed_attempt_is_charged_and_job_id_not_reused(self):
        with self.assertRaises(RuntimeError):
            with self.job("failed"):
                raise RuntimeError("fixture failure")
        job = json.loads(self.path.read_text())["jobs"][0]
        self.assertEqual(job["status"], "failed")
        self.assertGreaterEqual(job["gpu_hours"], 0)
        with self.assertRaises(ValueError):
            with self.job("failed"):
                pass

    def test_card_count_changes_wall_deadline_and_charge(self):
        with patch("experiments.radar_domain.table_job_budget.threading.Thread"):
            with patch("experiments.radar_domain.table_job_budget.time.monotonic", return_value=100):
                with self.job("two_cards", 2, gpu_count=2) as job:
                    self.assertEqual(job.deadline_monotonic, 3700)
                    with patch("experiments.radar_domain.table_job_budget.time.monotonic", return_value=280):
                        job._finish("completed")
        recorded = json.loads(self.path.read_text())["jobs"][0]
        self.assertEqual(recorded["elapsed_seconds"], 180)
        self.assertAlmostEqual(recorded["gpu_hours"], .1)

    def test_deadline_check_and_watchdog_preserve_timeout(self):
        with patch("experiments.radar_domain.table_job_budget.threading.Thread"):
            with self.job("timeout") as job:
                job.deadline_monotonic = 0
                with self.assertRaises(BudgetExceeded):
                    job.check()
                with patch("experiments.radar_domain.table_job_budget.os._exit") as terminate:
                    job._watch()
                    terminate.assert_called_once_with(124)
                job.deadline_monotonic = 10**20
        recorded = json.loads(self.path.read_text())["jobs"][0]
        self.assertEqual(recorded["status"], "timed_out")

    def test_invalid_limits_and_tampered_ledger_are_rejected(self):
        for cap in (float("nan"), float("inf"), 0, -1, 9, True):
            with self.assertRaises(ValueError):
                self.job("invalid", cap)
        with self.job("first"):
            pass
        ledger = json.loads(self.path.read_text()); ledger["round_cap_gpu_hours"] = 100
        self.path.write_text(json.dumps(ledger))
        with self.assertRaises(ValueError):
            with self.job("second"):
                pass

    def test_production_cannot_switch_to_a_fresh_budget_ledger(self):
        self.assertEqual(require_round_ledger(ROUND_LEDGER_PATH.resolve()), ROUND_LEDGER_PATH)
        with self.assertRaises(ValueError):
            require_round_ledger(self.path)


if __name__ == "__main__":
    unittest.main()
