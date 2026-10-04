"""Diagnostic selection is independent of outcomes and cannot duplicate work."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback import rl_signal_runs as runs


class SignalControllerTest(unittest.TestCase):
    def test_selection_is_order_and_outcome_independent(self):
        questions = [{"id": f"train:{i}", "question": "Q"} for i in range(30)]
        eligible = [{"id": r["id"], "answer": "unused"} for r in questions]
        chosen = runs.select_ids(questions, eligible)
        self.assertEqual(len(chosen), 16)
        changed = [{"id": r["id"], "answer": "different", "correct": False}
                   for r in reversed(eligible)]
        self.assertEqual(chosen, runs.select_ids(list(reversed(questions)), changed))

    def test_selection_rejects_duplicate_or_nontrain_ids(self):
        questions = [{"id": f"train:{i}"} for i in range(20)]
        for eligible in [questions + questions[:1], questions + [{"id": "dev:0"}], questions[:2]]:
            with self.subTest(eligible=eligible), self.assertRaises(ValueError):
                runs.select_ids(questions, eligible)

    def test_failed_review_cannot_reach_gpu_checks(self):
        with patch.object(runs, "require_new"), patch.object(runs, "read", return_value={
                "gate": {"conditional_signal_check_eligible": False}}), \
                patch.object(runs, "require_idle_gpus") as gpu:
            with self.assertRaises(ValueError):
                runs.freeze()
            gpu.assert_not_called()

    def test_existing_protocol_prevents_relaunch(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                runs.PROTOCOL.parent.mkdir(parents=True)
                runs.PROTOCOL.write_text("{}")
                with self.assertRaises(FileExistsError):
                    runs.require_new()
            finally:
                os.chdir(previous)

    def test_command_is_diagnostic_without_formal_updates(self):
        with patch.object(runs, "read", return_value={"CPATH": "/private/headers"}):
            command = runs.command("clean_sft_v1", "clean_rl_signal_smoke_v1")
        self.assertEqual(command[command.index("--smoke-steps") + 1], "2")
        self.assertEqual(command[command.index("--accumulation") + 1], "32")
        self.assertEqual(command[command.index("--eligible-ids") + 1], str(runs.IDS))
        self.assertNotIn("--max-steps", command)
        self.assertEqual(runs.COUNT * 4, 1 * runs.ACCUMULATION * 2)


if __name__ == "__main__":
    unittest.main()
