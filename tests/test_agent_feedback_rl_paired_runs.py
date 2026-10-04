"""CPU-only decision, completion and scheduling gates for one formal RL pair."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback import rl_paired_runs as runs
from experiments.agent_feedback.replication_runs import inference_config
from experiments.agent_feedback.train_grpo_signal import reward_group_records, signal_summary


class PairedRLControllerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, previous)
        self.calls, self.bad_count, self.bad_order = [], False, False
        self.write(runs.DECISION, {"gate": {"allow_formal_rl": True}, "training_timeout_hours_per_arm": 4})
        self.write(runs.ROOT / "python_headers.json", {"CPATH": "/private/python-dev/headers"})
        self.snapshot = {"timeout_hours": 4, "versions": {"torch": "fixture"}, "manifest": {},
            "dev_ids": list(range(500)), "eligible_ids": [f"train:{i}" for i in range(4999)],
            "inputs_sha256": {str(runs.DECISION): runs.digest(runs.DECISION)}}
        for path in (runs.QUESTIONS, runs.GOLD, runs.ELIGIBLE, runs.NATIVE_PROTOCOL,
                     runs.CODE / "train_grpo_signal.py", runs.CODE / "train_grpo.py", runs.CODE / "environment.py"):
            self.write(path, {"fixture": str(path)})
        for _, adapter, *_ in runs.RUNS:
            self.write(runs.ROOT / adapter / "model/adapter_config.json", {"fixture": adapter})
            (runs.ROOT / adapter / "model/adapter_model.safetensors").write_bytes(adapter.encode())

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def training_output(self, adapter, name):
        output = runs.ROOT / name
        count = 199 if self.bad_count else 200
        ids = self.snapshot["eligible_ids"][:count]
        if self.bad_order and name == "recovery_grpo_v1":
            ids = list(reversed(ids))
        groups = reward_group_records([id for id in ids for _ in range(4)], [0.0] * (count * 4),
                    [0.0] * (count * 4), [10] * (count * 4), [20] * (count * 4))
        self.jsonl(output / "reward_group_signal.jsonl", groups)
        steps = [{"all_gradients_finite": True, "gradient_l2_after_clip": 0.0,
                  "learning_rates": [0.0 if i == 0 else 5e-6]} for i in range(200)]
        # All-zero gradients in this fixture are valid finite outcomes, not an
        # excuse to silently add steps, change sampling, or skip evaluation.
        signals = signal_summary(groups, steps, {"all_before_finite": True, "all_after_finite": True,
            "changed_elements": 0, "l2_delta": 0.0, "sha256_before": "same", "sha256_after": "same"},
            self.snapshot["eligible_ids"])
        signals.update(optimizer_callbacks_match_trainer_steps=True,
                       reward_groups_audit_sha256=runs.digest(output / "reward_group_signal.jsonl"))
        paths = {"questions": runs.QUESTIONS, "gold": runs.GOLD, "eligible_ids": runs.ELIGIBLE,
                 "protocol": runs.NATIVE_PROTOCOL, "script": runs.CODE / "train_grpo_signal.py",
                 "frozen_grpo_entry": runs.CODE / "train_grpo.py", "environment": runs.CODE / "environment.py",
                 "adapter_config": runs.ROOT / adapter / "model/adapter_config.json",
                 "adapter_weights": runs.ROOT / adapter / "model/adapter_model.safetensors"}
        hashes = {key: runs.digest(path) for key, path in paths.items()}
        self.write(output / "run_result.json", {"status": "completed", "smoke": False,
            "formal_result_eligible": True, "optimizer_steps": 200, "world_size": 1, "seed": runs.SEED,
            "native_contract": runs.NATIVE_CONTRACT, "metrics": {"train_loss": 0.0},
            "model_artifact": str(output / "model"), "sha256": hashes, "learning_signal": signals})
        self.write(output / "run_started.json", {"training_questions": 4999,
            "config": runs.training_config(adapter, name), "versions": self.snapshot["versions"], "sha256": hashes,
            "signal_audit": {"planned_question_groups": 200},
            "grpo_config": {"max_steps": 200, "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4, "num_generations": 4, "beta": 0.0,
                "learning_rate": 5e-6, "num_iterations": 1, "seed": runs.SEED, "data_seed": runs.SEED,
                "generation_batch_size": 4, "steps_per_generation": 4}})
        self.write(output / "model/adapter_config.json", {"fixture": name})
        (output / "model/adapter_model.safetensors").write_bytes(name.encode())

    def fake_job(self, name, gpu, command, hours):
        self.calls.append((name, gpu, command, hours))
        for _, adapter, train_name, dev, _, _ in runs.RUNS:
            if name == train_name:
                self.training_output(adapter, train_name)
            elif name == dev:
                self.jsonl(runs.ROOT / f"{dev}.jsonl", [{"id": i} for i in self.snapshot["dev_ids"]])
                self.write(runs.ROOT / f"{dev}.runtime.json", {"status": "completed", "selected_questions": 500,
                    "source_questions": 500, "totals": {"questions": 500},
                    "config": inference_config(train_name, dev)})

    def fake_cpu(self, command, **kwargs):
        if "--output" in command:
            self.write(Path(command[command.index("--output") + 1]), {"fixture": "comparison"})

    def run_mock(self, gpu_error=None):
        def idle():
            if not self.calls:
                self.assertFalse(runs.MARKER.exists())
            if gpu_error:
                raise gpu_error
        with patch.object(runs, "require_syx"), patch.object(runs, "validate_inputs", return_value=self.snapshot), \
                patch.object(runs, "require_runtime"), patch.object(runs, "require_idle_gpus", side_effect=idle), \
                patch.object(runs, "require_budget", return_value=12) as budget, \
                patch.object(runs, "launch", side_effect=self.fake_job), \
                patch.object(runs.subprocess, "run", side_effect=self.fake_cpu) as cpu, \
                patch("sys.argv", ["rl_paired_runs"]):
            runs.main()
        return budget, cpu

    def test_complete_mock_uses_original_adapters_fixed_budget_and_four_dev_comparisons(self):
        budget, cpu = self.run_mock()
        self.assertEqual([call.args for call in budget.call_args_list], [(10,), (2,)])
        jobs = {name: (gpu, command, hours) for name, gpu, command, hours in self.calls}
        self.assertEqual(len(jobs), 4)
        for _, adapter, name, dev, _, gpu in runs.RUNS:
            self.assertEqual((jobs[name][0], jobs[name][2]), (gpu, 4))
            self.assertEqual((jobs[dev][0], jobs[dev][2]), (gpu, 1))
            command = jobs[name][1]
            self.assertIn("CPATH=/private/python-dev/headers", command)
            self.assertNotIn("--smoke-steps", command)
            self.assertEqual(command[command.index("--adapter") + 1], str(runs.ROOT / adapter / "model"))
            self.assertEqual(command[command.index("--eligible-ids") + 1], str(runs.ELIGIBLE))
            self.assertEqual(command[command.index("--accumulation") + 1], "4")
        self.assertEqual(cpu.call_count, 6)
        labels = {(call.args[0][call.args[0].index("--candidate-label") + 1],
                   call.args[0][call.args[0].index("--baseline-label") + 1])
                  for call in cpu.call_args_list if "--candidate-label" in call.args[0]}
        self.assertEqual(labels, {("D", "B"), ("B", "A"), ("D", "C"), ("D", "P")})
        summary = runs.read(runs.SUMMARY)
        self.assertFalse(summary["automatic_next_round"])
        self.assertFalse(summary["held_out_evaluation"])
        self.assertEqual(summary["training"]["B"]["question_groups"], 200)
        self.assertFalse(summary["training"]["B"]["learning_signal"]["all_eligible_questions_observed"])
        self.assertFalse(summary["training"]["B"]["learning_signal"]["verified_nonzero_learning_signal"])
        with self.assertRaises(FileExistsError):
            runs.require_new_outputs()

    def test_gate_is_mandatory_and_rejects_unbound_decision_before_gpu(self):
        for gate, message in ((False, "not authorized"), (True, "must bind")):
            self.write(runs.DECISION, {"gate": {"allow_formal_rl": gate},
                                      "training_timeout_hours_per_arm": 4, "inputs_sha256": {}})
            with self.subTest(gate=gate), patch.object(runs, "require_idle_gpus") as gpu, \
                    self.assertRaisesRegex(ValueError, message):
                runs.validate_inputs()
            gpu.assert_not_called()
        self.assertFalse(runs.MARKER.exists())

    def test_timeout_rejects_nonfinite_nonpositive_boolean_and_excessive_values(self):
        for value in (0, -1, 8.01, float("nan"), float("inf"), True, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runs.timeout_hours({"training_timeout_hours_per_arm": value})
        self.assertEqual(runs.timeout_hours({"training_timeout_hours_per_arm": 4}), 4.0)

    def test_gpu_gate_precedes_exclusive_marker(self):
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.run_mock(gpu_error=RuntimeError("busy"))
        self.assertFalse(runs.MARKER.exists())
        self.assertFalse(self.calls)

    def test_wrong_group_count_stops_before_dev_and_keeps_failure_marker(self):
        self.bad_count = True
        with self.assertRaisesRegex(ValueError, "fixed configuration"):
            self.run_mock()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(runs.MARKER.exists())
        self.assertFalse(runs.SUMMARY.exists())

    def test_different_group_order_stops_before_dev(self):
        self.bad_order = True
        with self.assertRaisesRegex(ValueError, "different training question group orders"):
            self.run_mock()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(runs.MARKER.exists())


if __name__ == "__main__":
    unittest.main()
