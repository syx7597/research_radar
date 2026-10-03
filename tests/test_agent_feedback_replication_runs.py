"""CPU-only gates and mocked scheduling for conditional paired replication."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback import replication_runs as controller


class ReplicationControllerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        previous = Path.cwd()
        os.chdir(temp.name)
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, previous)
        self.calls, self.bad_tokens = [], False
        self.versions = {"transformers": "fixture-version"}
        self.write(controller.ROOT / "initial_sft_v1/run_result.json", {"status": "completed"})
        self.write(controller.ADAPTER / "adapter_config.json", {"peft_type": "LORA"})
        (controller.ADAPTER / "adapter_model.safetensors").write_bytes(b"initial-weights")
        self.write(controller.ROOT / "protocol.json", {"seed": 20261003})
        self.write(controller.ROOT / "continuation_code_manifest_v1.json", {"files": controller.code_identity()})
        for path in ("recovery_vs_clean_dev.json", "recovery_vs_program_dev.json"):
            self.write(controller.ROOT / path, {"fixture": path})
        self.manifest = {"seed": 20261003, "max_context": 8192, "arms": {},
                         "budget": {"total": 4, "ordinary": 2, "paired_suffix": 2}}
        for arm in ("A", "C"):
            path = controller.DATA / f"{arm}.jsonl"
            self.jsonl(path, [{"id": 0}, {"id": 1}])
            self.manifest["arms"][arm] = {"path": str(path), "sha256": controller.digest(path),
                "records": 2, "supervised_tokens": 4, "ordinary_supervised_tokens": 2,
                "paired_suffix_supervised_tokens": 2}
        self.write(controller.MANIFEST, self.manifest)
        ids = list(range(500))
        dev = {"count": 500, "ids": ids}
        for key, path in (("questions", controller.DEV_QUESTIONS), ("gold", controller.DEV_GOLD)):
            self.jsonl(path, [{"id": i, "question": "fixture", "answer": "fixture"} for i in ids])
            dev[key] = {"path": str(path), "sha256": controller.digest(path)}
        self.write(controller.ROOT / "split_manifest.json", {"splits": {"dev": dev}})
        self.jsonl(controller.ROOT / "program_dev_v1.jsonl", [{"id": i} for i in ids])
        for arm, name, dev_name in controller.OLD_RUNS:
            self.training_output(arm, name, controller.ORIGINAL_SEED)
            self.dev_output(name, dev_name)
        self.review = {"gate": {"allow_paired_replication": True}, "inputs_sha256": {
            str(path): controller.digest(path) for path in controller.reviewed_paths()}}
        self.write(controller.REVIEW, self.review)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def training_output(self, arm, name, seed):
        output = controller.ROOT / name
        spec = self.manifest["arms"][arm]
        self.write(output / "run_result.json", {"status": "completed", "metrics": {"train_loss": 0.3},
            "actual_supervised_tokens": 3 if self.bad_tokens else 4, "data_sha256": spec["sha256"],
            "seed": seed, "script_sha256": controller.code_identity()["experiments/agent_feedback/train_sft.py"]})
        self.write(output / "input_manifest.json", {"examples": 2, "supervised_tokens": 4,
            "config": controller.training_config(arm, name, seed), "versions": self.versions,
            "continuation": {"arm": arm, "manifest_sha256": controller.digest(controller.MANIFEST),
                "artifact_sha256": spec["sha256"], "expected_supervised_tokens": 4,
                "adapter_weights_sha256": controller.digest(controller.ADAPTER / "adapter_model.safetensors"),
                "adapter_config_sha256": controller.digest(controller.ADAPTER / "adapter_config.json")}})
        self.write(output / "model/adapter_config.json", {"peft_type": "LORA"})
        (output / "model/adapter_model.safetensors").write_bytes(b"continued-weights")

    def dev_output(self, name, dev_name):
        path = controller.ROOT / f"{dev_name}.jsonl"
        self.jsonl(path, [{"id": i} for i in range(500)])
        self.write(path.with_suffix(".runtime.json"), {"status": "completed", "selected_questions": 500,
            "source_questions": 500, "totals": {"questions": 500},
            "config": controller.inference_config(name, dev_name)})

    def fake_job(self, name, gpu, command, hours):
        self.calls.append((name, gpu, command, hours))
        for arm, train_name, dev_name, _ in controller.RUNS:
            if name == train_name:
                self.training_output(arm, train_name, controller.SEED)
            elif name == dev_name:
                self.dev_output(train_name, dev_name)

    def run_controller(self, *, gpu_error=None):
        def idle():
            if not self.calls:
                self.assertFalse(controller.MARKER.exists())
            if gpu_error:
                raise gpu_error
        with patch.object(controller, "require_syx"), patch.object(controller, "require_runtime"), \
                patch.object(controller, "require_idle_gpus", side_effect=idle) as gpu, \
                patch.object(controller, "require_budget", return_value=9) as budget, \
                patch.object(controller, "run_job", side_effect=self.fake_job), \
                patch.object(controller.subprocess, "run") as cpu, patch("sys.argv", ["replication_runs"]):
            controller.main()
        return gpu, budget, cpu

    def test_complete_pipeline_changes_only_continuation_seed(self):
        old = {str(path): controller.digest(path) for path in controller.reviewed_paths()}
        gpu, budget, cpu = self.run_controller()
        self.assertEqual(gpu.call_count, 2)
        self.assertEqual([call.args for call in budget.call_args_list], [(6,), (2,)])
        jobs = {name: (gpu, command, hours) for name, gpu, command, hours in self.calls}
        self.assertEqual(len(jobs), 4)
        self.assertEqual(sum(row[2] for row in jobs.values()), 6)
        for arm, name, dev_name, gpu in controller.RUNS:
            self.assertEqual((jobs[name][0], jobs[name][2]), (gpu, 2))
            self.assertEqual((jobs[dev_name][0], jobs[dev_name][2]), (gpu, 1))
            for flag, value in (("--seed", "20261004"), ("--adapter", str(controller.ADAPTER)),
                                ("--data", str(controller.DATA / f"{arm}.jsonl")), ("--epochs", "1")):
                self.assertEqual(jobs[name][1][jobs[name][1].index(flag) + 1], value)
            self.assertEqual(jobs[dev_name][1][jobs[dev_name][1].index("--seed") + 1], "20261003")
        self.assertEqual(cpu.call_count, 4)
        self.assertEqual(cpu.call_args_list[2].args[0][-1], str(controller.ROOT / controller.COMPARISONS[0][2]))
        self.assertEqual(cpu.call_args_list[3].args[0][-1], str(controller.ROOT / controller.COMPARISONS[1][2]))
        protocol = controller.read(controller.PROTOCOL)
        self.assertIn("NOT a second end-to-end seed", protocol["scope"])
        self.assertEqual(protocol["python_hash_seed"], 20261003)
        self.assertFalse(protocol["rl"])
        self.assertFalse(protocol["held_out_evaluation"])
        self.assertIn(str(controller.REVIEW), protocol["inputs_sha256"])
        controller.require_hashes(old)
        with self.assertRaises(FileExistsError):
            self.run_controller()

    def test_closed_gate_and_changed_reviewed_inputs_block_before_marker(self):
        self.review["gate"]["allow_paired_replication"] = False
        self.write(controller.REVIEW, self.review)
        with self.assertRaisesRegex(ValueError, "does not allow"):
            self.run_controller()
        self.review["gate"]["allow_paired_replication"] = True
        self.write(controller.REVIEW, self.review)
        self.write(controller.ROOT / "recovery_vs_clean_dev.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "reviewed file changed"):
            self.run_controller()
        self.assertFalse(controller.MARKER.exists())
        self.assertFalse(self.calls)

    def test_busy_gpu_or_previous_output_blocks_without_marker(self):
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.run_controller(gpu_error=RuntimeError("busy"))
        self.assertFalse(controller.MARKER.exists())
        self.assertFalse(controller.PROTOCOL.exists())
        self.write(controller.ROOT / "jobs/clean_sft_seed20261004.json", {"status": "failed"})
        with self.assertRaises(FileExistsError):
            self.run_controller()
        self.assertFalse(self.calls)

    def test_short_or_unresolved_ledger_blocks(self):
        path = controller.ledger.ROOT / "existing.json"
        row = {"status": "completed", "started_at": 0, "finished_at": 67 * 3600, "gpus": [0]}
        self.write(path, row)
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            controller.require_budget(6)
        row.update(finished_at=None, status="running")
        self.write(path, row)
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            controller.require_budget(6)
        row.update(finished_at=9 * 3600, status="completed")
        self.write(path, row)
        self.assertEqual(controller.require_budget(6), 9)

    def test_wrong_training_budget_stops_before_dev_and_retains_marker(self):
        self.bad_tokens = True
        with self.assertRaisesRegex(ValueError, "frozen budget"):
            self.run_controller()
        self.assertEqual({row[0] for row in self.calls}, {row[1] for row in controller.RUNS})
        self.assertTrue(controller.MARKER.exists())
        self.assertTrue(controller.PROTOCOL.exists())

    def test_changed_training_implementation_rejected_even_if_review_rebound(self):
        path = controller.ROOT / "continuation_code_manifest_v1.json"
        prior = controller.read(path)
        prior["files"]["experiments/agent_feedback/environment.py"] = "changed"
        self.write(path, prior)
        self.review["inputs_sha256"][str(path)] = controller.digest(path)
        self.write(controller.REVIEW, self.review)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            controller.validate_inputs()


if __name__ == "__main__":
    unittest.main()
