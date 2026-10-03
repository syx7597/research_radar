"""CPU-only checks for the fixed continuation scheduling and completion gates."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.agent_feedback import continuation_runs as controller


class ContinuationControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        old_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, old_cwd)
        self.calls = []
        self.bad_tokens = False
        self.bad_adapter = False
        self.write(controller.ROOT / "initial_sft_v1/run_result.json", {"status": "completed"})
        self.write(controller.ADAPTER / "adapter_config.json", {"peft_type": "LORA"})
        (controller.ADAPTER / "adapter_model.safetensors").write_bytes(b"initial-weights")
        ids = list(range(500))
        dev = {"count": 500, "ids": ids}
        for key, path in (("questions", controller.DEV_QUESTIONS), ("gold", controller.DEV_GOLD)):
            self.jsonl(path, [{"id": i, "question": "fixture", "answer": "fixture"} for i in ids])
            dev[key] = {"path": str(path), "sha256": controller.digest(path)}
        self.write(controller.ROOT / "split_manifest.json", {"splits": {"dev": dev}})
        self.jsonl(controller.ROOT / "program_dev_v1.jsonl", [{"id": i} for i in ids])
        self.manifest = {"seed": 20261003, "max_context": 8192, "arms": {},
                         "budget": {"total": 4, "ordinary": 2, "paired_suffix": 2}}
        for arm in ("A", "C"):
            path = controller.DATA / f"{arm}.jsonl"
            self.jsonl(path, [{"id": 0}, {"id": 1}])
            self.manifest["arms"][arm] = {"path": str(path), "sha256": controller.digest(path),
                "records": 2, "supervised_tokens": 4, "ordinary_supervised_tokens": 2,
                "paired_suffix_supervised_tokens": 2}
        self.write(controller.MANIFEST, self.manifest)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def fake_job(self, name, gpu, command, hours):
        self.calls.append((name, gpu, command, hours))
        if name in {"clean_sft_v1", "recovery_sft_v1"}:
            arm = "A" if name == "clean_sft_v1" else "C"
            spec = self.manifest["arms"][arm]
            output = controller.ROOT / name
            self.write(output / "run_result.json", {"status": "completed", "metrics": {"train_loss": 0.3},
                "actual_supervised_tokens": 3 if self.bad_tokens else 4, "data_sha256": spec["sha256"]})
            self.write(output / "input_manifest.json", {"examples": 2, "supervised_tokens": 4,
                "continuation": {"arm": arm, "manifest_sha256": controller.digest(controller.MANIFEST),
                    "artifact_sha256": spec["sha256"], "expected_supervised_tokens": 4,
                    "adapter_weights_sha256": "wrong" if self.bad_adapter else
                        controller.digest(controller.ADAPTER / "adapter_model.safetensors")}})
            self.write(output / "model/adapter_config.json", {"peft_type": "LORA"})
            (output / "model/adapter_model.safetensors").write_bytes(b"continued-weights")
        else:
            path = controller.ROOT / f"{name}.jsonl"
            self.jsonl(path, [{"id": i} for i in range(500)])
            self.write(path.with_suffix(".runtime.json"), {"status": "completed", "selected_questions": 500,
                "source_questions": 500, "totals": {"questions": 500},
                "config": {"adapter": command[command.index("--adapter") + 1]}})

    def run_controller(self):
        with patch.object(controller, "require_syx"), patch.object(controller, "require_idle_gpus") as idle, \
                patch.object(controller, "run_job", side_effect=self.fake_job), \
                patch.object(controller.subprocess, "run") as cpu, patch("sys.argv", ["continuation_runs"]):
            controller.main()
        return idle, cpu

    def test_complete_mock_pipeline_uses_fixed_arms_and_raw_dev_predictions(self):
        initial_hash = controller.digest(controller.ADAPTER / "adapter_model.safetensors")
        idle, cpu = self.run_controller()
        self.assertEqual(idle.call_count, 2)
        jobs = {name: (gpu, command, hours) for name, gpu, command, hours in self.calls}
        self.assertEqual(set(jobs), {"clean_sft_v1", "recovery_sft_v1", "clean_dev_v1", "recovery_dev_v1"})
        for arm, name, dev_name, gpu in controller.RUNS:
            self.assertEqual((jobs[name][0], jobs[name][2]), (gpu, 6))
            self.assertEqual((jobs[dev_name][0], jobs[dev_name][2]), (gpu, 3))
            command = jobs[name][1]
            self.assertIn("--pretokenized", command)
            for flag, value in (("--epochs", "1"), ("--max-steps", "-1"), ("--limit", "0"),
                                ("--data", str(controller.DATA / f"{arm}.jsonl")),
                                ("--adapter", str(controller.ADAPTER))):
                self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(cpu.call_count, 4)
        for call, label, stem in zip(cpu.call_args_list[2:], ("A", "P"), ("clean", "program")):
            command = call.args[0]
            self.assertEqual(command[command.index("--baseline-label") + 1], label)
            self.assertEqual(command[command.index("--baseline") + 1], str(controller.ROOT / f"{stem}_dev_v1.jsonl"))
            self.assertEqual(command[command.index("--candidate") + 1], str(controller.ROOT / "recovery_dev_v1.jsonl"))
        self.assertEqual(initial_hash, controller.digest(controller.ADAPTER / "adapter_model.safetensors"))
        with self.assertRaises(FileExistsError):
            self.run_controller()

    def test_wrong_actual_budget_blocks_dev_and_keeps_failure_marker(self):
        self.bad_tokens = True
        with self.assertRaises(ValueError):
            self.run_controller()
        self.assertEqual({row[0] for row in self.calls}, {"clean_sft_v1", "recovery_sft_v1"})
        self.assertTrue((controller.ROOT / "continuation_runs_started.json").exists())

    def test_wrong_initial_adapter_identity_blocks_dev(self):
        self.bad_adapter = True
        with self.assertRaises(ValueError):
            self.run_controller()
        self.assertEqual(len(self.calls), 2)

    def test_missing_manifest_changed_corpus_or_existing_job_rejected_before_gpu(self):
        controller.MANIFEST.unlink()
        with self.assertRaises(FileNotFoundError):
            controller.validate_inputs()
        self.write(controller.MANIFEST, self.manifest)
        path = controller.DATA / "C.jsonl"
        original = path.read_bytes()
        path.write_bytes(b"incomplete")
        with self.assertRaises(ValueError):
            controller.validate_inputs()
        path.write_bytes(original)
        self.write(controller.ROOT / "jobs/clean_sft_v1.json", {"status": "failed"})
        with self.assertRaises(FileExistsError):
            controller.validate_inputs()

    def test_real_and_effective_identity_must_both_be_nonroot_syx(self):
        for real, effective, name, allowed in ((0, 1000, "syx", False), (1000, 0, "syx", False),
                (1000, 1000, "admin", False), (1000, 1000, "syx", True)):
            with self.subTest(real=real, effective=effective, name=name), \
                    patch.object(controller.os, "getuid", return_value=real), \
                    patch.object(controller.os, "geteuid", return_value=effective), \
                    patch.object(controller.pwd, "getpwuid", return_value=SimpleNamespace(pw_name=name)):
                if allowed:
                    controller.require_syx()
                else:
                    with self.assertRaises(PermissionError):
                        controller.require_syx()


if __name__ == "__main__":
    unittest.main()
