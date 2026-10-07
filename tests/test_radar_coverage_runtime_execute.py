"""Meaningful CPU failure tests; no CUDA, model, real questions, or references."""
import copy
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.radar_domain import coverage_runtime_execute as execute


def gpu_xml(process="", used=3, utilization=0):
    return f'''<nvidia_smi_log><driver_version>test</driver_version><gpu>
      <uuid>GPU-test</uuid><product_name>test-card</product_name>
      <mig_mode><current_mig>Disabled</current_mig></mig_mode>
      <fb_memory_usage><used>{used} MiB</used><total>49140 MiB</total></fb_memory_usage>
      <utilization><gpu_util>{utilization} %</gpu_util></utilization>
      <processes>{process}</processes></gpu></nvidia_smi_log>'''


def cases():
    msg = [{"role": "system", "content": "fixture"}, {"role": "user", "content": json.dumps(
        {"question": "synthetic old question", "evidence": {"chunks": [{"chunk_id": "old:p1"}]}})}]
    return [{"case_id": name, "input_tokens": 3, "prompt_ids": [1, 2, 3],
             "messages": msg, "truncated": False} for name in execute.CASE_IDS]


class RuntimeExecution(unittest.TestCase):
    def test_real_and_effective_user_both_required(self):
        with patch.object(execute.os, "getuid", return_value=1000), patch.object(execute.os, "geteuid", return_value=1001), \
             patch.object(execute.pwd, "getpwuid", side_effect=lambda uid: SimpleNamespace(pw_name="syx" if uid == 1000 else "admin")):
            with self.assertRaisesRegex(ValueError, "syx"):
                execute.identity()
        with patch.object(execute.os, "getuid", return_value=1000), patch.object(execute.os, "geteuid", return_value=1000), \
             patch.object(execute.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="syx")):
            self.assertEqual(execute.identity()["effective_user"], "syx")

    def test_only_idle_physical_card_accepted(self):
        self.assertEqual(execute.parse_gpu_xml(gpu_xml(), "0")["uuid"], "GPU-test")
        for text in (gpu_xml('<process_info><pid>999</pid></process_info>'), gpu_xml(used=400), gpu_xml(utilization=1)):
            with self.assertRaises(ValueError):
                execute.parse_gpu_xml(text, "GPU-test")
        with self.assertRaises(ValueError):
            execute.parse_gpu_xml(gpu_xml(), "0,1")

    def test_changed_public_or_plan_rejected_before_input_read(self):
        with patch.object(execute, "digest", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "public preparation"):
                execute.load_plan(Path("unused"))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            public = root / "public.json"
            plan = root / "plan.json"
            plan.write_text("changed")
            public.write_text(json.dumps({"local_plan": {"path": "plan.json", "sha256": execute.PLAN_SHA}}))
            with patch.object(execute, "ROOT", root), patch.object(execute, "PUBLIC", public), \
                 patch.object(execute, "PUBLIC_SHA", execute.digest(public)):
                with self.assertRaisesRegex(ValueError, "SHA binding"):
                    execute.load_plan(plan)

    def test_only_small_idle_xorg_graphics_process_is_allowed(self):
        process = ('<process_info><pid>123</pid><type>G</type>'
                   '<process_name>/usr/lib/xorg/Xorg</process_name>'
                   '<used_memory>4 MiB</used_memory></process_info>')
        result = execute.parse_gpu_xml(gpu_xml(process), "0")
        self.assertEqual(len(result["allowed_idle_display_processes"]), 1)
        for changed in (process.replace('>G<', '>C+G<'), process.replace('>G<', '>C<'),
                        process.replace('4 MiB', '128 MiB'), process.replace('Xorg', 'python')):
            with self.assertRaises(ValueError):
                execute.parse_gpu_xml(gpu_xml(changed), "0")
        with self.assertRaises(ValueError):
            execute.parse_gpu_xml(gpu_xml(process, utilization=1), "0")

    def test_all_prompt_ids_recomputed_without_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["truncation"] is False
                return [1, 2, 3]
        plan = {"cases": cases()}
        execute.verify_prompts(Tokenizer(), plan)
        plan["cases"][2]["prompt_ids"][1] = 9
        with self.assertRaisesRegex(ValueError, "Prompt IDs"):
            execute.verify_prompts(Tokenizer(), plan)
        plan = {"cases": cases()[:2]}
        with self.assertRaisesRegex(ValueError, "roster"):
            execute.verify_prompts(Tokenizer(), plan)

    def test_model_view_uses_only_attested_files_and_checks_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model, out = root / "checkpoint", root / "output"
            model.mkdir(); out.mkdir()
            weights, weight_map = [], {}
            for n in (1, 2):
                name = f"model-{n:05d}-of-00002.safetensors"
                header = json.dumps({f"tensor{n}": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
                path = model / name
                path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
                weights.append({"file": name, "bytes": path.stat().st_size, "sha256": execute.digest(path)})
                weight_map[f"tensor{n}"] = name
            for name in ("config.json", "generation_config.json"):
                (model / name).write_text("{}")
            index = model / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": weight_map}))
            (model / "curl.part").write_text("irrelevant download")
            (model / "adapter_config.json").write_text("must never enter view")
            identity = {"expected_base_weights": {"files": weights},
                        "expected_config_sha256": execute.digest(model / "config.json"),
                        "expected_generation_config_sha256": execute.digest(model / "generation_config.json")}
            files = execute.verify_model(model, identity)
            view = execute.model_view(out, files)
            expected = {Path(p).name: p for p in files}
            execute.check_model_view(view, expected)
            self.assertNotIn("adapter_config.json", {p.name for p in view.iterdir()})
            (view / "config.json").unlink()
            (view / "config.json").symlink_to(model / "curl.part")
            with self.assertRaisesRegex(ValueError, "target"):
                execute.check_model_view(view, expected)
            index.write_text(json.dumps({"weight_map": {"tensor1": weights[1]["file"]}}))
            with self.assertRaisesRegex(ValueError, "Shard index"):
                execute.verify_model(model, identity)
            (model / weights[0]["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                execute.verify_model(model, identity)

    def test_tokenizer_override_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                (path / name).write_text("{}")
            (path / "chat_template.jinja").write_text("changed")
            pinned = execute.digest(path / "config.json")
            identity = {"expected_tokenizer_sha256": {"tokenizer.json": pinned, "tokenizer_config.json": pinned},
                        "expected_config_sha256": pinned}
            with patch.object(execute.preparation, "CONFIG_SHA256", pinned):
                with self.assertRaisesRegex(ValueError, "Unpinned tokenizer"):
                    execute.tokenizer_precheck(path, identity)

    def test_greedy_config_does_not_inherit_sampling_defaults(self):
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def get_generation_mode(self):
                return SimpleNamespace(value="greedy_search")
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(GenerationConfig=Config)}):
            config = execute.greedy_config({"eos_token_id": [3, 4], "do_sample": True,
                                             "temperature": 0.6, "top_p": 0.8}, SimpleNamespace(pad_token_id=0, eos_token_id=3))
        self.assertFalse(config.do_sample)
        self.assertEqual((config.temperature, config.top_p, config.num_beams, config.max_new_tokens), (1.0, 1.0, 1, 768))
        self.assertEqual(config.eos_token_id, [3, 4])
        model = Mock()
        execute.generate_one(model, "input", "mask", config)
        call = model.generate.call_args.kwargs
        self.assertNotIn("use_model_defaults", call)
        self.assertIs(model.generation_config, config)
        self.assertFalse(call["generation_config"].do_sample)
        self.assertEqual(call["generation_config"].max_new_tokens, 768)

    def test_shared_format_checks_membership_not_truth(self):
        case = cases()[0]
        valid = {"answer_text": "fixture", "citations": [{"chunk_id": "old:p1"}]}
        self.assertEqual(execute.parse_format(json.dumps(valid), case), valid)
        execute.parse_format('{"answer_text":"证据不足","citations":[]}', case)
        for bad in ('{"answer_text":"x","citations":[{"chunk_id":"new:p1"}]}',
                    '{"answer_text":"x","answer_text":"y","citations":[]}',
                    '{"answer_text":"x","citations":[],"score":1}'):
            with self.assertRaises(ValueError):
                execute.parse_format(bad, case)

    def test_watchdog_reaps_own_term_ignoring_child(self):
        with tempfile.TemporaryDirectory() as d:
            child = [sys.executable, "-B", "-c",
                     "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)"]
            result = execute.watch_child(child, dict(os.environ), Path(d), wall_seconds=0.8, grace=0.15)
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["returncode"], -signal.SIGKILL)
            self.assertTrue(result["process_reaped"])
            self.assertFalse(result["GPU_hours_is_lower_bound"])
            self.assertLess(result["gpu_child_wall_seconds"], 2)

    def test_watchdog_completed_and_failure_are_distinct(self):
        for code, status in ((0, "completed"), (3, "worker_failed")):
            with tempfile.TemporaryDirectory() as d:
                result = execute.watch_child([sys.executable, "-B", "-c", f"raise SystemExit({code})"],
                                             dict(os.environ), Path(d), wall_seconds=1, grace=0.1)
            self.assertEqual(result["status"], status)

    def test_unreaped_timeout_and_interrupt_preserve_uncertain_accounting(self):
        timeout = subprocess.TimeoutExpired("fake", 1)
        for waits in ([timeout, timeout, timeout], [InterruptedError("stopped"), timeout]):
            process = Mock(pid=123456, returncode=None)
            process.wait.side_effect = waits
            with tempfile.TemporaryDirectory() as d, patch.object(execute.subprocess, "Popen", return_value=process), \
                 patch.object(execute, "terminate_owned_group") as stop:
                result = execute.watch_child(["fake"], {}, Path(d), wall_seconds=1, grace=0.1)
            self.assertEqual(result["status"], "cleanup_unverified")
            self.assertFalse(result["process_reaped"])
            self.assertTrue(result["GPU_hours_is_lower_bound"])
            self.assertTrue(all(call.args[0] is process for call in stop.call_args_list))
            self.assertTrue(all(call.kwargs["timeout"] <= 1 for call in process.wait.call_args_list))

    def test_partial_case_json_does_not_hide_watchdog_failure(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            (output / "old_base.json").write_text('{"format_error":null}')
            (output / "old_padded_8192.json").write_text('{"case_id":')
            status = {"status": "timeout"}
            status.update(execute.collect_cases(output))
        self.assertEqual(status["status"], "timeout")
        self.assertEqual(status["format_passed_cases"], ["old_base"])
        self.assertEqual(status["incomplete_case_files"], ["old_padded_8192.json"])

    def test_cpu_failure_saved_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            with patch.object(execute, "identity", side_effect=ValueError("wrong user")), \
                 patch.object(execute, "watch_child") as watch:
                result = execute.execute(Path("unused"), output, Path("unused"), "0")
            watch.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["new_GPU_hours"], 0)
            self.assertEqual(result["new_model_runs"], 0)
            self.assertFalse(result["formal_evaluation_permitted"])
            self.assertTrue((output / "runtime.json").exists())
            with self.assertRaises(FileExistsError):
                execute.execute(Path("unused"), output, Path("unused"), "0")


if __name__ == "__main__":
    unittest.main()
