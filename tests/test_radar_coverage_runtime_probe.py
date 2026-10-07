"""CPU-only probe preparation tests; no model, GPU, SSH or real QA references."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import coverage_runtime_probe as probe


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("truncation") is not False:
            raise AssertionError("Truncation must be disabled")
        payload = json.loads(messages[1]["content"])
        repeats = payload["evidence"].get("runtime_padding", "").count(probe.PADDING)
        return [7] * (100 + repeats * 3)


class RuntimePreparation(unittest.TestCase):
    def test_three_old_cases_use_shared_prompt_and_preserve_evidence(self):
        question = {"id": "old_fixture", "question": "旧开发题？"}
        evidence = {"chunks": [{"chunk_id": "old:p001", "text": "old source"}]}
        cases = probe.probe_cases(FakeTokenizer(), question, evidence)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0]["input_tokens"], 100)
        for case in cases:
            self.assertEqual(case["messages"][0]["content"], probe.SYSTEM_PROMPT)
            supplied = json.loads(case["messages"][1]["content"])
            self.assertEqual(supplied["question"], question["question"])
            self.assertEqual(supplied["evidence"]["chunks"], evidence["chunks"])
            self.assertEqual(len(case["prompt_ids"]), case["input_tokens"])
            self.assertFalse(case["truncated"])
            self.assertLessEqual(case["input_tokens"] + 768, 32768)
        self.assertTrue(31000 <= cases[-1]["input_tokens"] <= 31064)
        self.assertNotIn("runtime_padding", evidence)

    def test_reference_keys_and_context_overflow_are_rejected(self):
        with self.assertRaises(ValueError):
            probe.probe_cases(FakeTokenizer(), {"id": "x", "question": "q", "answer": "gold"}, {"chunks": []})
        with self.assertRaises(ValueError):
            probe.probe_cases(FakeTokenizer(), {"id": "x", "question": "q"}, {"chunks": [], "reference": "gold"})
        with patch.object(FakeTokenizer, "apply_chat_template", return_value=[1] * 32768):
            with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
                probe.token_ids(FakeTokenizer(), [])

    def test_plan_never_claims_remote_or_formal_readiness(self):
        protocol = {"model": "pinned-model", "base_weights": {"revision": "fixed", "files": []}, "tokenizer_sha256": {}}
        questions = [{"id": "old1", "question": "旧题"}]
        with patch.object(probe.source_rag, "BM25Index") as index:
            index.return_value.retrieve.return_value = []
            plan = probe.build_plan(FakeTokenizer(), protocol, questions, [], {})
        self.assertFalse(plan["model_run_ready"])
        self.assertFalse(plan["formal_evaluation_permitted"])
        self.assertFalse(plan["model_identity"]["local_tokenizer_verified"])
        self.assertFalse(plan["model_identity"]["remote_weights_verified"])
        self.assertFalse(plan["model_identity"]["remote_runtime_verified"])
        self.assertEqual(plan["future_execution_boundaries"]["max_gpu_wall_seconds"], 300)
        self.assertEqual(plan["future_execution_boundaries"]["required_os_user"], "syx")
        self.assertEqual(plan["new_model_runs"], 0)

    def test_changed_old_protocol_is_rejected_before_data_reads(self):
        with patch.object(probe, "sha", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "protocol identity"):
                probe.old_inputs()

    def test_unpinned_chat_template_override_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "chat_template.jinja").write_text("overridden template")
            with patch.object(probe, "sha", return_value=probe.CONFIG_SHA256):
                with self.assertRaisesRegex(ValueError, "Unpinned tokenizer auxiliary"):
                    probe.verify_tokenizer(folder, {"tokenizer_sha256": {}})


if __name__ == "__main__":
    unittest.main()
