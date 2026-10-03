"""Fail-closed integration of completed continuation artifacts into SFT."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.agent_feedback.train_sft import (
    digest, validate_continuation_manifest, validate_continuation_options,
)


class ContinuationLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"test-adapter-identity")
        self.identity = {"tokenizer_config.json": "config-digest", "tokenizer.json": "vocabulary-digest"}
        self.rows = [{"id": "continuation:0", "kind": "ordinary", "input_ids": [1, 2, 3, 4],
                      "labels": [-100, -100, 3, 4], "supervised_tokens": 2},
                     {"id": "continuation:1", "kind": "paired_suffix", "input_ids": [1, 2, 5, 6],
                      "labels": [-100, -100, 5, 6], "supervised_tokens": 2}]
        self.manifest = {"max_context": 8192,
                         "budget": {"total": 4, "ordinary": 2, "paired_suffix": 2},
                         "tokenizer": {"files_sha256": dict(self.identity), "transformers": "5.18.0"},
                         "arms": {}}
        for arm in ("A", "C"):
            path = self.root / f"{arm}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            self.manifest["arms"][arm] = {"path": str(path), "sha256": digest(path), "records": 2,
                "supervised_tokens": 4, "ordinary_supervised_tokens": 2,
                "paired_suffix_supervised_tokens": 2, "input_tokens": 8, "max_input_length": 4}
        self.path = self.root / "continuation_data.json"
        self.save_manifest()
        self.args = SimpleNamespace(pretokenized=True, adapter=str(self.adapter), epochs=1,
                                    max_steps=-1, limit=0, max_length=8192,
                                    continuation_manifest=str(self.path), data=str(self.root / "A.jsonl"))

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def validate(self, rows=None, identity=None, version="5.18.0"):
        return validate_continuation_manifest(self.args, self.rows if rows is None else rows,
                    self.identity if identity is None else identity, version)

    def test_complete_pair_loads_and_records_adapter_identity(self):
        validate_continuation_options(self.args)
        audit = self.validate()
        self.assertEqual(audit["arm"], "A")
        self.assertEqual(audit["expected_supervised_tokens"], 4)
        self.assertEqual(audit["adapter_weights_sha256"], digest(self.adapter / "adapter_model.safetensors"))
        self.args.data = str(self.root / "C.jsonl")
        self.assertEqual(self.validate()["arm"], "C")

    def test_continuation_requires_exact_epoch_and_existing_adapter(self):
        for name, value in [("adapter", None), ("epochs", 2), ("max_steps", 2), ("limit", 1)]:
            args = deepcopy(self.args)
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_continuation_options(args)
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_continuation_options(self.args)
        # Initial SFT/P retain their existing configurations and do not depend
        # on any continuation files or completed adapter.
        validate_continuation_options(SimpleNamespace(pretokenized=False))

    def test_missing_completion_manifest_and_changed_peer_fail(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.save_manifest()
        (self.root / "C.jsonl").write_text("incomplete peer artifact\n")
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_selected_artifact_or_unlisted_input_fails(self):
        self.args.data = str(self.root / "copy.jsonl")
        with self.assertRaises(ValueError):
            self.validate()
        self.args.data = str(self.root / "A.jsonl")
        (self.root / "A.jsonl").write_text("wrong data\n")
        with self.assertRaises(ValueError):
            self.validate()

    def test_actual_budget_and_manifest_half_budget_are_checked(self):
        with self.assertRaises(ValueError):
            self.validate(rows=self.rows[:1])
        self.manifest["budget"]["paired_suffix"] = 3
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.validate()

    def test_tokenizer_hash_version_and_unrecorded_template_are_checked(self):
        with self.assertRaises(ValueError):
            self.validate(identity={**self.identity, "tokenizer.json": "changed"})
        with self.assertRaises(ValueError):
            self.validate(version="5.19.0")
        with self.assertRaises(ValueError):
            self.validate(identity={**self.identity, "chat_template.jinja": "unrecorded"})


if __name__ == "__main__":
    unittest.main()
