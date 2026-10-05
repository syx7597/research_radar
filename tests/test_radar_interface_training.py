"""CPU admission, exact tokenization and continuation-freeze boundary tests."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.radar_domain import interface_training as training


class PrefixStableTokenizer:
    """Minimal template supporting the real action-only boundary checks."""
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        tokens = [10 if "tools" in kwargs else 11]
        for message in messages:
            tokens.extend(ord(char) for char in message["role"] + ":" + message["content"])
            tokens.append(0)
        if add_generation_prompt:
            tokens.extend(ord(char) for char in "assistant:")
        return tokens


class RadarInterfaceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "pinned-model"
        self.model.mkdir()
        for name in ("tokenizer_config.json", "tokenizer.json"):
            (self.model / name).write_text("{}")
        self.admission = self.directory / "admission.json"
        self.audit = self.directory / "independent_ai_audit.json"
        self.audit.write_text(json.dumps({"status": "passed"}))
        self.question_path = self.directory / "train.questions.jsonl"
        self.trajectory_paths = {kind: self.directory / f"train.{kind}_trajectories.jsonl"
                                 for kind in ("agent", "program")}
        self.cache_paths = {kind: self.directory / "tokenized" / f"{kind}.jsonl"
                            for kind in self.trajectory_paths}
        self.out = self.directory / "results"
        self.preparation = self.out / "preparation.json"
        self.protocol = self.out / "protocol.json"
        self.original = self.directory / "original_protocol.json"
        old = json.loads(training.ORIGINAL_PROTOCOL.read_text())
        old["model"] = str(self.model)
        old["tokenizer_sha256"] = {name: training.sha(self.model / name)
                                   for name in ("tokenizer_config.json", "tokenizer.json")}
        self.original.write_text(json.dumps(old))
        self.questions = [{"id": "example-zh", "question": "取出记录"},
                          {"id": "example-en", "question": "Retrieve the record"}]
        rows = {kind: [] for kind in self.trajectory_paths}
        for question in self.questions:
            common = [{"role": "system", "content": "frozen contract"},
                      {"role": "user", "content": question["question"]}]
            agent = common + [
                {"role": "assistant", "content": "find"}, {"role": "tool", "content": "entity"},
                {"role": "assistant", "content": "query"}, {"role": "tool", "content": "record"},
                {"role": "assistant", "content": "finish"}, {"role": "tool", "content": "done"}]
            program = common + [{"role": "assistant", "content": "Find<arg>entity<func>QueryAttr<arg>attribute"}]
            for kind, messages in (("agent", agent), ("program", program)):
                rows[kind].append({"id": question["id"], "messages": deepcopy(messages),
                                   "supervise": [message["role"] == "assistant" for message in messages]})
        self.write_jsonl(self.question_path, self.questions)
        for kind, path in self.trajectory_paths.items():
            self.write_jsonl(path, rows[kind])
        self.refresh_admission()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(training, ADMISSION=self.admission, AUDIT=self.audit, QUESTION_PATH=self.question_path,
                TRAIN_PATHS=self.trajectory_paths, CACHE_PATHS=self.cache_paths, OUT=self.out,
                PREPARATION=self.preparation, PROTOCOL=self.protocol, ORIGINAL_PROTOCOL=self.original))
        self.stack.enter_context(patch.object(training, "versions", lambda names: {name: "test-version" for name in names}))
        self.stack.enter_context(patch.dict(sys.modules, {
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: PrefixStableTokenizer()))}))

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def refresh_admission(self):
        hashes = {str(path): training.sha(path) for path in [self.question_path, *self.trajectory_paths.values()]}
        hashes[str(self.directory / "holdout.references.jsonl")] = "must never open"
        self.admission.write_text(json.dumps({"status": "cpu_validated", "files_sha256": hashes}))

    def test_cpu_prepare_preserves_pairing_masks_and_never_opens_heldout(self):
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if "holdout" in path.name or "dev." in path.name or "references" in path.name:
                raise AssertionError("Preparation opened evaluation/reference data")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open):
            result = training.prepare(model=str(self.model))
        self.assertTrue(result["cpu_only"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["truncation"])
        agent = training.read_jsonl(self.cache_paths["agent"])
        program = training.read_jsonl(self.cache_paths["program"])
        self.assertEqual([row["id"] for row in agent], [row["id"] for row in program])
        self.assertEqual([len(row["spans"]) for row in agent], [3, 3])
        self.assertEqual([len(row["spans"]) for row in program], [1, 1])
        self.assertTrue(all(row["input_ids"][0] == 10 for row in agent))
        self.assertTrue(all(row["input_ids"][0] == 11 for row in program))
        self.assertTrue(all(row["supervised_tokens"] < len(row["input_ids"]) for row in agent + program))
        self.assertEqual(training.verify_preparation(), result)
        with self.assertRaises(FileExistsError):
            training.prepare(model=str(self.model))

    def test_admission_rejects_trajectory_question_or_order_drift(self):
        rows = training.read_jsonl(self.trajectory_paths["program"])
        rows.reverse()
        self.write_jsonl(self.trajectory_paths["program"], rows)
        self.refresh_admission()
        with self.assertRaisesRegex(ValueError, "exact question order"):
            training.load_admitted_training()
        rows.reverse()
        rows[0]["messages"][1]["content"] = "a different question"
        self.write_jsonl(self.trajectory_paths["program"], rows)
        self.refresh_admission()
        with self.assertRaisesRegex(ValueError, "question instances"):
            training.load_admitted_training()

    def test_admission_rejects_loss_on_tool_observations(self):
        rows = training.read_jsonl(self.trajectory_paths["agent"])
        rows[0]["supervise"][3] = True
        self.write_jsonl(self.trajectory_paths["agent"], rows)
        self.refresh_admission()
        with self.assertRaisesRegex(ValueError, "canonical assistant"):
            training.load_admitted_training()

    def test_context_overflow_does_not_leave_partial_caches(self):
        old = json.loads(self.original.read_text())
        old["inference"]["max_context"] = 8
        self.original.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "Context overflow"):
            training.prepare(model=str(self.model), max_length=8)
        self.assertFalse(self.preparation.exists())
        self.assertFalse(any(path.exists() for path in self.cache_paths.values()))

    def test_freeze_binds_shared_agent_cache_and_matched_pair_seeds(self):
        training.prepare(model=str(self.model))
        result = training.freeze(epochs=2, learning_rate=2e-5)
        self.assertEqual(result["seeds"]["A1"], result["seeds"]["C1"])
        self.assertEqual(result["seeds"]["A2"], result["seeds"]["C2"])
        self.assertNotEqual(result["seeds"]["A1"], result["seeds"]["A2"])
        self.assertEqual(set(result["caches"]), {"agent", "program"})
        self.assertEqual(result["config"]["epochs"], 2)
        self.assertFalse(result["automatic_next_round"])
        self.assertEqual(training.verify("C2", weights=False), result)
        with self.assertRaises(FileExistsError):
            training.freeze(epochs=2, learning_rate=2e-5)
        self.cache_paths["agent"].write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            training.verify("P", weights=False)

    def test_freeze_requires_passed_audit_after_cpu_preparation(self):
        self.audit.unlink()
        training.prepare(model=str(self.model))
        with self.assertRaises(FileNotFoundError):
            training.freeze(epochs=2, learning_rate=2e-5)
        self.audit.write_text(json.dumps({"status": "pending"}))
        with self.assertRaisesRegex(ValueError, "passed independent AI audit"):
            training.freeze(epochs=2, learning_rate=2e-5)
        self.assertFalse(self.protocol.exists())
        self.audit.write_text(json.dumps({"status": "passed"}))
        result = training.freeze(epochs=2, learning_rate=2e-5)
        self.assertEqual(result["inputs_sha256"][str(self.audit)], training.sha(self.audit))

    def test_nonfinite_training_or_gradient_logs_fail_admission(self):
        evidence = training.validate_loss_history({"train_loss": 0.2}, [{"loss": 0.3, "grad_norm": 1.5}])
        self.assertEqual(evidence["finite_logged_gradient_entries"], 1)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                training.validate_loss_history({"train_loss": value}, [])
            for key in ("loss", "grad_norm", "train_loss"):
                with self.subTest(value=value, key=key), self.assertRaisesRegex(ValueError, "Non-finite"):
                    training.validate_loss_history({"train_loss": 0.2}, [{key: value}])
        with self.assertRaisesRegex(ValueError, "missing"):
            training.validate_loss_history({}, [])

    def test_invalid_token_cache_labels_and_counts_fail(self):
        valid = {"id": "row", "input_ids": [1, 2], "labels": [-100, 2], "supervised_tokens": 1}
        self.assertEqual(training.validate_examples([valid], 2)["supervised_tokens"], 1)
        for changed in (dict(valid, labels=[-100, 3]), dict(valid, supervised_tokens=2),
                        dict(valid, input_ids=[1, 2, 3]), dict(valid, labels=[-100, -100], supervised_tokens=0)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                training.validate_examples([changed], 2)


if __name__ == "__main__":
    unittest.main()
