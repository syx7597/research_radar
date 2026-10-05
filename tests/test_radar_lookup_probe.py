"""Bounded bilingual diagnostic: input isolation and unchanged runtime contracts."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.radar_domain import development_probe as original
from experiments.radar_domain import lookup_probe as probe


class FakeSequence(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return FakeSequence(value) if isinstance(item, slice) else value

    def tolist(self):
        return list(self)


class FakeInputs(dict):
    def __init__(self, batch_size):
        super().__init__(input_ids=SimpleNamespace(shape=(batch_size, 2)))

    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [1, 1]

    def pad(self, rows, **kwargs):
        return FakeInputs(len(rows["input_ids"]))

    def decode(self, content, **kwargs):
        return "Find<arg>EL/M-2080<func>QueryAttr<arg>power"


class FakeModel:
    def cuda(self):
        return self

    def eval(self):
        pass

    def generate(self, input_ids, **kwargs):
        return [FakeSequence([1, 1, 9, 2]) for _ in range(input_ids.shape[0])]


class RadarLookupProbeTests(unittest.TestCase):
    def make_protocol(self, directory):
        directory = Path(directory)
        questions = {}
        for language in probe.LANGUAGES:
            path = directory / f"questions.{language}.jsonl"
            path.write_bytes((probe.DATA / path.name).read_bytes())
            questions[language] = str(path)
        old = json.loads(original.PROTOCOL.read_text())
        protocol = {
            "questions": questions, "question_count": 12,
            "inputs_sha256": {name: probe.sha(name) for name in questions.values()},
            "reference_sha256": {str(directory / "references.json"): "must never open"},
            "inference": old["inference"], "model": "unused-model",
            "models": deepcopy(old["models"]), "tokenizer_sha256": {},
            "scope": "Mocked runtime contract check; no model evaluation",
        }
        for model in protocol["models"].values():
            model["files_sha256"] = {}
        path = directory / "protocol.json"
        path.write_text(json.dumps(protocol))
        return path, protocol

    def test_languages_are_paired_but_loaded_as_separate_question_inputs(self):
        protocol = {"questions": {language: str(probe.DATA / f"questions.{language}.jsonl")
                                  for language in probe.LANGUAGES}, "question_count": 12}
        zh = probe.load_questions(protocol, "zh")
        en = probe.load_questions(protocol, "en")
        self.assertEqual([row["id"] for row in zh], [row["id"] for row in en])
        self.assertTrue(all(z["question"] != e["question"] for z, e in zip(zh, en)))
        self.assertTrue(all(set(row) == {"id", "question"} for row in zh + en))
        self.assertTrue(all("source_context_id=" in row["question"] for row in zh + en))
        with self.assertRaisesRegex(ValueError, "language"):
            probe.load_questions(protocol, "de")

    def test_inference_questions_reject_gold_or_program_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            _, protocol = self.make_protocol(directory)
            path = Path(protocol["questions"]["zh"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["canonical_program"] = "Find<arg>forbidden"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "gold/program"):
                probe.load_questions(protocol, "zh")

    def test_verify_checks_all_bound_views_without_reading_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = self.make_protocol(directory)
            with patch.object(probe, "PROTOCOL", path):
                self.assertEqual(probe.verify("P", weights=False), protocol)
                # Even a zh run verifies the bound en view, before any model load.
                Path(protocol["questions"]["en"]).write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    probe.verify("P", weights=False)
            self.assertFalse((Path(directory) / "references.json").exists())

    def test_freeze_preserves_original_runtime_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            with patch.object(probe, "PROTOCOL", path):
                probe.freeze()
                frozen = json.loads(path.read_text())
                old = original.verify("P", weights=False)
                for key in ("models", "model", "tokenizer_sha256", "base_weights", "inference", "metrics"):
                    self.assertEqual(frozen[key], old[key])
                self.assertEqual(frozen["languages"], ["zh", "en"])
                self.assertEqual(frozen["run_count"], 10)
                self.assertEqual(frozen["total_rows"], 120)
                self.assertEqual(frozen["maximum_additional_gpu_hours"], 2.5)
                self.assertEqual(frozen["maximum_hours_per_run"], 0.25)
                self.assertEqual(frozen["kb"], str(probe.KB_PATH))
                for name, digest in old["inputs_sha256"].items():
                    self.assertEqual(frozen["inputs_sha256"][name], digest)
                self.assertFalse(set(frozen["inputs_sha256"]) & set(frozen["reference_sha256"]))
                self.assertIn("not an entirely English interface", frozen["language_scope"])
                self.assertFalse(frozen["automatic_next_round"])
                self.assertFalse(frozen["training"])
                with self.assertRaises(FileExistsError):
                    probe.freeze()

    def test_mocked_program_generation_keeps_language_metadata_and_no_reference_access(self):
        model = FakeModel()
        modules = {
            "torch": SimpleNamespace(bfloat16="mock-bfloat16", inference_mode=nullcontext),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
                set_seed=lambda seed: None),
            "peft": SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda model, adapter: model)),
        }
        real_open = Path.open
        opened = []
        def guarded_open(path, *args, **kwargs):
            opened.append(str(path))
            if path.name == "references.json":
                raise AssertionError("Inference opened reference records")
            return real_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = self.make_protocol(directory)
            with patch.object(probe, "PROTOCOL", path), patch.object(probe, "OUT", Path(directory)), \
                    patch.dict(sys.modules, modules), patch.object(Path, "open", guarded_open):
                probe.generate("P", "en")
                with self.assertRaises(FileExistsError):
                    probe.generate("P", "en")
            rows = [json.loads(line) for line in (Path(directory) / "P_en.jsonl").read_text().splitlines()]
            runtime = json.loads((Path(directory) / "P_en.runtime.json").read_text())
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["label"] == "P" and row["language"] == "en" for row in rows))
            self.assertTrue(all(row["prediction"][0]["fact_id"] == "radar-review-001-12" for row in rows))
            self.assertTrue(all(row["calls"] == 3 and row["invalid_calls"] == 0 for row in rows))
            self.assertEqual(runtime["language"], "en")
            self.assertEqual(runtime["config"], protocol["inference"])
            self.assertEqual(runtime["model"], protocol["models"]["P"])
            self.assertEqual(runtime["totals"]["questions"], 12)
            self.assertEqual(runtime["output_sha256"], probe.sha(Path(directory) / "P_en.jsonl"))
            self.assertFalse(any(Path(name).name == "references.json" for name in opened))


if __name__ == "__main__":
    unittest.main()
