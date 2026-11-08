from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.radar_domain.table_program import TableExecutor, ProgramBuilder, make_reference, compact, program_answer_spec
from experiments.radar_domain.table_inference import (
    BUDGET, load_partition, run_episode, encode_prompt, trim_tokens, validate_questions, collector_training_identity)
from experiments.radar_domain.table_recovery_data import action_text
from experiments.radar_domain.table_runtime import sha, validate_runtime_evidence, _user_check, directory_identity


class FixtureTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return [ord(c) for c in compact(messages)] + [10]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class FixtureGenerator:
    def __init__(self, tokenizer, texts=None):
        self.tokenizer, self.texts, self.calls = tokenizer, list(texts or []), []

    def generate(self, prompt_ids, max_new_tokens):
        self.calls.append((len(prompt_ids), max_new_tokens))
        text = self.texts.pop(0) if self.texts else "bad"
        return (self.tokenizer.encode(text) + [self.tokenizer.eos_token_id])[:max_new_tokens]


def reference_fixture(ident="q1"):
    raw = {"record_id": "r1", "family_id": "fixture", "subjects": ["device"],
           "component": "scanner", "attribute": "power", "value": {"text": "3 W"}, "conditions_all": []}
    typed = {**deepcopy(raw), "raw_record": raw, "typed_conditions_all": [],
             "typed_value": {"kind": "number", "value": 3, "unit": "W"}}
    catalog = {"records": [typed]}; executor = TableExecutor(catalog)
    b = ProgramBuilder(); source = b.add("Scan", {})
    values = b.add("Project", {"columns": {"power": "typed_value"}}, [source])
    ref = make_reference(executor, ident, "Read the scanner power.", b.finish({"rows": values}))
    meta = {"id": ident, "origin": "real_development", "source_group": "fixture", "task_type": "simple",
            "answer_spec": program_answer_spec(ref["actions"])}
    return catalog, ref, meta


class TableInferenceTests(unittest.TestCase):
    def test_reference_control_flow_retains_full_answer_and_fixture_origin(self):
        catalog, ref, _ = reference_fixture()
        tokenizer = FixtureTokenizer(); generator = FixtureGenerator(tokenizer, [action_text(a) for a in ref["actions"]])
        row = run_episode({"id": ref["id"], "question": ref["question"]}, TableExecutor(catalog), tokenizer, generator)
        self.assertEqual(row["answer"], ref["answer"])
        self.assertEqual(row["stop_reason"], "finished")
        self.assertEqual(row["calls"], 4)
        self.assertEqual(row["provenance"]["kind"], "cpu_fixture")
        self.assertFalse(row["training_eligible"])
        self.assertNotIn("source_records", json.loads(row["messages"][-1]["content"]))
        self.assertEqual(len(row["answer"]["source_records"]), 1)

    def test_invalid_outputs_consume_exactly_call_budget(self):
        catalog, ref, _ = reference_fixture(); tokenizer = FixtureTokenizer()
        row = run_episode({"id": ref["id"], "question": ref["question"]}, TableExecutor(catalog), tokenizer, FixtureGenerator(tokenizer))
        self.assertEqual((row["calls"], row["invalid_calls"]), (24, 24))
        self.assertEqual(row["stop_reason"], "call_budget")
        self.assertIsNone(row["answer"])

    def test_context_is_remeasured_after_tool_observation(self):
        catalog, ref, _ = reference_fixture(); tokenizer = FixtureTokenizer()
        from experiments.radar_domain.table_program import prompt_messages
        initial = len(encode_prompt(tokenizer, prompt_messages(ref["question"])))
        generator = FixtureGenerator(tokenizer, [action_text(ref["actions"][0])])
        row = run_episode({"id": ref["id"], "question": ref["question"]}, TableExecutor(catalog), tokenizer, generator,
                          budgets={**BUDGET, "context_tokens": initial+300})
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(row["stop_reason"], "context_budget")
        self.assertGreaterEqual(row["turn_tokens"][0]["context_tokens_after_observation"], initial+300)

    def test_initial_context_failure_never_invokes_generator(self):
        catalog, ref, _ = reference_fixture(); tokenizer = FixtureTokenizer(); generator = FixtureGenerator(tokenizer)
        row = run_episode({"id": ref["id"], "question": ref["question"]}, TableExecutor(catalog), tokenizer, generator,
                          budgets={**BUDGET, "context_tokens": 2})
        self.assertEqual(generator.calls, [])
        self.assertEqual(row["calls"], 0)
        self.assertEqual(row["stop_reason"], "context_budget")

    def test_generation_allowance_never_exceeds_remaining_total(self):
        catalog, ref, _ = reference_fixture(); tokenizer = FixtureTokenizer(); generator = FixtureGenerator(tokenizer)
        row = run_episode({"id": ref["id"], "question": ref["question"]}, TableExecutor(catalog), tokenizer, generator,
                          budgets={**BUDGET, "per_turn_generated_tokens": 2, "total_generated_tokens": 3})
        self.assertEqual([n for _, n in generator.calls], [2, 1])
        self.assertEqual(row["generated_tokens"], 3)
        self.assertEqual(row["stop_reason"], "generation_budget")

    def test_eos_counts_as_generation_but_padding_does_not(self):
        tokenizer = FixtureTokenizer()
        self.assertEqual(trim_tokens([65, 2, 0, 0], tokenizer), ([65, 2], [65]))
        self.assertEqual(trim_tokens([65, 0, 0], tokenizer), ([65], [65]))

    def test_generation_rejects_gold_fields(self):
        with self.assertRaises(ValueError):
            validate_questions([{"id": "q", "question": "x", "answer": 3}])

    def test_partition_loader_never_opens_references_or_metadata(self):
        original = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            opened.append(str(path))
            if Path(path).name in {"references.jsonl", "metadata.jsonl"}:
                raise AssertionError("Generation read evaluation material")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            data = load_partition(split="train")
        self.assertEqual(len(data["questions"]), 576)
        self.assertEqual(len(data["catalog"]["records"]), 1728)

    def test_role_label_alone_cannot_establish_initial_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"claim.json"
            path.write_text(json.dumps({"status": "completed", "stage": "initial", "arm": "I"}))
            with self.assertRaises(ValueError):
                collector_training_identity(path, role="I_domain", adapter_dir=tmp,
                    parent_manifest="data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")

    def test_self_declared_reduced_population_is_not_the_frozen_split(self):
        parent = Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")
        with tempfile.TemporaryDirectory() as tmp:
            changed = json.loads(parent.read_text())
            changed["splits"]["dev"]["ids"] = changed["splits"]["dev"]["ids"][:1]
            path = Path(tmp)/"rewritten.json"; path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "registered frozen manifest"):
                load_partition(path, "dev")

    def test_expired_runtime_evidence_is_rejected_before_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"runtime.json"
            path.write_text(json.dumps({"schema": "radar_table_runtime_evidence_v1", "status": "passed",
                "scope": "cuda_device_and_input_binding_only", "purpose": "generation", "user": "syx", "euid": 1000,
                "created_at_utc": (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat(),
                "probe": {"cuda_tensor_check_passed": True}}))
            with patch("experiments.radar_domain.table_runtime._user_check"), patch("os.geteuid", return_value=1000), \
                    patch("experiments.radar_domain.table_runtime._cuda_probe", side_effect=AssertionError("no CUDA expected")):
                with self.assertRaisesRegex(ValueError, "expired"):
                    validate_runtime_evidence(path, purpose="generation", model_files_sha256={"x": "h"},
                        implementation_sha256={"y": "h"}, input_files_sha256={"z": "h"})

    def test_administrator_cannot_spoof_ordinary_account_with_environment(self):
        with patch("os.geteuid", return_value=1000), patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="ubuntu")), \
                patch.dict("os.environ", {"USER": "syx", "LOGNAME": "syx"}):
            with self.assertRaisesRegex(ValueError, "ordinary experiment account"):
                _user_check("syx")

    def test_base_identity_keeps_config_and_tokenizer_separate_from_adapter(self):
        bound = {"models/base/config.json": "config", "models/base/tokenizer.json": "tok",
                 "models/base/model.safetensors": "weights", "outputs/I/adapter_model.safetensors": "adapter"}
        self.assertEqual(directory_identity(bound, "models/base"),
                         {"config.json": "config", "tokenizer.json": "tok", "model.safetensors": "weights"})

    def test_changed_base_config_cannot_reuse_completed_adapter_proof(self):
        parent = Path("data/radar_domain_recovery_v2/frozen_20261107/split_manifest.json")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); adapter = folder/"adapter"; base = folder/"base"
            adapter.mkdir(); base.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (adapter/name).write_text("fixture")
            config = folder/"config.json"; config.write_text(json.dumps({"base_model_dir": str(base)}))
            prep = folder/"preparation.json"
            prep.write_text(json.dumps({"schema": "radar_table_training_preparation_v1", "status": "completed",
                "stage": "initial", "source_split": "train", "training_eligible": True,
                "split_manifest": {"sha256": sha(parent)},
                "source_ids": json.loads(parent.read_text())["splits"]["train"]["ids"]}))
            result = folder/"result.json"
            result.write_text(json.dumps({"schema": "radar_table_training_run_result_v1", "status": "completed",
                "stage": "initial", "arm": "I", "preparation": {"path": str(prep), "sha256": sha(prep)},
                "config": {"path": str(config), "sha256": sha(config)},
                "final_adapter_files_sha256": {str(p): sha(p) for p in adapter.iterdir()},
                "base_and_starting_model_files_sha256": {str(base/"config.json"): "original"}}))
            with patch("experiments.radar_domain.table_inference.model_files", return_value={str(base/"config.json"): "changed"}):
                with self.assertRaisesRegex(ValueError, "base weights/config/tokenizer differ"):
                    collector_training_identity(result, role="I_domain", adapter_dir=adapter,
                        parent_manifest=parent, base_model_dir=base)


if __name__ == "__main__":
    unittest.main()
