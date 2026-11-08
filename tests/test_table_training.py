from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain.table_training import (
    SPLIT, binding, claim_paired_settings, config_defaults, load_training_partition,
    make_matched_corpus, mask_to_count, prepare_continuation, read, sample_exact_budget,
    sha, validate_collector_training, validate_config, validate_examples, validate_matched_corpus)
from experiments.radar_domain.table_runtime_executor import TableRuntimeExecutor
from experiments.radar_domain.table_program import TableEpisode, ProgramBuilder, make_reference, execute_response, prompt_messages
from experiments.radar_domain.table_recovery_data import action_text, make_pair, answers_equal
from tests.test_radar_table_program import record


def example(ident, n=6, offset=0):
    ids = list(range(offset+10, offset+10+n+4))
    return {"id": ident, "source_id": ident, "kind": "ordinary", "training_eligible": True,
            "input_ids": ids, "labels": [-100, -100]+ids[2:n+2]+[-100, -100],
            "supervised_tokens": n, "spans": []}


class TableTrainingTests(unittest.TestCase):
    def invalid_configuration_fixture(self, invalid):
        executor = TableRuntimeExecutor({"records": [record(1), record(2)]})
        b = ProgramBuilder(); scan = b.add("Scan", {"family_id": "fixture"})
        projected = b.add("Project", {"columns": {"subject": "subject", "power": "typed_value"}}, [scan])
        reference = make_reference(executor, "fixture-invalid-config", "List the two source powers", b.finish({"rows": projected}))
        ep = TableEpisode(executor); messages = prompt_messages(reference["question"])
        for action in [reference["actions"][0], invalid, {"name": "finish", "arguments": {"answer_handle": -1}}]:
            text = action_text(action); observation = execute_response(ep, text)
            messages.extend([{"role": "assistant", "content": text},
                             {"role": "tool", "content": json.dumps(observation, ensure_ascii=False, sort_keys=True)}])
        row = {"id": reference["id"], "messages": messages, "events": ep.events, "calls": ep.calls, "answer": ep.answer,
               "provenance": {"kind": "hand_constructed_cpu_fixture"}, "training_eligible": False}
        return executor, reference, ep, row

    def test_actual_invalid_parameter_feedback_replays_and_recovers(self):
        for function, config, deps in (
            ("Filter", {"predicates": [{"column": "typed_value", "op": ">", "value": "abc", "unit": "W"}]}, [0]),
            ("Project", {"columns": []}, [0]), ("Scan", {"conditions": []}, [])):
            with self.subTest(function=function):
                action = {"name": "step", "arguments": {"function": function, "inputs": [json.dumps(config)], "dependencies": deps}}
                executor, ref, ep, row = self.invalid_configuration_fixture(action)
                self.assertEqual(len(ep.handles), 1)
                self.assertFalse(ep.events[1]["observation"]["ok"])
                self.assertEqual(ep.events[1]["observation"]["error"], "ValueError")
                pair = make_pair(executor, ref, row)
                self.assertEqual(pair["audit"]["status"], "accepted")
                self.assertEqual(pair["audit"]["shared_prefix"], 1)
                self.assertFalse(pair["audit"]["divergence_executable"])
                self.assertFalse(pair["audit"]["training_eligible"])
                self.assertFalse(pair["recovery"]["training_eligible"])
                self.assertFalse(pair["recovery"]["supervise"][4])
                self.assertTrue(answers_equal(pair["clean"]["answer"], pair["recovery"]["answer"]))

    def test_invalid_parameter_error_observation_tampering_is_rejected(self):
        action = {"name": "step", "arguments": {"function": "Project", "inputs": [json.dumps({"columns": []})], "dependencies": [0]}}
        executor, ref, _, row = self.invalid_configuration_fixture(action)
        row["events"][1]["observation"]["detail"] = "edited_error"
        pair = make_pair(executor, ref, row)
        self.assertEqual(pair["audit"]["reason"], "rollout_replay_mismatch")

    def test_actual_training_partition_is_complete_and_only_opens_train_files(self):
        paths = []
        original = Path.open
        def checked(path, *args, **kwargs):
            paths.append(str(path))
            self.assertNotIn("/dev/", str(path)); self.assertNotIn("/holdout/", str(path))
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", checked):
            split, train, refs, trajectories, catalog, bound = load_training_partition()
        self.assertEqual((len(refs), len(trajectories), len(catalog["records"])), (576, 576, 1728))
        self.assertTrue(paths)

    def test_supplement_manifest_cannot_be_training_partition(self):
        path = Path("data/radar_domain_recovery_v2/compositional_supplement_20261108/manifest.json")
        with self.assertRaises(ValueError): load_training_partition(path)

    def test_relabelled_nontraining_trace_fails_even_with_updated_file_hash(self):
        split = read(SPLIT)
        source = Path(split["splits"]["train"]["success_trajectories"]["path"])
        rows = [json.loads(s) for s in source.read_text().splitlines()]
        rows[0]["training_eligible"] = False
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/"traces.jsonl"
            fake.write_text("".join(json.dumps(r)+"\n" for r in rows))
            split["splits"]["train"]["success_trajectories"].update(path=str(fake), sha256=sha(fake))
            path = Path(tmp)/"split.json"; path.write_text(json.dumps(split))
            with self.assertRaises(ValueError): load_training_partition(path)

    def test_fixture_generation_rejected_before_pair_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp)/"generation.json"
            manifest.write_text(json.dumps({"schema": "radar_table_rollout_manifest_v1", "status": "completed",
                "origin": "hand_constructed_cpu_fixture", "split": "train", "split_sha256": sha(SPLIT)}))
            prepared = {"stage": "initial", "split_manifest": binding(SPLIT)}
            with patch("experiments.radar_domain.table_training.verify_preparation", return_value=(prepared, {})):
                with self.assertRaises(ValueError):
                    prepare_continuation("dummy", manifest, Path(tmp)/"output")
            self.assertFalse((Path(tmp)/"output").exists())

    def test_packed_fixture_and_bad_label_cannot_be_admitted(self):
        row = example("q"); row["training_eligible"] = False
        with self.assertRaises(ValueError): validate_examples([row])
        row = example("q"); row["labels"][2] += 1
        with self.assertRaises(ValueError): validate_examples([row])

    def test_budget_masks_targets_without_shortening_inputs(self):
        original = example("q", 7)
        capped = mask_to_count(original, 3)
        self.assertEqual(capped["input_ids"], original["input_ids"])
        self.assertEqual(sum(v != -100 for v in capped["labels"]), 3)
        self.assertEqual(original["supervised_tokens"], 7)
        self.assertEqual(capped["labels"][:2], [-100, -100])

    def test_exact_budget_cycles_and_pairing_preserve_every_record_identity(self):
        ordinary = [example("s1", 5), example("s2", 8)]
        pairs = [(example("p1", 6), example("p1", 6, 100)),
                 (example("p2", 9), example("p2", 9, 200))]
        a, c, budget = make_matched_corpus(ordinary, pairs, 101, 101)
        self.assertEqual(budget, {"total": 101, "ordinary": 51, "paired_suffix": 50})
        self.assertEqual(sum(r["supervised_tokens"] for r in a), 101)
        self.assertEqual(sum(r["supervised_tokens"] for r in c), 101)
        self.assertTrue(any(r["sampling_cycle"] > 0 for r in a))
        self.assertEqual(make_matched_corpus(ordinary, pairs, 101, 101), (a, c, budget))
        bad = deepcopy(c); bad.reverse()
        with self.assertRaises(ValueError): validate_matched_corpus(a, bad, budget)
        bad = deepcopy(c)
        ordinary_index = next(i for i, r in enumerate(bad) if r["kind"] == "ordinary")
        bad[ordinary_index]["input_ids"][0] += 1
        with self.assertRaises(ValueError): validate_matched_corpus(a, bad, budget)

    def test_nonmatched_pair_target_counts_fail(self):
        with self.assertRaises(ValueError):
            make_matched_corpus([example("s")], [(example("p", 5), example("p", 6))], 100, 1)

    def test_admin_extra_continuation_epochs_and_wrong_start_adapter_rejected(self):
        prepared = {"stage": "continuation", "max_context": 16384, "arms": {"A": {}, "C": {}},
                    "collector": {"base_model_dir": "models/base", "adapter_dir": "models/I"}}
        config = config_defaults("continuation")
        config.update(base_model_dir="models/base", adapter_dir="models/I", job_id="test-A")
        validate_config(config, prepared, "A")
        for change in ({"runtime_user": "ubuntu"}, {"epochs": 2}, {"adapter_dir": "models/C"},
                       {"max_gpu_hours": 9}, {"max_context": 8192}, {"ledger_path": "different_round.json"}):
            bad = {**config, **change}
            with self.assertRaises(ValueError): validate_config(bad, prepared, "A")

    def test_role_label_without_actual_initial_training_proof_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_collector_training({"collector_role": "I_domain"}, "not-opened")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.json"
            path.write_text(json.dumps({"schema": "radar_table_training_run_result_v1", "status": "cpu_only",
                                        "stage": "initial", "arm": "I"}))
            with self.assertRaises(ValueError):
                validate_collector_training({"collector_training_result": binding(path)}, "not-opened")

    def test_paired_settings_lock_each_seed_but_allow_separate_seed_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep = Path(tmp)/"preparation.json"; prep.write_text("{}")
            first = {**config_defaults("continuation"), "job_id": "A"}
            environment = {"libraries": {"torch": "fixture-v1", "peft": "fixture-v1"}, "cuda_runtime_version": "fixture-1"}
            bound = claim_paired_settings(prep, first, environment)
            second = {**first, "job_id": "C"}
            self.assertEqual(bound, claim_paired_settings(prep, second, environment))
            for change in ({"learning_rate": 1e-5}, {"max_gpu_hours": 2}, {"adapter_dir": "different/I"}):
                with self.assertRaises(ValueError): claim_paired_settings(prep, {**second, **change}, environment)
            changed = deepcopy(environment); changed["libraries"]["peft"] = "fixture-v2"
            with self.assertRaises(ValueError): claim_paired_settings(prep, second, changed)
            changed = deepcopy(environment); changed["cuda_runtime_version"] = "fixture-2"
            with self.assertRaises(ValueError): claim_paired_settings(prep, second, changed)
            replicated = {**second, "seed": second["seed"]+1}
            second_seed = claim_paired_settings(prep, replicated, environment)
            self.assertNotEqual(bound["path"], second_seed["path"])
            self.assertEqual(bound, claim_paired_settings(prep, first, environment))
            self.assertEqual(second_seed, claim_paired_settings(prep, {**replicated, "job_id": "second-C"}, environment))


if __name__ == "__main__":
    unittest.main()
