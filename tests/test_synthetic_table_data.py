"""Synthetic data integrity and independent-oracle tests; no model is invoked."""
from copy import deepcopy
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.radar_domain import synthetic_table_data as syn


class SyntheticSourceTests(unittest.TestCase):
    def test_source_facts_are_deterministic_but_split_specific(self):
        a = syn.generate_group("train", 1)
        self.assertEqual(a, syn.generate_group("train", 1))
        self.assertNotEqual(a, syn.generate_group("synthetic_dev", 1))
        self.assertNotEqual(a, syn.generate_group("train", 1, seed=syn.SEED + 1))

    def test_entire_declared_group_and_entity_isolation(self):
        group_ids, subjects, sources, question_ids, record_ids = set(), set(), set(), set(), set()
        for split, count in syn.GROUP_COUNTS.items():
            for index in range(1, count + 1):
                group = syn.generate_group(split, index)
                doc, records, chunk = syn.compile_group(group)
                specs = syn.make_task_specs(group)
                self.assertNotIn(group["source_group_id"], group_ids)
                self.assertNotIn(group["source_id"], sources)
                group_ids.add(group["source_group_id"])
                sources.add(group["source_id"])
                self.assertEqual(18, len(records))
                self.assertEqual(6, len(specs))
                self.assertEqual(6, len({s["category"] for s in specs}))
                for m in group["models"]:
                    self.assertNotIn(m["subject"], subjects)
                    subjects.add(m["subject"])
                for s in specs:
                    self.assertNotIn(s["id"], question_ids)
                    question_ids.add(s["id"])
                for r in records:
                    self.assertNotIn(r["record_id"], record_ids)
                    record_ids.add(r["record_id"])
        self.assertEqual((112, 336, 112, 672, 2016),
                         (len(group_ids), len(subjects), len(sources), len(question_ids), len(record_ids)))

    def test_all_citations_bind_exact_local_synthetic_rows(self):
        group = syn.generate_group("train", 8)
        document, records, chunk = syn.compile_group(group)
        self.assertEqual(syn.digest(document.encode()), chunk["text_sha256"])
        for r in records:
            self.assertEqual(syn.ORIGIN, r["origin"])
            self.assertTrue(r["synthetic"])
            self.assertFalse(r["equipment_fact"])
            self.assertEqual(r["record_id"], r["source_fact_id"])
            self.assertTrue(r["source"]["source_path"].startswith("data/radar_domain_recovery_v2/synthetic_20261107/sources/"))
            for s in r["citation"]["spans"]:
                self.assertEqual(s["quote"], document[s["start"]:s["end"]])
            self.assertEqual(r["raw_record"]["citation"], r["citation"])

    def test_source_generation_contains_no_question_or_answer_data(self):
        group = syn.generate_group("train", 3)
        _, records, chunk = syn.compile_group(group)
        for obj in [group, *records, chunk]:
            self.assertNotIn("answer", obj)
            self.assertNotIn("answer_data", obj)
            self.assertNotIn("question", obj)

    def test_oracle_has_empty_and_nonempty_cases(self):
        outcomes = []
        for index in range(1, 9):
            group = syn.generate_group("train", index)
            specs = syn.make_task_specs(group)
            numeric = next(s for s in specs if s["category"] == "numeric_filter")
            outcomes.append(bool(syn.oracle(group, numeric)["rows"]))
            empty = next(s for s in specs if s["category"] == "empty_result")
            self.assertEqual([], syn.oracle(group, empty)["rows"])
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)

    def test_oracle_range_membership_has_positive_and_negative_cases(self):
        verdicts = []
        negative_source_values = []
        for index in range(1, 20):
            group = syn.generate_group("train", index)
            spec = next(s for s in syn.make_task_specs(group) if s["category"] == "range_membership")
            verdicts += [r["within_range"] for r in syn.oracle(group, spec)["rows"]]
            negative_source_values += [m["temperature_C"][0] < 0 for m in group["models"]]
        self.assertEqual({True, False}, set(verdicts))
        self.assertTrue(any(negative_source_values))

    def test_unknown_joint_configuration_does_not_become_positive(self):
        group = syn.generate_group("synthetic_dev", 4)
        spec = next(s for s in syn.make_task_specs(group) if s["category"] == "unknown_configuration")
        actual = syn.oracle(group, spec)["joint_support"]
        self.assertTrue(actual["individual_domain_checks"]["gate_count"])
        self.assertTrue(actual["individual_domain_checks"]["sample_interval"])
        self.assertEqual("not_established", actual["status"])

    def test_mode_pairs_survive_compilation_without_cross_product(self):
        group = syn.generate_group("train", 12)
        _, records, _ = syn.compile_group(group)
        mode_records = [r for r in records if r["attribute"] == "mode_parameters"]
        self.assertEqual(3, len(mode_records))
        for source, record in zip(group["models"], mode_records):
            rows = record["typed_value"]["rows"]
            self.assertEqual(2, len(rows))
            expected = [(m["mode"], m["pulse_width_us"], m["prf_hz"]) for m in source["modes"]]
            actual = [(m["mode_code"], m["pulse_width"]["value"], m["prf"]["value"]) for m in rows]
            self.assertEqual(expected, actual)

    def test_actual_shared_executor_matches_independent_oracle_all_categories(self):
        from experiments.radar_domain.table_program import TableExecutor
        _, groups, table, _, specs = syn.build_sources(group_counts={"train": 2, "synthetic_dev": 1})
        executor = TableExecutor(table)
        group_map = {g["source_group_id"]: g for g in groups}
        for spec in specs:
            reference, check = syn.verify_reference(executor, group_map[spec["source_group_id"]], spec)
            self.assertTrue(check["actual_execution_completed"])
            self.assertTrue(check["oracle_match"])
            self.assertEqual(check["oracle_semantics_sha256"], check["actual_semantics_sha256"])
            self.assertFalse(reference["model_generated"])
            self.assertFalse(reference["model_failure"])
            self.assertLessEqual(len(reference["actions"]), 24)

    def test_oracle_detects_a_corrupted_typed_value(self):
        from experiments.radar_domain.table_program import TableExecutor
        group = syn.generate_group("train", 1)
        _, records, _ = syn.compile_group(group)
        spec = syn.make_task_specs(group)[0]
        damaged = deepcopy(records)
        target = next(r for r in damaged if r["subjects"] == [spec["subject"]] and r["attribute"] == "output_power")
        target["typed_value"]["value"] += 999
        with self.assertRaisesRegex(ValueError, "Independent synthetic oracle mismatch"):
            syn.verify_reference(TableExecutor({"records": damaged}), group, spec)

    def test_question_inputs_are_question_only_and_answer_files_are_separate(self):
        import json
        outputs = syn.build(group_counts={"train": 1, "synthetic_dev": 1})
        for split in ("train", "synthetic_dev"):
            questions = [json.loads(x) for x in outputs[syn.LOCAL / f"{split}.questions.jsonl"].decode().splitlines()]
            references = [json.loads(x) for x in outputs[syn.LOCAL / f"{split}.references.jsonl"].decode().splitlines()]
            self.assertEqual(6, len(questions))
            self.assertEqual({q["id"] for q in questions}, {r["id"] for r in references})
            self.assertTrue(all(set(q) == {"id", "question"} for q in questions))
            self.assertTrue(all({"id", "question", "actions", "answer"} <= set(r) for r in references))

    def test_answer_specs_contain_format_only(self):
        group = syn.generate_group("train", 1)
        for task in syn.make_task_specs(group):
            spec = syn.answer_spec(task["category"])
            self.assertEqual({"outputs"}, set(spec))
            for output in spec["outputs"]:
                self.assertTrue(set(output) <= {"name", "kind", "columns"})
                self.assertFalse(any(k in output for k in ("value", "row_count", "record_id", "status")))

    def test_builder_does_not_read_real_tasks_holdout_or_model_results(self):
        original_bytes = Path.read_bytes

        def read_bytes(path, *args, **kwargs):
            self.assertIn(str(path), {str(Path(syn.__file__)), str(syn.ROOT / "experiments/radar_domain/table_program.py")})
            return original_bytes(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", read_bytes), patch.object(Path, "read_text", side_effect=AssertionError("No source data reads are allowed")):
            outputs = syn.build(group_counts={"train": 1, "synthetic_dev": 1})
        self.assertIn(syn.LOCAL / "full_manifest.json", outputs)


if __name__ == "__main__":
    unittest.main()
