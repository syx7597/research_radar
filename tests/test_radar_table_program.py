"""Runtime invariants using small independent catalogs, without thesis answers."""
from copy import deepcopy
import json
import unittest

from experiments.radar_domain.table_program import (
    TableExecutor, TableEpisode, ProgramBuilder, execute_actions, compare, contains, digest)


def record(index, value=None):
    raw = {"record_id": f"r{index}", "family_id": "fixture",
           "subjects": [f"device{index}"], "component": "scanner",
           "attribute": "power", "value": {"text": f"{index} W"},
           "conditions_all": []}
    return {**deepcopy(raw), "raw_record": raw,
            "typed_value": value or {"kind": "number", "value": index, "unit": "W"},
            "typed_conditions_all": []}


class TableProgramTests(unittest.TestCase):
    def setUp(self):
        self.executor = TableExecutor({"records": [record(i) for i in range(1, 8)]})

    def call(self, ep, function, config, deps=()):
        return ep.step(function, [json.dumps(config)], list(deps))

    def test_observation_sample_does_not_truncate_answer(self):
        ep = TableEpisode(self.executor)
        obs = self.call(ep, "Scan", {"family_id": "fixture"})
        self.assertEqual((obs["row_count"], len(obs["sample"])), (7, 3))
        self.assertTrue(obs["sample_truncated"])
        self.call(ep, "Project", {"columns": {"subject": "subject", "power": "typed_value"}}, [0])
        self.call(ep, "Assemble", {"names": ["devices"]}, [1])
        receipt = ep.finish(2)
        self.assertEqual(len(ep.answer["outputs"]["devices"]["rows"]), 7)
        self.assertEqual(len(ep.answer["source_records"]), 7)
        self.assertEqual(receipt["answer_sha256"], digest(ep.answer))
        self.assertNotIn("source_records", receipt)

    def test_rejected_actions_spend_budget_without_allocating_handles(self):
        ep = TableEpisode(self.executor, max_calls=2)
        self.assertFalse(self.call(ep, "Scan", {"record_ids": ["r1"]})["ok"])
        self.assertEqual((ep.calls, len(ep.handles)), (1, 0))
        self.assertTrue(self.call(ep, "Scan", {})["ok"])
        self.assertFalse(self.call(ep, "Scan", {})["ok"])
        self.assertTrue(ep.done)
        self.assertEqual((ep.calls, len(ep.handles)), (2, 1))

    def test_only_existing_integer_handles_and_assembled_answers(self):
        ep = TableEpisode(self.executor)
        self.call(ep, "Scan", {})
        for dependency in (-1, 10, True, "0"):
            self.assertFalse(self.call(ep, "Project", {"columns": {}}, [dependency])["ok"])
        self.assertFalse(ep.finish(0)["ok"])
        self.assertIsNone(ep.answer)

    def test_projection_cannot_inject_literal_answer(self):
        ep = TableEpisode(self.executor)
        self.call(ep, "Scan", {})
        self.assertFalse(self.call(ep, "Project", {"columns": {"power": 12345}}, [0])["ok"])
        self.assertFalse(self.call(ep, "Project", {"columns": {"power": "invented_path"}}, [0])["ok"])

    def test_compatible_units_and_strict_domain_boundaries(self):
        self.assertTrue(compare({"kind": "number", "value": 1, "unit": "kW"}, "==", 1000, "W"))
        with self.assertRaises(ValueError):
            compare({"kind": "number", "value": 1, "unit": "kW"}, "==", 1000, "A")
        bound = {"kind": "bound", "value": 5, "unit": "W", "operator": "<"}
        self.assertFalse(contains(bound, 5, "W"))
        self.assertTrue(contains(bound, 4, "W"))
        with self.assertRaises(ValueError):
            compare(bound, "<=", 5, "W")
        self.assertTrue(compare(bound, "<=", 5, "W", bound_endpoint=True))

    def test_approximation_and_tolerance_are_not_exact_scalars(self):
        for kind in ("approximate", "tolerance"):
            with self.assertRaises(ValueError):
                compare({"kind": kind, "value": 1, "unit": "W"}, ">", 0, "W")

    def test_empty_table_retains_scope_and_actual_predicate(self):
        b = ProgramBuilder()
        scan = b.add("Scan", {"family_id": "fixture"})
        predicate = {"column": "typed_value", "op": ">", "value": 100, "unit": "W"}
        selection = b.add("Filter", {"predicates": [predicate]}, [scan])
        ep = execute_actions(self.executor, "power >100 W", b.finish({"selected": selection}))
        out = ep.answer["outputs"]["selected"]
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["selection_context"][1]["predicates"], [predicate])
        self.assertEqual(set(ep.answer["source_records"]), {f"r{i}" for i in range(1, 8)})

    def test_failed_join_keeps_input_tables_intact(self):
        ep = TableEpisode(self.executor)
        self.call(ep, "Scan", {})
        self.call(ep, "Project", {"columns": {"key": "component", "value": "typed_value"}}, [0])
        before = deepcopy(ep.handles)
        self.assertFalse(self.call(ep, "Join", {"on": ["key"]}, [1, 1])["ok"])
        self.assertEqual(ep.handles, before)

    def test_catalog_source_identity_and_envelope_integrity(self):
        for field in ("record_id", "family_id", "attribute", "component", "subjects", "value"):
            row = record(1)
            row[field] = "modified"
            with self.assertRaises(ValueError):
                TableExecutor({"records": [row]})


if __name__ == "__main__":
    unittest.main()
