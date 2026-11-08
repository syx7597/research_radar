"""Admission checks for the separately frozen composition evaluation."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from experiments.radar_domain.freeze_compositional_supplement import (
    schema_only, live_operations, sha, verify_bindings)
from experiments.radar_domain.table_program import ProgramBuilder


class CompositionAdmissionTests(unittest.TestCase):
    def test_format_cannot_supply_values_counts_or_expected_status(self):
        for output in ({"name": "rows", "kind": "table", "columns": ["x"], "expected_rows": 1},
                       {"name": "decision", "kind": "assessment", "status": "not_established"}):
            with self.assertRaises(ValueError):
                schema_only({"outputs": [output]})

    def test_format_order_normalizes_without_erasing_column_identity(self):
        a = {"outputs": [{"name": "rows", "kind": "table", "columns": ["b", "a"]},
                         {"name": "decision", "kind": "assessment"}]}
        b = deepcopy(a); b["outputs"].reverse(); b["outputs"][1]["columns"].reverse()
        self.assertEqual(schema_only(a), schema_only(b))
        b["outputs"][1]["columns"][0] = "different"
        self.assertNotEqual(schema_only(a), schema_only(b))

    def test_dead_filter_does_not_count_as_compositional_output(self):
        b = ProgramBuilder()
        original = b.add("Scan", {})
        b.add("Filter", {"predicates": [{"column": "value", "op": ">", "value": 1}]}, [original])
        self.assertEqual(live_operations(b.finish({"result": original})), {"Scan", "Assemble"})

    def test_join_contributors_count_in_both_branches(self):
        b = ProgramBuilder()
        a = b.add("Scan", {"attribute": "a"})
        c = b.add("Scan", {"attribute": "b"})
        c = b.add("Project", {"columns": {"key": "subject"}}, [c])
        joined = b.add("Join", {"on": ["key"]}, [a, c])
        self.assertEqual(live_operations(b.finish({"result": joined})), {"Scan", "Project", "Join", "Assemble"})

    def test_future_and_boolean_dependencies_are_rejected(self):
        for dependency in (True, 1, 99, -1):
            b = ProgramBuilder(); a = b.add("Scan", {})
            b.add("Filter", {}, [dependency])
            with self.assertRaises(ValueError):
                live_operations(b.finish({"result": 1}))

    def test_changed_source_binding_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"; path.write_text('{"value":1}')
            bindings = {str(path): sha(path)}
            verify_bindings(bindings)
            path.write_text('{"value":2}')
            with self.assertRaises(ValueError):
                verify_bindings(bindings)


if __name__ == "__main__":
    unittest.main()
