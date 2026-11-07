from copy import deepcopy
import json
from pathlib import Path
import unittest

from experiments.radar_domain.table_scoring import CatalogIndex, score_answer, canonical


def cells(value):
    if isinstance(value, dict):
        if value.get("kind") in {"number", "bound", "range", "approximate", "options", "correspondence"}:
            yield value
        for child in value.values():
            yield from cells(child)
    elif isinstance(value, list):
        for child in value:
            yield from cells(child)


class TableScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.references = {r["id"]: r for r in (
            json.loads(line) for line in Path(
                "data/radar_domain_recovery_v2/programs_dev_20261107/references.jsonl").read_text().splitlines())}
        cls.catalog = CatalogIndex("data/radar_domain_recovery_v2/typed_sources_20261107/typed_records.json")

    def answer(self, number):
        return deepcopy(self.references[f"RDR-DEV-{number:02d}"]["answer"])

    def result(self, changed, number):
        return score_answer(changed, self.references[f"RDR-DEV-{number:02d}"]["answer"], self.catalog)

    def test_all_twelve_reference_schemas_and_row_order(self):
        for number in range(1, 13):
            answer = self.answer(number)
            for output in answer["outputs"].values():
                if output["kind"] == "table":
                    output["rows"].reverse()
                    output["columns"].reverse()
                    for row in output["rows"]:
                        for ids in row["evidence"].values():
                            ids.reverse()
            self.assertTrue(self.result(answer, number)["correct"])

    def test_equivalent_numeric_units_but_not_wrong_units(self):
        answer = self.answer(1)
        cell = next(c for c in cells(answer["outputs"]) if c.get("unit") == "kW" and "value" in c)
        cell["value"] *= 1000
        cell["unit"] = "W"
        self.assertTrue(self.result(answer, 1)["correct"])
        cell["unit"] = "A"
        self.assertFalse(self.result(answer, 1)["correct"])

    def test_strict_bound_and_range_endpoint_are_semantic(self):
        answer = self.answer(3)
        cell = next(c for c in cells(answer["outputs"]) if c["kind"] == "bound")
        cell["operator"] = ">" if cell["operator"] == ">=" else "<"
        self.assertFalse(self.result(answer, 3)["correct"])
        answer = self.answer(2)
        cell = next(c for c in cells(answer["outputs"]) if c["kind"] == "range")
        cell["lower_inclusive"] = False
        self.assertFalse(self.result(answer, 2)["correct"])

    def test_source_content_and_cell_evidence_tampering_rejected(self):
        answer = self.answer(1)
        source = next(iter(answer["source_records"].values()))
        source["value"]["text"] = "changed source value"
        self.assertFalse(self.result(answer, 1)["well_formed"])
        answer = self.answer(1)
        row = next(o["rows"][0] for o in answer["outputs"].values() if o["kind"] == "table")
        row["evidence"][next(iter(row["evidence"]))] = ["invented-record"]
        self.assertFalse(self.result(answer, 1)["well_formed"])

    def test_empty_results_preserve_actual_search_conditions(self):
        answer = self.answer(9)
        empty = next(o for o in answer["outputs"].values() if o["kind"] == "table" and not o["rows"])
        condition = next(c for c in empty["selection_context"] if c["operator"] == "Filter")
        condition["predicates"][0]["value"] += .1
        self.assertFalse(self.result(answer, 9)["correct"])
        del empty["selection_context"]
        self.assertFalse(self.result(answer, 9)["well_formed"])

    def test_unknown_is_not_false_or_empty(self):
        answer = self.answer(10)
        assessment = next(o for o in answer["outputs"].values() if o["kind"] == "assessment")
        self.assertEqual(assessment["result"]["status"], "not_established")
        assessment["result"]["status"] = "established"
        self.assertFalse(self.result(answer, 10)["correct"])
        self.assertFalse(self.result(None, 10)["correct"])

    def test_derivation_conditions_do_not_disappear_when_flags_coincide(self):
        answer = self.answer(8)
        output = next(o for o in answer["outputs"].values() if "derived_conditions" in o)
        expression = next(iter(output["derived_conditions"][0].values()))
        expression["value"] -= 1
        self.assertFalse(self.result(answer, 8)["correct"])

    def test_format_schema_contains_no_target_facts(self):
        answer = self.answer(9)
        spec = {"outputs": [{"name": name, "kind": output["kind"], **(
            {"columns": output["columns"]} if output["kind"] == "table" else {})}
            for name, output in answer["outputs"].items()]}
        self.assertTrue(score_answer(answer, answer, self.catalog, spec)["correct"])
        spec["outputs"][0]["expected_rows"] = 0
        with self.assertRaises(ValueError):
            score_answer(answer, answer, self.catalog, spec)

    def test_option_order_and_pairing_are_distinct(self):
        a = {"kind": "options", "options": [{"kind": "number", "value": 1, "unit": "m"},
                                           {"kind": "number", "value": 2, "unit": "m"}]}
        b = deepcopy(a); b["options"].reverse()
        self.assertEqual(canonical(a), canonical(b))
        correspondence = {"kind": "correspondence", "rows": [{"voltage": 12, "current": 8},
                                                            {"voltage": 24, "current": 4}]}
        wrong = {"kind": "correspondence", "rows": [{"voltage": 12, "current": 4},
                                                  {"voltage": 24, "current": 8}]}
        self.assertNotEqual(canonical(correspondence), canonical(wrong))

    def test_equivalent_option_units_include_parent_unit(self):
        original = {"kind": "options", "unit": "kW", "options": [
            {"kind": "number", "value": 1, "unit": "kW"},
            {"kind": "number", "value": 2, "unit": "kW"}]}
        equivalent = {"kind": "options", "unit": "W", "options": [
            {"kind": "number", "value": 2000, "unit": "W"},
            {"kind": "number", "value": 1000, "unit": "W"}]}
        self.assertEqual(canonical(original), canonical(equivalent))
        alias = {"kind": "options", "unit": "μs", "options": [
            {"kind": "number", "value": 2, "unit": "μs"}]}
        scaled = {"kind": "options", "unit": "ns", "options": [
            {"kind": "number", "value": 2000, "unit": "ns"}]}
        self.assertEqual(canonical(alias), canonical(scaled))

    def test_options_wrong_parent_or_member_unit_remains_incorrect(self):
        original = {"kind": "options", "unit": "W", "options": [
            {"kind": "number", "value": 1000, "unit": "W"}]}
        wrong_parent = deepcopy(original)
        wrong_parent["unit"] = "A"
        self.assertNotEqual(canonical(original), canonical(wrong_parent))
        wrong_member = deepcopy(original)
        wrong_member["options"][0]["unit"] = "A"
        self.assertNotEqual(canonical(original), canonical(wrong_member))


if __name__ == "__main__":
    unittest.main()
