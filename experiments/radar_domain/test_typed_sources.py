"""Semantic regressions for source typing; expected cases are source-grounded AI checks."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.radar_domain import typed_sources as ts


class TypedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = ts.build()
        cls.table = cls.outputs[ts.LOCAL / "typed_records.json"]
        cls.rows = {r["record_id"]: r for r in cls.table["records"]}
        cls.numeric_review = json.loads((ts.ROOT / "data/radar_domain_recovery_v1/numeric_review_20261106/record_review.json").read_text())

    def test_entire_original_snapshot_and_citations_are_preserved(self):
        original = [r for name in ts.PACKET_NAMES
                    for r in json.loads((ts.PACKET_ROOT / name).read_text())["records"]]
        self.assertEqual(502, len(self.rows))
        self.assertEqual({r["record_id"] for r in original}, set(self.rows))
        for record in original:
            row = self.rows[record["record_id"]]
            self.assertEqual(record, row["raw_record"])
            self.assertEqual(record["citation"], row["citation"])
            self.assertEqual(record["value"]["form"], row["typed_value"]["source_form"])

    def test_all_195_independently_reviewed_signed_literals(self):
        checked = 0
        for row in self.numeric_review["rows"]:
            if "independent_expected_signed_number" not in row:
                continue
            actual = self.rows[row["record_id"]]["typed_value"]
            self.assertIn(actual["kind"], {"number", "bound"}, row["record_id"])
            self.assertEqual(Decimal(row["independent_expected_signed_number"]), Decimal(str(actual["value"])), row["record_id"])
            checked += 1
        self.assertEqual(195, checked)

    def test_all_75_reviewed_bound_comparators(self):
        checked = 0
        for row in self.numeric_review["rows"]:
            if "reviewed_source_boundary" not in row:
                continue
            self.assertEqual(row["reviewed_source_boundary"]["comparator"],
                             self.rows[row["record_id"]]["typed_value"]["operator"], row["record_id"])
            checked += 1
        self.assertEqual(75, checked)

    def test_strict_endpoint_is_not_made_inclusive(self):
        lower = self.rows["WX-0018"]["typed_value"]
        upper = self.rows["WX-0027"]["typed_value"]
        self.assertEqual((">", 100), (lower["operator"], lower["value"]))
        self.assertEqual(("<", -29), (upper["operator"], upper["value"]))
        self.assertEqual(24, sum(r["typed_value"].get("operator") in {"<", ">"} for r in self.rows.values()))

    def test_separated_minus_and_locale_numbers(self):
        self.assertEqual(-8, self.rows["MRR-0047"]["typed_value"]["value"])
        self.assertEqual(40.1, self.rows["MRR-0018"]["typed_value"]["value"])
        self.assertEqual(24.23, self.rows["MRR-0020"]["typed_value"]["value"])
        self.assertEqual(15000, self.rows["WX-0064"]["typed_value"]["value"])

    def test_maximum_role_is_consistent_without_rewriting_raw_form(self):
        for rid in ("WX-0049", "RAY-0034", "RAY-0047", "marine-0150"):
            tv = self.rows[rid]["typed_value"]
            self.assertEqual(("bound", "<=", "maximum"), (tv["kind"], tv["operator"], tv["quantity_role"]))
        self.assertEqual("scalar", self.rows["RAY-0047"]["raw_record"]["value"]["form"])
        self.assertEqual("typical", self.rows["WX-0088"]["typed_value"]["quantity_role"])
        self.assertEqual("<=", self.rows["WX-0088"]["typed_value"]["operator"])

    def test_minute_unit_is_not_a_minimum_quantity_role(self):
        tv = self.rows["marine-0043"]["typed_value"]
        self.assertEqual(("min", "unspecified"), (tv["unit"], tv["quantity_role"]))
        self.assertEqual("nominal", self.rows["RAY-0043"]["typed_value"]["quantity_role"])

    def test_hv_pairs_are_not_a_cartesian_product(self):
        tv = self.rows["WX-0010"]["typed_value"]
        self.assertEqual([(4, 4), (2, 2)], [(r["peak_power_H"]["value"], r["peak_power_V"]["value"]) for r in tv["rows"]])
        self.assertEqual(["4 kW + 4 kW", "2 kW + 2 kW"], [r["configuration_label"] for r in tv["rows"]])
        self.assertFalse(tv["independent_member_cartesian_product_supported"])

    def test_voltage_branch_preserves_current_and_allowable_interval(self):
        rows = self.rows["marine-0045"]["typed_value"]["rows"]
        self.assertEqual(2, len(rows))
        self.assertEqual([(12, 10.8, 15.6, 9.5), (24, 21.6, 31.2, 5)],
                         [(r["nominal_supply"]["value"], r["allowable_supply"]["lower"],
                           r["allowable_supply"]["upper"], r["maximum_supply_current"]["value"]) for r in rows])

    def test_discrete_options_and_same_row_prf_pairing(self):
        tv = self.rows["WX-0083"]["typed_value"]
        self.assertEqual("options", tv["kind"])
        self.assertEqual([1, 1.8, 2.44], [v["value"] for v in tv["options"]])
        self.assertFalse(tv["continuous_interval_supported"])
        row = self.rows["RAY-0091"]["typed_value"]["rows"][0]
        self.assertEqual((47000, "ns", 1.2, "kHz"), (row["pulse_width"]["value"], row["pulse_width"]["unit"], row["prf"]["value"], row["prf"]["unit"]))
        self.assertEqual(1.5, self.rows["RAY-0082"]["typed_conditions_all"][0]["typed_value"]["value"])

    def test_shared_subjects_do_not_create_source_facts(self):
        row = self.rows["marine-0085"]
        self.assertEqual(["JMA-5222-7", "JMA-5222-9"], row["subjects"])
        self.assertEqual("marine-0085", row["source_fact_id"])
        self.assertNotIn("independent_sample_count", row)
        self.assertEqual(502, len({r["source_fact_id"] for r in self.rows.values()}))

    def test_printed_dual_units_are_preserved_without_recomputing(self):
        row = self.rows["marine-0136"]["typed_value"]["rows"][0]
        self.assertEqual((5.2, "kg", 11.4, "lb"), (row["primary"]["value"], row["primary"]["unit"], row["equivalent_printed"]["value"], row["equivalent_printed"]["unit"]))
        self.assertFalse(self.rows["marine-0136"]["typed_value"]["conversion_inferred"])
        self.assertEqual("text", self.rows["marine-0140"]["typed_value"]["kind"])

    def test_unknown_joint_configuration_stays_unknown(self):
        gate = self.rows["MRR-0045"]["typed_value"]
        self.assertEqual((16, 256, "count"), (gate["lower"], gate["upper"], gate["unit"]))
        self.assertIsNone(gate["every_integer_setting_supported"])
        for rid in ("MRR-0041", "MRR-0045", "MRR-0046"):
            cond = next(c for c in self.rows[rid]["typed_conditions_all"] if c["dimension"] == "parameter combinations")
            self.assertEqual("not_all_parameter_combinations_supported", cond["typed_value"]["constraint_code"])
            self.assertFalse(cond["typed_value"]["joint_cartesian_product_supported"])

    def test_uncertain_forms_and_identifiers_do_not_become_exact_numbers(self):
        for rid in ("WX-0006", "WX-0108", "MRR-0026", "WX-0065"):
            self.assertEqual("text", self.rows[rid]["typed_value"]["kind"], rid)
            self.assertFalse(self.rows[rid]["typed_value"]["numeric_operations_supported"])
        self.assertEqual("tolerance", self.rows["marine-0037"]["typed_value"]["kind"])
        for rid in ("WX-0148", "MRR-0051"):
            self.assertEqual("approximate", self.rows[rid]["typed_value"]["kind"])
            self.assertFalse(self.rows[rid]["typing_capabilities"]["exact_numeric_compare"])

    def test_generic_ellipsis_sign_units_and_role_do_not_depend_on_source_id(self):
        base = deepcopy(self.rows["WX-0002"]["raw_record"])
        base["record_id"] = "GENERIC-UNSEEN"
        for source, form, unit, expected in [
            ("2...20 seconds", "range", "seconds", (2, 20)),
            ("–40 to 80 °C", "range", "°C", (-40, 80)),
            ("24.0...24.25 GHz", "range", "GHz", (24, 24.25))]:
            base["value"] = {"form": form, "text": source, "unit_raw": unit}
            v = ts.type_record(base)["typed_value"]
            self.assertEqual(expected, (v["lower"], v["upper"]))
        base["value"] = {"form": "upper_bound", "text": "<550 μA", "unit_raw": "μA"}
        self.assertEqual(("<", 550, "μA"), tuple(ts.type_record(base)["typed_value"][k] for k in ("operator", "value", "unit")))
        base["value"] = {"form": "scalar", "text": "8 mA", "unit_raw": "mA"}
        base["conditions_all"] = [{"dimension": "quantity role", "value": "maximum"}]
        self.assertEqual("maximum", ts.type_record(base)["typed_value"]["quantity_role"])

    def test_public_adapter_preserves_extra_citations_and_explicit_uncertainty(self):
        base = deepcopy(self.rows["WX-0002"]["raw_record"])
        base["record_id"] = "GENERIC-UNCERTAINTY"
        base["additional_citations"] = [{"chunk_id": "other:page", "spans": [{"start": 1, "end": 4, "quote": "abc"}]}]
        base["value"] = {"form": "tolerance", "text": "9410±30 MHz", "unit_raw": "MHz"}
        out = ts.type_record(base)
        self.assertEqual(base, out["raw_record"])
        self.assertEqual("tolerance", out["typed_value"]["kind"])
        self.assertEqual(9410, out["typed_value"]["center"]["value"])
        self.assertEqual(30, out["typed_value"]["magnitude"]["value"])
        self.assertFalse(out["typing_capabilities"]["exact_numeric_compare"])
        self.assertTrue(all(e["additional_citations"] == base["additional_citations"] for e in out["field_evidence"].values()))
        condition = ts.condition_value({"dimension": "measuring distance", "value": "1 to 30 m (3 to 100 ft)"})
        self.assertEqual((1, 30, "m"), tuple(condition["rows"][0]["primary"][k] for k in ("lower", "upper", "unit")))

    def test_required_65_records_have_original_review_and_field_evidence(self):
        review = json.loads((ts.LOCAL / "source_review.json").read_text())
        self.assertEqual(65, len(review["required_record_ids"]))
        for rid in review["required_record_ids"]:
            r = self.rows[rid]
            self.assertTrue(r["original_source_checked_this_run"])
            self.assertNotEqual("text", r["typed_value"]["kind"])
            self.assertTrue(r["field_evidence"])
            for evidence in r["field_evidence"].values():
                self.assertEqual(rid, evidence["source_record_id"])
                self.assertTrue(evidence["source_spans"])

    def test_builder_does_not_open_task_or_answer_files(self):
        original_text, original_bytes = Path.read_text, Path.read_bytes

        def allowed(path):
            name = str(path)
            self.assertNotIn("development_task_candidates", name)
            self.assertNotIn("references.jsonl", name)
            self.assertNotIn("replays", name)
            self.assertNotIn("rollouts", name)

        def read_text(path, *args, **kwargs):
            allowed(path)
            return original_text(path, *args, **kwargs)

        def read_bytes(path, *args, **kwargs):
            allowed(path)
            return original_bytes(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text), patch.object(Path, "read_bytes", read_bytes):
            outputs = ts.build()
        self.assertEqual(502, len(outputs[ts.LOCAL / "typed_records.json"]["records"]))


if __name__ == "__main__":
    unittest.main()
