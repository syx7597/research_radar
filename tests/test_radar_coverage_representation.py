"""Synthetic relationship mutations test the complete equal-information contract."""
from copy import deepcopy
import json
import math
import unittest

from experiments.radar_domain import coverage_representation as cr


def reading(rid="r1", subjects=None, component="receiver", attribute="power", conditions=None, value="10 W"):
    return {"record_id": rid, "family_id": "fixture-family",
            "subjects": ["A", "B"] if subjects is None else subjects,
            "component": component, "attribute": attribute,
            "value": {"form": "scalar", "text": value, "unit_raw": "W"},
            "conditions_all": [{"dimension": "mode", "value": "active"}] if conditions is None else conditions,
            "citation": {"chunk_id": "fixture:p001", "text_sha256": "fixture-hash",
                         "spans": [{"start": 4, "end": 8, "quote": "10 W"}]},
            "locator": "Table 1, power", "annotation_note": "保留技术限定；not a gold answer"}


def bound_leaves(encoded):
    """Return the parsed tree and its leaf payload objects for tamper fixtures."""
    tree = json.loads(encoded)
    leaves = []

    def walk(groups):
        for node in groups:
            if "groups" in node:
                walk(node["groups"])
            else:
                leaves.extend(node["records"])

    walk(tree)
    return tree, leaves


class EqualInformation(unittest.TestCase):
    def test_empty_and_complete_round_trip_without_mutating_input(self):
        self.assertEqual(cr.encode_flat([]), "[]")
        self.assertEqual(cr.encode_bound([]), "[]")
        self.assertEqual(cr.validate_equal_information([])["record_count"], 0)
        records = [reading(), reading("r2", value="12 W")]
        original = deepcopy(records)
        for encode, decode in ((cr.encode_flat, cr.decode_flat), (cr.encode_bound, cr.decode_bound)):
            serialized = encode(records)
            self.assertIn("保留技术限定", serialized)
            self.assertEqual(decode(serialized), records)
        report = cr.validate_equal_information(records)
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["flat_utf8_bytes"], len(cr.encode_flat(records).encode("utf-8")))
        self.assertEqual(records, original)

    def test_continuous_prefix_groups_preserve_order_and_shared_subjects(self):
        records = [reading("1"), reading("2", value="11 W"),
                   reading("3", conditions=[]), reading("4", attribute="gain"),
                   reading("5", component="antenna"), reading("6", subjects=["C"]),
                   reading("7"), reading("8", subjects=["B", "A"])]
        tree = json.loads(cr.encode_bound(records))
        # Returning to the first subject list after C starts a new root run.
        self.assertEqual([n["subjects"] for n in tree], [["A", "B"], ["C"], ["A", "B"], ["B", "A"]])
        self.assertEqual([n["component"] for n in tree[0]["groups"]], ["receiver", "antenna"])
        attributes = tree[0]["groups"][0]["groups"]
        self.assertEqual([n["attribute"] for n in attributes], ["power", "gain"])
        condition_groups = attributes[0]["groups"]
        self.assertEqual(len(condition_groups), 2)
        self.assertEqual([r["record_id"] for r in condition_groups[0]["records"]], ["1", "2"])
        self.assertEqual(len(cr.decode_bound(cr.encode_bound(records))), len(records))
        cr.validate_equal_information(records)

    def test_duplicate_multiplicity_and_decoded_independence(self):
        record = reading()
        records = [record, record, deepcopy(record)]
        decoded = cr.decode_bound(cr.encode_bound(records))
        cr.validate_equal_information(records)
        self.assertEqual(len(decoded), 3)
        decoded[0]["subjects"].append("extra")
        self.assertEqual(decoded[1]["subjects"], ["A", "B"])
        self.assertEqual(record["subjects"], ["A", "B"])
        for altered in (records[:2], records + [record]):
            with self.assertRaises(ValueError):
                cr.validate_equal_information(records, bound=cr.encode_bound(altered))

    def test_null_numbers_booleans_unknown_fields_and_signed_zero(self):
        record = reading()
        record["value"].update({"unit_raw": None, "raw_number": 7, "decimal": 7.0,
                                "enabled": False, "negative_zero": -0.0})
        record["extension"] = [None, 0, 0.0, False, "0", {"nested": [2.5, True]}]
        # Structural field names in a leaf extension do not collide with nodes.
        record["groups"] = "literal original field"
        record["records"] = ["literal original field"]
        for decode, encode in ((cr.decode_flat, cr.encode_flat), (cr.decode_bound, cr.encode_bound)):
            output = decode(encode([record]))[0]
            self.assertIs(type(output["value"]["raw_number"]), int)
            self.assertIs(type(output["value"]["decimal"]), float)
            self.assertIs(type(output["value"]["enabled"]), bool)
            self.assertIsNone(output["value"]["unit_raw"])
            self.assertEqual(math.copysign(1, output["value"]["negative_zero"]), -1)
        cr.validate_equal_information([record])

    def test_strict_types_also_control_grouping(self):
        records = [reading("1", conditions=[{"value": 1}]),
                   reading("2", conditions=[{"value": 1.0}]),
                   reading("3", conditions=[{"value": True}])]
        tree = json.loads(cr.encode_bound(records))
        self.assertEqual(len(tree[0]["groups"][0]["groups"][0]["groups"]), 3)
        cr.validate_equal_information(records)
        for field, before, after in (("n", 1, True), ("n", 1, 1.0), ("n", -0.0, 0.0),
                                     ("n", None, "null")):
            original = reading(); original[field] = before
            tampered = deepcopy(original); tampered[field] = after
            for side in ("flat", "bound"):
                encoded = getattr(cr, f"encode_{side}")([tampered])
                with self.assertRaises(ValueError):
                    cr.validate_equal_information([original], **{side: encoded})

    def test_subject_or_condition_value_swap_is_not_equal_information(self):
        for second in (reading("r2", subjects=["C"], value="20 W"),
                       reading("r2", conditions=[{"dimension": "mode", "value": "standby"}], value="20 W")):
            records = [reading(), second]
            for side in ("flat", "bound"):
                encoded = getattr(cr, f"encode_{side}")(records)
                tree, leaves = (json.loads(encoded), None) if side == "flat" else bound_leaves(encoded)
                leaves = tree if leaves is None else leaves
                leaves[0]["value"], leaves[1]["value"] = leaves[1]["value"], leaves[0]["value"]
                with self.assertRaises(ValueError):
                    cr.validate_equal_information(records, **{side: json.dumps(tree)})

    def test_record_and_nested_sequence_reordering_rejected(self):
        records = [reading(), reading("r2", subjects=["C"], value="20 W")]
        for side in ("flat", "bound"):
            with self.assertRaises(ValueError):
                cr.validate_equal_information(records, **{side: getattr(cr, f"encode_{side}")(records[::-1])})
        for field in ("subjects", "conditions_all"):
            original = reading(conditions=[{"dimension": "mode", "value": "active"},
                                           {"dimension": "temperature", "value": 20}])
            changed = deepcopy(original); changed[field].reverse()
            with self.assertRaises(ValueError):
                cr.validate_equal_information([original], bound=cr.encode_bound([changed]))

    def test_every_record_field_and_citation_detail_is_checked(self):
        records = [reading()]
        for side in ("flat", "bound"):
            encoded = getattr(cr, f"encode_{side}")(records)
            for field in ("record_id", "family_id", "value", "citation", "locator", "annotation_note"):
                tree, leaves = (json.loads(encoded), None) if side == "flat" else bound_leaves(encoded)
                leaf = tree[0] if leaves is None else leaves[0]
                del leaf[field]
                with self.assertRaises(ValueError):
                    cr.validate_equal_information(records, **{side: json.dumps(tree)})
            tree, leaves = (json.loads(encoded), None) if side == "flat" else bound_leaves(encoded)
            leaf = tree[0] if leaves is None else leaves[0]
            leaf["citation"]["spans"][0]["start"] = 5
            with self.assertRaises(ValueError):
                cr.validate_equal_information(records, **{side: json.dumps(tree)})

    def test_group_field_omission_override_and_ghost_branch_rejected(self):
        encoded = cr.encode_bound([reading()])
        tree = json.loads(encoded); del tree[0]["subjects"]
        with self.assertRaises(ValueError):
            cr.validate_equal_information([reading()], bound=json.dumps(tree))
        tree, leaves = bound_leaves(encoded); leaves[0]["subjects"] = ["other"]
        with self.assertRaises(ValueError):
            cr.decode_bound(json.dumps(tree))
        tree = json.loads(encoded); tree[0]["unexpected"] = "new claim"
        with self.assertRaises(ValueError):
            cr.decode_bound(json.dumps(tree))
        with self.assertRaises(ValueError):
            cr.validate_equal_information([], bound='[{"subjects":["ghost"],"groups":[]}]')

    def test_non_json_values_and_ambiguous_encoded_objects_rejected(self):
        for value in (float("nan"), float("inf"), (1, 2), {1: "coerced key"}):
            record = reading(); record["extra"] = value
            for encode in (cr.encode_flat, cr.encode_bound):
                with self.assertRaises(ValueError):
                    encode([record])
        record = reading(); record["cycle"] = record
        with self.assertRaises(ValueError):
            cr.encode_bound([record])
        for decode in (cr.decode_flat, cr.decode_bound):
            for encoded in ('[{"x":1,"x":2}]', '[NaN]', '[1e999]'):
                with self.assertRaises(ValueError):
                    decode(encoded)


if __name__ == "__main__":
    unittest.main()
