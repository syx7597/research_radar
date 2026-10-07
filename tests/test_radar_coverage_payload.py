"""Synthetic tests distinguish semantic projection from complete reconstruction."""
from copy import deepcopy
import json
import unittest

from experiments.radar_domain import coverage_payload as payload
from experiments.radar_domain import coverage_representation as representation


def reading(rid="r1", subject="A", condition="active", value="10 W", quote="10 W plus source-only detail"):
    return {"record_id": rid, "family_id": "fixture-family", "subjects": [subject, "shared"],
            "component": "receiver", "attribute": "power",
            "value": {"form": "scalar", "text": value, "unit_raw": None},
            "conditions_all": [{"dimension": "mode", "value": condition}],
            "citation": {"chunk_id": "fixture:p001", "text_sha256": "source-text-hash",
                         "spans": [{"start": 4, "end": 4 + len(quote), "quote": quote}]},
            "locator": "Table 1, power", "annotation_note": "必须保留的技术条件"}


class CitationProjection(unittest.TestCase):
    def assert_typed_equal(self, expected, actual):
        representation.validate_equal_information(expected, flat=representation.encode_flat(actual),
                                                  bound=representation.encode_bound(actual))

    def test_only_two_citation_fields_move_and_all_semantic_fields_stay(self):
        original = [reading()]
        saved = deepcopy(original)
        semantic, sidecar = payload.project_records(original)
        self.assertEqual(set(semantic[0]), payload.RECORD_FIELDS)
        self.assertEqual(semantic[0]["citation"], {"chunk_id": "fixture:p001"})
        for key in payload.RECORD_FIELDS - {"citation"}:
            self.assertEqual(semantic[0][key], original[0][key])
        removed = sidecar["entries"][0]["removed_citation"]
        self.assertEqual(set(removed), {"text_sha256", "spans"})
        self.assertEqual(removed["spans"], original[0]["citation"]["spans"])
        self.assertEqual(sidecar["entries"][0]["position"], 0)
        self.assertEqual(sidecar["entries"][0]["record_id"], "r1")
        self.assertTrue(sidecar["full_reconstruction_requires_sidecar"])
        self.assert_typed_equal(original, payload.restore_records(semantic, sidecar))
        self.assert_typed_equal(saved, original)
        semantic[0]["subjects"].append("changed")
        removed["spans"][0]["quote"] = "changed"
        self.assert_typed_equal(saved, original)

    def test_empty_projection_and_full_restore(self):
        semantic, sidecar = payload.project_records([])
        self.assertEqual(semantic, [])
        self.assertEqual(sidecar["entries"], [])
        self.assertEqual(payload.restore_records(semantic, sidecar), [])
        with self.assertRaises(ValueError):
            payload.restore_records([], None)

    def test_duplicate_occurrences_shared_subjects_and_original_sequence_survive(self):
        one = reading(); two = reading("r2", subject="B", condition="standby", value="2 W")
        original = [one, one, two, deepcopy(one)]
        semantic, sidecar = payload.project_records(original)
        self.assertEqual([r["record_id"] for r in semantic], ["r1", "r1", "r2", "r1"])
        self.assertEqual([e["position"] for e in sidecar["entries"]], [0, 1, 2, 3])
        self.assertEqual(len(semantic), 4)
        self.assert_typed_equal(original, payload.restore_records(semantic, sidecar))
        restored = payload.restore_records(semantic, sidecar)
        restored[0]["citation"]["spans"][0]["quote"] = "changed"
        self.assertNotEqual(restored[1]["citation"]["spans"][0]["quote"], "changed")
        self.assertEqual(semantic[0]["subjects"], ["A", "shared"])

    def test_semantic_types_are_preserved_and_type_changes_invalidate_binding(self):
        for value in (None, 1, 1.0, True, "1", -0.0):
            original = [reading(value=value)]
            semantic, sidecar = payload.project_records(original)
            self.assert_typed_equal(original, payload.restore_records(semantic, sidecar))
        for before, after in ((1, True), (1, 1.0), (None, "null"), (-0.0, 0.0)):
            semantic, sidecar = payload.project_records([reading(value=before)])
            semantic[0]["value"]["text"] = after
            with self.assertRaises(ValueError):
                payload.restore_records(semantic, sidecar)

    def test_payload_subject_condition_value_and_citation_changes_are_detected(self):
        original = [reading(), reading("r2", subject="B", condition="standby", value="2 W")]
        semantic, sidecar = payload.project_records(original)
        mutations = [lambda r: r[0]["subjects"].reverse(),
                     lambda r: r[0]["conditions_all"][0].update(value="standby"),
                     lambda r: r[0]["citation"].update(chunk_id="different:p001"),
                     lambda r: r[0].update(annotation_note="omitted technical condition"),
                     lambda r: r.reverse()]
        for mutate in mutations:
            changed = deepcopy(semantic); mutate(changed)
            with self.assertRaises(ValueError):
                payload.restore_records(changed, sidecar)
        changed = deepcopy(semantic)
        changed[0]["value"], changed[1]["value"] = changed[1]["value"], changed[0]["value"]
        with self.assertRaises(ValueError):
            payload.restore_records(changed, sidecar)
        with self.assertRaises(ValueError):
            payload.restore_records(semantic[:1], sidecar)

    def test_sidecar_positions_ids_per_record_hash_and_removed_data_are_bound(self):
        original = [reading(), reading("r2", value="2 W")]
        semantic, sidecar = payload.project_records(original)
        mutations = [lambda s: s["entries"].reverse(),
                     lambda s: s["entries"][0].update(position=True),
                     lambda s: s["entries"][0].update(record_id="r2"),
                     lambda s: s["entries"][0].update(semantic_record_sha256="wrong"),
                     lambda s: s["entries"][0]["removed_citation"].update(text_sha256="wrong"),
                     lambda s: s["entries"][0]["removed_citation"]["spans"][0].update(quote="different")]
        for mutate in mutations:
            changed = deepcopy(sidecar); mutate(changed)
            with self.assertRaises(ValueError):
                payload.restore_records(semantic, changed)

    def test_identical_semantic_duplicates_cannot_swap_different_source_spans(self):
        original = [reading(quote="first quote"), reading(quote="second quote")]
        semantic, sidecar = payload.project_records(original)
        self.assertEqual(semantic[0], semantic[1])
        changed = deepcopy(sidecar); changed["entries"].reverse()
        with self.assertRaises(ValueError):
            payload.restore_records(semantic, changed)
        # Even resetting the visible indices does not hide swapped source text.
        for index, entry in enumerate(changed["entries"]):
            entry["position"] = index
        with self.assertRaisesRegex(ValueError, "complete records hash"):
            payload.restore_records(semantic, changed)

    def test_unknown_or_missing_current_schema_fields_are_rejected(self):
        mutations = [lambda r: r.update(new_field="extra"),
                     lambda r: r.pop("annotation_note"),
                     lambda r: r["citation"].update(source_uri="unregistered"),
                     lambda r: r["citation"].pop("text_sha256"),
                     lambda r: r["citation"]["spans"][0].update(new_offset=4),
                     lambda r: r["value"].update(normalized_value=10),
                     lambda r: r["conditions_all"][0].update(operator="AND")]
        for mutate in mutations:
            record = reading(); mutate(record)
            with self.assertRaises(ValueError):
                payload.project_records([record])
        semantic, sidecar = payload.project_records([reading()])
        for changed in ({**sidecar, "extra": "not ignored"}, {k: v for k, v in sidecar.items() if k != "entries"}):
            with self.assertRaises(ValueError):
                payload.restore_records(semantic, changed)
        changed = deepcopy(semantic); changed[0]["citation"]["spans"] = []
        with self.assertRaises(ValueError):
            payload.restore_records(changed, sidecar)

    def test_semantic_round_trip_does_not_claim_complete_source_reconstruction(self):
        original = [reading()]
        semantic, sidecar = payload.project_records(original)
        representation.validate_equal_information(semantic)
        self.assertNotIn("plus source-only detail", representation.encode_flat(semantic))
        self.assertIn("plus source-only detail", json.dumps(sidecar))
        with self.assertRaises(ValueError):
            representation.validate_equal_information(original, flat=representation.encode_flat(semantic))
        with self.assertRaises(ValueError):
            payload.restore_records(semantic, None)
        with self.assertRaises(TypeError):
            payload.restore_records(semantic)
        self.assert_typed_equal(original, payload.restore_records(semantic, sidecar))

    def test_json_sidecar_round_trip_and_object_key_order_are_stable(self):
        original = [reading()]
        semantic, sidecar = payload.project_records(original)
        reordered = [{key: semantic[0][key] for key in reversed(semantic[0])}]
        serialized_sidecar = json.loads(json.dumps(sidecar, ensure_ascii=False))
        self.assert_typed_equal(original, payload.restore_records(reordered, serialized_sidecar))


if __name__ == "__main__":
    unittest.main()
