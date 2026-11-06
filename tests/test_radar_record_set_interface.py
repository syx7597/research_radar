from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.radar_domain.record_set_interface import (
    RecordSetEpisode, SourceRecordStore, execute_response, load_bridge,
    prototype_queries, run_actions, tool_schemas, from_graph)
from experiments.radar_domain.kopl_record_bridge import build_graph, load_snapshot, reading_id


class RecordSetInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor, cls.store = load_bridge()
        cls.prefix = prototype_queries()[1]["program"][:-1]

    def episode_with_set(self, number="15 A", max_calls=24):
        episode = RecordSetEpisode(self.executor, self.store, max_calls=max_calls)
        program = deepcopy(self.prefix)
        program[-1]["inputs"][1] = number
        for action in program:
            result = episode.step(**action)
            self.assertTrue(result["ok"], result)
        return episode

    def test_selected_records_preserve_values_conditions_and_citations(self):
        episode = self.episode_with_set()
        result = episode.finish_records(3)
        self.assertTrue(result["ok"])
        rows = result["answer"]["records"]
        self.assertEqual([r["record_id"] for r in rows], ["RAY-0051", "RAY-0052"])
        self.assertEqual([r["value"]["text"] for r in rows], ["15 A", "15 A"])
        self.assertEqual([r["conditions_all"][0]["value"] for r in rows], ["24 V", "12 V"])
        for row in rows:
            self.assertEqual(row, self.store._records[reading_id(row["record_id"])])
            source = result["answer"]["sources"][row["citation"]["chunk_id"]]
            self.assertTrue(source["source_uri"].startswith("https://"))
            self.assertNotIn("text", source)  # no whole-page answer replacement
        other = self.episode_with_set("20 A")
        other.finish_records(3)
        self.assertEqual([r["record_id"] for r in other.answer["records"]], ["RAY-0050"])
        self.assertEqual(set(episode.answer["sources"]), set(other.answer["sources"]))
        self.assertNotEqual(episode.answer["records"], other.answer["records"])

    def test_empty_collection_and_zero_count_are_valid_distinct_answer_types(self):
        episode = self.episode_with_set("99 A")
        result = episode.finish_records(3)
        self.assertEqual(result["answer"]["records"], [])
        self.assertTrue(episode.done)
        scalar = self.episode_with_set("99 A")
        self.assertTrue(scalar.step("Count", [], [3])["ok"])
        result = scalar.finish(4)
        self.assertEqual(result["answer"]["value"], {"type": "int", "value": 0})
        self.assertEqual(scalar.prediction, "0")
        self.assertEqual(result["answer"]["source_records"], [])

    def test_invalid_or_nonreading_handles_rejected_then_can_recover(self):
        for handle in (True, -1, 99, 0):
            episode = self.episode_with_set()
            count = len(episode.handles)
            self.assertFalse(episode.finish_records(handle)["ok"], handle)
            self.assertFalse(episode.done)
            self.assertIsNone(episode.answer)
            self.assertEqual(len(episode.handles), count)
            self.assertTrue(episode.finish_records(3)["ok"])
        episode = self.episode_with_set()
        episode.step("Count", [], [3])
        self.assertFalse(episode.finish_records(4)["ok"])
        self.assertTrue(episode.finish(4)["ok"])

    def test_legacy_first_item_functions_guard_multirecord_input(self):
        for function, inputs in (("What", []), ("QueryAttr", ["record_id"])):
            episode = self.episode_with_set()
            result = episode.step(function, inputs, [3])
            self.assertFalse(result["ok"])
            self.assertIn("single_entity_required", result["detail"])
            self.assertEqual(len(episode.handles), 4)
            self.assertTrue(episode.finish_records(3)["ok"])

    def test_count_keeps_its_scalar_value_and_actual_support(self):
        episode = self.episode_with_set()
        episode.step("Count", [], [3])
        result = episode.finish(4)
        self.assertEqual(episode.prediction, "2")
        self.assertEqual(result["answer"]["value"], {"type": "int", "value": 2})
        self.assertEqual([r["record_id"] for r in result["answer"]["source_records"]],
                         ["RAY-0051", "RAY-0052"])

    def test_full_snapshot_has_no_silent_output_truncation(self):
        episode = RecordSetEpisode(self.executor, self.store)
        episode.step("FindAll", [], [])
        episode.step("FilterConcept", ["Specification reading"], [0])
        result = episode.finish_records(1)
        rows = result["answer"]["records"]
        self.assertEqual(len(rows), 502)
        for row in rows:
            self.assertEqual(row, self.store._records[reading_id(row["record_id"])])
        ids = [r["record_id"] for r in rows]
        self.assertEqual(ids, sorted(set(ids)))

    def test_result_mutation_does_not_change_source_store_or_final_answer(self):
        episode = self.episode_with_set()
        result = episode.finish_records(3)
        result["answer"]["records"][0]["value"]["text"] = "fabricated"
        self.assertEqual(episode.answer["records"][0]["value"]["text"], "15 A")
        other = self.episode_with_set()
        self.assertEqual(other.finish_records(3)["answer"]["records"][0]["value"]["text"], "15 A")

    def test_format_validation_and_last_call_completion(self):
        episode = self.episode_with_set(max_calls=5)
        result = execute_response(episode, '<tool_call>{"name":"finish_records","arguments":{"record_handle":3}}</tool_call>')
        self.assertTrue(result["ok"])
        self.assertEqual(episode.calls, 5)
        self.assertTrue(episode.done)
        self.assertIsNotNone(episode.answer)
        episode = self.episode_with_set()
        result = execute_response(episode, '<tool_call>{"name":"finish_records","arguments":{"record_handle":3,"answer":"injected"}}</tool_call>')
        self.assertFalse(result["ok"])
        self.assertIsNone(episode.answer)
        self.assertEqual([s["function"]["name"] for s in tool_schemas()], ["finish", "finish_records", "step"])
        result = execute_response(episode, '<tool_call>{"name":"step","arguments":{"function":[],"inputs":[],"dependencies":[]}}</tool_call>')
        self.assertFalse(result["ok"])

    def test_projection_mismatch_is_rejected(self):
        rows, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(rows)
        bad = deepcopy(graph)
        attr = next(a for a in bad["entities"][reading_id("RAY-0051")]["attributes"] if a["key"] == "value_text")
        attr["value"]["value"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Execution field differs"):
            from_graph(bad, chunks)

    def test_explicit_legacy_graph_rejects_negative_sign_error(self):
        with self.assertRaisesRegex(ValueError, "Numeric projection differs"):
            load_bridge(
                Path("data/radar_domain_recovery_v1/kopl_bridge/kb.json"),
                Path("data/radar_domain_recovery_v1/kopl_bridge/source_chunks.json"))

    def test_changed_quantity_type_unit_or_missing_projection_is_rejected(self):
        rows, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(rows)
        for change in ("unit", "type", "missing", "bool"):
            with self.subTest(change=change):
                bad = deepcopy(graph)
                attrs = bad["entities"][reading_id("RAY-0051")]["attributes"]
                attr = next(a for a in attrs if a["key"] == "scalar_value")
                if change == "unit":
                    attr["value"]["unit"] = "mA"
                elif change == "type":
                    attr["value"] = {"type": "string", "value": "15"}
                elif change == "bool":
                    attr["value"]["value"] = True
                else:
                    attrs.remove(attr)
                with self.assertRaisesRegex(ValueError, "Numeric projection differs"):
                    from_graph(bad, chunks)

    def test_duplicate_extra_or_nonprojectable_quantity_is_rejected(self):
        rows, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(rows)
        range_id = next(r["record_id"] for r in rows if r["value"]["form"] == "range")
        for change in ("duplicate", "other_numeric_key", "unexpected_quantity_key", "range_as_scalar"):
            with self.subTest(change=change):
                bad = deepcopy(graph)
                attrs = bad["entities"][reading_id("RAY-0051")]["attributes"]
                copied = deepcopy(next(a for a in attrs if a["key"] == "scalar_value"))
                if change == "other_numeric_key":
                    copied["key"] = "upper_bound_value"
                elif change == "unexpected_quantity_key":
                    copied["key"] = "invented_numeric_value"
                elif change == "range_as_scalar":
                    attrs = bad["entities"][reading_id(range_id)]["attributes"]
                attrs.append(copied)
                with self.assertRaisesRegex(ValueError, "[Nn]umeric projection"):
                    from_graph(bad, chunks)

    def test_source_text_changed_outside_spans_is_rejected(self):
        rows, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(rows)
        changed = deepcopy(chunks)
        key = rows[0]["citation"]["chunk_id"]
        changed[key]["text"] += " appended text outside every existing citation span"
        self.assertEqual(changed[key]["text_sha256"], chunks[key]["text_sha256"])
        for row in rows:
            if row["citation"]["chunk_id"] == key:
                for span in row["citation"]["spans"]:
                    self.assertEqual(changed[key]["text"][span["start"]:span["end"]], span["quote"])
        with self.assertRaisesRegex(ValueError, "Actual source text hash differs"):
            from_graph(graph, changed)

    def test_corrected_graph_explicit_load_keeps_real_negative_quantity(self):
        rows, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(rows)
        with tempfile.TemporaryDirectory() as directory:
            kb = Path(directory) / "kb.json"; kb.write_text(json.dumps(graph))
            source = Path(directory) / "chunks.json"; source.write_text(json.dumps(chunks))
            executor, store = load_bridge(kb, source)
        attrs = executor.engine.entities[reading_id("MRR-0047")]["attributes"]
        quantity = next(a["value"] for a in attrs if a["value"].type == "quantity")
        self.assertEqual((quantity.value, quantity.unit), (-8.0, "dBZ"))
        self.assertIn("- 8", store._records[reading_id("MRR-0047")]["value"]["text"])
        self.assertEqual(store.count, 502)


if __name__ == "__main__":
    unittest.main()
