from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.condition_consistency.executor import KoPLExecutor
from experiments.radar_domain.kopl_record_bridge import (
    numeric_projection, load_snapshot, build_graph, prototype_queries, run_program, reading_id)


class BridgeTests(unittest.TestCase):
    def test_multivalue_or_unclear_units_do_not_become_single_numbers(self):
        for form, text, unit in [("range", "0 to 8 rpm", "rpm"), ("options", "2 or 4 kW", "kW"),
                                 ("correspondence", "4 kW + 4 kW", "kW"),
                                 ("scalar", "388 mm (15.3 in)", "mm; in"),
                                 ("scalar", "1", None), ("approximate", "about 4 kW", "kW")]:
            result, reason = numeric_projection({"value": {"form": form, "text": text, "unit_raw": unit}})
            self.assertIsNone(result, (form, text))

    def test_bound_endpoint_has_a_different_key_from_scalar(self):
        r = {"value": {"form": "upper_bound", "text": "Max. 3.1 kW", "unit_raw": "kW"}}
        bound, _ = numeric_projection(r)
        self.assertEqual((bound["key"], bound["number"], bound["unit"]), ("upper_bound_value", 3.1, "kW"))
        r["value"]["form"] = "scalar"
        scalar, _ = numeric_projection(r)
        self.assertEqual(scalar["key"], "scalar_value")

    def test_full_snapshot_and_shared_subjects_roundtrip(self):
        records, _, _ = load_snapshot()
        graph, sidecar, _ = build_graph(records)
        self.assertEqual(len(sidecar), 502)
        self.assertEqual(sum(v["instanceOf"] == ["C-reading"] for v in graph["entities"].values()), 502)
        for r in records:
            attrs = graph["entities"][reading_id(r["record_id"])]["attributes"]
            self.assertEqual(json.loads(next(a["value"]["value"] for a in attrs if a["key"] == "source_record_json")), r)
            self.assertEqual(sidecar[reading_id(r["record_id"])]["record"]["citation"], r["citation"])
        self.assertGreater(sum(len(r["subjects"]) > 1 for r in records), 0)

    def test_real_queries_and_threshold_change_use_actual_graph(self):
        records, _, _ = load_snapshot()
        graph, mapping, _ = build_graph(records)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "kb.json"; p.write_text(json.dumps(graph))
            executor = KoPLExecutor(p)
            first, second = prototype_queries()
            episode, selected = run_program(executor, first["program"])
            self.assertEqual(episode.prediction, "1")
            self.assertEqual([mapping[x]["record"]["record_id"] for x in selected], ["WX-0051"])
            changed = deepcopy(first["program"]); changed[-2]["inputs"][1] = "6 kW"
            episode, selected = run_program(executor, changed)
            self.assertEqual(episode.prediction, "2")
            self.assertEqual({mapping[x]["record"]["record_id"] for x in selected}, {"WX-0049", "WX-0051"})
            episode, selected = run_program(executor, second["program"])
            self.assertEqual(episode.prediction, "2")
            self.assertEqual({mapping[x]["record"]["record_id"] for x in selected}, {"RAY-0051", "RAY-0052"})
            changed = deepcopy(second["program"]); changed[-2]["inputs"][1] = "99 A"
            episode, selected = run_program(executor, changed)
            self.assertEqual((episode.prediction, selected), ("0", []))


if __name__ == "__main__":
    unittest.main()
