from copy import deepcopy
import unittest

from experiments.radar_domain.dual_view_audit import check_views, read, GRAPH, SOURCE, TYPED, CHUNKS


class DualViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.sidecar, cls.typed, cls.chunks = [read(x) for x in (GRAPH, SOURCE, TYPED, CHUNKS)]

    def test_complete_existing_views_retain_shared_subjects(self):
        counts, scopes = check_views(self.graph, self.sidecar, self.typed, self.chunks)
        self.assertEqual(counts["readings"], 502)
        self.assertEqual(counts["subject_nodes"], 23)
        self.assertEqual(counts["shared_subject_readings"], 203)
        self.assertGreater(len(scopes), 502)

    def test_removing_or_redirecting_a_graph_edge_is_detected(self):
        graph = deepcopy(self.graph)
        node = next(v for v in graph["entities"].values() if v["instanceOf"] == ["C-subject"])
        node["relations"].pop()
        with self.assertRaises(ValueError):
            check_views(graph, self.sidecar, self.typed, self.chunks)

    def test_typed_condition_and_original_fact_tampering_detected(self):
        for key in ("conditions_all", "raw_record"):
            typed = deepcopy(self.typed)
            if key == "conditions_all":
                typed["records"][0][key] = [{"dimension": "invented", "value": "condition"}]
            else:
                typed["records"][0][key]["value"]["text"] = "altered source text"
            with self.assertRaises(ValueError):
                check_views(self.graph, self.sidecar, typed, self.chunks)

    def test_source_citation_text_tampering_detected(self):
        chunks = deepcopy(self.chunks)
        citation = self.typed["records"][0]["citation"]
        chunks[citation["chunk_id"]]["text"] += "altered"
        with self.assertRaises(ValueError):
            check_views(self.graph, self.sidecar, self.typed, chunks)


if __name__ == "__main__":
    unittest.main()
