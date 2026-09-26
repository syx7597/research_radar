"""Regression coverage for official Find's in-place concept-ID accumulation.

Pure adapter tests always run. Integration tests use the genuine downloaded
RuleExecutor against a tiny temporary KB, and skip only if its source is absent.
No public dataset, model weights or GPU are needed.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .executor import ROOT, KoPLExecutor, isolate_find_state


class MutatingFind:
    """The official method's observable list-aliasing behavior, including errors."""
    def __init__(self):
        self.entity_name_to_ids = defaultdict(list, {"shared": ["E"]})
        self.concept_name_to_ids = {"shared": ["C"], "concept": ["ONLY_C"]}
        self.fail = False

    def Find(self, dependencies, inputs):
        name = inputs[0]
        ids = self.entity_name_to_ids[name]
        if name in self.concept_name_to_ids:
            ids += self.concept_name_to_ids[name]
        if self.fail:
            raise RuntimeError("injected official failure after mutation")
        return ids, None


class FindIsolationTests(unittest.TestCase):
    def test_existing_list_identity_and_previous_return_value_stay_independent(self):
        engine = MutatingFind()
        original = engine.entity_name_to_ids["shared"]
        isolate_find_state(engine)
        first, _ = engine.Find([], ["shared"])
        second, _ = engine.Find([], ["shared"])
        self.assertEqual(first, ["E", "C"])
        self.assertEqual(second, ["E", "C"])
        self.assertIs(engine.entity_name_to_ids["shared"], original)
        self.assertEqual(original, ["E"])
        second.append("not_a_persistent_entity")
        self.assertEqual(first, ["E", "C"])
        self.assertEqual(original, ["E"])

    def test_concept_only_and_unknown_names_do_not_create_persistent_entries(self):
        engine = MutatingFind()
        isolate_find_state(engine)
        for _ in range(3):
            self.assertEqual(engine.Find([], ["concept"]), (["ONLY_C"], None))
            self.assertEqual(engine.Find([], ["unknown"]), ([], None))
            self.assertEqual(dict(engine.entity_name_to_ids), {"shared": ["E"]})

    def test_failure_restores_both_existing_and_absent_entries(self):
        engine = MutatingFind()
        original = engine.entity_name_to_ids["shared"]
        isolate_find_state(engine)
        engine.fail = True
        for name in ("shared", "concept", "unknown"):
            with self.assertRaisesRegex(RuntimeError, "injected official failure"):
                engine.Find([], [name])
            self.assertEqual(dict(engine.entity_name_to_ids), {"shared": ["E"]})
            self.assertIs(engine.entity_name_to_ids["shared"], original)


OFFICIAL_SOURCE = ROOT / "external/kqa_pro_baselines/Program/executor_rule.py"


@unittest.skipUnless(OFFICIAL_SOURCE.is_file(), "Official RuleExecutor source is not downloaded")
class OfficialExecutorStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kopl_state_test_")
        self.addCleanup(self.temp.cleanup)
        kb = {
            "concepts": {
                "C_ENGLISH": {"name": "English", "instanceOf": []},
                "C_SHARED": {"name": "Shared", "instanceOf": []},
            },
            "entities": {
                "E_ALPHA": {
                    "name": "Alpha", "instanceOf": ["C_SHARED"],
                    "attributes": [{"key": "height", "value": {"type": "quantity", "value": 2, "unit": "metre"}, "qualifiers": {}}],
                    "relations": [{"predicate": "language", "object": "C_ENGLISH", "direction": "forward", "qualifiers": {}}],
                },
                "E_SHARED": {"name": "Shared", "instanceOf": ["C_SHARED"], "attributes": [], "relations": []},
            },
        }
        path = Path(self.temp.name) / "kb.json"
        path.write_text(json.dumps(kb))
        self.executor = KoPLExecutor(path, backend="baseline")
        self.initial_name_map = copy.deepcopy(dict(self.executor.engine.entity_name_to_ids))
        self.initial_entities = copy.deepcopy(self.executor.engine.entities)
        self.initial_concepts = copy.deepcopy(self.executor.engine.concepts)

    def assert_prediction(self, program, expected):
        result = self.executor.execute(program)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["prediction"], expected)
        self.assertEqual(dict(self.executor.engine.entity_name_to_ids), self.initial_name_map)
        self.assertEqual(self.executor.engine.entities, self.initial_entities)
        self.assertEqual(self.executor.engine.concepts, self.initial_concepts)

    def test_repeated_concept_and_colliding_name_counts_are_constant(self):
        for _ in range(4):
            self.assert_prediction("Find <arg> English <func> Count", "1")
            self.assert_prediction("Find <arg> Shared <func> Count", "2")
            self.assert_prediction("Find <arg> Unknown <func> Count", "0")

    def test_interleaved_relation_and_set_programs_preserve_state(self):
        programs = [
            ("Find <arg> English <func> Relate <arg> language <arg> backward <func> Count", "1"),
            ("Find <arg> Alpha <func> Relate <arg> language <arg> forward <func> Count", "1"),
            ("Find <arg> English <func> Find <arg> Alpha <func> Or <func> Count", "2"),
            ("Find <arg> Shared <func> Find <arg> Alpha <func> And <func> Count", "0"),
        ]
        for order in (programs, list(reversed(programs)), programs):
            for program, expected in order:
                self.assert_prediction(program, expected)
            self.assert_prediction("Find <arg> English <func> Count", "1")
            self.assert_prediction("Find <arg> Shared <func> Count", "2")

    def test_extra_local_repair_prefix_execution_does_not_change_future_baseline(self):
        from .repair import Repairer
        repairer = Repairer(self.executor, mode="local")
        row = {"id": "concept_prefix", "question": "How many entities use English?", "candidates": [
            {"program_text": "Find <arg> English <func> Relate <arg> language typo <arg> backward <func> Count"}]}
        for _ in range(3):
            result = repairer.repair_row(row)
            repairs = result["candidates"][1:]
            self.assertEqual(len(repairs), 1)
            self.assertEqual(repairs[0]["edit"]["to"], "language")
            self.assertEqual(repairs[0]["prediction"], "1")
            self.assert_prediction("Find <arg> English <func> Count", "1")
            self.assert_prediction("Find <arg> Shared <func> Count", "2")


if __name__ == "__main__":
    unittest.main()
