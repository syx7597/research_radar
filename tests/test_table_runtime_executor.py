from copy import deepcopy
from decimal import InvalidOperation
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from tests.test_table_inference import FixtureTokenizer, FixtureGenerator, reference_fixture
from experiments.radar_domain.table_program import TableExecutor, execute_actions, action
from experiments.radar_domain.table_runtime_executor import TableRuntimeExecutor
from experiments.radar_domain.table_inference import run_episode, BUDGET
from experiments.radar_domain.table_recovery_data import action_text, replay, make_pair


class RuntimeExecutorTests(unittest.TestCase):
    def test_bad_model_numeric_columns_and_conditions_continue_and_replay(self):
        catalog, ref, _ = reference_fixture()
        bad = [action("Filter", {"predicates": [{"column": "typed_value", "op": "<", "value": "abc", "unit": "W"}]}, [0]),
               action("Project", {"columns": []}, [0]), action("Scan", {"conditions": []})]
        for invalid in bad:
            executor, tokenizer = TableRuntimeExecutor(catalog), FixtureTokenizer()
            actions = ref["actions"][:1] + [invalid] + ref["actions"][1:]
            row = run_episode({"id": ref["id"], "question": ref["question"]}, executor, tokenizer,
                              FixtureGenerator(tokenizer, [action_text(a) for a in actions]),
                              budgets={**BUDGET, "per_turn_generated_tokens": 512})
            self.assertEqual(row["stop_reason"], "finished")
            self.assertEqual((row["calls"], row["invalid_calls"]), (5, 1))
            self.assertEqual(row["events"][1]["observation"]["error"], "ValueError")
            self.assertTrue(row["events"][1]["observation"]["detail"].startswith("invalid_operation_configuration:"))
            self.assertEqual(row["events"][2]["observation"]["handle"], 1)
            self.assertEqual(row["answer"], ref["answer"])
            self.assertEqual(replay(executor, ref, row, 24).answer, ref["answer"])

    def test_failed_invalid_configuration_can_form_a_replayed_recovery_pair(self):
        catalog, ref, _ = reference_fixture(); tokenizer = FixtureTokenizer(); executor = TableRuntimeExecutor(catalog)
        bad = action("Filter", {"predicates": [{"column": "typed_value", "op": "<", "value": "abc", "unit": "W"}]}, [0])
        row = run_episode({"id": ref["id"], "question": ref["question"]}, executor, tokenizer,
                          FixtureGenerator(tokenizer, [action_text(ref["actions"][0]), action_text(bad)]),
                          budgets={**BUDGET, "per_turn_generated_tokens": 512})
        self.assertEqual(row["events"][1]["observation"]["error"], "ValueError")
        pair = make_pair(executor, ref, row)
        self.assertEqual(pair["audit"]["status"], "accepted")
        self.assertEqual(pair["audit"]["shared_prefix"], 1)
        self.assertFalse(pair["audit"]["training_eligible"])
        self.assertEqual(pair["recovery"]["answer"], ref["answer"])

    def test_only_registered_parameter_exceptions_are_normalized(self):
        catalog, _, _ = reference_fixture(); executor = TableRuntimeExecutor(catalog)
        for exception in (InvalidOperation, AttributeError, OverflowError):
            with patch.object(TableExecutor, "operate", side_effect=exception("bad model parameter")):
                with self.assertRaisesRegex(ValueError, "invalid_operation_configuration:" + exception.__name__):
                    executor.operate("Filter", {}, [])
        for exception in (OSError, MemoryError, RuntimeError):
            with patch.object(TableExecutor, "operate", side_effect=exception("infrastructure failure")):
                with self.assertRaises(exception):
                    executor.operate("Filter", {}, [])

    def test_all_complete_real_development_programs_preserve_valid_events_and_answers(self):
        folder = Path("data/radar_domain_recovery_v2/frozen_20261107/dev")
        catalog = json.loads((folder/"catalog.json").read_text())
        references = [json.loads(s) for s in (folder/"references.jsonl").read_text().splitlines()
                      if json.loads(s)["id"].startswith("RDR-DEV-")]
        self.assertEqual(len(references), 12)
        original, runtime = TableExecutor(catalog), TableRuntimeExecutor(catalog)
        for ref in references:
            a = execute_actions(original, ref["question"], ref["actions"])
            b = execute_actions(runtime, ref["question"], ref["actions"])
            self.assertEqual((a.messages, a.events, a.answer), (b.messages, b.events, b.answer))


if __name__ == "__main__":
    unittest.main()
