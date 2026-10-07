"""Budget selection and common prompt boundaries without tokenizer/network."""
import unittest

from experiments.radar_domain.coverage_preflight import messages, SYSTEM_PROMPT
from experiments.radar_domain.coverage_semantic_preflight import context_choice


class PreflightContract(unittest.TestCase):
    def test_generated_reserve_and_every_arm_determine_one_common_cap(self):
        self.assertEqual(context_choice([12, 7424]), 8192)
        self.assertEqual(context_choice([12, 7425]), 32768)
        self.assertEqual(context_choice([32000, 12]), 32768)
        self.assertIsNone(context_choice([12, 32001]))
        with self.assertRaises(ValueError):
            context_choice([])

    def test_same_system_contract_for_raw_flat_and_bound(self):
        for evidence in ({"chunks": []}, [], [{"subjects": ["fixture"], "groups": []}]):
            result = messages("fixture question", evidence)
            self.assertEqual(result[0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertEqual(len(result), 2)
            self.assertNotIn("expected_fact", result[1]["content"])


if __name__ == "__main__":
    unittest.main()
