from copy import deepcopy
import unittest

from experiments.agent_feedback.state_transition_analysis import analyze, STATES


class StateTransitionsTest(unittest.TestCase):
    def fixtures(self):
        gold, a, c = [], [], []
        for x in STATES:
            for y in STATES:
                ident = x + y
                gold.append({"id": ident, "answer": "yes"})
                for rows, status in ((a, x), (c, y)):
                    rows.append({"id": ident, "prediction": None if status == "unfinished" else
                                 "yes" if status == "correct" else "no",
                                 "stop_reason": "call_budget" if status == "unfinished" else "finished"})
        return a, c, gold

    def test_all_nine_cells_and_balanced_net(self):
        report, cases = analyze(*self.fixtures(), "seed")
        self.assertTrue(all(v == 1 for row in report["row_A_column_C"].values() for v in row.values()))
        self.assertEqual(report["decomposition"]["total_correct_net"], 0)
        self.assertEqual(len(cases), 8)

    def test_missing_rows_and_duplicate_ids_rejected(self):
        a, c, gold = self.fixtures()
        with self.assertRaises(ValueError):
            analyze(a[:-1], c, gold, "seed")
        with self.assertRaises(ValueError):
            analyze(a + [a[0]], c, gold, "seed")

    def test_inconsistent_finish_and_missing_prediction_rejected(self):
        a, c, gold = self.fixtures()
        bad = deepcopy(a)
        bad[0]["stop_reason"] = "finished"
        with self.assertRaises(ValueError):
            analyze(bad, c, gold, "seed")
        del a[0]["prediction"]
        with self.assertRaises(ValueError):
            analyze(a, c, gold, "seed")

    def test_selection_independent_of_file_order(self):
        a, c, gold = self.fixtures()
        first, _ = analyze(a, c, gold, "seed")
        second, _ = analyze(a[::-1], c[::-1], gold[::-1], "seed")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
