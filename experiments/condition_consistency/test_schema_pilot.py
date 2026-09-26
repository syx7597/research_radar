"""Regression for finished-batch padding; final constraint checks stay strict."""
import unittest

from .schema_pilot import PaddingSafeCallback, check_trace, field_issues
from .schema_constraints import SchemaFieldConstraint
from .test_schema_constraints import TinyTokenizer


class PaddingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.schema = {"attribute": ["mass"], "relation": ["country"], "qualifier": ["start time"]}
        self.constraint = SchemaFieldConstraint(self.tokenizer, self.schema)
        self.callback = PaddingSafeCallback(self.constraint)

    def prefix(self, text):
        return [2, 0] + self.tokenizer.encode(text)

    def test_finished_placeholder_can_only_continue_padding(self):
        tokens = self.prefix("QueryAttr <arg> mas") + [1]
        self.assertEqual(self.callback(3, tokens), [1])
        self.assertEqual(self.callback(3, tokens + [1]), [1])
        self.assertEqual(self.callback.padding_calls, 2)

    def test_placeholder_exception_cannot_validate_a_returned_field(self):
        tokens = self.prefix("QueryAttr <arg> mas") + [1, 2]
        self.assertIsNotNone(check_trace(tokens, self.constraint))

    def test_normal_field_rules_and_returned_program_unchanged(self):
        tokens = self.prefix("QueryAttr <arg> mas")
        self.assertEqual(self.callback(3, tokens), self.constraint(3, tokens))
        self.assertIsNone(check_trace(self.prefix("QueryAttr <arg> mass") + [2], self.constraint))
        self.assertEqual(self.callback.padding_calls, 0)

    def test_role_violations_are_separate_from_unconstrained_slots(self):
        issues, count = field_issues("Find <arg> mystery <func> FilterConcept <arg> unknown "
                                    "<func> QueryAttr <arg> country", self.schema)
        self.assertEqual(count, 1)
        self.assertEqual(issues[0]["role"], "attribute")
        self.assertEqual(issues[0]["value"], "country")


if __name__ == "__main__":
    unittest.main()
