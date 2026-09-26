"""Behavioral checks with a tiny tokenizer; real BART coverage is a separate audit."""
import re
from types import SimpleNamespace
import unittest

from experiments.condition_consistency.schema_constraints import (
    ConstraintPrefixError, SchemaFieldConstraint,
)


class TinyTokenizer:
    def __init__(self):
        pieces = ["<s>", "<pad>", "</s>", "<func>", "<arg>"] + list(
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-é<>")
        self.vocab = {piece: i for i, piece in enumerate(pieces)}
        self.inverse = {i: piece for piece, i in self.vocab.items()}
        self.bos_token_id, self.pad_token_id, self.eos_token_id = 0, 1, 2
        self.all_special_tokens = ["<s>", "<pad>", "</s>"]
        self.all_special_ids = [0, 1, 2]

    def encode(self, text, add_special_tokens=False):
        pieces = re.findall(r"<func>|<arg>|</s>|<pad>|<s>|.", text)
        return [self.vocab[piece] for piece in pieces]

    def decode(self, ids, **kwargs):
        return "".join(self.inverse[i] for i in ids
                       if not kwargs.get("skip_special_tokens", False) or i not in self.all_special_ids)

    def get_vocab(self):
        return self.vocab


class SchemaConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.constraint = SchemaFieldConstraint(self.tokenizer, {
            "attribute": ["date", "date of birth", "mass", "café"],
            "relation": ["part of", "country"], "qualifier": ["start time"],
            "concept": ["human"],
        })

    def ids(self, text):
        return [2, 0] + self.tokenizer.encode(text)

    def replay(self, text):
        ids = self.ids(text) + [2]
        for position in range(1, len(ids)):
            self.assertIn(ids[position], self.constraint(0, ids[:position]),
                          msg=f"Rejected token at {position}: {text}")

    def test_all_field_roles_and_spacing(self):
        for function, field in (("QueryAttr", "café"), ("Relate", "part of"),
                                ("QFilterStr", "start time")):
            for before in ("", " "):
                for after in ("", " "):
                    self.replay(f"Find <arg> X <func> {function} <arg>{before}{field}{after}")

    def test_short_field_also_allows_longer_field(self):
        state = self.constraint.inspect(self.ids("QueryAttr <arg> date"))
        self.assertIn(2, state["allowed"])
        self.assertIn(self.tokenizer.vocab[" "], state["allowed"])
        self.replay("QueryAttr <arg> date of birth")

    def test_partial_field_cannot_end_or_change_argument(self):
        allowed = self.constraint(0, self.ids("QueryAttr <arg> dat"))
        for piece in ("</s>", "<arg>", "<func>", "<pad>", "<s>"):
            self.assertNotIn(self.tokenizer.vocab[piece], allowed)

    def test_wrong_role_is_rejected(self):
        state = self.constraint.inspect(self.ids("QueryAttr <arg> country"))
        self.assertEqual(state["allowed"], [])
        with self.assertRaises(ConstraintPrefixError):
            self.constraint(0, self.ids("Relate <arg> mass"))

    def test_illegal_field_never_falls_back_to_free(self):
        with self.assertRaises(ConstraintPrefixError):
            self.constraint(0, self.ids("QueryAttr <arg> nonsense"))

    def test_unrestricted_arguments_stay_unrestricted(self):
        for text in ("Find <arg> Unknown", "FilterConcept <arg> Unknown",
                     "FilterStr <arg> mass <arg> Unknown",
                     "Relate <arg> country <arg> forward"):
            state = self.constraint.inspect(self.ids(text))
            self.assertEqual(state["reason"], "non_field_argument")
            self.assertIsNone(state["allowed"])

    def test_role_transitions_and_branch_boundaries(self):
        self.replay("Find <arg> X <func> QueryAttrQualifier <arg> mass <arg> 42 "
                    "<arg> start time")
        self.replay("QueryAttrUnderCondition <arg> date <arg> start time <arg> 2000")
        self.replay("QueryRelationQualifier <arg> country <arg> start time")

    def test_unknown_function_is_visible_fail_open(self):
        state = self.constraint.inspect(self.ids("Invented <arg> country"))
        self.assertEqual(state["reason"], "unknown_function")
        self.assertIsNone(state["allowed"])

    def test_special_token_cannot_disguise_known_function(self):
        prefix = self.ids("Query<s>Attr <arg> ")
        state = self.constraint.inspect(prefix)
        self.assertEqual(state["role"], "attribute")
        self.assertNotIn(self.tokenizer.vocab["n"], state["allowed"])

    def test_fragmented_delimiter_is_a_hard_failure(self):
        prefix = self.ids("QueryAttr ") + [self.tokenizer.vocab[c] for c in "<arg>"]
        self.assertEqual(self.constraint.inspect(prefix)["reason"], "non_atomic_delimiter")
        with self.assertRaises(ConstraintPrefixError):
            self.constraint(0, prefix)

    def test_decoder_start_eos_and_padding(self):
        self.assertEqual(self.constraint.inspect([2])["mode"], "free")
        self.assertEqual(self.constraint.inspect([2, 0])["mode"], "free")
        ended = self.ids("QueryAttr <arg> date") + [2, 1]
        self.assertEqual(self.constraint.inspect(ended)["allowed"], [1])
        with self.assertRaises(ConstraintPrefixError):
            self.constraint(0, ended + self.tokenizer.encode("x"))

    def test_beam_reordering_does_not_share_slot_state(self):
        prefix_a = self.ids("QueryAttr <arg> dat")
        prefix_b = self.ids("Find <arg> Anything")
        first = self.constraint(0, prefix_a)
        self.constraint(0, prefix_b)
        self.assertEqual(first, self.constraint(0, prefix_a))
        self.assertEqual(first, self.constraint(8, prefix_a))

    def test_forced_eos_requires_explicit_protocol_change(self):
        original = SimpleNamespace(forced_eos_token_id=2, num_beams=4)
        with self.assertRaises(ValueError):
            self.constraint.generation_kwargs(original)
        self.assertEqual(original.forced_eos_token_id, 2)
        config = SimpleNamespace(forced_eos_token_id=None, num_beams=4)
        kwargs = self.constraint.generation_kwargs(config)
        self.assertIs(kwargs["prefix_allowed_tokens_fn"], self.constraint)
        self.assertIsNot(kwargs["generation_config"], config)
        self.assertEqual(kwargs["generation_config"].num_beams, 4)

    def test_malformed_schema_or_special_delimiters_rejected(self):
        with self.assertRaises(ValueError):
            SchemaFieldConstraint(self.tokenizer, {"attribute": [" date"],
                "relation": ["country"], "qualifier": ["start time"]})
        self.tokenizer.all_special_tokens.append("<arg>")
        with self.assertRaises(ValueError):
            SchemaFieldConstraint(self.tokenizer, {"attribute": ["date"],
                "relation": ["country"], "qualifier": ["start time"]})


if __name__ == "__main__":
    unittest.main()
