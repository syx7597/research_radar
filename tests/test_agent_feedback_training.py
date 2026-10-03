"""Mask invariants using a fake tokenizer; not real-model template validation."""
import unittest

from experiments.agent_feedback.training_data import program_trajectory, tokenize_trajectory


class FakeTokenizer:
    """Each role, content character and end marker has an explicit token."""

    headers = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
    end = 5

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        assert tokenize
        tokens = []
        for message in messages:
            tokens.append(self.headers[message["role"]])
            tokens.extend(ord(c) + 10 for c in message["content"])
            tokens.append(self.end)
        if add_generation_prompt:
            tokens.append(self.headers["assistant"])
        return tokens


class NonPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        tokens = super().apply_chat_template(messages, **kwargs)
        # Simulate a template that rewrites an earlier turn when more messages
        # are supplied. The training code must reject this, not guess offsets.
        if tokens:
            tokens[0] = len(messages) + 1000
        return tokens


class TrainingMaskTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.row = {"id": "q1", "messages": [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "wrong action"},
            {"role": "tool", "content": "actual error feedback"},
            {"role": "assistant", "content": "correct recovery action"},
            {"role": "tool", "content": "actual execution result"},
            {"role": "assistant", "content": "finish"},
        ], "supervise": [False, False, False, False, True, False, True]}

    def test_bad_actions_and_tool_observations_are_context_only(self):
        result = tokenize_trajectory(self.tokenizer, self.row)
        self.assertEqual(result["input_ids"], self.tokenizer.apply_chat_template(self.row["messages"]))
        expected_indices = set()
        offset = 0
        for index, message in enumerate(self.row["messages"]):
            content_length = len(message["content"])
            if self.row["supervise"][index]:
                # Include content and end token, never the assistant header.
                expected_indices.update(range(offset + 1, offset + content_length + 2))
            offset += content_length + 2
        actual_indices = {i for i, token in enumerate(result["labels"]) if token != -100}
        self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(result["supervised_tokens"], len(expected_indices))
        self.assertEqual([span[0] for span in result["spans"]], [4, 6])
        for index in expected_indices:
            self.assertEqual(result["labels"][index], result["input_ids"][index])

    def test_label_positions_preserve_causal_prediction_boundary(self):
        result = tokenize_trajectory(self.tokenizer, self.row)
        for _, start, end in result["spans"]:
            self.assertEqual(result["input_ids"][start - 1], self.tokenizer.headers["assistant"])
            self.assertEqual(result["labels"][start - 1], -100)
            self.assertEqual(result["labels"][end - 1], self.tokenizer.end)

    def test_non_assistant_supervision_is_rejected(self):
        for index in (0, 1, 3, 5):
            row = {**self.row, "supervise": list(self.row["supervise"])}
            row["supervise"][index] = True
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Only assistant"):
                tokenize_trajectory(self.tokenizer, row)

    def test_mask_must_match_messages_and_include_a_target(self):
        for mask in ([False], [False] * len(self.row["messages"])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                tokenize_trajectory(self.tokenizer, {**self.row, "supervise": mask})

    def test_overflow_is_rejected_without_truncating_a_recovery_target(self):
        length = len(self.tokenizer.apply_chat_template(self.row["messages"]))
        with self.assertRaisesRegex(ValueError, "Context overflow"):
            tokenize_trajectory(self.tokenizer, self.row, max_length=length - 1)
        self.assertEqual(len(tokenize_trajectory(self.tokenizer, self.row, max_length=length)["input_ids"]), length)

    def test_template_prefix_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "previous token boundaries"):
            tokenize_trajectory(NonPrefixTokenizer(), self.row)

    def test_missing_explicit_mask_is_rejected(self):
        row = {k: value for k, value in self.row.items() if k != "supervise"}
        with self.assertRaisesRegex(ValueError, "explicit supervision mask"):
            tokenize_trajectory(self.tokenizer, row)

    def test_program_baseline_does_not_put_answer_into_prompt(self):
        row = program_trajectory({"id": "q1", "question": "Which entity?", "answer": "GOLD-ANSWER-MARKER",
                                  "program": [{"function": "Find", "inputs": ["Alpha"]},
                                              {"function": "What", "inputs": []}]})
        self.assertEqual(row["supervise"], [False, False, True])
        self.assertEqual(row["messages"][-1]["content"], "Find <arg> Alpha <func> What")
        self.assertNotIn("GOLD-ANSWER-MARKER", str(row["messages"]))
        self.assertTrue(tokenize_trajectory(self.tokenizer, row, tools=False)["supervised_tokens"] > 0)


if __name__ == "__main__":
    unittest.main()
