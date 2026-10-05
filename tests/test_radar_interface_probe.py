"""Per-question routing, baseline batching and complete-replay scoring gates."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback.environment import call_message, compact
from experiments.agent_feedback.inference import execute_response
from experiments.radar_domain import interface_analysis as analysis
from experiments.radar_domain import interface_data as data
from experiments.radar_domain import interface_probe as probe
from experiments.radar_domain import interface_training as training
from experiments.radar_domain.development_environment import (
    DevelopmentEpisode, DevelopmentKB, execute_program, prompt_messages, render_records,
)


class ReferenceGateReached(Exception):
    pass


class RadarInterfaceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        (self.raw / "kb").mkdir()
        self.out = self.root / "evaluation"
        self.out.mkdir()
        self.protocol_path = self.out / "protocol.json"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(training, "DATA", self.raw))
        self.stack.enter_context(patch.object(probe, "OUT", self.out))
        self.stack.enter_context(patch.object(probe, "PROTOCOL", self.protocol_path))
        self.kbs = {}
        for spec in data.split_plan()[:2]:
            document, _, _, _ = data.make_instance(spec)
            knowledge_id = spec["knowledge_id"]
            path = self.raw / "kb" / f"{knowledge_id}.json"
            path.write_bytes(data.encoded(document))
            self.kbs[knowledge_id] = DevelopmentKB(path)
        ids = list(self.kbs)
        self.questions = [
            {"id": f"{ids[0]}-zh", "question": "Retrieve a record.", "knowledge_id": ids[0],
             "dataset": "synthetic", "language": "zh"},
            {"id": f"{ids[1]}-en", "question": "Retrieve a record.", "knowledge_id": ids[1],
             "dataset": "synthetic", "language": "en"}]

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def trace(self, question, label, phase):
        kb = self.kbs[question["knowledge_id"]]
        record = kb.records[0]
        messages = prompt_messages(question["question"], kb, program=label == "P")
        if label == "P":
            program = f"Find<arg>{record['entity_name']}<func>QueryAttr<arg>{record['attribute']}"
            messages.append({"role": "assistant", "content": program})
            episode = execute_program(kb, program)
        else:
            episode = DevelopmentEpisode(kb)
            calls = [("step", {"function": "Find", "inputs": [record["entity_name"]], "dependencies": []}),
                     ("step", {"function": "QueryAttr", "inputs": [record["attribute"]], "dependencies": [0]}),
                     ("finish", {"answer_handle": 1})]
            for name, arguments in calls:
                message = call_message(name, arguments)
                messages.append(message)
                observation = execute_response(episode, message["content"])
                messages.append({"role": "tool", "content": compact(observation)})
        row = {**question, "label": label, "phase": phase, "messages": messages,
               "prediction": episode.prediction, "selected_handle": episode.selected,
               "events": episode.events, "calls": episode.calls, "invalid_calls": 0,
               "input_tokens": 20, "generated_tokens": 10, "total_tokens": 30,
               "rendered_evidence": render_records(episode.prediction),
               "stop_reason": "program_completed" if label == "P" else "finished"}
        if label == "P":
            row["program"] = program
        return row

    def completed_fixture(self):
        self.train_out = self.root / "training"
        self.train_out.mkdir()
        train_protocol_path = self.train_out / "protocol.json"
        train_protocol = {"caches": {"agent": {"supervised_tokens": 100, "sha256": "agent-cache"},
                                      "program": {"supervised_tokens": 40, "sha256": "program-cache"}},
                          "config": {"epochs": 5}, "seeds": training.SEEDS}
        train_protocol_path.write_text(json.dumps(train_protocol))
        self.stack.enter_context(patch.object(training, "OUT", self.train_out))
        self.stack.enter_context(patch.object(training, "PROTOCOL", train_protocol_path))
        for label in probe.MODELS:
            kind = "program" if label == "P" else "agent"
            spec = train_protocol["caches"][kind]
            record = {"status": "completed", "label": label,
                      "protocol_sha256": probe.sha(train_protocol_path),
                      "actual_supervised_tokens": spec["supervised_tokens"] * 5,
                      "actual_input_tokens": 10000, "microbatches": 10,
                      "sampled_order_sha256": "order-" + str(training.SEEDS[label]),
                      "sampled_examples": 10, "global_step": 2,
                      "cache_sha256": spec["sha256"], "seed": training.SEEDS[label]}
            (self.train_out / label).mkdir()
            (self.train_out / label / "run_result.json").write_text(json.dumps(record))
        original = json.loads(probe.old.PROTOCOL.read_text())
        protocol = {"inference": original["inference"], "inputs_sha256": {},
                    "reference_sha256": {str(self.raw / "holdout.references.jsonl"): "never read in replay"},
                    "total_new_rows": 20}
        self.protocol_path.write_text(json.dumps(protocol))
        def fixed_verify(label, phase, weights=True):
            return protocol, {"label": label, "phase": phase}
        self.stack.enter_context(patch.object(probe, "verify", fixed_verify))
        self.stack.enter_context(patch.object(probe, "question_rows", lambda phase: self.questions))
        for label in probe.MODELS:
            for phase in probe.PHASES:
                rows = [self.trace(question, label, phase) for question in self.questions]
                path = self.out / f"{label}_{phase}.jsonl"
                self.write_jsonl(path, rows)
                totals = Counter(questions=len(rows))
                for row in rows:
                    totals.update({key: row[key] for key in analysis.TOTAL_KEYS})
                runtime = {"label": label, "phase": phase, "status": "completed", "totals": dict(totals),
                           "protocol_sha256": probe.sha(self.protocol_path), "output_sha256": probe.sha(path),
                           "config": protocol["inference"], "model": fixed_verify(label, phase)[1]}
                path.with_suffix(".runtime.json").write_text(json.dumps(runtime))
        return protocol

    def test_distinct_knowledge_routes_do_not_share_records_or_source_evidence(self):
        first, second = self.questions
        kb1 = DevelopmentKB(probe.kb_path(first["knowledge_id"]))
        kb2 = DevelopmentKB(probe.kb_path(second["knowledge_id"]))
        row = self.trace(first, "P", "before")
        self.assertTrue(execute_program(kb1, row["program"]).prediction)
        self.assertEqual(execute_program(kb2, row["program"]).prediction, [])
        self.assertEqual(probe.kb_path("radar_development_v2"), probe.old.KB_PATH)
        for unsafe in ("../../other", "ifkb-../../other", "ifkb-garbage"):
            with self.assertRaises(ValueError):
                probe.kb_path(unsafe)

    def test_after_preserves_synthetic_batches_and_each_original_language_batch(self):
        synthetic = [{"id": f"s-{i}", "dataset": "synthetic", "language": "zh" if i % 2 == 0 else "en"}
                     for i in range(40)]
        radar = {lang: [{"id": f"r-{i}-{lang}", "dataset": "radar_seen", "language": lang}
                        for i in range(12)] for lang in ("zh", "en")}
        after = list(probe.inference_batches(synthetic + radar["zh"] + radar["en"], 8))
        before = list(probe.inference_batches(synthetic, 8))
        self.assertEqual(after[:5], before)
        self.assertEqual([len(batch) for batch in after], [8, 8, 8, 8, 8, 8, 4, 8, 4])
        for offset, lang in ((5, "zh"), (7, "en")):
            self.assertEqual(after[offset:offset + 2], [radar[lang][:8], radar[lang][8:]])

    def test_question_schema_rejects_embedded_gold(self):
        rows = [{"id": f"ifkb-{i:02x}-{lang}", "question": "Retrieve a record", "knowledge_id": f"ifkb-{i:02x}"}
                for i in range(20) for lang in ("zh", "en")]
        self.write_jsonl(self.raw / "holdout.questions.jsonl", rows)
        self.assertEqual(len(probe.question_rows("before")), 40)
        rows[0]["expected_fact_ids"] = ["must not enter generation"]
        self.write_jsonl(self.raw / "holdout.questions.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "Question-only"):
            probe.question_rows("before")

    def test_all_fixed_runs_must_complete_before_any_replay_or_reference_access(self):
        self.completed_fixture()
        last = self.out / "C2_after.runtime.json"
        last.unlink()
        with patch.object(analysis, "replay_row", wraps=analysis.replay_row) as replay, \
                patch.object(analysis, "load_references", side_effect=AssertionError("premature references")) as refs:
            with self.assertRaises(FileNotFoundError):
                analysis.build()
            self.assertEqual(replay.call_count, 0)
            refs.assert_not_called()

    def test_pair_token_or_order_mismatch_blocks_reference_access(self):
        self.completed_fixture()
        path = self.train_out / "C1" / "run_result.json"
        correct = json.loads(path.read_text())
        for key, changed, message in (
                ("actual_supervised_tokens", correct["actual_supervised_tokens"] + 1, "Training target budget mismatch"),
                ("actual_input_tokens", correct["actual_input_tokens"] + 1, "A/C mismatch"),
                ("sampled_order_sha256", "changed-sample-order", "A/C mismatch")):
            with self.subTest(key=key):
                path.write_text(json.dumps({**correct, key: changed}))
                with patch.object(analysis, "load_references", side_effect=AssertionError("premature references")) as refs:
                    with self.assertRaisesRegex(ValueError, message):
                        analysis.build()
                    refs.assert_not_called()
        path.write_text(json.dumps(correct))

    def test_all_runs_replay_before_reference_access_and_wrong_route_is_rejected(self):
        self.completed_fixture()
        real_replay = analysis.replay_row
        observed = []
        def counted(kb, row, question, protocol, label):
            observed.append((label, row["phase"], row["knowledge_id"]))
            return real_replay(kb, row, question, protocol, label)
        def gate(*args):
            self.assertEqual(len(observed), 20)
            raise ReferenceGateReached()
        with patch.object(analysis, "replay_row", counted), patch.object(analysis, "load_references", gate):
            with self.assertRaises(ReferenceGateReached):
                analysis.build()
        # Keep bytes bound but alter a route: semantic replay must fail before references.
        path = self.out / "C2_after.jsonl"
        rows = analysis.read_jsonl(path)
        rows[0]["knowledge_id"] = self.questions[1]["knowledge_id"]
        self.write_jsonl(path, rows)
        runtime_path = path.with_suffix(".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        runtime["output_sha256"] = probe.sha(path)
        runtime_path.write_text(json.dumps(runtime))
        with patch.object(analysis, "load_references", side_effect=AssertionError("premature references")) as refs:
            with self.assertRaisesRegex(ValueError, "Wrong row routing"):
                analysis.build()
            refs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
