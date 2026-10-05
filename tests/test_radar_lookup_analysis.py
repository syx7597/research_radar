"""Synthetic bilingual replay tests; no model inference or performance evidence."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agent_feedback.environment import compact
from experiments.agent_feedback.inference import execute_response
from experiments.radar_domain import lookup_analysis as analysis
from experiments.radar_domain import lookup_probe as probe
from experiments.radar_domain.development_environment import (
    DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages, render_records,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RadarLookupAnalysisTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        data, out = base / "data", base / "output"
        data.mkdir()
        out.mkdir()
        kb_path = data / "readings.json"
        kb_path.write_bytes(probe.KB_PATH.read_bytes())
        references = json.loads((probe.DATA / "references.json").read_text())
        dump(data / "references.json", references)
        questions = {}
        for language in probe.LANGUAGES:
            target = data / f"questions.{language}.jsonl"
            target.write_bytes((probe.DATA / target.name).read_bytes())
            questions[language] = analysis.read_jsonl(target)
        protocol = {
            "question_count": 12, "scope": "Synthetic temporary fixture; never model results",
            "languages": list(probe.LANGUAGES), "kb": str(kb_path),
            "questions": {language: str(data / f"questions.{language}.jsonl") for language in probe.LANGUAGES},
            "inference": {"max_calls": 24, "max_generated": 2048},
            "inputs_sha256": {str(data / name): probe.sha(data / name)
                              for name in ("readings.json", "questions.zh.jsonl", "questions.en.jsonl")},
            "reference_sha256": {str(data / "references.json"): probe.sha(data / "references.json")},
            "models": {label: {"program": label == "P", "files_sha256": {}} for label in probe.MODELS},
        }
        dump(out / "protocol.json", protocol)
        kb = DevelopmentKB(kb_path)
        all_rows = {}
        for label in probe.MODELS:
            for language in probe.LANGUAGES:
                rows = []
                for question, reference in zip(questions[language], references["records"]):
                    messages = prompt_messages(question["question"], kb, program=label == "P")
                    row = {"id": question["id"], "label": label, "language": language, "messages": messages}
                    program = reference["canonical_program"]
                    if label == "P":
                        episode = execute_program(kb, program)
                        row["program"] = program
                        messages.append({"role": "assistant", "content": program})
                    else:
                        episode = DevelopmentEpisode(kb)
                        for chunk in program.split("<func>"):
                            function, *inputs = chunk.split("<arg>")
                            call = {"name": "step", "arguments": {"function": function, "inputs": inputs,
                                    "dependencies": [] if function == "Find" else [0]}}
                            self.append_call(episode, messages, call)
                        self.append_call(episode, messages, {"name": "finish", "arguments": {"answer_handle": 1}})
                    self.update_episode(row, episode)
                    rows.append(row)
                all_rows[(label, language)] = rows
                self.write_run(out, label, language, rows, protocol)
        return data, out, protocol, all_rows

    @staticmethod
    def append_call(episode, messages, call):
        text = "<tool_call>" + compact(call) + "</tool_call>"
        observation = execute_response(episode, text)
        messages += [{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}]

    @staticmethod
    def update_episode(row, episode):
        row.update(prediction=episode.prediction, events=episode.events, selected_handle=episode.selected,
                   calls=episode.calls, invalid_calls=sum(not e["observation"].get("ok") for e in episode.events),
                   stop_reason="program_completed" if row["label"] == "P" else "finished",
                   input_tokens=100, generated_tokens=50, total_tokens=150,
                   rendered_evidence=render_records(episode.prediction) if episode.prediction is not None else None)

    @staticmethod
    def write_run(out, label, language, rows, protocol):
        output = out / f"{label}_{language}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        totals = Counter(questions=len(rows))
        for row in rows:
            totals.update({key: row[key] for key in analysis.TOTAL_KEYS})
        dump(output.with_suffix(".runtime.json"), {
            "label": label, "language": language, "status": "completed",
            "protocol_sha256": probe.sha(out / "protocol.json"), "output_sha256": probe.sha(output),
            "config": protocol["inference"], "model": protocol["models"][label],
            "totals": dict(totals), "seconds": 1.0,
        })

    @staticmethod
    def patched(data, out):
        stack = ExitStack()
        for name, value in (("DATA", data), ("OUT", out), ("PROTOCOL", out / "protocol.json"),
                            ("KB_PATH", data / "readings.json")):
            stack.enter_context(patch.object(probe, name, value))
        return stack

    def test_all_120_replays_precede_reference_access_and_ten_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            original = analysis.load_references
            with self.patched(data, out), patch.object(analysis, "replay_row", wraps=analysis.replay_row) as replay:
                def after_replay(*args):
                    self.assertEqual(replay.call_count, 120)
                    return original(*args)
                with patch.object(analysis, "load_references", side_effect=after_replay):
                    summary = analysis.build()["summary.json"]
            self.assertEqual(summary["cpu_replay"]["rows"], 120)
            self.assertEqual(len(summary["runs"]), 10)
            for run in summary["runs"].values():
                self.assertEqual(run["selection_counts"]["exact_target_selection"], 12)
                self.assertEqual(run["trace_question_counts"]["exact_target_ever_returned"], 12)
            for paired in summary["paired_language_comparison"].values():
                self.assertEqual(paired["counts"]["both_exact"], 12)
                self.assertFalse(paired["formal_significance_tested"])

    def test_tenth_run_must_complete_before_any_replay_or_reference_access(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            runtime_path = out / "C2_en.runtime.json"
            runtime = json.loads(runtime_path.read_text())
            runtime["status"] = "running"
            dump(runtime_path, runtime)
            with self.patched(data, out), patch.object(analysis, "replay_row") as replay, \
                    patch.object(analysis, "load_references") as refs:
                with self.assertRaisesRegex(ValueError, "not explicitly completed"):
                    analysis.build()
                replay.assert_not_called()
                refs.assert_not_called()

    def test_last_row_replay_failure_blocks_reference_access(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, protocol, rows = self.fixture(temp)
            rows[("C2", "en")][-1]["events"][0]["observation"]["count"] = 999
            self.write_run(out, "C2", "en", rows[("C2", "en")], protocol)
            with self.patched(data, out), patch.object(analysis, "load_references") as refs:
                with self.assertRaisesRegex(ValueError, "events replay mismatch"):
                    analysis.build()
                refs.assert_not_called()

    def test_language_metadata_and_actual_question_must_both_match(self):
        for mode in ("row", "runtime", "prompt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                data, out, protocol, rows = self.fixture(temp)
                if mode == "row":
                    rows[("A1", "en")][0]["language"] = "zh"
                if mode == "prompt":
                    rows[("A1", "en")][0]["messages"][:2] = deepcopy(rows[("A1", "zh")][0]["messages"][:2])
                self.write_run(out, "A1", "en", rows[("A1", "en")], protocol)
                if mode == "runtime":
                    runtime_path = out / "A1_en.runtime.json"
                    runtime = json.loads(runtime_path.read_text())
                    runtime["language"] = "zh"
                    dump(runtime_path, runtime)
                with self.patched(data, out), patch.object(analysis, "load_references") as refs:
                    with self.assertRaisesRegex(ValueError, "language mismatch|initial prompt drift"):
                        analysis.build()
                    refs.assert_not_called()

    def test_id_coverage_and_runtime_config_model_hash_bindings(self):
        for mode in ("id", "output_hash", "config", "model", "protocol_hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                data, out, protocol, rows = self.fixture(temp)
                if mode == "id":
                    rows[("P", "en")][0]["id"] = "wrong-question"
                    self.write_run(out, "P", "en", rows[("P", "en")], protocol)
                else:
                    runtime_path = out / "P_en.runtime.json"
                    runtime = json.loads(runtime_path.read_text())
                    key = {"output_hash": "output_sha256", "protocol_hash": "protocol_sha256"}.get(mode, mode)
                    runtime[key] = "unexpected"
                    dump(runtime_path, runtime)
                with self.patched(data, out), patch.object(analysis, "replay_row") as replay, \
                        patch.object(analysis, "load_references") as refs:
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        analysis.build()
                    replay.assert_not_called()
                    refs.assert_not_called()

    def test_reference_hash_is_enforced_after_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            reference = data / "references.json"
            reference.write_bytes(reference.read_bytes() + b"\n")
            with self.patched(data, out), self.assertRaisesRegex(ValueError, "reference hash mismatch"):
                analysis.build()

    def test_target_seen_then_unfinished_is_not_counted_as_final_success(self):
        kb = DevelopmentKB(probe.KB_PATH)
        episode = DevelopmentEpisode(kb, max_calls=3)
        messages = []
        for call in (
            {"name": "step", "arguments": {"function": "Find", "inputs": ["EL/M-2080"], "dependencies": []}},
            {"name": "step", "arguments": {"function": "QueryAttr", "inputs": ["power"], "dependencies": [0]}},
            {"name": "step", "arguments": {"function": "QueryAttr", "inputs": ["power", "extra"], "dependencies": [0]}},
        ):
            self.append_call(episode, messages, call)
        row = {"id": "synthetic", "label": "C1", "language": "zh"}
        self.update_episode(row, episode)
        row["stop_reason"] = "call_budget"
        reference = {"expected_fact_ids": ["radar-review-001-12"], "acceptable_context_fact_ids": []}
        result = analysis.trace_diagnosis(row, reference, {r["fact_id"]: r for r in kb.records})
        self.assertTrue(result["trace_metrics"]["exact_target_ever_returned"])
        self.assertTrue(result["trace_metrics"]["exact_target_returned_but_unfinished"])
        self.assertTrue(result["trace_metrics"]["has_argument_error"])
        self.assertFalse(result["metrics"]["exact_target_selection"])
        self.assertEqual(result["target_returned_event_indices"], [1])

    def test_language_pair_counts_all_four_outcomes_without_pooling(self):
        diagnoses = []
        for label in probe.MODELS:
            for index, (zh, en) in enumerate(((True, True), (False, True), (True, False), (False, False))):
                for language, value in (("zh", zh), ("en", en)):
                    diagnoses.append({"label": label, "language": language, "id": str(index),
                                      "metrics": {"exact_target_selection": value},
                                      "stop_reason": "synthetic", "selected_fact_ids": []})
        for result in analysis.paired_comparisons(diagnoses).values():
            self.assertEqual(result["question_pairs"], 4)
            self.assertEqual(set(result["counts"].values()), {1})


if __name__ == "__main__":
    unittest.main()
