"""Replay and scoring separation tests with synthetic traces, no model calls."""
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
from experiments.radar_domain import development_analysis as analysis
from experiments.radar_domain import development_probe as probe
from experiments.radar_domain.development_environment import (
    DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages, render_records,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RadarDevelopmentAnalysisTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        data, out = base / "data", base / "output"
        data.mkdir()
        out.mkdir()
        (data / "readings.json").write_bytes((probe.DATA / "readings.json").read_bytes())
        questions = analysis.read_jsonl(probe.DATA / "questions.jsonl")[1:3]
        (data / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
        refs = json.loads((probe.DATA / "references.json").read_text())
        refs["records"] = refs["records"][1:3]
        dump(data / "references.json", refs)
        protocol = {
            "question_count": 2, "scope": "Synthetic temporary fixture; never model results",
            "inference": {"max_calls": 24, "max_generated": 2048},
            "inputs_sha256": {str(data / f): probe.sha(data / f) for f in ("readings.json", "questions.jsonl")},
            "reference_sha256": {str(data / "references.json"): probe.sha(data / "references.json")},
            "models": {label: {"program": label == "P", "files_sha256": {}} for label in probe.MODELS},
        }
        dump(out / "protocol.json", protocol)
        kb = DevelopmentKB(data / "readings.json")
        all_rows = {}
        for label in probe.MODELS:
            rows = []
            for question, event in zip(questions, ("in_service", "initial_operational_capacity")):
                messages = prompt_messages(question["question"], kb, program=label == "P")
                row = {"id": question["id"], "label": label, "messages": messages}
                if label == "P":
                    program = f"Find<arg>AN/MPQ-65<func>QueryAttrUnderCondition<arg>service_entry<arg>event_type<arg>{event}"
                    episode = execute_program(kb, program)
                    row["program"] = program
                    messages.append({"role": "assistant", "content": program})
                else:
                    episode = DevelopmentEpisode(kb)
                    calls = [
                        {"name": "step", "arguments": {"function": "Find", "inputs": ["AN/MPQ-65"], "dependencies": []}},
                        {"name": "step", "arguments": {"function": "QueryAttrUnderCondition", "inputs": ["service_entry", "event_type", event], "dependencies": [0]}},
                        {"name": "finish", "arguments": {"answer_handle": 1}},
                    ]
                    for call in calls:
                        text = "<tool_call>" + compact(call) + "</tool_call>"
                        observation = execute_response(episode, text)
                        messages += [{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}]
                row.update(prediction=episode.prediction, events=episode.events,
                           selected_handle=episode.selected, calls=episode.calls, invalid_calls=0,
                           stop_reason="program_completed" if label == "P" else "finished",
                           input_tokens=100, generated_tokens=50, total_tokens=150,
                           rendered_evidence=render_records(episode.prediction))
                rows.append(row)
            all_rows[label] = rows
            self.write_run(out, label, rows, protocol)
        return data, out, protocol, all_rows

    def write_run(self, out, label, rows, protocol):
        output = out / f"{label}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        totals = Counter(questions=len(rows))
        for row in rows:
            totals.update({key: row[key] for key in analysis.TOTAL_KEYS})
        dump(output.with_suffix(".runtime.json"), {
            "label": label, "status": "completed", "protocol_sha256": probe.sha(out / "protocol.json"),
            "output_sha256": probe.sha(output), "config": protocol["inference"],
            "model": protocol["models"][label], "totals": dict(totals), "seconds": 1.0,
        })

    def patched(self, data, out):
        stack = ExitStack()
        stack.enter_context(patch.object(probe, "DATA", data))
        stack.enter_context(patch.object(probe, "OUT", out))
        stack.enter_context(patch.object(probe, "PROTOCOL", out / "protocol.json"))
        return stack

    def test_full_replay_verifies_predictions_and_aggregates(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            with self.patched(data, out):
                artifacts = analysis.build()
            summary = artifacts["summary.json"]
            self.assertEqual(summary["cpu_replay"]["rows"], 10)
            self.assertEqual(summary["cpu_replay"]["mismatches"], 0)
            for label in probe.MODELS:
                self.assertEqual(summary["models"][label]["selection_counts"]["exact_target_selection"], 2)
                self.assertEqual(summary["models"][label]["execution_totals"]["total_tokens"], 300)
            self.assertFalse(summary["semantic_answer_scored"])

    def test_reference_not_opened_if_any_trace_replay_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, protocol, rows = self.fixture(temp)
            rows["C2"][0]["events"][0]["observation"]["count"] = 999
            self.write_run(out, "C2", rows["C2"], protocol)
            with self.patched(data, out), patch.object(analysis, "load_references") as reference_access:
                with self.assertRaisesRegex(ValueError, "events replay mismatch"):
                    analysis.build()
                reference_access.assert_not_called()

    def test_all_models_must_complete_before_reference_access(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            runtime_path = out / "A2.runtime.json"
            runtime = json.loads(runtime_path.read_text())
            runtime["status"] = "running"
            dump(runtime_path, runtime)
            with self.patched(data, out), patch.object(analysis, "load_references") as reference_access:
                with self.assertRaisesRegex(ValueError, "not explicitly completed"):
                    analysis.build()
                reference_access.assert_not_called()

    def test_runtime_hash_and_reference_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            output = out / "P.jsonl"
            output.write_bytes(output.read_bytes() + b"\n")
            with self.patched(data, out), self.assertRaisesRegex(ValueError, "output hash mismatch"):
                analysis.build()
        with tempfile.TemporaryDirectory() as temp:
            data, out, _, _ = self.fixture(temp)
            reference = data / "references.json"
            reference.write_bytes(reference.read_bytes() + b"\n")
            with self.patched(data, out), self.assertRaisesRegex(ValueError, "reference hash mismatch"):
                analysis.build()

    def test_tool_reply_and_rendered_evidence_are_replayed(self):
        for field in ("tool", "rendered_evidence"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                data, out, protocol, rows = self.fixture(temp)
                if field == "tool":
                    rows["A1"][0]["messages"][3]["content"] = "{}"
                else:
                    rows["A1"][0][field] = "unsupported answer"
                self.write_run(out, "A1", rows["A1"], protocol)
                with self.patched(data, out), self.assertRaisesRegex(ValueError, "replay mismatch"):
                    analysis.build()

    def test_generation_budget_stop_requires_exhaustion(self):
        with tempfile.TemporaryDirectory() as temp:
            data, out, protocol, rows = self.fixture(temp)
            row = rows["A1"][0]
            row.update(messages=row["messages"][:2], prediction=None, events=[], selected_handle=None,
                       calls=0, invalid_calls=0, input_tokens=0, generated_tokens=0, total_tokens=0,
                       rendered_evidence=None, stop_reason="generation_budget")
            self.write_run(out, "A1", rows["A1"], protocol)
            with self.patched(data, out), patch.object(analysis, "load_references") as reference_access:
                with self.assertRaisesRegex(ValueError, "generation-budget stop before exhaustion"):
                    analysis.build()
                reference_access.assert_not_called()

    def test_allowed_context_is_distinct_from_exact_and_unrelated_extra(self):
        ref = {"expected_fact_ids": ["target"], "acceptable_context_fact_ids": ["context"]}
        metrics = analysis.selection_metrics([{"fact_id": "target"}, {"fact_id": "context"}], ref)
        self.assertFalse(metrics["exact_target_selection"])
        self.assertTrue(metrics["target_covered"])
        self.assertTrue(metrics["target_with_allowed_context"])
        self.assertTrue(metrics["extra_selection"])
        self.assertFalse(metrics["unrelated_extra_selection"])
        overbroad = analysis.selection_metrics([{"fact_id": "target"}, {"fact_id": "other"}], ref)
        self.assertTrue(overbroad["target_covered"])
        self.assertFalse(overbroad["target_with_allowed_context"])
        self.assertTrue(overbroad["unrelated_extra_selection"])

    def test_empty_answer_is_finished_but_unfinished_is_not_empty_answer(self):
        ref = {"expected_fact_ids": ["target"], "acceptable_context_fact_ids": []}
        empty, missing = [analysis.selection_metrics(value, ref) for value in ([], None)]
        self.assertTrue(empty["finished"])
        self.assertTrue(empty["empty_selection"])
        self.assertFalse(empty["exact_target_selection"])
        self.assertTrue(missing["unfinished"])
        self.assertFalse(missing["empty_selection"])

    def test_overbroad_condition_is_observation_not_language_cause(self):
        kb = DevelopmentKB(probe.DATA / "readings.json")
        records = {item["fact_id"]: item for item in kb.records}
        episode = execute_program(kb, "Find<arg>AN/SPG-51<func>QueryAttr<arg>frequency")
        row = {"id": "temporary", "label": "P", "prediction": episode.prediction,
               "events": episode.events, "calls": episode.calls, "invalid_calls": 0,
               "stop_reason": "program_completed", "input_tokens": 1, "generated_tokens": 1, "total_tokens": 2}
        ref = {"expected_fact_ids": ["radar-review-001-05"], "acceptable_context_fact_ids": []}
        diagnosis = analysis.diagnose(row, ref, records)
        self.assertEqual(diagnosis["selection_observation_tags"], ["condition_selection_difference"])
        self.assertIsNone(diagnosis["first_error"])
        self.assertFalse(diagnosis["semantic_answer_scored"])


if __name__ == "__main__":
    unittest.main()
