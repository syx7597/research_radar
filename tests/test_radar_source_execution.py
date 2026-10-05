"""Source execution isolation, real selector replay and supervised stage gates."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import source_answers as answers
from experiments.radar_domain import source_pipeline as pipeline
from experiments.radar_domain import source_preflight as preflight
from experiments.radar_domain import source_probe as probe
from experiments.radar_domain import source_rag as rag
from experiments.radar_domain import interface_data as synthetic
from experiments.radar_domain.development_environment import DevelopmentKB, execute_program, prompt_messages, render_records
from experiments.radar_domain.development_analysis import TOTAL_KEYS


class SourceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.out = self.root / "data", self.root / "results"
        self.data.mkdir()
        self.out.mkdir()
        self.protocol_path = self.out / "protocol.json"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(probe, "DATA", self.data))
        self.stack.enter_context(patch.object(probe, "OUT", self.out))
        self.stack.enter_context(patch.object(probe, "PROTOCOL", self.protocol_path))
        self.stack.enter_context(patch.object(pipeline, "JOB_ROOT", self.root / "jobs"))
        document, _, _, _ = synthetic.make_instance(synthetic.split_plan()[0])
        for record in document["records"]:
            record["citation"].update(source_id="source-a", chunk_id="chunk-a")
            record["scope_notes_zh"] = "private interpretation sentinel"
        (self.data / "kb.json").write_text(json.dumps(document))
        self.kb = DevelopmentKB(self.data / "kb.json")
        source_text = "Manufacturer original source: frequency and power are described."
        chunk = {"source_id": "source-a", "chunk_id": "chunk-a", "title": "Manufacturer brochure",
                 "text": source_text, "entities": [], "page_start": 1, "page_end": 1,
                 "text_sha256": hashlib.sha256(source_text.encode()).hexdigest()}
        self.write_jsonl(self.data / "chunks.jsonl", [chunk])
        self.question = {"id": "question-1", "question": "What frequency is described?"}
        self.write_jsonl(self.data / "questions.jsonl", [self.question])
        old = json.loads(probe.previous.PROTOCOL.read_text())
        self.protocol = {"question_count": 1, "models": {"P": {"adapter": "fixed-checkpoint", "program": True}},
                         "inference": old["inference"], "inputs_sha256": {}, "reference_sha256": {},
                         "maximum_hours_per_selector": 0.35, "maximum_hours_per_answer_arm": 0.25,
                         "maximum_additional_gpu_hours": 3.25}
        self.protocol_path.write_text(json.dumps(self.protocol))
        record = self.kb.records[0]
        program = f"Find<arg>{record['entity_name']}<func>QueryAttr<arg>{record['attribute']}"
        episode = execute_program(self.kb, program)
        self.selection = {"id": self.question["id"], "label": "P", "program": program,
                "messages": prompt_messages(self.question["question"], self.kb, program=True) + [{"role": "assistant", "content": program}],
                "prediction": episode.prediction, "events": episode.events, "selected_handle": episode.selected,
                "calls": episode.calls, "invalid_calls": 0, "input_tokens": 20, "generated_tokens": 10, "total_tokens": 30,
                "rendered_evidence": render_records(episode.prediction), "stop_reason": "program_completed"}
        self.selection_path = self.out / "P.selection.jsonl"
        self.runtime_path = self.selection_path.with_suffix(".runtime.json")
        self.write_selection()

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def write_selection(self):
        self.write_jsonl(self.selection_path, [self.selection])
        totals = {"questions": 1, **{key: self.selection[key] for key in TOTAL_KEYS}}
        runtime = {"status": "completed", "label": "P", "protocol_sha256": probe.sha(self.protocol_path),
                   "output_sha256": probe.sha(self.selection_path), "model": self.protocol["models"]["P"],
                   "config": self.protocol["inference"], "totals": totals}
        self.runtime_path.write_text(json.dumps(runtime))

    def test_answers_use_only_original_chunks_and_no_reference_file(self):
        real_read = Path.read_text
        def guarded(path, *args, **kwargs):
            if "reference" in path.name:
                raise AssertionError("Inference accessed semantic references")
            return real_read(path, *args, **kwargs)
        with patch.object(Path, "read_text", guarded):
            selected, hashes = answers.evidence_rows("P", self.protocol)
            retrieved, _ = answers.evidence_rows("RAG", self.protocol)
        self.assertEqual(len(hashes), 2)
        first = rag.answer_messages(self.question["question"], selected[0]["evidence"])
        second = rag.answer_messages(self.question["question"], retrieved[0]["evidence"])
        self.assertEqual(first, second)
        user = json.loads(first[1]["content"])
        self.assertEqual(set(user), {"question", "source_chunks"})
        self.assertEqual(set(user["source_chunks"][0]), set(rag.EVIDENCE_KEYS))
        self.assertNotIn("private interpretation sentinel", json.dumps(first))
        self.assertNotIn("fact_id", json.dumps(first))

    def test_selector_requires_completion_fixed_model_row_label_and_totals(self):
        correct = json.loads(self.runtime_path.read_text())
        for key, changed, message in (("status", "failed", "not complete"),
                                      ("model", {"adapter": "different"}, "model/config drift"),
                                      ("totals", {**correct["totals"], "calls": 99}, "totals mismatch")):
            with self.subTest(key=key):
                self.runtime_path.write_text(json.dumps({**correct, key: changed}))
                with self.assertRaisesRegex(ValueError, message):
                    answers.evidence_rows("P", self.protocol)
        self.runtime_path.write_text(json.dumps(correct))
        self.selection["label"] = "C1"
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "row label"):
            answers.evidence_rows("P", self.protocol)
        self.runtime_path.unlink()
        with self.assertRaises(FileNotFoundError):
            answers.evidence_rows("P", self.protocol)

    def test_selector_predictions_are_reexecuted_not_trusted_after_rehashing(self):
        self.selection["prediction"] = []
        self.write_selection()  # Hash is valid: tool replay must catch the corruption.
        with self.assertRaisesRegex(ValueError, "prediction replay mismatch"):
            answers.evidence_rows("P", self.protocol)

    def test_every_fixed_gpu_command_uses_the_shared_supervisor_and_budget(self):
        jobs = pipeline.build_jobs(self.protocol)
        self.assertEqual(len(jobs), 11)
        self.assertEqual(len({job["name"] for job in jobs}), 11)
        self.assertAlmostEqual(sum(job["maximum_gpu_hours"] for job in jobs), 3.25)
        for job in jobs:
            command = job["command"]
            self.assertEqual(command[1:3], ["-m", "experiments.agent_feedback.run_job"])
            self.assertEqual(command[command.index("--gpus") + 1], str(job["gpu"]))
            self.assertEqual(float(command[command.index("--max-hours") + 1]), job["maximum_gpu_hours"])
            self.assertIn("--", command)
            self.assertEqual(job["name"], f"radar_sources_v1_{job['label']}_{job['stage']}")
        self.assertEqual([job["label"] for job in jobs if job["stage"] == "select" and job["gpu"] == 0], ["A1", "P"])

    def pipeline_context(self):
        out = self.root / "pipeline-results"
        out.mkdir()
        protocol_path = out / "protocol.json"
        protocol_path.write_text(json.dumps(self.protocol))
        self.stack.enter_context(patch.object(probe, "OUT", out))
        self.stack.enter_context(patch.object(probe, "PROTOCOL", protocol_path))
        self.stack.enter_context(patch.object(probe, "verify", lambda *a, **k: self.protocol))
        self.stack.enter_context(patch.object(pipeline.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="syx")))
        self.stack.enter_context(patch.object(pipeline, "ensure_idle", lambda *a: None))
        self.stack.enter_context(patch.object(pipeline, "total_gpu_seconds", lambda now: 0))
        return out

    def test_administrator_account_is_rejected_before_any_gpu_work(self):
        with patch.object(pipeline.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="ubuntu")), \
                patch.object(pipeline, "ensure_idle", side_effect=AssertionError("GPU touched")) as idle:
            with self.assertRaises(PermissionError):
                pipeline.main()
            idle.assert_not_called()

    def test_failed_selection_stage_never_starts_answers_or_base_final_hash(self):
        out = self.pipeline_context()
        with patch.object(pipeline, "run_stage", return_value={"status": "failed", "stage": "select", "jobs_returned": 0}) as stage, \
                patch.object(preflight, "hash_base", side_effect=AssertionError("unnecessary hash")) as base:
            with self.assertRaises(SystemExit):
                pipeline.main()
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(stage.call_args.args[0], "select")
            base.assert_not_called()
        completed = json.loads((out / "pipeline_completed.json").read_text())
        self.assertFalse(completed["eligible_for_analysis"])
        self.assertIsNone(completed["base_files_final"])
        with self.assertRaises(FileExistsError):
            pipeline.main()

    def test_successful_stages_require_final_base_hash_before_scoring(self):
        out = self.pipeline_context()
        calls = []
        def stage(name, jobs, protocol):
            calls.append(name)
            return {"status": "completed", "stage": name, "jobs_returned": 5 if name == "select" else 6}
        def base(protocol):
            self.assertEqual(calls, ["select", "answer"])
            return {"base-shard": {"sha256": "fixed"}}
        with patch.object(pipeline, "run_stage", stage), patch.object(preflight, "hash_base", base):
            pipeline.main()
        completed = json.loads((out / "pipeline_completed.json").read_text())
        self.assertTrue(completed["eligible_for_analysis"])
        self.assertEqual(completed["jobs_returned"], 11)
        self.assertEqual(completed["protocol_sha256"], probe.sha(probe.PROTOCOL))
        self.assertEqual(completed["base_files_final"]["base-shard"]["sha256"], "fixed")

    def test_final_base_hash_failure_marks_whole_pipeline_unscorable(self):
        out = self.pipeline_context()
        with patch.object(pipeline, "run_stage", side_effect=lambda stage, jobs, p: {"status": "completed", "stage": stage, "jobs_returned": 5 if stage == "select" else 6}), \
                patch.object(preflight, "hash_base", side_effect=ValueError("base drift")):
            with self.assertRaises(SystemExit):
                pipeline.main()
        completed = json.loads((out / "pipeline_completed.json").read_text())
        self.assertEqual(completed["status"], "failed")
        self.assertFalse(completed["eligible_for_analysis"])
        self.assertIn("base drift", completed["final_error"])


if __name__ == "__main__":
    unittest.main()
