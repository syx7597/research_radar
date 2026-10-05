"""Completion gates, conservative claim matching and private blind-review keys."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.radar_domain import source_analysis as analysis
from experiments.radar_domain import source_probe as probe


def claim(**changes):
    row = {"subject": "Example", "attribute": "frequency", "value_kind": "scalar", "value": "1",
           "min_value": None, "max_value": None, "unit": "GHz", "condition": None, "event": None,
           "unknown_scope": None, "bound_inclusive": None,
           "citations": [{"source_id": "source-a", "chunk_id": "chunk-a"}]}
    row.update(changes)
    return row


class SourceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out, self.data = self.root / "results", self.root / "data"
        self.out.mkdir()
        self.data.mkdir()
        self.protocol_path = self.out / "protocol.json"
        self.protocol = {"model": "pinned-base", "models": {label: {"adapter": label} for label in probe.MODELS},
                         "inference": {"fixed": "selector"}, "answer_inference": {"fixed": "answer"},
                         "inputs_sha256": {}, "reference_sha256": {},
                         "base_weights": {"files": [{"file": "base.safetensors", "bytes": 10, "sha256": "pinned-digest"}]}}
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.completion = {"status": "completed", "jobs_returned": 11, "eligible_for_analysis": True,
                           "protocol_sha256": probe.sha(self.protocol_path),
                           "base_files_final": {"pinned-base/base.safetensors": {"bytes": 10, "sha256": "pinned-digest"}}}
        self.completion_path = self.out / "pipeline_completed.json"
        self.completion_path.write_text(json.dumps(self.completion))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(probe, "OUT", self.out))
        self.stack.enter_context(patch.object(probe, "DATA", self.data))
        self.stack.enter_context(patch.object(probe, "PROTOCOL", self.protocol_path))
        self.stack.enter_context(patch.object(probe, "verify", lambda *a, **k: self.protocol))
        self.stack.enter_context(patch.object(probe, "questions", lambda: [{"id": "qid-1", "question": "Frequency?"}]))
        self.stack.enter_context(patch.object(analysis, "DevelopmentKB", lambda path: object()))

    def write_run_headers(self):
        for stage, arms in (("selection", probe.MODELS), ("answers", probe.ARMS)):
            for arm in arms:
                path = self.out / f"{arm}.{stage}.jsonl"
                path.write_text(json.dumps({"id": "qid-1"}) + "\n")
                runtime = {"status": "completed", "protocol_sha256": probe.sha(self.protocol_path),
                           "output_sha256": probe.sha(path)}
                if stage == "selection":
                    runtime.update(label=arm, model=self.protocol["models"][arm], config=self.protocol["inference"])
                else:
                    runtime.update(arm=arm, model=self.protocol["model"], adapter=None, config=self.protocol["answer_inference"])
                path.with_suffix(".runtime.json").write_text(json.dumps(runtime))

    def test_failed_or_partial_controller_blocks_reference_access(self):
        for changed in ({**self.completion, "status": "failed"}, {**self.completion, "jobs_returned": 10}):
            with self.subTest(changed=changed):
                self.completion_path.write_text(json.dumps(changed))
                with patch.object(analysis, "load_references", side_effect=AssertionError("references opened")) as refs:
                    with self.assertRaisesRegex(ValueError, "eleven GPU jobs"):
                        analysis.build()
                    refs.assert_not_called()

    def test_changed_final_base_identity_blocks_reference_access(self):
        broken = deepcopy(self.completion)
        broken["base_files_final"]["pinned-base/base.safetensors"]["sha256"] = "wrong-digest"
        self.completion_path.write_text(json.dumps(broken))
        with patch.object(analysis, "load_references", side_effect=AssertionError("references opened")) as refs:
            with self.assertRaisesRegex(ValueError, "Final base identity"):
                analysis.build()
            refs.assert_not_called()

    def test_last_missing_completion_manifest_prevents_replay_and_reference_access(self):
        self.write_run_headers()
        (self.out / "C2.answers.runtime.json").unlink()
        with patch.object(analysis, "replay_row", side_effect=AssertionError("premature replay")) as replay, \
                patch.object(analysis, "load_references", side_effect=AssertionError("references opened")) as refs:
            with self.assertRaises(FileNotFoundError):
                analysis.build()
            replay.assert_not_called()
            refs.assert_not_called()

    def test_basic_unit_conversions_match_without_mixing_si_prefix_case(self):
        self.assertTrue(analysis.strict_claim_match(claim(value="1", unit="GHz"), claim(value="1000", unit="MHz")))
        self.assertTrue(analysis.strict_claim_match(claim(value="0.2", unit="us"), claim(value="200", unit="ns")))
        self.assertTrue(analysis.strict_claim_match(claim(value="1", unit="MW"), claim(value="1000000", unit="W")))
        self.assertTrue(analysis.strict_claim_match(claim(value="1", unit="mW"), claim(value="0.001", unit="W")))
        self.assertFalse(analysis.strict_claim_match(claim(unit="MW"), claim(unit="mW")))
        self.assertFalse(analysis.strict_claim_match(claim(unit="mHz"), claim(unit="MHz")))
        self.assertFalse(analysis.strict_claim_match(claim(unit="Ms"), claim(unit="ms")))

    def test_options_ranges_approximation_bounds_and_conditions_remain_distinct(self):
        options = claim(value_kind="options", value=["0.2", "0.8"], unit="us")
        scaled = claim(value_kind="options", value=["800", "200"], unit="ns")
        continuous = claim(value_kind="range", value=None, min_value="0.2", max_value="0.8", unit="us")
        self.assertTrue(analysis.strict_claim_match(options, scaled))
        self.assertFalse(analysis.strict_claim_match(options, continuous))
        self.assertFalse(analysis.strict_claim_match(claim(value_kind="approximate"), claim()))
        self.assertFalse(analysis.strict_claim_match(claim(condition="peak"), claim(condition="average")))
        lower = claim(value_kind="lower_bound", value=None, min_value="1", bound_inclusive=False)
        self.assertFalse(analysis.strict_claim_match(lower, {**lower, "bound_inclusive": True}))
        self.assertFalse(analysis.strict_claim_match(claim(), claim(citations=[{"source_id": "other", "chunk_id": "chunk-a"}])))

    def blind_fixture(self):
        questions = [{"id": "private-qid", "question": "Frequency?"}]
        reference = {"private-qid": {"semantic_claims": [claim()]}}
        chunk = {"source_id": "source-a", "chunk_id": "chunk-a", "title": "Source", "text": "Frequency: 1 GHz."}
        row = {"id": "private-qid", "evidence": {"chunks": [chunk], "mapping": {"record_count": 1}},
               "generated_text": '{"answer_text":"1 GHz"}', "parsed": {"answer_text": "1 GHz", "claims": [claim()]}}
        generated = {"A1": [{**deepcopy(row), "arm": "A1"}], "C1": [{**deepcopy(row), "arm": "C1"}]}
        return questions, reference, generated

    def test_blind_review_hides_arms_and_deduplicates_only_identical_visible_inputs(self):
        questions, refs, generated = self.blind_fixture()
        packet, key = analysis.prepare_blind(questions, refs, generated)
        self.assertEqual(len(packet["records"]), 1)
        self.assertEqual(len(key["records"]), 2)
        visible = json.dumps(packet, ensure_ascii=False)
        for forbidden in ('"arm":', '"A1"', '"C1"', "private-qid", "record_count"):
            self.assertNotIn(forbidden, visible)
        self.assertEqual({row["arm"] for row in key["records"]}, {"A1", "C1"})
        self.assertEqual(len({row["blind_id"] for row in key["records"]}), 1)
        generated["C1"][0]["generated_text"] = "An answer with different wording."
        self.assertEqual(len(analysis.prepare_blind(questions, refs, generated)[0]["records"]), 2)

    def test_blind_identity_key_is_written_only_to_private_data_directory(self):
        questions, refs, generated = self.blind_fixture()
        summary = {"selection_replay": {"rows": 5}, "answer_input_and_parser_replay": {"rows": 6}}
        written = {}
        with patch.object(analysis, "build", return_value=(summary, {"records": []}, questions, refs, generated)), \
                patch.object(probe, "write_new", lambda path, value: written.setdefault(path, value)), \
                patch("sys.argv", ["source_analysis"]):
            analysis.main()
        key_path = self.data / "blind_answer_key.json"
        self.assertIn(key_path, written)
        self.assertNotIn(self.out / "blind_answer_key.json", written)
        self.assertEqual({row["arm"] for row in written[key_path]["records"]}, {"A1", "C1"})
        public_packet = written[self.data / "blind_answer_review.json"]
        self.assertNotIn('"arm":', json.dumps(public_packet))


if __name__ == "__main__":
    unittest.main()
