"""Protect separation of reference authoring, retrieval and evidence mapping."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from experiments.radar_domain import coverage_evidence as ce
from experiments.radar_domain import coverage_questions as cq
from experiments.radar_domain.coverage_gate import validate_labels
from experiments.radar_domain.question_snapshot import review_outcomes, validate_history_binding, validate_response_binding
from experiments.radar_domain.coverage_design import question_sha


class CoverageBoundaries(unittest.TestCase):
    def question_doc(self):
        text = "Type FMCW"
        c = {"family_id": "family", "text": text,
             "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        evidence = {"chunk_id": "s:p001", "text_sha256": c["text_sha256"],
                    "spans": [{"start": 0, "end": len(text), "quote": text}]}
        q = {"question_id": "q1", "question_zh": "工作体制？", "family_id": "family",
             "primary_type": "simple_attribute", "review_status": "pending", "split": "candidate_not_admitted",
             "model_outputs_seen": False, "source_version": "fixture version", "equivalence_id": "i1",
             "intent": {"subjects": ["fixture"]}, "support": [evidence],
             "reference": {"answer_zh": "FMCW", "required_facts_zh": ["FMCW"], "source_conditions": []}}
        q["question_sha256"] = hashlib.sha256(q["question_zh"].encode()).hexdigest()
        d = {"schema": "radar_source_questions_v1", "human_gold": False, "evaluation_frozen": False,
             "model_run_ready": False, "reading_selection_sha256": "fixture",
             "author_declarations": {k: k == "fresh_isolated_context" for k in cq.AUTHOR_FLAGS},
             "questions": [q]}
        return d, {"s:p001": c}

    def test_family_substitution_and_changed_reference_span_rejected(self):
        d, chunks = self.question_doc()
        with patch.object(cq, "sha", return_value="fixture"):
            cq.validate(d, chunks)
            for kind in ("family", "span"):
                bad = copy.deepcopy(d)
                if kind == "family":
                    bad["questions"][0]["family_id"] = "different"
                else:
                    bad["questions"][0]["support"][0]["spans"][0]["quote"] = "Type pulse"
                with self.assertRaises(ValueError):
                    cq.validate(bad, chunks)

    def test_author_seen_old_answers_cannot_declare_independent(self):
        d, chunks = self.question_doc(); d["author_declarations"]["old_QA_answers_read"] = True
        with patch.object(cq, "sha", return_value="fixture"), self.assertRaises(ValueError):
            cq.validate(d, chunks)

    def test_question_rewording_invalidates_bound_identity(self):
        d, chunks = self.question_doc(); d["questions"][0]["question_zh"] = "其他问题？"
        with patch.object(cq, "sha", return_value="fixture"), self.assertRaises(ValueError):
            cq.validate(d, chunks)

    def test_rejected_review_cannot_be_overridden_by_empty_open_issue_list(self):
        row = {"question_id": "q", "status": "reject", "source_semantics": "supported"}
        with self.assertRaises(ValueError):
            review_outcomes([row], {"q"})
        row["status"] = "scoring_clarification_requested"
        with self.assertRaises(ValueError):
            review_outcomes([row], set())
        self.assertEqual(review_outcomes([row], {"q"}), {"q"})

    def test_reused_question_id_cannot_reuse_stale_historical_review(self):
        row = {"question_id": "q", "question_sha256": "old", "family_id": "f",
               "same_target_or_derivative_identified": False,
               "unresolved_question_derivative_identified": False}
        history = {"inputs_sha256": {"old.json": "old_packet"}, "rows": [row]}
        packets = [{"path": "new.json", "sha256": "new_packet"}]
        questions = {"q": {"question_sha256": "new", "question_zh": "New wording", "family_id": "f"}}
        with self.assertRaises(ValueError):
            validate_history_binding(history, packets, questions)
        history["inputs_sha256"]["new.json"] = "new_packet"
        with self.assertRaises(ValueError):
            validate_history_binding(history, packets, questions)
        row["question_sha256"] = question_sha("New wording")
        validate_history_binding(history, packets, questions)

    def test_closure_and_adjudication_must_bind_same_author_response(self):
        selected = {"response.json": "current"}
        decision = {"author_response": "response.json", "inputs_sha256": {"response.json": "current"}}
        validate_response_binding(decision, selected, selected)
        for binding in ({}, {"response.json": "old"}):
            with self.assertRaises(ValueError):
                validate_response_binding(decision, binding, selected)
            with self.assertRaises(ValueError):
                validate_response_binding({**decision, "inputs_sha256": binding}, selected, selected)

    def test_retrieval_keeps_distractors_and_rejects_gold_filter(self):
        corpus = {"chunks": [], "page_to_all_records": {"p": ["target", "different-condition", "shared"]}}
        with patch.object(ce, "load_corpus", return_value=corpus), patch.object(ce, "sha", return_value="fixture"), patch.object(ce.source_rag, "BM25Index") as index:
            index.return_value.retrieve.return_value = [{"chunk_id": "p", "bm25_score": 1.0}]
            q = {"question_id": "q", "question": "fixture"}
            r = ce.retrieve_questions([q])
            self.assertEqual(r["rows"][0]["all_reading_ids"], corpus["page_to_all_records"]["p"])
            with self.assertRaises(ValueError):
                ce.retrieve_questions([{**q, "expected_fact_ids": ["target"]}])
            with self.assertRaises(ValueError):
                ce.retrieve_questions([q, q])

    def test_competition_must_have_distinct_witnesses_in_actual_evidence(self):
        q = {"question_sha256": "fixture", "family_id": "f", "support": [{"chunk_id": "p"}]}
        row = {"question_id": "q", "question_sha256": "fixture", "family_id": "f",
               "competing": True, "complex_subset": True, "reason_zh": "Different modes.",
               "witness_record_ids": ["r1", "r2"], "all_reference_chunks_retrieved": True}
        obs = {"q": {"retrieved_chunks": ["p"], "all_reading_ids": ["r1", "r2"]}}
        records = {r: {"family_id": "f"} for r in ("r1", "r2")}
        validate_labels([row], obs, records, {"q": q})
        for witnesses in (["r1", "r1"], ["r1", "elsewhere"], ["r1"]):
            with self.assertRaises(ValueError):
                validate_labels([{**row, "witness_record_ids": witnesses}], obs, records, {"q": q})
        with self.assertRaises(ValueError):
            validate_labels([{**row, "competing": False, "witness_record_ids": []}], obs, records, {"q": q})
        with self.assertRaises(ValueError):
            validate_labels([row, row], obs, records, {"q": q})


if __name__ == "__main__":
    unittest.main()
