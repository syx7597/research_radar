"""Independent source-first question authoring and provenance checks.

This interface never reads curated record values, old QA or model predictions.
References are authored from original source text, not computed by an executor.
Checks bind text/locators and declarations; they do not certify answer semantics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.radar_domain.source_readings import ROOT, load_chunks, sha, spans, write_new

LOCAL = ROOT / "data/radar_sources_v2/qa_v1"
READINGS_SELECTION = ROOT / "artifacts/thesis_direction_review/radar_readings_v1/selection.json"
TYPES = {"simple_attribute", "quantity_form", "qualified_attribute", "table_binding", "bounded_listing"}
AUTHOR_FLAGS = {"fresh_isolated_context", "prior_reading_annotation_author", "old_QA_answers_read",
                "model_predictions_read", "references_computed_from_executor",
                "references_generated_from_curated_values"}


class QuestionPacket:
    def __init__(self, author):
        self.author, self.questions = author, []
        self.chunks = load_chunks()

    def add(self, *, question_id, family, primary, question, answer, facts, support,
            subjects, component, attribute, selectors=(), conditions=(),
            equivalence=None, competition_note="", binding_note="", version="",
            acceptable_variants=()):
        """support = [(chunk_id, [unique quote or (quote, occurrence), ...]), ...].

        facts: independent source-based answer assertions in Chinese.
        conditions: [{text_zh, explicit_in_question: bool}, ...].
        selectors: [[dimension, value], ...]; not executor code or target IDs.
        competition/binding notes are author hypotheses, not a passed gate.
        """
        evidence = []
        for cid, quotes in support:
            c = self.chunks[cid]
            evidence.append({"chunk_id": cid, "text_sha256": c["text_sha256"],
                             "spans": spans(c["text"], quotes)})
        self.questions.append({"question_id": question_id, "family_id": family,
                               "primary_type": primary, "question_zh": question,
                               "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                               "equivalence_id": equivalence or question_id,
                               "source_version": version,
                               "intent": {"subjects": list(subjects), "component": component,
                                          "attribute": attribute,
                                          "selectors": [{"dimension": k, "value": v} for k, v in selectors]},
                               "reference": {"answer_zh": answer, "required_facts_zh": list(facts),
                                             "source_conditions": list(conditions),
                                             "acceptable_variants_zh": list(acceptable_variants)},
                               "support": evidence, "author_competition_note_zh": competition_note,
                               "author_binding_note_zh": binding_note,
                               "annotation_origin": "AI_independent_source_author",
                               "review_status": "pending", "split": "candidate_not_admitted",
                               "model_outputs_seen": False})

    def save(self, name):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Output must be a JSON basename")
        doc = {"schema": "radar_source_questions_v1", "author": self.author,
               "author_declarations": {"fresh_isolated_context": True,
                                       "prior_reading_annotation_author": False,
                                       "old_QA_answers_read": False, "model_predictions_read": False,
                                       "references_computed_from_executor": False,
                                       "references_generated_from_curated_values": False},
               "reading_selection_sha256": sha(READINGS_SELECTION),
               "reference_grade": "AI_source_authored_pending_independent_reviews",
               "human_gold": False, "evaluation_frozen": False, "model_run_ready": False,
               "questions": self.questions}
        validate(doc, self.chunks)
        write_new(LOCAL / name, doc)


def validate(doc, chunks=None):
    chunks = load_chunks() if chunks is None else chunks
    if doc["schema"] != "radar_source_questions_v1" or doc["human_gold"] or doc["evaluation_frozen"] or doc["model_run_ready"]:
        raise ValueError("Invalid authoring-stage status")
    if doc["reading_selection_sha256"] != sha(READINGS_SELECTION):
        raise ValueError("Changed upstream reading snapshot")
    attestation = doc["author_declarations"]
    if (set(attestation) != AUTHOR_FLAGS or any(type(v) is not bool for v in attestation.values())
            or not attestation["fresh_isolated_context"]
            or any(v for k, v in attestation.items() if k != "fresh_isolated_context")):
        raise ValueError("Invalid independent author declaration")
    ids, texts = set(), set()
    for q in doc["questions"]:
        if q["question_id"] in ids or q["question_zh"] in texts:
            raise ValueError("Duplicate question ID or wording")
        ids.add(q["question_id"]); texts.add(q["question_zh"])
        if (q["primary_type"] not in TYPES or q["review_status"] != "pending"
                or q["split"] != "candidate_not_admitted" or q["model_outputs_seen"]):
            raise ValueError("Invalid question class or admission status")
        if hashlib.sha256(q["question_zh"].encode()).hexdigest() != q["question_sha256"]:
            raise ValueError("Question bytes changed")
        if not q["source_version"] or not q["equivalence_id"] or not q["intent"]["subjects"]:
            raise ValueError("Missing identity/version/intent")
        r = q["reference"]
        if not r["answer_zh"] or not r["required_facts_zh"] or not q["support"]:
            raise ValueError("Missing independently authored reference or source")
        if any(not c["text_zh"] or type(c["explicit_in_question"]) is not bool for c in r["source_conditions"]):
            raise ValueError("Malformed source condition")
        for evidence in q["support"]:
            c = chunks[evidence["chunk_id"]]
            if q["family_id"] != c["family_id"] or evidence["text_sha256"] != c["text_sha256"]:
                raise ValueError("Reference crosses a source-family component")
            if not evidence["spans"]:
                raise ValueError("Missing reference quote")
            for s in evidence["spans"]:
                if not 0 <= s["start"] < s["end"] <= len(c["text"]) or c["text"][s["start"]:s["end"]] != s["quote"]:
                    raise ValueError("Reference quote differs from source")


def audit(paths):
    chunks, questions, inputs = load_chunks(), [], {}
    for p in paths:
        p = Path(p).resolve()
        d = json.loads(p.read_text()); validate(d, chunks)
        questions += d["questions"]; inputs[str(p.relative_to(ROOT))] = sha(p)
    if len({q["question_id"] for q in questions}) != len(questions) or len({q["question_zh"] for q in questions}) != len(questions):
        raise ValueError("Duplicate questions across author packets")
    return {"schema": "radar_source_question_authoring_audit_v1", "inputs_sha256": inputs,
            "questions": len(questions), "by_family": dict(Counter(q["family_id"] for q in questions)),
            "by_primary_type": dict(Counter(q["primary_type"] for q in questions)),
            "distinct_declared_intents": len({q["equivalence_id"] for q in questions}),
            "evaluation_frozen": False, "human_gold": False, "model_run_ready": False,
            "limits": ["Identity, quotes and author declarations only; not semantic correctness or historical isolation.",
                       "Author competition notes are hypotheses, not a measured necessity gate."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packets", nargs="+")
    args = p.parse_args()
    print(json.dumps(audit(args.packets), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
