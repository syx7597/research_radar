"""Fixed source-only retrieval and complete page-to-reading mapping for v2.

No reference answers, intent/type labels, target IDs or condition filters enter
retrieval. This module does not implement, run or claim a representation gain.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from experiments.radar_domain import source_rag
from experiments.radar_domain.source_readings import ROOT, load_chunks, sha, validate, write_new

LOCAL = ROOT / "data/radar_sources_v2/qa_v1"
SELECTION = ROOT / "artifacts/thesis_direction_review/radar_readings_v1/selection.json"
CORPUS = LOCAL / "retrieval_corpus.json"


def build_corpus():
    chunks, records = load_chunks(), []
    selection = json.loads(SELECTION.read_text())
    inputs = {str(SELECTION.relative_to(ROOT)): sha(SELECTION),
              "experiments/radar_domain/source_rag.py": sha(ROOT / "experiments/radar_domain/source_rag.py"),
              "experiments/radar_domain/source_readings.py": sha(ROOT / "experiments/radar_domain/source_readings.py")}
    for entry in selection["accepted_packets"]:
        path = ROOT / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Changed accepted reading packet")
        d = json.loads(path.read_text()); validate(d, chunks)
        records.extend(d["records"]); inputs[entry["path"]] = sha(path)
    source_chunks = [{"source_id": c["source_id"], "chunk_id": c["chunk_id"],
                      "text": c["text"], "title": c["source_id"], "entities": [],
                      "page_start": c["page_one_based"] or 1, "page_end": c["page_one_based"] or 1,
                      "text_sha256": c["text_sha256"]} for c in chunks.values()]
    source_rag.validate_chunks(source_chunks, raw=True)
    mapping = {cid: sorted(r["record_id"] for r in records if r["citation"]["chunk_id"] == cid) for cid in chunks}
    return {"schema": "radar_coverage_retrieval_corpus_v1", "inputs_sha256": inputs,
            "retriever": {"k1": source_rag.K1, "b": source_rag.B, "top_k": source_rag.TOP_K,
                          "generic_query_expansion": source_rag.QUERY_EXPANSION,
                          "policy": "Reuse prior fixed tokenizer and vocabulary unchanged; no new-QA tuning."},
            "title_policy": "Archived source_id only, uniformly applied; no question-derived aliases or answer text.",
            "HTML_page_number_note": "page_start/end=1 are interface placeholders; chunk_id ending html remains authoritative.",
            "record_order": "retrieval chunk order, then lexicographic record_id within each full chunk",
            "chunks": source_chunks, "page_to_all_records": mapping,
            "reading_count": len(records)}


def load_corpus():
    d = json.loads(CORPUS.read_text())
    if d != json.loads(json.dumps(build_corpus())):
        raise ValueError("Saved retrieval corpus or configuration changed")
    return d


def retrieve_questions(questions):
    """Only a separate minimal {question_id, question} list is accepted."""
    d = load_corpus(); index = source_rag.BM25Index(d["chunks"])
    rows, seen = [], set()
    for q in questions:
        if set(q) != {"question_id", "question"} or q["question_id"] in seen:
            raise ValueError("Retrieval accepts unique IDs and question text only")
        seen.add(q["question_id"])
        found = index.retrieve(q["question"])
        selected = [c["chunk_id"] for c in found]
        rows.append({"question_id": q["question_id"],
                     "question_sha256": hashlib.sha256(q["question"].encode()).hexdigest(),
                     "retrieved_chunks": selected, "scores": [c["bm25_score"] for c in found],
                     "all_reading_ids": [rid for cid in selected for rid in d["page_to_all_records"][cid]]})
    return {"schema": "radar_coverage_retrieval_observation_v1", "corpus_sha256": sha(CORPUS),
            "status": "pre_model_fixed_retrieval_observation_not_a_tuning_set",
            "rows": rows, "model_runs": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--questions", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.prepare:
        write_new(CORPUS, build_corpus())
        print(json.dumps({"prepared_chunks": 26, "readings": 502, "model_runs": 0}))
    elif args.questions:
        if args.output is None or not args.output.resolve().is_relative_to(LOCAL):
            raise ValueError("Explicit local observation output required")
        rows = [json.loads(s) for s in args.questions.read_text().splitlines() if s.strip()]
        result = retrieve_questions(rows)
        result["questions_sha256"] = sha(args.questions)
        write_new(args.output, result)
        print(json.dumps({"retrieved_questions": len(rows), "model_runs": 0}))
    else:
        print(json.dumps({"verified_chunks": len(load_corpus()["chunks"])}))


if __name__ == "__main__":
    main()
