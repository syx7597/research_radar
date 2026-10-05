"""Build a private, source-bound AI development packet; publish only metadata.

Semantic readings/questions are authored in the private curation file before
executor validation. A successful program checks interface consistency, not
source truth. Full-page source chunks are identical for KG and RAG evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .development_environment import DevelopmentKB, execute_program, render_records

DATA = Path("data/radar_sources_v1")
ARTIFACTS = Path("artifacts/thesis_direction_review/radar_sources_v1")
VERSION = "radar_sources_v1"
NOTES = "仅限绑定的厂商文档快照；AI解读供开发探索，未经独立人工核验，不宣称设备实际参数真值。"
REVIEW = "AI_only_development"
SEMANTIC_FIELDS = ("entity_name", "attribute", "value_kind", "value", "min_value", "max_value", "unit_std", "condition_raw", "event_type")
EXCLUDED = ("AN/MPQ-65", "Patriot", "AN/APY-9", "AN/SPG-51", "AN/SPN-35", "AN/SPG-59", "EL/M-2080")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value, lines=False):
    if lines:
        return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value).encode()
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def write_or_check(path, raw, check):
    if check:
        if not path.exists() or path.read_bytes() != raw:
            raise ValueError(f"Frozen packet reproduction mismatch: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise FileExistsError(f"Refusing to overwrite packet: {path}")
        else:
            path.write_bytes(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_program(intent):
    start = "Find<arg>" + intent["entity_name"] + "<func>"
    if intent.get("condition_raw") is not None:
        return start + "QueryAttrUnderCondition<arg>" + intent["attribute"] + "<arg>condition_raw<arg>" + intent["condition_raw"]
    return start + "QueryAttr<arg>" + intent["attribute"]


def claim_semantics(claim):
    """An independently authored answer claim mapped to the KB representation."""
    value = claim["value"]
    if claim["value_kind"] == "options":
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return {"entity_name": claim["subject"], "attribute": claim["attribute"],
            "value_kind": claim["value_kind"], "value": value,
            "min_value": claim["min_value"], "max_value": claim["max_value"],
            "unit_std": claim["unit"], "condition_raw": claim["condition"], "event_type": claim["event"]}


def build(check=False):
    spec_path = DATA / "curation.json"
    spec = json.loads(spec_path.read_text())
    require(spec["review_status"] == REVIEW and spec["human_gold"] is False, "AI-only source interpretation required")
    sources = {source["source_id"]: source for source in spec["sources"]}
    chunks, source_inputs, chunk_map = [], {str(spec_path): sha(spec_path)}, {}
    for source in sources.values():
        pdf = DATA / "pdf" / (source["source_id"] + ".pdf")
        require(sha(pdf) == source["pdf_sha256"], "Original PDF hash changed")
        source_inputs[str(pdf)] = sha(pdf)
        for page in range(1, source["page_count"] + 1):
            path = DATA / "text" / f'{source["source_id"]}.p{page:02}.txt'
            raw = path.read_bytes()
            source_inputs[str(path)] = sha(path)
            chunk = {"source_id": source["source_id"], "chunk_id": f'{source["source_id"]}:p{page:02}',
                     "title": source["title"], "entities": source["entities"], "text": raw.decode(),
                     "page_start": page, "page_end": page, "text_sha256": sha(path)}
            require(chunk["text"].strip(), "Empty extracted page")
            chunks.append(chunk)
            chunk_map[(source["source_id"], page)] = chunk
    records = []
    for item in spec["readings"]:
        source = sources[item["source_id"]]
        chunk = chunk_map[(item["source_id"], item["page"])]
        text = chunk["text"]
        evidence = item["evidence_text"]
        require(text.count(evidence) == 1, "Evidence span must occur exactly once on its source page")
        start = text.index(evidence)
        record = deepcopy(item["reading"])
        require(record["entity_name"] in source["entities"], "Entity outside source family")
        require(not any(name.casefold() in record["entity_name"].casefold() for name in EXCLUDED), "Seen interface family entered new packet")
        record.update(fact_id=item["fact_id"], source_subject=record["entity_name"],
                      scope_notes_zh=NOTES, unknown_scope=None, value_status="source_reported",
                      original_entity_attribution_usable=True, review_status=REVIEW,
                      independent_gold=False, human_verified=False, equipment_truth_verified=False,
                      citation={"source_id": source["source_id"], "chunk_id": chunk["chunk_id"],
                                "source_uri": source["url"], "source_sha256": source["pdf_sha256"],
                                "source_file": f'data/radar_sources_v1/pdf/{source["source_id"]}.pdf',
                                "locator": {"pdf_page": item["page"],
                                            "text_file": f'data/radar_sources_v1/text/{source["source_id"]}.p{item["page"]:02}.txt',
                                            "text_sha256": chunk["text_sha256"], "char_start": start,
                                            "char_end": start + len(evidence), "offset_unit": "unicode_codepoint",
                                            "table_context": item["table_context"]},
                                "evidence_text": evidence})
        records.append(record)
    require(len({r["fact_id"] for r in records}) == len(records), "Duplicate fact ID")
    by_fact = {row["fact_id"]: row for row in records}
    questions, references = [], []
    for item in spec["questions"]:
        questions.append({"id": item["id"], "question": item["question"]})
        require(len(item["expected_fact_ids"]) == len(item["semantic_claims"]) == 1, "This packet uses single-intent questions")
        record = by_fact[item["expected_fact_ids"][0]]
        claim = item["semantic_claims"][0]
        require(claim_semantics(claim) == {key: record[key] for key in SEMANTIC_FIELDS}, "Authored semantic claim disagrees with source reading")
        require(claim["citations"] == [{"source_id": record["citation"]["source_id"], "chunk_id": record["citation"]["chunk_id"]}], "Claim source differs from authored reading")
        require(item["intent"]["entity_name"] == record["entity_name"] and item["intent"]["attribute"] == record["attribute"], "Authored lookup intent disagrees")
        references.append({**deepcopy(item), "canonical_program": canonical_program(item["intent"]),
                           "source_id": record["citation"]["source_id"], "family_id": sources[record["citation"]["source_id"]]["family_id"],
                           "review_status": REVIEW, "human_gold": False})
    require(len({q["id"] for q in questions}) == len(questions), "Duplicate question ID")
    kb_doc = {"version": VERSION, "split": "development_only", "independent_gold": False,
              "human_verified": False, "review_status": REVIEW, "records": records}
    generated = {DATA / "kb.json": encoded(kb_doc), DATA / "questions.jsonl": encoded(questions, True),
                 DATA / "references.jsonl": encoded(references, True), DATA / "chunks.jsonl": encoded(chunks, True)}
    # Write the independent semantic declarations before execution; execution
    # never authors or repairs labels. --check is a deterministic CPU replay.
    for path, raw in generated.items():
        write_or_check(path, raw, check)
    kb = DevelopmentKB(DATA / "kb.json")
    for reference in references:
        episode = execute_program(kb, reference["canonical_program"])
        require(episode.done and episode.prediction is not None, "Canonical lookup did not finish")
        require([r["fact_id"] for r in episode.prediction] == reference["expected_fact_ids"], "Canonical lookup does not match authored target")
        require(episode.prediction[0] == by_fact[reference["expected_fact_ids"][0]], "Executor changed a reading")
        record = episode.prediction[0]
        rendered = render_records(episode.prediction)
        preserved = [record[key] for key in ("fact_id", "entity_name", "attribute", "value", "min_value", "max_value", "unit_std", "condition_raw") if record[key] is not None]
        preserved += [record["citation"][key] for key in ("source_uri", "source_sha256", "evidence_text")]
        require(all(value in rendered for value in preserved), "Evidence renderer changed an authored field")
        wrong = deepcopy(reference["intent"])
        wrong["entity_name"] = "__source_packet_missing_entity__"
        require(execute_program(kb, canonical_program(wrong)).prediction == [], "Empty lookup is not empty")
    index = {"version": VERSION, "review_scope": REVIEW, "sources": list(sources.values()),
             "isolation": spec["isolation"], "extraction": spec["extraction"],
             "source_only_chunking": "Every complete extracted PDF page; no question-dependent windows or answer annotations"}
    write_or_check(ARTIFACTS / "source_index.json", encoded(index), check)
    bindings = {**source_inputs, **{str(path): hashlib.sha256(raw).hexdigest() for path, raw in generated.items()},
                "experiments/radar_domain/source_packet.py": sha(__file__),
                str(ARTIFACTS / "source_index.json"): sha(ARTIFACTS / "source_index.json"),
                "experiments/radar_domain/development_environment.py": sha("experiments/radar_domain/development_environment.py")}
    manifest = {"version": VERSION, "status": "cpu_validated", "review_status": REVIEW,
                "human_gold": False, "independent_final_evaluation": False, "model_runs": 0, "training_runs": 0,
                "purpose": "Source-isolated AI exploratory evaluation with fixed existing checkpoints and a same-source RAG route",
                "source_count": len(sources), "family_count": len(sources), "entity_count": len(kb.entities),
                "question_count": len(questions), "languages": ["zh"], "record_count": len(records), "chunk_count": len(chunks),
                "question_counts_by_source": dict(Counter(ref["source_id"] for ref in references)),
                "record_counts_by_value_kind": dict(Counter(r["value_kind"] for r in records)),
                "qualified_question_count": sum(ref["intent"].get("condition_raw") is not None for ref in references),
                "validation": {"canonical_programs": len(references), "exact_targets": len(references),
                               "authored_semantic_claims_match_readings": len(references), "empty_lookup_checks": len(references),
                               "source_span_checks": len(records), "independent_ai_review": "separate independent_ai_audit.json"},
                "isolation": spec["isolation"], "files_sha256": bindings,
                "limits": ["CPU success is not model performance or factual truth validation",
                           "AI-authored and AI-reviewed; no independent human gold",
                           "The four source/family groups are the correlation units, not 24 independent sources",
                           "Some models/families existed in historical unused corpus; legacy QA lineage is not certified",
                           "Raw PDF text has extraction artifacts; original PDF, page layout and source versions remain authoritative",
                           "KB is a curated subset of source facts; RAG and final answers receive the identical complete raw page chunks",
                           "No source-silent unknowns, invented events or manufactured conditions were added",
                           "Full source text, structured real readings and QA remain in ignored private data; public metadata only"]}
    write_or_check(ARTIFACTS / "manifest.json", encoded(manifest), check)
    return {"status": manifest["status"], "questions": len(questions), "records": len(records), "sources": len(sources), "chunks": len(chunks), "check": check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.check), ensure_ascii=False))
