"""Apply the independently verified EEC column-location correction as a new version."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

from .coverage_external_review import digest, encoded, review_html

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "data/radar_sources_v2/external_review_v1"
NEW = ROOT / "data/radar_sources_v2/external_review_v2_20261104"
AUDIT = ROOT / "artifacts/thesis_direction_review/radar_source_reaudit_20261104"
PRIVATE = ROOT / "data/radar_sources_v2/source_reaudit_20261104"
ID = "RQ2-RNG-07"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return digest(path.read_bytes())


def change(row, text):
    result = deepcopy(row)
    support = result["support"][0]
    assert support["chunk_id"] == "eec_ranger_x_band:p003"
    assert digest(text.encode()) == support["text_sha256"]
    old = support["spans"][1]
    new = dict(old, start=3015, end=3054)
    assert (old["start"], old["end"]) == (943, 982)
    assert text[old["start"]:old["end"]] == text[new["start"]:new["end"]] == old["quote"]
    support["spans"][1] = new
    restored = deepcopy(result)
    restored["support"][0]["spans"][1] = old
    assert restored == row, "Only the documented evidence location may change"
    return result


def revise_v2():
    if NEW.exists() or (AUDIT / "applied_revision.json").exists():
        raise FileExistsError("Revision already exists; never overwrite its provenance")
    audit = read(AUDIT / "summary.json")
    assert audit["reviewer"] == "AI" and audit["human_gold"] is False
    assert audit["needs_revision_ids"] == [ID]
    assert audit["counts"]["answer_changes_required"] == 0
    for name, expected in audit["input_hashes"].items():
        assert sha(ROOT / name) == expected, name
    author_path = ROOT / "data/radar_sources_v2/qa_v1/weather.author.json"
    chunk_path = ROOT / "data/radar_sources_v2/curation_v1/chunks.json"
    refs, authors = read(OLD / "references.json"), read(author_path)
    source = next(x for x in read(chunk_path)["chunks"] if x["chunk_id"] == "eec_ranger_x_band:p003")
    old_refs, old_authors = deepcopy(refs), deepcopy(authors)
    for rows in (refs["rows"], authors["questions"]):
        indices = [i for i, row in enumerate(rows) if row["question_id"] == ID]
        assert len(indices) == 1
        i = indices[0]
        rows[i] = change(rows[i], source["text"])
    assert sum(a != b for a, b in zip(refs["rows"], old_refs["rows"])) == 1
    assert sum(a != b for a, b in zip(authors["questions"], old_authors["questions"])) == 1
    protocol = {
        "schema": "radar_source_location_revision_v1", "date": "2026-11-04",
        "parent_protocol_sha256": old_refs["protocol_sha256"],
        "purpose": "Correct model-column source locator; preserve question and answer semantics",
        "reviewer_type": "AI", "human_gold": False,
        "question_id": ID, "old_span": [943, 982], "new_span": [3015, 3054],
        "source_text_sha256": source["text_sha256"],
        "question_changed": False, "answer_changed": False, "score_changed": False,
        "inputs_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [OLD / "references.json", OLD / "questions.json", author_path,
                      chunk_path, AUDIT / "summary.json", PRIVATE / "comparison.json"]
        },
    }
    protocol_bytes = encoded(protocol)
    new_protocol_sha = digest(protocol_bytes)
    refs["protocol_sha256"] = new_protocol_sha
    refs["parent_reference_sha256"] = sha(OLD / "references.json")
    authors["source_evidence_revision"] = {
        "protocol_sha256": new_protocol_sha, "question_id": ID,
        "scope": "Evidence locator only; existing questions remain exposed development material",
    }
    # Archive bytes, questions and empty response rows are copied unchanged.
    shutil.copytree(OLD, NEW)
    (NEW / "revision_protocol.json").write_bytes(protocol_bytes)
    reference_bytes = encoded(refs)
    (NEW / "references.json").write_bytes(reference_bytes)
    questions = read(NEW / "questions.json")["rows"]
    (NEW / "index.html").write_bytes(review_html(questions, digest(reference_bytes), new_protocol_sha))
    blank = read(NEW / "blank_responses.json")
    assert blank["reviewer_identity"] is None and blank["human_gold"] is False
    blank["protocol_sha256"] = new_protocol_sha
    (NEW / "blank_responses.json").write_bytes(encoded(blank))
    corrected_author = PRIVATE / "weather.author.corrected.json"
    corrected_author.write_bytes(encoded(authors))
    # Old files, archive pages and wording remain bound to the prior evaluation.
    for name, expected in protocol["inputs_sha256"].items():
        assert sha(ROOT / name) == expected
    assert sha(NEW / "questions.json") == sha(OLD / "questions.json")
    for source_path in (OLD / "sources").iterdir():
        assert sha(NEW / "sources" / source_path.name) == sha(source_path)
    report = dict(protocol, new_packet=str(NEW.relative_to(ROOT)),
        actual_human_reviews=0, historical_evaluation_changed=False,
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
            NEW / "references.json", NEW / "revision_protocol.json", NEW / "index.html",
            NEW / "questions.json", NEW / "blank_responses.json", corrected_author]},
        old_reference_sha256=sha(OLD / "references.json"))
    (AUDIT / "applied_revision.json").write_bytes(encoded(report))
    print(json.dumps({"corrected_question": ID, "kind": "evidence_location_only",
                      "answers_changed": 0, "new_packet": str(NEW.relative_to(ROOT))}, ensure_ascii=False))


def revise_v1():
    directory = ROOT / "data/radar_sources_v1/source_reaudit_20261104"
    public = ROOT / "artifacts/thesis_direction_review/radar_sources_v1_reaudit_20261104"
    target, report_path = directory / "kb.corrected.json", public / "applied_revision.json"
    if target.exists() or report_path.exists():
        raise FileExistsError("Four-source revision already exists")
    comparison = read(directory / "comparison.json")
    assert comparison["reviewer"] == "AI" and comparison["human_gold"] is False
    revisions = comparison["required_revisions"]
    assert len(revisions) == 1
    change = revisions[0]
    assert change["question_id"] == "source_zh_24" and change["fact_id"] == "j34tempS"
    assert not change["answer_label_change_required"] and not change["question_change_required"]
    original_path = ROOT / "data/radar_sources_v1/kb.json"
    original = read(original_path)
    revised = deepcopy(original)
    matches = [r for r in revised["records"] if r["fact_id"] == change["fact_id"]]
    assert len(matches) == 1
    locator = matches[0]["citation"]["locator"]
    assert locator["table_context"] == change["current"]
    locator["table_context"] = change["proposed"]
    restored = deepcopy(revised)
    next(r for r in restored["records"] if r["fact_id"] == change["fact_id"])["citation"]["locator"]["table_context"] = change["current"]
    assert restored == original
    original_hash = sha(original_path)
    target.write_bytes(encoded(revised))
    report_path.write_bytes(encoded({
        "schema": "radar_source_context_revision_v1", "date": "2026-11-04",
        "question_id": change["question_id"], "fact_id": change["fact_id"],
        "field": change["field"], "old_context": change["current"],
        "new_context": change["proposed"], "reason": change["reason_zh"],
        "reviewer": "AI", "human_gold": False, "actual_human_reviews": 0,
        "answers_changed": 0, "questions_changed": 0, "historical_scores_changed": False,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
            [original_path, directory / "comparison.json", public / "summary.json"]},
        "corrected_artifact": str(target.relative_to(ROOT)), "corrected_sha256": sha(target),
    }))
    assert sha(original_path) == original_hash
    print(json.dumps({"corrected_question": change["question_id"],
                      "kind": "table_context_only", "answers_changed": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    (revise_v1 if args.scope == "v1" else revise_v2)()


if __name__ == "__main__":
    main()
