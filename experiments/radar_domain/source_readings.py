"""Source-conditioned reading annotation; local text stays in ignored data/.

This module checks provenance and structure, not scientific truth or QA gold.
One reading can apply to several explicitly named subjects. Such expansion is
not additional independent evidence. No model, question or answer generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_sources_v2"
LOCAL = ROOT / "data/radar_sources_v2/curation_v1"
FORMS = {"scalar", "range", "options", "upper_bound", "lower_bound", "approximate",
         "tolerance", "text", "correspondence"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def compact(text):
    """Only collapse whitespace; do not correct punctuation, units or values."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_id(source, page=None):
    return f"{source}:p{page:03d}" if page is not None else f"{source}:html"


def build_chunks():
    scope_path = PUBLIC / "annotation_scope.json"
    scope = json.loads(scope_path.read_text())
    sources = {}
    inputs = {str(scope_path.relative_to(ROOT)): sha(scope_path)}
    for name in ("weather_admission.json", "marine_admission.json", "replacement_admission.json"):
        p = PUBLIC / name
        inputs[str(p.relative_to(ROOT))] = sha(p)
        for s in json.loads(p.read_text())["sources"]:
            sources[s["source_id"]] = s
    chunks = []
    for selection in scope["page_scope"]:
        s = sources[selection["source_id"]]
        source_path = s.get("file") or s["local_relative_path"]
        if sha(ROOT / source_path) != s["sha256"]:
            raise ValueError("Original source changed")
        for page in selection.get("pages_one_based", [None]):
            if page is None:
                suffix = "/text.txt"
            elif source_path.startswith("data/radar_sources_v2/weather/"):
                suffix = f".p{page:02d}.txt"
            else:
                suffix = f"/pages/{page:03d}.txt"
            matches = [a for a in s["derived_local_artifacts"] if a["path"].endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one selected text: {s['source_id']} {page}")
            derived = matches[0]
            if sha(ROOT / derived["path"]) != derived["sha256"]:
                raise ValueError("Derived source text changed")
            text = compact((ROOT / derived["path"]).read_text())
            chunks.append({"chunk_id": chunk_id(s["source_id"], page),
                           "family_id": selection["family_id"], "source_id": s["source_id"],
                           "page_one_based": page, "source_path": source_path,
                           "source_sha256": s["sha256"],
                           "source_uri": s.get("final_uri") or s["resolved_url"],
                           "derived_path": derived["path"], "derived_sha256": derived["sha256"],
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    return {"schema": "source_reading_chunks_v1", "inputs_sha256": inputs,
            "normalization": "collapse_whitespace_only", "chunks": chunks}


def prepare():
    write_new(LOCAL / "chunks.json", build_chunks())


def load_chunks():
    path = LOCAL / "chunks.json"
    doc = json.loads(path.read_text())
    # Reconstruct from the complete declared scope, not only caller-supplied
    # hashes. This also rejects omitted pages and substituted auxiliary sources.
    if doc != build_chunks():
        raise ValueError("Chunk manifest differs from declared source/page scope")
    chunks = {}
    for c in doc["chunks"]:
        if c["chunk_id"] in chunks:
            raise ValueError("Duplicate chunk ID")
        for stem in ("source", "derived"):
            if sha(ROOT / c[f"{stem}_path"]) != c[f"{stem}_sha256"]:
                raise ValueError("Changed archived bytes")
        expected = compact((ROOT / c["derived_path"]).read_text())
        if c["text"] != expected or hashlib.sha256(expected.encode()).hexdigest() != c["text_sha256"]:
            raise ValueError("Changed compact text")
        chunks[c["chunk_id"]] = c
    return chunks


def spans(text, quotes):
    """Strings must be unique. An explicit (quote, occurrence) may disambiguate."""
    found = []
    for item in quotes:
        quote, occurrence = (item, None) if isinstance(item, str) else item
        quote = compact(quote)
        if not quote:
            raise ValueError("Empty citation")
        positions = [m.start() for m in re.finditer(re.escape(quote), text)]
        if not positions or (occurrence is None and len(positions) != 1):
            raise ValueError(f"Citation absent or ambiguous ({len(positions)}): {quote!r}")
        i = 0 if occurrence is None else occurrence
        if not isinstance(i, int) or not 0 <= i < len(positions):
            raise ValueError("Invalid citation occurrence")
        start = positions[i]
        found.append({"start": start, "end": start + len(quote), "quote": quote})
    if not found:
        raise ValueError("Missing citation")
    return found


class Packet:
    """Authoring convenience. citations support locating, not proving, semantics.

    add(source, page, subjects, component, attribute, form, value, unit,
        conditions, quotes, locator, note="")
    conditions: ordered list of [dimension, source-faithful value] pairs; AND.
    Value text preserves options/correspondences (no Cartesian expansion).
    The options form is a printed enumeration, not an assertion that choices
    are mutually exclusive or exhaustive; source conjunctions/qualifiers stay.
    Use form=correspondence for paired values; unit can be null if missing or a
    source-faithful string for multiple explicitly printed units. Approximate
    and bound signs must remain in value. Shared rows may list multiple subjects.
    """
    def __init__(self, author, prefix):
        self.author, self.prefix = author, prefix
        self.chunks = load_chunks()
        self.records, self.ledger = [], []

    def cite(self, source, page, quotes):
        c = self.chunks[chunk_id(source, page)]
        return {"chunk_id": c["chunk_id"], "text_sha256": c["text_sha256"],
                "spans": spans(c["text"], quotes)}

    def add(self, source, page, subjects, component, attribute, form, value, unit,
            conditions, quotes, locator, note=""):
        c = self.chunks[chunk_id(source, page)]
        r = {"record_id": f"{self.prefix}-{len(self.records)+1:04d}",
             "family_id": c["family_id"], "subjects": [subjects] if isinstance(subjects, str) else subjects,
             "component": component, "attribute": attribute,
             "value": {"form": form, "text": value, "unit_raw": unit},
             "conditions_all": [{"dimension": k, "value": v} for k, v in conditions],
             "citation": self.cite(source, page, quotes), "locator": locator,
             "annotation_note": note}
        self.records.append(r)
        return r["record_id"]

    def mark(self, source, page, status, topic, reason, quotes):
        c = self.chunks[chunk_id(source, page)]
        self.ledger.append({"entry_id": f"{self.prefix}-L{len(self.ledger)+1:03d}",
                            "family_id": c["family_id"], "status": status, "topic": topic,
                            "reason": reason, "citation": self.cite(source, page, quotes)})

    def save(self, name, coverage):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Packet name must be a local JSON basename")
        doc = {"schema": "source_readings_v1", "author": self.author,
               "origin": "AI_source_annotation", "human_gold": False,
               "model_outputs_seen": False, "questions_created": 0,
               "chunks_sha256": sha(LOCAL / "chunks.json"),
               "coverage": coverage, "records": self.records, "ledger": self.ledger}
        validate(doc, self.chunks)
        write_new(LOCAL / name, doc)


def validate(doc, chunks):
    if doc["schema"] != "source_readings_v1" or doc["human_gold"] or doc["model_outputs_seen"] or doc["questions_created"]:
        raise ValueError("Invalid annotation status")
    if doc["chunks_sha256"] != sha(LOCAL / "chunks.json"):
        raise ValueError("Chunk manifest changed")
    ids = set()
    for row in doc["records"] + doc["ledger"]:
        rid = row.get("record_id") or row["entry_id"]
        if rid in ids:
            raise ValueError("Duplicate annotation ID")
        ids.add(rid)
        cite = row["citation"]
        c = chunks[cite["chunk_id"]]
        if row["family_id"] != c["family_id"] or cite["text_sha256"] != c["text_sha256"]:
            raise ValueError("Citation identity mismatch")
        if not cite["spans"]:
            raise ValueError("No citation spans")
        for s in cite["spans"]:
            if not 0 <= s["start"] < s["end"] <= len(c["text"]) or c["text"][s["start"]:s["end"]] != s["quote"]:
                raise ValueError("Citation span mismatch")
        if "record_id" in row:
            if (not row["subjects"] or len(set(row["subjects"])) != len(row["subjects"])
                    or not all(isinstance(s, str) and s.strip() for s in row["subjects"])
                    or not row["component"] or not row["attribute"] or not row["locator"]):
                raise ValueError("Incomplete reading identity")
            v = row["value"]
            if v["form"] not in FORMS or not isinstance(v["text"], str) or not v["text"].strip():
                raise ValueError("Invalid value form")
            if v["unit_raw"] is not None and (not isinstance(v["unit_raw"], str) or not v["unit_raw"].strip()):
                raise ValueError("Invalid source unit")
            if any(not d["dimension"] or not d["value"] for d in row["conditions_all"]):
                raise ValueError("Empty condition")
        elif row["status"] not in {"excluded", "unresolved", "context_only"}:
            raise ValueError("Invalid ledger status")
    declared = [x["chunk_id"] for x in doc["coverage"]]
    observed = {r["citation"]["chunk_id"] for r in doc["records"] + doc["ledger"]}
    if len(declared) != len(set(declared)) or set(declared) != observed:
        raise ValueError("Coverage must match annotated chunk set")
    if any(not x.get("scope_review") for x in doc["coverage"]):
        raise ValueError("Missing whole-page coverage declaration")


def audit(paths):
    chunks, rows, ledger, coverage, inputs = load_chunks(), [], [], [], {}
    for path in paths:
        path = Path(path)
        doc = json.loads(path.read_text())
        validate(doc, chunks)
        rows += doc["records"]
        ledger += doc["ledger"]
        coverage += doc["coverage"]
        inputs[str(path.relative_to(ROOT))] = sha(path)
    record_ids = [r["record_id"] for r in rows]
    ledger_ids = [r["entry_id"] for r in ledger]
    covered = [c["chunk_id"] for c in coverage]
    if (len(record_ids + ledger_ids) != len(set(record_ids + ledger_ids))
            or len(covered) != len(set(covered))):
        raise ValueError("Duplicate IDs or page ownership across packets")
    return {"schema": "source_readings_structural_audit_v1", "inputs_sha256": inputs,
            "chunks_sha256": sha(LOCAL / "chunks.json"),
            "counts": {"readings": len(rows), "covered_chunks": len(covered), "scope_chunks": len(chunks),
                       "families": len({r["family_id"] for r in rows}),
                       "qualified_readings": sum(bool(r["conditions_all"]) for r in rows),
                       "multiple_qualifier_readings": sum(len(r["conditions_all"]) >= 2 for r in rows),
                       "ledger_entries": len(ledger)},
            "by_family": dict(Counter(r["family_id"] for r in rows)),
            "by_form": dict(Counter(r["value"]["form"] for r in rows)),
            "by_ledger_status": dict(Counter(r["status"] for r in ledger)),
            "uncovered_chunks": sorted(set(chunks) - set(covered)),
            "status": "structural_only_semantic_review_separate", "human_gold": False,
            "evaluation_frozen": False, "model_run_ready": False,
            "new_questions": 0, "new_training_runs": 0, "new_model_runs": 0,
            "limits": ["Citation bytes do not certify semantic binding or completeness.",
                       "Reading counts are not independent facts or evaluation sample counts.",
                       "Necessity gate depends on independent questions and actual retrieval, not record counts."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("packets", nargs="*")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.packets:
        print(json.dumps(audit([Path(p).resolve() for p in args.packets]), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"verified_chunks": len(load_chunks())}))


if __name__ == "__main__":
    main()
