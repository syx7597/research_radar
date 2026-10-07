"""Audit declared source/split/exposure metadata only, using no PDF or QA text.

This checks registered identities and declarations, not fact truth, actual human
review, complete lineage, or source/QA bytes against their declared hashes.
Origins: ai/human/mixed. Reviews: ai_reviewed/human_reviewed/human_gold/unresolved.
equivalence_id may be null; unregistered paraphrases cannot be detected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

SPLITS = {"development", "evaluation_candidate"}
ORIGINS = {"ai", "human", "mixed"}
REVIEWS = {"ai_reviewed", "human_reviewed", "human_gold", "unresolved"}
SOURCE_FIELDS = {"source_id", "canonical_document_id", "family_ids", "source_uri", "source_sha256", "split"}
QUESTION_FIELDS = {"question_id", "question_sha256", "source_ids", "family_ids", "equivalence_id",
                   "primary_type", "split", "annotation_origin", "review_status", "model_outputs_seen"}
REGISTRY_FIELDS = {"canonical_document_ids": "document", "source_sha256": "source_hash",
                   "source_uris": "uri", "family_ids": "family", "question_ids": "question",
                   "question_sha256": "question_hash"}
LIMITS = (
    ("metadata_only", "Only declared metadata is audited; no source or QA bytes, factual correctness, or complete packet admission are verified."),
    ("incomplete_identity_registry", "Unregistered aliases, mirrors, translations and historical lineage may remain undetected."),
    ("pretraining_unknown", "Base-model pretraining exposure is unknown and is not checked."),
    ("semantic_rewrites_not_detected", "Only registered question IDs, hashes and equivalence IDs are compared; semantic rewrites are not detected."),
    ("review_declarations_only", "Origin, review and exposure flags are declarations; even human_gold does not certify actual independent human review."),
)


def _string(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _strings(value, *, nonempty=False):
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(_string(item) for item in value) and len(set(value)) == len(value))


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def normalize_uri(value):
    """Only scheme/host case and fragment; preserve path, query and userinfo."""
    if not _string(value) or any(char.isspace() or ord(char) < 32 for char in value):
        raise ValueError("nonempty URI required")
    parsed = urlsplit(value)
    if not parsed.scheme or (parsed.scheme.lower() in {"http", "https"} and not parsed.netloc):
        raise ValueError("absolute source URI required")
    user, marker, host = parsed.netloc.rpartition("@")
    netloc = user + marker + host.lower() if marker else parsed.netloc.lower()
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, ""))


def audit_packet(packet, exposure_registry):
    """Return a read-only metadata audit; malformed declarations block admission."""
    blockers, disclosures = [], [{"code": code, "message": message} for code, message in LIMITS]

    def issue(target, code, message, **metadata):
        target.append({"code": code, "message": message, **metadata})

    def rows(name):
        value = packet.get(name) if isinstance(packet, dict) else None
        if not isinstance(value, list) or not value:
            issue(blockers, "invalid_collection", "A nonempty metadata list is required.", collection=name)
            return []
        return value

    sources, questions = {}, {}
    for kind, required, destination in (("sources", SOURCE_FIELDS, sources),
                                        ("questions", QUESTION_FIELDS, questions)):
        id_key = "source_id" if kind == "sources" else "question_id"
        for offset, row in enumerate(rows(kind)):
            if not isinstance(row, dict) or set(row) != required:
                issue(blockers, "invalid_row_schema", "Row fields must match the metadata-only contract.", collection=kind, row_index=offset)
                continue
            identity = row[id_key]
            if not _string(identity):
                issue(blockers, "invalid_id", "A nonempty, trimmed identifier is required.", collection=kind, row_index=offset)
                continue
            if identity in destination:
                issue(blockers, "duplicate_id", "Identifiers must be unique within their collection.", collection=kind, identifier=identity)
                continue
            valid = row["split"] in SPLITS if isinstance(row["split"], str) else False
            valid = valid and _strings(row["family_ids"])
            if kind == "sources":
                valid = valid and _string(row["canonical_document_id"]) and _hash(row["source_sha256"])
                try:
                    uri = normalize_uri(row["source_uri"])
                except (ValueError, TypeError):
                    valid = False
            else:
                valid = (valid and _hash(row["question_sha256"]) and _strings(row["source_ids"], nonempty=True)
                         and (row["equivalence_id"] is None or _string(row["equivalence_id"]))
                         and _string(row["primary_type"]) and isinstance(row["annotation_origin"], str)
                         and row["annotation_origin"] in ORIGINS and isinstance(row["review_status"], str)
                         and row["review_status"] in REVIEWS and type(row["model_outputs_seen"]) is bool)
            if not valid:
                issue(blockers, "invalid_row_metadata", "Invalid split, identity, hash, list, review enum or boolean declaration.", collection=kind, identifier=identity)
                continue
            destination[identity] = dict(row)
            if kind == "sources":
                destination[identity]["source_uri"] = uri
            if not row["family_ids"]:
                target = blockers if row["split"] == "evaluation_candidate" else disclosures
                issue(target, "missing_family_ids", "Family registration is missing; candidates require explicit families.", collection=kind, identifier=identity)
            if kind == "questions":
                if row["annotation_origin"] != "human" and row["review_status"] == "human_gold":
                    issue(blockers, "ai_origin_human_gold", "AI or mixed authorship cannot be declared human_gold in this contract.", question_id=identity)
                for flag, active in (("unresolved_review", row["review_status"] == "unresolved"),
                                     ("model_outputs_seen", row["model_outputs_seen"])):
                    if active:
                        target = blockers if row["split"] == "evaluation_candidate" else disclosures
                        issue(target, flag, "This question remains development-only under the declared status.", question_id=identity)
                if row["equivalence_id"] is None:
                    issue(disclosures, "no_equivalence_id", "No question-equivalence identity was registered.", question_id=identity)

    exposed = set()
    for field, namespace in REGISTRY_FIELDS.items():
        entries = exposure_registry.get(field) if isinstance(exposure_registry, dict) else None
        if not isinstance(entries, list) or not all(_string(entry) for entry in entries):
            issue(blockers, "invalid_exposure_registry", "Every exposure field must be an array of nonempty identity strings.", field=field)
            continue
        for entry in entries:
            try:
                if namespace.endswith("hash"):
                    if not _hash(entry):
                        raise ValueError("invalid hash")
                    entry = entry.lower()
                elif namespace == "uri":
                    entry = normalize_uri(entry)
                exposed.add((namespace, entry))
            except ValueError:
                issue(blockers, "invalid_exposure_registry", "Invalid registered hash or URI.", field=field)

    # Explicit identity nodes join document mirrors, shared families and questions.
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def join(anchor, node):
        parents[find(node)] = find(anchor)

    for source in sources.values():
        anchor = ("source", source["source_id"])
        for node in [("document", source["canonical_document_id"]),
                     ("source_hash", source["source_sha256"].lower()), ("uri", source["source_uri"]),
                     *(("family", family) for family in source["family_ids"])]:
            join(anchor, node)
    for question in questions.values():
        anchor = ("question", question["question_id"])
        join(anchor, ("question_hash", question["question_sha256"].lower()))
        for family in question["family_ids"]:
            join(anchor, ("family", family))
        if question["equivalence_id"] is not None:
            join(anchor, ("equivalence", question["equivalence_id"]))
        for source_id in question["source_ids"]:
            if source_id not in sources:
                issue(blockers, "missing_source_reference", "Question references no valid source declaration.", question_id=question["question_id"], source_id=source_id)
            else:
                join(anchor, ("source", source_id))

    grouped = defaultdict(list)
    for node in parents:
        grouped[find(node)].append(node)
    components = []
    for nodes in grouped.values():
        nodes = sorted(nodes)
        source_ids = sorted(value for kind, value in nodes if kind == "source")
        question_ids = sorted(value for kind, value in nodes if kind == "question")
        splits = sorted({sources[s]["split"] for s in source_ids} | {questions[q]["split"] for q in question_ids})
        component_id = "component-" + hashlib.sha256(json.dumps(nodes, ensure_ascii=False).encode()).hexdigest()[:16]
        hits = [{"kind": kind, "identity": value} for kind, value in nodes if (kind, value) in exposed]
        component = {"component_id": component_id, "source_ids": source_ids, "question_ids": question_ids,
                     "family_ids": sorted(value for kind, value in nodes if kind == "family"),
                     "splits": splits, "exposure_matches": hits}
        components.append(component)
        if len(splits) > 1:
            issue(blockers, "cross_split_component", "Connected declared identities span development and evaluation_candidate.", component_id=component_id)
        if hits:
            target = blockers if "evaluation_candidate" in splits else disclosures
            issue(target, "candidate_exposure" if target is blockers else "development_reuse",
                  "This component overlaps the declared exposure registry.", component_id=component_id, matches=hits)

    coverage = {"source_count": len(sources), "question_count": len(questions),
                "family_count": len({f for row in [*sources.values(), *questions.values()] for f in row["family_ids"]}),
                "component_count": len(components),
                "question_counts_by_primary_type": dict(sorted(Counter(q["primary_type"] for q in questions.values()).items())),
                "evaluation_candidate_question_count": sum(q["split"] == "evaluation_candidate" for q in questions.values()),
                "evaluation_candidate_source_count": sum(s["split"] == "evaluation_candidate" for s in sources.values())}
    return {"status": "blocked" if blockers else "passed", "blockers": blockers, "disclosures": disclosures,
            "coverage": coverage, "components": sorted(components, key=lambda c: c["component_id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; audits cannot be overwritten")
    try:
        packet_bytes = args.packet.read_bytes()
        exposure_bytes = args.exposure.read_bytes()
        packet = json.loads(packet_bytes)
        exposure = json.loads(exposure_bytes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = audit_packet(packet, exposure)
    code_path = Path(__file__).resolve()
    report["inputs_sha256"] = {
        str(args.packet.resolve()): hashlib.sha256(packet_bytes).hexdigest(),
        str(args.exposure.resolve()): hashlib.sha256(exposure_bytes).hexdigest(),
        str(code_path): hashlib.sha256(code_path.read_bytes()).hexdigest(),
    }
    # Exclusive creation never overwrites a prior audit, even for blocked inputs.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "blockers": len(report["blockers"]), "output": str(args.output)}))
    return int(report["status"] == "blocked")


if __name__ == "__main__":
    raise SystemExit(main())
