"""Candidate citation-only projection; E_sem is not the complete original E.

project_records(E) removes only citation.text_sha256 and citation.spans from
each occurrence. In particular, quotes may contain source text absent from the
remaining semantic fields. A flat/bound round trip of E_sem proves preservation
of E_sem only, not preservation of that additional source text or complete E.

restore_records(E_sem, sidecar) is the full reconstruction interface. The sidecar
binds occurrence positions, record IDs, typed semantic content, and the complete
original content. Its integrity hashes are consistency checks, not signatures;
callers remain responsible for binding the sidecar artifact to their snapshot.
No source files, QA, retrieval results or model outputs are read by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from experiments.radar_domain import coverage_representation as representation


SIDECAR_SCHEMA = "radar_citation_projection_sidecar_v1"
RECORD_FIELDS = frozenset({"record_id", "family_id", "subjects", "component", "attribute",
                           "value", "conditions_all", "citation", "locator", "annotation_note"})
CITATION_FIELDS = frozenset({"chunk_id", "text_sha256", "spans"})
REMOVED_FIELDS = ("text_sha256", "spans")
SIDECAR_FIELDS = frozenset({"schema", "removed_citation_fields", "full_reconstruction_requires_sidecar",
                           "semantic_records_sha256", "full_records_sha256", "entries"})
ENTRY_FIELDS = frozenset({"position", "record_id", "semantic_record_sha256", "removed_citation"})


def _object_fields(value, fields, location):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"Unknown or missing schema fields at {location}")


def _removed_citation_schema(removed, location):
    _object_fields(removed, set(REMOVED_FIELDS), location)
    if type(removed["text_sha256"]) is not str or type(removed["spans"]) is not list:
        raise ValueError(f"Invalid citation metadata at {location}")
    for index, span in enumerate(removed["spans"]):
        _object_fields(span, {"start", "end", "quote"}, f"{location}.spans[{index}]")
        if (type(span["start"]) is not int or type(span["end"]) is not int
                or type(span["quote"]) is not str):
            raise ValueError(f"Invalid citation span types at {location}.spans[{index}]")


def _records_schema(records, *, projected):
    if type(records) is not list:
        raise ValueError("Records must be an ordered list")
    for index, record in enumerate(records):
        location = f"records[{index}]"
        _object_fields(record, RECORD_FIELDS, location)
        if type(record["record_id"]) is not str or not record["record_id"]:
            raise ValueError(f"Missing string record_id at {location}")
        _object_fields(record["value"], {"form", "text", "unit_raw"}, f"{location}.value")
        if type(record["conditions_all"]) is not list:
            raise ValueError(f"Invalid conditions_all at {location}")
        for condition in record["conditions_all"]:
            _object_fields(condition, {"dimension", "value"}, f"{location}.conditions_all")
        citation = record["citation"]
        _object_fields(citation, {"chunk_id"} if projected else CITATION_FIELDS, f"{location}.citation")
        if type(citation["chunk_id"]) is not str or not citation["chunk_id"]:
            raise ValueError(f"Missing citation chunk_id at {location}")
        if not projected:
            _removed_citation_schema({key: citation[key] for key in REMOVED_FIELDS}, f"{location}.citation")


def _typed_sha256(value):
    """Canonical object keys, unchanged array order and JSON scalar types.

    Inputs have already passed the existing codec's strict JSON checks. JSON
    spells booleans, integers, floats, strings and null differently; Python's
    bool/int/float equality is never used to determine content identity.
    """
    content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def project_records(records):
    """Return (E_sem, audit_sidecar), preserving all occurrences and field types.

    Only the two declared citation fields leave the payload. Both returned
    objects are independent of the input. Unknown schema fields are rejected
    instead of silently being projected away or added to a future payload.
    """
    _records_schema(records, projected=False)
    representation.validate_equal_information(records)
    semantic = representation.decode_flat(representation.encode_flat(records))
    entries = []
    for position, record in enumerate(semantic):
        removed = {field: record["citation"].pop(field) for field in REMOVED_FIELDS}
        entries.append({"position": position, "record_id": record["record_id"],
                        "semantic_record_sha256": _typed_sha256(record),
                        "removed_citation": removed})
    sidecar = {"schema": SIDECAR_SCHEMA, "removed_citation_fields": list(REMOVED_FIELDS),
               "full_reconstruction_requires_sidecar": True,
               "semantic_records_sha256": _typed_sha256(semantic),
               "full_records_sha256": _typed_sha256(records), "entries": entries}
    _records_schema(semantic, projected=True)
    return semantic, sidecar


def restore_records(semantic, sidecar):
    """Reconstruct complete E only with a matching, position-bound sidecar.

    A missing sidecar, stale payload, changed type, swapped occurrence, dropped
    duplicate, or changed citation metadata raises ValueError. No field or
    metadata is inferred from the remaining payload.
    """
    _records_schema(semantic, projected=True)
    # Public codec validation rejects non-JSON Python types and non-finite values
    # before hashing; decoding also detaches the restored records from callers.
    restored = representation.decode_flat(representation.encode_flat(semantic))
    _object_fields(sidecar, SIDECAR_FIELDS, "sidecar")
    if (sidecar["schema"] != SIDECAR_SCHEMA
            or type(sidecar["removed_citation_fields"]) is not list
            or sidecar["removed_citation_fields"] != list(REMOVED_FIELDS)
            or sidecar["full_reconstruction_requires_sidecar"] is not True):
        raise ValueError("Invalid projection sidecar contract")
    entries = sidecar["entries"]
    if type(entries) is not list or len(entries) != len(restored):
        raise ValueError("Sidecar occurrence count differs from payload")
    if sidecar["semantic_records_sha256"] != _typed_sha256(restored):
        raise ValueError("Ordered semantic payload hash mismatch")
    for position, (record, entry) in enumerate(zip(restored, entries)):
        _object_fields(entry, ENTRY_FIELDS, f"sidecar.entries[{position}]")
        if type(entry["position"]) is not int or entry["position"] != position:
            raise ValueError("Sidecar occurrence position mismatch")
        if type(entry["record_id"]) is not str or entry["record_id"] != record["record_id"]:
            raise ValueError("Sidecar record identity mismatch")
        if entry["semantic_record_sha256"] != _typed_sha256(record):
            raise ValueError("Sidecar typed record content hash mismatch")
        removed = entry["removed_citation"]
        _removed_citation_schema(removed, f"sidecar.entries[{position}].removed_citation")
        record["citation"].update(deepcopy(removed))
    _records_schema(restored, projected=False)
    representation.validate_equal_information(restored)
    if sidecar["full_records_sha256"] != _typed_sha256(restored):
        raise ValueError("Reconstructed complete records hash mismatch")
    return restored
