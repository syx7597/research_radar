"""Exact, evidence-local citation-ID normalization; no repair or semantic scoring.

Only citations[i] strings or citations[i].chunk_id strings may change. Original
model text is always preserved, and source output files are never rewritten.
The CLI requires a separately frozen analysis protocol before reading outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import coverage_representation as representation
from .source_readings import ROOT, sha, write_new

RULES_VERSION = "exact_supplied_id_v1"
ARMS = ("raw", "flat", "bound")
INPUT_PATH = "data/radar_sources_v2/representation_v1/semantic_payload_v1/inputs.jsonl"
INPUT_SHA = "f315d92f8ed6b0cba237e3496b60b44fdff4749f85f2efcceb37f789d559c1d1"
CODE_PATHS = ("experiments/radar_domain/coverage_citation_normalize.py",
              "experiments/radar_domain/coverage_representation.py",
              "experiments/radar_domain/source_readings.py")
REJECTED_ID_STATUSES = {"unknown_id", "ambiguous_record_id", "ambiguous_chunk_record_id", "unsupported_citation_shape"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_complete_object(text):
    """Accept a whole JSON object, or exactly one whole lower-case json fence."""
    require(type(text) is str, "Model output must be text")
    body, wrapper = text.strip(), "json"
    if body.startswith("```"):
        match = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", body)
        require(match is not None, "Only a whole single json fence is accepted")
        body, wrapper = match.group(1), "json_fence"
    parsed = representation._load(body)
    require(type(parsed) is dict, "The complete output must be a JSON object")
    return parsed, wrapper


def citation_index(arm, supplied_evidence):
    """Derive IDs only from the exact evidence supplied to this model request."""
    require(arm in ARMS, "Unknown evidence arm")
    chunk_ids, records = set(), {}
    if arm == "raw":
        require(type(supplied_evidence) is dict and type(supplied_evidence.get("chunks")) is list,
                "Raw evidence requires its actual chunks array")
        for chunk in supplied_evidence["chunks"]:
            require(type(chunk) is dict and type(chunk.get("chunk_id")) is str and chunk["chunk_id"],
                    "Invalid supplied raw chunk ID")
            chunk_ids.add(chunk["chunk_id"])
        return chunk_ids, records
    encoded = compact(supplied_evidence)
    readings = representation.decode_flat(encoded) if arm == "flat" else representation.decode_bound(encoded)
    for record in readings:
        citation, rid = record.get("citation"), record.get("record_id")
        require(type(rid) is str and rid and type(citation) is dict and set(citation) == {"chunk_id"}
                and type(citation["chunk_id"]) is str and citation["chunk_id"], "Invalid supplied E_sem citation identity")
        chunk_ids.add(citation["chunk_id"])
        records.setdefault(rid, set()).add(citation["chunk_id"])
    return chunk_ids, records


def resolve_id(identifier, chunk_ids, record_to_chunks):
    """No prefix, spelling, whitespace, joining or source-reference heuristics."""
    targets = record_to_chunks.get(identifier)
    if identifier in chunk_ids:
        if targets is not None and targets != {identifier}:
            return identifier, "ambiguous_chunk_record_id"
        return identifier, "already_chunk_id"
    if targets is None:
        return identifier, "unknown_id"
    if len(targets) != 1:
        return identifier, "ambiguous_record_id"
    return next(iter(targets)), "record_id_replaced"


def validate_only_id_changes(before, after, changes):
    """Typed roundtrip proves every change is one logged citation-ID replacement."""
    restored = deepcopy(after)
    positions = set()
    for change in changes:
        index, field = change["index"], change["field"]
        require(type(index) is int and index >= 0 and index not in positions, "Duplicate or invalid replacement position")
        positions.add(index)
        item = restored["citations"][index]
        require(type(change["before"]) is str and type(change["after"]) is str
                and change["before"] != change["after"], "Invalid replacement values")
        if field is None:
            require(type(item) is str and item == change["after"], "Changed citation string differs from log")
            restored["citations"][index] = change["before"]
        else:
            require(field == "chunk_id" and type(item) is dict and item.get(field) == change["after"],
                    "Only explicit chunk_id strings may change")
            item[field] = change["before"]
    require(representation._difference(before, restored) is None, "A non-citation field, type, order or duplicate changed")


def normalize_citations(text, arm, supplied_evidence):
    """Return new structures; unresolved/ambiguous references stay verbatim.

    Refusal is per citation. Independently resolvable references may be replaced
    in the same output, while invalid references remain in their original slots.
    No assertion of answer quality or citation support is made.
    """
    require(type(text) is str, "Model output must be text")
    chunks, records = citation_index(arm, supplied_evidence)
    result = {"original_text": text, "parse_status": "rejected", "wrapper": None,
              "parsed_before": None, "parsed_after": None, "citation_list_status": "unparsed",
              "changes": [], "citation_diagnostics": [], "changed_reference_count": 0,
              "rejected_reference_count": 0, "normalization_status": "unchanged"}
    try:
        before, wrapper = parse_complete_object(text)
    except (ValueError, TypeError, RecursionError):
        result["parse_error_code"] = "not_one_complete_json_object_or_single_json_fence"
        return result
    after = deepcopy(before)
    result.update(parse_status="parsed", wrapper=wrapper, parsed_before=before, parsed_after=after)
    if "citations" not in before:
        result["citation_list_status"] = "missing"
        return result
    if type(before["citations"]) is not list:
        result["citation_list_status"] = "invalid_type"
        return result
    result["citation_list_status"] = "list"
    for index, item in enumerate(before["citations"]):
        if type(item) is str:
            identifier, field = item, None
        elif type(item) is dict and type(item.get("chunk_id")) is str:
            identifier, field = item["chunk_id"], "chunk_id"
        else:
            result["citation_diagnostics"].append({"index": index, "status": "unsupported_citation_shape"})
            continue
        replacement, status = resolve_id(identifier, chunks, records)
        result["citation_diagnostics"].append({"index": index, "field": field, "before": identifier,
                                               "after": replacement, "status": status})
        if status == "record_id_replaced":
            if field is None:
                after["citations"][index] = replacement
            else:
                after["citations"][index][field] = replacement
            result["changes"].append({"index": index, "field": field, "before": identifier, "after": replacement})
    validate_only_id_changes(before, after, result["changes"])
    result["changed_reference_count"] = len(result["changes"])
    result["rejected_reference_count"] = sum(d["status"] in REJECTED_ID_STATUSES for d in result["citation_diagnostics"])
    result["normalization_status"] = "changed" if result["changes"] else "unchanged"
    return result


def repository_path(name):
    require(type(name) is str and not Path(name).is_absolute(), "Expected a repository-relative binding")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT.resolve()), "Binding leaves repository")
    return path


def load_bound(binding):
    require(set(binding) == {"path", "sha256"}, "Invalid artifact binding")
    path = repository_path(binding["path"])
    require(sha(path) == binding["sha256"], "Bound artifact changed")
    return representation._load(path.read_text())


def build(protocol_path, protocol_sha, private_output, public_output):
    """Run only a separately registered analysis; never open references/scores."""
    require(sha(protocol_path) == protocol_sha, "Normalization protocol SHA changed")
    protocol = representation._load(Path(protocol_path).read_text())
    require(protocol["schema"] == "radar_citation_normalization_protocol_v1" and protocol["status"] == "frozen"
            and protocol["rules_version"] == RULES_VERSION, "A frozen exact-ID normalization protocol is required")
    require(set(CODE_PATHS) <= set(protocol["code_sha256"]), "Missing normalizer/codec code bindings")
    for name, digest in protocol["code_sha256"].items():
        require(name.endswith(".py") and sha(repository_path(name)) == digest, "Bound code changed")
    execution = load_bound(protocol["execution_protocol"])
    audit = load_bound(protocol["execution_audit"])
    require(execution["status"] == "frozen" and audit["passed"] is True
            and audit["protocol_sha256"] == protocol["execution_protocol"]["sha256"], "Wrong completed execution identity")
    require(protocol["inputs"] == execution["inputs"] == {"path": INPUT_PATH, "sha256": INPUT_SHA}, "Wrong fixed supplied model evidence")
    inputs_path = repository_path(INPUT_PATH)
    require(sha(inputs_path) == INPUT_SHA, "Original model inputs changed")
    inputs = [representation._load(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    qids = execution["question_ids"]
    require(type(qids) is list and len(qids) == len(set(qids)) == 96, "Fixed 96-question inventory required")
    expected_pairs = [(qid, arm) for qid in qids for arm in ARMS]
    require([(r["question_id"], r["arm"]) for r in inputs] == expected_pairs, "Frozen 288-row ordering changed")
    require(all(set(r) == {"question_id", "arm", "messages", "input_tokens", "truncated"}
                and r["truncated"] is False for r in inputs), "Unexpected input fields/truncation")
    outputs = protocol["model_outputs_sha256"]
    expected_paths = {str(Path(execution["execution"]["output_root"]) / arm / f"{qid}.json") for qid, arm in expected_pairs}
    require(set(outputs) == expected_paths, "Model output bindings must contain exactly the frozen 288 outputs")
    attestation = load_bound(audit["private_audit"])
    require(attestation["passed"] is True and attestation["protocol_sha256"] == audit["protocol_sha256"], "Mechanical attestation changed")
    for name, digest in outputs.items():
        path = repository_path(name)
        require(sha(path) == digest == attestation["artifact_sha256"].get(str(path)), "Original output differs from mechanical audit")
    private_output, public_output = Path(private_output).resolve(), Path(public_output).resolve()
    manifest_path = private_output.with_suffix(".manifest.json")
    require(private_output.is_relative_to(ROOT.resolve() / "data") and private_output.suffix == ".json"
            and not private_output.exists() and not manifest_path.exists(), "Use a new private JSON file under data")
    require(public_output.is_relative_to(ROOT.resolve() / "results") and not public_output.exists(), "Use a new public count file under results")
    counts = {arm: Counter() for arm in ARMS}
    normalized_rows = []
    for row in inputs:
        qid, arm = row["question_id"], row["arm"]
        name = str(Path(execution["execution"]["output_root"]) / arm / f"{qid}.json")
        original = representation._load(repository_path(name).read_text())
        require((original["question_id"], original["arm"], original["status"]) == (qid, arm, "generated"), "Original output identity changed")
        supplied = representation._load(row["messages"][1]["content"])["evidence"]
        normalized = normalize_citations(original["generated_text"], arm, supplied)
        normalized_rows.append({"question_id": qid, "arm": arm, "original_output_sha256": outputs[name],
                  "input_row_sha256": hashlib.sha256(compact(row).encode()).hexdigest(), **normalized})
        c = counts[arm]
        c["outputs"] += 1
        c["parsed_outputs"] += normalized["parse_status"] == "parsed"
        c["parse_rejected_outputs"] += normalized["parse_status"] == "rejected"
        c["fenced_json_outputs"] += normalized["wrapper"] == "json_fence"
        c["changed_outputs"] += bool(normalized["changes"])
        c["changed_references"] += normalized["changed_reference_count"]
        c["rejected_references"] += normalized["rejected_reference_count"]
        c["empty_citation_lists"] += normalized["citation_list_status"] == "list" and not normalized["parsed_before"]["citations"]
        c["missing_citation_fields"] += normalized["citation_list_status"] == "missing"
        c["invalid_citation_list_types"] += normalized["citation_list_status"] == "invalid_type"
        c.update({"reference_" + key: value for key, value in Counter(d["status"] for d in normalized["citation_diagnostics"]).items()})
    # Detect changes while reading; never alter original or previously frozen files.
    require(sha(inputs_path) == INPUT_SHA and sha(protocol_path) == protocol_sha
            and all(sha(repository_path(p)) == h for p, h in outputs.items()), "Bound input changed during normalization")
    private_output.parent.mkdir(parents=True, exist_ok=True)
    write_new(private_output, {"schema": "radar_citation_normalization_results_v1", "rules_version": RULES_VERSION,
                              "protocol_sha256": protocol_sha, "rows": normalized_rows})
    private_manifest = {"schema": "radar_citation_normalization_private_v1", "rules_version": RULES_VERSION,
        "protocol_sha256": protocol_sha, "input_sha256": INPUT_SHA, "model_outputs_sha256": outputs,
        "code_sha256": protocol["code_sha256"], "normalized_json_sha256": sha(private_output),
        "rows": 288, "original_output_files_rewritten": False, "semantic_scoring_performed": False}
    write_new(manifest_path, private_manifest)
    public = {"schema": "radar_citation_normalization_counts_v1", "rules_version": RULES_VERSION,
        "completed": True, "protocol_sha256": protocol_sha, "execution_protocol_sha256": protocol["execution_protocol"]["sha256"],
        "execution_audit_sha256": protocol["execution_audit"]["sha256"], "input_sha256": INPUT_SHA,
        "arms": {arm: dict(counts[arm]) for arm in ARMS}, "semantic_scoring_performed": False,
        "reference_files_opened": False, "original_output_files_rewritten": False,
        "private_output": {"path": str(private_output.relative_to(ROOT)), "sha256": sha(private_output)},
        "private_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha(manifest_path)},
        "limits": "Exact supplied-ID substitutions only. Unknown and ambiguous citations remain unchanged. Parsing/normalization counts are not correctness or citation-support judgments."}
    public_output.parent.mkdir(parents=True, exist_ok=True)
    write_new(public_output, public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "private-output", "public-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = build(args.protocol, args.protocol_sha256, args.private_output, args.public_output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
