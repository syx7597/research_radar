"""Audit existing graph and typed views against their common source readings.

This checks source identities, relations, citation spans and source-field query
selection. It does not equate the two interfaces' numeric/compositional coverage.
No question, model output or answer file is opened.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .kopl_record_bridge import reading_id, subject_id, chain, run_program
from .table_program import TableExecutor
from experiments.condition_consistency.executor import KoPLExecutor

GRAPH = Path("data/radar_domain_recovery_v1/kopl_bridge_20261106/kb.json")
SOURCE = GRAPH.parent / "source_records.json"
CHUNKS = GRAPH.parent / "source_chunks.json"
TYPED = Path("data/radar_domain_recovery_v2/typed_sources_20261107/typed_records.json")
PUBLIC = Path("results/radar_domain/dual_view_audit_20261113")
PRIVATE = Path("data/radar_domain_recovery_v2/dual_view_audit_20261113")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")


def check_views(graph, sidecar, typed, chunks):
    """Pure integrity checks, independent of the query implementations."""
    records = typed["records"]
    by_id = {r["record_id"]: r for r in records}
    if len(by_id) != len(records) or {reading_id(r) for r in by_id} != set(sidecar):
        raise ValueError("Reading identities differ across views")
    reading_nodes = {k for k, v in graph["entities"].items() if v["instanceOf"] == ["C-reading"]}
    subject_nodes = {k for k, v in graph["entities"].items() if v["instanceOf"] == ["C-subject"]}
    if reading_nodes != set(sidecar) or reading_nodes | subject_nodes != set(graph["entities"]):
        raise ValueError("Graph contains missing or unrecognized nodes")
    expected_edges, actual_edges = set(), set()
    span_count, multi_subject = 0, 0
    scopes = {}
    for rid, row in by_id.items():
        node = graph["entities"][reading_id(rid)]
        attrs = {a["key"]: a["value"]["value"] for a in node["attributes"]
                 if a["value"]["type"] == "string"}
        raw = row["raw_record"]
        if (json.loads(attrs["source_record_json"]) != raw or sidecar[reading_id(rid)]["record"] != raw
                or row["source_fact_id"] != rid or raw["record_id"] != rid
                or hashlib.sha256(stable(raw).encode()).hexdigest() != row["raw_record_sha256"]):
            raise ValueError("Source record content or hash differs")
        for key in ("family_id", "subjects", "component", "attribute", "value", "conditions_all", "citation", "locator"):
            if row[key] != raw[key]:
                raise ValueError("Typed envelope changed original source field: " + key)
        for key in ("family_id", "component", "attribute"):
            if attrs[key] != raw[key]:
                raise ValueError("Graph query projection differs: " + key)
        if (json.loads(attrs["conditions_json"]) != raw["conditions_all"]
                or json.loads(attrs["subjects_json"]) != raw["subjects"]
                or json.loads(attrs["citation_json"]) != raw["citation"]):
            raise ValueError("Graph source metadata differs")
        citation = raw["citation"]; chunk = chunks[citation["chunk_id"]]
        if hashlib.sha256(chunk["text"].encode()).hexdigest() != citation["text_sha256"]:
            raise ValueError("Source text differs from citation identity")
        for span in citation["spans"]:
            if chunk["text"][span["start"]:span["end"]] != span["quote"]:
                raise ValueError("Source citation span differs")
            span_count += 1
        multi_subject += len(raw["subjects"]) > 1
        conditions = {c["dimension"]: c["value"] for c in raw["conditions_all"]}
        if len(conditions) != len(raw["conditions_all"]):
            raise ValueError("Duplicate condition dimensions need separate query treatment")
        for subject in raw["subjects"]:
            sid = subject_id(subject)
            if sid not in subject_nodes or graph["entities"][sid]["name"] != subject:
                raise ValueError("Subject identity changed")
            expected_edges.add((sid, reading_id(rid)))
            config = {"family_id": raw["family_id"], "subject": subject,
                      "component": raw["component"], "attribute": raw["attribute"], "conditions": conditions}
            scopes[stable(config)] = config
    reverse_edges = set()
    for sid in subject_nodes:
        for rel in graph["entities"][sid]["relations"]:
            if rel["predicate"] != "has specification" or rel["direction"] != "forward":
                raise ValueError("Unexpected subject relation")
            actual_edges.add((sid, rel["object"]))
    for rid in reading_nodes:
        for rel in graph["entities"][rid]["relations"]:
            if rel["predicate"] != "has specification" or rel["direction"] != "backward":
                raise ValueError("Unexpected reading relation")
            reverse_edges.add((rel["object"], rid))
    if actual_edges != expected_edges or reverse_edges != expected_edges:
        raise ValueError("Graph subject-reading links differ from source subjects")
    if {s for s, _ in expected_edges} != subject_nodes:
        raise ValueError("Unused graph subject")
    return {"readings": len(records), "subject_nodes": len(subject_nodes), "reading_nodes": len(reading_nodes),
            "subject_reading_edges": len(expected_edges), "shared_subject_readings": multi_subject,
            "citation_spans_checked": span_count, "source_families": len({r["family_id"] for r in records})}, list(scopes.values())


def audit_queries(scopes, raw_records, graph_executor, table_executor):
    checks = []
    for config in sorted(scopes, key=stable):
        actions = [("Find", [config["subject"]]), ("Relate", ["has specification", "forward"])]
        actions += [("FilterStr", [key, config[key]]) for key in ("family_id", "component", "attribute")]
        actions += [("FilterStr", ["condition:"+k, v]) for k, v in sorted(config["conditions"].items())]
        actions.append(("Count", []))
        _, selected = run_program(graph_executor, chain(actions))
        graph_ids = {x.removeprefix("reading:") for x in selected}
        result = table_executor.operate("Scan", config, [])
        table_ids = {rid for row in result["rows"] for rid in row["evidence"]["typed_value"]}
        expected = {r["record_id"] for r in raw_records if config["subject"] in r["subjects"]
            and all(r[k] == config[k] for k in ("family_id", "component", "attribute"))
            and all(any(c["dimension"] == key and c["value"] == value for c in r["conditions_all"])
                    for key, value in config["conditions"].items())}
        if not graph_ids == table_ids == expected:
            raise ValueError("Source-field query selection differs between graph and table")
        checks.append({"scope": config, "selected_record_ids": sorted(expected), "graph_and_table_match_source": True})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    source_manifest = read("results/radar_domain/kopl_bridge_20261106/summary.json")
    for path in (GRAPH, SOURCE, CHUNKS):
        expected = source_manifest["outputs_sha256"][str(path)]
        if sha(path) != expected:
            raise ValueError("Existing graph artifact changed: " + str(path))
    design_path = PUBLIC / "design.json"
    inputs = [GRAPH, SOURCE, CHUNKS, TYPED, Path("experiments/radar_domain/kopl_record_bridge.py"),
              Path("experiments/radar_domain/table_program.py"), Path(__file__)]
    bindings = {str(path): sha(path) for path in inputs}
    if not args.check:
        if PUBLIC.exists() or PRIVATE.exists():
            raise FileExistsError("Preserve the recorded audit; use --check")
        write_new(design_path, {"schema": "radar_dual_view_design_v1", "date": "2026-11-13",
            "scope": "Existing complete 502-reading development source snapshot; source identity and source-field selection only",
            "question_answers_or_model_outputs_used": False, "expected_readings": 502,
            "query_selection": "All unique source-derived subject/family/component/attribute/condition conjunctions",
            "inputs_sha256": bindings})
    design = read(design_path)
    if design["inputs_sha256"] != bindings:
        raise ValueError("Design inputs differ")
    graph, sidecar, typed, chunks = [read(x) for x in (GRAPH, SOURCE, TYPED, CHUNKS)]
    counts, scopes = check_views(graph, sidecar, typed, chunks)
    if counts["readings"] != 502:
        raise ValueError("Full recorded source scope required")
    checks = audit_queries(scopes, [r["raw_record"] for r in typed["records"]], KoPLExecutor(GRAPH), TableExecutor(typed))
    details = PRIVATE / "query_checks.json"
    detail_text = json.dumps(checks, ensure_ascii=False, indent=2)+"\n"
    summary = {"schema": "radar_dual_view_audit_v1", "date": "2026-11-13", "status": "passed", **counts,
        "source_field_queries_checked": len(checks), "source_field_query_matches": len(checks),
        "family_query_counts": dict(Counter(x["scope"]["family_id"] for x in checks)),
        "source_record_roundtrip_passed": counts["readings"], "source_data_changed": False,
        "model_runs": 0, "server_accessed": False, "human_gold": False,
        "scope": "Source/provenance identity and conjunctive source-field selection. Numeric and compositional operator equivalence is a separate question.",
        "private_checks": {"path": str(details), "sha256": hashlib.sha256(detail_text.encode()).hexdigest()},
        "design_sha256": sha(design_path), "inputs_sha256": bindings}
    outputs = {details: detail_text, PUBLIC / "summary.json": json.dumps(summary, ensure_ascii=False, indent=2)+"\n"}
    for path, text in outputs.items():
        if args.check:
            if path.read_text() != text:
                raise ValueError("Saved audit differs: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream: stream.write(text)
    print(json.dumps({"status": "checked" if args.check else "passed", **counts, "source_field_queries": len(checks)}))


if __name__ == "__main__":
    main()
