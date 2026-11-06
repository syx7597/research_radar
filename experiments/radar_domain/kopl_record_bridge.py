"""CPU-only bridge from the complete source-reading snapshot to original KoPL.

No QA file or required-record list is an input to graph construction. Each source
reading remains one node even if it has several subjects. Numbers are projections
with explicit value-form keys; all original fields remain available losslessly.
"""
from collections import Counter
from copy import deepcopy
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from experiments.agent_feedback.environment import Episode
from experiments.agent_feedback.recovery import sha256
from experiments.condition_consistency.executor import KoPLExecutor, validate_program

SELECTION = Path("artifacts/thesis_direction_review/radar_readings_v1/selection.json")
CHUNKS = Path("data/radar_sources_v2/curation_v1/chunks.json")
PRIVATE = Path("data/radar_domain_recovery_v1/kopl_bridge")
PUBLIC = Path("results/radar_domain/kopl_bridge_20261104")
NUMBER = re.compile(r"(?:[+-]\s*)?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
NUMERIC_KEYS = {"scalar": "scalar_value", "lower_bound": "lower_bound_value",
                "upper_bound": "upper_bound_value"}
# Exact source units, no unit conversion, lowercasing or mixed-unit guessing.
UNITS = set(("° W dB m A kW dBi % rpm degrees ft mm hPa kg km Watts dBz dBm MHz "
             "kn mW Hz bits years meters bit bins products channel Kg Knots Nm NM "
             "GHz VDC kHz ms s dBZ seconds").split()) | {
                 "ft.", "V dc", "range bins", "targets", "m/s", "°/s²"}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def numeric_projection(record):
    value = record["value"]
    form, unit = value["form"], value["unit_raw"]
    if form not in NUMERIC_KEYS:
        return None, "value_form_retained_as_text"
    if unit not in UNITS:
        return None, "ambiguous_or_unregistered_unit"
    text = value["text"].replace("−", "-")
    numbers = NUMBER.findall(text)
    if len(numbers) != 1:
        return None, "not_exactly_one_number"
    # PDF-derived values can separate a sign from its digits, e.g. "- 8 dBZ".
    # Preserve that source lexeme and normalize whitespace only for conversion.
    decimal = Decimal(re.sub(r"\s+", "", numbers[0]))
    number = float(decimal)
    if not decimal.is_finite() or abs(number) == float("inf"):
        return None, "non_finite"
    return {"key": NUMERIC_KEYS[form], "number": number, "unit": unit,
            "source_lexeme": numbers[0], "form": form}, "projected"


def string_attr(key, value):
    return {"key": key, "value": {"type": "string", "value": value}, "qualifiers": {}}


def subject_id(name):
    return "subject:" + hashlib.sha256(name.encode()).hexdigest()[:20]


def reading_id(ident):
    return "reading:" + ident


def load_snapshot():
    manifest = json.loads(SELECTION.read_text())
    rows, paths, seen = [], [SELECTION, CHUNKS], set()
    for entry in manifest["accepted_packets"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Accepted source packet changed")
        data = json.loads(path.read_text())
        for row in data["records"]:
            if row["record_id"] in seen:
                raise ValueError("Duplicate source reading ID")
            seen.add(row["record_id"])
            rows.append(row)
        paths.append(path)
    if len(rows) != 502:
        raise ValueError("Expected the complete frozen 502-reading snapshot")
    chunks = {r["chunk_id"]: r for r in json.loads(CHUNKS.read_text())["chunks"]}
    for row in rows:
        citation = row["citation"]
        chunk = chunks[citation["chunk_id"]]
        if citation["text_sha256"] != chunk["text_sha256"]:
            raise ValueError("Source-text binding differs")
        if hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["text_sha256"]:
            raise ValueError("Archived source text changed")
        for span in citation["spans"]:
            if chunk["text"][span["start"]:span["end"]] != span["quote"]:
                raise ValueError("Citation span differs from its actual source")
    return rows, chunks, paths


def build_graph(records):
    graph = {"concepts": {"C-subject": {"name": "Source subject", "instanceOf": []},
                          "C-reading": {"name": "Specification reading", "instanceOf": []}},
             "entities": {}}
    sidecar, counts = {}, Counter()
    for row in records:
        eid = reading_id(row["record_id"])
        if eid in graph["entities"]:
            raise ValueError("Duplicate reading node")
        value = row["value"]
        attrs = [string_attr(key, val) for key, val in {
            "record_id": row["record_id"], "family_id": row["family_id"],
            "component": row["component"], "attribute": row["attribute"],
            "value_form": value["form"], "value_text": value["text"],
            "unit_raw_json": compact(value["unit_raw"]),
            "subjects_json": compact(row["subjects"]),
            "conditions_json": compact(row["conditions_all"]),
            "source_chunk": row["citation"]["chunk_id"],
            "citation_json": compact(row["citation"]),
            "source_record_json": compact(row),
        }.items()]
        # Full conditions retain order and duplicates in the JSON attributes.
        # Deduplicated filter projections avoid counting one reading twice.
        projected_conditions = set()
        for condition in row["conditions_all"]:
            key, val = "condition:" + condition["dimension"], condition["value"]
            if (key, val) not in projected_conditions:
                attrs.append(string_attr(key, val))
                projected_conditions.add((key, val))
        numeric, reason = numeric_projection(row)
        counts[reason] += 1
        if numeric:
            attrs.append({"key": numeric["key"],
                          "value": {"type": "quantity", "value": numeric["number"], "unit": numeric["unit"]},
                          "qualifiers": {}})
        node = {"name": row["record_id"], "instanceOf": ["C-reading"],
                "attributes": attrs, "relations": []}
        graph["entities"][eid] = node
        for subject in row["subjects"]:
            sid = subject_id(subject)
            old = graph["entities"].setdefault(sid, {"name": subject, "instanceOf": ["C-subject"],
                "attributes": [string_attr("subject_name", subject)], "relations": []})
            if old["name"] != subject:
                raise ValueError("Subject ID collision")
            relation = {"predicate": "has specification", "direction": "forward",
                        "object": eid, "qualifiers": {}}
            if relation not in old["relations"]:
                old["relations"].append(relation)
                node["relations"].append({"predicate": "has specification", "direction": "backward",
                                          "object": sid, "qualifiers": {}})
        sidecar[eid] = {"record": deepcopy(row), "numeric_projection": numeric,
                        "numeric_status": reason}
    return graph, sidecar, dict(counts)


def chain(actions):
    return [{"function": fn, "inputs": inputs, "dependencies": [] if i == 0 else [i-1]}
            for i, (fn, inputs) in enumerate(actions)]


def prototype_queries():
    return [
        {"id": "bridge-dev-wrs-count",
         "question": "WRS300资料中，最大系统耗电标示值严格低于4 kW的发射配置有多少个？",
         "derived_from_development_task": "RDR-DEV-01", "complete_original_task": False,
         "program": chain([("Find", ["WRS300"]), ("Relate", ["has specification", "forward"]),
             ("FilterStr", ["component", "system"]), ("FilterStr", ["attribute", "power_consumption"]),
             ("FilterStr", ["condition:quantity_role", "maximum"]),
             ("FilterNum", ["upper_bound_value", "4 kW", "<"]), ("Count", [])])},
        {"id": "bridge-dev-cyclone-count",
         "question": "Cyclone Pro资料列出的保护器件记录中，额定电流为15 A的电压—器件组合有多少个？",
         "derived_from_development_task": "RDR-DEV-11", "complete_original_task": False,
         "program": chain([("Find", ["Cyclone Pro"]), ("Relate", ["has specification", "forward"]),
             ("FilterStr", ["attribute", "current_rating"]), ("FilterNum", ["scalar_value", "15 A", "="]),
             ("Count", [])])},
    ]


def run_program(executor, program):
    episode = Episode(executor)
    program = validate_program(program)
    for action in program:
        observation = episode.step(**action)
        if not observation["ok"]:
            raise ValueError(f"Prototype execution failed: {observation}")
    selected = list(episode.handles[program[-1]["dependencies"][0]].value[0])
    observation = episode.finish(len(episode.handles)-1)
    if not observation["ok"]:
        raise ValueError("Could not submit prototype result")
    return episode, selected


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, default=PRIVATE)
    parser.add_argument("--public-dir", type=Path, default=PUBLIC)
    parser.add_argument("--date", default="2026-11-06")
    args = parser.parse_args()
    private, public = args.private_dir, args.public_dir
    if private.exists() or public.exists():
        raise FileExistsError("Bridge output already exists")
    records, chunks, paths = load_snapshot()
    graph, sidecar, counts = build_graph(records)
    for row in records:
        attrs = graph["entities"][reading_id(row["record_id"])]["attributes"]
        restored = json.loads(next(a["value"]["value"] for a in attrs if a["key"] == "source_record_json"))
        if restored != row:
            raise AssertionError("Source record did not round trip")
    write_new(private / "kb.json", graph)
    write_new(private / "source_records.json", sidecar)
    write_new(private / "source_chunks.json", chunks)
    executor = KoPLExecutor(private / "kb.json")
    prototypes = []
    for spec in prototype_queries():
        episode, selected = run_program(executor, spec["program"])
        # Independent declarative predicates scan ALL source readings.
        if spec["id"].endswith("wrs-count"):
            expected = [r for r in records if "WRS300" in r["subjects"] and r["component"] == "system"
                and r["attribute"] == "power_consumption"
                and {"dimension": "quantity_role", "value": "maximum"} in r["conditions_all"]
                and (n := numeric_projection(r)[0]) is not None and n["key"] == "upper_bound_value"
                and n["unit"] == "kW" and n["number"] < 4]
            configurations = {compact(r["conditions_all"]) for r in expected}
            assert len(configurations) == len(expected)
        else:
            expected = [r for r in records if "Cyclone Pro" in r["subjects"]
                and r["attribute"] == "current_rating" and (n := numeric_projection(r)[0]) is not None
                and n["key"] == "scalar_value" and n["unit"] == "A" and n["number"] == 15]
            assert all(r["component"] in {"inline fuse", "thermal breaker"} for r in expected)
        assert set(selected) == {reading_id(r["record_id"]) for r in expected}
        assert episode.prediction == str(len(expected))
        record = {**spec, "prediction": episode.prediction, "calls": episode.calls,
                  "selected_record_ids": [sidecar[e]["record"]["record_id"] for e in selected],
                  "source_chunks": sorted({sidecar[e]["record"]["citation"]["chunk_id"] for e in selected}),
                  "events": episode.events, "selected_source_records": [sidecar[e]["record"] for e in selected]}
        prototypes.append(record)
    write_new(private / "prototype_replays.json", prototypes)
    # Confirm actual original single-item answer behavior on a multi-item set.
    prefix = prototype_queries()[1]["program"][:-1]
    what = executor.execute(prefix + [{"function": "What", "inputs": [], "dependencies": [len(prefix)-1]}])
    query = executor.execute(prefix + [{"function": "QueryAttr", "inputs": ["record_id"], "dependencies": [len(prefix)-1]}])
    paths += [Path(__file__).relative_to(Path.cwd()), Path("experiments/condition_consistency/executor.py"),
              Path("experiments/agent_feedback/environment.py"),
              Path("external/kqa_pro_baselines/Program/executor_rule.py")]
    report = {"schema": "radar_kopl_record_bridge_cpu_v1", "date": args.date,
              "development_only": True, "human_gold": False, "gpu_model_calls": 0,
              "training_started": False, "source_readings": len(records),
              "reading_nodes": len(sidecar), "subject_nodes": len(graph["entities"]) - len(sidecar),
              "all_source_records_roundtrip": True, "gold_or_required_record_ids_in_graph_builder": False,
              "numeric_projection_counts": counts,
              "non_projected_records": len(records) - counts.get("projected", 0),
              "numeric_semantics": "Exact-unit projection only; scalar and lower/upper bound endpoints have distinct keys; original value.form/text/unit and all source fields retained.",
              "original_value_forms": dict(Counter(r["value"]["form"] for r in records)),
              "prototypes": [{k: v for k, v in p.items() if k not in {"events", "selected_source_records"}} for p in prototypes],
              "set_answer_probe": {"input_records": prototypes[1]["selected_record_ids"],
                                   "What": what, "QueryAttr_record_id": query,
                                   "conclusion": "Original What/QueryAttr return the first entity/value, not an answer set."},
              "scope": "Two Count subquestions executed on full 502-reading graph. The twelve multi-record candidate tasks and end-to-end model improvement are not evaluated.",
              "inputs_sha256": {str(p): sha256(p) for p in paths},
              "outputs_sha256": {str(p): sha256(p) for p in sorted(private.glob("*.json"))}}
    write_new(public / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("source_readings", "subject_nodes", "numeric_projection_counts", "prototypes", "set_answer_probe")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
