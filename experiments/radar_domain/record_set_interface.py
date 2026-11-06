"""Domain-only source-record finalization over the unchanged KoPL executor.

finish_records submits the complete executed reading set. finish retains scalar
Count/single-reading outputs and attaches their actual source readings. No answer
key, task ID, required-record list, or answer-generation model is accepted.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import tempfile

from experiments.agent_feedback.environment import (
    Episode, ENTITY_FUNCTIONS, FUNCTION_HELP, tool_schemas as public_tool_schemas)
from experiments.agent_feedback.recovery import sha256
from experiments.condition_consistency.executor import KoPLExecutor
from .kopl_record_bridge import (
    NUMERIC_KEYS, build_graph, load_snapshot, numeric_projection, reading_id,
    compact, prototype_queries, write_new)

PUBLIC = Path("results/radar_domain/domain_interface_20261106")
PRIVATE = Path("data/radar_domain_recovery_v1/domain_interface_20261106")
SOURCE_FIELDS = ("chunk_id", "source_id", "source_uri", "source_sha256", "page_one_based")
SINGLE_READING_FUNCTIONS = {"What", "QueryAttr", "QueryAttrUnderCondition", "QueryAttrQualifier"}
SCALAR_SOURCE_FUNCTIONS = SINGLE_READING_FUNCTIONS | {"Count", "SelectAmong", "SelectBetween"}
DOMAIN_SYSTEM = (
    "Answer the question by executing KoPL operations against the source-reading graph. "
    "Each specification reading retains its subjects, component, value form, text, unit, "
    "conditions and source citation. Use exactly one tool call per turn. "
    "Use step(function, inputs, dependencies) to compute query results. "
    "Use finish_records(record_handle) to submit ALL reading entities in a computed set; "
    "an empty set is a valid answer within this graph. The tool returns complete source "
    "records; do not generate values or citations yourself. "
    "For Count or a scalar from a single reading, use finish(answer_handle). "
    "What and QueryAttr require a single entity; they do not enumerate a set. "
    "Errors create no handle and earlier handles remain available. At most 24 calls.\n"
    + FUNCTION_HELP
)


class SourceRecordStore:
    """Validate source records actually stored in the full execution graph."""

    def __init__(self, executor, chunks):
        self._records = {}
        self._chunks = deepcopy(chunks)
        checked_chunks = set()
        for eid, entity in executor.engine.entities.items():
            if "C-reading" not in entity["instanceOf"]:
                continue
            attrs = {}
            for attr in entity["attributes"]:
                attrs.setdefault(attr["key"], []).append(attr["value"])
            raw = attrs.get("source_record_json", [])
            if len(raw) != 1 or raw[0].type != "string":
                raise ValueError("Reading lacks one complete source record")
            record = json.loads(raw[0].value)
            if eid != reading_id(record["record_id"]) or entity["name"] != record["record_id"]:
                raise ValueError("Reading identity differs from source record")
            projections = {
                "record_id": record["record_id"], "family_id": record["family_id"],
                "component": record["component"], "attribute": record["attribute"],
                "value_form": record["value"]["form"], "value_text": record["value"]["text"],
                "unit_raw_json": compact(record["value"]["unit_raw"]),
                "subjects_json": compact(record["subjects"]),
                "conditions_json": compact(record["conditions_all"]),
                "citation_json": compact(record["citation"]),
                "source_chunk": record["citation"]["chunk_id"],
            }
            for key, expected in projections.items():
                actual = attrs.get(key, [])
                if len(actual) != 1 or actual[0].type != "string" or actual[0].value != expected:
                    raise ValueError(f"Execution field differs from source record: {key}")
            numeric, _ = numeric_projection(record)
            for key, values in attrs.items():
                if key not in NUMERIC_KEYS.values() and any(v.type == "quantity" for v in values):
                    raise ValueError(f"Unexpected numeric projection: {key}")
            for key in NUMERIC_KEYS.values():
                actual = attrs.get(key, [])
                if numeric is None or numeric["key"] != key:
                    if actual:
                        raise ValueError(f"Unexpected numeric projection: {key}")
                elif (len(actual) != 1 or actual[0].type != "quantity"
                      or type(actual[0].value) not in (int, float)
                      or actual[0].value != numeric["number"]
                      or actual[0].unit != numeric["unit"]):
                    raise ValueError(f"Numeric projection differs from source record: {key}")
            citation = record["citation"]
            chunk = self._chunks[citation["chunk_id"]]
            if citation["chunk_id"] not in checked_chunks:
                actual_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
                if actual_hash != chunk["text_sha256"]:
                    raise ValueError("Actual source text hash differs from cached hash")
                checked_chunks.add(citation["chunk_id"])
            if citation["text_sha256"] != chunk["text_sha256"]:
                raise ValueError("Citation binds different source text")
            for span in citation["spans"]:
                if chunk["text"][span["start"]:span["end"]] != span["quote"]:
                    raise ValueError("Citation span differs from source text")
            self._records[eid] = record
        if not self._records:
            raise ValueError("The graph has no source-reading nodes")

    def materialize(self, entity_ids):
        if not isinstance(entity_ids, list) or any(type(e) is not str for e in entity_ids):
            raise ValueError("Expected actual reading entity IDs")
        if any(e not in self._records for e in entity_ids):
            raise ValueError("Selected entities must all be source readings")
        ordered = sorted(set(entity_ids), key=lambda eid: self._records[eid]["record_id"])
        records = [deepcopy(self._records[e]) for e in ordered]
        chunks = sorted({r["citation"]["chunk_id"] for r in records})
        sources = {key: {field: deepcopy(self._chunks[key].get(field)) for field in SOURCE_FIELDS}
                   for key in chunks}
        return records, sources

    @property
    def count(self):
        return len(self._records)


def typed_scalar(value):
    if type(value) in (str, int, float, bool):
        return {"type": type(value).__name__, "value": value}
    if type(value).__name__ == "ValueClass":
        raw = value.value.isoformat() if isinstance(value.value, date) else value.value
        result = {"type": value.type, "value": raw}
        if value.type == "quantity":
            result["unit"] = value.unit
        return result
    raise ValueError("Unsupported scalar answer type")


class RecordSetEpisode(Episode):
    """A shared domain interface; behavior never branches on checkpoint labels."""

    def __init__(self, executor, store, max_calls=24):
        super().__init__(executor, max_calls)
        self.store = store
        self.answer = None

    def _entities(self, handle):
        if type(handle) is not int or not 0 <= handle < len(self.handles):
            raise ValueError("unknown_record_handle")
        item = self.handles[handle]
        if item.function not in ENTITY_FUNCTIONS:
            raise ValueError("record_handle_requires_entity_result")
        ids, _ = item.value
        if not isinstance(ids, list):
            raise ValueError("malformed_entity_result")
        return ids

    def step(self, function, inputs, dependencies):
        # Domain contract guards the legacy first-item behavior explicitly.
        if isinstance(function, str) and function in SINGLE_READING_FUNCTIONS:
            try:
                self._check(function, inputs, dependencies)
                if len(self._entities(dependencies[0])) != 1:
                    raise ValueError("single_entity_required_use_finish_records_for_sets")
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                try:
                    self._begin()
                    result = {"ok": False, "error": "ValueError", "detail": str(exc)}
                except ValueError as limit:
                    result = {"ok": False, "error": "ValueError", "detail": str(limit)}
                self.events.append({"tool": "step", "arguments": {
                    "function": function, "inputs": inputs, "dependencies": dependencies},
                    "observation": result})
                return result
        return super().step(function, inputs, dependencies)

    def _support_ids(self, answer_handle):
        item = self.handles[answer_handle]
        if item.function in SCALAR_SOURCE_FUNCTIONS:
            return [eid for dep in item.dependencies for eid in self._entities(dep)]
        if item.function.startswith("Verify") and len(item.dependencies) == 1:
            return self._support_ids(item.dependencies[0])
        raise ValueError("unsupported_domain_scalar_terminal")

    def _finish_event(self, tool, args, producer):
        try:
            self._begin()
            payload, selected, prediction = producer()
            self.answer, self.selected, self.prediction = payload, selected, prediction
            self.done = True
            result = {"ok": True, "done": True, "answer": deepcopy(payload)}
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            result = {"ok": False, "error": "ValueError", "detail": str(exc)}
        self.events.append({"tool": tool, "arguments": args, "observation": result})
        return result

    def finish_records(self, record_handle):
        def produce():
            ids = self._entities(record_handle)
            records, sources = self.store.materialize(ids)
            answer = {"kind": "records", "records": records, "sources": sources,
                      "input_entity_count": len(ids), "unique_record_count": len(records),
                      "scope": "selected_readings_in_current_graph"}
            return answer, record_handle, compact(answer)
        return self._finish_event("finish_records", {"record_handle": record_handle}, produce)

    def finish(self, answer_handle):
        def produce():
            if type(answer_handle) is not int or not 0 <= answer_handle < len(self.handles):
                raise ValueError("unknown_answer_handle")
            item = self.handles[answer_handle]
            ids = self._support_ids(answer_handle)
            records, sources = self.store.materialize(ids)
            value = typed_scalar(item.value)
            answer = {"kind": "scalar", "value": value, "source_records": records,
                      "sources": sources, "input_entity_count": len(ids),
                      "unique_record_count": len(records),
                      "scope": "computed_from_current_graph"}
            return answer, answer_handle, str(item.value)
        return self._finish_event("finish", {"answer_handle": answer_handle}, produce)


def tool_schemas():
    schemas = deepcopy(public_tool_schemas())
    for schema in schemas:
        if schema["function"]["name"] == "finish":
            schema["function"]["description"] = (
                "Submit Count or a scalar from source readings, preserving actual supporting records.")
    schemas.append({"type": "function", "function": {
        "name": "finish_records",
        "description": "Submit every source reading in an actual entity-result handle, including an empty set. Values, conditions and citations are returned by the executor.",
        "parameters": {"type": "object", "properties": {
            "record_handle": {"type": "integer", "description": "Handle of a computed reading-entity set."}},
            "required": ["record_handle"], "additionalProperties": False}}})
    return sorted(schemas, key=lambda s: s["function"]["name"])


def prompt_messages(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A question string is required")
    return [{"role": "system", "content": DOMAIN_SYSTEM}, {"role": "user", "content": question}]


def execute_response(episode, text):
    try:
        if not isinstance(text, str):
            raise ValueError("tool_response_must_be_text")
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
        if len(matches) != 1:
            raise ValueError("exactly_one_tool_call_required")
        call = json.loads(matches[0])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("tool_call_requires_name_and_arguments")
        expected = {"step": {"function", "inputs", "dependencies"},
                    "finish": {"answer_handle"}, "finish_records": {"record_handle"}}
        name, args = call["name"], call["arguments"]
        if (not isinstance(name, str) or name not in expected
                or not isinstance(args, dict) or set(args) != expected[name]):
            raise ValueError("unknown_tool_or_invalid_arguments")
    except (ValueError, TypeError) as exc:
        try:
            episode._begin()
            result = {"ok": False, "error": "ActionFormatError", "detail": str(exc)}
        except ValueError as limit:
            result = {"ok": False, "error": "ValueError", "detail": str(limit)}
        episode.events.append({"tool": "invalid", "arguments": {}, "observation": result})
    else:
        result = getattr(episode, name)(**args)
    if episode.calls >= episode.max_calls:
        episode.done = True
    return result


def from_graph(graph, chunks):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kb.json"
        path.write_text(json.dumps(graph, ensure_ascii=False))
        executor = KoPLExecutor(path)
    return executor, SourceRecordStore(executor, chunks)


def load_bridge(kb_path=None, chunks_path=None):
    """Load explicit full source inputs, or build from the current bridge code."""
    if (kb_path is None) != (chunks_path is None):
        raise ValueError("Both explicit KB and source-chunk paths are required")
    if kb_path is None:
        records, chunks, _ = load_snapshot()
        graph, _, _ = build_graph(records)
        executor, store = from_graph(graph, chunks)
    else:
        executor = KoPLExecutor(kb_path)
        store = SourceRecordStore(executor, json.loads(Path(chunks_path).read_text()))
    if store.count != 502:
        raise ValueError("All 502 competing source readings must remain available")
    return executor, store


def run_actions(executor, store, program, finish_kind="records"):
    episode = RecordSetEpisode(executor, store)
    for action in program:
        result = execute_response(episode, "<tool_call>" + compact({
            "name": "step", "arguments": action}) + "</tool_call>")
        if not result["ok"]:
            raise ValueError(result)
    tool, key = ("finish_records", "record_handle") if finish_kind == "records" else ("finish", "answer_handle")
    result = execute_response(episode, "<tool_call>" + compact({
        "name": tool, "arguments": {key: len(episode.handles)-1}}) + "</tool_call>")
    if not result["ok"]:
        raise ValueError(result)
    return episode


def main():
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("Retain previous interface evidence")
    records, chunks, source_paths = load_snapshot()
    graph, _, _ = build_graph(records)
    executor, store = from_graph(graph, chunks)
    if store.count != 502:
        raise ValueError("Full 502-reading graph required")
    base = prototype_queries()[1]["program"][:-1]
    variants = []
    for label, number, terminal in (("15A_set", "15 A", "records"),
                                    ("20A_set", "20 A", "records"),
                                    ("99A_empty_set", "99 A", "records"),
                                    ("15A_count", "15 A", "scalar")):
        program = deepcopy(base)
        program[-1]["inputs"][1] = number
        if terminal == "scalar":
            program.append({"function": "Count", "inputs": [], "dependencies": [len(program)-1]})
        episode = run_actions(executor, store, program, terminal)
        records = episode.answer.get("records", episode.answer.get("source_records"))
        variants.append({"case": label, "program": program, "calls": episode.calls,
                         "answer": episode.answer, "events": episode.events,
                         "record_ids": [r["record_id"] for r in records]})
    all_records = run_actions(executor, store, [
        {"function": "FindAll", "inputs": [], "dependencies": []},
        {"function": "FilterConcept", "inputs": ["Specification reading"], "dependencies": [0]}])
    assert len(all_records.answer["records"]) == 502
    for record in all_records.answer["records"]:
        assert record == store._records[reading_id(record["record_id"])]
    write_new(PRIVATE / "replays.json", variants)
    write_new(PRIVATE / "all_records_output.json", all_records.answer)
    write_new(PRIVATE / "kb.json", graph)
    write_new(PRIVATE / "source_chunks.json", chunks)
    write_new(PRIVATE / "tool_contract.json", {"tools": tool_schemas(), "system_prompt": DOMAIN_SYSTEM})
    public_paths = [Path(__file__).relative_to(Path.cwd()),
                    Path("experiments/radar_domain/kopl_record_bridge.py"),
                    *source_paths,
                    Path("experiments/agent_feedback/environment.py"),
                    Path("experiments/agent_feedback/inference.py"),
                    Path("experiments/condition_consistency/executor.py")]
    frozen = json.loads(Path("results/agent_feedback/same_state_v1/design_protocol.json").read_text())
    for path in public_paths[-3:]:
        assert sha256(path) == frozen["inputs_sha256"][str(path)]
    report = {"schema": "radar_domain_set_interface_cpu_v1", "date": "2026-11-06",
              "revision": "numeric_and_source_text_integrity_v2",
              "development_only": True, "human_gold": False, "gpu_model_calls": 0,
              "training_started": False, "source_readings_available": store.count,
              "complete_readings_returned_in_full_set_check": len(all_records.answer["records"]),
              "all_original_record_fields_preserved": True,
              "source_consistency_validation": {
                  "numeric_field_type_value_unit": True,
                  "missing_duplicate_unexpected_numeric_projections_rejected": True,
                  "source_text_sha256_recomputed": True},
              "final_answer_generated_from": "actual executed handles and bound source records",
              "answer_key_or_candidate_tasks_read_by_runtime": False,
              "tool_names": [s["function"]["name"] for s in tool_schemas()],
              "cases": [{"case": v["case"], "record_ids": v["record_ids"], "calls": v["calls"],
                        "answer_kind": v["answer"]["kind"], "scalar": v["answer"].get("value")}
                       for v in variants],
              "scope": "Collection/scalar interface only. The 12 multi-record tasks, range/correspondence parsing, joins and source-unsupported configuration judgments are not completed.",
              "inputs_sha256": {str(p): sha256(p) for p in public_paths},
              "outputs_sha256": {str(p): sha256(p) for p in PRIVATE.glob("*.json")}}
    archive = Path("data/radar_domain_recovery_v1/domain_interface_20261106_initial")
    if archive.exists():
        report["previous_cpu_revision"] = {
            "archive": str(archive),
            "summary_sha256": sha256(archive / "public/summary.json"),
            "archive_manifest_sha256": sha256(archive / "archive_manifest.json"),
            "reason": "Add quantity/source-text consistency checks without changing source records or prior model results."}
    write_new(PUBLIC / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
