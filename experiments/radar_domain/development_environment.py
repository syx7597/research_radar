"""Exact, source-scoped tools for the exposed AI-reviewed development packet.

This is a new interface compatibility probe, not the public KoPL benchmark
executor and not admission of radar facts. No question ID, reference answer, or
reference query is accepted by the execution layer. Rendering is deterministic;
its source qualifications are reviewed data, not model-generated reasoning.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from experiments.agent_feedback.environment import tool_schemas

MAX_CALLS = 24
SIGNATURES = {"Find": (0, 1), "QueryAttr": (1, 1),
              "QueryAttrUnderCondition": (1, 3)}
TERMINALS = {"QueryAttr", "QueryAttrUnderCondition"}
QUALIFIERS = {"condition_raw", "event_type"}
REVIEW_SCOPE = "AI_only_development"
RECORD_FIELDS = (
    "fact_id", "entity_name", "source_subject", "attribute", "event_type",
    "condition_raw", "value_kind", "value_status", "value", "min_value",
    "max_value", "unit_std", "scope_notes_zh", "unknown_scope", "citation",
    "original_entity_attribution_usable",
)
ATTRIBUTE_MEANINGS = {
    "type_description": "来源主体的类型描述", "service_entry": "来源中的服役/初始作战能力事件",
    "lifecycle_event": "生命周期事件（包括取消，不等于服役）", "frequency": "频率",
    "prf": "脉冲重复频率", "pulse_width": "脉冲宽度", "power": "功率",
}
FUNCTION_HELP = """Supported functions only:
Find(name): exact source lookup anchor; no dependencies, one string input. This does NOT assert that the source describes that equipment.
QueryAttr(h, key): all source readings of that attribute; one Find handle dependency, one string input.
QueryAttrUnderCondition(h, key, qualifier_key, qualifier_value): exact attribute and qualifier match; one Find handle dependency, three string inputs. qualifier_key is condition_raw or event_type. Preserve exact capitalization.
QueryAttr and QueryAttrUnderCondition return separate structured records, including ranges, unknown values, source subject and evidence. An omitted selector retains all matching records. A missing field is not an unconditional fact. Empty results mean no matching reading in this packet, not zero or real-world absence.
Never replace source subject, conditions, event type, unit, interval endpoints, unknown scope or source limitations with an equipment claim. The records are AI-reviewed source readings for development, not independently verified facts.
"""
SYSTEM_PROMPT = (
    "Answer the question by selecting relevant source readings through real tools. "
    "Make exactly one tool call per turn, then read its observation. "
    "Use step(function, inputs, dependencies); inputs are strings, dependencies are "
    "integer handles from earlier successful steps. Errors create no handle; earlier "
    "handles remain available. Use finish(answer_handle) to select a QueryAttr or "
    "QueryAttrUnderCondition result, then respond Done. Do not answer from memory. "
    "At most 24 tool calls. The final evidence card is rendered separately; your task "
    "is source-record selection, not free-form semantic adjudication.\n" + FUNCTION_HELP
)


class DevelopmentKB:
    """Load validated development readings; source bytes were checked upstream.

    The file hash is exposed for run freezing. Citation hashes bind upstream source
    snapshots; this class does not authenticate or independently re-review them.
    """

    def __init__(self, path: str | Path):
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("development document must be an object")
        if (document.get("independent_gold") is not False or
                document.get("human_verified") is not False or
                document.get("split") != "development_only"):
            raise ValueError("only explicitly non-gold, non-human development readings are allowed")
        if not isinstance(document.get("version"), str) or not document["version"]:
            raise ValueError("development version is required")
        records = document.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("development records must be a nonempty list")
        seen = set()
        for record in records:
            self._validate_record(record)
            if record["fact_id"] in seen:
                raise ValueError("duplicate fact_id")
            seen.add(record["fact_id"])
        self.version = document["version"]
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self._records = tuple(deepcopy(records))
        self.entities = tuple(sorted({r["entity_name"] for r in records}))
        self.attributes = tuple(sorted({r["attribute"] for r in records}))

    @staticmethod
    def _validate_record(record: dict):
        if not isinstance(record, dict) or set(RECORD_FIELDS) - set(record):
            raise ValueError("record is missing required source-scoped fields")
        if record.get("review_status") != REVIEW_SCOPE:
            raise ValueError("record must retain AI development review status")
        if any(record.get(key) is True for key in ("independent_gold", "human_verified", "equipment_truth_verified")):
            raise ValueError("development records cannot assert independent truth admission")
        for key in ("fact_id", "entity_name", "source_subject", "attribute", "value_kind", "value_status"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"nonempty {key} required")
        for key in ("event_type", "condition_raw", "value", "min_value", "max_value", "unit_std", "unknown_scope"):
            if record[key] is not None and (not isinstance(record[key], str) or not record[key]):
                raise ValueError(f"{key} must be a nonempty string or null")
        if type(record["original_entity_attribution_usable"]) is not bool:
            raise ValueError("original attribution usability must be boolean")
        notes = record["scope_notes_zh"]
        if not ((isinstance(notes, str) and notes) or
                (isinstance(notes, list) and notes and all(isinstance(n, str) and n for n in notes))):
            raise ValueError("source scope notes are required")
        citation = record["citation"]
        if not isinstance(citation, dict):
            raise ValueError("citation must be an object")
        if not all(isinstance(citation.get(k), str) and citation[k]
                   for k in ("source_uri", "source_sha256", "evidence_text")):
            raise ValueError("citation source, hash and evidence are required")
        if not re.fullmatch(r"[0-9a-f]{64}", citation["source_sha256"]):
            raise ValueError("invalid source SHA256")
        if not isinstance(citation.get("locator"), dict) or not citation["locator"]:
            raise ValueError("citation requires parsed source locator")
        if record["value_kind"] == "range":
            if record["value"] is not None or record["min_value"] is None or record["max_value"] is None:
                raise ValueError("range requires two endpoints and no scalar collapse")
            try:
                lower, upper = Decimal(record["min_value"]), Decimal(record["max_value"])
                if not lower.is_finite() or not upper.is_finite() or lower > upper:
                    raise ValueError("invalid range ordering")
            except InvalidOperation as exc:
                raise ValueError("range endpoints must be numeric") from exc
        elif record["min_value"] is not None or record["max_value"] is not None:
            raise ValueError("non-range cannot contain interval endpoints")
        if record["value_kind"] == "unknown":
            if any(record[k] is not None for k in ("value", "min_value", "max_value")) or not record["unknown_scope"]:
                raise ValueError("unknown value must be null and source-scoped")
        elif record["value_kind"] != "range" and record["value"] is None:
            raise ValueError("known non-range reading requires its value")

    @property
    def records(self) -> list[dict]:
        return deepcopy(list(self._records))

    def find(self, name: str) -> list[int]:
        return [i for i, record in enumerate(self._records) if record["entity_name"] == name]

    def query(self, indices: list[int], attribute: str, qualifier_key: str | None = None,
              qualifier_value: str | None = None) -> list[dict]:
        if qualifier_key is not None and qualifier_key not in QUALIFIERS:
            raise ValueError("unsupported_qualifier_key")
        return [deepcopy(self._records[i]) for i in indices
                if self._records[i]["attribute"] == attribute and
                (qualifier_key is None or self._records[i][qualifier_key] == qualifier_value)]

    def schema_prompt(self) -> str:
        attributes = [f"{key} ({ATTRIBUTE_MEANINGS.get(key, key)})" for key in self.attributes]
        qualifiers = {key: sorted({record[key] for record in self._records if record[key] is not None})
                      for key in sorted(QUALIFIERS)}
        return ("\nExact source lookup anchors: " + json.dumps(self.entities, ensure_ascii=False) +
                "\nAttribute keys: " + "; ".join(attributes) +
                "\nExact qualifier vocabulary: " + json.dumps(qualifiers, ensure_ascii=False, separators=(",", ":")))


@dataclass
class DevelopmentHandle:
    function: str
    value: Any
    inputs: list[str]
    dependencies: list[int]


class DevelopmentEpisode:
    def __init__(self, kb: DevelopmentKB, max_calls: int = MAX_CALLS):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.kb, self.max_calls = kb, max_calls
        self.handles: list[DevelopmentHandle] = []
        self.events: list[dict] = []
        self.calls, self.done, self.prediction, self.selected = 0, False, None, None

    def _begin(self):
        if self.done:
            raise ValueError("episode_finished")
        if self.calls >= self.max_calls:
            self.done = True
            raise ValueError("call_budget_exhausted")
        self.calls += 1

    def _check(self, function, inputs, dependencies):
        if not isinstance(function, str) or function not in SIGNATURES:
            raise ValueError("unknown_function")
        dep_count, arg_count = SIGNATURES[function]
        if (not isinstance(inputs, list) or len(inputs) != arg_count or
                any(not isinstance(x, str) or not x or len(x) > 2048 for x in inputs)):
            raise ValueError(f"{function} requires {arg_count} nonempty bounded string inputs")
        if (not isinstance(dependencies, list) or len(dependencies) != dep_count or
                any(type(x) is not int or not 0 <= x < len(self.handles) for x in dependencies)):
            raise ValueError(f"{function} requires {dep_count} existing integer handles")
        if any(self.handles[x].function != "Find" for x in dependencies):
            raise ValueError("incompatible_dependency_type")
        if function == "QueryAttrUnderCondition" and inputs[1] not in QUALIFIERS:
            raise ValueError("unsupported_qualifier_key")

    def _observation(self, handle: int) -> dict:
        item = self.handles[handle]
        result = {"ok": True, "handle": handle, "scope": REVIEW_SCOPE}
        if item.function == "Find":
            records = [self.kb._records[i] for i in item.value]
            names = sorted({r["entity_name"] for r in records})
            result.update(type="entities", count=len(names), sample=[{"name": name} for name in names],
                          attributes=sorted({r["attribute"] for r in records}),
                          qualifiers={key: sorted({r[key] for r in records if r[key] is not None})
                                      for key in sorted(QUALIFIERS)},
                          source_subjects=sorted({r["source_subject"] for r in records}),
                          attribution_warning=any(not r["original_entity_attribution_usable"] for r in records))
        else:
            result.update(type="value", count=len(item.value),
                          value=[{key: deepcopy(record[key]) for key in RECORD_FIELDS} for record in item.value])
        return result

    def _record_event(self, tool: str, arguments: dict, result: dict):
        self.events.append({"tool": tool, "arguments": deepcopy(arguments), "observation": deepcopy(result)})
        if self.calls >= self.max_calls:
            self.done = True
        return result

    def step(self, function: str, inputs: list[str], dependencies: list[int]) -> dict:
        try:
            self._begin()
            self._check(function, inputs, dependencies)
            if function == "Find":
                value = self.kb.find(inputs[0])
            else:
                indices = self.handles[dependencies[0]].value
                value = self.kb.query(indices, inputs[0], *(inputs[1:] if len(inputs) > 1 else []))
            handle = len(self.handles)
            self.handles.append(DevelopmentHandle(function, value, list(inputs), list(dependencies)))
            result = self._observation(handle)
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:240]}
        return self._record_event("step", {"function": function, "inputs": inputs, "dependencies": dependencies}, result)

    def finish(self, answer_handle: int) -> dict:
        try:
            self._begin()
            if type(answer_handle) is not int or not 0 <= answer_handle < len(self.handles):
                raise ValueError("unknown_answer_handle")
            item = self.handles[answer_handle]
            if item.function not in TERMINALS:
                raise ValueError("finish_requires_answer_function")
            self.prediction = deepcopy(item.value)
            self.selected, self.done = answer_handle, True
            result = {"ok": True, "done": True, "answer": deepcopy(self.prediction), "scope": REVIEW_SCOPE}
        except (ValueError, TypeError) as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:240]}
        return self._record_event("finish", {"answer_handle": answer_handle}, result)


def prompt_messages(question: str, kb: DevelopmentKB, program: bool = False) -> list[dict]:
    if not isinstance(question, str) or not question:
        raise ValueError("nonempty question text required")
    if program:
        instruction = ("Translate the question into a complete executable KoPL program. "
                       "Output only function and literal inputs separated by <arg>, with steps separated by <func>. "
                       "Find has no dependencies; each QueryAttr or QueryAttrUnderCondition uses the immediately "
                       "preceding Find result. The final function must produce an answer. "
                       "Do not output JSON, markdown, finish or explanatory text.\n" + FUNCTION_HELP)
    else:
        instruction = SYSTEM_PROMPT
    return [{"role": "system", "content": instruction + kb.schema_prompt()},
            {"role": "user", "content": question}]


def execute_program(kb: DevelopmentKB, text: str) -> DevelopmentEpisode:
    """Execute P's linear syntax with the same query semantics as agent steps.

    All literals must be exact. There is no code execution, fuzzy normalization,
    alternate syntax extraction, or repair of failed programs. A failed step ends
    P's one-shot episode; successful terminal output is selected via finish.
    """
    episode = DevelopmentEpisode(kb)
    try:
        if not isinstance(text, str) or not text.strip() or len(text) > 65536:
            raise ValueError("program must be bounded nonempty text")
        chunks = text.strip().split("<func>")
        if len(chunks) + 1 > MAX_CALLS:
            raise ValueError("program_exceeds_call_budget")
        parts = [[part.strip() for part in chunk.split("<arg>")] for chunk in chunks]
        if parts[-1][0] not in TERMINALS:
            raise ValueError("program_requires_terminal_answer_function")
        for item in parts:
            function, inputs = item[0], item[1:]
            dependencies = [] if function == "Find" else [len(episode.handles) - 1]
            observation = episode.step(function, inputs, dependencies)
            if not observation["ok"]:
                episode.done = True
                return episode
        episode.finish(len(episode.handles) - 1)
    except ValueError as exc:
        try:
            episode._begin()
        except ValueError:
            pass
        episode.events.append({"tool": "invalid", "arguments": {}, "observation": {
            "ok": False, "error": "ProgramFormatError", "detail": str(exc)[:240]}})
    episode.done = True
    return episode


def render_records(records: list[dict]) -> str:
    """Deterministic evidence cards, explicitly not a model semantic answer."""
    if not isinstance(records, list):
        raise ValueError("render_records requires a selected record list")
    lines = ["AI 审阅开发证据卡（确定性字段展示，非模型生成回答；未经独立人工事实核验）"]
    if not records:
        return "\n".join(lines + ["当前开发资料中没有匹配记录；这不表示现实中不存在该信息，也不表示数值为零。"])
    for record in records:
        citation = record["citation"]
        kind = record["value_kind"]
        if kind == "unknown":
            value = f"未知（范围：{record['unknown_scope']}；未记录可用数值）"
        elif kind == "range":
            value = f"{record['min_value']}–{record['max_value']}"
        else:
            value = record["value"]
        if record["unit_std"] is not None:
            value += " " + record["unit_std"]
        notes = record["scope_notes_zh"]
        if isinstance(notes, list):
            notes = "；".join(notes)
        lines.extend([f"[{record['fact_id']}] 来源检索锚点：{record['entity_name']}；来源主体：{record['source_subject']}。",
                      f"字段：{record['attribute']}；快照记载：{value}；状态：{record['value_status']}。"])
        if record["event_type"] is not None:
            lines.append(f"事件：{record['event_type']}。")
        if record["condition_raw"] is not None:
            lines.append(f"原文条件/部件标签：{record['condition_raw']}（保留原文，不扩展解释）。")
        else:
            lines.append("条件字段未记载，不代表无条件适用。")
        if not record["original_entity_attribution_usable"]:
            lines.append("原始设备归属断言已隔离：此记录不能直接支持检索锚点设备的相应事实。")
        lines.extend([f"适用边界：{notes}", f"原文：{citation['evidence_text']}",
                      f"来源：{citation['source_uri']}；定位：{json.dumps(citation['locator'], ensure_ascii=False, separators=(',', ':'))}；SHA256：{citation['source_sha256']}"])
    return "\n".join(lines)
