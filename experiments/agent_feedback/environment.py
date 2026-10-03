"""Episode-local, data-only KoPL tools; observations never contain gold labels.

The shared official engine is read-only after initialization (Find aliasing is
isolated by the existing adapter). Handles are appended only on successful calls.
An empty set is a valid result. A separate finish call selects an answer handle.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from experiments.condition_consistency.executor import (
    ENTITY_FUNCTIONS, FACT_FUNCTIONS, SIGNATURES, TERMINALS, VALUE_FUNCTIONS,
    KoPLExecutor, compare_answers, validate_program,
)

MAX_CALLS = 24
COMPARISONS = {"=", "!=", "<", ">"}
FUNCTION_HELP = """FindAll(): all entities. Find(name): entities/concepts with that name.
FilterConcept(h, concept): keep entities of a concept.
FilterStr(h,key,value), FilterNum/Year/Date(h,key,value,op): filter attributes.
QFilterStr(h,key,value), QFilterNum/Year/Date(h,key,value,op): filter qualifiers on attached facts.
Relate(h,predicate,direction): follow a relation, direction forward or backward.
And(h1,h2), Or(h1,h2): intersect or union entity sets.
What(h): entity name. Count(h): number of entities.
SelectBetween(h1,h2,key,op): compare, op greater or less.
SelectAmong(h,key,op): select extreme, op largest or smallest.
QueryAttr(h,key): attribute value.
QueryAttrUnderCondition(h,key,qualifier_key,qualifier_value): conditional attribute.
VerifyStr(h,value), VerifyNum/Year/Date(h,value,op): verify a queried value.
QueryRelation(h1,h2): relation between entities.
QueryAttrQualifier(h,key,value,qualifier_key): qualifier of an attribute fact.
QueryRelationQualifier(h1,h2,predicate,qualifier_key): qualifier of a relation.
Numeric/date operators are =, !=, <, >. Quantities include the exact unit, e.g. '5 metre'.
"""
SYSTEM_PROMPT = (
    "Answer the question by executing KoPL tools against the knowledge base. "
    "Make exactly one tool call per turn, then read its actual observation. "
    "Use step(function, inputs, dependencies); inputs are strings, dependencies are "
    "integer handles returned by earlier successful steps. Do not copy entity lists. "
    "Keep every question condition. Empty results, zero and no can be correct. "
    "Errors do not create handles; earlier handles remain available. "
    "Use finish(answer_handle) to select a result from an answer-producing function; "
    "then respond Done. Do not answer from memory. At most 24 tool calls.\n" + FUNCTION_HELP
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "<tool_call>\n" +
            compact({"name": name, "arguments": arguments}) + "\n</tool_call>"}


def prompt_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}]


def tool_schemas() -> list[dict]:
    schemas = [
        {"type": "function", "function": {"name": "step",
         "description": "Execute one KoPL function and retain its result as an integer handle.",
         "parameters": {"type": "object", "properties": {
             "function": {"type": "string", "description": "KoPL function name from the instruction."},
             "inputs": {"type": "array", "items": {"type": "string"}, "description": "Literal string arguments, excluding dependency handles."},
             "dependencies": {"type": "array", "items": {"type": "integer"}, "description": "Handles returned by previous successful calls."}},
             "required": ["function", "inputs", "dependencies"]},
         "return": {"type": "object", "description": "Result type, handle, bounded observation, or an execution error."}}},
        {"type": "function", "function": {"name": "finish",
         "description": "Finish the episode using a previously computed answer result.",
         "parameters": {"type": "object", "properties": {
             "answer_handle": {"type": "integer", "description": "Integer handle of an answer-producing KoPL function."}},
             "required": ["answer_handle"]},
         "return": {"type": "object", "description": "The computed answer and termination status, or an error."}}},
    ]
    # Match native TRL environment reflection ordering before any model training.
    return sorted(schemas, key=lambda schema: schema["function"]["name"])


@dataclass
class Handle:
    function: str
    value: Any
    inputs: list[str]
    dependencies: list[int]


class Episode:
    """No gold program or answer is accepted by this execution class."""

    def __init__(self, executor: KoPLExecutor, max_calls: int = MAX_CALLS):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.executor = executor
        self.max_calls = max_calls
        self.handles: list[Handle] = []
        self.calls = 0
        self.done = False
        self.prediction = None
        self.selected = None
        self.events = []

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
                any(not isinstance(x, str) or len(x) > 2048 for x in inputs)):
            raise ValueError(f"{function} requires {arg_count} bounded string inputs")
        if (not isinstance(dependencies, list) or len(dependencies) != dep_count or
                any(type(x) is not int or not 0 <= x < len(self.handles) for x in dependencies)):
            raise ValueError(f"{function} requires {dep_count} existing integer handles")
        allowed = VALUE_FUNCTIONS if function.startswith("Verify") else ENTITY_FUNCTIONS
        if any(self.handles[x].function not in allowed for x in dependencies):
            raise ValueError("incompatible_dependency_type")
        if function.startswith("QFilter") and self.handles[dependencies[0]].function not in FACT_FUNCTIONS:
            raise ValueError("qualifier_filter_requires_attached_facts")
        if function in {"FilterNum", "FilterYear", "FilterDate", "QFilterNum", "QFilterYear", "QFilterDate", "VerifyNum", "VerifyYear", "VerifyDate"} and inputs[-1] not in COMPARISONS:
            raise ValueError("unsupported_comparison")
        if function == "Relate" and inputs[-1] not in {"forward", "backward"}:
            raise ValueError("unsupported_relation_direction")
        if function == "SelectBetween" and inputs[-1] not in {"greater", "less"}:
            raise ValueError("unsupported_comparison")
        if function == "SelectAmong" and inputs[-1] not in {"largest", "smallest"}:
            raise ValueError("unsupported_selection")

    def _observation(self, handle: int) -> dict:
        item = self.handles[handle]
        result = {"ok": True, "handle": handle}
        if item.function in ENTITY_FUNCTIONS:
            ids, facts = item.value
            engine = self.executor.engine
            result.update(type="entities_with_facts" if facts is not None else "entities",
                          count=len(ids),
                          sample=[{"id": x, "name": (engine.entities.get(x) or engine.concepts.get(x))["name"]}
                                  for x in sorted(set(ids))[:3]])
            # Bounded schema observations derived ONLY from the actual result.
            # Do not scan all FindAll entities or select fields using gold labels.
            if len(ids) <= 8:
                attrs, rels = set(), set()
                for eid in ids:
                    entity = engine.entities.get(eid) or engine.concepts.get(eid)
                    attrs.update(a["key"] for a in entity.get("attributes", []))
                    rels.update((r["predicate"], r["direction"]) for r in entity.get("relations", []))
                result["attributes"] = sorted(attrs)[:16]
                result["relations"] = [list(x) for x in sorted(rels)[:16]]
            if facts is not None:
                result["qualifiers"] = sorted({k for f in facts for k in f.get("qualifiers", {})})[:16]
        else:
            result.update(type="value" if item.function in VALUE_FUNCTIONS else "answer",
                          value=str(item.value)[:512] if item.value is not None else None)
        return result

    def step(self, function: str, inputs: list[str], dependencies: list[int]) -> dict:
        """Execute one KoPL function and retain its result as an integer handle.

        Args:
            function: KoPL function name from the instruction.
            inputs: Literal string arguments, excluding dependency handles.
            dependencies: Handles returned by previous successful calls.

        Returns:
            Result type, handle, bounded observation, or an execution error.
        """
        try:
            self._begin()
            self._check(function, inputs, dependencies)
            values = [self.handles[x].value for x in dependencies]
            value = getattr(self.executor.engine, function)(values, inputs)
            handle = len(self.handles)
            self.handles.append(Handle(function, value, list(inputs), list(dependencies)))
            try:
                result = self._observation(handle)
            except Exception:
                self.handles.pop()
                raise
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:240]}
        self.events.append({"tool": "step", "arguments": {"function": function,
                            "inputs": inputs, "dependencies": dependencies}, "observation": result})
        return result

    def finish(self, answer_handle: int) -> dict:
        """Finish the episode using a previously computed answer result.

        Args:
            answer_handle: Integer handle of an answer-producing KoPL function.

        Returns:
            The computed answer and termination status, or an error.
        """
        try:
            self._begin()
            if type(answer_handle) is not int or not 0 <= answer_handle < len(self.handles):
                raise ValueError("unknown_answer_handle")
            item = self.handles[answer_handle]
            if item.function not in TERMINALS:
                raise ValueError("finish_requires_answer_function")
            raw = item.value
            answers = [] if raw is None else ([str(x) for x in raw] if isinstance(raw, list) else [str(raw)])
            self.prediction = answers[0] if answers else "None"
            self.selected, self.done = answer_handle, True
            result = {"ok": True, "done": True, "answer": self.prediction}
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:240]}
        self.events.append({"tool": "finish", "arguments": {"answer_handle": answer_handle}, "observation": result})
        return result


def gold_trajectory(executor: KoPLExecutor, row: dict) -> dict:
    episode = Episode(executor)
    messages = prompt_messages(row["question"])
    for action in validate_program(row["program"]):
        observation = episode.step(**action)
        if not observation["ok"]:
            raise ValueError(f"Gold step failed for {row['id']}: {observation}")
        messages.extend([call_message("step", action), {"role": "tool", "content": compact(observation)}])
    arguments = {"answer_handle": len(episode.handles) - 1}
    observation = episode.finish(**arguments)
    messages.extend([call_message("finish", arguments), {"role": "tool", "content": compact(observation)},
                     {"role": "assistant", "content": "Done."}])
    return {"id": row["id"], "messages": messages,
            "supervise": [m["role"] == "assistant" for m in messages],
            "prediction": episode.prediction,
            "correct": compare_answers(row["answer"], episode.prediction),
            "calls": episode.calls, "source": "gold_success"}
