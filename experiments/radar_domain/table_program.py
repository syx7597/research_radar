"""Read-only typed-table queries with explicit append-only handles.

All operations use supplied source facts. The executor has no task IDs, expected
answers or candidate required-record lists. Reference authoring/scoring is offline.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def compact(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(compact(x).encode()).hexdigest()


def path_get(value, path):
    keys = path if isinstance(path, list) else path.split(".")
    for key in keys:
        value = value[key]
    return value


UNIT_ALIASES = {"°": "deg", "degrees": "deg", "degree": "deg", "°C": "degC",
                "C": "degC", "ft.": "ft", "VDC": "V DC", "V dc": "V DC",
                "Watts": "W", "%RH": "% RH"}
UNIT_SCALE = {"W": ("W", "1"), "kW": ("W", "1000"), "mW": ("W", ".001"),
              "m": ("m", "1"), "km": ("m", "1000"), "cm": ("m", ".01"), "mm": ("m", ".001"),
              "Hz": ("Hz", "1"), "kHz": ("Hz", "1000"), "MHz": ("Hz", "1000000"),
              "GHz": ("Hz", "1000000000"), "s": ("s", "1"), "ms": ("s", ".001"),
              "us": ("s", ".000001"), "ns": ("s", ".000000001")}


def number(value, unit=None, bound_endpoint=False):
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind != "number" and not (kind == "bound" and bound_endpoint):
            raise ValueError("Numeric comparison requires a number or explicit bound endpoint")
        unit, value = value.get("unit"), value["value"]
    if type(value) not in (str, int, float):
        raise ValueError("Finite numeric value required")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Finite numeric value required")
    unit = UNIT_ALIASES.get(unit, unit)
    dimension, factor = UNIT_SCALE.get(unit, (unit, "1"))
    return result * Decimal(factor), dimension


def compare(left, op, right, unit=None, bound_endpoint=False):
    if op == "in":
        return any(compare(left, "==", item) for item in right)
    if isinstance(left, dict) and left.get("kind") in {"number", "bound"} or op in {"<", "<=", ">", ">="}:
        a, ua = number(left, bound_endpoint=bound_endpoint)
        b, ub = number(right, unit=unit)
        if ua != ub:
            raise ValueError(f"Incompatible units: {ua!r} and {ub!r}")
    else:
        a, b = compact(left), compact(right)
    operations = {"==": lambda: a == b, "!=": lambda: a != b, "<": lambda: a < b,
                  "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}
    if op not in operations:
        raise ValueError("Unsupported comparison")
    return operations[op]()


def contains(domain, value, unit=None):
    kind = domain.get("kind")
    x, ux = number(value, unit)
    if kind == "range":
        low, ul = number(domain["lower"], domain.get("unit"))
        high, uh = number(domain["upper"], domain.get("unit"))
        if ux != ul or ux != uh:
            raise ValueError("Range containment units differ")
        return (x >= low if domain["lower_inclusive"] else x > low) and (
            x <= high if domain["upper_inclusive"] else x < high)
    if kind == "bound":
        return compare(value if isinstance(value, dict) else {"kind": "number", "value": value, "unit": unit},
                       domain["operator"], domain["value"], domain.get("unit"))
    if kind == "options":
        return any(compare(item, "==", value, unit) for item in domain["options"])
    if kind == "number":
        return compare(domain, "==", value, unit)
    raise ValueError("Containment requires a declared numeric domain")


def predicate(row, spec):
    left = path_get(row["values"], spec["column"])
    if spec["op"] == "contains":
        return contains(left, spec["value"], spec.get("unit"))
    return compare(left, spec["op"], spec["value"], spec.get("unit"),
                   spec.get("value_mode") == "bound_endpoint")


def source_ids(rows, columns=None):
    return sorted({rid for row in rows for key, ids in row["evidence"].items()
                   if columns is None or key in columns for rid in ids})


def table(rows, context, columns=None):
    return {"kind": "table", "rows": rows, "selection_context": deepcopy(context),
            "columns": list(columns) if columns is not None else sorted({k for r in rows for k in r["values"]})}


def distinct(rows):
    indexed = {}
    for row in rows:
        key = compact(row["values"])
        if key not in indexed:
            indexed[key] = deepcopy(row)
        else:
            for col, ids in row["evidence"].items():
                indexed[key]["evidence"][col] = sorted(set(indexed[key]["evidence"].get(col, [])) | set(ids))
    return list(indexed.values())


class TableExecutor:
    """A supplied full typed source catalog; never a question-specific subset."""
    def __init__(self, catalog):
        if isinstance(catalog, (str, Path)):
            catalog = json.loads(Path(catalog).read_text())
        records = catalog["records"]
        self.records = deepcopy(records)
        self.by_id = {r["record_id"]: r for r in self.records}
        if not records or len(self.by_id) != len(records):
            raise ValueError("Complete nonempty unique source catalog required")
        for row in records:
            for key in ("family_id", "subjects", "component", "attribute", "typed_value",
                        "typed_conditions_all", "raw_record"):
                if key not in row:
                    raise ValueError(f"Missing typed source field: {key}")
            if row["raw_record"]["record_id"] != row["record_id"]:
                raise ValueError("Source identity mismatch")
            for key in ("family_id", "subjects", "component", "attribute", "value",
                        "conditions_all", "citation", "locator"):
                if key in row["raw_record"] and row.get(key) != row["raw_record"][key]:
                    raise ValueError("Typed envelope differs from original source: " + key)
        self.scope = {"kind": "source_catalog",
                      "catalog_sha256": digest(sorted(records, key=lambda r: r["record_id"]))}

    def operate(self, function, config, dependencies):
        if function == "Scan":
            if dependencies:
                raise ValueError("Scan has no dependencies")
            allowed = {"family_id", "subject", "component", "attribute", "conditions",
                       "condition_absent", "explode_subjects"}
            if set(config) - allowed:
                raise ValueError("Scan accepts source semantics, not record-ID or answer selection")
            rows = []
            for record in self.records:
                if any(key in config and record[key] not in (config[key] if isinstance(config[key], list) else [config[key]])
                       for key in ("family_id", "component", "attribute")):
                    continue
                wanted = config.get("subject")
                wanted = wanted if isinstance(wanted, list) else [wanted] if wanted is not None else None
                subjects = [s for s in record["subjects"] if wanted is None or s in wanted]
                if not subjects:
                    continue
                raw_conditions, conditions = {}, {}
                for item in record["typed_conditions_all"]:
                    key = item["dimension"]
                    if key in raw_conditions:
                        raise ValueError("Repeated condition dimension requires explicit handling")
                    raw_conditions[key], conditions[key] = item["value"], item["typed_value"]
                if any(raw_conditions.get(k) != v for k, v in config.get("conditions", {}).items()):
                    continue
                if any(k in raw_conditions for k in config.get("condition_absent", [])):
                    continue
                for subject in subjects if config.get("explode_subjects", True) else [None]:
                    values = {key: deepcopy(record[key]) for key in ("family_id", "subjects", "component", "attribute")}
                    values.update(subject=subject, typed_value=deepcopy(record["typed_value"]),
                                  value_text=record["raw_record"]["value"]["text"],
                                  conditions=deepcopy(conditions), raw_conditions=deepcopy(raw_conditions))
                    rows.append({"values": values, "evidence": {key: [record["record_id"]] for key in values}})
            context = [{"operator": "Scan", "config": deepcopy(config),
                        "candidate_record_ids": source_ids(rows)}]
            return table(rows, context, ["family_id", "subjects", "component", "attribute",
                "subject", "typed_value", "value_text", "conditions", "raw_conditions"])
        if function == "Assemble":
            names = config["names"]
            if len(names) != len(dependencies) or len(names) != len(set(names)) or not names:
                raise ValueError("Each answer output needs a unique name and dependency")
            outputs, used = {}, set()
            for name, value in zip(names, dependencies):
                if value["kind"] == "table":
                    item = {"kind": "table", "columns": value["columns"][:], "rows": deepcopy(value["rows"])}
                    derived = [c["columns"] for c in value["selection_context"] if c["operator"] == "Derive"]
                    if derived:
                        item["derived_conditions"] = deepcopy(derived)
                    if not value["rows"]:
                        item["selection_context"] = deepcopy(value["selection_context"])
                        for ctx in value["selection_context"]:
                            used.update(ctx.get("candidate_record_ids", []))
                    used.update(source_ids(value["rows"]))
                elif value["kind"] == "assessment":
                    item = deepcopy(value)
                    used.update(value["support_record_ids"])
                else:
                    raise ValueError("Only tables and computed assessments may enter an answer")
                outputs[name] = item
            answer = {"schema": "radar_table_answer_v1", "scope": deepcopy(self.scope),
                      "outputs": outputs, "source_records": {
                          rid: deepcopy(self.by_id[rid]["raw_record"]) for rid in sorted(used)}}
            return {"kind": "answer", "answer": answer}
        expected = 2 if function == "Join" else 1
        if len(dependencies) != expected or any(d.get("kind") != "table" for d in dependencies):
            raise ValueError(f"{function} requires {expected} table dependencies")
        base = dependencies[0]
        rows = base["rows"]
        context = deepcopy(base["selection_context"])
        if function == "Expand":
            output = []
            for row in rows:
                values = path_get(row["values"], config["path"])
                if not isinstance(values, list):
                    raise ValueError("Expand requires an explicit source list")
                top = config["path"].split(".")[0] if isinstance(config["path"], str) else config["path"][0]
                for value in values:
                    new = deepcopy(row)
                    new["values"][config.get("as", "variant")] = deepcopy(value)
                    new["evidence"][config.get("as", "variant")] = deepcopy(row["evidence"][top])
                    output.append(new)
            return table(output, context, base["columns"] + [config.get("as", "variant")])
        if function == "Project":
            output = []
            for row in rows:
                values, evidence = {}, {}
                for alias, path in config["columns"].items():
                    if not isinstance(path, (str, list)):
                        raise ValueError("Project copies source paths; literal answers are not accepted")
                    values[alias] = deepcopy(path_get(row["values"], path))
                    top = path.split(".")[0] if isinstance(path, str) else path[0]
                    evidence[alias] = deepcopy(row["evidence"][top])
                output.append({"values": values, "evidence": evidence})
            return table(distinct(output) if config.get("distinct", True) else output, context, config["columns"])
        if function == "Join":
            other = dependencies[1]
            keys = config["on"]
            if not isinstance(keys, list) or not keys or any(type(k) is not str for k in keys):
                raise ValueError("Join requires explicit equality keys")
            output = []
            for left in rows:
                for right in other["rows"]:
                    if not all(compare(left["values"][k], "==", right["values"][k]) for k in keys):
                        continue
                    values, evidence = deepcopy(left["values"]), deepcopy(left["evidence"])
                    for key, value in right["values"].items():
                        if key in values and not compare(values[key], "==", value):
                            raise ValueError("Conflicting unjoined columns: " + key)
                        values[key] = deepcopy(value)
                        evidence[key] = sorted(set(evidence.get(key, [])) | set(right["evidence"][key]))
                    output.append({"values": values, "evidence": evidence})
            context.extend(deepcopy(other["selection_context"]))
            context.append({"operator": "Join", "keys": keys,
                            "candidate_record_ids": sorted(set(source_ids(rows)) | set(source_ids(other["rows"])))})
            return table(distinct(output), context, list(dict.fromkeys(base["columns"] + other["columns"])))
        if function == "Filter":
            specs = config["predicates"]
            if not isinstance(specs, list) or not specs:
                raise ValueError("Filter requires actual predicates")
            selected = []
            for row in rows:
                keep = all(predicate(row, spec) for spec in specs)
                if keep != config.get("negate", False):
                    selected.append(deepcopy(row))
            context.append({"operator": "Filter", "predicates": deepcopy(specs),
                            "negate": config.get("negate", False), "candidate_record_ids": source_ids(rows)})
            return table(selected, context, base["columns"])
        if function == "Derive":
            output = deepcopy(rows)
            for row in output:
                for name, spec in config["columns"].items():
                    row["values"][name] = predicate(row, spec)
                    top = spec["column"].split(".")[0] if isinstance(spec["column"], str) else spec["column"][0]
                    row["evidence"][name] = deepcopy(row["evidence"][top])
            context.append({"operator": "Derive", "columns": deepcopy(config["columns"]),
                            "candidate_record_ids": source_ids(rows)})
            return table(output, context, list(dict.fromkeys(base["columns"] + list(config["columns"]))))
        if function == "Assess":
            mode = config["mode"]
            if mode == "same_value":
                column = config["column"]
                if not rows:
                    result = {"status": "not_established", "reason": "no_values"}
                else:
                    same = all(compare(rows[0]["values"][column], "==", r["values"][column]) for r in rows[1:])
                    result = {"status": "established", "all_equal": same, "column": column}
                used = source_ids(rows, [column])
            elif mode == "listed_domain":
                column = config["column"]
                cells = [deepcopy(r["values"][column]) for r in rows]
                if any(c.get("kind") != "number" for c in cells):
                    raise ValueError("Listed-domain assessment requires explicit scalar alternatives")
                unique = {compact(c): c for c in cells}
                result = {"status": "established", "value_type": "discrete_options",
                          "options": list(unique.values()), "continuous_interval_supported": False}
                used = source_ids(rows, [column])
            elif mode == "joint_support":
                name_col, domain_col = config["parameter_column"], config["domain_column"]
                by_parameter = {r["values"][name_col]: r for r in rows}
                if len(by_parameter) != len(rows):
                    raise ValueError("Joint assessment needs one domain per parameter")
                checks = {}
                for name, requested in config["requested"].items():
                    if name not in by_parameter:
                        raise ValueError("Missing parameter domain")
                    checks[name] = contains(by_parameter[name]["values"][domain_col], requested)
                restrictions = []
                for row in rows:
                    for value in row["values"].values():
                        if isinstance(value, dict) and value.get("constraint_code"):
                            restrictions.append(value["constraint_code"])
                result = {"status": "not_established" if all(checks.values()) else "contradicted_by_individual_domain",
                          "requested": deepcopy(config["requested"]), "individual_domain_checks": checks,
                          "source_restrictions": sorted(set(restrictions)),
                          "reason": "Individual parameter domains do not assert a jointly supported tuple"}
                used = source_ids(rows)
            else:
                raise ValueError("Unknown assessment mode")
            return {"kind": "assessment", "result": result, "support_record_ids": used}
        raise ValueError("Unknown table operation")


class TableEpisode:
    def __init__(self, executor, max_calls=24):
        self.executor, self.max_calls = executor, max_calls
        self.handles, self.events = [], []
        self.calls, self.done, self.answer, self.prediction, self.selected = 0, False, None, None, None

    def _begin(self):
        if self.done:
            raise ValueError("episode_finished")
        if self.calls >= self.max_calls:
            self.done = True
            raise ValueError("call_budget_exhausted")
        self.calls += 1

    def step(self, function, inputs, dependencies):
        try:
            self._begin()
            if (not isinstance(function, str) or not isinstance(inputs, list) or len(inputs) != 1
                    or type(inputs[0]) is not str or not isinstance(dependencies, list)
                    or any(type(h) is not int or not 0 <= h < len(self.handles) for h in dependencies)):
                raise ValueError("step requires one JSON-config string and existing handles")
            config = json.loads(inputs[0])
            if not isinstance(config, dict):
                raise ValueError("Operation config must be an object")
            value = self.executor.operate(function, config, [self.handles[h]["value"] for h in dependencies])
            handle = len(self.handles)
            self.handles.append({"function": function, "inputs": deepcopy(inputs),
                                 "dependencies": dependencies[:], "value": value})
            result = {"ok": True, "handle": handle, "type": value["kind"]}
            if value["kind"] == "table":
                result.update(row_count=len(value["rows"]), columns=value["columns"][:],
                              sample=[{"values": compact_cell(r["values"]),
                                       "source_record_ids": source_ids([r])}
                                      for r in value["rows"][:3]],
                              sample_truncated=len(value["rows"]) > 3)
                if function == "Scan":
                    directory = {
                        "attributes": sorted({r["values"]["attribute"] for r in value["rows"]}),
                        "components": sorted({r["values"]["component"] for r in value["rows"]}),
                        "condition_dimensions": sorted({k for r in value["rows"] for k in r["values"]["conditions"]}),
                        "subjects": sorted({s for r in value["rows"] for s in r["values"]["subjects"]})}
                    result["schema_directory"] = {
                        k: {"items": vals[:64], "total": len(vals), "truncated": len(vals) > 64}
                        for k, vals in directory.items()}
            elif value["kind"] == "assessment":
                result["result"] = compact_cell(value["result"])
            else:
                result["outputs"] = {k: v["kind"] for k, v in value["answer"]["outputs"].items()}
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
        self.events.append({"tool": "step", "arguments": {"function": function, "inputs": inputs,
                            "dependencies": dependencies}, "observation": result})
        return result

    def finish(self, answer_handle):
        try:
            self._begin()
            if type(answer_handle) is not int or not 0 <= answer_handle < len(self.handles):
                raise ValueError("unknown_answer_handle")
            value = self.handles[answer_handle]["value"]
            if value["kind"] != "answer":
                raise ValueError("finish requires an Assemble result")
            self.answer = deepcopy(value["answer"])
            self.selected, self.done = answer_handle, True
            self.prediction = compact(self.answer)
            result = {"ok": True, "done": True, "answer_sha256": digest(self.answer),
                      "outputs": {name: {"kind": output["kind"],
                          "rows": len(output["rows"]) if output["kind"] == "table" else None}
                          for name, output in self.answer["outputs"].items()}}
        except (ValueError, TypeError) as exc:
            result = {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
        self.events.append({"tool": "finish", "arguments": {"answer_handle": answer_handle}, "observation": result})
        return result


def action(function, config, dependencies=()):
    return {"name": "step", "arguments": {"function": function, "inputs": [compact(config)],
                                          "dependencies": list(dependencies)}}


def compact_cell(value):
    """Bound observations only; handles and final answer retain original values."""
    if isinstance(value, str):
        return value if len(value) <= 120 else {"text_sample": value[:120],
                                              "characters": len(value), "truncated": True}
    if isinstance(value, list):
        return {"items": [compact_cell(x) for x in value[:3]], "total": len(value),
                "truncated": len(value) > 3}
    if isinstance(value, dict):
        if value.get("kind") in {"number", "bound", "range", "approximate", "tolerance", "text"}:
            keys = {"kind", "value", "unit", "operator", "lower", "upper", "lower_inclusive",
                    "upper_inclusive", "quantity_role", "text", "normalized", "constraint_code",
                    "central_value", "tolerance", "count_kind", "discrete_step"}
            return {k: compact_cell(v) for k, v in value.items() if k in keys}
        if value.get("kind") in {"options", "correspondence"}:
            return {k: compact_cell(v) for k, v in value.items()
                    if k in {"kind", "options", "rows", "unit", "quantity_role", "pairing",
                             "continuous_interval_supported", "independent_member_cartesian_product_supported"}}
        return {k: compact_cell(v) for k, v in value.items()}
    return value


def tool_schemas():
    return [{"type": "function", "function": {"name": "step",
             "description": "Execute a read-only typed-table operation; append one handle only on success.",
             "parameters": {"type": "object", "properties": {
                 "function": {"type": "string", "enum": ["Scan", "Expand", "Project", "Join", "Filter", "Derive", "Assess", "Assemble"]},
                 "inputs": {"type": "array", "items": {"type": "string"},
                            "description": "One JSON configuration string."},
                 "dependencies": {"type": "array", "items": {"type": "integer"}}},
                 "required": ["function", "inputs", "dependencies"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "finish",
             "description": "Submit the complete answer stored by Assemble; return a compact receipt.",
             "parameters": {"type": "object", "properties": {
                 "answer_handle": {"type": "integer"}}, "required": ["answer_handle"], "additionalProperties": False}}}]


def execute_response(episode, text):
    try:
        calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
        if len(calls) != 1:
            raise ValueError("exactly_one_tool_call_required")
        call = json.loads(calls[0])
        if set(call) != {"name", "arguments"} or call["name"] not in {"step", "finish"}:
            raise ValueError("Invalid tool call")
        expected = {"step": {"function", "inputs", "dependencies"}, "finish": {"answer_handle"}}
        if set(call["arguments"]) != expected[call["name"]]:
            raise ValueError("Invalid tool arguments")
    except (ValueError, TypeError, KeyError) as exc:
        try:
            episode._begin()
        except ValueError:
            pass
        result = {"ok": False, "error": "ActionFormatError", "detail": str(exc)}
        episode.events.append({"tool": "invalid", "arguments": {}, "observation": result})
    else:
        result = getattr(episode, call["name"])(**call["arguments"])
    if episode.calls >= episode.max_calls:
        episode.done = True
    return result


def prompt_messages(question):
    return [{"role": "system", "content": (
        "Query the supplied typed source catalog using read-only table operations. "
        "Each step takes function, one JSON configuration string in inputs, and explicit handle dependencies. "
        "Operations: Scan, Expand, Project, Join, Filter, Derive, Assess, Assemble. "
        "Scan config filters family_id, subject, component, attribute, conditions; may set explode_subjects=false. "
        "Expand {path:'typed_value.rows',as:'variant'} unfolds printed paired rows; typed_value.options unfolds declared alternatives. "
        "Project {columns:{name:'source.path'}} copies fields. Join {on:['key']} uses explicit equality keys. "
        "Filter {predicates:[{column:'name',op:'<=',value:4,unit:'kW',value_mode:'bound_endpoint'}],negate:false} uses conjunction. "
        "Derive {columns:{flag:{column:'range',op:'contains',value:-20,unit:'degC'}}} computes flags and keeps criteria. "
        "Assess modes: same_value with column; listed_domain with column; joint_support with parameter_column,domain_column,requested. "
        "Assemble {names:['rows','assessment']} names each corresponding dependency. "
        "Project copies source paths; Filter compares with exact units and explicit bound_endpoint mode. "
        "Use Assemble to name actual output tables/assessments, then finish(answer_handle). "
        "Preserve ranges, discrete alternatives, paired fields, source conditions and evidence. "
        "A missing joint statement is not_established, not false. At most 24 calls.")},
        {"role": "user", "content": question}]


def execute_actions(executor, question, actions, max_calls=24):
    episode = TableEpisode(executor, max_calls)
    episode.messages = prompt_messages(question)
    for item in actions:
        if episode.done:
            raise ValueError("Reference continues after termination")
        text = "<tool_call>\n" + compact(item) + "\n</tool_call>"
        observation = execute_response(episode, text)
        episode.messages.extend([{"role": "assistant", "content": text},
                                 {"role": "tool", "content": compact(observation)}])
        if not observation["ok"]:
            raise ValueError(observation)
    if not episode.done or episode.answer is None:
        raise ValueError("Reference did not submit an answer")
    return episode


def make_reference(executor, ident, question, actions):
    episode = execute_actions(executor, question, actions)
    return {"id": ident, "question": question, "actions": deepcopy(actions), "answer": episode.answer}


class ProgramBuilder:
    def __init__(self):
        self.actions = []

    def add(self, function, config, dependencies=()):
        handle = len(self.actions)
        self.actions.append(action(function, config, dependencies))
        return handle

    def finish(self, outputs):
        handle = self.add("Assemble", {"names": list(outputs)}, list(outputs.values()))
        self.actions.append({"name": "finish", "arguments": {"answer_handle": handle}})
        return self.actions


def program_answer_spec(actions):
    """Infer names/column schema from operations alone, without executing facts."""
    schemas = []
    final = None
    for item in actions:
        if item["name"] == "finish":
            if schemas[item["arguments"]["answer_handle"]]["kind"] != "answer":
                raise ValueError("Final handle is not an answer schema")
            continue
        args = item["arguments"]; name = args["function"]; config = json.loads(args["inputs"][0])
        deps = [schemas[h] for h in args["dependencies"]]
        if name == "Scan":
            spec = {"kind": "table", "columns": ["family_id", "subjects", "component", "attribute",
                    "subject", "typed_value", "value_text", "conditions", "raw_conditions"]}
        elif name == "Expand":
            spec = {"kind": "table", "columns": deps[0]["columns"] + [config.get("as", "variant")]}
        elif name == "Project":
            spec = {"kind": "table", "columns": list(config["columns"])}
        elif name == "Join":
            spec = {"kind": "table", "columns": list(dict.fromkeys(deps[0]["columns"] + deps[1]["columns"]))}
        elif name == "Derive":
            spec = {"kind": "table", "columns": list(dict.fromkeys(deps[0]["columns"] + list(config["columns"])))}
        elif name == "Filter":
            spec = deepcopy(deps[0])
        elif name == "Assess":
            spec = {"kind": "assessment"}
        elif name == "Assemble":
            final = {"outputs": [{"name": label, **deepcopy(dep)} for label, dep in zip(config["names"], deps)]}
            spec = {"kind": "answer"}
        else:
            raise ValueError("Unknown operation schema")
        schemas.append(spec)
    if final is None:
        raise ValueError("No answer assembly")
    return final


def _project_scan(builder, scope, columns, expand=None):
    handle = builder.add("Scan", scope)
    if expand:
        handle = builder.add("Expand", {"path": expand, "as": "variant"}, [handle])
    return builder.add("Project", {"columns": columns}, [handle])


def _join(builder, left, right, *keys):
    return builder.add("Join", {"on": list(keys)}, [left, right])


def _filter(builder, handle, predicates, negate=False):
    return builder.add("Filter", {"predicates": predicates, "negate": negate}, [handle])


def _condition(column, op, value, unit=None, bound=False):
    result = {"column": column, "op": op, "value": value}
    if unit is not None:
        result["unit"] = unit
    if bound:
        result["value_mode"] = "bound_endpoint"
    return result


def development_programs():
    """Human-authored reference query ASTs. No expected answer values are read."""
    programs = {}
    b = ProgramBuilder()
    scope = {"family_id": "vaisala_wrs300", "subject": "WRS300"}
    peak = _project_scan(b, {**scope, "component": "transmitter", "attribute": "peak_power"},
        {"configuration": "variant.configuration_label", "peak_power_H": "variant.peak_power_H",
         "peak_power_V": "variant.peak_power_V"}, "typed_value.rows")
    maximum = _project_scan(b, {**scope, "component": "system", "attribute": "power_consumption",
        "conditions": {"quantity_role": "maximum"}}, {"configuration": "raw_conditions.transmitter_configuration",
        "system_consumption_max": "typed_value"})
    typical = _project_scan(b, {**scope, "component": "system", "attribute": "power_consumption",
        "conditions": {"quantity_role": "typical"}}, {"configuration": "raw_conditions.transmitter_configuration",
        "system_consumption_typical": "typed_value"})
    combined = _join(b, _join(b, peak, maximum, "configuration"), typical, "configuration")
    predicates = [_condition("system_consumption_max", "<=", 4, "kW", True)]
    programs["RDR-DEV-01"] = b.finish({"selected": _filter(b, combined, predicates),
                                      "excluded": _filter(b, combined, predicates, True)})

    b = ProgramBuilder()
    scope = {"family_id": "vaisala_wrs300", "subject": "WRS300",
             "component": ["units in radome", "radar cabinet"]}
    temp = _project_scan(b, {**scope, "attribute": "operating_temperature",
        "condition_absent": ["quantity_role"]}, {"component": "component", "operating_temperature": "typed_value"})
    humidity = _project_scan(b, {**scope, "attribute": "operating_humidity"},
        {"component": "component", "operating_humidity": "typed_value",
         "moisture_state": "raw_conditions.moisture_state"})
    recommended = _project_scan(b, {**scope, "attribute": "operating_temperature",
        "conditions": {"quantity_role": "recommended"}}, {"component": "component",
        "recommended_temperature": "typed_value"})
    programs["RDR-DEV-02"] = b.finish({"rows": _join(b, temp, humidity, "component"),
                                      "recommended": recommended})

    b = ProgramBuilder()
    scope = {"family_id": "eec_ranger_x_band", "subject": "RANGER-X5"}
    diameters = _project_scan(b, {**scope, "component": "antenna", "attribute": "available_diameters"},
        {"antenna_diameter": "variant"}, "typed_value.options")
    gain = _project_scan(b, {**scope, "component": "antenna", "attribute": "gain"},
        {"antenna_diameter": "conditions.antenna_diameter", "gain_min": "typed_value"})
    beam = _project_scan(b, {**scope, "component": "antenna", "attribute": "half_power_beam_width",
        "conditions": {"quantity_role": "typical"}}, {"antenna_diameter": "conditions.antenna_diameter",
        "typical_half_power_beam_width_max": "typed_value"})
    base = _join(b, _join(b, diameters, gain, "antenna_diameter"), beam, "antenna_diameter")
    predicates = [_condition("gain_min", ">=", 40, "dB", True),
                  _condition("typical_half_power_beam_width_max", "<=", 1.3, "deg", True)]
    selected, excluded = _filter(b, base, predicates), _filter(b, base, predicates, True)
    radome = _project_scan(b, {**scope, "component": "radome", "attribute": "required_diameter"},
        {"antenna_diameter": "conditions.antenna_diameter", "required_radome_diameter": "variant.primary"},
        "typed_value.rows")
    selected = _join(b, selected, radome, "antenna_diameter")
    excluded = b.add("Project", {"columns": {"antenna_diameter": "antenna_diameter"}}, [excluded])
    programs["RDR-DEV-03"] = b.finish({"selected": selected, "excluded": excluded})

    b = ProgramBuilder()
    scope = {"family_id": "eec_ranger_x_band", "subject": ["RANGER-X1", "RANGER-X5"]}
    power = _project_scan(b, {**scope, "component": "transmitter", "attribute": "peak_power"},
        {"subject": "subject", "peak_power_per_channel": "variant.per_channel",
         "peak_power_total": "variant.total"}, "typed_value.rows")
    sensitivity = _project_scan(b, {**scope, "component": "system", "attribute": "sensitivity_reflectivity"},
        {"subject": "subject", "sensitivity_reflectivity_raw": "value_text", "distance": "conditions.distance"})
    rows = _join(b, power, sensitivity, "subject")
    assessment = b.add("Assess", {"mode": "same_value", "column": "distance"}, [rows])
    programs["RDR-DEV-04"] = b.finish({"rows": rows,
        "same_distance_sensitivity_comparison_supported": assessment})

    b = ProgramBuilder()
    scope = {"family_id": "furuno_drs_nxt", "component": "antenna unit", "explode_subjects": False}
    length = _project_scan(b, {**scope, "attribute": "antenna length"},
        {"antenna_model": "raw_conditions.antenna model", "length": "typed_value"})
    gain = _project_scan(b, {**scope, "attribute": "antenna gain"},
        {"antenna_model": "raw_conditions.antenna model", "gain": "typed_value"})
    base = _join(b, length, gain, "antenna_model")
    predicates = [_condition("length", "<=", 4, "ft"), _condition("gain", ">=", 28, "dBi")]
    selected, excluded = _filter(b, base, predicates), _filter(b, base, predicates, True)
    width = _project_scan(b, {**scope, "attribute": "horizontal beam width"},
        {"antenna_model": "raw_conditions.antenna model", "horizontal_beam_width": "typed_value"})
    effective = _project_scan(b, {**scope, "attribute": "RezBoost effective beam width",
        "conditions": {"processing": "RezBoost"}}, {"antenna_model": "raw_conditions.antenna model",
        "RezBoost_effective_beam_width": "typed_value"})
    selected = _join(b, _join(b, selected, width, "antenna_model"), effective, "antenna_model")
    excluded = b.add("Project", {"columns": {"antenna_model": "antenna_model"}}, [excluded])
    programs["RDR-DEV-05"] = b.finish({"selected": selected, "excluded": excluded})

    b = ProgramBuilder()
    scope = {"family_id": "furuno_drs_nxt", "subject": ["DRS6A-NXT", "DRS12A-NXT", "DRS25A-NXT"]}
    power = _project_scan(b, {**scope, "component": "radar function", "attribute": "output power"},
        {"subject": "subject", "output_power": "typed_value"})
    supply = _project_scan(b, {**scope, "component": "power supply", "attribute": "supply voltage and maximum current"},
        {"subject": "subject", "nominal_supply": "variant.nominal_supply",
         "allowable_supply": "variant.allowable_supply", "maximum_supply_current": "variant.maximum_supply_current"},
        "typed_value.rows")
    supply = _filter(b, supply, [_condition("nominal_supply", "==", 24, "V DC")])
    base = _join(b, power, supply, "subject")
    predicates = [_condition("output_power", ">=", 100, "W"),
                  _condition("maximum_supply_current", "<=", 5, "A", True)]
    programs["RDR-DEV-06"] = b.finish({"selected": _filter(b, base, predicates),
                                      "excluded": _filter(b, base, predicates, True)})

    b = ProgramBuilder()
    scope = {"family_id": "jrc_jma5200mk2"}
    power = _project_scan(b, {**scope, "component": "scanner", "attribute": "transmitting power"},
        {"subject": "subject", "transmitting_power": "typed_value"})
    width = _project_scan(b, {**scope, "component": "scanner antenna", "attribute": "horizontal beam width",
        "conditions": {"beam-width level": "3db", "plane": "horizontal"}},
        {"subject": "subject", "horizontal_3dB_beam_width": "typed_value"})
    base = _join(b, power, width, "subject")
    selected = _filter(b, base, [_condition("transmitting_power", "==", 25, "kW"),
                                 _condition("horizontal_3dB_beam_width", "<=", 1, "deg")])
    length = _project_scan(b, {**scope, "component": "scanner antenna", "attribute": "antenna length"},
        {"subject": "subject", "antenna_length": "typed_value"})
    speed = _project_scan(b, {**scope, "component": "scanner", "attribute": "rotation speed"},
        {"subject": "subject", "rotation_speed": "typed_value"})
    programs["RDR-DEV-07"] = b.finish({"selected": _join(b, _join(b, selected, length, "subject"), speed, "subject")})

    b = ProgramBuilder()
    base = _project_scan(b, {"family_id": "jrc_jma5200mk2", "attribute": "ambient temperature",
        "component": ["scanner", "processor, display and keyboard"], "explode_subjects": False},
        {"component": "component", "allowed_temperature": "typed_value"})
    assessed = b.add("Derive", {"columns": {"within_stated_range":
        _condition("allowed_temperature", "contains", -20, "degC")}}, [base])
    programs["RDR-DEV-08"] = b.finish({"rows": assessed})

    b = ProgramBuilder()
    scope = {"family_id": "garmin_gmr_fantom", "component": "antenna", "explode_subjects": False}
    weight = _project_scan(b, {**scope, "attribute": "weight"},
        {"antenna_length": "conditions.antenna length", "antenna_weight": "variant.primary"}, "typed_value.rows")
    width = _project_scan(b, {**scope, "attribute": "horizontal beam width",
        "conditions": {"plane": "horizontal"}},
        {"antenna_length": "conditions.antenna length", "horizontal_beam_width": "typed_value"})
    base = _join(b, weight, width, "antenna_length")
    selected = _filter(b, base, [_condition("antenna_weight", "<=", 6, "kg"),
                                _condition("horizontal_beam_width", "<=", 1.5, "deg")])
    programs["RDR-DEV-09"] = b.finish({"candidates": base, "selected": selected})

    b = ProgramBuilder()
    params = _project_scan(b, {"family_id": "metek_mrr", "subject": "MRR-PRO", "component": "radar",
        "attribute": ["number of height gates", "height resolution", "time interval for averaged spectra"]},
        {"parameter": "attribute", "domain": "typed_value", "adjustability": "raw_conditions.adjustability",
         "combination_constraint": "conditions.parameter combinations"})
    requested = {"number of height gates": {"kind": "number", "value": 256, "unit": "count"},
                 "height resolution": {"kind": "number", "value": 10, "unit": "m"},
                 "time interval for averaged spectra": {"kind": "number", "value": 1, "unit": "s"}}
    joint = b.add("Assess", {"mode": "joint_support", "parameter_column": "parameter",
        "domain_column": "domain", "requested": requested}, [params])
    programs["RDR-DEV-10"] = b.finish({"individual_parameters": params, "joint_configuration": joint})

    b = ProgramBuilder()
    base = _project_scan(b, {"family_id": "raymarine_cyclone", "subject": "Cyclone Pro",
        "component": ["inline fuse", "thermal breaker"], "attribute": "current_rating"},
        {"component": "component", "supply_voltage": "conditions.supply_voltage", "rating": "typed_value"})
    selected = _filter(b, base, [_condition("rating", "==", 15, "A")])
    programs["RDR-DEV-11"] = b.finish({"all_combinations": base, "selected_15A": selected})

    b = ProgramBuilder()
    base = b.add("Scan", {"family_id": "raymarine_cyclone", "subject": "Cyclone Pro",
        "component": "transmitter", "attribute": "pulse_width_and_prf_by_range_scale"})
    base = _filter(b, base, [_condition("conditions.range_scale", ">=", 12, "NM"),
                            _condition("conditions.range_scale", "<=", 96, "NM")])
    base = b.add("Expand", {"path": "typed_value.rows", "as": "variant"}, [base])
    base = b.add("Project", {"columns": {"range_scale": "conditions.range_scale",
        "pulse_width": "variant.pulse_width", "PRF": "variant.prf"}}, [base])
    selected = _filter(b, base, [_condition("PRF", "<=", 1.2, "kHz")])
    domain = b.add("Assess", {"mode": "listed_domain", "column": "range_scale"}, [selected])
    programs["RDR-DEV-12"] = b.finish({"selected": selected, "selection_domain": domain})
    return programs


def numeric_in(cell, unit, bound=False):
    """Offline semantic checking helper; preserves bound checks separately."""
    value, dimension = number(cell, bound_endpoint=bound)
    scale, target_dimension = number(1, unit)
    if dimension != target_dimension:
        raise ValueError("Expected answer unit mismatch")
    return float(value / scale)


def range_in(cell, unit):
    if cell.get("kind") != "range" or not cell["lower_inclusive"] or not cell["upper_inclusive"]:
        raise ValueError("Expected closed printed range")
    return [numeric_in({"kind": "number", "value": cell[k], "unit": cell["unit"]}, unit)
            for k in ("lower", "upper")]


def bound_in(cell, unit):
    if cell.get("kind") != "bound":
        raise ValueError("Expected explicit bound")
    return {"op": cell["operator"], "value": numeric_in(cell, unit, True)}


def validate_development_answer(task_id, answer, expected):
    """Independent offline view against the previously authored candidate answer.

    This function is not reachable from TableExecutor/TableEpisode operations.
    It validates all original requested answer fields, including negative/empty/
    unsupported results; it never repairs or supplies a runtime answer.
    """
    outputs = answer["outputs"]
    rows = lambda label: [r["values"] for r in outputs[label]["rows"]]
    if task_id == "RDR-DEV-01":
        result = {}
        for label in ("selected", "excluded"):
            result[label] = []
            for r in rows(label):
                assert r["system_consumption_max"]["kind"] == "bound"
                assert r["system_consumption_max"]["quantity_role"] == "maximum"
                assert r["system_consumption_typical"]["quantity_role"] == "typical"
                result[label].append({"peak_power_H_kW": numeric_in(r["peak_power_H"], "kW"),
                    "peak_power_V_kW": numeric_in(r["peak_power_V"], "kW"),
                    "system_consumption_max_kW": numeric_in(r["system_consumption_max"], "kW", True),
                    "system_consumption_typical_kW": numeric_in(r["system_consumption_typical"], "kW")})
    elif task_id == "RDR-DEV-02":
        recommendations = {r["component"]: range_in(r["recommended_temperature"], "degC") for r in rows("recommended")}
        result = {"rows": []}
        for r in rows("rows"):
            row = {"component": r["component"], "operating_temperature_C": range_in(r["operating_temperature"], "degC"),
                   "operating_humidity_percent_RH": range_in(r["operating_humidity"], "% RH"),
                   "moisture_state": r["moisture_state"]}
            if r["component"] in recommendations:
                row["recommended_temperature_C"] = recommendations[r["component"]]
            result["rows"].append(row)
    elif task_id == "RDR-DEV-03":
        result = {"selected": [{"antenna_diameter_m": numeric_in(r["antenna_diameter"], "m"),
            "gain_min_dB": bound_in(r["gain_min"], "dB"),
            "typical_half_power_beam_width_max_deg": bound_in(r["typical_half_power_beam_width_max"], "deg"),
            "required_radome_diameter_m": numeric_in(r["required_radome_diameter"], "m")} for r in rows("selected")],
            "excluded_antenna_diameters_m": [numeric_in(r["antenna_diameter"], "m") for r in rows("excluded")]}
    elif task_id == "RDR-DEV-04":
        assessment = outputs["same_distance_sensitivity_comparison_supported"]["result"]
        assert assessment["status"] == "established" and type(assessment["all_equal"]) is bool
        result = {"rows": [{"subject": r["subject"], "peak_power_per_channel_W": numeric_in(r["peak_power_per_channel"], "W"),
            "peak_power_total_W": numeric_in(r["peak_power_total"], "W"),
            "sensitivity_reflectivity_raw": r["sensitivity_reflectivity_raw"],
            "distance_km": numeric_in(r["distance"], "km")} for r in rows("rows")],
            "same_distance_sensitivity_comparison_supported": assessment["all_equal"]}
    elif task_id == "RDR-DEV-05":
        result = {"selected": [{"antenna_model": r["antenna_model"], "length_ft": numeric_in(r["length"], "ft"),
            "gain_dBi": numeric_in(r["gain"], "dBi"), "horizontal_beam_width_deg": numeric_in(r["horizontal_beam_width"], "deg"),
            "RezBoost_effective_beam_width_deg": range_in(r["RezBoost_effective_beam_width"], "deg")} for r in rows("selected")],
            "excluded": [r["antenna_model"] for r in rows("excluded")]}
    elif task_id == "RDR-DEV-06":
        result = {"selected": [{"subject": r["subject"], "output_power_W": numeric_in(r["output_power"], "W"),
            "nominal_supply_V_DC": numeric_in(r["nominal_supply"], "V DC"),
            "allowable_supply_V_DC": range_in(r["allowable_supply"], "V DC"),
            "maximum_supply_current_A": numeric_in(r["maximum_supply_current"], "A", True)} for r in rows("selected")],
            "excluded": [{"subject": r["subject"], "output_power_W": numeric_in(r["output_power"], "W"),
            "maximum_supply_current_at_24V_A": numeric_in(r["maximum_supply_current"], "A", True)} for r in rows("excluded")]}
        assert all(numeric_in(r["nominal_supply"], "V DC") == 24 for r in rows("excluded"))
    elif task_id == "RDR-DEV-07":
        result = {"selected": [{"subject": r["subject"], "transmitting_power_kW": numeric_in(r["transmitting_power"], "kW"),
            "antenna_length_ft": numeric_in(r["antenna_length"], "ft"),
            "horizontal_3dB_beam_width_deg": numeric_in(r["horizontal_3dB_beam_width"], "deg"),
            "rotation_speed_rpm": numeric_in(r["rotation_speed"], "rpm")} for r in rows("selected")]}
    elif task_id == "RDR-DEV-08":
        criterion = outputs["rows"]["derived_conditions"][0]["within_stated_range"]
        assert criterion == _condition("allowed_temperature", "contains", -20, "degC")
        result = {"ambient_temperature_C": criterion["value"], "rows": [
            {"component": r["component"], "allowed_temperature_C": range_in(r["allowed_temperature"], "degC"),
             "within_stated_range": r["within_stated_range"]} for r in rows("rows")]}
    elif task_id == "RDR-DEV-09":
        result = {"candidates": [{"antenna_length_ft": numeric_in(r["antenna_length"], "ft"),
            "antenna_weight_kg": numeric_in(r["antenna_weight"], "kg"),
            "horizontal_beam_width_deg": numeric_in(r["horizontal_beam_width"], "deg")} for r in rows("candidates")],
            "selected": rows("selected")}
        assert outputs["selected"]["selection_context"]
    elif task_id == "RDR-DEV-10":
        domains = {r["parameter"]: r["domain"] for r in rows("individual_parameters")}
        interval, gate, height = (domains[k] for k in ("time interval for averaged spectra", "number of height gates", "height resolution"))
        assessment = outputs["joint_configuration"]["result"]
        assert assessment["status"] == "not_established"
        assert "not_all_parameter_combinations_supported" in assessment["source_restrictions"]
        assert all(assessment["individual_domain_checks"].values())
        request = assessment["requested"]
        result = {"individual_parameters": {
            "height_gate_count": {"form": "range", "minimum": range_in(gate, "count")[0], "maximum": range_in(gate, "count")[1]},
            "height_resolution_m": {"form": "range", "minimum": range_in(height, "m")[0], "maximum": range_in(height, "m")[1]},
            "averaged_spectra_interval_s": {"form": "lower_bound", **bound_in(interval, "s")}},
            "requested_joint_configuration": {"height_gate_count": numeric_in(request["number of height gates"], "count"),
                "height_resolution_m": numeric_in(request["height resolution"], "m"),
                "averaged_spectra_interval_s": numeric_in(request["time interval for averaged spectra"], "s")},
            "joint_feasibility": "not_established_by_source"}
    elif task_id == "RDR-DEV-11":
        result = {"all_combinations": [{"supply_voltage_V": numeric_in(r["supply_voltage"], "V"),
            "component": r["component"], "rating_A": numeric_in(r["rating"], "A")} for r in rows("all_combinations")],
            "selected_15A": [{"supply_voltage_V": numeric_in(r["supply_voltage"], "V"),
            "component": r["component"]} for r in rows("selected_15A")]}
        assert all(numeric_in(r["rating"], "A") == 15 for r in rows("selected_15A"))
    elif task_id == "RDR-DEV-12":
        assessment = outputs["selection_domain"]["result"]
        result = {"selected": [{"range_scale_NM": numeric_in(r["range_scale"], "NM"),
            "pulse_width_ns": numeric_in(r["pulse_width"], "ns"), "PRF_kHz": numeric_in(r["PRF"], "kHz")}
            for r in rows("selected")], "selection_value_type": assessment["value_type"],
            "continuous_interval_supported": assessment["continuous_interval_supported"]}
    else:
        raise ValueError("Unknown development validation mapping")
    def unordered(x):
        if type(x) in (int, float):
            return str(Decimal(str(x)).normalize())
        if isinstance(x, list):
            return sorted((unordered(v) for v in x), key=compact)
        if isinstance(x, dict):
            return {k: unordered(v) for k, v in x.items()}
        return x
    if unordered(result) != unordered(expected):
        raise ValueError(f"Complete candidate-answer mismatch: {task_id}: {compact(result)}")
    return result


def main():
    from experiments.agent_feedback.recovery import sha256
    private = Path("data/radar_domain_recovery_v2/programs_dev_20261107")
    public = Path("results/radar_domain/programs_dev_20261107")
    catalog_path = Path("data/radar_domain_recovery_v2/typed_sources_20261107/typed_records.json")
    candidate_path = Path("data/radar_domain_recovery_v1/development_task_candidates.json")
    if private.exists() or public.exists():
        raise FileExistsError("Retain completed development compilation")
    catalog = json.loads(catalog_path.read_text())
    tasks = json.loads(candidate_path.read_text())["tasks"]
    if len(tasks) != 12 or len(catalog["records"]) != 502:
        raise ValueError("Use the full 502-record development catalog and all twelve tasks")
    executor = TableExecutor(catalog)
    programs = development_programs()
    references, traces, comparisons, specs = [], [], [], {}
    for task in tasks:
        ident, question = task["task_id"], task["question_zh"]
        actions = programs[ident]
        spec = program_answer_spec(actions)
        episode = execute_actions(executor, question, actions)
        reference = {"id": ident, "question": question, "actions": actions, "answer": episode.answer}
        projection = validate_development_answer(ident, episode.answer, task["answer_data"])
        for output_spec in spec["outputs"]:
            actual = episode.answer["outputs"][output_spec["name"]]
            assert actual["kind"] == output_spec["kind"]
            if actual["kind"] == "table":
                assert actual["columns"] == output_spec["columns"]
        for rid, raw in episode.answer["source_records"].items():
            assert raw == executor.by_id[rid]["raw_record"]
        references.append(reference); specs[ident] = spec
        traces.append({"id": ident, "messages": episode.messages, "events": episode.events,
                       "calls": episode.calls, "answer_sha256": digest(episode.answer),
                       "development_only": True, "model_generated": False})
        comparisons.append({"id": ident, "all_original_answer_fields_match": True,
                            "derived_view": projection, "expected_view": task["answer_data"]})
    private.mkdir(parents=True); public.mkdir(parents=True)
    for name, rows in (("references.jsonl", references), ("reference_replays.jsonl", traces)):
        with (private/name).open("x") as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    for name, value in (("answer_specs.json", {"schema": "radar_answer_specs_v1", "answer_specs": specs}),
                        ("independent_candidate_comparison.json", comparisons),
                        ("tool_contract.json", {"tools": tool_schemas(), "system_prompt": prompt_messages("")[0]["content"]})):
        (private/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    report = {"schema": "radar_complete_development_programs_v1", "date": "2026-11-07",
        "development_only": True, "human_gold": False, "model_generated": False,
        "complete_original_tasks": 12, "complete_task_replays_passed": len(references),
        "complete_candidate_answer_comparisons_passed": len(comparisons),
        "source_catalog_records": len(catalog["records"]), "max_calls": max(t["calls"] for t in traces),
        "gpu_model_calls": 0, "training_started": False,
        "runtime_reads_answer_data": False, "reference_actions_manually_authored_from_question": True,
        "source_evidence_preserved": True, "all_model_arms_share_operations": True,
        "cases": [{"id": r["id"], "calls": t["calls"], "outputs": {
            k: {"kind": v["kind"], "rows": len(v["rows"])} if v["kind"] == "table" else
            {"kind": v["kind"], "status": v["result"]["status"]} for k,v in r["answer"]["outputs"].items()}}
            for r,t in zip(references,traces)],
        "inputs_sha256": {str(p): sha256(p) for p in (catalog_path,candidate_path,Path(__file__))},
        "outputs_sha256": {str(p): sha256(p) for p in private.iterdir() if p.is_file()},
        "scope": "Full source-grounded development reference programs, not model prediction accuracy or a new held-out benchmark."}
    (public/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"complete_tasks":12,"passed":len(references),"max_calls":report["max_calls"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
