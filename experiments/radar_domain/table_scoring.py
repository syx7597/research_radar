"""Structured source-supported answer scoring for the frozen domain task contract."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from .table_program import compact, digest, UNIT_ALIASES, UNIT_SCALE

AUDIT_CELL_FIELDS = {"source_form", "source_text", "unit_raw", "source_lexeme",
                     "exact_fraction", "endpoint_policy", "review_note", "parse_rule"}
EXTRA_UNITS = {"μs": "us", "µs": "us", "μA": "uA", "µA": "uA", "dBz": "dBZ",
               "Kg": "kg", "ft.": "ft"}
EXTRA_SCALE = {"A": ("A", "1"), "mA": ("A", ".001"), "uA": ("A", ".000001")}


def quantity(value, unit):
    if type(value) not in (int, float, str):
        raise ValueError("Quantity needs a finite number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric literal") from exc
    if not number.is_finite():
        raise ValueError("Nonfinite quantity")
    unit = EXTRA_UNITS.get(unit, UNIT_ALIASES.get(unit, unit))
    dimension, factor = {**UNIT_SCALE, **EXTRA_SCALE}.get(unit, (unit, "1"))
    return str((number * Decimal(factor)).normalize()), dimension


def canonical(value):
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (int, float):
        if not Decimal(str(value)).is_finite():
            raise ValueError("Nonfinite JSON")
        return {"decimal": str(Decimal(str(value)).normalize())}
    if isinstance(value, list):
        return [canonical(x) for x in value]
    if not isinstance(value, dict):
        raise ValueError("Unsupported answer type")
    out = {k: deepcopy(v) for k, v in value.items()}
    kind = out.get("kind")
    if kind in {"number", "bound", "range", "options", "correspondence", "approximate", "tolerance"}:
        for key in AUDIT_CELL_FIELDS:
            out.pop(key, None)
    if kind in {"number", "bound"}:
        if kind == "bound" and out.get("operator") not in {"<", "<=", ">", ">="}:
            raise ValueError("A bound needs its actual comparator")
        number, unit = quantity(out["value"], out.get("unit"))
        out["value"], out["unit"] = {"decimal": number}, unit
    elif kind == "range":
        low, unit = quantity(out["lower"], out.get("unit"))
        high, high_unit = quantity(out["upper"], out.get("unit"))
        if unit != high_unit or Decimal(low) > Decimal(high):
            raise ValueError("Invalid interval")
        if type(out.get("lower_inclusive")) is not bool or type(out.get("upper_inclusive")) is not bool:
            raise ValueError("Interval endpoint semantics required")
        out.update(lower={"decimal": low}, upper={"decimal": high}, unit=unit)
    elif kind in {"approximate", "tolerance", "options"} and "unit" in out:
        unit = EXTRA_UNITS.get(out["unit"], UNIT_ALIASES.get(out["unit"], out["unit"]))
        out["unit"] = {**UNIT_SCALE, **EXTRA_SCALE}.get(unit, (unit, "1"))[0]
    result = {key: canonical(val) for key, val in sorted(out.items())}
    if kind == "options" or out.get("value_type") == "discrete_options":
        result["options"] = sorted(result["options"], key=compact)
    elif kind == "correspondence":
        result["rows"] = sorted(result["rows"], key=compact)
    return result


class CatalogIndex:
    def __init__(self, catalog):
        if isinstance(catalog, (str, Path)):
            catalog = json.loads(Path(catalog).read_text())
        rows = catalog["records"]
        self.by_id = {r["record_id"]: r["raw_record"] for r in rows}
        if len(self.by_id) != len(rows):
            raise ValueError("Duplicate source identity in catalog")
        self.sha256 = digest(sorted(rows, key=lambda r: r["record_id"]))


def canonical_predicate(spec):
    result = deepcopy(spec)
    if type(result.get("value")) in (int, float) and "unit" in result:
        value, unit = quantity(result["value"], result["unit"])
        result["value"], result["unit"] = {"decimal": value}, unit
    return canonical(result)


def canonical_derivations(groups):
    if not isinstance(groups, list):
        raise ValueError("Executed derivations must be a list")
    expressions = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Named derivations required")
        expressions.extend({"column": name, "predicate": canonical_predicate(spec)}
                           for name, spec in group.items())
    return sorted(expressions, key=compact)


def context_signature(context):
    """Ignore conjunction and independent operation order, retain actual criteria."""
    if not isinstance(context, list) or not context:
        raise ValueError("An empty table requires executed selection context")
    clauses = []
    for item in context:
        op = item.get("operator")
        if op == "Scan":
            config = deepcopy(item["config"])
            config.setdefault("explode_subjects", True)
            for key in ("subject", "family_id", "attribute", "component", "condition_absent"):
                if isinstance(config.get(key), list):
                    config[key] = sorted(config[key], key=compact)
            ids = item.get("candidate_record_ids")
            if not isinstance(ids, list):
                raise ValueError("Scan scope records missing")
            clauses.append({"operator": "Scan", "config": canonical(config),
                            "candidate_record_ids": sorted(set(ids))})
        elif op == "Filter":
            predicates = sorted([canonical_predicate(x) for x in item["predicates"]], key=compact)
            if item.get("negate", False):
                clauses.append({"operator": "not_all", "predicates": predicates})
            else:
                clauses.extend({"operator": "predicate", "predicate": x} for x in predicates)
        elif op == "Join":
            clauses.append({"operator": "Join", "keys": sorted(item["keys"])})
        elif op == "Derive":
            clauses.append(canonical({k: v for k, v in item.items() if k != "candidate_record_ids"}))
        else:
            raise ValueError("Unknown selection-context operation")
    return sorted({compact(x): x for x in clauses}.values(), key=compact)


def validate_spec(answer, spec):
    if not isinstance(spec, dict) or set(spec) != {"outputs"} or not isinstance(spec["outputs"], list):
        raise ValueError("A value-free answer format specification is required")
    expected = {}
    for row in spec["outputs"]:
        allowed = {"name", "kind", "columns"} if row.get("kind") == "table" else {"name", "kind"}
        if set(row) != allowed or row["kind"] not in {"table", "assessment"} or row["name"] in expected:
            raise ValueError("Format specification contains invalid or answer-bearing fields")
        if row["kind"] == "table" and (not isinstance(row["columns"], list)
                or any(type(c) is not str for c in row["columns"]) or len(set(row["columns"])) != len(row["columns"])):
            raise ValueError("Invalid declared output columns")
        expected[row["name"]] = row
    if set(answer["outputs"]) != set(expected):
        raise ValueError("Output names differ from the supplied format")
    for name, item in answer["outputs"].items():
        requirement = expected[name]
        if item["kind"] != requirement["kind"]:
            raise ValueError("Answer kind differs")
        if item["kind"] == "table" and set(item["columns"]) != set(requirement["columns"]):
            raise ValueError("Columns differ from the supplied format")


def answer_signature(answer, catalog=None, spec=None):
    if not isinstance(answer, dict) or set(answer) != {"schema", "scope", "outputs", "source_records"}:
        raise ValueError("Complete structured answer required")
    if answer["schema"] != "radar_table_answer_v1" or not isinstance(answer["outputs"], dict) or not answer["outputs"]:
        raise ValueError("Unsupported answer schema or empty output collection")
    if (not isinstance(answer["scope"], dict) or set(answer["scope"]) != {"kind", "catalog_sha256"}
            or answer["scope"]["kind"] != "source_catalog"):
        raise ValueError("The full executed source catalog must be identified")
    by_id = None
    if catalog is not None:
        index = catalog if isinstance(catalog, CatalogIndex) else CatalogIndex(catalog)
        by_id = index.by_id
        if index.sha256 != answer["scope"]["catalog_sha256"]:
            raise ValueError("Source catalog identity differs")
    raw = answer["source_records"]
    if not isinstance(raw, dict):
        raise ValueError("Complete source records required")
    for rid, row in raw.items():
        if not isinstance(row, dict) or row.get("record_id") != rid:
            raise ValueError("Invalid source identity")
        if by_id is not None and (rid not in by_id or compact(row) != compact(by_id[rid])):
            raise ValueError("Source record altered or absent from catalog")
    used = set()
    outputs = {}
    for name, output in answer["outputs"].items():
        if not isinstance(name, str) or not isinstance(output, dict):
            raise ValueError("Invalid named output")
        if output.get("kind") == "table":
            allowed = {"kind", "columns", "rows", "selection_context", "derived_conditions"}
            if set(output) - allowed or not {"kind", "columns", "rows"} <= set(output):
                raise ValueError("Invalid table output")
            columns, rows = output["columns"], output["rows"]
            if (not isinstance(columns, list) or not columns or len(columns) != len(set(columns))
                    or any(type(x) is not str for x in columns) or not isinstance(rows, list)):
                raise ValueError("Declared table columns required, including for empty results")
            normalized_rows = []
            for row in rows:
                if (not isinstance(row, dict) or set(row) != {"values", "evidence"}
                        or set(row["values"]) != set(columns) or set(row["evidence"]) != set(columns)):
                    raise ValueError("Every answer cell needs a named value and its provenance")
                evidence = {}
                for column, ids in row["evidence"].items():
                    if not isinstance(ids, list) or not ids or any(type(i) is not str or i not in raw for i in ids):
                        raise ValueError("Unbound answer-cell evidence")
                    evidence[column] = sorted(set(ids)); used.update(ids)
                normalized_rows.append({"values": canonical(row["values"]), "evidence": evidence})
            result = {"kind": "table", "columns": sorted(columns),
                      "rows": sorted(normalized_rows, key=compact)}
            if not rows:
                result["selection_context"] = context_signature(output.get("selection_context"))
                for ctx in output["selection_context"]:
                    for rid in ctx.get("candidate_record_ids", []):
                        if rid not in raw:
                            raise ValueError("Empty-answer search scope lacks its source records")
                        used.add(rid)
            if "derived_conditions" in output:
                conditions = output["derived_conditions"]
                if not isinstance(conditions, list):
                    raise ValueError("Executed derivation conditions required")
                result["derived_conditions"] = canonical_derivations(conditions)
            outputs[name] = result
        elif output.get("kind") == "assessment":
            if set(output) != {"kind", "result", "support_record_ids"}:
                raise ValueError("Invalid assessment output")
            result = output["result"]
            if not isinstance(result, dict) or result.get("status") not in {
                    "established", "not_established", "contradicted_by_individual_domain"}:
                raise ValueError("Explicit established/unknown/contradicted status required")
            ids = output["support_record_ids"]
            if not isinstance(ids, list) or any(i not in raw for i in ids):
                raise ValueError("Assessment source evidence is unbound")
            used.update(ids)
            outputs[name] = {"kind": "assessment", "result": canonical(result),
                             "support_record_ids": sorted(set(ids))}
        else:
            raise ValueError("Unsupported output type")
    if used != set(raw):
        raise ValueError("Source inventory differs from actually used output/scope evidence")
    if spec is not None:
        validate_spec(answer, spec)
    # Original source bytes/metadata are exact; semantic numeric normalization
    # applies only to computed answer cells, not to archived provenance.
    return compact({"scope": answer["scope"], "outputs": outputs,
                    "source_records": raw})


def answers_equal(left, right, catalog=None):
    if left is None or right is None:
        return False
    try:
        return answer_signature(left, catalog) == answer_signature(right, catalog)
    except (ValueError, TypeError, KeyError):
        return False


def score_answer(prediction, reference, catalog, spec=None):
    # Reference validation errors are dataset errors, never silently model errors.
    expected = answer_signature(reference, catalog, spec)
    if prediction is None:
        return {"correct": False, "well_formed": False, "reason": "no_submitted_answer"}
    try:
        actual = answer_signature(prediction, catalog, spec)
    except (ValueError, TypeError, KeyError) as exc:
        return {"correct": False, "well_formed": False, "reason": str(exc)}
    return {"correct": actual == expected, "well_formed": True,
            "reason": "matched" if actual == expected else "answer_or_evidence_differs"}
