"""Source-bound typed views of all 502 accepted radar readings.

This builder reads source packets, source chunks, and a source-review manifest.
It never reads task answers, reference answers, model outputs, or task programs.
Unparsed values remain typed text and cannot participate in numeric operations.
Run: python -m experiments.radar_domain.typed_sources --write / --check
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "data/radar_domain_recovery_v2/typed_sources_20261107"
PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_typed_sources_20261107"
PACKET_ROOT = ROOT / "data/radar_sources_v2/curation_v1"
PACKET_NAMES = ("weather.v2.json", "marine.v2.json", "raymarine.draft.json", "metek.v2.json")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
UNIT_ALIASES = {
    "Watts": "W", "Kg": "kg", "degrees": "°", "ft.": "ft", "meters": "m",
    "seconds": "s", "min.": "min", "µs": "μs", "usec": "μs", "RPM": "rpm",
    "% RH": "%RH", "VDC": "V DC", "Vdc": "V DC", "V dc": "V DC",
    "VAC": "V AC", "bits": "bit", "lbs": "lb", "lb.": "lb", "in.": "in",
    "Knots": "kn", "kts": "kn", "dBz": "dBZ",
    "inch": "in", "ch": "channel", "µA": "μA", "uA": "μA", "µs": "μs",
}
SIMPLE_UNITS = set(("° W dB m A kW dBi % rpm ft mm hPa kg km dBm MHz kn mW Hz bit years "
                    "bins products channel GHz kHz ms s dBZ targets count pulses pts port pixels "
                    "ns μs °C °F %RH cm in lb NM min mA μA kVA g oz cycles sensors ppm").split()) | {
                        "m/s", "km/h", "mph", "°/s²", "range bins", "V DC", "V AC", "V", "% of full scale"}
ROLE_DIMENSIONS = {"quantity_role", "quantity role", "value role", "power_consumption_role", "current role"}
NUMERIC_CONDITION_UNITS = {
    "antenna_diameter": "m", "antenna length": "ft", "antenna_size_designation": "ft",
    "distance": "km", "supply_voltage": "V", "range_scale": "NM", "pulse_width": "μs",
    "NEBW": "MHz", "ambient temperature": "°C", "range gates": "count",
    "specified target count": "targets", "z": "m", "Δz": "m", "Δt": "s",
    "measuring distance": "m", "range scale": "NM", "target range": "NM",
    "temperature": "°C", "integration time": "s", "measurement time": "s",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_numbers(text, record_id=""):
    text = text.replace("−", "-")
    # Ellipses delimit intervals. They are never decimal punctuation.
    text = re.sub(r"\.{2,}|…", " to ", text)
    # An en dash is a sign at an initial/open-parenthesis numeric position, but
    # remains a range separator between endpoints (e.g. 0–100).
    text = re.sub(r"(^|[(\[])\s*–\s*(?=\d)", r"\1-", text)
    text = re.sub(r"\bto\s+–\s*(?=\d)", "to -", text)
    # These locale cases were explicitly checked in the numeric source review.
    if record_id in {"MRR-0018", "MRR-0020"}:
        text = text.replace(",", ".")
    if record_id == "WX-0064":
        text = text.replace("15,000", "15000")
    # Do not consume a hyphen between numeric endpoints as a unary minus.
    text = re.sub(r"(?<=\d)\s*-(?=\s*\d)", " to ", text)
    text = re.sub(r"(^|[\s<(])([+-])\s+(?=\d)", r"\1\2", text)
    return text


def numbers(text, record_id=""):
    return [float(Decimal(x)) for x in NUMBER.findall(normalize_numbers(text, record_id))]


def unit(raw, record=None):
    if raw is None:
        return None
    if raw == "Nm" and record and record["family_id"] == "raymarine_cyclone":
        return "NM"  # Source range-scale heading; never reinterpret torque Nm.
    result = UNIT_ALIASES.get(raw, raw)
    return result if result in SIMPLE_UNITS else None


def number(value, unit_name, role="unspecified", **extra):
    return {"kind": "number", "value": value, "unit": unit_name, "quantity_role": role, **extra}


def bound(value, operator, unit_name, role="unspecified", **extra):
    assert operator in {"<", "<=", ">", ">="}
    return {"kind": "bound", "operator": operator, "value": value, "unit": unit_name,
            "quantity_role": role, **extra}


def interval(lower, upper, unit_name, role="unspecified", **extra):
    if lower > upper:
        raise ValueError("Reversed source interval")
    return {"kind": "range", "lower": lower, "upper": upper, "lower_inclusive": True,
            "upper_inclusive": True, "unit": unit_name, "quantity_role": role,
            "endpoint_policy": "ordinary printed specification endpoints; not a guarantee of all settings or joint combinations",
            **extra}


def text_value(text, reason, **extra):
    return {"kind": "text", "text": text, "reason": reason, "numeric_operations_supported": False, **extra}


def role_normalized(raw):
    low = raw.lower().strip()
    if low.startswith(("typ", "typical")):
        return "typical"
    if low.startswith(("max", "maximum")):
        return "maximum"
    if low.startswith(("min", "minimum")):
        return "minimum"
    if low in {"peak", "peak power"}:
        return "peak"
    return {"nominal": "nominal", "recommended": "recommended", "average": "average",
            "standby": "standby", "rms": "rms", "continuous": "continuous"}.get(low, "unspecified")


def quantity_role(record):
    for cond in record["conditions_all"]:
        if cond["dimension"] in ROLE_DIMENSIONS:
            result = role_normalized(cond["value"])
            if result != "unspecified":
                return result
    raw = record["value"]["text"].lower()
    attr = record["attribute"].lower()
    form = record["value"]["form"]
    if (form != "range" and re.search(r"\bmax(?:imum)?\.?\b", raw)) or attr.startswith(("maximum", "max.")):
        return "maximum"
    minimum_word = re.search(r"\bminimum\b", raw) or (form == "lower_bound" and re.search(r"\bmin\.", raw))
    if minimum_word or attr.startswith(("minimum", "min.")):
        return "minimum"
    if attr.startswith("nominal"):
        return "nominal"
    if re.search(r"\bnominal\b", raw):
        return "nominal"
    if re.search(r"\btyp(?:ical)?\.?\b", raw):
        return "typical"
    if "peak_power" in attr or "peak power" in attr:
        return "peak"
    return "unspecified"


def comparator(record, role):
    raw = record["value"]["text"].lower()
    if "≤" in raw or "<=" in raw:
        return "<="
    if "≥" in raw or ">=" in raw:
        return ">="
    if "<" in raw or "less than" in raw:
        return "<"
    if ">" in raw or "greater than" in raw:
        return ">"
    if record["value"]["form"] == "upper_bound":
        return "<="
    if record["value"]["form"] == "lower_bound":
        return ">="
    if role == "maximum":
        return "<="
    # A named minimum quantity in a scalar record is not automatically a domain bound.
    return None


def paired_power(text):
    rows = []
    for branch in text.removeprefix("H + V:").strip().split(" or "):
        m = re.fullmatch(r"\s*([\d.]+)\s*kW\s*\+\s*([\d.]+)\s*kW\s*", branch)
        if not m:
            return None
        rows.append({"peak_power_H": number(float(m[1]), "kW", "peak"),
                     "peak_power_V": number(float(m[2]), "kW", "peak"),
                     "configuration_label": branch.strip()})
    return {"kind": "correspondence", "rows": rows, "pairing": "printed_H_V_configurations",
            "independent_member_cartesian_product_supported": False}


def tolerance_value(values, unit_name, value_text):
    if len(values) not in {1, 2} or "±" not in value_text:
        return None
    return {"kind": "tolerance", "center": number(values[0], unit_name) if len(values) == 2 else None,
            "magnitude": number(abs(values[-1]), unit_name),
            "operator": "<" if "<" in value_text and "<=" not in value_text else "<=",
            "unit": unit_name, "numeric_operations_supported": False,
            "comparison_scope": "absolute_deviation_from_source_reference",
            "reference_center_inferred": False}


def approximate_value(values, unit_name, role="unspecified"):
    if len(values) == 1:
        nominal = number(values[0], unit_name, role)
    elif len(values) == 2 and values[0] <= values[1]:
        nominal = interval(values[0], values[1], unit_name, role)
    else:
        return None
    return {"kind": "approximate", "nominal": nominal, "unit": unit_name,
            "quantity_role": role, "numeric_operations_supported": False,
            "reason": "source_explicit_approximation_not_an_exact_domain"}


def printed_unit_views(record, role):
    """Parse explicit, compatible unit representations without converting values.

    A source such as '15 m (49 ft 3 in)' is intentionally refused: the latter
    is a compound length, not two independently equivalent representations.
    """
    raw = record["value"]
    raw_units = [x.strip() for x in (raw["unit_raw"] or "").split(";")]
    units = [unit(x, record) for x in raw_units]
    if len(raw_units) < 2 or any(x is None for x in units):
        return None
    families = [set(x.split()) for x in ["m km cm mm ft in", "kg g lb oz", "m/s km/h mph kn",
                                        "W kW mW", "s ms μs ns min", "Hz kHz MHz GHz", "°C °F"]]
    if not any(set(units) <= family for family in families):
        return None
    if len(units) > 2 and ";" not in raw["text"]:
        return None
    found = []
    for i, (spelling, canonical) in enumerate(zip(raw_units, units)):
        alternatives = {spelling.rstrip("."), canonical}
        patterns = []
        for name in alternatives:
            pattern = re.escape(name).replace(r"\ ", r"\s*")
            pattern = pattern.replace("°C", r"°\s*C").replace("°F", r"°\s*F")
            patterns.append(pattern)
        regex = re.compile(r"(?<![A-Za-z])(?:" + "|".join(sorted(patterns, key=len, reverse=True)) + r")(?![A-Za-z/])")
        found += [(m.start(), m.end(), i) for m in regex.finditer(raw["text"])]
    found.sort()
    groups, cursor = [[] for _ in units], 0
    for start, end, i in found:
        if start < cursor:
            continue
        groups[i].extend(numbers(raw["text"][cursor:start], record["record_id"]))
        cursor = end
    if any(not values for values in groups) or numbers(raw["text"][cursor:]):
        return None
    cells = []
    for values, un in zip(groups, units):
        form = raw["form"]
        if form in {"scalar", "upper_bound", "lower_bound", "correspondence"} and len(values) == 1:
            op = comparator(record, role)
            cell = bound(values[0], op, un, role) if op else number(values[0], un, role)
        elif form == "range" and len(values) == 2 and values[0] <= values[1]:
            cell = interval(values[0], values[1], un, role)
        elif form == "tolerance":
            cell = tolerance_value(values, un, raw["text"])
        elif form == "approximate":
            cell = approximate_value(values, un, role)
        else:
            return None
        if cell is None:
            return None
        cells.append(cell)
    names = ["primary", "equivalent_printed"] + [f"equivalent_printed_{i}" for i in range(2, len(cells))]
    return {"kind": "correspondence", "rows": [dict(zip(names, cells))],
            "pairing": "same_quantity_in_printed_units", "conversion_inferred": False}


def condition_value(cond):
    dim, raw = cond["dimension"], cond["value"]
    if dim == "transmitter_configuration":
        result = paired_power(raw)
        if result:
            return result
    if dim == "parameter combinations":
        return text_value(raw, "source_combination_restriction", constraint_code="not_all_parameter_combinations_supported",
                          joint_cartesian_product_supported=False)
    if dim in ROLE_DIMENSIONS:
        return text_value(raw, "categorical_quantity_role", normalized=role_normalized(raw))
    if dim in NUMERIC_CONDITION_UNITS:
        target_unit = NUMERIC_CONDITION_UNITS[dim]
        if dim in {"range_scale", "range scale"}:
            m = re.fullmatch(r"(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+\.\d+)\s*N[mM]", raw)
            if m:
                value = sum(Fraction(x) for x in m[1].split())
                return number(float(value), "NM", source_lexeme=m[1], exact_fraction=str(value))
        has_interval = bool(re.search(r"\.{2,}|…|\bto\b|\bbetween\b|(?<=\d)\s*[–-]\s*(?=\d)", raw))
        if has_interval:
            units = target_unit
            if target_unit == "m" and re.search(r"\bft\.?\b", raw):
                units = "m; ft"
            if target_unit == "°C" and re.search(r"°\s*F", raw):
                units = "°C; °F"
            temporary = {"record_id": "__typed_condition__", "family_id": "condition", "attribute": dim,
                "value": {"form": "range", "text": raw, "unit_raw": units}, "conditions_all": []}
            return parse_value(temporary)
        vals = numbers(raw)
        # The dimension registry is source-domain-specific: model IDs never enter here.
        if len(vals) == 1 and not re.search(r"[<>±]|or|to|and", raw):
            return number(vals[0], target_unit)
    return text_value(raw, "categorical_or_unparsed_condition_retained")


def parse_value(record):
    raw = record["value"]; rid = record["record_id"]
    form, value_text, raw_unit = raw["form"], raw["text"], raw["unit_raw"]
    role = quantity_role(record)
    simple_unit = unit(raw_unit, record)
    vals = numbers(value_text, rid)
    result = None

    if form == "text":
        result = text_value(value_text, "categorical_source_value_not_numeric")
    elif raw_unit and ";" in raw_unit and (view := printed_unit_views(record, role)) is not None:
        result = view
    elif form == "tolerance" and simple_unit:
        result = tolerance_value(vals, simple_unit, value_text)
    elif form == "approximate" and simple_unit:
        result = approximate_value(vals, simple_unit, role)
    elif form == "range" and simple_unit and len(vals) == 1 and "±" in value_text:
        result = interval(-abs(vals[0]), abs(vals[0]), simple_unit, role,
                          symmetric_range_explicit_in_source=True)
    elif record["attribute"] == "peak_power" and value_text.startswith("H + V:"):
        result = paired_power(value_text)
    elif record["attribute"] == "peak_power" and value_text.startswith("per channel:"):
        m = re.fullmatch(r"per channel: ([\d.]+) Watts / total: ([\d.]+) Watts", value_text)
        if m:
            result = {"kind": "correspondence", "rows": [{"per_channel": number(float(m[1]), "W", "peak"),
                      "total": number(float(m[2]), "W", "peak")}], "pairing": "printed_per_channel_and_total",
                      "computed_total": False}
    elif record["attribute"] == "supply voltage and maximum current":
        rows = []
        for branch in value_text.split("; "):
            m = re.fullmatch(r"([\d.]+) VDC nominal → ([\d.]+)–([\d.]+) V allowable, ([\d.]+) A max\.", branch)
            if not m:
                rows = []; break
            rows.append({"nominal_supply": number(float(m[1]), "V DC", "nominal"),
                         "allowable_supply": interval(float(m[2]), float(m[3]), "V DC"),
                         "maximum_supply_current": bound(float(m[4]), "<=", "A", "maximum")})
        if rows:
            result = {"kind": "correspondence", "rows": rows, "pairing": "printed_supply_branches",
                      "independent_member_cartesian_product_supported": False}
    elif record["attribute"] == "pulse_width_and_prf_by_range_scale":
        m = re.fullmatch(r"([\d.]+) Pulse width \(ns\) ([\d.]+) PRF \(kHz\)", value_text)
        if m:
            result = {"kind": "correspondence", "rows": [{"pulse_width": number(float(m[1]), "ns"),
                      "prf": number(float(m[2]), "kHz")}], "pairing": "same_printed_range_scale_row",
                      "independent_member_cartesian_product_supported": False}
    elif form == "correspondence" and (m := re.fullmatch(
            r"Pulse length ([\d.]+) [μµ]s; PRR approximately ([\d.]+) Hz", value_text)):
        result = {"kind": "correspondence", "rows": [{"pulse_width": number(float(m[1]), "μs"),
                  "prf": approximate_value([float(m[2])], "Hz")}],
                  "pairing": "same_printed_pulse_length_and_approximate_repetition_rate",
                  "independent_member_cartesian_product_supported": False}
    elif rid in {"WX-0114", "WX-0115", "WX-0117"}:
        if len(vals) == 2:
            result = {"kind": "correspondence", "rows": [{"primary": number(vals[0], "m"),
                      "equivalent_printed": number(vals[1], "ft")}], "pairing": "same_quantity_in_printed_units",
                      "conversion_inferred": False}
    elif form == "scalar" and raw_unit in {"kg; lb.", "Kg; lbs", "mm; in", "cm; in."} and len(vals) == 2:
        units = [unit(x.strip(), record) for x in raw_unit.split(";")]
        result = {"kind": "correspondence", "rows": [{"primary": number(vals[0], units[0], role),
                  "equivalent_printed": number(vals[1], units[1], role)}], "pairing": "same_quantity_in_printed_units",
                  "conversion_inferred": False}
    elif raw_unit is None and rid in {"MRR-0023", "MRR-0045"}:
        if form == "scalar" and len(vals) == 1:
            result = number(vals[0], "count", count_kind="range_gates")
        elif form == "range" and len(vals) == 2:
            result = interval(vals[0], vals[1], "count", count_kind="height_gates", discrete_step=None,
                              every_integer_setting_supported=None)
    elif form in {"scalar", "lower_bound", "upper_bound"} and simple_unit and len(vals) == 1:
        op = comparator(record, role)
        result = bound(vals[0], op, simple_unit, role) if op else number(vals[0], simple_unit, role)
    elif form == "range" and simple_unit and len(vals) == 2:
        result = interval(vals[0], vals[1], simple_unit, role)
    elif form == "options" and simple_unit:
        # Restrict to numeric tokens, their printed unit, and list separators.
        rest = value_text
        for spelling in sorted({raw_unit, simple_unit}, key=len, reverse=True):
            rest = rest.replace(spelling, "")
        rest = NUMBER.sub("", rest)
        if not re.sub(r"\s|,|/|\bor\b|\band\b", "", rest) and vals:
            result = {"kind": "options", "options": [number(v, simple_unit, role) for v in vals],
                      "unit": simple_unit, "continuous_interval_supported": False}
    # No one-number shortcut for approximate values, tolerances, or arbitrary tuples.
    if result is None:
        reason = {"approximate": "approximation_not_exact_numeric_domain",
                  "tolerance": "tolerance_structure_requires_explicit_parser",
                  "correspondence": "unregistered_correspondence_schema",
                  "options": "categorical_or_mixed_options_preserved_without_splitting"}.get(form,
                      "multiple_quantities_or_unregistered_unit_require_explicit_parser")
        result = text_value(value_text, reason)
    result.update({"source_form": form, "source_text": value_text, "unit_raw": raw_unit,
                   "quantity_role": result.get("quantity_role", role)})
    return result


def numeric_leaf_paths(obj, prefix):
    for key, value in obj.items():
        path = f"{prefix}.{key}"
        if key in {"value", "lower", "upper", "operator", "unit", "lower_inclusive", "upper_inclusive",
                   "configuration_label", "constraint_code", "normalized", "quantity_role", "count_kind"} and not isinstance(value, (dict, list)):
            yield path
        elif isinstance(value, dict):
            yield from numeric_leaf_paths(value, path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    yield from numeric_leaf_paths(item, f"{path}[{i}]")


def field_evidence(record, typed_value, conditions):
    """Row/cell provenance for every derived value, endpoint, unit, and condition.

    Source offsets always address the source chunk, not a re-tokenized string.
    Locator and typed field path jointly identify tuple members; no false claim
    of a unique numeric-token span when the printed row repeats the same number.
    """
    citation = record["citation"]
    base = {"source_record_id": record["record_id"], "chunk_id": citation["chunk_id"],
            "text_sha256": citation["text_sha256"], "source_spans": citation["spans"],
            "source_locator": record["locator"], "precision": "source_row_cell_with_member_path",
            "additional_citations": deepcopy(record.get("additional_citations", []))}
    fields = {}
    for path in numeric_leaf_paths(typed_value, "typed_value"):
        fields[path] = {**base, "raw_field": "value", "typed_member_path": path,
                        "interpretation": "preserve named member within the cited row/cell; original source form retained"}
        if path == "typed_value.quantity_role":
            for i, cond in enumerate(record["conditions_all"]):
                if cond["dimension"] in ROLE_DIMENSIONS:
                    fields[path]["raw_field"] = f"conditions_all[{i}]"
                    break
    for i, cond in enumerate(conditions):
        spans = citation["spans"]
        if cond["dimension"] == "parameter combinations":
            spans = [x for x in spans if "Not all combinations" in x["quote"]]
        fields[f"typed_conditions_all[{i}]"] = {**base, "source_spans": spans,
            "raw_field": f"conditions_all[{i}]", "condition_dimension": cond["dimension"],
            "interpretation": "source row/header/footnote or named tuple condition, not an inferred cross-row condition"}
        for path in numeric_leaf_paths(cond["typed_value"], f"typed_conditions_all[{i}].typed_value"):
            fields[path] = {**fields[f"typed_conditions_all[{i}]"], "typed_member_path": path}
    if not fields:
        fields["typed_value.text"] = {**base, "raw_field": "value.text"}
    return fields


def type_record(raw_record):
    """Public source-only adapter. No QA, source subset, or score inputs.

    The caller owns source byte/span validation and may attach source metadata.
    Additional citations remain attached to both the raw record and field evidence.
    """
    required = {"record_id", "family_id", "subjects", "component", "attribute", "value",
                "conditions_all", "citation", "locator"}
    if not required <= set(raw_record):
        raise ValueError(f"Missing raw source fields: {sorted(required - set(raw_record))}")
    record = deepcopy(raw_record)
    tv = parse_value(record)
    conditions = [{"dimension": x["dimension"], "value": x["value"], "raw": x["value"],
                   "typed_value": condition_value(x)} for x in record["conditions_all"]]
    kind = tv["kind"]
    typed = deepcopy(record)
    typed.update({"source_fact_id": record["record_id"], "raw_record": record,
        "raw_record_sha256": hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "typed_value": tv, "typed_conditions_all": conditions,
        "field_evidence": field_evidence(record, tv, conditions),
        "typing_status": "typed_text_retained" if kind == "text" else "structured",
        "typing_capabilities": {"source_display": True, "exact_numeric_compare": kind == "number",
            "explicit_bound_endpoint_compare": kind == "bound", "domain_contains": kind in {"number", "bound", "range", "options"},
            "named_tuple_expansion": kind == "correspondence", "uncertainty_structure": kind in {"tolerance", "approximate"},
            "numeric_fallback_from_text": False},
        "original_source_checked_this_run": False, "reviewer": "AI", "human_gold": False,
        "sample_independence": "not_established_by_source_record_identity",
        "source_evidence_group_id": hashlib.sha256(json.dumps({"chunk_id": record["citation"]["chunk_id"],
            "spans": [(s["start"], s["end"]) for s in record["citation"]["spans"]]}, sort_keys=True).encode()).hexdigest()[:24],
        "shared_subjects_are_query_views_only": len(record["subjects"]) > 1})
    return typed


def validate_citation(record, chunk):
    if record["citation"]["text_sha256"] != chunk["text_sha256"]:
        raise ValueError(f"Chunk hash binding mismatch: {record['record_id']}")
    for span in record["citation"]["spans"]:
        actual = chunk["text"][span["start"]:span["end"]]
        # The accepted snapshot stores display-normalized quotes separately from
        # exact source codepoint spans. Reproduce its whitespace normalization.
        if " ".join(actual.split()) != " ".join(span["quote"].split()):
            raise ValueError(f"Invalid source span: {record['record_id']}")


def build():
    selection_path = ROOT / "artifacts/thesis_direction_review/radar_readings_v1/selection.json"
    selection = json.loads(selection_path.read_text())
    expected_packets = {x["path"]: x["sha256"] for x in selection["accepted_packets"]}
    review_path = LOCAL / "source_review.json"
    review = json.loads(review_path.read_text())
    chunk_path = PACKET_ROOT / "chunks.json"
    chunk_obj = json.loads(chunk_path.read_text())
    chunks = {x["chunk_id"]: x for x in chunk_obj["chunks"]}
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (selection_path, chunk_path, review_path)}
    source_records, records, ledger = [], [], []
    for name in PACKET_NAMES:
        path = PACKET_ROOT / name
        rel = str(path.relative_to(ROOT))
        if sha(path) != expected_packets[rel]:
            raise ValueError(f"Accepted snapshot changed: {rel}")
        inputs[rel] = sha(path)
        packet = json.loads(path.read_text())
        source_records.extend(packet["records"])
        ledger.extend(packet["ledger"])
    if len(source_records) != 502 or len({r["record_id"] for r in source_records}) != 502:
        raise ValueError("Original 502-record snapshot completeness violated")
    # Verify all sources, not only the 65 development-task records.
    for chunk in chunks.values():
        for path_key, hash_key in (("source_path", "source_sha256"), ("derived_path", "derived_sha256")):
            p = ROOT / chunk[path_key]
            if sha(p) != chunk[hash_key]:
                raise ValueError(f"Archived source changed: {p}")
            inputs[chunk[path_key]] = sha(p)
        if hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["text_sha256"]:
            raise ValueError(f"Source chunk text changed: {chunk['chunk_id']}")
    visual_ids = {rid for c in review["source_checks"] for rid in c["checked_record_ids"]}
    if visual_ids != set(review["required_record_ids"]) or len(visual_ids) != 65:
        raise ValueError("Required 65-record original-source coverage mismatch")
    for source_check in review["source_checks"]:
        if sha(ROOT / source_check["source_path"]) != source_check["source_sha256"]:
            raise ValueError("Reviewed original source changed")
        if "render_path" in source_check and sha(ROOT / source_check["render_path"]) != source_check["render_sha256"]:
            raise ValueError("Reviewed PDF render changed")
    for record in source_records:
        chunk = chunks[record["citation"]["chunk_id"]]
        validate_citation(record, chunk)
        typed = type_record(record)
        typed.update({"source": {k: chunk[k] for k in ("source_id", "source_path", "source_sha256", "source_uri", "page_one_based")},
            "original_source_checked_this_run": record["record_id"] in visual_ids or record["record_id"] in review["additional_checked_record_ids"]})
        records.append(typed)
    required_rows = [r for r in records if r["record_id"] in visual_ids]
    unresolved_required = [r["record_id"] for r in required_rows if r["typed_value"]["kind"] == "text"]
    if unresolved_required:
        raise ValueError(f"Required source values remain untyped: {unresolved_required}")
    kinds = Counter(r["typed_value"]["kind"] for r in records)
    bound_rows = [r for r in records if r["typed_value"]["kind"] == "bound"]
    counts = {"source_records": 502, "typed_records": len(records), "source_fact_ids": len({r["source_fact_id"] for r in records}),
              "required_records": 65, "required_records_structured": len(required_rows),
              "typed_value_kinds": dict(sorted(kinds.items())),
              "structured_numeric_or_correspondence_records": sum(kinds[k] for k in {"number", "bound", "range", "options", "correspondence"}),
              "structured_uncertain_records": kinds["tolerance"] + kinds["approximate"],
              "bound_operators": dict(sorted(Counter(r["typed_value"]["operator"] for r in bound_rows).items())),
              "strict_bound_records": sum(r["typed_value"]["operator"] in {"<", ">"} for r in bound_rows),
              "shared_subject_source_records": sum(len(r["subjects"]) > 1 for r in records),
              "original_source_evidence_units_reviewed_this_run": len(review["source_checks"]),
              "ledger_entries_preserved": len(ledger)}
    common = {"schema": "radar_typed_sources_v1", "reviewer": "AI", "human_gold": False,
              "typing_revision": 2,
              "development_only": True, "historical_sources_exposed": True,
              "input_sha256": inputs, "counts": counts}
    table = {**common, "records": records,
             "capabilities": ["source-bound exact numbers", "explicit strict and inclusive bounds", "printed intervals",
                              "numeric discrete options", "registered paired correspondence", "typed numeric and categorical conditions",
                              "explicit tolerance and approximate structures without exact-value substitution"],
             "limitations": ["Typed text is preserved and is not numeric-executable.",
                             "Intervals do not establish continuous device setting availability or joint feasibility.",
                             "Expanding subjects or correspondence rows never creates independent source facts.",
                             "Printed alternative units are retained without forcing their rounded values to be exactly equivalent.",
                             "No gold answer or model output is a builder input."]}
    contract = {"schema": "radar_typing_contract_v1", "reviewer": "AI", "human_gold": False,
        "table_path": str((LOCAL / "typed_records.json").relative_to(ROOT)),
        "record_identity": "Exactly one typed row per original record_id; source_fact_id is unchanged. Subject/tuple expansion creates views only.",
        "value_kinds": {
            "number": ["value", "unit", "quantity_role"],
            "bound": ["operator", "value", "unit", "quantity_role"],
            "range": ["lower", "upper", "lower_inclusive", "upper_inclusive", "unit"],
            "options": ["options (array of typed numbers)", "unit", "continuous_interval_supported=false"],
            "correspondence": ["rows (array of named typed cells)", "pairing"],
            "tolerance": ["center (typed number or null)", "magnitude (typed number)", "operator (absolute deviation)", "numeric_operations_supported=false"],
            "approximate": ["nominal (typed number or range)", "numeric_operations_supported=false"],
            "text": ["text", "reason", "numeric_operations_supported=false"]},
        "bound_operators": ["<", "<=", ">", ">="],
        "quantity_roles": ["unspecified", "maximum", "minimum", "typical", "peak", "nominal", "recommended", "average", "standby", "rms", "continuous"],
        "original_form": "typed_value.source_form and raw_record.value.form preserve the source form even when maximum-role scalar gains a bound query view.",
        "conditions": "typed_conditions_all keeps each dimension/raw/value verbatim alongside typed_value. Unparsed conditions remain typed text.",
        "unit_aliases": UNIT_ALIASES,
        "unit_rules": ["Only explicit unit spelling aliases are applied by the builder; no physical-unit magnitude conversion is inferred.",
                       "Raymarine range-scale Nm normalizes to NM only in that declared family context, never for torque.",
                       "count is a semantic count unit for source-labeled gates, not an invented physical unit.",
                       "Primary and equivalent_printed are two printed representations of one quantity, including source rounding."],
        "correspondence_field_names": {"H/V power": ["peak_power_H", "peak_power_V", "configuration_label"],
            "channel power": ["per_channel", "total"], "supply branch": ["nominal_supply", "allowable_supply", "maximum_supply_current"],
            "range row": ["pulse_width", "prf"], "printed units": ["primary", "equivalent_printed"]},
        "provenance": "Each derived numeric/condition field has a field_evidence entry with original source chunk, codepoint spans, row/cell locator, raw field and typed member path.",
        "evidence_precision": "row/cell with named member path; not a claim of unique numerical-token coordinates within repeated-number table rows",
        "unsupported_inferences": ["text-to-number fallback", "independent Cartesian expansion of paired members", "continuous filling of discrete options",
                                   "joint configuration feasibility from separate marginal domains", "statistical sample independence from expanded subjects"],
        "combination_constraint_code": "not_all_parameter_combinations_supported"}
    contract["public_adapter"] = "experiments.radar_domain.typed_sources.type_record(raw_record)"
    contract["public_adapter_validation_boundary"] = "The caller validates source bytes and citation spans. The adapter preserves raw metadata and additional_citations and declares per-record capabilities."
    summary = {**common, "required_record_ids": sorted(visual_ids), "unstructured_required_record_ids": [],
               "all_original_records_preserved": True, "original_citations_preserved": True,
               "all_source_and_derived_hashes_verified": True, "all_citation_spans_verified": True,
               "source_form_preserved": True, "strict_comparators_preserved": True,
               "quantity_role_normalized_separately_from_source_form": True,
               "model_outputs_read": False, "model_scores_read": False, "gold_answer_data_builder_input": False,
               "original_data_modified": False, "new_GPU_hours": 0, "training_started": False,
               "hard_case_records": {"MRR-0047": "Separated minus sign retained in source-reported quantity",
                   "RAY-0034,RAY-0047": "Maximum quantity role normalized consistently; original scalar form remains unchanged",
                   "WX-0010": "Only the printed paired H/V configuration rows are represented",
                   "marine-0045": "Nominal voltage, allowable interval and maximum current stay within each printed branch",
                   "MRR-0041,MRR-0045,MRR-0046": "Combination-restriction footnote retained; no Cartesian feasibility inference"},
               "typed_text_reason_counts": dict(sorted(Counter(r["typed_value"].get("reason") for r in records if r["typing_status"] == "typed_text_retained").items())),
               "typed_text_record_ids": [r["record_id"] for r in records if r["typing_status"] == "typed_text_retained"],
               "builder": "experiments/radar_domain/typed_sources.py", "builder_sha256": sha(Path(__file__)),
               "pre_extension_snapshot_manifest": "data/radar_domain_recovery_v2/typed_sources_20261107/snapshots/pre_parser_extension_20261107/manifest.json",
               "table_path": str((LOCAL / "typed_records.json").relative_to(ROOT)),
               "table_sha256": hashlib.sha256(json_bytes(table)).hexdigest(),
               "typing_contract_path": str((LOCAL / "typing_contract.json").relative_to(ROOT)),
               "typing_contract_sha256": hashlib.sha256(json_bytes(contract)).hexdigest(),
               "source_review_manifest": str(review_path.relative_to(ROOT)),
               "contains_full_qa": False,
               "semantic_limit": "AI source structuring, not human gold or equipment truth verification. 65 required records received original-source review this run; remaining safe structures use the accepted source snapshot, while unsupported forms remain text."}
    return {LOCAL / "typed_records.json": table,
            LOCAL / "typing_contract.json": contract,
            LOCAL / "source_ledger.json": {"schema": "preserved_source_ledger_v1", "entries": ledger},
            LOCAL / "source_chunks.json": chunk_obj,
            PUBLIC / "summary.json": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    for path, value in outputs.items():
        data = json_bytes(value)
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        elif not path.exists() or path.read_bytes() != data:
            raise ValueError(f"Artifact does not reproduce: {path}")
    print(json.dumps({"status": "written" if args.write else "reproduced", "counts": outputs[LOCAL / "typed_records.json"]["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
