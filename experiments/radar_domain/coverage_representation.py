"""Lossless JSON views of an already selected, ordered reading list E.

This module has no file, question, reference, retrieval or model interface.
Flat records are self-contained. Bound records inherit the unchanged fields
subjects -> component -> attribute -> conditions_all from enclosing groups.
Only consecutive equal prefixes share a group; every other field stays in the
leaf record. Lists retain order and duplicate multiplicity. JSON object member
order is not semantic, but no object keys are sorted or renamed when encoding.

These codecs validate information preservation, not source-reading semantics.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math


GROUP_FIELDS = ("subjects", "component", "attribute", "conditions_all")


def _check_json(value, path="records", active=None):
    """Reject Python values that JSON would silently coerce or cannot preserve."""
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number at {path}")
        return
    if kind not in (list, dict):
        raise ValueError(f"Not a JSON value at {path}: {kind.__name__}")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError(f"Circular JSON value at {path}")
    active.add(id(value))
    try:
        if kind is list:
            for index, child in enumerate(value):
                _check_json(child, f"{path}[{index}]", active)
        else:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError(f"Non-string JSON object key at {path}")
                _check_json(child, f"{path}[{key!r}]", active)
    finally:
        active.remove(id(value))


def _validate_records(records):
    if type(records) is not list:
        raise ValueError("E must be a list of reading objects")
    _check_json(records)
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ValueError(f"Reading at records[{index}] must be an object")
        missing = [field for field in GROUP_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Missing reading fields at records[{index}]: {missing}")


def _difference(left, right, path="records"):
    """Return the first typed JSON difference; bool/int/float are distinct."""
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if type(left) is dict:
        if left.keys() != right.keys():
            return f"{path}: object fields differ"
        for key in left:
            difference = _difference(left[key], right[key], f"{path}[{key!r}]")
            if difference:
                return difference
    elif type(left) is list:
        if len(left) != len(right):
            return f"{path}: list lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif type(left) is float:
        # Preserve signed zero too; all non-finite values are rejected earlier.
        if left.hex() != right.hex():
            return f"{path}: float values differ"
    elif left != right:
        return f"{path}: values differ"
    return None


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _load(encoded):
    if type(encoded) is not str:
        raise ValueError("Encoded representation must be a JSON string")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON object field: {key}")
            obj[key] = value
        return obj

    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON number: {value}")

    decoded = json.loads(encoded, object_pairs_hook=unique_object,
                         parse_constant=reject_constant)
    # JSON such as 1e999 parses as infinity without invoking parse_constant.
    _check_json(decoded, "representation")
    return decoded


def encode_flat(records):
    """Encode E as a compact JSON array of complete self-contained records."""
    _validate_records(records)
    return _dump(records)


def decode_flat(encoded):
    """Decode complete records, rejecting coercions and ambiguous object keys."""
    records = _load(encoded)
    _validate_records(records)
    return records


def _groups(records, depth):
    field = GROUP_FIELDS[depth]
    result, start = [], 0
    while start < len(records):
        end = start + 1
        while end < len(records) and _difference(records[start][field], records[end][field]) is None:
            end += 1
        run = records[start:end]
        node = {field: records[start][field]}
        if depth + 1 < len(GROUP_FIELDS):
            node["groups"] = _groups(run, depth + 1)
        else:
            node["records"] = [{key: value for key, value in record.items()
                                if key not in GROUP_FIELDS} for record in run]
        result.append(node)
        start = end
    return result


def encode_bound(records):
    """Group only adjacent equal prefixes; retain all remaining fields in leaves.

    The root is an array. Each of its nodes has subjects and groups; successive
    groups have component, attribute, then conditions_all. Conditions nodes
    contain records instead of groups. Shared subjects remain one ordered list.
    No field is inferred: all four grouping fields must exist, even if null.
    """
    _validate_records(records)
    return _dump(_groups(records, 0))


def decode_bound(encoded):
    """Restore every complete record in traversal order without shared aliases."""
    root = _load(encoded)
    if type(root) is not list:
        raise ValueError("Bound representation root must be an array")
    records = []

    def visit(nodes, depth, inherited):
        field = GROUP_FIELDS[depth]
        child_key = "records" if depth + 1 == len(GROUP_FIELDS) else "groups"
        for node in nodes:
            if type(node) is not dict or set(node) != {field, child_key}:
                raise ValueError(f"Invalid bound group fields at depth {depth}")
            children = node[child_key]
            # Empty internal branches could introduce unsupported group fields
            # that disappear on decoding; the only empty encoding is root [].
            if type(children) is not list or not children:
                raise ValueError("Bound groups must contain a nonempty child array")
            context = {**inherited, field: node[field]}
            if child_key == "groups":
                visit(children, depth + 1, context)
            else:
                for payload in children:
                    if type(payload) is not dict or any(k in payload for k in GROUP_FIELDS):
                        raise ValueError("Bound leaf must be an object without inherited fields")
                    records.append(deepcopy({**context, **payload}))

    visit(root, 0, {})
    _validate_records(records)
    return records


def validate_equal_information(records, flat=None, bound=None):
    """Require decode(flat) == decode(bound) == E with strict recursive types.

    Optional serialized arguments enable checking stored or externally produced
    representations, not merely testing the encoders against their own output.
    Raises ValueError on any missing/extra field, changed binding, reordered list
    or multiplicity/type difference. Returns only sizes for downstream preflight;
    byte lengths are not model token counts and impose no truncation policy.
    """
    _validate_records(records)
    flat = encode_flat(records) if flat is None else flat
    bound = encode_bound(records) if bound is None else bound
    for name, decoded in (("flat", decode_flat(flat)), ("bound", decode_bound(bound))):
        difference = _difference(records, decoded)
        if difference:
            raise ValueError(f"{name} representation differs at {difference}")
    return {"record_count": len(records), "flat_utf8_bytes": len(flat.encode("utf-8")),
            "bound_utf8_bytes": len(bound.encode("utf-8"))}
