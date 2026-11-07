"""Deterministic synthetic source groups and executor-verified successful references.

All facts in this module's output are invented for tool-use development. They
are not equipment specifications. The builder never reads real-task gold,
holdout tasks, model outputs, or model failures. The reference oracle computes
from generated latent source objects, independently of the typed-table executor.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "data/radar_domain_recovery_v2/synthetic_20261107"
RESULTS = ROOT / "results/radar_domain/synthetic_20261107"
SEED = 20261107
GROUP_COUNTS = {"train": 96, "synthetic_dev": 16}
ORIGIN = "synthetic_generated_non_equipment_fact"


def bytes_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(rows):
    return ("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows)).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def q(value, unit, role="unspecified"):
    return {"kind": "number", "value": value, "unit": unit, "quantity_role": role}


def span(lower, upper, unit):
    return {"kind": "range", "lower": lower, "upper": upper,
            "lower_inclusive": True, "upper_inclusive": True, "unit": unit,
            "quantity_role": "unspecified"}


def text(value, reason="synthetic_categorical_condition", **extra):
    return {"kind": "text", "text": value, "reason": reason,
            "numeric_operations_supported": False, **extra}


def stable_rng(seed, split, index):
    key = f"synthetic-radar-table-v1|{seed}|{split}|{index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def generate_group(split, index, seed=SEED):
    """Invent source facts first; no questions, answers, or real sources are inputs."""
    if split not in GROUP_COUNTS:
        raise ValueError("Undeclared synthetic split")
    prefix = "T" if split == "train" else "D"
    gid = f"SYN-{prefix}-{index:04d}"
    rng = stable_rng(seed, split, index)
    source_id = f"synthetic_{prefix.lower()}_{index:04d}"
    powers = rng.sample(list(range(30, 331, 15)), 3)
    masses = rng.sample([x / 2 for x in range(20, 121)], 3)
    lowers = rng.sample([-45, -30, -15, 0], 3)
    uppers = rng.sample([25, 40, 55, 70], 3)
    models = []
    for n in range(3):
        token = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
        name = f"合成雷达{prefix}{index:04d}-{token}-{n + 1}"
        widths = sorted(rng.sample([0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2], 2))
        prfs = rng.sample(list(range(600, 3601, 150)), 2)
        mode_codes = [f"C{rng.randrange(10, 50)}", f"C{rng.randrange(50, 99)}"]
        modes = [{"mode": mode_codes[k], "pulse_width_us": widths[k], "prf_hz": prfs[k]} for k in range(2)]
        gate_low = rng.choice([8, 12, 16, 24])
        gate_high = rng.choice([96, 128, 192, 256, 384])
        interval_min = rng.choice([1, 2, 3, 4, 5])
        models.append({"subject": name, "power_W": powers[n], "mass_kg": masses[n],
                       "temperature_C": [lowers[n], uppers[n]], "modes": modes,
                       "gate_count": [gate_low, gate_high], "sample_interval_min_s": interval_min,
                       "joint_configuration_constraint": "Not all combinations of gate count and sample interval are supported; no joint compatibility table is supplied."})
    return {"source_group_id": gid, "source_id": source_id, "family_id": f"synthetic:{gid}",
            "split": split, "origin": ORIGIN, "synthetic": True, "equipment_fact": False,
            "seed": seed, "models": models}


def compile_group(group):
    """Compile invented document rows to the same schema as the real typed sources."""
    sid, gid = group["source_id"], group["source_group_id"]
    chunk_id = sid + ":document"
    source_path = f"data/radar_domain_recovery_v2/synthetic_20261107/sources/{sid}.txt"
    lines = [f"SYNTHETIC SOURCE {sid} / {gid}",
             "Origin: synthetic_generated_non_equipment_fact. All entities and values are invented.",
             "These data are NOT manufacturer facts, engineering recommendations, or independently observed measurements.",
             "Each row applies only to its explicitly named synthetic subject; correspondence members stay paired.", ""]
    specs = []
    for m in group["models"]:
        subject = m["subject"]
        modes_text = "; ".join(f"{v['mode']}: pulse_width {v['pulse_width_us']} μs, PRF {v['prf_hz']} Hz" for v in m["modes"])
        specs.extend([
            (subject, "transmitter", "output_power", "scalar", f"{m['power_W']} W", "W", [], q(m["power_W"], "W")),
            (subject, "system", "mass", "scalar", f"{m['mass_kg']} kg", "kg", [], q(m["mass_kg"], "kg")),
            (subject, "system", "operating_temperature", "range", f"{m['temperature_C'][0]} to {m['temperature_C'][1]} °C", "°C", [], span(*m["temperature_C"], "°C")),
            (subject, "transmitter", "mode_parameters", "correspondence", modes_text, "μs; Hz", [],
             {"kind": "correspondence", "rows": [{"mode_code": v["mode"], "pulse_width": q(v["pulse_width_us"], "μs"), "prf": q(v["prf_hz"], "Hz")} for v in m["modes"]],
              "pairing": "same_synthetic_mode_row", "independent_member_cartesian_product_supported": False}),
            (subject, "processor", "gate_count", "range", f"{m['gate_count'][0]} to {m['gate_count'][1]} gates", "count",
             [{"dimension": "parameter combinations", "value": m["joint_configuration_constraint"]}],
             {**span(*m["gate_count"], "count"), "count_kind": "gates", "every_integer_setting_supported": None}),
            (subject, "processor", "sample_interval", "lower_bound", f">= {m['sample_interval_min_s']} s", "s",
             [{"dimension": "parameter combinations", "value": m["joint_configuration_constraint"]}],
             {"kind": "bound", "operator": ">=", "value": m["sample_interval_min_s"], "unit": "s", "quantity_role": "minimum"}),
        ])
    records = []
    for i, (subject, component, attribute, form, value_text, unit_raw, conditions, tv) in enumerate(specs, 1):
        rid = f"{gid}:R{i:03d}"
        line = f"{rid} | subject={subject} | component={component} | attribute={attribute} | value={value_text}"
        for c in conditions:
            line += f" | {c['dimension']}={c['value']}"
        start = len("\n".join(lines)) + 1
        lines.append(line)
        record = {"record_id": rid, "family_id": group["family_id"], "subjects": [subject],
                  "component": component, "attribute": attribute,
                  "value": {"form": form, "text": value_text, "unit_raw": unit_raw},
                  "conditions_all": conditions,
                  "citation": {"chunk_id": chunk_id, "text_sha256": None,
                               "spans": [{"start": start, "end": start + len(line), "quote": line}]},
                  "locator": {"source_group_id": gid, "line_one_based": len(lines), "row_id": rid,
                              "column_names": ["subject", "component", "attribute", "value", "conditions"]},
                  "annotation_note": "Invented source fact for a synthetic tool-use task; not equipment truth."}
        typed = deepcopy(record)
        typed.update({"raw_record": record, "source_fact_id": rid, "source_group_id": gid,
                      "origin": ORIGIN, "synthetic": True, "equipment_fact": False,
                      "reviewer": "programmatic_synthetic_generator", "human_gold": False,
                      "typed_value": {**tv, "source_form": form, "source_text": value_text, "unit_raw": unit_raw},
                      "typed_conditions_all": [{"dimension": c["dimension"], "value": c["value"], "raw": c["value"],
                          "typed_value": text(c["value"], "explicit_synthetic_combination_restriction",
                              constraint_code="not_all_parameter_combinations_supported", joint_cartesian_product_supported=False)} for c in conditions],
                      "sample_independence": "source_group_is_split_unit; record_views_are_not_independent_groups",
                      "shared_subjects_are_query_views_only": False})
        records.append(typed)
    document = "\n".join(lines) + "\n"
    source_hash = digest(document.encode())
    for r in records:
        r["citation"]["text_sha256"] = source_hash
        r["raw_record"]["citation"]["text_sha256"] = source_hash
        r["source"] = {"source_id": sid, "source_path": source_path,
                       "source_sha256": source_hash, "source_uri": source_path, "page_one_based": None,
                       "origin": ORIGIN}
        r["field_evidence"] = {"typed_value": {"source_record_id": r["record_id"],
            "chunk_id": chunk_id, "text_sha256": source_hash,
            "source_spans": r["citation"]["spans"], "source_locator": r["locator"],
            "raw_field": "value", "precision": "exact_generated_source_row"}}
        for s in r["citation"]["spans"]:
            if document[s["start"]:s["end"]] != s["quote"]:
                raise AssertionError("Synthetic source citation construction failed")
    chunk = {"chunk_id": chunk_id, "family_id": group["family_id"], "source_id": sid,
             "source_group_id": gid, "source_path": source_path, "source_sha256": source_hash,
             "source_uri": source_path, "page_one_based": None, "text": document,
             "text_sha256": source_hash, "origin": ORIGIN, "synthetic": True, "equipment_fact": False}
    return document, records, chunk


def make_task_specs(group):
    """Question constants can depend on source facts; answers are computed separately."""
    split = group["split"]
    index = int(group["source_group_id"].rsplit("-", 1)[1])
    rng = stable_rng(group["seed"] + 71, split, index)
    models = group["models"]
    chosen = rng.choice(models)
    mode = rng.choice(chosen["modes"])
    target = min(models, key=lambda m: m["mass_kg"])
    if index % 2:
        numeric_power, numeric_mass = target["power_W"], target["mass_kg"]
    else:
        numeric_power, numeric_mass = min(m["power_W"] for m in models), min(m["mass_kg"] for m in models) - 0.5
    temp = rng.choice([-35, -20, -5, 30, 45, 60, 75])
    sid = group["source_id"]
    lead = f"仅依据合成资料{sid}"
    specs = [
        {"category": "simple", "subject": chosen["subject"],
         "question": f"{lead}，{chosen['subject']}的发射输出功率是多少？请给出主体与以W计的值。"},
        {"category": "qualified_pair", "subject": chosen["subject"], "mode": mode["mode"],
         "question": f"{lead}，{chosen['subject']}采用{mode['mode']}模式时，脉冲宽度和PRF分别是多少？请保留这个模式内的配对。"},
        {"category": "numeric_filter", "minimum_power_W": numeric_power, "maximum_mass_kg": numeric_mass,
         "question": f"{lead}，列出输出功率至少{numeric_power} W且整机质量不超过{numeric_mass} kg的全部合成型号及这两项数值；若没有则返回空列表。"},
        {"category": "range_membership", "temperature_C": temp,
         "question": f"{lead}，将{temp} °C与三个合成型号各自的工作温度范围比较，逐项给出范围与是否包含该温度。"},
        {"category": "empty_result", "minimum_power_W": max(m["power_W"] for m in models) + rng.choice([5, 10, 20]),
         "question": None},
        {"category": "unknown_configuration", "subject": chosen["subject"],
         "gate_count": chosen["gate_count"][1], "sample_interval_s": chosen["sample_interval_min_s"],
         "question": f"{lead}，对{chosen['subject']}，{chosen['gate_count'][1]}个门和{chosen['sample_interval_min_s']} s采样间隔分别是否满足单项范围？资料是否确认二者可同时采用？请分别报告单项判断和联合可行性。"},
    ]
    specs[4]["question"] = f"{lead}，有哪些合成型号的输出功率达到或超过{specs[4]['minimum_power_W']} W？返回全部匹配型号和功率；不要从资料外补充型号。"
    for n, spec in enumerate(specs, 1):
        spec.update({"id": f"{group['source_group_id']}:Q{n:02d}", "source_group_id": group["source_group_id"],
                     "source_id": sid, "family_id": group["family_id"], "split": split,
                     "origin": ORIGIN, "synthetic": True, "equipment_fact": False})
    return specs


def oracle(group, spec):
    """Plain Python oracle over latent source facts; no typed cells or executor calls."""
    models = group["models"]
    category = spec["category"]
    if category == "simple":
        m = next(m for m in models if m["subject"] == spec["subject"])
        return {"rows": [{"subject": m["subject"], "power_W": m["power_W"]}]}
    if category == "qualified_pair":
        m = next(m for m in models if m["subject"] == spec["subject"])
        mode = next(v for v in m["modes"] if v["mode"] == spec["mode"])
        return {"rows": [{"subject": m["subject"], "mode": mode["mode"],
                          "pulse_width_us": mode["pulse_width_us"], "prf_hz": mode["prf_hz"]}]}
    if category in {"numeric_filter", "empty_result"}:
        rows = []
        for m in models:
            if m["power_W"] < spec["minimum_power_W"]:
                continue
            if category == "numeric_filter" and m["mass_kg"] > spec["maximum_mass_kg"]:
                continue
            row = {"subject": m["subject"], "power_W": m["power_W"]}
            if category == "numeric_filter":
                row["mass_kg"] = m["mass_kg"]
            rows.append(row)
        return {"rows": rows}
    if category == "range_membership":
        return {"temperature_C": spec["temperature_C"],
                "rows": [{"subject": m["subject"], "lower_temperature_C": m["temperature_C"][0],
                           "upper_temperature_C": m["temperature_C"][1],
                           "within_range": m["temperature_C"][0] <= spec["temperature_C"] <= m["temperature_C"][1]}
                         for m in models]}
    if category == "unknown_configuration":
        m = next(m for m in models if m["subject"] == spec["subject"])
        checks = {"gate_count": m["gate_count"][0] <= spec["gate_count"] <= m["gate_count"][1],
                  "sample_interval": spec["sample_interval_s"] >= m["sample_interval_min_s"]}
        return {"joint_support": {"status": "not_established" if all(checks.values()) else "contradicted_by_individual_domain",
            "individual_domain_checks": checks,
            "requested": {"gate_count": {"kind": "number", "value": spec["gate_count"], "unit": "count"},
                          "sample_interval": {"kind": "number", "value": spec["sample_interval_s"], "unit": "s"}},
            "source_restrictions": ["not_all_parameter_combinations_supported"] if m["joint_configuration_constraint"] else []}}
    raise ValueError(f"Unknown synthetic task category: {category}")


def answer_spec(category):
    """Formatting only: never includes values, row counts, record IDs, or verdicts."""
    if category == "unknown_configuration":
        return {"outputs": [{"name": "joint_support", "kind": "assessment"}]}
    columns = {"simple": ["subject", "power_W"],
               "qualified_pair": ["subject", "mode", "pulse_width_us", "prf_hz"],
               "numeric_filter": ["subject", "power_W", "mass_kg"],
               "range_membership": ["subject", "lower_temperature_C", "upper_temperature_C", "within_range"],
               "empty_result": ["subject", "power_W"]}[category]
    return {"outputs": [{"name": "rows", "kind": "table", "columns": sorted(columns)}]}


def build_sources(seed=SEED, group_counts=None):
    """Build source and question artifacts without computing any reference answer."""
    counts = dict(GROUP_COUNTS if group_counts is None else group_counts)
    groups, records, chunks, specs = [], [], [], []
    outputs, source_group_by_id = {}, {}
    split_groups = {split: [] for split in counts}
    identifiers = set()

    def assign(identifier, gid):
        if identifier in identifiers:
            raise ValueError(f"Synthetic identifier collision: {identifier}")
        identifiers.add(identifier)
        source_group_by_id[identifier] = gid

    for split, count in counts.items():
        for index in range(1, count + 1):
            group = generate_group(split, index, seed)
            document, rows, chunk = compile_group(group)
            tasks = make_task_specs(group)
            gid = group["source_group_id"]
            groups.append(group)
            records.extend(rows)
            chunks.append(chunk)
            specs.extend(tasks)
            outputs[ROOT / chunk["source_path"]] = document.encode()
            split_groups[split].append(gid)
            assign(gid, gid)
            assign(group["source_id"], gid)
            assign(chunk["chunk_id"], gid)
            for model in group["models"]:
                assign(model["subject"], gid)
            for record in rows:
                assign(record["record_id"], gid)
            for spec in tasks:
                assign(spec["id"], gid)
    metadata = {"origin": ORIGIN, "synthetic": True, "equipment_fact": False,
                "human_gold": False, "seed": seed, "model_outputs_used": False,
                "real_source_records_used": False, "generator_holdout_accessed": False,
                "holdout_questions_or_answers_used": False}
    table = {"schema": "radar_typed_sources_v1", **metadata, "records": records,
             "capability_scope": "generated exact numbers, inclusive ranges, bounds, paired modes, and explicit joint-configuration restrictions"}
    chunk_table = {"schema": "synthetic_source_chunks_v1", **metadata, "chunks": chunks}
    outputs[LOCAL / "typed_records.json"] = bytes_json(table)
    outputs[LOCAL / "chunks.json"] = bytes_json(chunk_table)
    outputs[LOCAL / "source_latents.json"] = bytes_json({"schema": "synthetic_latent_sources_v1", **metadata, "groups": groups})
    outputs[LOCAL / "task_specs.json"] = bytes_json({"schema": "synthetic_task_specs_v1", **metadata, "tasks": specs})
    outputs[LOCAL / "source_group_by_id.json"] = bytes_json({"schema": "source_group_by_id_v1", **metadata,
        "split_groups": split_groups, "source_group_by_id": source_group_by_id,
        "split_unit": "entire_source_group", "final_global_split_freeze_owned_by_root": True})
    outputs[LOCAL / "answer_specs.json"] = bytes_json({"schema": "synthetic_answer_specs_v1", **metadata,
        "answer_specs": {s["id"]: answer_spec(s["category"]) for s in specs},
        "content_policy": "Outer keys identify questions; each specification contains output names/kinds/columns only, no answer values, counts, source/entity IDs, or statuses."})
    for split in counts:
        outputs[LOCAL / f"{split}.questions.jsonl"] = jsonl_bytes(
            [{"id": s["id"], "question": s["question"]} for s in specs if s["split"] == split])
    return outputs, groups, table, chunk_table, specs


def make_actions(spec):
    """Author generic source queries; no expected answers enter the action stream."""
    from experiments.radar_domain.table_program import ProgramBuilder
    b = ProgramBuilder()

    def scan(attribute, subject=None):
        config = {"family_id": spec["family_id"], "attribute": attribute}
        if subject is not None:
            config["subject"] = subject
        return b.add("Scan", config)

    def project(handle, columns):
        return b.add("Project", {"columns": columns}, [handle])

    category = spec["category"]
    if category == "simple":
        h = scan("output_power", spec["subject"])
        h = project(h, {"subject": "subject", "power_W": "typed_value.value"})
    elif category == "qualified_pair":
        h = scan("mode_parameters", spec["subject"])
        h = b.add("Expand", {"path": "typed_value.rows", "as": "variant"}, [h])
        h = b.add("Filter", {"predicates": [{"column": "variant.mode_code", "op": "==", "value": spec["mode"]}]}, [h])
        h = project(h, {"subject": "subject", "mode": "variant.mode_code",
                        "pulse_width_us": "variant.pulse_width.value", "prf_hz": "variant.prf.value"})
    elif category == "numeric_filter":
        p = project(scan("output_power"), {"subject": "subject", "power": "typed_value"})
        m = project(scan("mass"), {"subject": "subject", "mass": "typed_value"})
        h = b.add("Join", {"on": ["subject"]}, [p, m])
        h = b.add("Filter", {"predicates": [
            {"column": "power", "op": ">=", "value": spec["minimum_power_W"], "unit": "W"},
            {"column": "mass", "op": "<=", "value": spec["maximum_mass_kg"], "unit": "kg"}]}, [h])
        h = project(h, {"subject": "subject", "power_W": "power.value", "mass_kg": "mass.value"})
    elif category == "range_membership":
        h = project(scan("operating_temperature"), {"subject": "subject", "temperature": "typed_value"})
        h = b.add("Derive", {"columns": {"within_range": {
            "column": "temperature", "op": "contains", "value": spec["temperature_C"], "unit": "degC"}}}, [h])
        h = project(h, {"subject": "subject", "lower_temperature_C": "temperature.lower",
                        "upper_temperature_C": "temperature.upper", "within_range": "within_range"})
    elif category == "empty_result":
        h = scan("output_power")
        h = b.add("Filter", {"predicates": [{"column": "typed_value", "op": ">=",
                                             "value": spec["minimum_power_W"], "unit": "W"}]}, [h])
        h = project(h, {"subject": "subject", "power_W": "typed_value.value"})
    elif category == "unknown_configuration":
        h = scan(["gate_count", "sample_interval"], spec["subject"])
        h = project(h, {"parameter": "attribute", "domain": "typed_value",
                        "restriction": ["conditions", "parameter combinations"]})
        h = b.add("Assess", {"mode": "joint_support", "parameter_column": "parameter",
            "domain_column": "domain", "requested": {
                "gate_count": {"kind": "number", "value": spec["gate_count"], "unit": "count"},
                "sample_interval": {"kind": "number", "value": spec["sample_interval_s"], "unit": "s"}}}, [h])
        return b.finish({"joint_support": h})
    else:
        raise ValueError("Unknown synthetic task category")
    return b.finish({"rows": h})


def semantic_answer(answer, category):
    """Strip executor provenance for the oracle comparison, never for reference storage."""
    if category == "unknown_configuration":
        item = answer["outputs"]["joint_support"]
        if item["kind"] != "assessment":
            raise ValueError("Joint-support answer must be an actual assessment")
        result = item["result"]
        return {"joint_support": {key: deepcopy(result[key]) for key in
                ("status", "individual_domain_checks", "source_restrictions", "requested")}}
    item = answer["outputs"]["rows"]
    if item["kind"] != "table":
        raise ValueError("Source-query answer must be an actual table")
    rows = [deepcopy(row["values"]) for row in item["rows"]]
    rows.sort(key=lambda r: r.get("subject", ""))
    result = {"rows": rows}
    if category == "range_membership":
        result["temperature_C"] = item["derived_conditions"][0]["within_range"]["value"]
    return result


def canonical_semantics(value):
    if isinstance(value, dict):
        return {k: canonical_semantics(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonical_semantics(x) for x in value]
    if type(value) is float and value.is_integer():
        return int(value)
    return value


def verify_reference(executor, group, spec):
    from experiments.radar_domain.table_program import make_reference
    actions = make_actions(spec)
    reference = make_reference(executor, spec["id"], spec["question"], actions)
    actual = semantic_answer(reference["answer"], spec["category"])
    expected = oracle(group, spec)
    if "rows" in expected:
        expected["rows"].sort(key=lambda r: r.get("subject", ""))
    if actual != expected:
        raise ValueError(f"Independent synthetic oracle mismatch for {spec['id']}: {actual!r} != {expected!r}")
    allowed_prefix = group["source_group_id"] + ":R"
    used_ids = reference["answer"]["source_records"]
    if not used_ids or any(not rid.startswith(allowed_prefix) for rid in used_ids):
        raise ValueError("Synthetic answer escaped its source group or lost evidence")
    if "subject" in spec and any(spec["subject"] not in r["subjects"] for r in used_ids.values()):
        raise ValueError("Qualified synthetic answer used a different subject")
    for output_spec in answer_spec(spec["category"])["outputs"]:
        output = reference["answer"]["outputs"][output_spec["name"]]
        if output["kind"] != output_spec["kind"] or ("columns" in output_spec and output["columns"] != output_spec["columns"]):
            raise ValueError("Executed answer does not satisfy its value-free format specification")
    reference.update({"origin": ORIGIN, "synthetic": True, "equipment_fact": False,
        "source_group_id": group["source_group_id"], "split": group["split"],
        "category": spec["category"], "human_gold": False,
        "trajectory_kind": "programmatic_executor_success_reference",
        "oracle_verified": True, "model_generated": False, "model_failure": False})
    check = {"id": spec["id"], "source_group_id": group["source_group_id"],
        "split": group["split"], "category": spec["category"], "origin": ORIGIN,
        "actual_execution_completed": True, "oracle_match": True,
        "actual_semantics_sha256": digest(bytes_json(canonical_semantics(actual))),
        "oracle_semantics_sha256": digest(bytes_json(canonical_semantics(expected))),
        "action_count": len(actions), "source_record_ids": sorted(used_ids),
        "empty_result": "rows" in actual and not actual["rows"],
        "unknown_result": "joint_support" in actual and actual["joint_support"]["status"] == "not_established"}
    return reference, check


def build(seed=SEED, group_counts=None):
    from experiments.radar_domain import table_program
    outputs, groups, table, chunks, specs = build_sources(seed, group_counts)
    executor = table_program.TableExecutor(table)
    by_group = {g["source_group_id"]: g for g in groups}
    references, checks = [], []
    for spec in specs:
        reference, check = verify_reference(executor, by_group[spec["source_group_id"]], spec)
        references.append(reference)
        checks.append(check)
    split_names = list(dict.fromkeys(g["split"] for g in groups))
    counts_by_split = {}
    for split in split_names:
        refs = [r for r in references if r["split"] == split]
        split_checks = [r for r in checks if r["split"] == split]
        counts_by_split[split] = {"source_groups": sum(g["split"] == split for g in groups),
            "questions": len(refs), "executed_success_references": len(refs),
            "oracle_verified_references": len(refs),
            "categories": dict(sorted(Counter(r["category"] for r in refs).items())),
            "empty_answers": sum(c["empty_result"] for c in split_checks),
            "unknown_joint_configuration_answers": sum(c["unknown_result"] for c in split_checks)}
        outputs[LOCAL / f"{split}.references.jsonl"] = jsonl_bytes(refs)
    outputs[RESULTS / "oracle_checks.jsonl"] = jsonl_bytes(checks)
    fingerprint = {digest(bytes_json([{k: v for k, v in m.items() if k != "subject"}
                                     for m in group["models"]])) for group in groups}
    if len(fingerprint) != len(groups):
        raise ValueError("A synthetic group merely renamed an identical numerical source profile")
    if max(c["action_count"] for c in checks) > 24:
        raise ValueError("Reference exceeds shared tool call budget")
    input_hashes = {str(Path(p).relative_to(ROOT)): digest(Path(p).read_bytes())
                    for p in (__file__, table_program.__file__)}
    manifest = {"schema": "synthetic_radar_table_data_manifest_v1", "origin": ORIGIN,
        "synthetic": True, "equipment_fact": False, "human_gold": False,
        "seed": seed, "generation_version": 1, "counts_by_split": counts_by_split,
        "total_source_groups": len(groups), "total_source_records": len(table["records"]),
        "total_questions": len(specs), "total_executed_success_references": len(references),
        "distinct_numerical_source_profiles": len(fingerprint),
        "source_group_split_isolation_verified": True,
        "record_view_expansion_changes_source_group_identity": False,
        "source_group_by_id_path": str((LOCAL / "source_group_by_id.json").relative_to(ROOT)),
        "all_citations_point_to_local_synthetic_documents": True,
        "question_only_inputs": {s: str((LOCAL / f"{s}.questions.jsonl").relative_to(ROOT)) for s in split_names},
        "reference_files": {s: str((LOCAL / f"{s}.references.jsonl").relative_to(ROOT)) for s in split_names},
        "independent_oracle": "oracle(group,spec) reads latent generated source objects; it does not read typed cells, actions, executor outputs, or reference answers",
        "reference_answer_provenance": "Unmodified answer returned by the shared TableExecutor via make_reference after every action actually executes",
        "oracle_mismatches": 0, "generated_model_failures": 0, "model_runs": 0,
        "training_started": False, "server_started": False, "new_GPU_hours": 0,
        "real_source_answers_or_holdout_used": False, "old_real_dev_moved_to_train": False,
        "global_split_frozen": False, "global_split_freeze_owner": "root",
        "synthetic_dev_scope": "same generator with isolated source groups; development checks, not independent real-source generalization evidence",
        "input_sha256": input_hashes,
        "output_sha256": {str(p.relative_to(ROOT)): digest(v) for p, v in sorted(outputs.items())},
        "reproduce": "cache/proposal_env/bin/python -m experiments.radar_domain.synthetic_table_data --check",
        "limitations": ["All generated values and entities are invented; no claim of real radar behavior or physical joint feasibility.",
                        "Successful references are programmatic tool executions, not model-generated successful or failed trajectories.",
                        "Synthetic development groups share generator templates with synthetic training groups.",
                        "Final cross-dataset train/dev/holdout freeze is performed separately by the root coordinator."]}
    outputs[LOCAL / "full_manifest.json"] = bytes_json(manifest)
    outputs[RESULTS / "summary.json"] = bytes_json({"schema": "synthetic_radar_table_data_summary_v1",
        "origin": ORIGIN, "synthetic": True, "equipment_fact": False, "human_gold": False,
        "counts_by_split": counts_by_split, "total_groups": len(groups),
        "total_source_records": len(table["records"]), "oracle_mismatches": 0,
        "all_references_executed": True, "model_failures_fabricated": False,
        "minimum_action_count": min(c["action_count"] for c in checks),
        "maximum_action_count": max(c["action_count"] for c in checks),
        "manifest_path": str((LOCAL / "full_manifest.json").relative_to(ROOT)),
        "manifest_sha256": digest(outputs[LOCAL / "full_manifest.json"]),
        "global_split_frozen": False, "model_or_training_runs": 0})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--write", action="store_true")
    choice.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    # Keep validation receipts fresh when the shared executor is rebound before
    # the coordinator freezes the global split. Tests call build(), not main().
    import importlib
    import io
    import unittest
    test_module = importlib.import_module("tests.test_synthetic_table_data")
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    test_log = io.StringIO()
    result = unittest.TextTestRunner(stream=test_log).run(suite)
    if not result.wasSuccessful():
        raise ValueError(test_log.getvalue())
    manifest = json.loads(outputs[LOCAL / "full_manifest.json"])
    outputs[RESULTS / "validation.json"] = bytes_json({
        "schema": "synthetic_table_validation_v1", "origin": ORIGIN, "human_gold": False,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "core_artifacts_reproduced": len(outputs), "executed_references": manifest["total_executed_success_references"],
        "oracle_mismatches": 0, "tampered_typed_value_rejected_by_independent_oracle": True,
        "question_only_inputs_separate_from_references": True, "answer_specs_format_only": True,
        "generator_no_real_or_holdout_data_reads_tested": True,
        "manifest_sha256": digest(outputs[LOCAL / "full_manifest.json"]),
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "tests_sha256": digest(Path(test_module.__file__).read_bytes()),
        "commands": ["cache/proposal_env/bin/python -m unittest tests.test_synthetic_table_data -q",
                     "cache/proposal_env/bin/python -m experiments.radar_domain.synthetic_table_data --check"]})
    for path, content in outputs.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        elif not path.exists() or path.read_bytes() != content:
            raise ValueError(f"Synthetic artifact does not reproduce: {path}")
    summary = json.loads(outputs[RESULTS / "summary.json"])
    print(json.dumps({"status": "written" if args.write else "reproduced",
                      "counts_by_split": summary["counts_by_split"],
                      "oracle_mismatches": 0, "artifact_count": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

