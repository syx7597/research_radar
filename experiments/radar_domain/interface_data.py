"""Build grouped, fictitious successful interface traces without model calls.

This teaches the existing tool contract, not radar facts or failure recovery.
Split assignments, wording templates and fictional source instances are fixed
before trajectories are generated. All artifacts are reproducible from this file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import tempfile

from experiments.agent_feedback.environment import compact
from experiments.agent_feedback.inference import execute_response
from .development_environment import DevelopmentKB, DevelopmentEpisode, execute_program, prompt_messages

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/radar_interface_v1"
PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_interface_synthetic_v1"
ORIGINAL = ROOT / "artifacts/thesis_direction_review/radar_development_v2/readings.json"
VERSION, SEED = "radar_interface_synthetic_v1", 20261005
FAMILIES = ("plain", "conditional_scalar", "conditional_range", "event", "range", "unknown")
SPLIT_COUNTS = {"train": (10, 10, 10, 10, 10, 10), "dev": (4, 3, 3, 3, 4, 3), "holdout": (3, 4, 3, 4, 3, 3)}
FRAMES = {
    "train": [("请从{name}的合成资料中读取{field}。", "Read {field} from the synthetic records for {name}."),
              ("查找{name}，取出其中的{field}。", "Find {name} and retrieve its {field}.")],
    "dev": [("资料包里，{name}对应的{field}是哪条记录？", "Which record in the packet is the {field} for {name}?"),
            ("定位到{name}后，返回资料所记的{field}。", "After locating {name}, return the recorded {field}.")],
    "holdout": [("针对{name}，我需要资料中保留的{field}。", "For {name}, I need the {field} preserved in the source data."),
                ("把{name}在资料中的{field}检索出来。", "Look up the source entry giving the {field} of {name}.")],
}
EVENT_NAMES = {"in_service": ("服役", "in-service"), "initial_operational_capacity": ("初始作战能力", "initial operational capacity"),
               "cancellation": ("取消", "cancellation"), "retirement": ("退役", "retirement")}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def token(value):
    return digest(value.encode())[:12]


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def split_plan():
    """Freeze group/family/template choices before any question or answer exists."""
    plan, ordinal = [], 0
    for split, counts in SPLIT_COUNTS.items():
        families = [family for family, count in zip(FAMILIES, counts) for _ in range(count)]
        random.Random(f"{SEED}:{split}:family_order").shuffle(families)
        family_seen = Counter()
        for index, family in enumerate(families):
            ordinal += 1
            plan.append({"split": split, "family": family, "ordinal": ordinal,
                         "family_index": family_seen[family],
                         "knowledge_id": "ifkb-" + token(f"{SEED}:instance:{ordinal}"),
                         "template_id": f"{split}-wording-{index % 2}", "template_index": index % 2})
            family_seen[family] += 1
    return plan


def make_instance(spec):
    """Every complete KB has three entities, all keys, and competing values."""
    rng = random.Random(f"{SEED}:{spec['knowledge_id']}")
    names = ["SYN-" + token(f"{spec['knowledge_id']}:entity:{index}").upper() for index in range(3)]
    rng.shuffle(names)
    unknown_role = rng.randrange(3)
    rows = []
    def add(role, attribute, kind, *, condition=None, event=None, unit=None):
        index = len(rows)
        # Arbitrary, deliberately nonphysical values. No original radar answers.
        number = 21001 + spec["ordinal"] * 101 + index * 2
        record = {"fact_id": "synfact-" + token(f"{spec['knowledge_id']}:{index}"),
                  "entity_name": names[role], "source_subject": names[role], "attribute": attribute,
                  "event_type": event, "condition_raw": condition, "value_kind": kind,
                  "value_status": "not_recorded" if kind == "unknown" else "stated",
                  "value": None if kind in {"range", "unknown"} else ("fictional_class_" + token(str(number)) if kind == "text" else str(number)),
                  "min_value": str(number) if kind == "range" else None,
                  "max_value": str(number + 1) if kind == "range" else None,
                  "unit_std": None if kind == "unknown" else unit,
                  "scope_notes_zh": "人工合成的虚构字段，仅验证接口；数值不具有物理真实性，不描述真实装备。",
                  "unknown_scope": "bound_source_field" if kind == "unknown" else None,
                  "original_entity_attribution_usable": True,
                  "review_status": "AI_only_development", "synthetic": True,
                  "equipment_truth_verified": False, "human_verified": False, "independent_gold": False}
        rows.append(record)
    for role in range(3):
        add(role, "type_description", "text")
        add(role, "pulse_width", "scalar", unit="us")
        for condition in ("surface", "air"):
            add(role, "prf", "scalar", condition=condition, unit="pps")
        for condition in ([None] if role == 0 else ["Illuminator", "Tracking Radar"]):
            add(role, "frequency", "range", condition=condition, unit="GHz")
        for attribute, events in (("service_entry", ("in_service", "initial_operational_capacity")),
                                  ("lifecycle_event", ("cancellation", "retirement"))):
            for event in events:
                add(role, attribute, "event", event=event, unit="year")
        add(role, "power", "unknown" if role == unknown_role else "scalar", unit="W")
    rng.shuffle(rows)
    source = {"version": VERSION, "synthetic": True, "knowledge_id": spec["knowledge_id"],
              "entries": [{key: row[key] for key in ("fact_id", "entity_name", "attribute", "event_type", "condition_raw", "value_kind", "value_status", "value", "min_value", "max_value", "unit_std")} for row in rows]}
    source_bytes = encoded(source)
    for index, row in enumerate(rows):
        row["citation"] = {"source_uri": "urn:synthetic-interface:" + spec["knowledge_id"],
                           "source_sha256": digest(source_bytes),
                           "source_file": str((RAW / "sources" / f"{spec['knowledge_id']}.json").relative_to(ROOT)),
                           "locator": {"json_pointer": f"/entries/{index}"},
                           "evidence_text": compact(source["entries"][index])}
    kb = {"version": VERSION, "split": "development_only", "partition": spec["split"],
          "synthetic": True, "human_verified": False, "independent_gold": False, "records": rows}
    family = spec["family"]
    role = rng.randrange(3)
    selector, expected_kind = {}, None
    if family == "plain":
        attribute = "type_description" if spec["family_index"] % 2 == 0 else "pulse_width"
        expected_kind = "text" if attribute == "type_description" else "scalar"
        field = ("类型说明条目", "type-description entry") if expected_kind == "text" else ("脉冲宽度条目", "pulse-width entry")
    elif family == "conditional_scalar":
        attribute, expected_kind = "prf", "scalar"
        value = ["surface", "air"][spec["family_index"] % 2]
        selector = {"condition_raw": value}
        field = (f"带 {value} 标签的脉冲重复频率条目", f"pulse-repetition-frequency entry labeled {value}")
    elif family == "conditional_range":
        role, attribute, expected_kind = rng.choice([1, 2]), "frequency", "range"
        value = ["Illuminator", "Tracking Radar"][spec["family_index"] % 2]
        selector = {"condition_raw": value}
        field = (f"带 {value} 标签的频率条目", f"frequency entry labeled {value}")
    elif family == "event":
        attribute, value = [("service_entry", "in_service"), ("service_entry", "initial_operational_capacity"),
                            ("lifecycle_event", "cancellation"), ("lifecycle_event", "retirement")][spec["family_index"] % 4]
        expected_kind = "event"
        selector = {"event_type": value}
        field = (EVENT_NAMES[value][0] + "事件条目", EVENT_NAMES[value][1] + " event entry")
    elif family == "range":
        role, attribute, expected_kind, field = 0, "frequency", "range", ("频率条目", "frequency entry")
    else:
        role, attribute, expected_kind, field = unknown_role, "power", "unknown", ("功率条目", "power entry")
    intent = {"entity_name": names[role], "attribute": attribute, "selector": selector, "expected_value_kind": expected_kind}
    return kb, source_bytes, intent, field


def semantic_select(records, intent):
    """Declarative target interpretation, independent of the program interpreter."""
    return [row for row in records if row["entity_name"] == intent["entity_name"] and row["attribute"] == intent["attribute"]
            and all(row[key] == value for key, value in intent["selector"].items())]


def canonical_program(intent):
    query = "QueryAttr<arg>" + intent["attribute"]
    if intent["selector"]:
        key, value = next(iter(intent["selector"].items()))
        query = "QueryAttrUnderCondition<arg>" + "<arg>".join((intent["attribute"], key, value))
    return "Find<arg>" + intent["entity_name"] + "<func>" + query


def validate_instance(kb, source, intent):
    expected = semantic_select(kb.records, intent)
    if len(expected) != 1 or expected[0]["value_kind"] != intent["expected_value_kind"]:
        raise ValueError("Declarative intent is ambiguous or has the wrong value kind")
    result = execute_program(kb, canonical_program(intent))
    if result.prediction != expected or any(not event["observation"]["ok"] for event in result.events):
        raise ValueError("Executable program disagrees with the declarative intent")
    entries = json.loads(source)["entries"]
    for row in kb.records:
        citation = row["citation"]
        index = int(citation["locator"]["json_pointer"].rsplit("/", 1)[1])
        if citation["source_sha256"] != digest(source) or citation["evidence_text"] != compact(entries[index]):
            raise ValueError("Synthetic citation does not bind its actual source entry")
        if any(row.get(key) != value for key, value in entries[index].items()):
            raise ValueError("Synthetic query fields differ from their bound source entry")
    for mutation in ({**intent, "entity_name": next(name for name in kb.entities if name != intent["entity_name"])},
                     {**intent, "attribute": next(key for key in kb.attributes if key != intent["attribute"])}):
        other = execute_program(kb, canonical_program(mutation))
        if other.prediction is None or expected[0]["fact_id"] in {row["fact_id"] for row in other.prediction}:
            raise ValueError("Wrong entity/attribute counterfactual was not separated from target")
    if intent["selector"]:
        key, value = next(iter(intent["selector"].items()))
        alternatives = [row[key] for row in kb.records if row["entity_name"] == intent["entity_name"] and row["attribute"] == intent["attribute"] and row[key] != value]
        if not alternatives:
            raise ValueError("Conditional task has no competing same-entity record")
        wrong = {**intent, "selector": {key: alternatives[0]}}
        other = execute_program(kb, canonical_program(wrong))
        if not other.prediction or expected[0] in other.prediction:
            raise ValueError("Wrong qualifier must select a competing record")
    if intent["expected_value_kind"] == "unknown":
        row = expected[0]
        if any(row[key] is not None for key in ("value", "min_value", "max_value", "unit_std")):
            raise ValueError("Synthetic unknown is not null-valued")
        empty = execute_program(kb, canonical_program({**intent, "selector": {"condition_raw": "synthetic_absent_label"}}))
        if empty.prediction != [] or result.prediction == []:
            raise ValueError("Explicit unknown and no matching record were collapsed")
    return expected


def make_traces(kb, question, intent):
    program = canonical_program(intent)
    agent = {key: question[key] for key in ("id", "knowledge_id")}
    agent["messages"] = prompt_messages(question["question"], kb)
    agent["supervise"] = [False, False]
    selector = list(intent["selector"].items())
    query_inputs = [intent["attribute"]] + (list(selector[0]) if selector else [])
    calls = [{"name": "step", "arguments": {"function": "Find", "inputs": [intent["entity_name"]], "dependencies": []}},
             {"name": "step", "arguments": {"function": "QueryAttrUnderCondition" if selector else "QueryAttr", "inputs": query_inputs, "dependencies": [0]}},
             {"name": "finish", "arguments": {"answer_handle": 1}}]
    episode = DevelopmentEpisode(kb)
    for call in calls:
        text = "<tool_call>" + compact(call) + "</tool_call>"
        observation = execute_response(episode, text)
        if not observation["ok"]:
            raise ValueError("Recovery or invalid calls are forbidden in interface training")
        agent["messages"].extend([{"role": "assistant", "content": text}, {"role": "tool", "content": compact(observation)}])
        agent["supervise"].extend([True, False])
    if episode.prediction != semantic_select(kb.records, intent):
        raise ValueError("Agent and declarative target disagree")
    one_shot = {key: question[key] for key in ("id", "knowledge_id")}
    one_shot.update(messages=prompt_messages(question["question"], kb, program=True) + [{"role": "assistant", "content": program}], supervise=[False, False, True])
    return agent, one_shot


def normalized_number(value):
    try:
        return Decimal(value) if value is not None else None
    except InvalidOperation:
        return None


def audit_exclusion(originals, all_records, question_texts):
    forbidden_text = {row["entity_name"] for row in originals} | {row["citation"]["source_uri"] for row in originals}
    forbidden_text |= {row["value"] for row in originals if row["value_kind"] == "text"}
    forbidden_numbers = {normalized_number(row[key]) for row in originals for key in ("value", "min_value", "max_value")} - {None}
    blob = json.dumps(all_records, ensure_ascii=False) + "\n" + "\n".join(question_texts)
    if any(text in blob for text in forbidden_text):
        raise ValueError("Original entity, source URI or answer text leaked into synthetic material")
    actual = {normalized_number(row[key]) for row in all_records for key in ("value", "min_value", "max_value")} - {None}
    if actual & forbidden_numbers:
        raise ValueError("An original numeric answer literal was reused")
    return {"original_entities_uris_text_answers_found": 0, "original_numeric_answer_values_found": 0,
            "numeric_rule": "Decimal-value equality in value/min_value/max_value, not arbitrary digit substrings of hashes or IDs."}


def build():
    plan, outputs, groups, records_all, texts_all = split_plan(), {}, [], [], []
    rows = {split: {kind: [] for kind in ("questions", "references", "agent_trajectories", "program_trajectories")} for split in SPLIT_COUNTS}
    ranks, family_kinds = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="radar_interface_cpu_") as temporary:
        for spec in plan:
            document, source, intent, field = make_instance(spec)
            filename = spec["knowledge_id"] + ".json"
            kb_bytes = encoded(document)
            kb_file = Path(temporary) / filename
            kb_file.write_bytes(kb_bytes)
            kb = DevelopmentKB(kb_file)
            expected = validate_instance(kb, source, intent)
            outputs[RAW / "kb" / filename] = kb_bytes
            outputs[RAW / "sources" / filename] = source
            split, target = spec["split"], expected[0]
            ranks[(split, kb.entities.index(intent["entity_name"]))] += 1
            family_kinds[(split, spec["family"], target["value_kind"])] += 1
            group = {**spec, "entity_aliases": list(kb.entities), "target_entity_schema_position": kb.entities.index(intent["entity_name"]),
                     "target_record_position": document["records"].index(target), "record_count": len(document["records"])}
            groups.append(group)
            for language, index in (("zh", 0), ("en", 1)):
                text = FRAMES[split][spec["template_index"]][index].format(name=intent["entity_name"], field=field[index])
                disclaimer = "全部记录均为虚构合成材料，仅用于接口练习。" if language == "zh" else "All records are fictitious synthetic material used only for interface practice."
                context = f"source_context_id={spec['knowledge_id']}; source_uri=urn:synthetic-interface:{spec['knowledge_id']}"
                question = {"id": spec["knowledge_id"] + "-" + language, "question": text + "\n" + disclaimer + "\n" + context,
                            "knowledge_id": spec["knowledge_id"]}
                agent, program = make_traces(kb, question, intent)
                rows[split]["questions"].append(question)
                rows[split]["references"].append({"id": question["id"], "knowledge_id": spec["knowledge_id"], "language": language,
                    "family": spec["family"], "template_id": spec["template_id"], "declarative_intent": intent,
                    "expected_fact_ids": [target["fact_id"]], "canonical_program": canonical_program(intent), "synthetic": True})
                rows[split]["agent_trajectories"].append(agent)
                rows[split]["program_trajectories"].append(program)
                texts_all.append(question["question"])
            records_all.extend(kb.records)
    for split, kinds in rows.items():
        for kind, values in kinds.items():
            outputs[RAW / f"{split}.{kind}.jsonl"] = jsonl(values)
    exclusion = audit_exclusion(json.loads(ORIGINAL.read_bytes())["records"], records_all, texts_all)
    alias_sets = [{name for group in groups if group["split"] == split for name in group["entity_aliases"]} for split in SPLIT_COUNTS]
    if any(alias_sets[i] & alias_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Entity aliases crossed grouped splits")
    for split in SPLIT_COUNTS:
        if {group["family"] for group in groups if group["split"] == split} != set(FAMILIES):
            raise ValueError("Every split must cover every task family")
        if any(ranks[(split, rank)] == 0 for rank in range(3)):
            raise ValueError("Target entity positions do not cover the whole schema catalog")
        if any(family_kinds[(split, "plain", kind)] == 0 for kind in ("scalar", "text")):
            raise ValueError("Every split must include both plain scalar and plain text tasks")
    quality = {"status": "cpu_validated", "group_count": len(plan), "bilingual_questions": 200,
               "successful_program_validations": 100, "successful_agent_trajectories": 200,
               "invalid_or_recovery_training_calls": 0, "known_unknown_vs_empty_checked": True,
               "wrong_entity_attribute_and_present_qualifier_checked": True,
               "source_sha_pointer_and_evidence_checked": True, "quarantined_groups": [],
               "target_schema_positions": {f"{split}:{rank}": count for (split, rank), count in sorted(ranks.items())},
               "family_value_kind_counts": {":".join(key): value for key, value in sorted(family_kinds.items())},
               "original_answer_exclusion": exclusion, "model_inference_runs": 0, "new_training_started": False}
    outputs[PUBLIC / "quality.json"] = encoded(quality)
    outputs[PUBLIC / "split_assignments.json"] = encoded({"version": VERSION, "seed": SEED, "groups": groups})
    outputs[PUBLIC / "examples.json"] = encoded({"scope": "Fictitious train-only examples, no real radar assertions.",
        "questions": rows["train"]["questions"][:4], "references": rows["train"]["references"][:4]})
    manifest = {"version": VERSION, "status": "cpu_validated", "seed": SEED, "synthetic": True,
        "split_counts": {split: {"groups": sum(counts), "questions": 2 * sum(counts)} for split, counts in SPLIT_COUNTS.items()},
        "question_schema": ["id", "question", "knowledge_id"], "trajectory_schema": ["id", "knowledge_id", "messages", "supervise"],
        "training_files": {"agent": "data/radar_interface_v1/train.agent_trajectories.jsonl", "program": "data/radar_interface_v1/train.program_trajectories.jsonl"},
        "knowledge_routing": "data/radar_interface_v1/kb/{knowledge_id}.json; each route selects a full three-entity KB, never a target-only view",
        "supervision": "Exactly three successful assistant actions for agents; one complete program for P. Tool observations are never supervised; no Done turn or recovery traces.",
        "partition_policy": "One fictitious KB/source instance and bilingual intent per group. Entities, KB instances and wording template IDs/text are disjoint across train/dev/holdout.",
        "partition_limits": "Task semantic families, tool sequence structures, attribute names, shared field phrases and protocol boilerplate are intentionally shared. This is not unseen-reasoning-structure or real-document-source generalization.",
        "original_data_exclusion": "Real entity names, real source URIs, text answers and normalized numeric answer values from the original twelve readings are prohibited in synthetic records/questions.",
        "explicit_shared_ontology_exceptions": {"functions": ["Find", "QueryAttr", "QueryAttrUnderCondition", "finish"],
            "attributes": ["type_description", "pulse_width", "frequency", "prf", "service_entry", "lifecycle_event", "power"],
            "qualifier_keys": ["condition_raw", "event_type"], "qualifier_labels": ["surface", "air", "Illuminator", "Tracking Radar", *EVENT_NAMES],
            "units": ["GHz", "us", "pps", "year", "W"], "reason": "Interface vocabulary and semantic labels may overlap; these are not reused equipment answers."},
        "physical_realism": "None. Arbitrary synthetic numeric fields are not plausible or verified equipment parameters.",
        "human_verified": False, "independent_equipment_gold": False, "model_inference_runs": 0, "new_training_started": False,
        "inputs_sha256": {str(item.relative_to(ROOT)): digest(item.read_bytes()) for item in (Path(__file__), Path(__file__).with_name("development_environment.py"), ORIGINAL)},
        "files_sha256": {str(item.relative_to(ROOT)): digest(value) for item, value in outputs.items()},
        "total_raw_bytes": sum(len(value) for item, value in outputs.items() if RAW in item.parents)}
    outputs[PUBLIC / "manifest.json"] = encoded(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    if args.check:
        for target, value in outputs.items():
            if target.read_bytes() != value:
                raise ValueError(f"Synthetic artifact drift: {target.relative_to(ROOT)}")
    else:
        if any(target.exists() for target in outputs):
            raise FileExistsError("Synthetic version exists; use --check or create a new version")
        for target, value in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(value)
    print(json.dumps({"mode": "verified" if args.check else "created", "groups": 100, "questions": 200,
                      "model_inference_runs": 0, "new_training_started": False}))


if __name__ == "__main__":
    main()
