"""
v2 Stage C integration: merge raw LLM extractions into the v2 KG.

Reads:  data/v2/stage_c_raw_extractions.json  (output of stage_c_extract.py)
Writes: data/v2/radarkg_v2_entities.json     (in-place, adds new Subsystem/Component/Conflict/Standard entities)
        data/v2/radarkg_v2_edges.json        (in-place, appends new edges)
        data/v2/stage_c_report.md            (statistics + sample results)

Steps:
  1. Group extractions by relation type
  2. For hasSubsystem: tail values get standardized via VALID_SUBSYSTEMS map; create radar-scoped Subsystem entities (e.g., "AN/APG-77:transmitter") so two radars don't share the same generic "transmitter" node
  3. For hasComponent: similar but with the parent Subsystem if context allows; otherwise tied to radar
  4. For meetsStandard: dedupe to existing Standard entities created in Stage B; new ones get added
  5. For usedIn: create Conflict entities (with a normalization map for common wars) and link
  6. Filter: confidence ≥ 0.5; reject obvious extraction noise (very short tails, single chars)
  7. Dedupe: (head, relation, tail) tuples
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


# Conflict normalization map (Chinese ↔ English ↔ canonical)
CONFLICT_CANON = {
    "海湾战争": "Gulf War 1991",
    "海湾战争1991": "Gulf War 1991",
    "Gulf War": "Gulf War 1991",
    "Gulf War 1991": "Gulf War 1991",
    "越南战争": "Vietnam War",
    "越战": "Vietnam War",
    "Vietnam War": "Vietnam War",
    "Vietnam": "Vietnam War",
    "伊拉克战争": "Iraq War 2003",
    "伊拉克": "Iraq War 2003",
    "Iraq War": "Iraq War 2003",
    "Iraq War 2003": "Iraq War 2003",
    "阿富汗战争": "War in Afghanistan",
    "阿富汗": "War in Afghanistan",
    "Afghanistan": "War in Afghanistan",
    "马岛战争": "Falklands War",
    "福克兰群岛战争": "Falklands War",
    "福克兰战争": "Falklands War",
    "Falklands War": "Falklands War",
    "利比亚": "Libya operations",
    "Operation Iraqi Freedom": "Iraq War 2003",
    "Operation Enduring Freedom": "War in Afghanistan",
    "Korean War": "Korean War",
    "朝鲜战争": "Korean War",
    "二战": "World War II",
    "World War II": "World War II",
}

# Standard normalization (handle MIL-STD-1553A → MIL-STD-1553 etc.)
STANDARD_CANON = {
    "MIL-STD-1553": "MIL-STD-1553",
    "MIL-STD-1553A": "MIL-STD-1553",
    "MIL-STD-1553B": "MIL-STD-1553",
    "1553": "MIL-STD-1553",
    "1553A": "MIL-STD-1553",
    "1553B": "MIL-STD-1553",
    "MIL-STD-1750": "MIL-STD-1750A",
    "MIL-STD-1750A": "MIL-STD-1750A",
    "MIL-STD-1760": "MIL-STD-1760",
    "1760": "MIL-STD-1760",
    "ARINC-429": "ARINC-429",
    "ARINC429": "ARINC-429",
    "ARINC 429": "ARINC-429",
    "ARINC-664": "ARINC-664",
    "ARINC-664P7": "ARINC-664",
    "AFDX": "AFDX",
    "RS-422": "RS-422",
    "RS422": "RS-422",
    "RS-485": "RS-485",
    "RS485": "RS-485",
}

COMPONENT_CANON = {
    "TWT": "Traveling-Wave Tube",
    "Traveling-Wave Tube": "Traveling-Wave Tube",
    "Traveling Wave Tube": "Traveling-Wave Tube",
    "行波管": "Traveling-Wave Tube",
    "MMIC": "MMIC",
    "T/R module": "T/R Module",
    "T/R-module": "T/R Module",
    "T/R Module": "T/R Module",
    "T/R 模块": "T/R Module",
    "T/R模块": "T/R Module",
    "phase shifter": "Phase Shifter",
    "phase-shifter": "Phase Shifter",
    "移相器": "Phase Shifter",
    "magnetron": "Magnetron",
    "Magnetron": "Magnetron",
    "磁控管": "Magnetron",
    "klystron": "Klystron",
    "Klystron": "Klystron",
    "速调管": "Klystron",
    "巴克码相位调制器": "Barker Code Phase Modulator",
}


def main():
    raw_path = ROOT / "data" / "v2" / "stage_c_raw_extractions.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run stage_c_extract.py first.")
        sys.exit(1)
    with open(raw_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    extractions = raw_data["extractions"]
    print(f"Raw extractions: {len(extractions)}")

    ent_path  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
    edge_path = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
    with open(ent_path, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(edge_path, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = {e["id"]: e for e in ent_data["entities"]}
    edges    = edge_data["edges"]
    print(f"Current v2 KG: {len(entities)} entities, {len(edges)} edges")

    # ────────  Filter & normalize
    new_edges = []
    new_entities_added = []
    rejected = Counter()
    relation_counters = Counter()

    # Seed seen_keys with existing edges so re-running this script is idempotent
    seen_keys: set = {(e["head"], e["relation"], e["tail"]) for e in edges}
    print(f"  pre-seeded with {len(seen_keys)} existing edge keys (idempotency guard)")

    for ext in extractions:
        rel  = ext["relation"]
        head = ext["head"]
        tail = ext["tail"].strip()
        conf = float(ext.get("confidence", 0.7))

        # Reject low-confidence
        if conf < 0.5:
            rejected["low_conf"] += 1; continue
        # Reject very short
        if len(tail) < 2 or tail in {"无", "—", "-", "(空)", "未知"}:
            rejected["empty_tail"] += 1; continue

        # Normalize tail per relation type
        if rel == "hasSubsystem":
            # ID it as radar-scoped (e.g., "AN/APG-77:transmitter") to avoid conflating
            # subsystems across radars
            sub_id = f"{head}:{tail}"
            sub_label = {
                "transmitter": "发射机", "receiver": "接收机",
                "antenna": "天线", "signal_processor": "信号处理机",
                "display_control": "显控", "power_supply": "电源",
                "cooling": "冷却系统",
            }.get(tail, tail)
            if sub_id not in entities:
                entities[sub_id] = {
                    "id": sub_id, "type": "Subsystem",
                    "name_en": tail, "name_zh": sub_label,
                    "subsystem_type": tail, "parent_radar": head,
                    "aliases": [],
                }
                new_entities_added.append(sub_id)
            tail_for_edge = sub_id
            tail_type = "Subsystem"

        elif rel == "hasComponent":
            canonical = COMPONENT_CANON.get(tail, tail)
            comp_id = canonical
            if comp_id not in entities:
                entities[comp_id] = {
                    "id": comp_id, "type": "Component",
                    "name_en": canonical, "aliases": [tail] if tail != canonical else [],
                }
                new_entities_added.append(comp_id)
            tail_for_edge = comp_id
            tail_type = "Component"

        elif rel == "meetsStandard":
            canonical = STANDARD_CANON.get(tail, tail)
            std_id = canonical
            if std_id not in entities:
                entities[std_id] = {
                    "id": std_id, "type": "Standard",
                    "name_en": canonical,
                    "aliases": [tail] if tail != canonical else [],
                    "category": "data_bus" if "STD" in canonical or "ARINC" in canonical else "",
                }
                new_entities_added.append(std_id)
            tail_for_edge = std_id
            tail_type = "Standard"

        elif rel == "usedIn":
            canonical = CONFLICT_CANON.get(tail, tail)
            conf_id = canonical
            if conf_id not in entities:
                entities[conf_id] = {
                    "id": conf_id, "type": "Conflict",
                    "name_en": canonical,
                    "aliases": [tail] if tail != canonical else [],
                }
                new_entities_added.append(conf_id)
            tail_for_edge = conf_id
            tail_type = "Conflict"

        else:
            rejected["unknown_rel"] += 1; continue

        # Dedupe
        key = (head, rel, tail_for_edge)
        if key in seen_keys:
            rejected["dup"] += 1; continue
        seen_keys.add(key)

        new_edges.append({
            "head":       head,
            "relation":   rel,
            "tail":       tail_for_edge,
            "head_type":  "Radar",
            "tail_type":  tail_type,
            "confidence": conf,
            "evidence":   ext.get("evidence", ""),
            "source":     "stage_c_llm",
            "source_pages": ext.get("source_pages", []),
        })
        relation_counters[rel] += 1

    # ────────  Save
    edges.extend(new_edges)
    ent_data["entities"] = list(entities.values())
    ent_data["count"] = len(entities)
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)
    with open(ent_path, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)

    # ────────  Report
    rep_path = ROOT / "data" / "v2" / "stage_c_report.md"
    new_ent_types = Counter(entities[eid]["type"] for eid in new_entities_added)
    sample_edges_per_rel = {}
    for r in relation_counters:
        sample_edges_per_rel[r] = [e for e in new_edges if e["relation"] == r][:5]

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"""# v2 Stage C Extraction Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Source**: PDF narrative re-extraction targeting v2 schema

## Summary

| Metric | Before | Added | After |
|---|---|---|---|
| Entities | {len(entities) - len(new_entities_added)} | +{len(new_entities_added)} | {len(entities)} |
| Edges    | {len(edges) - len(new_edges)} | +{len(new_edges)} | {len(edges)} |

## New edges by relation

{chr(10).join(f"- `{r}`: {c}" for r, c in relation_counters.most_common())}

## New entities by type

{chr(10).join(f"- {t}: {c}" for t, c in new_ent_types.most_common())}

## Rejected extractions

{chr(10).join(f"- {reason}: {c}" for reason, c in rejected.most_common())}

## Sample new edges per relation

""")
        for r, samples in sample_edges_per_rel.items():
            f.write(f"### {r}\n\n")
            for e in samples:
                f.write(f"- `{e['head']}` -[{e['relation']}]→ `{e['tail']}`  (conf={e['confidence']:.2f}, ev=\"{e['evidence'][:80]}\")\n")
            f.write("\n")

    # ────────  Stdout summary
    print(f"\n────── stage C integration done ──────")
    print(f"new edges:    +{len(new_edges)}")
    print(f"new entities: +{len(new_entities_added)}")
    print(f"rejected:     {dict(rejected)}")
    print()
    print("New edges by relation:")
    for r, c in relation_counters.most_common():
        print(f"  {r:18s} +{c}")
    print()
    print("New entity types:")
    for t, c in new_ent_types.most_common():
        print(f"  {t:15s} +{c}")
    print(f"\nReport: {rep_path}")
    print(f"Final v2 KG: {len(entities)} entities, {len(edges)} edges")


if __name__ == "__main__":
    main()
