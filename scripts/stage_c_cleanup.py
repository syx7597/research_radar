"""
Post-integration cleanup for Stage C extractions.

Removes:
  - usedIn edges where evidence contains hedging language ("未明确" / "未提及" /
    "可能参战" / "无实战" / "仅描述研发")
  - Orphan Conflict / Standard / Component / Subsystem entities (no edges referencing them)

Idempotent: safe to re-run after each Stage C extraction.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BAD_USED_IN_PATTERNS = [
    "未明确", "未提及", "仅描述", "可能参战", "无实战", "未实战", "无相关",
    "未直接", "未表明", "无直接",
]


def main():
    ent_path  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
    edge_path = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
    with open(edge_path, encoding="utf-8") as f:
        edge_data = json.load(f)
    with open(ent_path, encoding="utf-8") as f:
        ent_data = json.load(f)

    edges = edge_data["edges"]
    n_before = len(edges)

    # 1. Remove over-reaching usedIn
    removed = []
    clean_edges = []
    for e in edges:
        if e["relation"] == "usedIn" and any(p in e.get("evidence", "")
                                                for p in BAD_USED_IN_PATTERNS):
            removed.append(e)
            continue
        clean_edges.append(e)
    print(f"Removed {len(removed)} hedged usedIn edges")
    for r in removed[:10]:
        print(f"  {r['head']:30s} -> {r['tail']:25s}  ev: {r['evidence'][:60]}")

    # 2. Drop orphan Conflict/Component/Subsystem/Standard entities
    referenced = set()
    for e in clean_edges:
        referenced.add(e["head"]); referenced.add(e["tail"])

    entities = ent_data["entities"]
    n_ent_before = len(entities)
    cleaned_ents = []
    orphans = {"Conflict": [], "Component": [], "Subsystem": [], "Standard": []}
    for ent in entities:
        if ent["type"] in orphans and ent["id"] not in referenced:
            orphans[ent["type"]].append(ent["id"])
            continue
        cleaned_ents.append(ent)

    print(f"\nOrphan entities removed:")
    for typ, ids in orphans.items():
        if ids:
            print(f"  {typ}: {len(ids)}  e.g., {ids[:3]}")

    # Save
    edge_data["edges"] = clean_edges
    edge_data["count"] = len(clean_edges)
    ent_data["entities"] = cleaned_ents
    ent_data["count"] = len(cleaned_ents)
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ent_path, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    print(f"\nFinal v2 KG: {len(cleaned_ents)} entities, {len(clean_edges)} edges")
    print(f"  (was {n_ent_before} entities, {n_before} edges before cleanup)")


if __name__ == "__main__":
    main()
