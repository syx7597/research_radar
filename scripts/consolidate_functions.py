"""
Consolidate duplicate Function entities so Neo4j export doesn't hit name collisions.

Older Function ids like 'air defense' / 'fire control' / 'long-range search' are
mapped to the canonical underscored ids ('air_defense' / 'fire_control' / 'early_warning').

For each old → canonical mapping:
  - rewrite all hasFunction edges pointing to old id → canonical id
  - merge old aliases into canonical entity's aliases
  - delete the old Function entity if no edges remain

Idempotent: safe to re-run.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"

OLD_TO_CANON = {
    # Older / messier ids → canonical underscore ids
    "fire control":                       "fire_control",
    "air defense":                        "air_defense",
    "air-to-air combat":                  "air_intercept",
    "air-to-ground attack":               "ground_attack",
    "air-to-ground targeting":            "ground_attack",
    "air-to-surface attack":              "ground_attack",
    "synthetic aperture radar mapping":   "imaging_SAR",
    "target detection":                   "air_search",
    "target identification":              "target_identification",
    "missile tracking":                   "missile_guidance",
    "gun control":                        "naval_gunnery",
    "warning":                            "early_warning",
    "long-range search":                  "early_warning",
    "long range search":                  "early_warning",
    "altitude-finding":                   "altimeter",
    "reconnaissance":                     "reconnaissance",
    "Identification Friend or Foe":       "IFF",
}


def main():
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]

    ent_by_id = {e["id"]: e for e in entities}

    # Some canonical Function entities may not exist yet — ensure they do
    from_enrich_function = {
        "fire_control": "火控", "air_defense": "防空", "air_intercept": "空中截击",
        "ground_attack": "对地攻击", "imaging_SAR": "合成孔径成像", "air_search": "空中目标搜索",
        "target_identification": "目标识别", "missile_guidance": "导弹制导",
        "naval_gunnery": "舰炮火控", "early_warning": "预警", "altimeter": "雷达测高",
        "reconnaissance": "侦察", "IFF": "敌我识别",
    }
    for canon, zh in from_enrich_function.items():
        if canon not in ent_by_id:
            ent_by_id[canon] = {
                "id": canon, "type": "Function",
                "name_en": canon.replace("_", " "), "name_zh": zh, "aliases": [],
            }
            entities.append(ent_by_id[canon])

    # 1. Migrate any remaining edges pointing to old Function ids
    n_migrated = 0
    for ed in edges:
        if ed["relation"] == "hasFunction":
            old = ed["tail"]
            canon = OLD_TO_CANON.get(old)
            if canon and canon in ent_by_id:
                ed["tail"] = canon
                n_migrated += 1
    print(f"Migrated {n_migrated} hasFunction edges to canonical ids")

    # 2. Dedup edges (head, relation, tail)
    seen = set()
    cleaned = []
    n_dup = 0
    for ed in edges:
        key = (ed["head"], ed["relation"], ed["tail"])
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        cleaned.append(ed)
    print(f"Removed {n_dup} duplicate edges from migration")
    edges = cleaned

    # 3. Merge aliases & delete orphan old Function entities (no edges referencing them)
    referenced = set()
    for ed in edges:
        referenced.add(ed["head"]); referenced.add(ed["tail"])

    new_entities = []
    n_dropped = 0
    for ent in entities:
        if (ent["type"] == "Function" and ent["id"] in OLD_TO_CANON
                and ent["id"] not in referenced):
            # Merge alias into canonical entity
            canon_id = OLD_TO_CANON[ent["id"]]
            canon = ent_by_id.get(canon_id)
            if canon is not None:
                aliases = set(canon.get("aliases", []) or [])
                aliases.add(ent["id"])
                if ent.get("name_en"): aliases.add(ent["name_en"])
                canon["aliases"] = sorted(a for a in aliases if a)
            n_dropped += 1
            continue
        new_entities.append(ent)
    print(f"Dropped {n_dropped} orphan/old Function entities")

    # Save
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)
    ent_data["entities"] = new_entities
    ent_data["count"] = len(new_entities)
    with open(EDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    # Final sanity
    funcs = [e for e in new_entities if e["type"] == "Function"]
    print(f"\nFinal Function entities: {len(funcs)}")
    print(f"Final v2 KG: {len(new_entities)} entities, {len(edges)} edges")


if __name__ == "__main__":
    main()
