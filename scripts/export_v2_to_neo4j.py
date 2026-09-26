"""
Export v2 KG (entities with properties + edges) to Neo4j.

Reads:
  data/v2/radarkg_v2_entities.json
  data/v2/radarkg_v2_edges.json

Writes to Neo4j:
  - Nodes labeled by entity type (Radar / Manufacturer / Country / Platform / ...)
  - All entity properties become Neo4j node properties (numeric floats / strings / lists)
  - All edges become typed relationships
  - Original `id` becomes the Neo4j unique key

Run:
  python scripts/export_v2_to_neo4j.py [--reset]
  python scripts/export_v2_to_neo4j.py --uri bolt://localhost:7687 --user neo4j --password ...

Auth via env vars: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Map v2 entity types → Neo4j labels (multi-label supported via "Type1:Type2")
TYPE_TO_LABEL = {
    "Radar":            "Radar",
    "RadarSystem":      "Radar",        # collapse into one label, keep type as property
    "Manufacturer":     "Company",
    "Country":          "Country",
    "Platform":         "Platform",
    "NavalVessel":      "Platform",     # subtypes folded with `platform_subtype` property
    "AircraftPlatform": "Platform",
    "GroundPlatform":   "Platform",
    "Aircraft":         "Platform",
    "FrequencyBand":    "FrequencyBand",
    "Function":         "Function",
    "Weapon":           "Weapon",
    "Missile":          "Weapon",
    "RadarMode":        "RadarMode",
    "TechType":         "TechType",
    "Organization":     "Organization",
    # v2 new types
    "Subsystem":        "Subsystem",
    "Component":        "Component",
    "Standard":         "Standard",
    "Conflict":         "Conflict",
    "Location":         "Location",
    "Range":            "Range",
}

PLATFORM_SUBTYPES = {
    "NavalVessel": "naval", "AircraftPlatform": "aircraft",
    "GroundPlatform": "ground", "Aircraft": "aircraft",
}


def get_label(entity_type: str) -> str:
    return TYPE_TO_LABEL.get(entity_type, "Entity")


def serialize_value(v):
    """Neo4j supports str/int/float/bool/list of those. Skip None and complex objects."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        # only flat lists of primitives
        clean = [x for x in v if isinstance(x, (str, int, float, bool))]
        return clean
    return str(v)  # fallback


def serialize_props(ent: dict) -> dict:
    """Strip non-Neo4j-compatible fields and serialize."""
    drop_keys = {"type", "id"}
    out = {}
    for k, v in ent.items():
        if k in drop_keys: continue
        sv = serialize_value(v)
        if sv is not None and sv != "":
            out[k] = sv
    out["entity_type"] = ent.get("type", "Entity")
    out["id"] = ent.get("id", "")
    # Set `name` so Neo4j Browser picks it as the default caption.
    # Prefer name_zh > name_en > id (in that order, whichever is non-empty).
    out["name"] = (
        ent.get("name_en") or ent.get("name_zh") or ent.get("id", "")
    )
    # Add platform subtype if applicable
    sub = PLATFORM_SUBTYPES.get(ent.get("type", ""))
    if sub:
        out["platform_subtype"] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri",      default=os.getenv("NEO4J_URI",      "bolt://localhost:7687"))
    ap.add_argument("--user",     default=os.getenv("NEO4J_USER",     "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "neo4jneo4j"))
    ap.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    ap.add_argument("--reset",    action="store_true",
                    help="MATCH (n) DETACH DELETE n before import")
    ap.add_argument("--batch",    type=int, default=200)
    args = ap.parse_args()

    print(f"Loading v2 KG…")
    with open(ROOT / "data" / "v2" / "radarkg_v2_entities.json", encoding="utf-8") as f:
        entities = json.load(f)["entities"]
    with open(ROOT / "data" / "v2" / "radarkg_v2_edges.json", encoding="utf-8") as f:
        edges = json.load(f)["edges"]
    print(f"  {len(entities)} entities, {len(edges)} edges")

    # Distribution check
    from collections import Counter
    print(f"  entity types: {dict(Counter(e['type'] for e in entities).most_common())}")
    print(f"  edge relations: {len(set(e['relation'] for e in edges))} types")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("\nERROR: pip install neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session(database=args.database) as session:
            if args.reset:
                print("\nDropping existing data…")
                session.run("MATCH (n) DETACH DELETE n")
                # Also drop any pre-existing constraints/indexes left over from
                # earlier imports — they may bind unique on properties (e.g. `name`)
                # that conflict with this schema.
                print("Dropping existing constraints/indexes…")
                cons = list(session.run("SHOW CONSTRAINTS YIELD name").data())
                for row in cons:
                    try:
                        session.run(f"DROP CONSTRAINT {row['name']}")
                    except Exception as ex:
                        print(f"  ! drop constraint {row['name']} failed: {ex}")
                idxs = list(session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").data())
                for row in idxs:
                    try:
                        session.run(f"DROP INDEX {row['name']}")
                    except Exception as ex:
                        print(f"  ! drop index {row['name']} failed: {ex}")

            # Constraints (one per label)
            print("\nCreating unique constraints on `id`…")
            labels_used = sorted({get_label(e["type"]) for e in entities})
            for lbl in labels_used + ["Entity"]:
                try:
                    session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.id IS UNIQUE"
                    )
                except Exception as ex:
                    print(f"  ! constraint on {lbl} skipped: {ex}")

            # Bulk insert nodes
            print(f"\nMerging {len(entities)} nodes (batch={args.batch})…")
            for i in range(0, len(entities), args.batch):
                batch = entities[i:i+args.batch]
                with session.begin_transaction() as tx:
                    for e in batch:
                        lbl = get_label(e["type"])
                        props = serialize_props(e)
                        tx.run(
                            f"MERGE (n:{lbl} {{id:$id}}) SET n += $props",
                            id=e["id"], props=props,
                        )
                if (i // args.batch + 1) % 5 == 0 or i + args.batch >= len(entities):
                    print(f"  {min(i+args.batch, len(entities)):>5d}/{len(entities)}")

            # Bulk insert relationships — group by relation type for typed RELS
            print(f"\nCreating {len(edges)} relationships…")
            from collections import defaultdict
            edges_by_rel = defaultdict(list)
            for ed in edges:
                edges_by_rel[ed["relation"]].append(ed)

            for rel_type, rel_batch in edges_by_rel.items():
                # neo4j relationship types must be valid identifiers; replace any non-alnum with _
                rel_safe = rel_type if rel_type.isidentifier() else "REL_" + "".join(
                    c if c.isalnum() else "_" for c in rel_type)
                for i in range(0, len(rel_batch), args.batch):
                    batch = rel_batch[i:i+args.batch]
                    with session.begin_transaction() as tx:
                        for ed in batch:
                            head_lbl = get_label(ed.get("head_type", ""))
                            tail_lbl = get_label(ed.get("tail_type", ""))
                            edge_props = {
                                "evidence":   ed.get("evidence", "")[:300],
                                "confidence": float(ed.get("confidence", 0.0) or 0.0),
                                "source":     ed.get("source", ""),
                            }
                            tx.run(
                                f"""
                                MATCH (h {{id:$h_id}})
                                MATCH (t {{id:$t_id}})
                                MERGE (h)-[r:{rel_safe}]->(t)
                                SET r += $props
                                """,
                                h_id=ed["head"], t_id=ed["tail"], props=edge_props,
                            )
                print(f"  [{rel_type:25s}] {len(rel_batch)} edges loaded")

            # Verify
            print("\nFinal Neo4j state:")
            n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            n_rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"  nodes: {n_nodes}, relationships: {n_rels}")
    finally:
        driver.close()

    print("\n✓ Done. Open Neo4j Browser:")
    print(f"  URI: {args.uri}")
    print(f"  Database: {args.database}")
    print("  Try: MATCH (r:Radar)-[:hasSubsystem]->(s) RETURN r,s LIMIT 50")


if __name__ == "__main__":
    main()
