"""Import the current kg_v3 files into Neo4j without losing edge records.

The current source files contain 10,744 entity records and 21,928 unique edge
records. Some frequency-band edge endpoints are absent from entities.json, so
the importer materializes them as explicitly marked KGPlaceholder nodes. This
keeps source entity counts separate from referential-completeness nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENTITIES_PATH = ROOT / "kg_v3" / "entities.json"
EDGES_PATH = ROOT / "kg_v3" / "edges.json"


def safe_identifier(value: str, prefix: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value or "")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def primitive(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return str(value)


def compact_props(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        cleaned = primitive(value)
        if cleaned is not None and cleaned != "" and cleaned != []:
            result[key] = cleaned
    return result


def load_source() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities = json.loads(ENTITIES_PATH.read_text(encoding="utf-8"))
    edges = json.loads(EDGES_PATH.read_text(encoding="utf-8"))
    if not isinstance(entities, list) or not isinstance(edges, list):
        raise ValueError("kg_v3 entities.json and edges.json must contain JSON arrays")
    return entities, edges


def prepare_graph(entities: list[dict[str, Any]], edges: list[dict[str, Any]]):
    source_nodes = []
    candidates_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    name_counts = Counter(str(entity["name"]) for entity in entities)

    for index, entity in enumerate(entities):
        record_id = f"v3e:{index:05d}"
        entity_type = str(entity.get("type") or "Entity")
        attributes = entity.get("attributes") or []
        node = {
            "record_id": record_id,
            "label": safe_identifier(entity_type, "Entity"),
            "entity_type": entity_type,
            "name": str(entity["name"]),
            "props": compact_props({
                "record_id": record_id,
                "name": str(entity["name"]),
                "entity_type": entity_type,
                "aliases": entity.get("aliases") or [],
                "domain": entity.get("domain"),
                "attributes_json": json.dumps(attributes, ensure_ascii=False, separators=(",", ":")),
                "attribute_count": len(attributes),
                "source_record": True,
                "duplicate_name_count": name_counts[str(entity["name"])],
                "kg_version": "kg_v3_current",
            }),
        }
        source_nodes.append(node)
        candidates_by_name[node["name"]].append(node)

    missing_types: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if str(edge["head"]) not in candidates_by_name:
            missing_types[str(edge["head"])].add(str(edge.get("head_type") or "Entity"))
        if str(edge["tail"]) not in candidates_by_name:
            missing_types[str(edge["tail"])].add(str(edge.get("tail_type") or "Entity"))

    placeholder_nodes = []
    placeholders_by_name = {}
    for index, name in enumerate(sorted(missing_types)):
        types = sorted(missing_types[name])
        entity_type = types[0] if len(types) == 1 else "UnresolvedEntity"
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
        record_id = f"v3p:{index:04d}:{digest}"
        node = {
            "record_id": record_id,
            "label": safe_identifier(entity_type, "Endpoint"),
            "entity_type": entity_type,
            "name": name,
            "props": compact_props({
                "record_id": record_id,
                "name": name,
                "entity_type": entity_type,
                "candidate_types": types,
                "source_record": False,
                "unresolved_endpoint": True,
                "kg_version": "kg_v3_current",
            }),
        }
        placeholder_nodes.append(node)
        placeholders_by_name[name] = node

    resolution = Counter()

    def resolve(name: str, expected_type: str) -> dict[str, Any]:
        candidates = candidates_by_name.get(name)
        if not candidates:
            resolution["placeholder"] += 1
            return placeholders_by_name[name]
        if len(candidates) == 1:
            resolution["unique_name"] += 1
            return candidates[0]
        typed = [node for node in candidates if node["entity_type"] == expected_type]
        if len(typed) == 1:
            resolution["duplicate_name_resolved_by_type"] += 1
            return typed[0]
        resolution["duplicate_name_canonicalized"] += 1
        return sorted(typed or candidates, key=lambda node: node["record_id"])[0]

    relationships = []
    triple_keys = set()
    for index, edge in enumerate(edges):
        head_name = str(edge["head"])
        tail_name = str(edge["tail"])
        relation = str(edge["relation"])
        triple_key = (head_name, relation, tail_name)
        if triple_key in triple_keys:
            raise ValueError(f"Duplicate edge triple at source index {index}: {triple_key}")
        triple_keys.add(triple_key)

        head = resolve(head_name, str(edge.get("head_type") or "Entity"))
        tail = resolve(tail_name, str(edge.get("tail_type") or "Entity"))
        edge_id = f"v3r:{index:05d}"
        relationships.append({
            "edge_id": edge_id,
            "relation": relation,
            "rel_type": safe_identifier(relation, "REL"),
            "head_id": head["record_id"],
            "tail_id": tail["record_id"],
            "props": compact_props({
                "edge_id": edge_id,
                "relation": relation,
                "head_name": head_name,
                "tail_name": tail_name,
                "head_type": edge.get("head_type"),
                "tail_type": edge.get("tail_type"),
                "evidence": edge.get("evidence") or [],
                "tier": edge.get("tier"),
                "confidence": float(edge.get("confidence") or 0.0),
                "source_kinds": edge.get("source_kinds") or [],
                "doc_ids": edge.get("doc_ids") or [],
                "corroborated": bool(edge.get("corroborated", False)),
                "relabeled_from": edge.get("_relabeled_from"),
                "kg_version": "kg_v3_current",
            }),
        })

    audit = {
        "source_entities": len(entities),
        "source_edges": len(edges),
        "unique_entity_names": len(candidates_by_name),
        "duplicate_name_groups": sum(1 for nodes in candidates_by_name.values() if len(nodes) > 1),
        "placeholder_nodes": len(placeholder_nodes),
        "neo4j_total_nodes": len(source_nodes) + len(placeholder_nodes),
        "resolution": dict(resolution),
    }
    return source_nodes, placeholder_nodes, relationships, audit


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    return text.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fieldnames) + "\n")
        for row in rows:
            handle.write("\t".join(csv_value(row.get(key)) for key in fieldnames) + "\n")


def export_load_csv_bundle(
    bundle_dir: Path,
    source_nodes: list[dict[str, Any]],
    placeholder_nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    batch_size: int,
) -> Path:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    node_fields = [
        "record_id", "label", "name", "entity_type", "aliases_json", "domain",
        "attributes_json", "attribute_count", "source_record", "duplicate_name_count",
        "candidate_types_json", "unresolved_endpoint", "kg_version",
    ]
    source_rows = []
    for node in source_nodes:
        props = node["props"]
        source_rows.append({
            **props,
            "label": node["label"],
            "aliases_json": json.dumps(props.get("aliases", []), ensure_ascii=False),
        })
    placeholder_rows = []
    for node in placeholder_nodes:
        props = node["props"]
        placeholder_rows.append({
            **props,
            "label": node["label"],
            "candidate_types_json": json.dumps(
                props.get("candidate_types", []), ensure_ascii=False
            ),
        })
    write_csv(bundle_dir / "source_nodes.csv", node_fields, source_rows)
    write_csv(bundle_dir / "placeholder_nodes.csv", node_fields, placeholder_rows)

    rel_fields = [
        "edge_id", "rel_type", "relation", "head_id", "tail_id", "head_name",
        "tail_name", "head_type", "tail_type", "evidence_json", "tier", "confidence",
        "source_kinds_json", "doc_ids_json", "corroborated", "relabeled_from", "kg_version",
    ]
    rel_rows = []
    for rel in relationships:
        props = rel["props"]
        rel_rows.append({
            **props,
            "rel_type": rel["rel_type"],
            "head_id": rel["head_id"],
            "tail_id": rel["tail_id"],
            "evidence_json": json.dumps(props.get("evidence", []), ensure_ascii=False),
            "source_kinds_json": json.dumps(props.get("source_kinds", []), ensure_ascii=False),
            "doc_ids_json": json.dumps(props.get("doc_ids", []), ensure_ascii=False),
        })
    write_csv(bundle_dir / "relationships.csv", rel_fields, rel_rows)

    source_path = str(ENTITIES_PATH).replace("\\", "\\\\").replace("'", "\\'")
    edge_path = str(EDGES_PATH).replace("\\", "\\\\").replace("'", "\\'")
    imported_at = datetime.now(timezone.utc).isoformat()
    statements = f"""
MATCH (n) DETACH DELETE n;
DROP CONSTRAINT constraint_19392f4 IF EXISTS;
DROP CONSTRAINT constraint_1c224eb6 IF EXISTS;
DROP CONSTRAINT constraint_1ed05907 IF EXISTS;
DROP CONSTRAINT constraint_342f0343 IF EXISTS;
DROP CONSTRAINT constraint_6958ccf6 IF EXISTS;
DROP CONSTRAINT constraint_77b6241c IF EXISTS;
DROP CONSTRAINT constraint_7c52de27 IF EXISTS;
DROP CONSTRAINT constraint_8979eba8 IF EXISTS;
DROP CONSTRAINT constraint_8b8cdf19 IF EXISTS;
DROP CONSTRAINT constraint_9382f685 IF EXISTS;
DROP CONSTRAINT constraint_97a885ff IF EXISTS;
DROP CONSTRAINT constraint_a0148cca IF EXISTS;
DROP CONSTRAINT constraint_afa83276 IF EXISTS;
DROP CONSTRAINT constraint_b6f2e6c0 IF EXISTS;
DROP CONSTRAINT constraint_ce358054 IF EXISTS;
DROP CONSTRAINT constraint_de87de21 IF EXISTS;
DROP CONSTRAINT constraint_f850a545 IF EXISTS;
CREATE CONSTRAINT kg_node_record_id IF NOT EXISTS
FOR (n:KGNode) REQUIRE n.record_id IS UNIQUE;
CREATE INDEX kg_node_name IF NOT EXISTS FOR (n:KGNode) ON (n.name);
CREATE INDEX kg_node_type IF NOT EXISTS FOR (n:KGNode) ON (n.entity_type);

LOAD CSV WITH HEADERS FROM 'file:///kg_v3_current/source_nodes.csv' AS row
FIELDTERMINATOR '\\t'
CALL (row) {{
  CREATE (n:KGNode:KGEntity)
  SET n:$(row.label),
      n.record_id = row.record_id,
      n.name = row.name,
      n.entity_type = row.entity_type,
      n.aliases_json = row.aliases_json,
      n.domain = CASE row.domain WHEN '' THEN null ELSE row.domain END,
      n.attributes_json = row.attributes_json,
      n.attribute_count = toInteger(row.attribute_count),
      n.source_record = true,
      n.duplicate_name_count = toInteger(row.duplicate_name_count),
      n.kg_version = row.kg_version
}} IN TRANSACTIONS OF {batch_size} ROWS;

LOAD CSV WITH HEADERS FROM 'file:///kg_v3_current/placeholder_nodes.csv' AS row
FIELDTERMINATOR '\\t'
CALL (row) {{
  CREATE (n:KGNode:KGPlaceholder)
  SET n:$(row.label),
      n.record_id = row.record_id,
      n.name = row.name,
      n.entity_type = row.entity_type,
      n.candidate_types_json = row.candidate_types_json,
      n.source_record = false,
      n.unresolved_endpoint = true,
      n.kg_version = row.kg_version
}} IN TRANSACTIONS OF {batch_size} ROWS;

LOAD CSV WITH HEADERS FROM 'file:///kg_v3_current/relationships.csv' AS row
FIELDTERMINATOR '\\t'
CALL (row) {{
  MATCH (h:KGNode {{record_id: row.head_id}})
  MATCH (t:KGNode {{record_id: row.tail_id}})
  CREATE (h)-[r:$(row.rel_type)]->(t)
  SET r.edge_id = row.edge_id,
      r.relation = row.relation,
      r.head_name = row.head_name,
      r.tail_name = row.tail_name,
      r.head_type = row.head_type,
      r.tail_type = row.tail_type,
      r.evidence_json = row.evidence_json,
      r.tier = row.tier,
      r.confidence = toFloat(row.confidence),
      r.source_kinds_json = row.source_kinds_json,
      r.doc_ids_json = row.doc_ids_json,
      r.corroborated = row.corroborated = 'true',
      r.relabeled_from = CASE row.relabeled_from WHEN '' THEN null ELSE row.relabeled_from END,
      r.kg_version = row.kg_version
}} IN TRANSACTIONS OF {batch_size} ROWS;

CREATE (:KGImport {{
  version: 'kg_v3_current',
  imported_at: '{imported_at}',
  entities_path: '{source_path}',
  edges_path: '{edge_path}'
}});

MATCH (e:KGEntity) WITH count(e) AS source_entities
MATCH (p:KGPlaceholder) WITH source_entities, count(p) AS placeholders
MATCH (n) WITH source_entities, placeholders, count(n) AS total_nodes
MATCH ()-[r]->()
RETURN source_entities, placeholders, total_nodes, count(r) AS relationships;
""".strip()
    cypher_path = bundle_dir / "import.cypher"
    cypher_path.write_text(statements + "\n", encoding="utf-8")
    return cypher_path


def run_cypher_shell(args, cypher_path: Path) -> None:
    command = [
        str(args.cypher_shell),
        "-a", args.uri,
        "-u", args.user,
        "-p", args.password,
        "-d", args.database,
        "--format", "plain",
        "--fail-fast",
        "-f", str(cypher_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "neo4jneo4j"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--neo4j-home",
        type=Path,
        default=Path(r"C:\neo4j\neo4j-community-2025.03.0"),
    )
    parser.add_argument("--cypher-shell", type=Path)
    parser.add_argument("--reset", action="store_true", help="Required for destructive import")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entities, edges = load_source()
    source_nodes, placeholders, relationships, audit = prepare_graph(entities, edges)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if args.dry_run:
        return
    if not args.reset:
        raise SystemExit("Refusing to modify Neo4j without --reset")

    args.cypher_shell = args.cypher_shell or args.neo4j_home / "bin" / "cypher-shell.bat"
    staging_dir = ROOT / "reports" / "neo4j_import" / "kg_v3_current"
    cypher_path = export_load_csv_bundle(
        staging_dir, source_nodes, placeholders, relationships, args.batch_size
    )
    neo4j_bundle = args.neo4j_home / "import" / "kg_v3_current"
    if neo4j_bundle.exists():
        shutil.rmtree(neo4j_bundle)
    shutil.copytree(staging_dir, neo4j_bundle)
    print(f"Prepared LOAD CSV bundle: {neo4j_bundle}")
    run_cypher_shell(args, neo4j_bundle / cypher_path.name)


if __name__ == "__main__":
    main()
