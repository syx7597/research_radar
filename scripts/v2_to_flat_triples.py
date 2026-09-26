"""
v2 → flat triple adapter.

Emits graphrag_index/merged_triples_v2.json in the v1-compatible flat triple
schema, so the existing pipeline (qa_strategy_pipeline / qa_router /
HybridRetriever / qa_pipeline) can read v2 without code changes.

Conversions:
  - Each edge in radarkg_v2_edges.json → one triple (head, relation, tail)
  - Each entity property in radarkg_v2_entities.json → triples:
      str / int / float / bool   → 1 triple with tail_type=Literal
      list[primitive]            → N triples (one per element); tail_type
                                   is the list-element type if known
  - Skips: id, type, aliases, name_en, name_zh, canonical_name, entity_type
    (these are entity identifiers, not facts)

Total expected: ~6000 edges + ~3000 attribute triples = ~9000 flat triples.
"""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

# Properties that are entity metadata, not facts
SKIP_PROPS = {
    "id", "type", "aliases", "name_en", "name_zh", "canonical_name",
    "entity_type", "platform_subtype", "subsystem_type", "parent_radar",
    # Description-only fields are kept as "literal triples" so retrieval can match them
}

# Multi-valued list properties → tail_type when expanded
LIST_PROP_TAIL_TYPE = {
    "frequency_bands":   "FrequencyBand",
    "tech_types":        "TechType",
    "modes":             "RadarMode",
    "functions":         "Function",
    "cooling_methods":   "Literal",
    "subsidiaries":      "Manufacturer",
}

# Some list properties are equivalent to existing edges (e.g. frequency_bands
# is already an edge `hasFrequencyBand`). In that case we DON'T re-emit them
# as flat triples to avoid double-counting. The pipeline already gets that info
# through the edge.
PROPERTY_REDUNDANT_WITH_EDGE = {
    "frequency_bands":   "hasFrequencyBand",
    "tech_types":        "hasTechType",
    "modes":             "hasMode",
    "functions":         "hasFunction",
    "operator_primary":  "operatedBy",
    "developer":         "developedBy",
    "country_of_origin": "countryOfOrigin",
}


def main():
    in_ent  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
    in_edge = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
    out_path = ROOT / "graphrag_index" / "merged_triples_v2.json"

    print(f"Reading entities: {in_ent}")
    with open(in_ent, encoding="utf-8") as f:
        ents = json.load(f)["entities"]
    print(f"Reading edges:    {in_edge}")
    with open(in_edge, encoding="utf-8") as f:
        edges = json.load(f)["edges"]
    print(f"  {len(ents)} entities, {len(edges)} edges")

    flat_triples = []

    # 1. all edges pass through directly
    for ed in edges:
        flat_triples.append({
            "head":       ed["head"],
            "head_type":  ed.get("head_type", ""),
            "relation":   ed["relation"],
            "tail":       ed["tail"],
            "tail_type":  ed.get("tail_type", ""),
            "confidence": ed.get("confidence", 0.5),
            "evidence":   ed.get("evidence", ""),
            "source":     ed.get("source", "v2_edge"),
        })

    # 2. entity properties → literal-tail triples (skipping redundant ones)
    n_attr = 0
    skipped_redundant = 0
    for ent in ents:
        head = ent["id"]
        head_type = ent.get("type", "Entity")
        for k, v in ent.items():
            if k in SKIP_PROPS: continue
            if v is None or v == "" or v == []: continue
            if k in PROPERTY_REDUNDANT_WITH_EDGE:
                # already captured as edge in step 1
                skipped_redundant += 1
                continue

            if isinstance(v, list):
                tail_type = LIST_PROP_TAIL_TYPE.get(k, "Literal")
                for elem in v:
                    if elem is None or elem == "": continue
                    flat_triples.append({
                        "head": head, "head_type": head_type,
                        "relation": k,
                        "tail": str(elem), "tail_type": tail_type,
                        "confidence": 0.95, "evidence": f"v2 attribute {k}",
                        "source": "v2_attribute",
                    })
                    n_attr += 1
            else:
                flat_triples.append({
                    "head": head, "head_type": head_type,
                    "relation": k,
                    "tail": str(v), "tail_type": "Literal",
                    "confidence": 0.95, "evidence": f"v2 attribute {k}",
                    "source": "v2_attribute",
                })
                n_attr += 1

    print(f"\n  edges → triples:           {len(edges)}")
    print(f"  attributes → triples:      {n_attr}")
    print(f"  attributes skipped (redundant with edge): {skipped_redundant}")
    print(f"  total flat triples:        {len(flat_triples)}")

    # Save in v1-compatible schema
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flat_triples, f, ensure_ascii=False, indent=2)
    print(f"\nWritten: {out_path}")

    # Distribution stats
    print("\nRelation distribution (top 25):")
    rel_count = Counter(t["relation"] for t in flat_triples)
    for r, c in rel_count.most_common(25):
        print(f"  {r:25s} {c}")
    print(f"\n  unique relations: {len(rel_count)}")

    # Quick sanity check: did we lose any heads?
    head_set = set(t["head"] for t in flat_triples)
    ent_set = set(e["id"] for e in ents)
    missing_heads = ent_set - head_set
    print(f"\nEntities with NO outgoing triples: {len(missing_heads)}")
    if missing_heads:
        print(f"  e.g.: {list(missing_heads)[:8]}")


if __name__ == "__main__":
    main()
