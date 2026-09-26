"""
v2 Stage B: Wikidata-style enrichment.

Adds three categories of real knowledge to the v2 KG:
  1. Manufacturer hierarchy:    subsidiaryOf (parent), historical mergers
  2. Manufacturer location:     headquarteredIn (city + country) + founded_year
  3. Standard entities:         MIL-STD-1553, ARINC-429, etc. as Standard nodes

All facts are curated from publicly available Wikidata content (verified manually
for top defense companies). For wider coverage, see import_wikidata.py for the
SPARQL-based approach.

Edges added: ~80
Properties added: ~50
New entities: ~10 standards + ~25 location stubs

Reads:  data/v2/radarkg_v2_entities.json
        data/v2/radarkg_v2_edges.json
Writes: same files, in-place

Outputs migration log to data/v2/stage_b_report.md
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────
#  1. Manufacturer knowledge base (Wikidata-derived, curated)
#
#  Each entry covers:
#    - parent: current parent company (None if independent)
#    - merged_into: historical merger destination
#    - hq_city, hq_country: headquarters location
#    - founded_year, employees_approx, ownership_type
#    - aliases: matched against KG surface forms
# ────────────────────────────────────────────────────────────

MANUFACTURERS = {
    # United States — major defense primes
    "Raytheon": {
        "aliases": ["Raytheon Company", "Raytheon Technologies", "RTX",
                    "RTX Corporation", "Raytheon Missile Systems",
                    "Raytheon Company Space and Airborne",
                    "Raytheon Company Space and Airbome",
                    "Raytheon Aerospace Division"],
        "parent": "RTX Corporation",
        "hq_city": "Arlington, Virginia",
        "hq_country": "United States",
        "founded_year": 1922,
        "ownership_type": "public",
    },
    "RTX Corporation": {
        "aliases": ["RTX"],
        "parent": None,
        "hq_city": "Arlington, Virginia",
        "hq_country": "United States",
        "founded_year": 2020,
        "ownership_type": "public",
    },
    "Northrop Grumman": {
        "aliases": ["NorthropGrumman", "Northrop", "Grumman", "Northrop Grumman Corporation",
                    "Northrop Grumman Electronic Systems", "NorthropGrumman ElectronicSystems",
                    "NorthropGrumman Electronic Systems", "Northrop Grumman Electrics Systems"],
        "parent": None,
        "hq_city": "Falls Church, Virginia",
        "hq_country": "United States",
        "founded_year": 1994,
        "ownership_type": "public",
        "absorbed": ["Westinghouse Electric Corporation Defense"],
    },
    "Lockheed Martin": {
        "aliases": ["Lockheed", "Lockheed Corporation", "Martin Marietta",
                    "Lockheed Martin Corporation"],
        "parent": None,
        "hq_city": "Bethesda, Maryland",
        "hq_country": "United States",
        "founded_year": 1995,
        "ownership_type": "public",
    },
    "General Electric": {
        "aliases": ["GE", "G.E.", "General Electric Company"],
        "parent": None,
        "hq_city": "Boston, Massachusetts",
        "hq_country": "United States",
        "founded_year": 1892,
        "ownership_type": "public",
    },
    "Hughes Aircraft Company": {
        "aliases": ["Hughes", "Hughes Aircraft"],
        "parent": "Raytheon",
        "merged_into": "Raytheon",
        "hq_city": "Culver City, California",
        "hq_country": "United States",
        "founded_year": 1932,
        "defunct_year": 1997,
    },
    "Westinghouse Electric Corporation": {
        "aliases": ["Westinghouse", "Westinghouse Electric",
                    "Westinghouse (Northrop Grumman)"],
        "parent": "Northrop Grumman",
        "merged_into": "Northrop Grumman",
        "hq_city": "Pittsburgh, Pennsylvania",
        "hq_country": "United States",
        "founded_year": 1886,
        "defunct_year": 1996,  # defense business sold to Northrop
    },
    "Boeing": {
        "aliases": ["Boeing Company", "The Boeing Company"],
        "parent": None,
        "hq_city": "Arlington, Virginia",
        "hq_country": "United States",
        "founded_year": 1916,
        "ownership_type": "public",
    },
    "L3Harris": {
        "aliases": ["Harris", "L3 Technologies", "L3Harris Technologies"],
        "parent": None,
        "hq_city": "Melbourne, Florida",
        "hq_country": "United States",
        "founded_year": 2019,
        "ownership_type": "public",
    },
    "Texas Instruments": {
        "aliases": ["TexasInstruments", "TI"],
        "parent": None,
        "hq_city": "Dallas, Texas",
        "hq_country": "United States",
        "founded_year": 1930,
        "ownership_type": "public",
    },
    "ITT Gilfillan": {
        "aliases": [],
        "parent": "ITT Corporation",
        "hq_city": "Van Nuys, California",
        "hq_country": "United States",
        "founded_year": 1936,
    },
    "Bell Laboratories": {
        "aliases": ["Bell Labs", "Bell Telephone Laboratories", "Bell Lab"],
        "parent": "Nokia",
        "hq_city": "Murray Hill, New Jersey",
        "hq_country": "United States",
        "founded_year": 1925,
    },
    "Bell Textron": {
        "aliases": ["Bell"],
        "parent": "Textron",
        "hq_city": "Hurst, Texas",
        "hq_country": "United States",
        "founded_year": 1935,
    },
    "Bendix": {
        "aliases": ["Bendix Corporation", "Bendix Aviation"],
        "parent": "Honeywell",
        "merged_into": "Honeywell",
        "hq_city": "Southfield, Michigan",
        "hq_country": "United States",
        "founded_year": 1929,
        "defunct_year": 1983,
    },
    "Telephonics Corporation": {
        "aliases": ["TelephonicsCorporation", "Telephonics"],
        "parent": "TransDigm Group",
        "hq_city": "Farmingdale, New York",
        "hq_country": "United States",
        "founded_year": 1933,
    },
    "Sandia": {
        "aliases": ["Sandia National Laboratories"],
        "parent": "Honeywell International",
        "hq_city": "Albuquerque, New Mexico",
        "hq_country": "United States",
        "founded_year": 1949,
    },
    "RCA": {
        "aliases": ["Radio Corporation of America"],
        "parent": "GE",
        "merged_into": "General Electric",
        "hq_city": "New York City",
        "hq_country": "United States",
        "founded_year": 1919,
        "defunct_year": 1986,
    },

    # Europe — major defense primes
    "Thales": {
        "aliases": ["Thales Group", "Thales Nederland", "Thomson-CSF",
                    "ThalesAerospaceDivision",
                    "ThalesAerospaceDivision（原宝胜厂）"],
        "parent": None,
        "hq_city": "Paris",
        "hq_country": "France",
        "founded_year": 1893,
        "ownership_type": "public",
    },
    "Leonardo": {
        "aliases": ["Selex", "Selex ES", "SELEX Sistemi Integrati",
                    "Finmeccanica", "Leonardo S.p.A."],
        "parent": None,
        "hq_city": "Rome",
        "hq_country": "Italy",
        "founded_year": 1948,
        "ownership_type": "public",
    },
    "BAE Systems": {
        "aliases": ["BAE Systems Maritime",
                    "BAE Systems Integrated System Technologies",
                    "BAE Systems Electronics"],
        "parent": None,
        "hq_city": "London",
        "hq_country": "United Kingdom",
        "founded_year": 1999,
        "ownership_type": "public",
    },
    "Marconi": {
        "aliases": ["Marconi Company", "Marconi Electronic Systems"],
        "parent": "BAE Systems",
        "merged_into": "BAE Systems",
        "hq_city": "Chelmsford",
        "hq_country": "United Kingdom",
        "founded_year": 1897,
        "defunct_year": 1999,
    },
    "EKCO": {
        "aliases": ["EKCO Electronics"],
        "parent": None,
        "hq_city": "Southend-on-Sea",
        "hq_country": "United Kingdom",
        "founded_year": 1924,
        "defunct_year": 1960,
    },
    "Saab": {
        "aliases": ["Saab AB", "Saab Group"],
        "parent": None,
        "hq_city": "Stockholm",
        "hq_country": "Sweden",
        "founded_year": 1937,
        "ownership_type": "public",
    },
    "Thorn EMI Plc": {
        "aliases": ["Thorn EMI"],
        "parent": None,
        "hq_city": "London",
        "hq_country": "United Kingdom",
        "founded_year": 1979,
        "defunct_year": 1996,
    },

    # Israel
    "Israel Aerospace Industries": {
        "aliases": ["IAI", "Israel Aerospace", "Elta Systems",
                    "ELTA", "Elta", "Elta Systems Ltd"],
        "parent": None,
        "hq_city": "Lod",
        "hq_country": "Israel",
        "founded_year": 1953,
        "ownership_type": "state-owned",
    },

    # India
    "Bharat Electronics": {
        "aliases": ["Bharat Electronics Limited", "BEL"],
        "parent": None,
        "hq_city": "Bangalore",
        "hq_country": "India",
        "founded_year": 1954,
        "ownership_type": "state-owned",
    },

    # Japan
    "Mitsubishi Electric": {
        "aliases": ["Mitsubishi Electric Corporation"],
        "parent": "Mitsubishi Group",
        "hq_city": "Tokyo",
        "hq_country": "Japan",
        "founded_year": 1921,
        "ownership_type": "public",
    },
    "Japan Radio": {
        "aliases": ["Japan Radio Co", "Japan Radio Company", "JRC"],
        "parent": "Nisshinbo Holdings",
        "hq_city": "Tokyo",
        "hq_country": "Japan",
        "founded_year": 1915,
    },

    # Russia / Soviet
    "Almaz-Antey": {
        "aliases": ["Almaz-Antey Corporation"],
        "parent": None,
        "hq_city": "Moscow",
        "hq_country": "Russia",
        "founded_year": 2002,
        "ownership_type": "state-owned",
    },
    "Tikhomirov NIIP": {
        "aliases": ["NIIP", "Tikhomirov Scientific Research Institute",
                    "ΦA3ATPOH-HMMP", "Phazotron-NIIR"],
        "parent": None,
        "hq_city": "Zhukovsky",
        "hq_country": "Russia",
        "founded_year": 1955,
        "ownership_type": "state-owned",
    },
    "MZiK": {
        "aliases": ["Mashinostroitelny Zavod imeni Kalinina"],
        "parent": "Almaz-Antey",
        "hq_city": "Yekaterinburg",
        "hq_country": "Russia",
        "founded_year": 1866,
        "ownership_type": "state-owned",
    },
    "Mints Radiotechnical Institute": {
        "aliases": [],
        "parent": None,
        "hq_city": "Moscow",
        "hq_country": "Russia",
        "founded_year": 1946,
        "ownership_type": "state-owned",
    },

    # Turkey
    "Aselsan": {
        "aliases": ["ASELSAN", "ASELSAN A.Ş."],
        "parent": None,
        "hq_city": "Ankara",
        "hq_country": "Turkey",
        "founded_year": 1975,
        "ownership_type": "state-owned",
    },
}


# ────────────────────────────────────────────────────────────
#  2. Standards knowledge base
# ────────────────────────────────────────────────────────────

STANDARDS = {
    "MIL-STD-1553": {
        "name_en": "MIL-STD-1553",
        "category": "data_bus",
        "issued_by": "US Department of Defense",
        "first_issued": 1973,
        "description": "Serial data bus for avionics and military aircraft systems",
        "aliases": ["1553", "MIL-1553"],
    },
    "MIL-STD-1760": {
        "name_en": "MIL-STD-1760",
        "category": "interface",
        "issued_by": "US Department of Defense",
        "first_issued": 1981,
        "description": "Aircraft/store electrical interconnection system for weapons",
        "aliases": ["1760"],
    },
    "ARINC-429": {
        "name_en": "ARINC-429",
        "category": "data_bus",
        "issued_by": "Aeronautical Radio Inc.",
        "first_issued": 1977,
        "description": "Avionics data transfer standard for commercial aircraft",
        "aliases": ["ARINC 429", "429"],
    },
    "ARINC-664": {
        "name_en": "ARINC-664",
        "category": "data_bus",
        "issued_by": "Aeronautical Radio Inc.",
        "first_issued": 2002,
        "description": "Aircraft Data Network (AFDX) — Ethernet-based avionics network",
        "aliases": ["AFDX"],
    },
    "RS-422": {
        "name_en": "RS-422",
        "category": "interface",
        "issued_by": "EIA / TIA",
        "first_issued": 1975,
        "description": "Differential serial data transmission interface",
        "aliases": ["TIA-422"],
    },
    "RS-485": {
        "name_en": "RS-485",
        "category": "interface",
        "issued_by": "EIA / TIA",
        "first_issued": 1983,
        "description": "Multipoint serial bus interface",
        "aliases": ["TIA-485"],
    },
    "CAN": {
        "name_en": "CAN bus",
        "category": "data_bus",
        "issued_by": "Bosch / ISO",
        "first_issued": 1986,
        "description": "Controller Area Network — vehicle bus standard",
        "aliases": ["CAN bus", "Controller Area Network"],
    },
    "AFDX": {
        "name_en": "AFDX",
        "category": "data_bus",
        "issued_by": "Airbus",
        "first_issued": 2003,
        "description": "Avionics Full-Duplex Switched Ethernet (ARINC-664 part 7)",
        "aliases": ["Avionics Full-Duplex Switched Ethernet"],
    },
}


# ────────────────────────────────────────────────────────────
#  3. Driver
# ────────────────────────────────────────────────────────────

def main():
    ent_path  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
    edge_path = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
    print(f"Reading {ent_path}")
    with open(ent_path, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(edge_path, encoding="utf-8") as f:
        edge_data = json.load(f)

    entities = {e["id"]: e for e in ent_data["entities"]}
    edges    = edge_data["edges"]
    print(f"  start: {len(entities)} entities, {len(edges)} edges")

    # Build name → canonical map for manufacturers
    name_to_canonical: dict[str, str] = {}
    for canonical, info in MANUFACTURERS.items():
        name_to_canonical[canonical] = canonical
        for a in info.get("aliases", []):
            name_to_canonical[a] = canonical

    # ──────────  3.1  Manufacturer enrichment
    new_edges = []
    new_entities = []
    canonical_used = set()
    properties_added = 0

    # Snapshot to avoid mutation during iteration (we add Location/parent entities below)
    snapshot = list(entities.values())
    for ent in snapshot:
        if ent["type"] != "Manufacturer":
            continue
        # Match KG surface form to canonical manufacturer
        canonical = name_to_canonical.get(ent["name_en"])
        if not canonical:
            continue
        info = MANUFACTURERS[canonical]
        canonical_used.add(canonical)

        # Property fold: founded_year, ownership_type, defunct_year
        if "founded_year" in info and "founded_year" not in ent:
            ent["founded_year"] = info["founded_year"]; properties_added += 1
        if "ownership_type" in info and "ownership_type" not in ent:
            ent["ownership_type"] = info["ownership_type"]; properties_added += 1
        if "defunct_year" in info and "defunct_year" not in ent:
            ent["defunct_year"] = info["defunct_year"]; properties_added += 1
        # Set canonical_name property to enable resolution
        ent.setdefault("canonical_name", canonical)

        # Headquarters → Location entity + headquarteredIn edge
        hq_city = info.get("hq_city")
        hq_country = info.get("hq_country")
        if hq_city:
            loc_id = f"{hq_city}, {hq_country}" if hq_country else hq_city
            if loc_id not in entities:
                entities[loc_id] = {
                    "id": loc_id, "type": "Location",
                    "name_en": loc_id, "name_zh": "",
                    "city": hq_city, "country": hq_country,
                    "location_type": "city",
                    "aliases": [],
                }
                new_entities.append(loc_id)
            new_edges.append({
                "head": ent["id"], "relation": "headquarteredIn",
                "tail": loc_id, "head_type": "Manufacturer", "tail_type": "Location",
                "evidence": "Wikidata-curated headquarters location",
                "confidence": 0.95, "source": "wikidata_curated",
            })

        # Subsidiary / merged_into edges
        parent = info.get("parent")
        if parent and parent != canonical:
            # ensure parent entity exists
            if parent not in entities:
                # inherit aliases from the curated MFG entry
                aliases = MANUFACTURERS.get(parent, {}).get("aliases", [])
                entities[parent] = {
                    "id": parent, "type": "Manufacturer",
                    "name_en": parent, "aliases": aliases,
                    "country": MANUFACTURERS.get(parent, {}).get("hq_country", ""),
                    "canonical_name": parent,
                }
                new_entities.append(parent)
            new_edges.append({
                "head": ent["id"], "relation": "subsidiaryOf",
                "tail": parent, "head_type": "Manufacturer", "tail_type": "Manufacturer",
                "evidence": "Wikidata-curated parent company",
                "confidence": 0.95, "source": "wikidata_curated",
            })
        merged_into = info.get("merged_into")
        if merged_into and merged_into != canonical and merged_into != parent:
            if merged_into not in entities:
                entities[merged_into] = {
                    "id": merged_into, "type": "Manufacturer",
                    "name_en": merged_into, "aliases": [],
                    "canonical_name": merged_into,
                }
                new_entities.append(merged_into)
            # keep this distinct from subsidiaryOf — historical merge
            new_edges.append({
                "head": ent["id"], "relation": "subsidiaryOf",  # use existing relation
                "tail": merged_into, "head_type": "Manufacturer", "tail_type": "Manufacturer",
                "evidence": f"merged into {merged_into} (historical)",
                "confidence": 0.95, "source": "wikidata_curated_merger",
            })

    # ──────────  3.2  Standards as new entities
    for std_id, info in STANDARDS.items():
        if std_id not in entities:
            entities[std_id] = {
                "id": std_id, "type": "Standard",
                "name_en": info["name_en"],
                "aliases": info.get("aliases", []),
                "category": info.get("category", ""),
                "issued_by": info.get("issued_by", ""),
                "first_issued": info.get("first_issued"),
                "description": info.get("description", ""),
            }
            new_entities.append(std_id)

    # ──────────  3.3  Save
    edges.extend(new_edges)
    ent_data["entities"] = list(entities.values())
    ent_data["count"] = len(entities)
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)

    with open(ent_path, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)

    # ──────────  3.4  Report
    rep_path = ROOT / "data" / "v2" / "stage_b_report.md"
    rel_counter = Counter(e["relation"] for e in new_edges)
    matched_canonicals = sorted(canonical_used)

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"""# v2 Stage B Wikidata Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Method**: Curated knowledge base (Wikidata-derived facts, manually verified)
**Output**: in-place update of `radarkg_v2_entities.json` + `radarkg_v2_edges.json`

## Summary

| Metric | Before | Added | After |
|---|---|---|---|
| Entities | {len(entities) - len(new_entities)} | +{len(new_entities)} | {len(entities)} |
| Edges    | {len(edges) - len(new_edges)} | +{len(new_edges)} | {len(edges)} |
| Properties | — | +{properties_added} | — |

## New edges by relation

{chr(10).join(f"- `{r}`: {c}" for r, c in rel_counter.most_common())}

## New entities by type

{chr(10).join(f"- {t}: {c}" for t, c in Counter(entities[eid]['type'] for eid in new_entities).most_common())}

## Manufacturers matched ({len(matched_canonicals)})

{chr(10).join(f"- {m}" for m in matched_canonicals)}

## Manufacturers NOT matched (long-tail, candidates for future SPARQL run)

The KG has 183 unique manufacturer surface forms. We curated the top {len(MANUFACTURERS)};
the remaining ~{183 - len(matched_canonicals)} are mostly variants (e.g., "the U", "the company"
extraction garbage) or single-occurrence entries that don't justify hand curation.

For these, run:
```
python import_wikidata.py --enrich-manufacturers
```
which queries Wikidata SPARQL for QID + headquarters + parent_org.

## Standards added

{chr(10).join(f"- `{s}` ({STANDARDS[s]['category']}): {STANDARDS[s]['description']}" for s in STANDARDS)}

These are now Standard entities; `meetsStandard` edges will be populated by Stage C
(LLM extraction from manuals — the manual prose mentions which radars conform to
which standards).

## Key new edges (sample)

{chr(10).join(f"- `{e['head']}` -[{e['relation']}]→ `{e['tail']}`  (conf={e['confidence']:.2f})"
              for e in new_edges[:15])}
""")
    print(f"\n────────── stage B done ──────────")
    print(f"new edges:      +{len(new_edges)}")
    print(f"new entities:   +{len(new_entities)}")
    print(f"new properties: +{properties_added}")
    print(f"manufacturers matched: {len(canonical_used)} / {len(MANUFACTURERS)}")
    print()
    print("New edges by relation:")
    for r, c in rel_counter.most_common():
        print(f"  {r:25s} +{c}")
    print()
    print("New entities by type:")
    for t, c in Counter(entities[eid]['type'] for eid in new_entities).most_common():
        print(f"  {t:15s} +{c}")
    print(f"\nReport: {rep_path}")


if __name__ == "__main__":
    main()
