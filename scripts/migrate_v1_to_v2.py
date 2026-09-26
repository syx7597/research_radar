"""
v1 → v2 KG migration (Stage A).

Reads:  graphrag_index/merged_triples.json   (5592 v1 triples)
        lexicon/relations.json
        lexicon/entity_aliases.json

Writes: data/v2/radarkg_v2_entities.json     (entities + properties)
        data/v2/radarkg_v2_edges.json        (inter-entity edges only)
        data/v2/radarkg_v2_schema.json       (machine-readable schema)
        data/v2/migration_report.md          (statistics + caveats)

Key transformations:
  1. 14 literal-tail relations → entity properties on Radar nodes
     (hasRange / hasFrequency / hasWeight / hasMTBF / hasPower / hasPeakPower /
      hasLRUCount / hasAntennaGain / hasECCM / hasCooling / status / price /
      researchedIn / deployedIn)

  2. Unit parsing — extracts numeric values from messy string literals:
     "20km(RCS1m²)"             → range_km=20.0, range_description="..."
     "9.2GHz~9.5GHz"            → frequency_GHz_min=9.2, frequency_GHz_max=9.5
     "约50kg"                   → weight_kg=50.0
     "1981年"                   → deployed_year=1981
     "20世纪70年代"             → researched_period="1970s"
     "120万美元"                 → price_usd=1_200_000

  3. Multi-valued relations (hasFrequencyBand / hasTechType / hasMode / hasFunction)
     are kept as edges AND duplicated as list-typed entity properties
     for fast lookup.

  4. Aliases are folded from entity_aliases.json into each entity's `aliases` field.

  5. Garbage filtering: extraction artifacts ("the same company", "which had previously",
     "the U", etc.) are dropped from edge tails.
"""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────

LITERAL_RELATIONS = {
    "hasRange":        "range",
    "hasFrequency":    "frequency",
    "hasWeight":       "weight",
    "hasMTBF":         "mtbf",
    "hasPower":        "power_consumption",
    "hasPeakPower":    "peak_power",
    "hasLRUCount":     "lru_count",
    "hasAntennaGain":  "antenna_gain",
    "hasECCM":         "eccm",
    "hasCooling":      "cooling",
    "status":          "status",
    "price":           "price",
    "researchedIn":    "researched",
    "deployedIn":      "deployed",
}

# Multi-valued entity-tail relations that should also be folded into entity property lists
MULTI_VALUED_RELATIONS = {
    "hasFrequencyBand": "frequency_bands",
    "hasTechType":      "tech_types",
    "hasMode":          "modes",
    "hasFunction":      "functions",
}

# Single-valued entity-tail relations folded as entity properties (PRIMARY value),
# but also kept as edges for graph queries.
SINGLE_VALUED_PROPS = {
    "developedBy":     "developer",
    "countryOfOrigin": "country_of_origin",
    "operatedBy":      "operator_primary",   # if multiple, store first; full list as edge
    "affiliatedTo":    "country",            # for Manufacturer
}

# Garbage tail blacklist (extraction artifacts)
TAIL_GARBAGE = {
    "the same company", "which had previously", "the U", "the company",
    "no", "n/a", "none", "unknown", "allies",
}


# ────────────────────────────────────────────────────────────
#  Value parsers — turn strings into structured numeric properties
# ────────────────────────────────────────────────────────────

UNIT_NUM_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:[~～-]\s*(?P<high>\d+(?:\.\d+)?))?\s*(?P<unit>[A-Za-zμ]+)?"
)


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_range(value: str) -> dict:
    """
    "20km(RCS1m²夏船，海态3)" → {range_km: 20.0, range_description: "..."}
    "37km~46km(海面石油污染)" → {range_km_low: 37.0, range_km_high: 46.0, ...}
    "3000m"                   → {range_km: 3.0}
    """
    out = {"range_description": value.strip()}
    m = UNIT_NUM_RE.search(value)
    if not m:
        return out
    low = _to_float(m.group("low"))
    high = _to_float(m.group("high"))
    unit = (m.group("unit") or "").lower()
    if low is None:
        return out
    if "km" in unit:
        scale = 1.0
    elif unit == "m":
        scale = 0.001
    elif "海里" in value or unit in ("nm", "nmi"):
        scale = 1.852
    else:
        scale = 1.0  # assume km if missing
    if high is not None:
        out["range_km_min"] = low * scale
        out["range_km_max"] = high * scale
        out["range_km"] = (low + high) / 2 * scale
    else:
        out["range_km"] = low * scale
    return out


def parse_frequency(value: str) -> dict:
    """
    "9.2GHz~9.5GHz"           → {frequency_GHz_min: 9.2, frequency_GHz_max: 9.5}
    "(9.375±0.039)GHz"        → {frequency_GHz: 9.375}
    "8GHz~12.5GHz"            → {frequency_GHz_min: 8, frequency_GHz_max: 12.5}
    "33.7GHz~35.5GHz"         → {frequency_GHz_min: 33.7, frequency_GHz_max: 35.5}
    """
    out = {"frequency_description": value.strip()}
    # find first numeric range or single
    nums = re.findall(r"(\d+(?:\.\d+)?)", value)
    if not nums:
        return out
    nums = [float(n) for n in nums]
    if "MHz" in value and "GHz" not in value:
        nums = [n / 1000 for n in nums]
    elif "kHz" in value:
        nums = [n / 1_000_000 for n in nums]
    if len(nums) >= 2 and "~" in value or len(nums) >= 2 and "-" in value:
        out["frequency_GHz_min"] = nums[0]
        out["frequency_GHz_max"] = nums[1]
        out["frequency_GHz"] = (nums[0] + nums[1]) / 2
    else:
        out["frequency_GHz"] = nums[0]
    return out


def parse_weight(value: str) -> dict:
    """
    "约50kg"                          → {weight_kg: 50}
    "20.95kg(彩显),15.95kg(黑白)"     → {weight_kg: 20.95, weight_description: "..."}
    "5kg(天线)"                       → {weight_kg: 5, weight_description: "..."}
    "约8t"                            → {weight_kg: 8000}
    """
    out = {"weight_description": value.strip()}
    m = UNIT_NUM_RE.search(value)
    if not m:
        return out
    low = _to_float(m.group("low"))
    unit = (m.group("unit") or "").lower()
    if low is None:
        return out
    if "kg" in unit or "公斤" in value:
        out["weight_kg"] = low
    elif unit == "t" or "吨" in value:
        out["weight_kg"] = low * 1000
    elif unit == "g":
        out["weight_kg"] = low / 1000
    return out


def parse_mtbf(value: str) -> dict:
    """
    "150h(Ka频段),220h(L频段)"   → {mtbf_hours: 150, mtbf_description: "..."}
    "1000h"                       → {mtbf_hours: 1000}
    """
    out = {"mtbf_description": value.strip()}
    m = re.search(r"(\d+(?:\.\d+)?)\s*h", value)
    if m:
        out["mtbf_hours"] = float(m.group(1))
    return out


def parse_power_kw(value: str) -> dict:
    """
    "25kW±1dB"                   → {peak_power_kW: 25}
    "500mW"                      → {peak_power_kW: 0.0005}
    "3.5kW"                      → {peak_power_kW: 3.5}
    "120kW~220kW"                → {peak_power_kW_min: 120, peak_power_kW_max: 220}
    """
    out = {}
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*(kW|MW|mW|W)", value)
    if not nums:
        return out
    converted = []
    for v, unit in nums:
        v = float(v)
        if unit == "kW": v_kw = v
        elif unit == "MW": v_kw = v * 1000
        elif unit == "mW": v_kw = v / 1_000_000
        elif unit == "W":  v_kw = v / 1000
        else: v_kw = v
        converted.append(v_kw)
    if len(converted) >= 2:
        out["peak_power_kW_min"] = min(converted)
        out["peak_power_kW_max"] = max(converted)
    out["peak_power_kW"] = converted[0]
    return out


def parse_lru_count(value: str) -> dict:
    """
    "4个"                        → {lru_count: 4}
    "主要部件有10余个"           → {lru_count_approx: 10}
    "3个(机顶罩内部件)"          → {lru_count: 3, lru_description: "..."}
    """
    out = {"lru_description": value.strip()}
    m = re.search(r"(\d+)", value)
    if m:
        out["lru_count"] = int(m.group(1))
    return out


def parse_antenna_gain(value: str) -> dict:
    """ "32dB"  → {antenna_gain_dB: 32} """
    out = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*dB", value)
    if m:
        out["antenna_gain_dB"] = float(m.group(1))
    return out


def parse_price(value: str) -> dict:
    """
    "300万美元"                   → {price_usd: 3_000_000}
    "120万美元"                   → {price_usd: 1_200_000}
    """
    out = {"price_description": value.strip()}
    m = re.search(r"(\d+(?:\.\d+)?)\s*万\s*美元", value)
    if m:
        out["price_usd"] = float(m.group(1)) * 10_000
        return out
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", value)
    if m:
        out["price_usd"] = float(m.group(1))
    return out


def parse_year(value: str) -> dict:
    """
    "1981年"                     → {year: 1981}
    "20世纪70年代"               → {decade: 1970}
    "20世纪90年代中期"            → {decade: 1990}
    "1994年~2000年"              → {year_min: 1994, year_max: 2000}
    """
    out = {"period_description": value.strip()}
    # 4-digit year(s)
    years = re.findall(r"(19\d{2}|20\d{2})", value)
    if len(years) >= 2:
        out["year_min"] = int(years[0]); out["year_max"] = int(years[1])
        out["year"] = int(years[0])
        return out
    if years:
        out["year"] = int(years[0]); return out
    # decade
    m = re.search(r"(\d{1,2})\s*世纪\s*(\d{2})\s*年代", value)
    if m:
        century = int(m.group(1))
        decade  = int(m.group(2))
        out["decade"] = (century - 1) * 100 + decade
    return out


def parse_status(value: str) -> dict:
    """
    "正在服役"                   → {status: "in_service"}
    "生产、使用中"                → {status: "in_service"}
    "已退役"                     → {status: "retired"}
    "研制中"                     → {status: "development"}
    """
    v = value.strip()
    out = {"status_description": v}
    if any(k in v for k in ["服役中", "正在服役", "使用中", "在役", "生产", "现役"]):
        out["status"] = "in_service"
    elif any(k in v for k in ["退役", "停用", "已停产"]):
        out["status"] = "retired"
    elif any(k in v for k in ["研制中", "开发中", "正在开发", "研制"]):
        out["status"] = "development"
    elif any(k in v for k in ["取消", "终止"]):
        out["status"] = "cancelled"
    else:
        out["status"] = "unknown"
    return out


def parse_eccm(value: str) -> dict:
    return {"eccm_description": value.strip()}


def parse_cooling(value: str) -> dict:
    out = {"cooling_description": value.strip()}
    methods = []
    if "风冷" in value or "air cooled" in value.lower(): methods.append("air")
    if "液冷" in value or "liquid" in value.lower(): methods.append("liquid")
    if "自然" in value: methods.append("natural")
    out["cooling_methods"] = methods
    return out


PARSER_MAP = {
    "range":             parse_range,
    "frequency":         parse_frequency,
    "weight":            parse_weight,
    "mtbf":              parse_mtbf,
    "peak_power":        parse_power_kw,
    "power_consumption": parse_power_kw,
    "lru_count":         parse_lru_count,
    "antenna_gain":      parse_antenna_gain,
    "price":             parse_price,
    "researched":        parse_year,
    "deployed":          parse_year,
    "status":            parse_status,
    "eccm":              parse_eccm,
    "cooling":           parse_cooling,
}


# ────────────────────────────────────────────────────────────
#  Main migration
# ────────────────────────────────────────────────────────────

def is_garbage_tail(tail: str) -> bool:
    if not tail or len(tail) > 60:
        return True
    if tail.strip().lower() in TAIL_GARBAGE:
        return True
    return False


def main():
    in_path = ROOT / "graphrag_index" / "merged_triples.json"
    out_dir = ROOT / "data" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {in_path}")
    with open(in_path, encoding="utf-8") as f:
        triples = json.load(f)
    print(f"  {len(triples)} input triples")

    # Load aliases for fold-in
    with open(ROOT / "lexicon" / "entity_aliases.json", encoding="utf-8") as f:
        alias_data = json.load(f)

    # build alias-to-canonical and canonical-to-aliases maps
    canon2aliases = defaultdict(set)
    for sec_name, sec in alias_data.items():
        if not isinstance(sec, dict):
            continue
        for canonical, rec in sec.items():
            if not isinstance(rec, dict):
                continue
            for a in [rec.get("zh", ""), rec.get("en", "")] + list(rec.get("aliases", [])):
                if a and a != canonical:
                    canon2aliases[canonical].add(a)

    # ──────────  Pass 1: collect entity types and properties from literal-tail triples
    entities: dict[str, dict] = {}
    edges: list[dict] = []
    stats = {
        "literal_to_property": 0,
        "edges_kept":          0,
        "edges_dropped_garbage": 0,
        "entities_seen":       0,
    }

    for t in triples:
        h, r, ta = t["head"], t["relation"], t["tail"]
        ht, tt = t.get("head_type", ""), t.get("tail_type", "")

        # Ensure head entity exists
        if h not in entities:
            entities[h] = {"id": h, "type": ht or "Radar", "name_en": h, "aliases": []}
            if h in canon2aliases:
                entities[h]["aliases"] = sorted(canon2aliases[h])
            stats["entities_seen"] += 1

        # Ensure tail entity exists if it's an entity type
        if tt and tt != "Literal" and not is_garbage_tail(ta):
            if ta not in entities:
                entities[ta] = {"id": ta, "type": tt, "name_en": ta, "aliases": []}
                if ta in canon2aliases:
                    entities[ta]["aliases"] = sorted(canon2aliases[ta])
                stats["entities_seen"] += 1

        # ── Branch 1: literal-tail → property
        if r in LITERAL_RELATIONS:
            prop_namespace = LITERAL_RELATIONS[r]
            parser = PARSER_MAP.get(prop_namespace)
            parsed = parser(ta) if parser else {f"{prop_namespace}_description": ta}
            ent = entities[h]
            for k, v in parsed.items():
                # don't overwrite richer existing values
                if k in ent and ent[k]:
                    continue
                ent[k] = v
            stats["literal_to_property"] += 1
            continue

        # ── Branch 2: entity-tail → keep as edge
        if is_garbage_tail(ta):
            stats["edges_dropped_garbage"] += 1
            continue

        # Single-valued property fold (in addition to keeping the edge)
        if r in SINGLE_VALUED_PROPS:
            prop = SINGLE_VALUED_PROPS[r]
            ent = entities[h]
            ent.setdefault(prop, ta)  # only set if not already

        # Multi-valued list property fold
        if r in MULTI_VALUED_RELATIONS:
            prop = MULTI_VALUED_RELATIONS[r]
            ent = entities[h]
            ent.setdefault(prop, [])
            if ta not in ent[prop]:
                ent[prop].append(ta)

        edges.append({
            "head":       h,
            "relation":   r,
            "tail":       ta,
            "head_type":  ht,
            "tail_type":  tt,
            "evidence":   t.get("evidence", ""),
            "confidence": t.get("confidence", 0.0),
            "source":     t.get("source", ""),
        })
        stats["edges_kept"] += 1

    # ──────────  Pass 2: emit schema description
    schema = {
        "version": "v2",
        "generated_from": str(in_path.name),
        "entity_types": sorted(set(e["type"] for e in entities.values())),
        "edge_relations": sorted(set(e["relation"] for e in edges)),
        "literal_relations_promoted": list(LITERAL_RELATIONS.keys()),
        "multi_valued_relations":      list(MULTI_VALUED_RELATIONS.keys()),
        "single_valued_property_folds": list(SINGLE_VALUED_PROPS.keys()),
    }

    # ──────────  Reports
    type_counts  = Counter(e["type"] for e in entities.values())
    rel_counts   = Counter(e["relation"] for e in edges)

    # save
    ent_path  = out_dir / "radarkg_v2_entities.json"
    edge_path = out_dir / "radarkg_v2_edges.json"
    schema_path = out_dir / "radarkg_v2_schema.json"

    with open(ent_path, "w", encoding="utf-8") as f:
        json.dump({"version": "v2", "count": len(entities),
                   "entities": list(entities.values())}, f, ensure_ascii=False, indent=2)
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump({"version": "v2", "count": len(edges), "edges": edges},
                  f, ensure_ascii=False, indent=2)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"\n────────── stage A migration done ──────────")
    print(f"input triples:          {len(triples)}")
    print(f"literal→property:       {stats['literal_to_property']}")
    print(f"edges kept:             {stats['edges_kept']}")
    print(f"edges dropped (garbage):{stats['edges_dropped_garbage']}")
    print(f"entities seen:          {stats['entities_seen']}")
    print()
    print("entity types:")
    for t, c in type_counts.most_common():
        print(f"  {t:25s} {c}")
    print()
    print("edge relations:")
    for r, c in rel_counts.most_common():
        print(f"  {r:25s} {c}")
    print(f"\nWritten:")
    print(f"  {ent_path}")
    print(f"  {edge_path}")
    print(f"  {schema_path}")

    # ──────────  Spot-check: sample radar entity to verify property structure
    print("\n────────── sample radar entity (AN/TPY-2) ──────────")
    sample = entities.get("AN/TPY-2")
    if sample:
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    # ──────────  Migration report
    report_path = out_dir / "migration_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# v1 → v2 Migration Report (Stage A)

**Date**: 2026-04-29
**Input**: `{in_path.name}` ({len(triples)} triples)
**Output**:
- `radarkg_v2_entities.json` ({len(entities)} entities)
- `radarkg_v2_edges.json` ({len(edges)} edges)

## Statistics

| Metric | Value |
|---|---|
| Input triples | {len(triples)} |
| Literal triples promoted to entity properties | {stats['literal_to_property']} |
| Inter-entity edges retained | {stats['edges_kept']} |
| Edges dropped as extraction garbage | {stats['edges_dropped_garbage']} |
| Total entities (after fold-in) | {len(entities)} |

## Entity types

{chr(10).join(f"- `{t}`: {c}" for t, c in type_counts.most_common())}

## Edges by relation

{chr(10).join(f"- `{r}`: {c}" for r, c in rel_counts.most_common())}

## Next steps (per KG_SCHEMA.md)

- Stage A2: structural inference (`scripts/kg_enrichment_v2.py`)
- Stage B: Wikidata enrichment
- Stage C: LLM-driven subsystem/component extraction

## Caveats

- Some literal values may have multiple parses captured (e.g., `peak_power_kW` AND
  `peak_power_kW_min/max` for ranges); downstream consumers should prefer the
  more specific keys.
- Status mapping is heuristic; manual review recommended.
- Property fold for `operator_primary` only stores the FIRST country seen; full
  list is available via the `operatedBy` edges.
""")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
