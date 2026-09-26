"""
v2 Stage A2: Structural inference enrichment.

Reads:  data/v2/radarkg_v2_edges.json
Writes: data/v2/radarkg_v2_edges.json (in-place, with new inferred edges added)
        data/v2/enrichment_report.md

Adds 5 new relation types via pure structural inference (no LLM):
  1. coDeployedWith  ↔  same platform → reciprocal pairs
  2. competitorOf    ↔  same band + same platform-class + different country
  3. coOperatedBy    ↔  countries that operate the same radar (Country—Country)
  4. replacedBy      →  inverse of upgradeOf
  5. precededBy      →  inverse of upgradeOf + derivedFrom

All inferred edges carry low confidence (~0.5-0.6) and source="inferred_*".
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────────────────────
#  Existing inference modules (reused from kg_enrichment.py)
# ────────────────────────────────────────────────────────────

# Platform → class mapping (copied from kg_enrichment.py to avoid circular imports)
PLATFORM_TYPE_HINTS = {
    "destroyer": "NavalVessel", "frigate": "NavalVessel", "cruiser": "NavalVessel",
    "carrier": "NavalVessel", "corvette": "NavalVessel", "submarine": "NavalVessel",
    "naval": "NavalVessel", "ship": "NavalVessel", "舰": "NavalVessel",
    "驱逐": "NavalVessel", "护卫": "NavalVessel", "巡洋": "NavalVessel",
    "fighter": "AircraftPlatform", "aircraft": "AircraftPlatform",
    "bomber": "AircraftPlatform", "helicopter": "AircraftPlatform",
    "uav": "AircraftPlatform", "airborne": "AircraftPlatform",
    "aew": "AircraftPlatform", "awacs": "AircraftPlatform",
    "预警机": "AircraftPlatform", "战斗机": "AircraftPlatform",
    "ground": "GroundPlatform", "mobile": "GroundPlatform", "fixed": "GroundPlatform",
    "vehicle": "GroundPlatform", "地面": "GroundPlatform",
    "固定": "GroundPlatform", "车载": "GroundPlatform",
}


def _platform_class(name: str) -> Optional[str]:
    if not name: return None
    lower = name.lower()
    for kw, cls in PLATFORM_TYPE_HINTS.items():
        if kw in lower:
            return cls
    return None


def make_edge(head, relation, tail, *, head_type="", tail_type="",
               confidence=0.5, evidence="", source="inferred"):
    return {
        "head": head, "relation": relation, "tail": tail,
        "head_type": head_type, "tail_type": tail_type,
        "confidence": confidence, "evidence": evidence, "source": source,
    }


# ────────────────────────────────────────────────────────────
#  1. coDeployedWith — radars sharing a platform
# ────────────────────────────────────────────────────────────

def infer_co_deployed(edges: list, min_conf: float = 0.65) -> list:
    by_platform: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for e in edges:
        if e["relation"] != "deployedOn": continue
        if e.get("confidence", 0) < min_conf: continue
        by_platform[e["tail"]].append((e["head"], e["confidence"], e.get("head_type", "")))

    out = []
    seen: set[frozenset] = set()
    for platform, members in by_platform.items():
        if len(members) < 2: continue
        for i, (r1, c1, ht1) in enumerate(members):
            for r2, c2, ht2 in members[i+1:]:
                if r1.lower() == r2.lower(): continue
                pair = frozenset([r1, r2])
                if pair in seen: continue
                seen.add(pair)
                conf = round(min(c1, c2) - 0.05, 3)
                ev = f"co-deployed on {platform}"
                out.append(make_edge(r1, "coDeployedWith", r2,
                                      head_type=ht1, tail_type=ht2,
                                      confidence=conf, evidence=ev,
                                      source="inferred_co_deployed"))
                out.append(make_edge(r2, "coDeployedWith", r1,
                                      head_type=ht2, tail_type=ht1,
                                      confidence=conf, evidence=ev,
                                      source="inferred_co_deployed"))
    return out


# ────────────────────────────────────────────────────────────
#  2. competitorOf — same band + platform-class, different country
# ────────────────────────────────────────────────────────────

def infer_competitors(edges: list, min_conf: float = 0.5) -> list:
    bands: dict[str, set[str]] = defaultdict(set)
    countries: dict[str, set[str]] = defaultdict(set)
    plat_cls: dict[str, set[str]] = defaultdict(set)
    head_types: dict[str, str] = {}

    for e in edges:
        if e.get("confidence", 0) < min_conf: continue
        h, r, t = e["head"], e["relation"], e["tail"]
        if not (h and t): continue
        head_types.setdefault(h, e.get("head_type", ""))
        if r == "hasFrequencyBand":
            bands[h].add(t)
        elif r == "operatedBy":
            countries[h].add(t)
        elif r == "deployedOn":
            cls = _platform_class(t) or e.get("tail_type", "")
            if cls: plat_cls[h].add(cls)

    candidates = [r for r in bands if bands[r] and countries[r] and plat_cls[r]]
    out = []
    seen: set[frozenset] = set()
    for i, r1 in enumerate(candidates):
        for r2 in candidates[i+1:]:
            if not (bands[r1] & bands[r2]): continue
            if not (plat_cls[r1] & plat_cls[r2]): continue
            if countries[r1] & countries[r2]: continue
            if r1.lower() == r2.lower(): continue
            pair = frozenset([r1, r2])
            if pair in seen: continue
            seen.add(pair)
            ev = f"shared band={list(bands[r1] & bands[r2])[0]}, " \
                 f"platform_class={list(plat_cls[r1] & plat_cls[r2])[0]}, " \
                 f"diff countries"
            out.append(make_edge(r1, "competitorOf", r2,
                                  head_type=head_types.get(r1, "RadarSystem"),
                                  tail_type=head_types.get(r2, "RadarSystem"),
                                  confidence=0.60, evidence=ev,
                                  source="inferred_competitor"))
            out.append(make_edge(r2, "competitorOf", r1,
                                  head_type=head_types.get(r2, "RadarSystem"),
                                  tail_type=head_types.get(r1, "RadarSystem"),
                                  confidence=0.60, evidence=ev,
                                  source="inferred_competitor"))
    return out


# ────────────────────────────────────────────────────────────
#  3. coOperatedBy — countries that operate the same radar
# ────────────────────────────────────────────────────────────

def infer_co_operated_by(edges: list, min_conf: float = 0.65) -> list:
    by_radar: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e["relation"] != "operatedBy": continue
        if e.get("confidence", 0) < min_conf: continue
        by_radar[e["head"]].add(e["tail"])

    pair_radar: dict[frozenset, list[str]] = defaultdict(list)
    for radar, countries in by_radar.items():
        if len(countries) < 2: continue
        clist = sorted(countries)
        for i, c1 in enumerate(clist):
            for c2 in clist[i+1:]:
                pair_radar[frozenset([c1, c2])].append(radar)

    out = []
    for pair, radars in pair_radar.items():
        if len(radars) < 2: continue   # require at least 2 shared radars
        c1, c2 = sorted(pair)
        ev = f"share {len(radars)} radars; e.g. {radars[0]}"
        out.append(make_edge(c1, "coOperatedBy", c2,
                              head_type="Country", tail_type="Country",
                              confidence=0.55, evidence=ev,
                              source="inferred_co_operated"))
        out.append(make_edge(c2, "coOperatedBy", c1,
                              head_type="Country", tail_type="Country",
                              confidence=0.55, evidence=ev,
                              source="inferred_co_operated"))
    return out


# ────────────────────────────────────────────────────────────
#  4. replacedBy — inverse of upgradeOf (forward chronological)
#  5. precededBy — inverse of upgradeOf + derivedFrom (backward chronological)
# ────────────────────────────────────────────────────────────

def infer_replaced_by_and_preceded(edges: list) -> list:
    out = []
    for e in edges:
        h, r, t = e["head"], e["relation"], e["tail"]
        ht, tt = e.get("head_type", ""), e.get("tail_type", "")
        conf = max(0.5, e.get("confidence", 0.5) - 0.1)
        if r == "upgradeOf":
            # h is upgrade of t → t was replaced by h
            out.append(make_edge(t, "replacedBy", h,
                                  head_type=tt, tail_type=ht,
                                  confidence=conf, evidence=f"inverse of upgradeOf",
                                  source="inferred_replaced_by"))
            # h precededBy t (h came after t, t came before h)
            out.append(make_edge(h, "precededBy", t,
                                  head_type=ht, tail_type=tt,
                                  confidence=conf, evidence=f"inverse of upgradeOf",
                                  source="inferred_preceded_by"))
        elif r == "derivedFrom":
            out.append(make_edge(h, "precededBy", t,
                                  head_type=ht, tail_type=tt,
                                  confidence=conf, evidence=f"inverse of derivedFrom",
                                  source="inferred_preceded_by"))
    return out


# ────────────────────────────────────────────────────────────
#  Driver
# ────────────────────────────────────────────────────────────

def main():
    edge_path = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
    print(f"Reading {edge_path}")
    with open(edge_path, encoding="utf-8") as f:
        data = json.load(f)
    edges = data["edges"]
    print(f"  {len(edges)} input edges")

    # run inference modules
    print("\nRunning inference modules…")
    inferences = {
        "coDeployedWith":  infer_co_deployed(edges),
        "competitorOf":    infer_competitors(edges),
        "coOperatedBy":    infer_co_operated_by(edges),
        "replacedBy/precededBy": infer_replaced_by_and_preceded(edges),
    }
    new_edges = []
    for name, batch in inferences.items():
        new_edges.extend(batch)
        print(f"  {name:30s} +{len(batch)} edges")

    # de-duplicate (head, relation, tail)
    existing_keys = {(e["head"], e["relation"], e["tail"]) for e in edges}
    fresh = [e for e in new_edges
             if (e["head"], e["relation"], e["tail"]) not in existing_keys]
    print(f"\nDe-duplicated: {len(new_edges)} → {len(fresh)} fresh edges")

    edges.extend(fresh)
    by_relation = Counter(e["relation"] for e in edges)
    print(f"\n────────── stage A2 enrichment done ──────────")
    print(f"total edges: {len(edges)}")
    print()
    for r, c in by_relation.most_common():
        print(f"  {r:25s} {c}")

    # write back
    with open(edge_path, "w", encoding="utf-8") as f:
        json.dump({"version": "v2", "count": len(edges), "edges": edges},
                  f, ensure_ascii=False, indent=2)
    print(f"\nWritten back: {edge_path}")

    # report
    rep_path = ROOT / "data" / "v2" / "enrichment_report.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"""# v2 Stage A2 Enrichment Report

**Date**: 2026-04-29
**Input**: {len(data['edges'])} edges (post Stage A)
**Output**: {len(edges)} edges (post Stage A2)

## Inferred edges added

| Module | Count |
|---|---|
""")
        for name, batch in inferences.items():
            f.write(f"| {name} | {len(batch)} |\n")
        f.write(f"\n**Total fresh** (after dedup): {len(fresh)}\n\n")
        f.write("## Edges by relation (post-enrichment)\n\n")
        for r, c in by_relation.most_common():
            f.write(f"- `{r}`: {c}\n")
    print(f"  {rep_path}")


if __name__ == "__main__":
    main()
