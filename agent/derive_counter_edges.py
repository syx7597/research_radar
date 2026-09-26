# -*- coding: utf-8 -*-
"""Derive auditable counter-relation edges from threat profiles + doctrine map.

For every radar with a threat profile, run the EW advisor and emit triples:
  (radar, vulnerableTo, jamming_technique)   for viable recommendations
  (radar, resistantTo, jamming_technique)    for avoided / ECCM-neutralised ones
  (radar, employsECCM, eccm_technique)       for the ECCM it employs

Each triple carries source = "doctrine_rule:<rule_id>" (or "kg_keyword" for
employsECCM), a confidence, and evidence_type="doctrine_heuristic", so these
edges plug into the same provenance model as the rest of the KG. They are
written to a SEPARATE file (not merged blindly) for review.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from threat_profile import build_profiles, OUT as PROFILES_OUT
from ew_advisor import EWAdvisor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ew" / "counter_edges.json"


def load_profiles():
    """Prefer the enriched (T1+T2+T3) profiles on disk; fall back to keyword tier."""
    if PROFILES_OUT.exists():
        return json.load(open(PROFILES_OUT, encoding="utf-8"))
    return build_profiles()


def derive(profiles=None):
    profiles = profiles or load_profiles()
    adv = EWAdvisor()
    triples = []

    def add(h, r, t, source, conf, evidence=""):
        triples.append({"head": h, "relation": r, "tail": t, "source": source,
                        "confidence": conf, "evidence": evidence,
                        "evidence_type": "doctrine_heuristic"})

    for name, prof in profiles.items():
        rec = adv.recommend(prof)
        for item in rec["recommended"]:
            cite = item["cites"][0]
            add(name, "vulnerableTo", item["id"], f"doctrine_rule:{cite['rule_id']}",
                item["confidence"], cite["rationale"])
        for item in rec["countered_by_eccm"]:
            cite = item["cites"][0]
            add(name, "resistantTo", item["id"], f"doctrine_rule:{cite['rule_id']}",
                round(1 - item["confidence"], 3),
                "被本机抗干扰抵消：" + "、".join(item["neutralised_by"]))
        for item in rec["avoid"]:
            cite = item["cites"][0]
            add(name, "resistantTo", item["id"], f"doctrine_rule:{cite['rule_id']}",
                cite["confidence"], cite["rationale"])
        # employsECCM from the profile's keyword extraction
        ev = prof.get("_evidence", {}).get("employs_eccm", {})
        kwmap = ev.get("evidence", {}) if isinstance(ev, dict) else {}
        for e in prof.get("employs_eccm", []):
            add(name, "employsECCM", e, "kg_keyword", ev.get("confidence", 0.6),
                kwmap.get(e, ""))
    return triples


if __name__ == "__main__":
    profs = load_profiles()
    triples = derive(profs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(triples, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    rels = Counter(t["relation"] for t in triples)
    print(f"profiles: {len(profs)}  ->  counter edges: {len(triples)}")
    for r, c in rels.most_common():
        print(f"  {r:16s} {c}")
    # how many radars got at least one vulnerableTo
    heads = {t["head"] for t in triples if t["relation"] == "vulnerableTo"}
    print(f"radars with >=1 vulnerableTo: {len(heads)}")
