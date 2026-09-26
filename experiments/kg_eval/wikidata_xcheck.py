"""
Independent accuracy check against Wikidata (no human, no LLM)
=============================================================
Takes a sample of NON-Wikidata-sourced triples on relations Wikidata also covers
(developedBy -> manufacturer P176; countryOfOrigin -> P495 / operator P137), links
the head radar to a Wikidata entity by name search, and compares the KG's tail to
Wikidata's value. Reports COVERAGE (how many heads were found in Wikidata) and,
on the covered subset, AGREEMENT (an independent precision estimate).

Honest by design: radar models are sparsely covered by Wikidata, so coverage will
be partial; the agreement rate is reported only on the matched subset and clearly
labelled as such.

Run:  python experiments/kg_eval/wikidata_xcheck.py
"""
import json, re, random, time
from pathlib import Path
from collections import Counter
import requests

ROOT = Path(__file__).resolve().parents[2]
T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
API = "https://www.wikidata.org/w/api.php"
HEAD = {"User-Agent": "radar-kg-eval/1.0 (academic thesis evaluation)"}
# KG stores countries in Chinese; map to Wikidata English/QID labels for matching
COUNTRY = {"美国": ["united states", "usa", "u.s."], "中国": ["china", "people's republic"],
           "俄罗斯": ["russia", "soviet"], "法国": ["france"], "英国": ["united kingdom", "britain"],
           "德国": ["germany"], "日本": ["japan"], "以色列": ["israel"], "意大利": ["italy"],
           "瑞典": ["sweden"], "印度": ["india"]}


def norm(s): return re.sub(r"[\s\-_.]", "", str(s)).strip().lower()


def wd_search(name):
    try:
        r = requests.get(API, params={"action": "wbsearchentities", "search": name,
                         "language": "en", "format": "json", "limit": 1}, headers=HEAD, timeout=15)
        hits = r.json().get("search", [])
        return hits[0]["id"] if hits else None
    except Exception:
        return None


def wd_claims(qid, prop):
    try:
        r = requests.get(API, params={"action": "wbgetentities", "ids": qid,
                         "props": "claims", "format": "json"}, headers=HEAD, timeout=15)
        claims = r.json()["entities"][qid]["claims"].get(prop, [])
        out = []
        for c in claims:
            try:
                out.append(c["mainsnak"]["datavalue"]["value"]["id"])   # target QID
            except Exception:
                pass
        return out
    except Exception:
        return []


def label(qid):
    try:
        r = requests.get(API, params={"action": "wbgetentities", "ids": qid,
                         "props": "labels", "languages": "en", "format": "json"},
                         headers=HEAD, timeout=15)
        return r.json()["entities"][qid]["labels"].get("en", {}).get("value", "")
    except Exception:
        return ""


def check(relation, prop, n, rng):
    pool = [t for t in T if t["relation"] == relation and str(t.get("source")) != "wikidata"]
    rng.shuffle(pool)
    found = agree = checked = 0
    examples = []
    for t in pool:
        if checked >= n:
            break
        qid = wd_search(t["head"])
        time.sleep(0.1)
        if not qid:
            continue
        targets = wd_claims(qid, prop)
        if not targets:
            continue
        checked += 1; found += 1
        wd_labels = [norm(label(q)) for q in targets[:3]]
        kg = t["tail"]
        if relation == "countryOfOrigin":
            cands = [norm(x) for x in COUNTRY.get(kg, [kg])]
            ok = any(any(c in wl or wl in c for wl in wd_labels) for c in cands if c)
        else:
            k = norm(kg)
            ok = any(k in wl or wl in k for wl in wd_labels if wl)
        agree += ok
        if len(examples) < 6:
            examples.append((t["head"][:24], kg, "/".join(label(q) for q in targets[:2])[:30], ok))
        time.sleep(0.1)
    return found, agree, checked, examples


def main():
    rng = random.Random(0)
    print("Independent cross-check vs Wikidata (non-Wikidata-sourced triples)\n")
    for rel, prop, lab in [("developedBy", "P176", "manufacturer"),
                           ("countryOfOrigin", "P495", "country of origin")]:
        print(f"=== {rel}  (Wikidata {prop} = {lab}) ===")
        found, agree, checked, ex = check(rel, prop, 40, rng)
        if checked:
            print(f"  linked & had {prop}: {checked} radar heads  |  agreement: {agree}/{checked} = {agree/checked:.0%}")
            for h, kg, wd, ok in ex:
                print(f"    {'OK ' if ok else 'DIFF'} {h:<24} KG={kg!r:<22} WD={wd!r}")
        else:
            print("  no heads could be linked to Wikidata with this property (coverage 0)")
        print()
    print("NOTE: agreement is an INDEPENDENT precision estimate, but only on the")
    print("Wikidata-covered subset; radar models are sparsely covered, so this")
    print("complements — does not replace — the manual annotation (annotation_sheet.csv).")


if __name__ == "__main__":
    main()
