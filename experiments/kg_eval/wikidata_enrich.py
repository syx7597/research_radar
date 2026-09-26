"""
Wikidata relation enrichment for EXISTING radars (add-edges + validate + expose gaps).
======================================================================================
Per the marginal-value argument: don't pull NEW radars — take the KG's existing ~1,140
radar entities, look them up in Wikidata, and pull their RELATIONS to (a) add missing
edges, (b) cross-validate the manual-extracted relations, (c) expose the blank quadrant
(radars with no Wikidata match / no lineage). Output is aligned to the KG schema.

Wikidata -> schema mapping:
  P176 manufacturer / P287 designer     -> developedBy
  P495 country of origin / P17 country  -> countryOfOrigin
  P1365 replaces                        -> replaces
  P1366 replaced by                     -> replacedBy
  P144 based on / P155 follows          -> derivedFrom
  P156 followed by                      -> (successor; inverse of derivedFrom)

Output: wikidata_enrich_out.json = {add_candidates, validation, coverage}.
Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/wikidata_enrich.py [--limit N] [--apply]
"""
import os, re, sys, json, time
from pathlib import Path
from collections import defaultdict

import requests

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "RadarKG-thesis/1.0 (academic KG cross-check; syx7597@gmail.com)"
# Behind the GFW the direct route resets larger foreign payloads; tunnel via Clash.
PROXIES = {"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"}

PMAP = {  # wikidata property -> (schema relation, tail_kind)
    "P176": "developedBy", "P287": "developedBy", "P495": "countryOfOrigin",
    "P17": "countryOfOrigin", "P1365": "replaces", "P1366": "replacedBy",
    "P144": "derivedFrom", "P155": "derivedFrom",
}


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def sparql(names):
    vals = " ".join('"' + n.replace("\\", "").replace('"', "") + '"@en' for n in names)
    props = " ".join(f"OPTIONAL{{?item wdt:{p} ?{p}.}}" for p in PMAP)
    q = f"""SELECT ?item ?name ?p31Label {' '.join(f'?{p} ?{p}Label' for p in PMAP)} WHERE {{
      VALUES ?name {{ {vals} }}
      ?item rdfs:label ?name .
      ?item wdt:P31 ?p31 .
      {props}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    last = None
    for prox in (PROXIES, None):                     # tunnel first, then direct fallback
        for attempt in range(3):
            try:
                r = requests.get(ENDPOINT, params={"query": q, "format": "json"},
                                 headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
                                 proxies=prox, timeout=90)
                r.raise_for_status()
                return r.json()["results"]["bindings"]
            except Exception as e:
                last = e; time.sleep(2 * (attempt + 1))
    raise last


def main(limit=None, apply=False, recompute=False):
    T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    radars = sorted({t["head"] for t in T if t.get("head_type") in ("Radar", "RadarSystem")})
    # prefer the english title where the head matches a corpus doc (better WD labels)
    en_by_norm = {norm(d["en_title"]): d["en_title"] for d in corpus}
    lookup = [en_by_norm.get(norm(h), h) for h in radars]
    lookup = sorted(set(lookup))
    if limit:
        lookup = lookup[:limit]
    print(f"查询 {len(lookup)} 个雷达的 Wikidata 关系...", flush=True)

    # bilingual country canonicalization (avoid zh-vs-en false conflicts: 俄罗斯 vs Russia)
    AL = json.load(open(ROOT / "lexicon" / "entity_aliases.json", encoding="utf-8")).get("countries", {})
    CANON = {}
    for key, info in AL.items():
        for f in [info.get("zh"), info.get("en"), key] + list(info.get("aliases", [])):
            if f:
                CANON[norm(f)] = norm(key)
    for f in ("苏联", "Soviet Union", "USSR", "俄罗斯", "Russia"):   # historical successor bucket
        CANON[norm(f)] = "__ru__"

    def canon(v):
        return CANON.get(norm(v), norm(v))

    def core(s):   # manufacturer canonicalization: drop corporate suffixes + known renames
        s = re.sub(r"\b(corporation|company|corp|inc|incorporated|ltd|limited|gmbh|llc|plc|ag|"
                   r"division|div|group|sector|electronicsystems|electronics|systems|aerospace)\b",
                   "", str(s).lower())
        s = re.sub(r"[^a-z0-9一-鿿]", "", s)
        return {"rtx": "raytheon", "geaerospace": "generalelectric", "ge": "generalelectric",
                "hughesaircraft": "hughes"}.get(s, s)

    def match(rel, wdv, kgset):
        if rel == "countryOfOrigin":
            return canon(wdv) in {canon(k) for k in kgset}
        cw = core(wdv)
        for k in kgset:
            ck = core(k)
            if cw and ck and (cw == ck or (len(cw) >= 5 and (cw in ck or ck in cw))):
                return True
        return False

    # existing KG relations for validation (store raw tails; match() canonicalizes)
    kg = defaultdict(lambda: defaultdict(set))
    for t in T:
        if t["relation"] in ("developedBy", "countryOfOrigin", "replaces", "replacedBy", "derivedFrom"):
            kg[norm(t["head"])][t["relation"]].add(t["tail"])

    RAW = ROOT / "experiments" / "kg_eval" / "wikidata_raw.json"
    wd = defaultdict(lambda: defaultdict(set))     # name -> relation -> {values}
    matched = set()
    if recompute and RAW.exists():
        for k, d in json.load(open(RAW, encoding="utf-8")).items():
            for r, vs in d.items():
                wd[k][r] = set(vs)
            matched.add(k)
        print(f"从缓存重算 {len(wd)} 条(不再联网)")
    else:
        RADARISH = re.compile(r"radar|sensor|detector|warning receiver|sonar|电达|雷达", re.I)
        name_types = defaultdict(set)
        name_props = defaultdict(lambda: defaultdict(set))
        B = 15
        for i in range(0, len(lookup), B):
            batch = lookup[i:i + B]
            try:
                rows = sparql(batch)
            except Exception as e:
                print(f"  batch {i}: ERR {e}", flush=True); time.sleep(2); continue
            for row in rows:
                nm = norm(row["name"]["value"])
                name_types[nm].add(row.get("p31Label", {}).get("value", ""))
                for p, rel in PMAP.items():
                    lab = row.get(p + "Label", {}).get("value")
                    if lab and not re.fullmatch(r"Q\d+", lab):
                        name_props[nm][rel].add(lab)
            print(f"  {min(i+B,len(lookup))}/{len(lookup)}  已扫 {len(name_types)}", flush=True)
            time.sleep(1)
        # gate: keep only names whose Wikidata type looks like a radar/sensor (drops homonyms)
        for nm, types in name_types.items():
            if any(RADARISH.search(t) for t in types if t):
                matched.add(nm)
                for rel, vals in name_props[nm].items():
                    wd[nm][rel] |= vals
        print(f"类型过滤: 扫到 {len(name_types)} -> 确认雷达 {len(matched)}")
        json.dump({k: {r: list(v) for r, v in d.items()} for k, d in wd.items()},
                  open(RAW, "w", encoding="utf-8"), ensure_ascii=False)

    # (a) add candidates, (b) validation, (c) coverage
    add, agree, conflict = [], 0, 0
    conflict_ex = []
    for nm, rels in wd.items():
        for rel, vals in rels.items():
            kgv = kg.get(nm, {}).get(rel, set())
            for v in vals:
                if not kgv:
                    add.append({"head": nm, "relation": rel, "tail": v, "source": "wikidata_enrich"})
                elif match(rel, v, kgv):
                    agree += 1
                else:
                    conflict += 1
                    if len(conflict_ex) < 20:
                        conflict_ex.append(f"{nm} {rel}: KG={list(kgv)} vs WD='{v}'")

    out = {"queried": len(lookup), "matched": len(matched),
           "add_candidates": add, "validation": {"agree": agree, "conflict": conflict,
           "conflict_examples": conflict_ex}}
    json.dump(out, open(ROOT / "experiments" / "kg_eval" / "wikidata_enrich_out.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 56)
    print(f"覆盖: {len(matched)}/{len(lookup)} 个雷达在 Wikidata 命中")
    print(f"(a) 补关系候选: {len(add)} 条(KG 没有、WD 有)")
    addrel = defaultdict(int)
    for a in add:
        addrel[a["relation"]] += 1
    print("     按关系:", dict(addrel))
    print(f"(b) 验证: 一致 {agree}  冲突 {conflict}")
    for e in conflict_ex[:8]:
        print("     冲突:", e)
    print(f"(c) 空白: {len(lookup)-len(matched)} 个雷达在 WD 无匹配(暴露空白象限)")
    print("\n输出 wikidata_enrich_out.json。--apply 未实现(先审 add_candidates 再入库)。")


if __name__ == "__main__":
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    main(limit=lim, apply="--apply" in sys.argv, recompute="--recompute" in sys.argv)
