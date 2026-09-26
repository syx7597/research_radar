"""
S2b Wikidata 结构化源：对 kg_v3 的雷达查 Wikidata，产出高置信关系边（金标 92%）。

与 S2 同属"确定性/结构化"层（不走 LLM 抽取）。Wikidata 是独立第三方源，
主要补 developedBy / countryOfOrigin / replaces / derivedFrom，并天然提供多源印证。

复用 experiments/kg_eval/wikidata_enrich.py 的 SPARQL + RADARISH 门控思路：
  - 按雷达英文名批量查，仅保留 Wikidata 类型像雷达/传感器的（去同名歧义）
  - country 归一到 lexicon 中文规范名（与 v3 其他边一致）
缓存 work/s2b_wikidata_raw.json，重跑免联网。

输出:  work/s2b_wikidata_edges.jsonl
运行:  python pipeline/v3/s2b_wikidata.py [--limit N]
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))
import schema as S  # noqa: E402

WORK = ROOT / "pipeline" / "v3" / "work"
KGV3 = ROOT / "kg_v3"
RAW  = WORK / "s2b_wikidata_raw.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "radar-kg-research/1.0 (academic; contact syx7597)"

PMAP = {"P176": ("developedBy", "Manufacturer"), "P287": ("developedBy", "Manufacturer"),
        "P495": ("countryOfOrigin", "Country"), "P17": ("countryOfOrigin", "Country"),
        "P1365": ("replaces", "RadarSystem"), "P1366": ("replacedBy", "RadarSystem"),
        "P144": ("derivedFrom", "RadarSystem"), "P155": ("derivedFrom", "RadarSystem")}
RADARISH = re.compile(r"radar|sensor|detector|warning receiver|sonar|雷达", re.I)


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def sparql(names):
    vals = " ".join('"' + n.replace("\\", "").replace('"', "") + '"@en' for n in names)
    props = " ".join(f"OPTIONAL{{?item wdt:{p} ?{p}.}}" for p in PMAP)
    q = f"""SELECT ?item ?name ?p31Label {' '.join(f'?{p}Label' for p in PMAP)} WHERE {{
      VALUES ?name {{ {vals} }}
      ?item rdfs:label ?name .
      ?item wdt:P31 ?p31 .
      {props}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    last = None
    for attempt in range(3):
        try:
            r = requests.get(ENDPOINT, params={"query": q, "format": "json"},
                             headers={"User-Agent": UA,
                                      "Accept": "application/sparql-results+json"},
                             timeout=90)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def country_canon(v):
    return S.resolve_alias(v, "Country")


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    ents = json.loads((KGV3 / "entities.json").read_text(encoding="utf-8"))
    radars = sorted({e["name"] for e in ents if e["type"] == "Radar"})
    # 只查 ascii 名（Wikidata 英文 label 匹配；中文型号命中率低且拖慢）
    lookup = [r for r in radars if re.search(r"[A-Za-z]", r) and len(r) <= 50]
    if limit:
        lookup = lookup[:limit]
    canon_by_norm = {norm(e["name"]): e["name"] for e in ents if e["type"] == "Radar"}
    # 查询变体：原名 + 去 radar/system 后缀（提高与 Wikidata label 的匹配率）
    query_to_canon = {}
    for r in lookup:
        query_to_canon.setdefault(r, r)
        variants = set()
        # 去 radar/system 后缀
        variants.add(re.sub(r"\s+(radar|radar system|system)$", "", r, flags=re.I).strip())
        # 引号内 NATO 代号/别名（'AN/FPS-95 "Cobra Mist"' → Cobra Mist + AN/FPS-95）
        for q in re.findall(r"[\"“”]([^\"“”]{2,30})[\"“”]", r):
            variants.add(q.strip())
        # 括号内代号（'AN/SPS-38 (XN-1)' → XN-1）+ 去引号/括号的主型号
        for q in re.findall(r"[\(（]([^)）]{2,30})[\)）]", r):
            variants.add(q.strip())
        variants.add(re.sub(r"[\"“”].*?[\"“”]|[\(（].*?[\)）]", "", r).strip())
        for v in variants:
            if v and v != r and 2 <= len(v) <= 45:
                query_to_canon.setdefault(v, r)
                canon_by_norm.setdefault(norm(v), r)
    lookup = sorted(query_to_canon)
    print(f"[S2b] 查询 {len(lookup)} 个名称变体的 Wikidata 关系...", flush=True)

    name_types = defaultdict(set)
    name_props = defaultdict(lambda: defaultdict(set))
    if RAW.exists():
        cached = json.loads(RAW.read_text(encoding="utf-8"))
        for nm, d in cached.items():
            name_types[nm] = set(d.get("_types", []))
            for rel, vs in d.items():
                if rel != "_types":
                    name_props[nm][rel] = set(vs)
        print(f"[S2b] 缓存已有 {len(cached)} 条")

    todo = [n for n in lookup if norm(n) not in name_types]
    B = 15
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        try:
            rows = sparql(batch)
        except Exception as e:
            print(f"  batch {i}: ERR {e}", flush=True)
            continue
        for row in rows:
            nm = norm(row["name"]["value"])
            name_types[nm].add(row.get("p31Label", {}).get("value", ""))
            for p, (rel, _) in PMAP.items():
                lab = row.get(p + "Label", {}).get("value")
                if lab and not re.fullmatch(r"Q\d+", lab):
                    name_props[nm][rel].add(lab)
        if (i // B) % 5 == 0:
            print(f"  {min(i+B,len(todo))}/{len(todo)} scanned {len(name_types)}", flush=True)
            RAW.write_text(json.dumps(
                {nm: {"_types": sorted(name_types[nm]),
                      **{r: sorted(v) for r, v in name_props[nm].items()}}
                 for nm in name_types}, ensure_ascii=False), encoding="utf-8")
        time.sleep(1)
    RAW.write_text(json.dumps(
        {nm: {"_types": sorted(name_types[nm]),
              **{r: sorted(v) for r, v in name_props[nm].items()}}
         for nm in name_types}, ensure_ascii=False), encoding="utf-8")

    out = (WORK / "s2b_wikidata_edges.jsonl").open("w", encoding="utf-8")
    n_edge = n_radar = 0
    rel_tail = {r: t for r, t in PMAP.values()}
    for nm, types in name_types.items():
        if not any(RADARISH.search(t) for t in types if t):   # 类型门控去同名歧义
            continue
        head = canon_by_norm.get(nm)
        if not head:
            continue
        got = False
        for rel, vals in name_props[nm].items():
            ttype = rel_tail[rel]
            for v in vals:
                tail = country_canon(v) if ttype == "Country" else v
                out.write(json.dumps({
                    "head": head, "head_type": "Radar", "relation": rel,
                    "tail": tail, "tail_type": ttype,
                    "evidence": f"Wikidata: {rel}={v}", "tier": "v3_wikidata",
                    "source_kind": "wikidata", "doc_id": "wikidata"},
                    ensure_ascii=False) + "\n")
                n_edge += 1
                got = True
        n_radar += got
    out.close()
    print(f"[S2b] 命中雷达 {n_radar}, 产出边 {n_edge} -> s2b_wikidata_edges.jsonl")


if __name__ == "__main__":
    main()
