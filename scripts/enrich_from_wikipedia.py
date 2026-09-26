"""
Priority-5: enrich the v2 KG by pulling Wikipedia article summaries and
LLM-extracting structured data (numeric specs, platforms, weapon compatibility).

Wikipedia summaries have richer prose than Wikidata SPARQL queries — they typically
contain frequency, peak power, range, deployment, and weapon compatibility.

Approach:
  1. For each radar entity in v2 KG, attempt Wikipedia lookup (en first, then zh).
     - Try id, name_en, aliases as search terms.
     - Use https://en.wikipedia.org/api/rest_v1/page/summary/{title}
  2. If a summary is found, LLM-extract:
       country, developer, year, frequency_GHz_min/max, range_km, peak_power_kW,
       weight_kg, platforms[], compatible_weapons[]
  3. Update entity attributes (don't overwrite existing) + add new edges.

Cache: data/v2/enrich_wikipedia_cache.json (article summaries, separate from LLM)
       data/v2/enrich_wikipedia_llm_cache.json (LLM extractions)
Report: data/v2/enrich_wikipedia_report.md
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
WIKI_CACHE = ROOT / "data" / "v2" / "enrich_wikipedia_cache.json"
LLM_CACHE = ROOT / "data" / "v2" / "enrich_wikipedia_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_wikipedia_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

USER_AGENT = "RadarKG-Research-Bot/1.0 (academic; contact: research@example.com)"

VALID_RANGES = {
    "frequency_GHz":     (0.05, 300.0),
    "frequency_GHz_min": (0.05, 300.0),
    "frequency_GHz_max": (0.05, 300.0),
    "range_km":          (0.1, 10000.0),
    "peak_power_kW":     (0.001, 100000.0),
    "year":              (1930, 2030),
    "weight_kg":         (0.5, 50000.0),
}


def _load_cache(p):
    if p.exists():
        with open(p, encoding="utf-8") as f: return json.load(f)
    return {}

def _save_cache(p, c):
    with open(p, "w", encoding="utf-8") as f: json.dump(c, f, ensure_ascii=False, indent=2)


_WIKI_CACHE = _load_cache(WIKI_CACHE)
_LLM_CACHE = _load_cache(LLM_CACHE)
_LOCK = threading.Lock()


# ────────────────────────────────────────────────────────────
#  Wikipedia summary fetcher
# ────────────────────────────────────────────────────────────

def fetch_wiki_summary(title: str, lang: str = "en") -> dict | None:
    """Try Wikipedia REST summary API. Returns {extract,description,title,url} or None."""
    cache_key = f"{lang}:{title}"
    with _LOCK:
        if cache_key in _WIKI_CACHE:
            return _WIKI_CACHE[cache_key]
    import requests
    enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{enc}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Skip disambiguation
            if data.get("type") == "disambiguation":
                result = None
            else:
                result = {
                    "title":       data.get("title", ""),
                    "description": data.get("description", ""),
                    "extract":     data.get("extract", ""),
                    "url":         data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
        elif r.status_code == 404:
            result = None
        else:
            result = None
    except Exception:
        result = None
    with _LOCK:
        _WIKI_CACHE[cache_key] = result
        if len(_WIKI_CACHE) % 30 == 0: _save_cache(WIKI_CACHE, _WIKI_CACHE)
    return result


def search_wiki(radar: dict) -> dict | None:
    """Try at most 2 lookups per radar to keep cost bounded."""
    # Build a small ordered candidate list, dedup, max 2 entries
    candidates = []
    seen = set()
    def push(lang, t):
        if not t or not isinstance(t, str): return
        t = t.strip()
        if len(t) < 3 or len(t) > 60: return
        key = (lang, t)
        if key in seen: return
        seen.add(key)
        candidates.append((lang, t))

    push("en", radar.get("id", ""))
    name_en = radar.get("name_en", "")
    if name_en != radar.get("id"):
        push("en", name_en)

    candidates = candidates[:2]   # cap

    for lang, t in candidates:
        # Strip 雷达 / "radar" suffix to widen match
        t_clean = re.sub(r"\s*(?:radar|雷达)\s*$", "", t, flags=re.I).strip()
        for query in (t,) if t_clean == t else (t, t_clean):
            res = fetch_wiki_summary(query, lang=lang)
            if res and res.get("extract") and len(res["extract"]) > 80:
                ext_l = res["extract"].lower()
                if any(k in ext_l for k in ("radar", "雷达", "antenna", "transmitter", "frequency", "波段", "天线")):
                    return res
    return None


# ────────────────────────────────────────────────────────────
#  LLM extraction
# ────────────────────────────────────────────────────────────

def call_llm(messages, max_tokens=600) -> str:
    import requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    with _LOCK:
        if key in _LLM_CACHE: return _LLM_CACHE[key]
    payload = {"model": "deepseek-chat", "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            with _LOCK:
                _LLM_CACHE[key] = txt
                if len(_LLM_CACHE) % 20 == 0: _save_cache(LLM_CACHE, _LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


SYSTEM = """你是雷达情报分析师。给定一个 Wikipedia 摘要（雷达条目），请抽取以下结构化字段：
- country         : 研发国（中文国名：美国/俄罗斯/中国/英国/法国/德国/意大利/以色列/瑞典/日本/印度/...）
- developer       : 制造商英文名
- year            : 服役/部署/首飞年份（YYYY）
- frequency_GHz   : 中心频率（GHz）
- frequency_GHz_min : 最低频率（GHz）
- frequency_GHz_max : 最高频率（GHz）
- range_km        : 探测距离 / 最大作用距离（km）
- peak_power_kW   : 峰值功率（kW）
- weight_kg       : 重量（kg）
- platforms       : 部署平台清单（飞机/舰船/SAM 系统名）
- compatible_weapons : 该雷达制导的导弹/武器列表

**规则：**
- 只在文本明确提到时给出。模糊→留 null/空数组。不要瞎猜。
- 单位换算（MHz→GHz, W→kW, m/海里→km）。
- platforms / compatible_weapons 用通用化型号（F-16, MiG-29, Su-27, AIM-120, R-77 等）。

只输出 JSON：
{
 "country":"...","developer":"...","year":...,
 "frequency_GHz":...,"frequency_GHz_min":...,"frequency_GHz_max":...,
 "range_km":...,"peak_power_kW":...,"weight_kg":...,
 "platforms":[...],"compatible_weapons":[...],
 "confidence": 0.0-1.0
}
未知字段写 null 或 []。
"""


def llm_extract(radar_id: str, summary: dict) -> dict:
    extract = summary.get("extract", "")[:2000]
    user = f"""雷达 ID: {radar_id}
Wikipedia 标题: {summary.get('title','')}
Wikipedia description: {summary.get('description','')}
摘要内容:
{extract}

请返回 JSON。"""
    raw = call_llm(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=600,
    )
    if raw.startswith("[LLM_ERROR"): return {}
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data


COUNTRY_NORM = {
    "United States": "美国", "USA": "美国", "U.S.": "美国", "美国": "美国",
    "Soviet Union": "俄罗斯", "USSR": "俄罗斯", "Russia": "俄罗斯", "苏联": "俄罗斯", "俄罗斯": "俄罗斯",
    "China": "中国", "PRC": "中国", "中国": "中国",
    "United Kingdom": "英国", "UK": "英国", "Britain": "英国", "英国": "英国",
    "France": "法国", "法国": "法国",
    "Germany": "德国", "West Germany": "德国", "德国": "德国",
    "Italy": "意大利", "意大利": "意大利",
    "Israel": "以色列", "以色列": "以色列",
    "Japan": "日本", "日本": "日本",
    "Sweden": "瑞典", "瑞典": "瑞典",
    "Netherlands": "荷兰", "荷兰": "荷兰",
    "India": "印度", "印度": "印度",
    "Canada": "加拿大", "加拿大": "加拿大",
    "Australia": "澳大利亚", "澳大利亚": "澳大利亚",
    "Turkey": "土耳其", "土耳其": "土耳其",
    "Spain": "西班牙", "西班牙": "西班牙",
    "Norway": "挪威", "挪威": "挪威",
    "Switzerland": "瑞士", "瑞士": "瑞士",
}

def norm_country(s: str) -> str:
    if not s: return ""
    return COUNTRY_NORM.get(s.strip(), s.strip())


def _f(x, field):
    if x is None: return None
    try: f = float(x)
    except (TypeError, ValueError): return None
    lo, hi = VALID_RANGES.get(field, (-1e18, 1e18))
    if not (lo <= f <= hi): return None
    if field == "year": return int(f)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-radars", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("Loading v2 KG…")
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]
    ent_by_id = {e["id"]: e for e in entities}
    radars = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    print(f"  {len(entities)} entities, {len(edges)} edges, {len(radars)} radars")

    existing_edges = {(e["head"], e["relation"], e["tail"]) for e in edges}

    radars_to_process = radars[:args.max_radars]
    print(f"  Will probe Wikipedia for {len(radars_to_process)} radars")

    # ── Step 1: Wikipedia probing (parallel)
    print("\n=== Step 1: Wikipedia probing ===")
    summary_cache = {}    # radar_id → summary dict | None
    progress = {"n": 0, "found": 0}
    plock = threading.Lock()

    def probe(r):
        s = search_wiki(r)
        with plock:
            progress["n"] += 1
            if s: progress["found"] += 1
            if progress["n"] % 50 == 0:
                _save_cache(WIKI_CACHE, _WIKI_CACHE)
                print(f"  [{progress['n']}/{len(radars_to_process)}] probed, found={progress['found']}")
        return r["id"], s

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(probe, r) for r in radars_to_process]
        for fut in as_completed(futures):
            rid, s = fut.result()
            if s: summary_cache[rid] = s

    _save_cache(WIKI_CACHE, _WIKI_CACHE)
    print(f"  Total Wikipedia hits: {len(summary_cache)}/{len(radars_to_process)}")

    # ── Step 2: LLM extract on hits (parallel)
    print(f"\n=== Step 2: LLM extract on {len(summary_cache)} Wikipedia hits ===")
    extractions = {}    # radar_id → extracted dict
    progress = {"n": 0}
    plock = threading.Lock()

    def extract(rid, summary):
        d = llm_extract(rid, summary)
        with plock:
            progress["n"] += 1
            if progress["n"] % 25 == 0:
                _save_cache(LLM_CACHE, _LLM_CACHE)
                print(f"  [{progress['n']}/{len(summary_cache)}] extracted")
        return rid, d

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(extract, rid, s) for rid, s in summary_cache.items()]
        for fut in as_completed(futures):
            rid, d = fut.result()
            if d: extractions[rid] = d

    _save_cache(LLM_CACHE, _LLM_CACHE)

    # ── Step 3: apply
    print("\n=== Step 3: apply ===")
    new_edges = []
    new_entities = []
    field_filled = Counter()
    new_compat = 0
    new_deployed = 0
    new_country = 0
    new_developer = 0

    # Lookups for normalization
    weapon_lookup = {e["id"].lower(): e["id"] for e in entities if e["type"] in ("Weapon", "Missile")}
    platform_lookup = {e["id"].lower(): e["id"] for e in entities if e["type"] in ("Platform", "NavalVessel", "AircraftPlatform", "GroundPlatform", "Aircraft")}

    for rid, d in extractions.items():
        r = ent_by_id.get(rid)
        if not r: continue
        conf = float(d.get("confidence", 0.5) or 0.5)
        if conf < args.min_conf: continue

        # Numeric attrs
        for field in ("frequency_GHz", "frequency_GHz_min", "frequency_GHz_max",
                      "range_km", "peak_power_kW", "year", "weight_kg"):
            v = _f(d.get(field), field)
            if v is not None and not r.get(field):
                r[field] = v
                field_filled[field] += 1

        # country
        country = norm_country(d.get("country", "") or "")
        if country and not r.get("country_of_origin"):
            r["country_of_origin"] = country
            new_country += 1
            key = (rid, "countryOfOrigin", country)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "countryOfOrigin",
                    "tail": country, "tail_type": "Country",
                    "confidence": round(conf, 2),
                    "evidence": "wikipedia: " + d.get("country", "")[:50],
                    "source": "enrich_wiki",
                })
                existing_edges.add(key)
                if country not in ent_by_id:
                    ent_by_id[country] = {"id": country, "type": "Country",
                                           "name_zh": country, "name_en": country, "aliases": []}
                    entities.append(ent_by_id[country])
                    new_entities.append(country)

        # developer
        dev = (d.get("developer") or "").strip()
        if dev and not r.get("developer"):
            r["developer"] = dev
            new_developer += 1
            key = (rid, "developedBy", dev)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "developedBy",
                    "tail": dev, "tail_type": "Manufacturer",
                    "confidence": round(conf, 2),
                    "evidence": "wikipedia",
                    "source": "enrich_wiki",
                })
                existing_edges.add(key)
                if dev not in ent_by_id:
                    ent_by_id[dev] = {"id": dev, "type": "Manufacturer",
                                       "name_en": dev, "aliases": []}
                    entities.append(ent_by_id[dev])
                    new_entities.append(dev)

        # platforms (deployedOn edges)
        plats = d.get("platforms", []) or []
        for p in plats[:6]:
            if not p or not isinstance(p, str): continue
            p = p.strip()
            if len(p) < 2 or len(p) > 50: continue
            p_norm = platform_lookup.get(p.lower(), p)
            if p_norm not in ent_by_id:
                ent_by_id[p_norm] = {"id": p_norm, "type": "Platform",
                                      "name_en": p_norm, "aliases": []}
                entities.append(ent_by_id[p_norm])
                new_entities.append(p_norm)
                platform_lookup[p_norm.lower()] = p_norm
            key = (rid, "deployedOn", p_norm)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "deployedOn",
                    "tail": p_norm, "tail_type": "Platform",
                    "confidence": round(conf, 2),
                    "evidence": "wikipedia",
                    "source": "enrich_wiki",
                })
                existing_edges.add(key)
                new_deployed += 1

        # compatible_weapons
        weapons = d.get("compatible_weapons", []) or []
        for w in weapons[:8]:
            if not w or not isinstance(w, str): continue
            w = w.strip()
            if len(w) < 2 or len(w) > 40: continue
            w_norm = weapon_lookup.get(w.lower(), w)
            if w_norm not in ent_by_id:
                ent_by_id[w_norm] = {"id": w_norm, "type": "Weapon",
                                      "name_en": w_norm, "aliases": []}
                entities.append(ent_by_id[w_norm])
                new_entities.append(w_norm)
                weapon_lookup[w_norm.lower()] = w_norm
            key = (rid, "compatibleWith", w_norm)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "compatibleWith",
                    "tail": w_norm, "tail_type": "Weapon",
                    "confidence": round(conf, 2),
                    "evidence": "wikipedia",
                    "source": "enrich_wiki",
                })
                existing_edges.add(key)
                new_compat += 1

    # ── Save
    edges.extend(new_edges)
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)
    ent_data["entities"] = entities
    ent_data["count"] = len(entities)

    if not args.dry_run:
        with open(EDGE_PATH, "w", encoding="utf-8") as f:
            json.dump(edge_data, f, ensure_ascii=False, indent=2)
        with open(ENT_PATH, "w", encoding="utf-8") as f:
            json.dump(ent_data, f, ensure_ascii=False, indent=2)

    radars_after = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    report = f"""# Wikipedia Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage

- Wikipedia hits: {len(summary_cache)} / {len(radars_to_process)} probed
- LLM extractions: {len(extractions)}
- New edges added: {len(new_edges)}
- New entities: {len(new_entities)}

## New attribute fills

{chr(10).join(f"- `{f}`: +{c}" for f, c in field_filled.most_common())}
- countryOfOrigin: +{new_country}
- developedBy: +{new_developer}
- deployedOn (edges): +{new_deployed}
- compatibleWith (edges): +{new_compat}
"""

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {REPORT_PATH}")

    print(f"""
────── Wikipedia enrichment summary ──────
wiki hits:           {len(summary_cache)}
llm extractions:     {len(extractions)}
new edges:           +{len(new_edges)}
new entities:        +{len(new_entities)}

attribute fills:
""" + "\n".join(f"  {f:25s} +{c}" for f, c in field_filled.most_common()))


if __name__ == "__main__":
    main()
