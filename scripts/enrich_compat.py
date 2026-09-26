"""
Priority-4 (compat half): enrich `compatibleWith` (radar→weapon) edges using LLM
parametric knowledge for fighter / SAM / interceptor radars whose weapon
compatibility is well-documented.

Approach:
  - Filter radars to those that look like fire-control radars OR are deployed on
    weapon-bearing platforms (fighters, SAM systems)
  - LLM is given radar id + name + country + developer + functions + platform list
    and asked which missiles/weapons this radar guides — drawing on training data.
  - Each returned weapon is normalized; new Weapon entities are auto-created.

Cache:  data/v2/enrich_compat_llm_cache.json
Writes: edges with source=enrich_compat_llm
Report: data/v2/enrich_compat_report.md
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import threading
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
CACHE_PATH = ROOT / "data" / "v2" / "enrich_compat_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_compat_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f: return json.load(f)
    return {}

def _save_cache(c):
    with open(CACHE_PATH, "w", encoding="utf-8") as f: json.dump(c, f, ensure_ascii=False, indent=2)

_LLM_CACHE = _load_cache()
_CACHE_LOCK = threading.Lock()


def call_llm(messages, max_tokens=400, temperature=0.0) -> str:
    import requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    with _CACHE_LOCK:
        if key in _LLM_CACHE: return _LLM_CACHE[key]
    payload = {"model": "deepseek-chat", "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            with _CACHE_LOCK:
                _LLM_CACHE[key] = txt
                if len(_LLM_CACHE) % 20 == 0: _save_cache(_LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


SYSTEM = """你是雷达-导弹武器系统专家。给定一个雷达型号（通常是机载火控雷达 / 舰载火控雷达 / 防空雷达），
请基于你的领域知识列出：该雷达**实际能够引导/制导/支持发射**的导弹或武器型号清单。

**规则：**
1. 仅在你对该雷达-武器组合有可靠把握时给出（如 AN/APG-77 → AIM-120、AIM-9X；AN/AWG-9 → AIM-54、AIM-7、AIM-9；F-16 APG-68 → AIM-120、AIM-7、AIM-9；S-300 雷达 → 5V55、48N6 系列；EL/M-2032 → Python-4/5、Derby、AMRAAM）。
2. 不确定 → 返回空数组。不要瞎猜或泛化。
3. weapon 用通用化型号（AIM-120 / AIM-9X / AIM-7M / R-77 / R-73 / Python-5 / 48N6 / RIM-66 等）；不要写 "中距空空导弹" 这种泛称。
4. 目标 4-8 个最权威的型号即可；多变体可以合并为系列名（如 AIM-120 而非 AIM-120A/B/C）。

只输出 JSON 数组：
[{"weapon":"AIM-120","confidence":0.0-1.0,"note":"<10字补充>"}, ...]
不要解释，没有就 []。
"""


def llm_extract(radar: dict, platforms: list[str]) -> list:
    user = f"""雷达 ID: {radar.get('id','')}
英文名: {radar.get('name_en','')}
country_of_origin: {radar.get('country_of_origin','')}
developer: {radar.get('developer','')}
functions: {', '.join(radar.get('functions',[]) or [])}
deployedOn: {', '.join(platforms)}

列出该雷达能引导的武器。"""
    raw = call_llm(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=300,
    )
    if raw.startswith("[LLM_ERROR"): return []
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        rows = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list): return []
    out = []
    for r in rows:
        if not isinstance(r, dict): continue
        w = str(r.get("weapon", "")).strip()
        if len(w) < 2 or len(w) > 40: continue
        out.append({
            "weapon": w,
            "confidence": float(r.get("confidence", 0.7)),
            "note": str(r.get("note", ""))[:40],
        })
    return out[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-llm", type=int, default=10000)
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=10)
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

    # Build platform map for each radar
    platform_of = defaultdict(list)
    for ed in edges:
        if ed["relation"] == "deployedOn":
            platform_of[ed["head"]].append(ed["tail"])

    # Filter: radars that look like they'd guide weapons
    weapon_funcs = {"fire_control", "missile_guidance", "air_intercept",
                    "air_defense", "missile_defense", "ground_attack",
                    "naval_gunnery", "weapon_locating", "torpedo_guidance"}
    candidates = []
    for r in radars:
        funcs = set(r.get("functions", []) or [])
        if not funcs & weapon_funcs: continue
        if not r.get("country_of_origin") or not r.get("developer"): continue
        candidates.append(r)
    print(f"  Candidates (weapon-related fn + country + developer): {len(candidates)}")
    candidates = candidates[:args.max_llm]

    # Existing weapon entities for normalization
    weapon_lookup = {}
    for e in entities:
        if e["type"] in ("Weapon", "Missile"):
            weapon_lookup[e["id"].lower()] = e["id"]
            for al in (e.get("aliases") or []) + [e.get("name_en", "")]:
                if al: weapon_lookup[al.lower()] = e["id"]

    new_edges = []
    new_entities_added = []
    progress = {"n": 0}
    plock = threading.Lock()

    def worker(r):
        plats = platform_of.get(r["id"], [])
        rows = llm_extract(r, plats)
        with plock:
            progress["n"] += 1
            if progress["n"] % 25 == 0:
                _save_cache(_LLM_CACHE)
                print(f"  [{progress['n']}/{len(candidates)}] compat calls")
        return r, rows

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r) for r in candidates]
        for fut in as_completed(futures):
            r, rows = fut.result()
            for row in rows:
                if row["confidence"] < args.min_conf: continue
                w = row["weapon"]
                w_norm = weapon_lookup.get(w.lower(), w)
                if w_norm not in ent_by_id:
                    ent_by_id[w_norm] = {
                        "id": w_norm, "type": "Weapon",
                        "name_en": w_norm, "aliases": [],
                    }
                    entities.append(ent_by_id[w_norm])
                    new_entities_added.append(w_norm)
                    weapon_lookup[w_norm.lower()] = w_norm
                key = (r["id"], "compatibleWith", w_norm)
                if key in existing_edges: continue
                new_edges.append({
                    "head": r["id"], "head_type": r["type"],
                    "relation": "compatibleWith",
                    "tail": w_norm, "tail_type": "Weapon",
                    "confidence": round(row["confidence"], 2),
                    "evidence": f"LLM domain knowledge: {row.get('note','')}"[:200],
                    "source": "enrich_compat_llm",
                })
                existing_edges.add(key)

    _save_cache(_LLM_CACHE)

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

    n_compat_total = sum(1 for e in edges if e["relation"] == "compatibleWith")
    by_w = Counter(e["tail"] for e in new_edges)
    by_radar = Counter(e["head"] for e in new_edges)

    report = f"""# compatibleWith Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage

- compatibleWith edges (before): 43
- compatibleWith edges (after):  {n_compat_total}
- New edges added:               +{len(new_edges)}
- New Weapon entities:           +{len(new_entities_added)}

## Top weapons referenced

{chr(10).join(f"- `{w}`: {c}" for w, c in by_w.most_common(20))}

## Radars with most weapons

{chr(10).join(f"- `{r}`: {c}" for r, c in by_radar.most_common(10))}

## Sample edges

{chr(10).join(f"- `{e['head']}` → `{e['tail']}`  (conf={e['confidence']}, {e['evidence'][:50]})" for e in new_edges[:20])}
"""

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)

    print(f"""
────── compatibleWith enrichment ──────
edges added:       +{len(new_edges)}
total compatible:  {n_compat_total}
new weapons:       +{len(new_entities_added)}
LLM calls:         {progress['n']}
""")


if __name__ == "__main__":
    main()
