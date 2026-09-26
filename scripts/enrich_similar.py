"""
Priority-4: enrich `similarTo` and `compatibleWith` edges.

similarTo (radar↔radar): two radars are similar if they belong to the same generation,
   country, role and frequency band. Currently 1 edge.
   Approach:
     1. Candidate generation by feature overlap (country + freq_band + function + tech_type)
     2. Score by # shared features + similar year
     3. Top-N candidate pairs → LLM verifies (yes/no/related_via_family) + confidence
     4. Add edges with source=enrich_similar_llm

compatibleWith (radar→weapon): currently 43 edges. Add by extracting weapon mentions
   from narrative_summary using LLM, normalizing to existing Weapon entities or
   creating new ones.

Writes:
  data/v2/radarkg_v2_edges.json    — appends similarTo / compatibleWith edges
  data/v2/radarkg_v2_entities.json — adds new Weapon entities as needed
  data/v2/enrich_similar_report.md
Caches:
  data/v2/enrich_similar_llm_cache.json
  data/v2/enrich_compat_llm_cache.json
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
ENT_PATH  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
SIM_CACHE = ROOT / "data" / "v2" / "enrich_similar_llm_cache.json"
COMPAT_CACHE = ROOT / "data" / "v2" / "enrich_compat_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_similar_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def _load_cache(p) -> dict:
    if p.exists():
        with open(p, encoding="utf-8") as f: return json.load(f)
    return {}

def _save_cache(p, c):
    with open(p, "w", encoding="utf-8") as f: json.dump(c, f, ensure_ascii=False, indent=2)


_CACHE_LOCK = threading.Lock()


def call_llm(messages, cache, cache_path, max_tokens=300, temperature=0.0) -> str:
    import requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    with _CACHE_LOCK:
        if key in cache:
            return cache[key]
    payload = {"model": "deepseek-chat", "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            with _CACHE_LOCK:
                cache[key] = txt
                if len(cache) % 20 == 0: _save_cache(cache_path, cache)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


# ────────────────────────────────────────────────────────────
#  similarTo
# ────────────────────────────────────────────────────────────

def feature_score(a: dict, b: dict) -> int:
    """Higher = more similar in graph features."""
    if a["country"] != b["country"] or not a["country"]:
        return 0
    s = 1  # same country baseline
    if a["freq_bands"] & b["freq_bands"]: s += 2
    if a["functions"] & b["functions"]: s += 2
    if a["tech_types"] & b["tech_types"]: s += 1
    if a["platforms"] & b["platforms"]: s += 1
    # Same decade bonus
    if a.get("year") and b.get("year") and abs(a["year"] - b["year"]) <= 10:
        s += 1
    # Penalize identical (would be the same entity)
    return s


SIM_SYSTEM = """你是雷达装备分析师。给两个雷达，请判断它们是否"在装备意义上相似"——
即：同一国家研发，相近年代，同一类型（机载火控/预警/防空火控等），频段重叠或同代次。

不需要二者完全相同，只要"用户在比较选型时会把它们视为同类候选"就算 similar。

派生型 / 升级型不算 similarTo（已经有 derivedFrom / upgradeOf 关系覆盖）。

只输出 JSON：
{"similar": true/false, "reason": "<10字解释>", "confidence": 0.0-1.0}
"""


def is_similar_llm(a: dict, b: dict, cache) -> tuple[bool, float, str]:
    user = f"""雷达 A:
  id: {a['id']}
  英文名: {a.get('name_en','')}
  country: {a['country']}
  freq: {sorted(a['freq_bands'])}
  function: {sorted(a['functions'])}
  tech: {sorted(a['tech_types'])}
  year: {a.get('year','?')}
  developer: {a.get('developer','?')}

雷达 B:
  id: {b['id']}
  英文名: {b.get('name_en','')}
  country: {b['country']}
  freq: {sorted(b['freq_bands'])}
  function: {sorted(b['functions'])}
  tech: {sorted(b['tech_types'])}
  year: {b.get('year','?')}
  developer: {b.get('developer','?')}

请判断 similar。"""
    raw = call_llm(
        [{"role": "system", "content": SIM_SYSTEM},
         {"role": "user", "content": user}],
        cache, SIM_CACHE,
        max_tokens=120,
    )
    if raw.startswith("[LLM_ERROR"): return False, 0.0, ""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return False, 0.0, ""
    return bool(data.get("similar")), float(data.get("confidence", 0.6)), str(data.get("reason", ""))


# ────────────────────────────────────────────────────────────
#  compatibleWith (radar → weapon)
# ────────────────────────────────────────────────────────────

COMPAT_SYSTEM = """你是雷达-武器系统分析师。从雷达条目的 narrative 中找出该雷达"能引导/与之搭配工作"的导弹/武器。

**只在文本里明确出现"配合 / 制导 / 引导 / 与...一起 / 搭载 / compatibility / used with X" 等措辞时抽取。**

武器尾节点用通用化的型号名（R-77 / AIM-120 / AIM-7 / R-27 / R-73 / Phoenix / AMRAAM 等）。

只输出 JSON 数组：
[{"weapon":"R-77","evidence":"≤30字原文片段","confidence":0.0-1.0}, ...]
没有就 []。
"""


def extract_compat_llm(radar: dict, cache) -> list:
    nar = (radar.get("narrative_summary") or "")[:1500]
    if len(nar) < 100: return []
    user = f"""雷达: {radar.get('id','')}
country: {radar.get('country_of_origin','')}
narrative: {nar}

抽取武器兼容性。"""
    raw = call_llm(
        [{"role": "system", "content": COMPAT_SYSTEM},
         {"role": "user", "content": user}],
        cache, COMPAT_CACHE,
        max_tokens=300,
    )
    if raw.startswith("[LLM_ERROR"): return []
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        rows = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return []
    valid = []
    for r in rows or []:
        if not isinstance(r, dict): continue
        w = str(r.get("weapon", "")).strip()
        if len(w) < 2 or len(w) > 40: continue
        valid.append({
            "weapon": w,
            "evidence": str(r.get("evidence", ""))[:200],
            "confidence": float(r.get("confidence", 0.7)),
        })
    return valid


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-similar-pairs", type=int, default=500,
                    help="Max LLM calls for similarTo verification")
    ap.add_argument("--max-compat", type=int, default=500,
                    help="Max LLM calls for compatibleWith extraction")
    ap.add_argument("--min-feat-score", type=int, default=4,
                    help="Min feature-overlap score to consider a pair as similarTo candidate")
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--skip-similar", action="store_true")
    ap.add_argument("--skip-compat", action="store_true")
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

    # Build derivedFrom / upgradeOf set so we don't double-emit those as similarTo
    family_pairs: set = set()
    for ed in edges:
        if ed["relation"] in ("derivedFrom", "upgradeOf"):
            family_pairs.add((ed["head"], ed["tail"]))
            family_pairs.add((ed["tail"], ed["head"]))

    # Build platform/edge maps
    platform_of = defaultdict(set)
    for ed in edges:
        if ed["relation"] == "deployedOn":
            platform_of[ed["head"]].add(ed["tail"])

    # Build feature dicts
    feat = {}
    for r in radars:
        feat[r["id"]] = {
            "id": r["id"],
            "name_en": r.get("name_en", ""),
            "country": r.get("country_of_origin", "") or "",
            "developer": r.get("developer", "") or "",
            "freq_bands": set(r.get("frequency_bands", []) or []),
            "tech_types": set(r.get("tech_types", []) or []),
            "functions": set(r.get("functions", []) or []),
            "platforms": platform_of.get(r["id"], set()),
            "year": r.get("year"),
        }

    new_edges = []
    new_entities_added = []

    # ──────  similarTo
    if not args.skip_similar:
        print("\n=== similarTo enrichment ===")
        # Generate candidates
        eligible_ids = [rid for rid, f in feat.items()
                        if f["country"] and (f["freq_bands"] or f["functions"])]
        print(f"  Eligible radars: {len(eligible_ids)}")

        candidates = []
        seen = set()
        for i, a_id in enumerate(eligible_ids):
            for b_id in eligible_ids[i+1:]:
                if (a_id, b_id) in family_pairs: continue
                key = tuple(sorted([a_id, b_id]))
                if key in seen: continue
                seen.add(key)
                score = feature_score(feat[a_id], feat[b_id])
                if score >= args.min_feat_score:
                    candidates.append((score, a_id, b_id))
        candidates.sort(reverse=True)
        candidates = candidates[:args.max_similar_pairs]
        print(f"  Top candidate pairs (score≥{args.min_feat_score}): {len(candidates)}")

        sim_cache = _load_cache(SIM_CACHE)
        n_yes = 0
        progress = {"n": 0}
        prog_lock = threading.Lock()

        def sim_worker(item):
            score, a_id, b_id = item
            ok, conf, reason = is_similar_llm(feat[a_id], feat[b_id], sim_cache)
            with prog_lock:
                progress["n"] += 1
                if progress["n"] % 50 == 0:
                    _save_cache(SIM_CACHE, sim_cache)
                    print(f"  [{progress['n']}/{len(candidates)}] sim verified")
            return score, a_id, b_id, ok, conf, reason

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(sim_worker, c) for c in candidates]
            for fut in as_completed(futures):
                score, a_id, b_id, ok, conf, reason = fut.result()
                if not ok or conf < args.min_conf: continue
                h, t = sorted([a_id, b_id])
                key = (h, "similarTo", t)
                if key in existing_edges: continue
                new_edges.append({
                    "head": h, "head_type": ent_by_id[h]["type"],
                    "relation": "similarTo",
                    "tail": t, "tail_type": ent_by_id[t]["type"],
                    "confidence": round(conf, 2),
                    "evidence": f"enriched similar (score={score}): {reason}"[:200],
                    "source": "enrich_similar_llm",
                })
                existing_edges.add(key)
                n_yes += 1

        _save_cache(SIM_CACHE, sim_cache)
        print(f"  similarTo added: +{n_yes}")

    # ──────  compatibleWith
    if not args.skip_compat:
        print("\n=== compatibleWith enrichment ===")
        compat_cache = _load_cache(COMPAT_CACHE)
        # Pick radars that have narrative + some likelihood of weapon mention
        candidates = [r for r in radars
                      if r.get("narrative_summary") and len(r["narrative_summary"]) > 200
                      and any(kw in r["narrative_summary"]
                              for kw in ("R-", "AIM-", "missile", "导弹", "AMRAAM", "Phoenix",
                                         "Sparrow", "Sidewinder", "MICA", "制导", "搭配"))]
        candidates = candidates[:args.max_compat]
        print(f"  Candidates with weapon-keyword narrative: {len(candidates)}")

        # Existing weapon entities (for normalization)
        weapon_ids = {e["id"]: e for e in entities if e["type"] in ("Weapon", "Missile")}
        weapon_lookup = {}  # canonical lower-case → entity id
        for wid, w in weapon_ids.items():
            weapon_lookup[wid.lower()] = wid
            for al in (w.get("aliases") or []) + [w.get("name_en", "")]:
                if al: weapon_lookup[al.lower()] = wid

        n_compat = 0
        progress = {"n": 0}
        prog_lock = threading.Lock()

        def compat_worker(r):
            extractions = extract_compat_llm(r, compat_cache)
            with prog_lock:
                progress["n"] += 1
                if progress["n"] % 25 == 0:
                    _save_cache(COMPAT_CACHE, compat_cache)
                    print(f"  [{progress['n']}/{len(candidates)}] compat calls")
            return r, extractions

        with ThreadPoolExecutor(max_workers=args.workers) as ex_pool:
            futures = [ex_pool.submit(compat_worker, r) for r in candidates]
            for fut in as_completed(futures):
                r, extractions = fut.result()
                for ex in extractions:
                    if ex["confidence"] < args.min_conf: continue
                    w = ex["weapon"]
                    w_norm = weapon_lookup.get(w.lower())
                    if not w_norm:
                        w_norm = w
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
                        "confidence": round(ex["confidence"], 2),
                        "evidence": ex["evidence"][:200],
                        "source": "enrich_compat_llm",
                    })
                    existing_edges.add(key)
                    n_compat += 1

        _save_cache(COMPAT_CACHE, compat_cache)
        print(f"  compatibleWith added: +{n_compat}, new Weapon entities: +{len(new_entities_added)}")

    # ──────  Save
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

    # ──────  Report
    n_sim = sum(1 for e in edges if e['relation']=='similarTo')
    n_compat = sum(1 for e in edges if e['relation']=='compatibleWith')
    report = f"""# similarTo / compatibleWith Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage

| Relation | Before | After | Delta |
|---|---|---|---|
| similarTo | 1 | {n_sim} | +{n_sim - 1} |
| compatibleWith | 43 | {n_compat} | +{n_compat - 43} |

## New entities added

- Weapons: +{len(new_entities_added)}

## Sample edges

### similarTo
"""
    for e in [e for e in new_edges if e['relation']=='similarTo'][:15]:
        report += f"- `{e['head']}` ↔ `{e['tail']}`  ({e.get('evidence','')[:60]})\n"
    report += "\n### compatibleWith\n"
    for e in [e for e in new_edges if e['relation']=='compatibleWith'][:15]:
        report += f"- `{e['head']}` → `{e['tail']}`  ({e.get('evidence','')[:60]})\n"

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {REPORT_PATH}")

    print(f"""
────── enrichment summary ──────
similarTo:       {n_sim} edges (was 1)
compatibleWith:  {n_compat} edges (was 43)
new entities:    +{len(new_entities_added)}
""")


if __name__ == "__main__":
    main()
