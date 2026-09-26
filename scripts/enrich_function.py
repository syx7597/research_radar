"""
Priority-2 enrichment: fill hasFunction relation for Radar / RadarSystem.

Currently 26 edges (1.2% of 888 radars). Target ≥50% with multi-function support.

Approach: LLM batch over every radar. Provide a canonical Function vocabulary
and ask the model to pick the 1-4 best fits given the radar's name + narrative
+ modes + tech_types + operator. Strict JSON output.

Writes:
  data/v2/radarkg_v2_entities.json   — sets `functions` attribute (list)
  data/v2/radarkg_v2_edges.json      — adds hasFunction edges (source=enrich_func_llm)
                                       and creates Function entities as needed
Cache:
  data/v2/enrich_function_llm_cache.json
Report:
  data/v2/enrich_function_report.md
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse

# ensure utf-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
CACHE_PATH = ROOT / "data" / "v2" / "enrich_function_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_function_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ────────────────────────────────────────────────────────────
#  Canonical Function vocabulary (id → display)
# ────────────────────────────────────────────────────────────
# Each function has a stable English ID. Aliases will be normalized to the ID.
FUNCTION_CANON = {
    # Air-to-air
    "air_search":           "空中目标搜索",
    "air_track":            "空中目标跟踪",
    "fire_control":         "火控",
    "missile_guidance":     "导弹制导",
    "IFF":                  "敌我识别",
    "air_intercept":        "空中截击",
    # Air-to-ground / sea
    "ground_mapping":       "地形测绘",
    "ground_attack":        "对地攻击",
    "sea_search":           "海面搜索",
    "surface_search":       "水面/海面搜索",
    "terrain_following":    "地形跟随",
    "terrain_avoidance":    "地形回避",
    "weather":              "气象探测",
    "navigation":           "导航",
    "altimeter":            "雷达测高",
    # Strategic / defense
    "early_warning":        "预警",
    "air_defense":          "防空",
    "missile_defense":      "反导",
    "battlefield_surveillance": "战场监视",
    "counter_battery":      "反炮兵",
    "airspace_surveillance": "空域监视",
    "OTH":                  "超视距探测",
    # Civil / specialized
    "ATC":                  "空中交通管制",
    "ASR":                  "机场监视",
    "imaging_SAR":          "合成孔径成像",
    "ISAR":                 "逆合成孔径成像",
    "GPR":                  "探地",
    "weapon_locating":      "武器定位",
    "target_designation":   "目标指示",
    "target_tracking":      "目标跟踪",
    "target_identification": "目标识别",
    "reconnaissance":       "侦察",
    "naval_gunnery":        "舰炮火控",
    "torpedo_guidance":     "鱼雷制导",
    "test_range":           "靶场测控",
}

# Aliases (lowercased English / Chinese phrases) → canonical id
FUNCTION_ALIASES = {
    "air-to-air combat": "air_intercept",
    "air to air combat": "air_intercept",
    "air-to-air": "air_intercept",
    "a/a": "air_intercept",
    "long-range search": "early_warning",
    "long range search": "early_warning",
    "long-range surveillance": "airspace_surveillance",
    "search": "air_search",
    "search radar": "air_search",
    "air search": "air_search",
    "tracking": "air_track",
    "track": "air_track",
    "target tracking": "target_tracking",
    "target detection": "air_search",
    "target identification": "target_identification",
    "target designation": "target_designation",
    "missile tracking": "missile_guidance",
    "missile guidance": "missile_guidance",
    "weapon guidance": "missile_guidance",
    "fire control": "fire_control",
    "fire-control": "fire_control",
    "gun control": "naval_gunnery",
    "gunnery": "naval_gunnery",
    "altitude-finding": "altimeter",
    "altitude finding": "altimeter",
    "height finder": "altimeter",
    "height-finder": "altimeter",
    "warning": "early_warning",
    "early warning": "early_warning",
    "synthetic aperture radar mapping": "imaging_SAR",
    "synthetic aperture": "imaging_SAR",
    "sar mapping": "imaging_SAR",
    "sar": "imaging_SAR",
    "ground mapping": "ground_mapping",
    "mapping": "ground_mapping",
    "air-to-ground attack": "ground_attack",
    "air-to-ground targeting": "ground_attack",
    "air-to-surface attack": "ground_attack",
    "air to ground": "ground_attack",
    "surface attack": "ground_attack",
    "ground target attack": "ground_attack",
    "reconnaissance": "reconnaissance",
    "recon": "reconnaissance",
    "air defense": "air_defense",
    "air-defense": "air_defense",
    "missile defense": "missile_defense",
    "anti-missile": "missile_defense",
    "abm": "missile_defense",
    "battlefield surveillance": "battlefield_surveillance",
    "counter-battery": "counter_battery",
    "counter battery": "counter_battery",
    "weapon locating": "weapon_locating",
    "weapon-locating": "weapon_locating",
    "atc": "ATC",
    "air traffic control": "ATC",
    "airport surveillance": "ASR",
    "asr": "ASR",
    "iff": "IFF",
    "identification friend or foe": "IFF",
    "navigation": "navigation",
    "navigational": "navigation",
    "weather": "weather",
    "weather radar": "weather",
    "terrain following": "terrain_following",
    "terrain-following": "terrain_following",
    "terrain avoidance": "terrain_avoidance",
    "terrain-avoidance": "terrain_avoidance",
    "isar": "ISAR",
    "ground penetrating": "GPR",
    "gpr": "GPR",
    "ground-penetrating": "GPR",
    "over-the-horizon": "OTH",
    "over the horizon": "OTH",
    "oth": "OTH",
    "airspace surveillance": "airspace_surveillance",
    "sea search": "sea_search",
    "surface search": "surface_search",
    "naval search": "sea_search",
    "test range": "test_range",
    "torpedo guidance": "torpedo_guidance",
    "intercept": "air_intercept",
    "interception": "air_intercept",
    "ground controlled interception": "air_intercept",
    "gci": "air_intercept",
    "altimeter": "altimeter",
}


def normalize_function(s: str) -> str:
    """Map raw label to canonical id, or '' if no match."""
    if not s: return ""
    s_clean = s.strip()
    if s_clean in FUNCTION_CANON:
        return s_clean
    sl = s_clean.lower()
    if sl in FUNCTION_ALIASES:
        return FUNCTION_ALIASES[sl]
    # Substring match against canonical IDs
    for fid in FUNCTION_CANON:
        if fid.replace("_", " ") in sl or fid in sl:
            return fid
    # Substring match against aliases
    for alias, fid in FUNCTION_ALIASES.items():
        if alias in sl:
            return fid
    return ""


# ────────────────────────────────────────────────────────────
#  LLM
# ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(c: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)

_LLM_CACHE = _load_cache()


def call_llm(messages, max_tokens=300, temperature=0.0) -> str:
    import requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    payload = {"model": "deepseek-chat", "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            _LLM_CACHE[key] = txt
            if len(_LLM_CACHE) % 20 == 0: _save_cache(_LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


def _vocab_block() -> str:
    return "\n".join(f"  {fid:25s} - {zh}" for fid, zh in FUNCTION_CANON.items())


LLM_SYSTEM = f"""你是雷达装备分析师。给定雷达型号及其上下文，请从下面的 function 候选清单中挑出最贴合的 1-4 个（多选）：

{_vocab_block()}

**判断要点：**
- 一般机载火控雷达 → fire_control, air_search, air_track（视情况加 ground_attack / ground_mapping）
- 预警机/远程对空 → early_warning, airspace_surveillance, air_search
- 舰载搜索/对海 → sea_search, surface_search 或 air_search（高空搜索）
- 防空火控 → fire_control, air_defense, missile_guidance
- 气象 → weather  /  导航 → navigation  /  地形回避/跟随用对应专项
- 反炮兵雷达 → counter_battery, weapon_locating
- ATC / 机场 → ATC, ASR

只输出 JSON：
{{"functions": ["fire_control", "air_search"], "confidence": 0.0-1.0}}
**不要解释，不要列出清单外的值，不能确定就给 [] 。**
"""

LLM_USER_TPL = """雷达 ID: {rid}
英文名: {name_en}
中文名: {name_zh}
平台/操作国: {operator}
modes: {modes}
tech_types: {tech_types}
frequency_bands: {freq_bands}
narrative 摘要: {narrative}

请选出 functions。"""


def llm_infer_functions(radar: dict) -> tuple[list[str], float]:
    rid = radar.get("id", "")
    user = LLM_USER_TPL.format(
        rid=rid,
        name_en=radar.get("name_en", "") or "",
        name_zh=radar.get("name_zh", "") or "",
        operator=radar.get("operator_primary", "") or "(无)",
        modes=", ".join(radar.get("modes", []) or []) or "(无)",
        tech_types=", ".join(radar.get("tech_types", []) or []) or "(无)",
        freq_bands=", ".join(radar.get("frequency_bands", []) or []) or "(无)",
        narrative=(radar.get("narrative_summary") or "")[:500] or "(无)",
    )
    raw = call_llm(
        [{"role": "system", "content": LLM_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=200,
    )
    if raw.startswith("[LLM_ERROR"):
        return [], 0.0
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return [], 0.0
    raw_funcs = data.get("functions", []) or []
    if not isinstance(raw_funcs, list): return [], 0.0
    norm = []
    for f in raw_funcs:
        nf = normalize_function(str(f))
        if nf and nf not in norm:
            norm.append(nf)
    return norm[:4], float(data.get("confidence", 0.7))


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-llm", type=int, default=10000)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-missing", action="store_true",
                    help="Skip radars that already have ≥1 hasFunction edge")
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

    # First, normalize the existing 26 hasFunction edges to canonical IDs (in place)
    normalized_existing = 0
    for ed in edges:
        if ed["relation"] == "hasFunction":
            nf = normalize_function(ed["tail"])
            if nf and nf != ed["tail"]:
                ed["tail"] = nf
                normalized_existing += 1
    print(f"  Normalized {normalized_existing} existing hasFunction edges to canonical")

    # Radars that already have a hasFunction edge
    has_fn = defaultdict(set)
    for ed in edges:
        if ed["relation"] == "hasFunction":
            has_fn[ed["head"]].add(ed["tail"])

    new_edges = []
    new_func_entities = set()
    by_func = Counter()
    llm_calls = 0
    skipped_low_conf = 0
    skipped_existing = 0
    skipped_empty = 0

    for r in radars:
        rid = r["id"]
        if args.only_missing and rid in has_fn:
            skipped_existing += 1
            continue
        # Skip generic concept entries (no name_en or trivial id)
        if not r.get("name_en") and not r.get("name_zh") and not (r.get("narrative_summary") or "").strip():
            skipped_empty += 1
            continue
        if llm_calls >= args.max_llm:
            break

        funcs, conf = llm_infer_functions(r)
        llm_calls += 1
        if llm_calls % 25 == 0:
            _save_cache(_LLM_CACHE)
            try:
                print(f"  LLM progress: {llm_calls} calls (last: {rid})")
            except UnicodeEncodeError:
                print(f"  LLM progress: {llm_calls} calls (last: {rid.encode('ascii', 'replace').decode()})")

        if conf < args.min_conf or not funcs:
            skipped_low_conf += 1
            continue

        # Update entity attribute
        cur = list(r.get("functions") or [])
        for f in funcs:
            if f not in cur: cur.append(f)
        r["functions"] = cur

        # Add edges + Function entities
        for f in funcs:
            key = (rid, "hasFunction", f)
            if key in existing_edges: continue
            new_edges.append({
                "head": rid, "head_type": r["type"],
                "relation": "hasFunction",
                "tail": f, "tail_type": "Function",
                "confidence": round(conf, 2),
                "evidence": "enriched: function inferred from radar context",
                "source": "enrich_func_llm",
            })
            existing_edges.add(key)
            by_func[f] += 1
            if f not in ent_by_id:
                ent_by_id[f] = {
                    "id": f, "type": "Function",
                    "name_en": f.replace("_", " "),
                    "name_zh": FUNCTION_CANON.get(f, ""),
                    "aliases": [],
                }
                entities.append(ent_by_id[f])
                new_func_entities.add(f)

    _save_cache(_LLM_CACHE)

    # Save
    edges.extend(new_edges)
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)
    ent_data["entities"] = entities
    ent_data["count"] = len(entities)

    if not args.dry_run:
        with open(ENT_PATH, "w", encoding="utf-8") as f:
            json.dump(ent_data, f, ensure_ascii=False, indent=2)
        with open(EDGE_PATH, "w", encoding="utf-8") as f:
            json.dump(edge_data, f, ensure_ascii=False, indent=2)

    radars_after = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    n_with_fn = sum(1 for r in radars_after if r.get("functions"))

    report = f"""# Function Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage

| Metric | Before | After |
|---|---|---|
| hasFunction edges | 26 | {sum(1 for e in edges if e['relation']=='hasFunction')} |
| Radars with ≥1 function | {sum(1 for r in radars_after if r.get('functions') and len(r.get('functions'))>=1) - sum(1 for f in by_func)} | {n_with_fn} ({100*n_with_fn/len(radars_after):.1f}%) |
| Function entities | 17 | {sum(1 for e in entities if e['type']=='Function')} |

## New edges by function

{chr(10).join(f"- `{f}` ({FUNCTION_CANON.get(f,'')}): {c}" for f, c in by_func.most_common())}

## LLM stats

- LLM calls:                   {llm_calls}
- Low-confidence rejected:     {skipped_low_conf}
- Skipped (already had func):  {skipped_existing}
- Skipped (empty radar):       {skipped_empty}
- Cache size after run:        {len(_LLM_CACHE)}
"""

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {REPORT_PATH}")

    print(f"""
────── function enrichment summary ──────
hasFunction edges added:   +{len(new_edges)}
total hasFunction edges:   {sum(1 for e in edges if e['relation']=='hasFunction')}
radars with ≥1 function:   {n_with_fn}/{len(radars_after)} ({100*n_with_fn/len(radars_after):.1f}%)
new Function entities:     +{len(new_func_entities)}
LLM calls:                 {llm_calls}
""")


if __name__ == "__main__":
    main()
