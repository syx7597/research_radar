"""
Priority-1 enrichment: fill country_of_origin and developer for Radar / RadarSystem entities.

Three passes:
  A. Rule-based on radar ID/name (AN/* → US, EL/M-* → Israel, P-XX → Russia, ...)
  B. Manufacturer chain — if radar has `developer` attribute matching a Manufacturer
     whose `country` is set, propagate.
  C. LLM (DeepSeek) batch fallback — uses name + name_zh + aliases + narrative_summary
     + operator_primary as context. Cached by radar id.

Writes back to:
  data/v2/radarkg_v2_entities.json   — fills country_of_origin / developer attrs
  data/v2/radarkg_v2_edges.json      — adds countryOfOrigin / developedBy edges with source
                                       in {enrich_rule, enrich_chain, enrich_llm}.
Cache:
  data/v2/enrich_country_dev_llm_cache.json
Report:
  data/v2/enrich_country_dev_report.md
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH  = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
CACHE_PATH = ROOT / "data" / "v2" / "enrich_country_dev_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_country_dev_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


# ────────────────────────────────────────────────────────────
#  Country normalization (canonical = Chinese)
# ────────────────────────────────────────────────────────────

COUNTRY_CANON = {
    "美国": "美国", "USA": "美国", "U.S.": "美国", "US": "美国", "United States": "美国",
    "United States of America": "美国", "America": "美国",
    "苏联": "俄罗斯", "USSR": "俄罗斯", "Soviet Union": "俄罗斯", "Soviet": "俄罗斯",
    "俄罗斯": "俄罗斯", "Russia": "俄罗斯", "Russian Federation": "俄罗斯",
    "中国": "中国", "China": "中国", "PRC": "中国", "People's Republic of China": "中国",
    "英国": "英国", "UK": "英国", "United Kingdom": "英国", "Britain": "英国", "Great Britain": "英国",
    "法国": "法国", "France": "法国",
    "德国": "德国", "Germany": "德国", "西德": "德国", "West Germany": "德国",
    "意大利": "意大利", "Italy": "意大利",
    "以色列": "以色列", "Israel": "以色列",
    "日本": "日本", "Japan": "日本",
    "瑞典": "瑞典", "Sweden": "瑞典",
    "荷兰": "荷兰", "Netherlands": "荷兰",
    "印度": "印度", "India": "印度",
    "加拿大": "加拿大", "Canada": "加拿大",
    "澳大利亚": "澳大利亚", "Australia": "澳大利亚",
    "土耳其": "土耳其", "Turkey": "土耳其",
    "波兰": "波兰", "Poland": "波兰",
    "西班牙": "西班牙", "Spain": "西班牙",
    "南非": "南非", "South Africa": "南非",
    "巴西": "巴西", "Brazil": "巴西",
    "韩国": "韩国", "South Korea": "韩国", "Korea": "韩国",
    "朝鲜": "朝鲜", "North Korea": "朝鲜",
    "伊朗": "伊朗", "Iran": "伊朗",
    "乌克兰": "乌克兰", "Ukraine": "乌克兰",
    "芬兰": "芬兰", "Finland": "芬兰",
    "捷克": "捷克", "Czech Republic": "捷克", "Czechoslovakia": "捷克",
    "罗马尼亚": "罗马尼亚", "Romania": "罗马尼亚",
    "希腊": "希腊", "Greece": "希腊",
    "挪威": "挪威", "Norway": "挪威",
    "瑞士": "瑞士", "Switzerland": "瑞士",
    "比利时": "比利时", "Belgium": "比利时",
    "巴基斯坦": "巴基斯坦", "Pakistan": "巴基斯坦",
    "新加坡": "新加坡", "Singapore": "新加坡",
    "台湾": "台湾", "Taiwan": "台湾",
    "白俄罗斯": "白俄罗斯", "Belarus": "白俄罗斯",
    "保加利亚": "保加利亚", "Bulgaria": "保加利亚",
    "南斯拉夫": "南斯拉夫", "Yugoslavia": "南斯拉夫",
    "丹麦": "丹麦", "Denmark": "丹麦",
    "葡萄牙": "葡萄牙", "Portugal": "葡萄牙",
    "西德": "德国", "东德": "德国",
}

def canon_country(s: str) -> str:
    if not s: return ""
    s = s.strip()
    if s in COUNTRY_CANON: return COUNTRY_CANON[s]
    # try first word
    head = s.split()[0] if s.split() else s
    return COUNTRY_CANON.get(head, s)


# ────────────────────────────────────────────────────────────
#  Pass A: rule-based patterns on radar IDs/names
# ────────────────────────────────────────────────────────────

# Each rule = (regex_to_match_id_or_name, country, optional_developer_hint)
RULE_PATTERNS = [
    # United States — JETDS Joint Electronics Type Designation System (AN/...)
    (re.compile(r"^AN/[A-Z]{3}-?\d", re.I), "美国", None),
    # Older US Navy/Army designators without AN/ prefix
    (re.compile(r"^(APQ|APG|APN|APS|APY|AWG|TPQ|TPS|FPS|SPS|SPG|SPY|SPN|UPS|GPN|MPQ|MPN|MPS|UPD|VPS|TPN|FPN|GPS|GPA)-\d", re.I), "美国", None),
    # AN/SPQ AN/SPG AN/SLQ — naval
    (re.compile(r"^AN/(SLQ|SPS|SPG|SPQ|SPY|SPN|FPS)-?\d", re.I), "美国", None),
    # Westinghouse / Raytheon model numbers
    (re.compile(r"^WX-?\d", re.I), "美国", None),
    # Honeywell Primus — General Aviation
    (re.compile(r"^Primus[-\s]\d", re.I), "美国", "Honeywell"),
    # Russia / Soviet
    (re.compile(r"^(P|РП|RP)-\d{1,3}([A-Z]?)$", re.I), "俄罗斯", None),
    (re.compile(r"^(9[SPCXNT]|5N|5P|5V|5J|55Ж|64Ж|76N|RV)\d", re.I), "俄罗斯", None),
    (re.compile(r"^Zhuk", re.I), "俄罗斯", "Phazotron-NIIR"),
    (re.compile(r"^Bars", re.I), "俄罗斯", "Tikhomirov NIIP"),
    (re.compile(r"^Irbis", re.I), "俄罗斯", "Tikhomirov NIIP"),
    (re.compile(r"^Sapsan", re.I), "俄罗斯", None),
    (re.compile(r"^Kopyo", re.I), "俄罗斯", "Phazotron-NIIR"),
    (re.compile(r"^(Mech|Smerch|Tor|Kub|Buk|Pantsir|S-300|S-400|S-500|Sosna|Tunguska)", re.I), "俄罗斯", None),
    (re.compile(r"^(Don|Volga|Daryal|Voronezh|Dnepr)", re.I), "俄罗斯", None),
    # United Kingdom — AMES / Marconi / Plessey / Type-XXX in naval (3-digit naval pattern)
    (re.compile(r"^AMES\b", re.I), "英国", None),
    (re.compile(r"^Type\s?9\d{2}\b"), "英国", None),       # Type 9xx → UK naval
    (re.compile(r"^Type\s?10\d{2}\b"), "英国", None),
    (re.compile(r"^Type\s?9[78]\d\b"), "英国", None),
    (re.compile(r"^Marconi\b", re.I), "英国", "Marconi"),
    (re.compile(r"^(Searchwater|Blue\s+\w+|Foxhunter|Ferranti|Plessey|Cossor)", re.I), "英国", None),
    (re.compile(r"^Chain\s+Home", re.I), "英国", None),
    (re.compile(r"^ASV\s+Mark", re.I), "英国", None),
    # Israel — IAI Elta
    (re.compile(r"^EL/[ML]-?\d", re.I), "以色列", "IAI Elta"),
    (re.compile(r"^ELM[-\s]?\d", re.I), "以色列", "IAI Elta"),
    (re.compile(r"^Elta\b", re.I), "以色列", "IAI Elta"),
    # France — Thales / Thomson / Dassault / DRBV / DRBR
    (re.compile(r"^(DRBV|DRBR|DRBI|DRBJ|DRBN|DRBC|RDM|RDI|RDY|RDX|RBE2|Cyrano)", re.I), "法国", None),
    (re.compile(r"^(Thomson|Thales)\b", re.I), "法国", None),
    (re.compile(r"^Iguane", re.I), "法国", None),
    (re.compile(r"^Anemone", re.I), "法国", None),
    (re.compile(r"^Agrion", re.I), "法国", None),
    # Germany / EU
    (re.compile(r"^(BAA|TRS|MPDR)\d", re.I), "德国", None),
    # Italy — Selex / Selenia / SMA / Galileo / Alenia
    (re.compile(r"^(Selex|Selenia|Galileo|Alenia|SMA|EMPAR)", re.I), "意大利", None),
    # Sweden — Ericsson / Saab — Erieye / PS-XX / Giraffe
    (re.compile(r"^(Erieye|Giraffe|UAR|UAP|PS-?[\d])", re.I), "瑞典", None),
    (re.compile(r"^Ericsson", re.I), "瑞典", "Ericsson"),
    (re.compile(r"^Saab", re.I), "瑞典", "Saab"),
    # Netherlands — Thales NL legacy as Signaal / DA-08 / SMART-L
    (re.compile(r"^(Signaal|SMART-L|SMART-S|APAR|DA-?\d|LW-?\d)", re.I), "荷兰", None),
    # Japan — JMSDF/JASDF OPS-XX, FCS-XX, J/APG-X, J/FPS-X
    (re.compile(r"^(OPS|FCS|J/APG|J/FPS|J/TPS|J/MPQ)-?\d", re.I), "日本", None),
    (re.compile(r"^Mitsubishi", re.I), "日本", "Mitsubishi"),
    # India — BEL / DRDO / Rajendra
    (re.compile(r"^(BEL|DRDO|Rajendra|Indra|Arudhra|Revathi|Aslesha|Rohini|Swordfish)", re.I), "印度", None),
    # China — Type-XXX (when followed by Chinese radar prefixes), JL-, JY-, LLQ-, JGS-, KJ-, YLC-, SLC-
    (re.compile(r"^(JL|JY|LLQ|JGS|KJ|YLC|SLC|JLG|JL-?\d|HQ|HT-?[\d]|JH-?\d)", re.I), "中国", None),
    (re.compile(r"^(KLJ|KLC|KLU|KLD|KLT)-?\d", re.I), "中国", None),
    # Iran
    (re.compile(r"^(Sepehr|Najm|Asr|Matla|Ghadir)", re.I), "伊朗", None),
    # Turkey
    (re.compile(r"^(Aselsan|MILDAR|MILGEM|KALKAN|HISAR)", re.I), "土耳其", None),
    # Brazil
    (re.compile(r"^(SABER|SCANTER)", re.I), "巴西", None),
]


def rule_infer(rid: str, name_en: str = "") -> tuple[str, str | None]:
    """Try to infer (country, developer_hint) from radar id/name. Returns ('','') if no rule matched."""
    for src in (rid, name_en):
        if not src: continue
        for pat, country, dev_hint in RULE_PATTERNS:
            if pat.search(src):
                return country, dev_hint
    return "", None


# ────────────────────────────────────────────────────────────
#  LLM with cache
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


def call_llm(messages, max_tokens=400, temperature=0.0) -> str:
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
            if len(_LLM_CACHE) % 10 == 0: _save_cache(_LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


LLM_SYSTEM = """你是雷达装备情报分析师。给定一个雷达型号及其上下文，请回答它的：
1. country_of_origin — 研发国（不是使用国）。用中文国名（美国 / 俄罗斯 / 中国 / 英国 / 法国 / 德国 / 意大利 / 以色列 / 瑞典 / 荷兰 / 日本 / 印度 / 加拿大 / 澳大利亚 / 土耳其 / 波兰 / 伊朗 / 西班牙 / 巴西 / 韩国 / 苏联 等。苏联归一化为"俄罗斯"）。
2. developer — 研发厂商英文名称（如 Northrop Grumman / Raytheon / Lockheed Martin / Phazotron-NIIR / IAI Elta / Thales / Marconi / Selex / Ericsson / NIIP / 中国电子科技集团 / 14 Institute 等）。

判断依据：型号命名规则（AN/* 是美国，EL/M-* 是以色列 IAI Elta，Type-9xx 是英国海军，DRBV/Cyrano 是法国，9S/9C/P-数字 是苏联/俄罗斯，OPS-xx 是日本，HQ-/JY-/YLC- 是中国，等等）+ 你的领域知识。

只在你有合理把握时给出，把握度低的留空。

严格输出 JSON：
{"country":"...","developer":"...","confidence":0.0-1.0}
不能确定就 country/developer 留空字符串，不要编造。
"""

LLM_USER_TPL = """雷达 ID: {rid}
英文名: {name_en}
中文名: {name_zh}
别名: {aliases}
现有 operator (使用国): {operator}
narrative 摘要: {narrative}

请返回 JSON。"""


def llm_infer(radar: dict) -> dict:
    """Returns {country: str, developer: str, confidence: float}"""
    rid = radar.get("id", "")
    user = LLM_USER_TPL.format(
        rid=rid,
        name_en=radar.get("name_en", "") or "",
        name_zh=radar.get("name_zh", "") or "",
        aliases=", ".join(radar.get("aliases", []) or []) or "(无)",
        operator=radar.get("operator_primary", "") or "(无)",
        narrative=(radar.get("narrative_summary") or "")[:400] or "(无)",
    )
    raw = call_llm(
        [{"role": "system", "content": LLM_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=200,
    )
    if raw.startswith("[LLM_ERROR"):
        return {"country": "", "developer": "", "confidence": 0.0}
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return {"country": "", "developer": "", "confidence": 0.0}
    return {
        "country": canon_country(str(data.get("country", "")).strip()),
        "developer": str(data.get("developer", "")).strip(),
        "confidence": float(data.get("confidence", 0.5)),
    }


# ────────────────────────────────────────────────────────────
#  Main pipeline
# ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-llm", type=int, default=10000,
                    help="Cap LLM calls (default unlimited)")
    ap.add_argument("--dry-run", action="store_true", help="Don't write files")
    ap.add_argument("--skip-llm", action="store_true", help="Only do passes A+B")
    args = ap.parse_args()

    print(f"Loading v2 KG…")
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]
    print(f"  {len(entities)} entities, {len(edges)} edges")

    ent_by_id = {e["id"]: e for e in entities}
    radars = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]

    # Build manufacturer→country map (from `country` attr OR existing edges)
    mfr_country: dict[str, str] = {}
    for e in entities:
        if e["type"] in ("Manufacturer", "Organization"):
            c = canon_country(e.get("country", "") or e.get("country_of_origin", ""))
            if c:
                mfr_country[e["id"]] = c
                # also map by name_en for fuzzy lookups
                if e.get("name_en"):
                    mfr_country.setdefault(e["name_en"], c)
                if e.get("canonical_name"):
                    mfr_country.setdefault(e["canonical_name"], c)
    for ed in edges:
        if ed["relation"] in ("headquarteredIn", "countryOfOrigin"):
            head = ed["head"]
            if head not in mfr_country:
                mfr_country[head] = canon_country(ed["tail"])
    print(f"  Manufacturer→country map: {len(mfr_country)} entries")

    # Build radar → developer map (from attribute OR developedBy edge)
    radar_dev: dict[str, str] = {}
    for r in radars:
        if r.get("developer"):
            radar_dev[r["id"]] = r["developer"]
    for ed in edges:
        if ed["relation"] == "developedBy" and ed["head"] not in radar_dev:
            radar_dev[ed["head"]] = ed["tail"]

    # Existing edges set (for dedupe)
    existing_edges = {(e["head"], e["relation"], e["tail"]) for e in edges}

    # Counters
    filled = {"country": 0, "developer": 0}
    by_method = Counter()
    new_edges: list[dict] = []
    llm_calls = 0
    llm_no_country = 0
    samples_per_method = defaultdict(list)

    for r in radars:
        rid = r["id"]
        already_country = bool(r.get("country_of_origin"))
        already_dev = bool(r.get("developer"))

        if already_country and already_dev:
            continue

        country = ""
        developer = ""
        method = ""

        # ── Pass A: rule
        if not already_country or not already_dev:
            c, dev_hint = rule_infer(rid, r.get("name_en", "") or "")
            if c:
                if not already_country:
                    country = c; method = "rule"
                if not already_dev and dev_hint:
                    developer = dev_hint
                    method = method or "rule"

        # ── Pass B: manufacturer chain
        if not country and not already_country:
            dev_id = radar_dev.get(rid)
            if dev_id and dev_id in mfr_country:
                country = mfr_country[dev_id]
                method = "chain"

        # ── Pass C: LLM
        if (not already_country and not country) or (not already_dev and not developer):
            if not args.skip_llm and llm_calls < args.max_llm:
                # Only LLM-call when we still need something
                res = llm_infer(r)
                llm_calls += 1
                if llm_calls % 25 == 0:
                    _save_cache(_LLM_CACHE)
                    print(f"  LLM progress: {llm_calls} calls (last: {rid})")
                if res["confidence"] >= 0.6:
                    if not country and not already_country and res["country"]:
                        country = res["country"]
                        method = method or "llm"
                    if not developer and not already_dev and res["developer"]:
                        developer = res["developer"]
                        method = method or "llm"
                else:
                    llm_no_country += 1

        # Apply
        if country and not already_country:
            r["country_of_origin"] = country
            filled["country"] += 1
            key = (rid, "countryOfOrigin", country)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "countryOfOrigin",
                    "tail": country, "tail_type": "Country",
                    "confidence": 0.9 if method == "rule" else (0.85 if method == "chain" else 0.75),
                    "evidence": f"enriched via {method}",
                    "source": f"enrich_{method}",
                })
                existing_edges.add(key)
                by_method[method] += 1
            if len(samples_per_method[f"country_{method}"]) < 5:
                samples_per_method[f"country_{method}"].append((rid, country))

            # Auto-create Country entity if missing
            if country not in ent_by_id:
                ent_by_id[country] = {
                    "id": country, "type": "Country",
                    "name_zh": country, "name_en": country,
                    "aliases": [],
                }
                entities.append(ent_by_id[country])

        if developer and not already_dev:
            r["developer"] = developer
            filled["developer"] += 1
            key = (rid, "developedBy", developer)
            if key not in existing_edges:
                new_edges.append({
                    "head": rid, "head_type": r["type"],
                    "relation": "developedBy",
                    "tail": developer, "tail_type": "Manufacturer",
                    "confidence": 0.9 if method == "rule" else (0.85 if method == "chain" else 0.7),
                    "evidence": f"enriched via {method}",
                    "source": f"enrich_{method}",
                })
                existing_edges.add(key)
            if len(samples_per_method[f"dev_{method}"]) < 5:
                samples_per_method[f"dev_{method}"].append((rid, developer))

            # Auto-create Manufacturer if missing
            if developer not in ent_by_id:
                ent_by_id[developer] = {
                    "id": developer, "type": "Manufacturer",
                    "name_en": developer, "aliases": [],
                }
                entities.append(ent_by_id[developer])

    _save_cache(_LLM_CACHE)

    # Append edges
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

    # Report
    radars_after = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    n_with_country = sum(1 for r in radars_after if r.get("country_of_origin"))
    n_with_dev = sum(1 for r in radars_after if r.get("developer"))
    n_total = len(radars_after)

    report = f"""# Country/Developer Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage

| Field | Before | Filled | After | Coverage |
|---|---|---|---|---|
| country_of_origin | {n_with_country - filled['country']} | +{filled['country']} | {n_with_country} | {100*n_with_country/n_total:.1f}% |
| developer         | {n_with_dev - filled['developer']} | +{filled['developer']} | {n_with_dev} | {100*n_with_dev/n_total:.1f}% |

## New edges by method

{chr(10).join(f"- {m}: {c}" for m, c in by_method.most_common())}

## LLM stats

- LLM calls made:                {llm_calls}
- LLM low-confidence (rejected): {llm_no_country}
- Cache size after run:          {len(_LLM_CACHE)}

## Sample fills

"""
    for k, samples in samples_per_method.items():
        report += f"### {k}\n\n"
        for rid, val in samples:
            report += f"- `{rid}` → {val}\n"
        report += "\n"

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {REPORT_PATH}")

    print(f"""
────── enrichment summary ──────
country_of_origin coverage: {100*n_with_country/n_total:.1f}%   (+{filled['country']})
developer        coverage: {100*n_with_dev/n_total:.1f}%   (+{filled['developer']})
edges added:               {len(new_edges)}
LLM calls:                 {llm_calls}
""")


if __name__ == "__main__":
    main()
