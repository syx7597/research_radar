"""
Priority-3 enrichment: fill numeric specs (frequency_GHz, range_km, peak_power_kW, year)
for Radar / RadarSystem entities.

Currently:
  - frequency_GHz: 12.5%, range_km: 16.1%, peak_power_kW: 12.7%, year: 6.6%

Approach: use LLM with the radar's full context (name, country, developer, descriptions,
narrative) to extract structured numeric fields. Validate ranges before accepting.
Don't overwrite existing values — only fill blanks.

Writes:
  data/v2/radarkg_v2_entities.json  — fills numeric fields in place
  data/v2/enrich_numeric_report.md
Cache:
  data/v2/enrich_numeric_llm_cache.json
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
CACHE_PATH = ROOT / "data" / "v2" / "enrich_numeric_llm_cache.json"
REPORT_PATH = ROOT / "data" / "v2" / "enrich_numeric_report.md"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Sanity ranges per field (reject anything outside these)
VALID_RANGES = {
    "frequency_GHz":     (0.05, 300.0),     # HF radars from 50 MHz to W-band
    "frequency_GHz_min": (0.05, 300.0),
    "frequency_GHz_max": (0.05, 300.0),
    "range_km":          (0.1, 10000.0),    # short to OTH
    "peak_power_kW":     (0.001, 100000.0),  # mW class to MW class
    "year":              (1930, 2030),
    "weight_kg":         (0.5, 50000.0),
    "antenna_gain_dB":   (5.0, 70.0),
    "mtbf_hours":        (10, 100000),
}


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
_CACHE_LOCK = threading.Lock()


def call_llm(messages, max_tokens=400, temperature=0.0) -> str:
    import requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    with _CACHE_LOCK:
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
            with _CACHE_LOCK:
                _LLM_CACHE[key] = txt
                if len(_LLM_CACHE) % 20 == 0: _save_cache(_LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2: return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


LLM_SYSTEM = """你是雷达装备情报分析师，请从给定雷达条目中抽取/推断这些数值规格：
- frequency_GHz_min : 最低工作频率（GHz）
- frequency_GHz_max : 最高工作频率（GHz）
- frequency_GHz     : 中心频率（如果只有单点）
- range_km          : 探测距离 / 作用距离（km，取最大值）
- peak_power_kW     : 峰值功率（kW）
- year              : 服役/部署年份（YYYY）
- weight_kg         : 重量（kg）
- antenna_gain_dB   : 天线增益（dB）
- mtbf_hours        : 平均故障间隔时间（小时）

**规则：**
1. 只在描述/原文里能找到 OR 你对该型号有可靠领域知识时给出（如 AN/APG-77、AN/APG-63、Cobra Dane 等知名雷达可凭知识填）。
2. 模糊/无法确定 → 字段留空（不要写 0 或猜测中位数）。
3. 单位换算后输出（MHz→GHz / W→kW / m→km / "数千" 不要写）。
4. 频段标识（X 波段、L 波段）不能直接当 frequency_GHz —— 必须有具体数字才填。
5. confidence 反映该型号你的信心：0.9+ 是知名美俄系雷达，0.5-0.7 是描述里有明确数字但来源单一。

只输出 JSON：
{"frequency_GHz":...,"frequency_GHz_min":...,"frequency_GHz_max":...,"range_km":...,"peak_power_kW":...,"year":...,"weight_kg":...,"antenna_gain_dB":...,"mtbf_hours":...,"confidence":0.0-1.0}
未知字段写 null。
"""

LLM_USER_TPL = """雷达 ID: {rid}
英文名: {name_en}
中文名: {name_zh}
country: {country}
developer: {developer}
operator: {operator}
frequency_description: {fd}
range_description: {rd}
weight_description: {wd}
peak_power 已有值: {pp}
narrative 摘要: {narrative}

请抽取数值规格。"""


def _f(x):
    """Coerce to float, return None on failure."""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip().rstrip("%")
        if not s or s.lower() in ("null", "none", "n/a", "unknown"):
            return None
        try: return float(s)
        except ValueError: return None
    return None


def _validate(field: str, val):
    """Return clean value or None if outside valid range."""
    f = _f(val)
    if f is None: return None
    lo, hi = VALID_RANGES.get(field, (-1e18, 1e18))
    if not (lo <= f <= hi): return None
    if field == "year": return int(f)
    return f


def llm_infer(radar: dict) -> tuple[dict, float]:
    rid = radar.get("id", "")
    user = LLM_USER_TPL.format(
        rid=rid,
        name_en=radar.get("name_en", "") or "",
        name_zh=radar.get("name_zh", "") or "",
        country=radar.get("country_of_origin", "") or "(未知)",
        developer=radar.get("developer", "") or "(未知)",
        operator=radar.get("operator_primary", "") or "(无)",
        fd=radar.get("frequency_description", "") or "(无)",
        rd=radar.get("range_description", "") or "(无)",
        wd=radar.get("weight_description", "") or "(无)",
        pp=radar.get("peak_power_kW", "") or "(无)",
        narrative=(radar.get("narrative_summary") or "")[:500] or "(无)",
    )
    raw = call_llm(
        [{"role": "system", "content": LLM_SYSTEM},
         {"role": "user", "content": user}],
        max_tokens=300,
    )
    if raw.startswith("[LLM_ERROR"):
        return {}, 0.0
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return {}, 0.0
    conf = float(data.get("confidence", 0.6))
    out = {}
    for field in VALID_RANGES:
        v = _validate(field, data.get(field))
        if v is not None: out[field] = v
    return out, conf


# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-llm", type=int, default=10000)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-missing", action="store_true",
                    help="Only call LLM if at least one numeric is missing")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel LLM workers (default 8)")
    args = ap.parse_args()

    print("Loading v2 KG…")
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    entities = ent_data["entities"]
    radars = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    print(f"  {len(entities)} entities, {len(radars)} radars")

    # Track before-coverage
    target_fields = ["frequency_GHz", "frequency_GHz_min", "frequency_GHz_max",
                     "range_km", "peak_power_kW", "year", "weight_kg"]
    before = {f: sum(1 for r in radars if r.get(f)) for f in target_fields}

    filled = Counter()
    samples = defaultdict(list)
    rejected_low_conf = 0
    rejected_empty = 0

    # Build work queue
    work = []
    for r in radars:
        if all(r.get(f) for f in target_fields):
            continue
        if not r.get("name_en") and not r.get("name_zh") and not (r.get("narrative_summary") or ""):
            rejected_empty += 1
            continue
        work.append(r)
    work = work[:args.max_llm]
    print(f"  Work queue: {len(work)} radars to enrich")

    progress = {"n": 0}
    progress_lock = threading.Lock()

    def worker(r):
        result, conf = llm_infer(r)
        with progress_lock:
            progress["n"] += 1
            if progress["n"] % 50 == 0:
                _save_cache(_LLM_CACHE)
                try:
                    print(f"  LLM progress: {progress['n']}/{len(work)} (last: {r['id']})")
                except UnicodeEncodeError:
                    print(f"  LLM progress: {progress['n']}/{len(work)}")
        return r, result, conf

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r) for r in work]
        for fut in as_completed(futures):
            r, result, conf = fut.result()
            rid = r["id"]
            if conf < args.min_conf:
                rejected_low_conf += 1
                continue
            for field, val in result.items():
                if field not in target_fields and field not in ("antenna_gain_dB", "mtbf_hours"):
                    continue
                if r.get(field):
                    continue
                if field == "frequency_GHz":
                    fmin = r.get("frequency_GHz_min") or result.get("frequency_GHz_min")
                    fmax = r.get("frequency_GHz_max") or result.get("frequency_GHz_max")
                    if fmin and fmax and not (fmin <= val <= fmax):
                        continue
                r[field] = val
                filled[field] += 1
                if len(samples[field]) < 5:
                    samples[field].append((rid, val))

    llm_calls = progress["n"]
    _save_cache(_LLM_CACHE)

    # Save (only entities; we don't add edges since these are attributes)
    ent_data["entities"] = entities
    if not args.dry_run:
        with open(ENT_PATH, "w", encoding="utf-8") as f:
            json.dump(ent_data, f, ensure_ascii=False, indent=2)

    # Report
    radars_after = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    after = {f: sum(1 for r in radars_after if r.get(f)) for f in target_fields}

    report = f"""# Numeric Spec Enrichment Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Coverage by field

| Field | Before | Filled | After | Coverage |
|---|---|---|---|---|
"""
    for f in target_fields:
        report += f"| {f} | {before[f]} | +{filled[f]} | {after[f]} | {100*after[f]/len(radars_after):.1f}% |\n"

    report += f"""
## LLM stats

- Calls:                    {llm_calls}
- Low-confidence rejected:  {rejected_low_conf}
- Empty entities skipped:   {rejected_empty}
- Cache size:               {len(_LLM_CACHE)}

## Sample fills

"""
    for f, items in samples.items():
        report += f"### {f}\n\n"
        for rid, val in items:
            report += f"- `{rid}` → {val}\n"
        report += "\n"

    if not args.dry_run:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport: {REPORT_PATH}")

    print("\n────── numeric enrichment summary ──────")
    for f in target_fields:
        delta = after[f] - before[f]
        print(f"  {f:25s} {before[f]:>3d} → {after[f]:>3d}   (+{delta}, {100*after[f]/len(radars_after):.1f}%)")
    print(f"\nLLM calls:  {llm_calls}")


if __name__ == "__main__":
    main()
