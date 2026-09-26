# -*- coding: utf-8 -*-
"""Enrich threat profiles beyond the keyword tier (T1) with:

  T2  LLM extraction over rich corpus text — only for radars MISSING a field and
      having enough text; the LLM MUST cite a verbatim quote and may answer
      "unknown"; quotes are verified to be substrings of the source text, so
      unsupported (hallucinated) values are dropped. Results are cached.
  T3  Doctrine priors — fill still-missing fields from strong correlates
      (AESA/PESA → phased; fire-control/SAM-guidance → monopulse). Marked as
      "推测/prior" at low confidence so explicit evidence always wins.

Merge is fill-missing-only in order T1 > T2 > T3, so there is never a conflict;
each field records which tier produced it (auditable). The advisor consumes the
enriched profiles unchanged.

Cost control: high-value radars (fire_control/SAM_guidance/track/acquisition)
first, optional --budget cap, and a persistent cache keyed by text hash.
"""
import json, re, sys, hashlib, argparse
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from qa_strategy_pipeline import llm_call
from threat_profile import build_profiles, OUT, TRIPLES

CORPUS = ROOT / "radar_corpus" / "corpus.json"
CACHE = ROOT / "cache" / "profile_llm_cache.json"
WEB_CACHE = ROOT / "cache" / "web_text_cache.json"

LLM_FIELDS = ["tracking_method", "scan_type", "freq_agile", "prf_agile", "lpi"]
ENUMS = {
    "tracking_method": ["monopulse", "conical_scan", "lobe_switching",
                        "sequential_lobing", "TWS", "range_gate", "cw_illuminator", "none"],
    "scan_type": ["circular", "sector", "raster", "phased", "track"],
}
HIGH_VALUE = {"fire_control", "SAM_guidance", "track", "acquisition"}
CONF_LLM = 0.75
TEXT_MIN = 200
TEXT_CAP = 2600


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _normspace(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def build_corpus_index():
    c = json.load(open(CORPUS, encoding="utf-8"))
    idx = {}
    for e in c:
        text = ((e.get("raw_text_en") or "") + "\n" + (e.get("raw_text_zh") or "")).strip()
        for key in (e.get("en_title"), e.get("id"), e.get("zh_title")):
            if key:
                idx.setdefault(_norm(key), text)
    return idx


def kg_snippets(triples):
    """Per-head concatenation of the descriptive tails already in the KG."""
    pool = defaultdict(list)
    keep = ("hasMode", "eccm_description", "hasTechType", "scan_method_description",
            "azimuth_description", "signal_processing_description", "transmitter_description")
    for x in triples:
        if x["relation"] in keep:
            pool[x["head"]].append(str(x["tail"]))
    return {h: " | ".join(v) for h, v in pool.items()}


# ----------------------- T2: LLM extraction -----------------------
EXTRACT_SYS = """You extract radar engineering parameters STRICTLY from the given text.
For each field, return the value AND a verbatim quote (substring) from the text that supports it.
If the text does NOT clearly state a field, return value "unknown" with quote "".
NEVER guess from world knowledge — only what the text supports.

Fields and allowed values:
- tracking_method: one of [monopulse, conical_scan, lobe_switching, sequential_lobing, TWS, range_gate, cw_illuminator, none, unknown]
  (TWS = track-while-scan; cw_illuminator = continuous-wave illuminator)
- scan_type: one of [circular, sector, raster, phased, unknown]  (phased = electronically scanned / AESA / PESA)
- freq_agile: true / false / unknown   (frequency agility / frequency hopping)
- prf_agile: true / false / unknown    (PRF stagger / jitter)
- lpi: true / false / unknown          (low probability of intercept)

Output ONLY JSON:
{"tracking_method":{"value":"...","quote":"..."},"scan_type":{"value":"...","quote":"..."},
"freq_agile":{"value":"...","quote":"..."},"prf_agile":{"value":"...","quote":"..."},"lpi":{"value":"...","quote":"..."}}"""


def llm_extract(head, text, want):
    """Return {field: {value, quote, confidence}} for fields in `want`, verified."""
    text = text[:TEXT_CAP]
    user = f"Radar: {head}\nText:\n\"\"\"\n{text}\n\"\"\"\n\nExtract the JSON now:"
    raw = llm_call([{"role": "system", "content": EXTRACT_SYS},
                    {"role": "user", "content": user}], max_tokens=500)
    m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    ntext = _normspace(text)
    out = {}
    for f in want:
        rec = obj.get(f) or {}
        val, quote = rec.get("value"), rec.get("quote", "")
        if val in (None, "unknown", "", "false") and f in ("tracking_method", "scan_type"):
            continue
        if f in ("tracking_method", "scan_type") and val not in ENUMS[f]:
            continue
        if f in ("freq_agile", "prf_agile", "lpi"):
            if str(val).lower() != "true":
                continue
            val = True
        # verify the quote is actually in the source text (anti-hallucination)
        if not quote or _normspace(quote)[:60] not in ntext:
            continue
        out[f] = {"value": val, "quote": quote[:160], "confidence": CONF_LLM}
    return out


# ----------------------- T3: doctrine priors -----------------------
def doctrine_priors(prof, tech_text, want):
    """Fill still-missing fields from strong correlates. Returns {field:{value,reason,confidence}}."""
    out = {}
    purposes = set(prof.get("purpose") or [])
    is_phased = bool(re.search(r"相控阵|AESA|PESA|电扫|电子扫描|phased", tech_text, re.I))
    if "scan_type" in want and is_phased:
        out["scan_type"] = {"value": "phased", "confidence": 0.8,
                            "reason": "技术体制含相控阵/AESA/PESA → 电扫描"}
    if "tracking_method" in want:
        eff_phased = is_phased or prof.get("scan_type") == "phased" or out.get("scan_type", {}).get("value") == "phased"
        if purposes & {"fire_control", "SAM_guidance"}:
            out["tracking_method"] = {"value": "monopulse", "confidence": 0.55,
                                      "reason": "火控/制导雷达（1965年后）普遍采用单脉冲测角"}
        elif eff_phased and (purposes & {"track", "acquisition"}):
            out["tracking_method"] = {"value": "monopulse", "confidence": 0.5,
                                      "reason": "相控阵跟踪/截获雷达通常用单脉冲"}
    return out


# ----------------------- orchestrate -----------------------
def _load_cache(path=CACHE):
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}


def _save_cache(c, path=CACHE):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def _web_variants(head):
    """Clean a noisy KG head (e.g. 'AN/APG-67/67(V)') into wiki-searchable queries."""
    cands = []
    base = head.split("(")[0].strip()
    cands.append(base)
    m = re.match(r"([A-Za-z]{1,4}/[A-Za-z]{2,5}-\d+)", head)   # AN/APG-67
    if m:
        cands.append(m.group(1))
    cands.append(re.sub(r"(/\d+)+(\(.*\))?$", "", head).strip())  # strip /67(V)
    seen, out = set(), []
    for c in cands:
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out += [c, f"{c} radar"]
    return out


def fetch_web_text(head, web_cache):
    """Fetch a Wikipedia intro for a radar model (cached). Returns text or ''."""
    if head in web_cache and web_cache[head]:
        return web_cache[head]
    from web_tool import wiki_search
    text = ""
    for q in _web_variants(head):
        title, extract, _url = wiki_search(q, "en")
        if title and extract and len(extract) > TEXT_MIN:
            text = extract
            break
    web_cache[head] = text
    return text


def enrich(budget=None, use_llm=True, use_web=False, web_budget=60):
    triples = json.load(open(TRIPLES, encoding="utf-8"))
    profs = build_profiles(triples)
    corpus = build_corpus_index()
    snips = kg_snippets(triples)
    web_cache = _load_cache(WEB_CACHE)
    web_calls = 0
    # tech text per head for priors
    techpool = defaultdict(list)
    for x in triples:
        if x["relation"] == "hasTechType":
            techpool[x["head"]].append(str(x["tail"]))

    cache = _load_cache()
    before = Counter()
    for p in profs.values():
        for f in LLM_FIELDS:
            if f in p:
                before[f] += 1

    # order: high-value first
    def hv(item):
        return 0 if (set(item[1].get("purpose") or []) & HIGH_VALUE) else 1
    ordered = sorted(profs.items(), key=hv)

    llm_calls = 0
    t2_fills, t3_fills = Counter(), Counter()
    for head, prof in ordered:
        ev = prof.setdefault("_evidence", {})
        # tag existing (T1) fields
        for f in LLM_FIELDS:
            if f in prof and f in ev and "tier" not in ev[f]:
                ev[f]["tier"] = "keyword"
        want = [f for f in LLM_FIELDS if f not in prof]
        if not want:
            continue

        # ---- T2: LLM over text (corpus first, web fallback for high-value) ----
        text = corpus.get(_norm(head), "")
        t2_src = "corpus_llm"
        if (not text or len(text) < TEXT_MIN):
            text = ""
            is_hv = bool(set(prof.get("purpose") or []) & HIGH_VALUE)
            if use_web and is_hv and web_calls < web_budget:
                wt = fetch_web_text(head, web_cache)
                web_calls += 1
                if web_calls % 10 == 0:
                    _save_cache(web_cache, WEB_CACHE)
                if wt:
                    text = wt
                    t2_src = "web_llm"
        full = (text + "\n" + snips.get(head, "")).strip()
        if use_llm and text and (budget is None or llm_calls < budget):
            key = head + ":" + hashlib.md5(full.encode("utf-8")).hexdigest()[:8]
            if key in cache:
                res = cache[key]
            else:
                res = llm_extract(head, full, want)
                cache[key] = res
                llm_calls += 1
                if llm_calls % 20 == 0:
                    _save_cache(cache)
            for f, r in res.items():
                if f in prof:
                    continue
                prof[f] = r["value"]
                ev[f] = {"value": r["value"], "tier": "llm", "source": t2_src,
                         "confidence": r["confidence"], "quote": r.get("quote", "")}
                t2_fills[f] += 1

        # ---- T3: doctrine priors for whatever is still missing ----
        want2 = [f for f in ("tracking_method", "scan_type") if f not in prof]
        if want2:
            pr = doctrine_priors(prof, " ".join(techpool.get(head, [])), want2)
            for f, r in pr.items():
                if f in prof:
                    continue
                prof[f] = r["value"]
                ev[f] = {"value": r["value"], "tier": "prior", "source": "doctrine_prior",
                         "confidence": r["confidence"], "reason": r.get("reason", "")}
                t3_fills[f] += 1

    _save_cache(cache)
    _save_cache(web_cache, WEB_CACHE)
    json.dump(profs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    after = Counter()
    for p in profs.values():
        for f in LLM_FIELDS:
            if f in p:
                after[f] += 1
    return {"n": len(profs), "llm_calls": llm_calls, "web_calls": web_calls,
            "before": before, "after": after, "t2": t2_fills, "t3": t3_fills}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None, help="max LLM calls (cost cap)")
    ap.add_argument("--no-llm", action="store_true", help="priors only (no LLM)")
    ap.add_argument("--web", action="store_true", help="web fallback for corpus-less high-value radars")
    ap.add_argument("--web-budget", type=int, default=60, help="max web fetches")
    args = ap.parse_args()
    r = enrich(budget=args.budget, use_llm=not args.no_llm,
               use_web=args.web, web_budget=args.web_budget)
    print(f"profiles: {r['n']}　LLM calls: {r['llm_calls']}　web fetches: {r['web_calls']}")
    print(f"{'field':18s}{'before':>8s}{'after':>8s}{'  (T2 llm / T3 prior)'}")
    for f in LLM_FIELDS:
        print(f"{f:18s}{r['before'][f]:>8d}{r['after'][f]:>8d}    ({r['t2'][f]} / {r['t3'][f]})")
