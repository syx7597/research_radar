# -*- coding: utf-8 -*-
"""Derive a radar's *threat profile* (purpose / tracking method / scan / agility /
employed ECCM) from the existing KG — the bridge between the descriptive KG and
the EW countermeasure advisor.

Strategy: deterministic mapping from already-structured fields where possible
(`hasFunction` → purpose, `hasFrequencyBand` → bands at high confidence), plus
keyword inference over descriptive text (`hasMode` / `hasTechType` /
`eccm_description`) for tracking method, scan type, agility and ECCM (medium
confidence). Every derived field carries its evidence + confidence + source
field, so the profile is auditable just like the rest of the system.

No LLM in the trust path: this is rule-based extraction over real KG values
(real data, with provenance), NOT inference invented to pad the graph.
"""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
TRIPLES = ROOT / "graphrag_index" / "merged_triples.json"
OUT = ROOT / "data" / "ew" / "threat_profiles.json"

# ---- hasFunction (English enum) -> purpose ----
PURPOSE_MAP = {
    "air_search": "search", "sea_search": "search", "surface_search": "search",
    "early_warning": "search", "airspace_surveillance": "search",
    "battlefield_surveillance": "search", "reconnaissance": "search",
    "OTH": "search", "ASR": "search", "ATC": "search",
    "air_defense": "acquisition", "target_designation": "acquisition",
    "weapon_locating": "acquisition", "counter_battery": "acquisition",
    "air_track": "track", "target_tracking": "track", "air_intercept": "track",
    "target_identification": "track",
    "fire_control": "fire_control", "naval_gunnery": "fire_control",
    "missile_guidance": "SAM_guidance", "missile_defense": "SAM_guidance",
    "navigation": "navigation", "terrain_avoidance": "navigation",
    "terrain_following": "navigation", "altimeter": "navigation",
    "weather": "navigation", "ground_mapping": "navigation",
    "imaging_SAR": "navigation", "ISAR": "navigation", "ground_attack": "navigation",
    "test_range": "navigation", "IFF": "navigation",
}

# ---- keyword -> tracking_method (searched over mode/tech/eccm text) ----
TRACKING_KW = [
    ("monopulse",        ["单脉冲"]),
    ("conical_scan",     ["圆锥扫描", "锥扫", "圆锥扫"]),
    ("lobe_switching",   ["波瓣切换", "波瓣转换", "瓣间切换"]),
    ("sequential_lobing",["顺序波瓣", "序列波瓣"]),
    ("TWS",              ["边扫描边跟踪", "边扫边跟", "TWS", "边搜索边跟踪"]),
    ("cw_illuminator",   ["连续波照射", "连续波", "CW照射"]),
    ("range_gate",       ["距离波门", "距离门跟踪"]),
]
# ---- keyword -> scan_type ----
PHASED_KW = ["相控阵", "有源相控阵", "无源相控阵", "电扫", "电子扫描", "AESA", "PESA"]
SCAN_KW = [
    ("circular", ["圆周扫描", "环扫", "360"]),
    ("sector",   ["扇扫", "扇形扫描"]),
    ("raster",   ["光栅扫描", "光栅"]),
]
# ---- keyword -> ECCM technique id (must match doctrine_map ids) ----
ECCM_KW = [
    ("freq_agility",       ["频率捷变", "捷变频", "跳频", "频率分集", "频率敏捷"]),
    ("pulse_compression",  ["脉冲压缩", "脉压"]),
    ("sidelobe_canceller", ["旁瓣对消", "副瓣对消"]),
    ("sidelobe_blanking",  ["旁瓣匿影", "副瓣匿影", "旁瓣消隐"]),
    ("monopulse",          ["单脉冲"]),
    ("mti_pulse_doppler",  ["动目标显示", "脉冲多普勒", "MTI", "PD体制"]),
    ("cfar",               ["恒虚警", "CFAR"]),
    ("prf_jitter",         ["重频参差", "PRF参差", "重频抖动", "脉冲重复频率参差", "参差"]),
]
FREQ_AGILE_KW = ["频率捷变", "捷变频", "跳频", "频率分集", "频率敏捷"]
PRF_AGILE_KW = ["重频参差", "PRF参差", "重频抖动", "参差", "重频跳变"]
LPI_KW = ["低截获", "低截获概率", "LPI", "低被截获"]

CONF_STRUCT = 0.9    # from structured enum field
CONF_KW = 0.6        # from descriptive-text keyword inference

# numeric attributes pulled from the KG (per-radar, used for engagement guidance)
NUMERIC_REL = {
    "frequency_GHz": ["frequency_GHz", "frequency_GHz_min"],
    "peak_power_kW": ["peak_power_kW", "peak_power_kW_min"],
    "range_km": ["range_km"],
    "antenna_gain_dB": ["antenna_gain_dB"],
}


def _firstnum(vals):
    """First parseable float from a list of tail values, else None."""
    for v in vals:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                continue
    return None


def _first_kw(text, kw_table):
    """Return (value, matched_keyword) for the first matching entry, else (None, None)."""
    for value, kws in kw_table:
        for kw in kws:
            if kw in text:
                return value, kw
    return None, None


def _all_kw(text, kw_table):
    """Return list of (value, matched_keyword) for all matching entries (dedup value)."""
    out, seen = [], set()
    for value, kws in kw_table:
        for kw in kws:
            if kw in text and value not in seen:
                out.append((value, kw)); seen.add(value); break
    return out


def _any_kw(text, kws):
    for kw in kws:
        if kw in text:
            return kw
    return None


def build_profiles(triples=None):
    if triples is None:
        triples = json.load(open(TRIPLES, encoding="utf-8"))
    byh = defaultdict(lambda: defaultdict(list))
    for x in triples:
        byh[x["head"]][x["relation"]].append(x["tail"])

    profiles = {}
    for head, rels in byh.items():
        prof, ev = {}, {}

        # purpose <- hasFunction (structured, high conf)
        purposes, pmap = [], {}
        for fn in rels.get("hasFunction", []):
            p = PURPOSE_MAP.get(fn)
            if p and p not in purposes:
                purposes.append(p); pmap.setdefault(p, fn)
        if purposes:
            prof["purpose"] = purposes
            ev["purpose"] = {"value": purposes, "confidence": CONF_STRUCT,
                             "source_field": "hasFunction", "evidence": pmap}

        # bands <- hasFrequencyBand (structured)
        bands = [b for b in rels.get("hasFrequencyBand", []) if b]
        if bands:
            prof["bands"] = sorted(set(bands))
            ev["bands"] = {"value": prof["bands"], "confidence": CONF_STRUCT,
                           "source_field": "hasFrequencyBand"}

        # numeric attributes (structured) — these differ per radar and drive
        # parameter-specific engagement guidance (band class / burnthrough / stand-off)
        for field, srcs in NUMERIC_REL.items():
            val = None
            for r in srcs:
                val = _firstnum(rels.get(r, []))
                if val is not None:
                    break
            if val is not None:
                prof[field] = val
                ev[field] = {"value": val, "confidence": CONF_STRUCT, "source_field": srcs[0]}

        # descriptive text pool for keyword inference
        pool_parts = []
        for r in ("hasMode", "hasTechType", "eccm_description", "scan_method_description",
                  "azimuth_description", "signal_processing_description", "transmitter_description"):
            pool_parts += [str(v) for v in rels.get(r, [])]
        text = " | ".join(pool_parts)

        # tracking_method <- keyword
        tm, kw = _first_kw(text, TRACKING_KW)
        if tm:
            prof["tracking_method"] = tm
            ev["tracking_method"] = {"value": tm, "confidence": CONF_KW,
                                     "source_field": "mode/tech text", "evidence": kw}

        # scan_type <- phased (from techtype) else keyword
        tech_text = " | ".join(str(v) for v in rels.get("hasTechType", []))
        ph = _any_kw(tech_text + " " + text, PHASED_KW)
        if ph:
            prof["scan_type"] = "phased"
            ev["scan_type"] = {"value": "phased", "confidence": CONF_KW,
                               "source_field": "hasTechType", "evidence": ph}
        else:
            st, kw = _first_kw(text, SCAN_KW)
            if st:
                prof["scan_type"] = st
                ev["scan_type"] = {"value": st, "confidence": CONF_KW,
                                   "source_field": "mode text", "evidence": kw}

        # agility / lpi flags
        fa = _any_kw(text, FREQ_AGILE_KW)
        if fa:
            prof["freq_agile"] = True
            ev["freq_agile"] = {"value": True, "confidence": CONF_KW,
                                "source_field": "eccm/mode text", "evidence": fa}
        pa = _any_kw(text, PRF_AGILE_KW)
        if pa:
            prof["prf_agile"] = True
            ev["prf_agile"] = {"value": True, "confidence": CONF_KW, "evidence": pa}
        lp = _any_kw(text, LPI_KW)
        if lp:
            prof["lpi"] = True
            ev["lpi"] = {"value": True, "confidence": CONF_KW, "evidence": lp}

        # employs_eccm <- keyword (multi)
        eccm_hits = _all_kw(text, ECCM_KW)
        if eccm_hits:
            prof["employs_eccm"] = [v for v, _ in eccm_hits]
            ev["employs_eccm"] = {"value": prof["employs_eccm"], "confidence": CONF_KW,
                                  "source_field": "eccm/mode text",
                                  "evidence": {v: k for v, k in eccm_hits}}

        # keep only radars with at least a purpose or tracking signal (real threats)
        if prof.get("purpose") or prof.get("tracking_method"):
            prof["name"] = head
            prof["_evidence"] = ev
            profiles[head] = prof
    return profiles


def save_profiles(profiles=None):
    profiles = profiles or build_profiles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(profiles, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return len(profiles)


INTEL = ROOT / "data" / "ew" / "documented_cases.json"
THREAT_RADARS = ROOT / "data" / "ew" / "threat_radars.json"

_NUM_FIELDS = ("frequency_GHz", "peak_power_kW", "range_km", "antenna_gain_dB")


def _record_to_profile(r, tier):
    """Convert a curated radar record (documented case OR threat-radar reference)
    into a threat profile carrying provenance + aliases + extra metadata."""
    prof = {k: r[k] for k in ("purpose", "tracking_method", "bands", "scan_type",
                              "freq_agile", "prf_agile", "lpi") + _NUM_FIELDS if k in r}
    prof["name"] = r["name"]
    prof["aliases"] = r.get("aliases", [])
    prof["role"] = r.get("role", "")
    if r.get("system"):
        prof["system"] = r["system"]
    if r.get("country"):
        prof["country"] = r["country"]
    if r.get("sources"):
        prof["sources"] = r["sources"]
    if r.get("case"):
        prof["documented_case"] = r["case"]
    src = "documented_case" if tier == "documented" else "open_reference"
    ev = {}
    for k in ("purpose", "tracking_method", "bands"):
        if k in r:
            ev[k] = {"value": r[k], "tier": tier, "source": src, "confidence": 0.9}
    prof["_evidence"] = ev
    return prof


def load_intel_profiles():
    """Curated adversary threat radars from (a) documented historical cases —
    real cited combat intelligence (tier='documented'), and (b) the SAM/IADS
    threat-radar reference library (tier='reference', open-source refs). These
    are the realistic *targets* of radar countermeasures."""
    out = {}
    for path, tier in ((INTEL, "documented"), (THREAT_RADARS, "reference")):
        if not path.exists():
            continue
        d = json.load(open(path, encoding="utf-8"))
        for r in d.get("radars", []):
            out[r["name"]] = _record_to_profile(r, tier)
    return out


_CACHE = None
_ALIAS = None


def _ensure_cache():
    global _CACHE, _ALIAS
    if _CACHE is not None:
        return
    base = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else build_profiles()
    intel = load_intel_profiles()
    base.update(intel)                       # documented radars take precedence
    _CACHE = base
    _ALIAS = {}
    for name, prof in _CACHE.items():
        _ALIAS[re.sub(r"\s+", "", name.lower())] = name
        for a in prof.get("aliases", []):
            _ALIAS.setdefault(re.sub(r"\s+", "", str(a).lower()), name)


def all_profile_names():
    _ensure_cache()
    return sorted(_CACHE.keys())


def profile_for(model, resolver=None):
    """Look up a radar's threat profile by model name or alias. `resolver` is an
    optional callable name->canonical (e.g. ProvenanceKG.resolve)."""
    _ensure_cache()
    if model in _CACHE:
        return _CACHE[model]
    nk = re.sub(r"\s+", "", model.lower())
    if nk in _ALIAS:
        return _CACHE[_ALIAS[nk]]
    if resolver:
        canon, found = resolver(model)
        if found and canon in _CACHE:
            return _CACHE[canon]
    for k, v in _CACHE.items():
        if re.sub(r"\s+", "", k.lower()) == nk:
            return v
    return None


if __name__ == "__main__":
    profs = build_profiles()
    n = save_profiles(profs)
    # coverage report
    from collections import Counter
    fields = Counter()
    for p in profs.values():
        for f in ("purpose", "bands", "tracking_method", "scan_type",
                  "freq_agile", "prf_agile", "lpi", "employs_eccm"):
            if f in p:
                fields[f] += 1
    print(f"threat profiles built: {n}")
    print("field coverage:")
    for f, c in fields.most_common():
        print(f"  {f:18s} {c:5d}  ({c/n*100:.0f}%)")
    tm = Counter(p["tracking_method"] for p in profs.values() if "tracking_method" in p)
    print("tracking_method distribution:", dict(tm))
