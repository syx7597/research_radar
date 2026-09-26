"""
Parse the 性能数据 section under each radar entry. Open schema: collect any
labeled subfield, parse numeric value + unit where recognized, store raw
description otherwise.

Outputs:
  - hasMode edges (from 工作方式 list)
  - Numeric attributes per known field (range_km / peak_power_kW / weight_kg / volume_m3 /
    pulse_width_us / prf_kHz_min/max / beam_width_deg / antenna_gain_dB / antenna_size_m /
    frequency_GHz_min/max / input_power_kVA / avg_power_W / mtbf_hours / lru_count /
    resolution_m / scan_range_deg / temp_min_C / temp_max_C / ...)
  - Open-schema raw description attrs for unrecognized labels: `<canonical_label>_description`
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "v2" / "ocr_cache"
ENT_PATH = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"
REPORT_PATH = ROOT / "data" / "v2" / "extract_performance_data_report.md"


# ────────────────────────────────────────────────────────────
#  Subfield → (canonical attr name, value parser)
#
#  Keys are PREFIX matches because OCR sometimes truncates labels
#  (e.g. "作用距" instead of "作用距离").
# ────────────────────────────────────────────────────────────

# Helper unit-aware float parsers
def _re_num():
    return r"(\d+(?:\.\d+)?)"

def parse_range_km(val):
    """e.g. '185km', '300海里', '40km(地面目标)~200km(空中)'."""
    # Try km first
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*(?:km|公里|千米)", val)
    if m: return {"range_km_min": float(m.group(1)), "range_km_max": float(m.group(2))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里|千米)", val)
    if m: return {"range_km": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*海里", val)
    if m: return {"range_km": float(m.group(1)) * 1.852}
    m = re.search(r"(\d+(?:\.\d+)?)\s*n?\s*mile", val, re.I)
    if m: return {"range_km": float(m.group(1)) * 1.609}
    return {}


def parse_peak_power_kW(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*(MW|kW|W)", val)
    if m:
        lo, hi, u = float(m.group(1)), float(m.group(2)), m.group(3)
        scale = {"MW": 1000, "kW": 1, "W": 0.001}[u]
        return {"peak_power_kW_min": lo*scale, "peak_power_kW_max": hi*scale}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(MW|kW|W)", val)
    if m:
        v, u = float(m.group(1)), m.group(2)
        scale = {"MW": 1000, "kW": 1, "W": 0.001}[u]
        return {"peak_power_kW": v * scale}
    return {}


def parse_avg_power_W(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kW|W)", val)
    if m:
        v = float(m.group(1))
        if m.group(2) == "kW": v *= 1000
        return {"avg_power_W": v}
    return {}


def parse_weight_kg(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg", val)
    if m: return {"weight_kg": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:磅|lb)", val, re.I)
    if m: return {"weight_kg": float(m.group(1)) * 0.4536}
    return {}


def parse_volume_m3(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*m[³³3]", val)
    if m: return {"volume_m3": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*L", val)
    if m: return {"volume_m3": float(m.group(1)) * 0.001}
    return {}


def parse_freq_GHz(val):
    """e.g. '8GHz~20GHz', '9.375GHz', 'X 8.55GHz'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*GHz", val)
    if m: return {"frequency_GHz_min": float(m.group(1)), "frequency_GHz_max": float(m.group(2))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*MHz", val)
    if m: return {"frequency_GHz_min": float(m.group(1))/1000, "frequency_GHz_max": float(m.group(2))/1000}
    m = re.search(r"(\d+(?:\.\d+)?)\s*GHz", val)
    if m: return {"frequency_GHz": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*MHz", val)
    if m: return {"frequency_GHz": float(m.group(1))/1000}
    return {}


def parse_prf_kHz(val):
    """脉冲重复频率 e.g. '200kHz', '1.5-2.0kHz'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*kHz", val)
    if m: return {"prf_kHz_min": float(m.group(1)), "prf_kHz_max": float(m.group(2))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*kHz", val)
    if m: return {"prf_kHz": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", val)
    if m: return {"prf_kHz": float(m.group(1))/1000}
    return {}


def parse_pulse_width_us(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～至-]\s*(\d+(?:\.\d+)?)\s*[μu]?s", val)
    if m: return {"pulse_width_us_min": float(m.group(1)), "pulse_width_us_max": float(m.group(2))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*[μu]s", val)
    if m: return {"pulse_width_us": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*ms", val)
    if m: return {"pulse_width_us": float(m.group(1)) * 1000}
    return {}


def parse_beam_width_deg(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[°度]", val)
    if m: return {"beam_width_deg": float(m.group(1))}
    return {}


def parse_antenna_size_m(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*(?:m|米)", val)
    if m: return {"antenna_width_m": float(m.group(1)), "antenna_height_m": float(m.group(2))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米)", val)
    if m: return {"antenna_size_m": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", val)
    if m: return {"antenna_size_m": float(m.group(1)) / 100}
    return {}


def parse_antenna_gain_dB(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*dBi?", val, re.I)
    if m: return {"antenna_gain_dB": float(m.group(1))}
    return {}


def parse_input_power_kVA(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*kVA", val, re.I)
    if m: return {"input_power_kVA": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*kW", val)
    if m: return {"input_power_kVA": float(m.group(1))}
    return {}


def parse_mtbf_hours(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*h", val)
    if m: return {"mtbf_hours": float(m.group(1))}
    return {}


def parse_lru_count(val):
    m = re.search(r"(\d+)\s*(?:个|LRU|lru)", val)
    if m: return {"lru_count": int(m.group(1))}
    return {}


def parse_resolution_m(val):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米)", val)
    if m: return {"resolution_m": float(m.group(1))}
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", val)
    if m: return {"resolution_m": float(m.group(1)) / 100}
    return {}


def parse_temp_C(val):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[~～至-]\s*(-?\d+(?:\.\d+)?)\s*[°℃]C?", val)
    if m: return {"temp_min_C": float(m.group(1)), "temp_max_C": float(m.group(2))}
    return {}


# Map: prefix → (canonical_attr_label, value_parser_or_None_for_description_only)
# value_parser returns dict of {attr_name: numeric_value}; otherwise just store raw description
SUBFIELD_VOCAB = [
    ("作用距离",      "range",                  parse_range_km),
    ("作用距",        "range",                  parse_range_km),
    ("探测距离",      "range",                  parse_range_km),
    ("探测距",        "range",                  parse_range_km),
    ("最大作用距离",  "range",                  parse_range_km),
    ("最大探测距离",  "range",                  parse_range_km),
    ("工作频率",      "frequency",              parse_freq_GHz),
    ("工作频",        "frequency",              parse_freq_GHz),
    ("中心频率",      "frequency",              parse_freq_GHz),
    ("载频",          "frequency",              parse_freq_GHz),
    ("峰值功率",      "peak_power",             parse_peak_power_kW),
    ("峰值功",        "peak_power",             parse_peak_power_kW),
    ("平均功率",      "avg_power",              parse_avg_power_W),
    ("平均功",        "avg_power",              parse_avg_power_W),
    ("脉冲宽度",      "pulse_width",            parse_pulse_width_us),
    ("脉冲宽",        "pulse_width",            parse_pulse_width_us),
    ("脉宽",          "pulse_width",            parse_pulse_width_us),
    ("重复频率",      "prf",                    parse_prf_kHz),
    ("重复频",        "prf",                    parse_prf_kHz),
    ("脉冲重复",      "prf",                    parse_prf_kHz),
    ("波束宽度",      "beam_width",             parse_beam_width_deg),
    ("波束宽",        "beam_width",             parse_beam_width_deg),
    ("天线尺寸",      "antenna_size",           parse_antenna_size_m),
    ("天线尺",        "antenna_size",           parse_antenna_size_m),
    ("天线型号",      "antenna_type",           None),
    ("天线型",        "antenna_type",           None),
    ("天线增益",      "antenna_gain",           parse_antenna_gain_dB),
    ("天线增",        "antenna_gain",           parse_antenna_gain_dB),
    ("扫描方式",      "scan_method",            None),
    ("扫描方",        "scan_method",            None),
    ("扫描范围",      "scan_range",             None),
    ("扫描范",        "scan_range",             None),
    ("扫描速率",      "scan_rate",              None),
    ("方位",          "azimuth",                None),
    ("俯仰",          "elevation",              None),
    ("分辨力",        "resolution",             parse_resolution_m),
    ("分辨率",        "resolution",             parse_resolution_m),
    ("输入功率",      "input_power",            parse_input_power_kVA),
    ("输入功",        "input_power",            parse_input_power_kVA),
    ("耗电",          "input_power",            parse_input_power_kVA),
    ("功耗",          "input_power",            parse_input_power_kVA),
    ("体积",          "volume",                 parse_volume_m3),
    ("质量",          "weight",                 parse_weight_kg),
    ("重量",          "weight",                 parse_weight_kg),
    ("MTBF",          "mtbf",                   parse_mtbf_hours),
    ("LRU",           "lru",                    parse_lru_count),
    ("BIT",           "bit",                    None),
    ("可靠性",        "reliability",            None),
    ("发射机",        "transmitter",            None),
    ("接收机",        "receiver",               None),
    ("接收通道",      "receiver",               None),
    ("信号处理",      "signal_processing",      None),
    ("显示器",        "display",                None),
    ("显示",          "display",                None),
    ("数据处理",      "data_processing",        None),
    ("分系统",        "subsystems",             None),
    ("射频带宽",      "rf_bandwidth",           None),
    ("工作温度",      "temp_range",             parse_temp_C),
    ("冷却",          "cooling",                None),
    ("电源",          "power_supply",           None),
    ("ECCM",          "eccm",                   None),
    ("ECM",           "ecm",                    None),
    ("驱动方式",      "drive_method",           None),
]


# Modes parsing — split list
def parse_modes(text):
    """Split a 工作方式 block into individual mode names."""
    out = []
    # Remove 空空: / 空地: prefixes which are category headers
    # Split on commas, 、, ;
    text = re.sub(r"(?:空空|空地|空海|对地|对空|对海)\s*[:：]", " ", text)
    items = re.split(r"[，,、；;]", text)
    for it in items:
        it = it.strip(" 。、（）()")
        if not it or len(it) < 2 or len(it) > 20: continue
        # Reject narrative-looking items
        if any(c in it for c in "包括以及该其使能可"): continue
        # Reject pure number/parenthetical
        if not any('一' <= c <= '鿿' for c in it) and not any(c.isalpha() for c in it): continue
        out.append(it)
    # Dedup keep order
    seen, dedup = set(), []
    for t in out:
        if t in seen: continue
        seen.add(t); dedup.append(t)
    return dedup[:20]


# ────────────────────────────────────────────────────────────
#  Block locator and parser
# ────────────────────────────────────────────────────────────

WIDE_LABEL_FIXES = [
    ("体\n制", "体制"), ("频\n段", "频段"), ("现\n状", "现状"),
    ("价\n格", "价格"), ("体\n积", "体积"), ("质\n量", "质量"),
]


def normalize_text(text):
    for old, new in WIDE_LABEL_FIXES:
        text = text.replace(old, new)
    return text


END_ANCHORS = ["技术特点", "参考文献", "附：", "附录", "概述", "研制、试验、销售、装备情况"]


def find_perf_block(norm: str) -> str:
    """Return the text of 性能数据 section, or empty."""
    m = re.search(r"性能数据(.*?)(?:" + "|".join(re.escape(a) for a in END_ANCHORS) + r"|$)",
                  norm, re.S)
    return m.group(1) if m else ""


def parse_perf_block(block: str) -> tuple[list[str], dict]:
    """
    Returns (modes_list, attr_dict).
    attr_dict: {canonical_attr: value-or-description}
    """
    modes = []
    attrs = {}
    open_descs = {}    # for fields we don't have parsers for, store as <name>_description

    # First, try to find 工作方式 block(s) - it can be very long (multi-line list)
    m = re.search(r"工作方式([\s\S]*?)(?:" + "|".join(re.escape(x) for x in
                  ["扫描范围","扫描方","作用距","作用距离","峰值功率","峰值功","脉冲宽","重复频","波束宽","天线尺",
                   "天线型","天线增","显示器","显示","体积","质量","重量","MTBF","LRU","BIT","发射机","接收",
                   "工作温度","分辨","信号处理","分系统","数据处理","驱动","耗电","功耗","输入功","可靠性","ECCM","ECM"]) +
                  r"|$)", block, re.S)
    if m:
        modes_raw = m.group(1)
        modes = parse_modes(modes_raw)

    # Then walk line by line for other subfields
    lines = block.split("\n")
    current_label = None
    current_attr = None
    current_parser = None
    current_value_accum = []

    def commit():
        nonlocal current_label, current_attr, current_parser, current_value_accum
        if current_label and current_value_accum:
            val_text = " ".join(current_value_accum).strip()
            # If we have a numeric parser, try it
            if current_parser:
                parsed = current_parser(val_text)
                for k, v in parsed.items():
                    if k not in attrs:
                        attrs[k] = v
            # Always store raw description too (open schema)
            desc_key = current_attr + "_description"
            if desc_key not in attrs and 2 <= len(val_text) <= 200:
                attrs[desc_key] = val_text[:200]
        current_label = current_attr = current_parser = None
        current_value_accum = []

    for line in lines:
        s = line.strip()
        if not s: continue
        if "。" in s: continue                       # narrative
        # Detect a subfield label at start
        matched_label = None
        for prefix, attr, parser in SUBFIELD_VOCAB:
            if s.startswith(prefix):
                matched_label = (prefix, attr, parser)
                break
        if matched_label:
            # Commit previous
            commit()
            prefix, attr, parser = matched_label
            current_label = prefix
            current_attr = attr
            current_parser = parser
            # Value is the rest of the line
            tail = s[len(prefix):].strip(" :：")
            if tail:
                current_value_accum.append(tail)
        else:
            # Continuation of current field
            if current_label and len(s) < 80:
                current_value_accum.append(s)
    commit()

    # Drop modes-related attrs that bled into open_descs
    return modes, attrs


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    print("Loading v2 KG…")
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]
    ent_by_id = {e["id"]: e for e in entities}
    print(f"  {len(entities)} entities, {len(edges)} edges")

    # Build radar→pages
    pdf_results = ROOT / "extraction_results" / "pdf_results.json"
    radar_pages = defaultdict(set)
    if pdf_results.exists():
        with open(pdf_results, encoding="utf-8") as f:
            data = json.load(f)
        for t in data[0]["triples"]:
            if t["head_type"] in ("Radar", "RadarSystem"):
                for p in (t.get("pages") or []):
                    radar_pages[t["head"]].add(p)

    radar_index = {e["id"]: e for e in entities if e["type"] in ("Radar", "RadarSystem")}
    # Alias → canonical lookup
    alias_lookup = {}
    for e in entities:
        if e["type"] in ("Radar", "RadarSystem"):
            for alias in (e.get("aliases", []) or []):
                if alias and alias != e["id"]:
                    alias_lookup[alias] = e["id"]
    existing = {(e["head"], e["relation"], e["tail"]) for e in edges}

    new_edges = []
    new_entities = []
    mode_added = 0
    attr_filled = Counter()
    attr_samples = defaultdict(list)
    pages_with_block = 0
    radars_touched = set()

    for ocr_path in sorted(CACHE_DIR.glob("page_*.txt")):
        page_no = int(re.search(r"(\d+)", ocr_path.name).group(1))
        if not (39 <= page_no <= 510): continue
        text = ocr_path.read_text(encoding="utf-8")
        if "性能数据" not in text: continue
        norm = normalize_text(text)
        block = find_perf_block(norm)
        if not block or len(block) < 50: continue
        pages_with_block += 1

        # Find radar for this page
        prelinked = [rid for rid, ps in radar_pages.items() if page_no in ps]
        radar = None
        if prelinked:
            in_text = [r for r in prelinked if r in norm]
            radar = in_text[0] if in_text else prelinked[0]
        else:
            # Match against canonical IDs first
            for rid in sorted(radar_index, key=len, reverse=True):
                if len(rid) < 5: continue
                if rid in norm[:800]:
                    radar = rid; break
            # Fallback: alias lookup → canonical id
            if radar is None:
                best_pos = 1e9
                for alias, canonical in alias_lookup.items():
                    if len(alias) < 5: continue
                    pos = norm[:800].find(alias)
                    if 0 <= pos < best_pos:
                        best_pos = pos
                        radar = canonical
        if radar is None or radar not in ent_by_id: continue

        r_ent = ent_by_id[radar]
        radars_touched.add(radar)

        # Parse
        modes, attrs = parse_perf_block(block)

        # Apply modes
        for m in modes:
            if m not in ent_by_id:
                ent_by_id[m] = {"id": m, "type": "RadarMode", "name_zh": m, "name_en": m, "aliases": []}
                entities.append(ent_by_id[m])
                new_entities.append(m)
            key = (radar, "hasMode", m)
            if key in existing: continue
            existing.add(key)
            new_edges.append({
                "head": radar, "head_type": r_ent["type"],
                "relation": "hasMode",
                "tail": m, "tail_type": "RadarMode",
                "confidence": 0.93,
                "evidence": f"page {page_no} 性能数据 工作方式",
                "source": "extract_perf_data",
            })
            mode_added += 1

        # Apply attrs (don't overwrite existing values)
        for k, v in attrs.items():
            if r_ent.get(k) in (None, "", 0, []):
                r_ent[k] = v
                attr_filled[k] += 1
                if len(attr_samples[k]) < 3:
                    attr_samples[k].append((radar, v))

    # Save
    edge_data["edges"] = edges + new_edges
    edge_data["count"] = len(edge_data["edges"])
    ent_data["entities"] = entities
    ent_data["count"] = len(entities)
    with open(EDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    print(f"\nPages with 性能数据 block: {pages_with_block}")
    print(f"Radars touched: {len(radars_touched)}")
    print(f"New hasMode edges: +{mode_added}")
    print(f"New entities (modes etc.): +{len(new_entities)}")
    print(f"\nAttribute fills (top 30):")
    for k, c in attr_filled.most_common(30):
        print(f"  {k:30s} +{c}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"""# Performance Data Extraction Report

Pages with 性能数据 block: {pages_with_block}
Radars touched: {len(radars_touched)}
New hasMode edges: +{mode_added}
New entities: +{len(new_entities)}

## Attribute fills
""")
        for k, c in attr_filled.most_common():
            f.write(f"- `{k}`: +{c}\n")
        f.write("\n## Sample fills\n\n")
        for k, items in list(attr_samples.items())[:20]:
            f.write(f"### {k}\n\n")
            for rid, v in items:
                f.write(f"- `{rid}` → {v}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
