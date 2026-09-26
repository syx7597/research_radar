"""
Clean up attribute values across Radar / RadarSystem entities:
  - frequency_bands: 'IJ' / 'I/J' / 'KU' → split to canonical list ['I','J'] / ['Ku']
  - status: 'in_service' / 'development' / 'retired' / 'unknown' → 服役中 / 研制中 / 退役 / (drop)
  - developer: insert spaces in CamelCase strings (TexasInstruments → Texas Instruments)
               strip OCR junk tails like '（原研制厂商'
               fix common OCR typos (Airbome → Airborne)
  - status_description: trim at first sentence/newline boundary if longer than ~30 chars

Also re-emits the corresponding edges (hasFrequencyBand) when bands list was reshaped.

Idempotent.
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
ENT_PATH = ROOT / "data" / "v2" / "radarkg_v2_entities.json"
EDGE_PATH = ROOT / "data" / "v2" / "radarkg_v2_edges.json"

# ────────────────────────────────────────────────────────────
#  frequency_bands normalization
# ────────────────────────────────────────────────────────────

VALID_BANDS = {"X", "Ku", "Ka", "K", "L", "S", "C", "P", "VHF", "UHF", "HF", "W",
               "I", "J", "Q", "V", "G", "F", "E", "D"}
BAND_CANON = {b.upper(): b for b in VALID_BANDS}
BAND_CANON["KU"] = "Ku"
BAND_CANON["KA"] = "Ka"


def normalize_bands(v) -> list[str]:
    """Take whatever shape and return a canonical list of bands."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if not isinstance(item, str): continue
        s = item.strip()
        # split combined token: 'IJ' / 'I/J' / 'I-J' / 'L/X' / 'S,X'
        tokens = re.split(r"[/\-,\s]+", s)
        for tok in tokens:
            tok = tok.strip()
            if not tok: continue
            # Single letter handling
            if len(tok) == 1 and tok.upper() in BAND_CANON:
                out.append(BAND_CANON[tok.upper()])
                continue
            # 'IJ' (two single-letter bands stuck together)
            if len(tok) == 2 and tok[0].upper() in BAND_CANON and tok[1].upper() in BAND_CANON:
                out.append(BAND_CANON[tok[0].upper()])
                out.append(BAND_CANON[tok[1].upper()])
                continue
            # Ku / Ka / VHF / UHF
            if tok.upper() in BAND_CANON:
                out.append(BAND_CANON[tok.upper()])
                continue
    # Dedupe preserve order
    seen, dedup = set(), []
    for t in out:
        if t in seen: continue
        seen.add(t); dedup.append(t)
    return dedup


# ────────────────────────────────────────────────────────────
#  status normalization
# ────────────────────────────────────────────────────────────

STATUS_CANON = {
    "服役中": "服役中", "在役": "服役中", "正在服役": "服役中", "现役": "服役中",
    "已服役": "服役中", "继续服役": "服役中",
    "in_service": "服役中", "in service": "服役中", "active": "服役中",
    "退役": "退役", "已退役": "退役", "停产": "退役",
    "retired": "退役", "out of service": "退役",
    "研制中": "研制中", "在研": "研制中", "研制阶段": "研制中",
    "development": "研制中", "in development": "研制中",
    "试验阶段": "试验阶段", "试验": "试验阶段", "试飞": "试验阶段", "testing": "试验阶段",
    "生产中": "生产中", "production": "生产中",
    "unknown": "", "未知": "", "n/a": "", "": "",
}


def normalize_status(s) -> str:
    if not s or not isinstance(s, str): return ""
    k = s.strip().lower()
    for src, dst in STATUS_CANON.items():
        if k == src.lower():
            return dst
    # Substring match (e.g. "in_service since 1985")
    for src, dst in sorted(STATUS_CANON.items(), key=lambda x: -len(x[0])):
        if src and src.lower() in k:
            return dst
    return s.strip()


# ────────────────────────────────────────────────────────────
#  Developer normalization
# ────────────────────────────────────────────────────────────

DEV_TYPO = {
    "Airbome": "Airborne",
    "Aerosystems": "AeroSystems",
    "原研制厂商": "",
}

DEV_KNOWN = [
    # Known companies that often appear concatenated
    "Texas Instruments", "Hughes Aircraft", "Northrop Grumman", "Lockheed Martin",
    "Raytheon", "Boeing", "General Electric", "Westinghouse", "Honeywell",
    "Thales Airborne Systems", "Thales Aerospace Division", "Thales",
    "Telephonics Corporation", "Emerson Electric", "EMS Technologies",
    "Marconi", "Selex", "Selenia", "Galileo", "Alenia",
    "Saab", "Ericsson", "BAE Systems", "Mitsubishi", "Toshiba",
    "Phazotron-NIIR", "Phazotron", "NIIP", "Tikhomirov", "Leninetz",
    "IAI Elta", "Elta", "IAI",
    "BEL", "DRDO", "Larsen & Toubro",
    "Indra", "Aselsan", "Hensoldt",
    "Thomson-CSF", "Thomson",
    "Przemyslowy Instytut Telekomunikacji",
    "Norden", "Bendix", "Sperry", "Magnavox", "ITT",
]


def normalize_developer(s: str) -> str:
    if not s or not isinstance(s, str): return s
    out = s
    # Apply typo fixes
    for typo, fix in DEV_TYPO.items():
        out = out.replace(typo, fix)
    # Strip Chinese parenthetical tail like '（原研制厂商' / '（原研制厂商Hughes'
    out = re.split(r"[（(]\s*原研制厂商", out)[0].strip()
    # Strip Chinese narrative suffix
    out = re.split(r"[，,]\s*由", out)[0].strip()
    out = out.rstrip("，,。：:")
    # Insert spaces in CamelCase company names by matching known patterns
    for company in sorted(DEV_KNOWN, key=len, reverse=True):
        compact = company.replace(" ", "")
        if compact in out and company not in out:
            out = out.replace(compact, company)
    # Generic camelCase → "Camel Case" when there's no space
    if " " not in out and len(out) > 8 and out.isascii():
        # Insert space before uppercase if preceded by lowercase
        out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    return out.strip()


# ────────────────────────────────────────────────────────────
#  status_description noise trim
# ────────────────────────────────────────────────────────────

STATUS_TRIM_BREAKS = ["工作方式", "技术特点", "性能数据", "参考文献", "概述"]


def trim_status_desc(s: str) -> str:
    if not s or not isinstance(s, str): return s
    # Trim at first break marker
    for br in STATUS_TRIM_BREAKS:
        idx = s.find(br)
        if idx > 0:
            s = s[:idx]
    # Trim at first 。
    if "。" in s:
        s = s.split("。")[0]
    # Truncate if absurdly long
    if len(s) > 60:
        s = s[:60]
    return s.strip()


# ────────────────────────────────────────────────────────────
def main():
    with open(ENT_PATH, encoding="utf-8") as f:
        ent_data = json.load(f)
    with open(EDGE_PATH, encoding="utf-8") as f:
        edge_data = json.load(f)
    entities = ent_data["entities"]
    edges = edge_data["edges"]

    radars = [e for e in entities if e["type"] in ("Radar", "RadarSystem")]
    counters = Counter()

    # ──── 1. frequency_bands & re-emit edges
    existing_band_edges = set()
    for ed in edges:
        if ed["relation"] == "hasFrequencyBand":
            existing_band_edges.add((ed["head"], ed["tail"]))
    new_edges_band = []
    for r in radars:
        old = r.get("frequency_bands")
        if not old: continue
        new = normalize_bands(old)
        if new != old:
            r["frequency_bands"] = new
            counters["bands_normalized"] += 1
        # Ensure edges exist for the bands
        for b in new:
            if (r["id"], b) in existing_band_edges: continue
            existing_band_edges.add((r["id"], b))
            new_edges_band.append({
                "head": r["id"], "head_type": r["type"],
                "relation": "hasFrequencyBand",
                "tail": b, "tail_type": "FrequencyBand",
                "confidence": 0.9,
                "evidence": "normalized from radar attribute",
                "source": "normalize_attrs",
            })
            counters["band_edges_added"] += 1
    edges.extend(new_edges_band)

    # Drop hasFrequencyBand edges whose tails are now invalid (e.g. 'IJ' as a single band)
    valid_bands = set(VALID_BANDS)
    pruned = 0
    cleaned = []
    for ed in edges:
        if ed["relation"] == "hasFrequencyBand" and ed["tail"] not in valid_bands:
            pruned += 1
            continue
        cleaned.append(ed)
    edges = cleaned
    counters["band_edges_pruned"] = pruned

    # ──── 2. status normalization
    for r in radars:
        old = r.get("status")
        if old:
            new = normalize_status(old)
            if new != old:
                if new:
                    r["status"] = new
                else:
                    r.pop("status", None)
                counters["status_normalized"] += 1
        # status_description trim
        sd = r.get("status_description")
        if sd:
            sd_new = trim_status_desc(sd)
            if sd_new != sd:
                r["status_description"] = sd_new
                counters["status_desc_trimmed"] += 1

    # ──── 3. developer normalization
    for r in radars:
        d = r.get("developer")
        if d:
            d_new = normalize_developer(d)
            if d_new != d:
                r["developer"] = d_new
                counters["developer_normalized"] += 1

    # Also normalize Manufacturer entity name_en + id when matching a CamelCase typo
    for e in entities:
        if e["type"] in ("Manufacturer", "Organization"):
            ne = normalize_developer(e.get("name_en", ""))
            if ne and ne != e.get("name_en"):
                e["name_en"] = ne
                counters["mfr_name_normalized"] += 1

    # ──── 4. Also normalize developedBy edges (tail = manufacturer name)
    for ed in edges:
        if ed["relation"] == "developedBy":
            new = normalize_developer(ed["tail"])
            if new and new != ed["tail"]:
                ed["tail"] = new
                counters["dev_edge_normalized"] += 1

    # Save
    edge_data["edges"] = edges
    edge_data["count"] = len(edges)
    ent_data["entities"] = entities
    ent_data["count"] = len(entities)
    with open(EDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    print("Normalization summary:")
    for k, v in counters.most_common():
        print(f"  {k:30s} {v}")
    print(f"\nFinal: {len(entities)} entities, {len(edges)} edges")


if __name__ == "__main__":
    main()
