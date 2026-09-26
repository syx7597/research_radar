"""
Parse the standardized field block under each radar title in the PDF (handles 2-column
OCR-zigzag layout).

For each radar entry in the manual, 9 fixed fields appear in a consistent order:
  体制 / 频段 / 研制厂商 / 研制时间 / 装备时间 / 装备机种 / 配用武器 / 价格 / 现状

Output → KG:
  - Relations: hasTechType, hasFrequencyBand, developedBy, deployedOn, compatibleWith
  - Attributes: rd_year_min/max, rd_period_description, year, deployment_year_description,
                price_usd, price_year, price_description, status

This script is regex-only (no LLM). Idempotent.
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
REPORT_PATH = ROOT / "data" / "v2" / "extract_standard_fields_report.md"


# ────────────────────────────────────────────────────────────
#  Step 1: text normalization (column-aware)
# ────────────────────────────────────────────────────────────

# Field markers that are typeset wide-spaced and get OCR'd as char + \n + char
WIDE_LABEL_FIXES = [
    ("体\n制", "体制"),
    ("频\n段", "频段"),
    ("现\n状", "现状"),
    ("价\n格", "价格"),
    ("体\n积", "体积"),
    ("质\n量", "质量"),
]

# Standard fields in canonical order
STANDARD_FIELDS = [
    "体制", "频段", "研制厂商", "研制时间", "装备时间",
    "装备机种", "配用武器", "价格", "现状",
]

# Field-value line prefix (first char of value line often is the 2nd char of marker)
FIELD_PREFIX_STRIP = {
    "体制":   re.compile(r"^制"),
    "频段":   re.compile(r"^段"),
    "现状":   re.compile(r"^状"),
    "价格":   re.compile(r"^格"),
    "体积":   re.compile(r"^积"),
    "质量":   re.compile(r"^量"),
    "研制时间": re.compile(r"^间"),
    "装备时间": re.compile(r"^间"),
    "装备机种": re.compile(r"^种"),
    "配用武器": re.compile(r"^器"),
    "研制厂商": re.compile(r"^商"),
}


def normalize_text(text: str) -> str:
    for old, new in WIDE_LABEL_FIXES:
        text = text.replace(old, new)
    return text


# ────────────────────────────────────────────────────────────
#  Step 2: split into field blocks
# ────────────────────────────────────────────────────────────

# A "next section" anchor is anything that ends a structured block
NEXT_SECTION_MARKERS = STANDARD_FIELDS + [
    "性能数据", "技术特点", "参考文献", "概述", "研制、试验、销售、装备情况",
    "附：", "附录",
]


def find_field_blocks(norm_text: str) -> dict[str, str]:
    """Locate each standard field marker and grab the raw text up to the next marker.
    Returns a dict {field_name: raw_value_text}."""
    # Build a regex that captures ANY anchor with position
    anchor_pat = re.compile(
        r"(?:^|\n)("
        + "|".join(re.escape(m) for m in NEXT_SECTION_MARKERS)
        + r")",
        re.M,
    )
    anchors = [(m.group(1), m.start(), m.end()) for m in anchor_pat.finditer(norm_text)]
    # Build map: for each STANDARD_FIELDS anchor, what's the text until next anchor
    blocks = {}
    for i, (name, start, end) in enumerate(anchors):
        if name not in STANDARD_FIELDS:
            continue
        next_start = anchors[i + 1][1] if i + 1 < len(anchors) else len(norm_text)
        raw = norm_text[end:next_start]
        # Only keep the FIRST occurrence per page (first radar entry on the page)
        if name not in blocks:
            blocks[name] = raw
    return blocks


# ────────────────────────────────────────────────────────────
#  Step 3: per-field value cleaners
# ────────────────────────────────────────────────────────────

def _clean_lines(raw: str, field: str) -> list[str]:
    """Split into lines, drop narrative (contains 。), strip first-char prefix if needed."""
    out = []
    for i, line in enumerate(raw.split("\n")):
        s = line.strip()
        if not s: continue
        # Narrative lines (full sentences) → skip
        if "。" in s and i > 0: continue
        if i == 0:
            pat = FIELD_PREFIX_STRIP.get(field)
            if pat: s = pat.sub("", s)
        s = s.strip()
        if s: out.append(s)
    return out


# Controlled vocabularies / matchers
TECH_VOCAB = [
    "脉冲多普勒", "PD", "连续波", "CW", "单脉冲", "脉冲", "合成孔径", "SAR",
    "逆合成孔径", "ISAR", "动目标指示", "MTI", "动目标显示",
    "相控阵", "有源相控阵", "AESA", "无源相控阵", "PESA",
    "频率扫描", "频扫", "相扫", "机械扫描", "电扫", "数字波束形成", "DBF",
    "调频连续波", "FMCW", "二次雷达", "脉冲压缩", "巴克码",
    "倒置卡塞格伦", "卡塞格伦", "平板缝阵", "缝隙阵列",
    "极化", "圆极化", "线极化", "可变极化",
    "地面动目标指示", "地动目标指示", "GMTI",
    "波束捷变", "频率捷变",
]


def clean_tech_types(raw: str) -> list[str]:
    lines = _clean_lines(raw, "体制")
    text = "，".join(lines)
    items = re.split(r"[，,、；;]", text)
    out = []
    for it in items:
        it = it.strip(" 。；;.")
        if not it: continue
        # Substring match against vocabulary, longest first — only accept vocab hits
        matched = None
        for v in sorted(TECH_VOCAB, key=len, reverse=True):
            if v in it:
                matched = v
                break
        if matched:
            out.append(matched)
        # Reject anything that's a sentence fragment (contains common narrative words)
        elif len(it) <= 8 and not any(c in it for c in "包括以及其中第使能可应那这"):
            out.append(it)
        if len(out) >= 8: break
    seen, dedup = set(), []
    for t in out:
        if t in seen: continue
        seen.add(t); dedup.append(t)
    return dedup


BAND_TOKENS = ["X", "Ku", "Ka", "K", "L", "S", "C", "P", "VHF", "UHF", "HF", "W",
               "I", "J", "Q", "V", "G", "F", "E", "D"]


def clean_freq_bands(raw: str) -> list[str]:
    lines = _clean_lines(raw, "频段")
    if not lines: return []
    # Only the first short line is the band value; subsequent lines are likely narrative
    first = lines[0]
    # Trim narrative tail (e.g. "IJ年11月推出..." → "IJ")
    first = re.split(r"(?:年|月|公司|雷达|系统|工作|波)", first, maxsplit=1)[0]
    out = []
    # Multi-char tokens first (Ku, Ka, VHF, UHF, HF)
    for tok in ("Ku", "Ka", "VHF", "UHF", "HF"):
        if tok in first:
            out.append(tok)
            first = first.replace(tok, " ")
    # Single-letter tokens
    for ch in first:
        if ch in BAND_TOKENS and len(ch) == 1:
            out.append(ch)
    seen, dedup = set(), []
    for t in out:
        if t in seen: continue
        seen.add(t); dedup.append(t)
    return dedup[:4]


def clean_developer(raw: str) -> str:
    lines = _clean_lines(raw, "研制厂商")
    # Take lines until we hit a non-company-looking line
    parts = []
    for line in lines:
        # If line is mostly Chinese AND short — likely narrative — skip
        if "。" in line: break
        if re.search(r"^[A-Z][a-zA-Z]", line) or re.search(r"[A-Za-z]{4,}", line):
            parts.append(line)
        elif parts and re.search(r"^[（(]", line):  # continuation in parens
            parts.append(line)
        elif parts and "原研制" in line:  # "原研制厂商XXX"
            parts.append(line)
        else:
            break
    if not parts: return ""
    name = " ".join(parts)
    # Strip narrative tail (e.g. "APG-70 的机内...")
    name = re.split(r"(?:。|；|;)", name)[0]
    # Strip trailing parenthetical
    name = re.sub(r"\s+APG-\d.*", "", name)
    return name.strip()


def clean_period(raw: str) -> tuple[str, int | None, int | None]:
    """研制时间 → (description, year_min, year_max)."""
    lines = _clean_lines(raw, "研制时间")
    text = " ".join(lines)
    years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", text)]
    text = re.split(r"(?:。|装备时间)", text)[0].strip()
    if not years:
        return text, None, None
    return text, min(years), max(years) if len(years) > 1 else None


def clean_deploy_year(raw: str) -> tuple[str, int | None]:
    lines = _clean_lines(raw, "装备时间")
    text = " ".join(lines)
    text = re.split(r"(?:。|装备机种)", text)[0].strip()
    m = re.search(r"(19\d{2}|20\d{2})", text)
    return text, int(m.group(1)) if m else None


def clean_platforms(raw: str) -> list[str]:
    """装备机种 → list of platform names."""
    lines = _clean_lines(raw, "装备机种")
    text = "，".join(lines)
    text = re.split(r"(?:。|配用武器)", text)[0]
    items = re.split(r"[，,、；;]", text)
    out = []
    for it in items:
        it = it.strip(" 。;；")
        if not it: continue
        # ASCII designation (F-15C, F/A-18E/F, AC-130U, Su-27 etc.)
        if re.match(r"^[A-Za-z][A-Za-z0-9/\-\(\)\+\.]{1,25}$", it):
            out.append(it)
        # Pure Chinese short name (e.g. 苏-27, 米格-29)
        elif re.match(r"^(?:苏|米格|歼|强|轰)-?\d", it):
            out.append(it)
    return out[:10]


WEAPON_PAT = re.compile(r"^(?:AIM|AGM|RIM|R-|R\d|K-|Kh-|MICA|AMRAAM|ASRAAM|Magic|Phoenix|Sidewinder|Sparrow|HARM|Maverick|Penguin|Harpoon|Exocet|ALARM|Stinger|Patriot|Std|Aster|ESSM|SeaSparrow|Akash|Brimstone|Storm|Taurus|JSOW|JDAM|Mk\s?\d|GBU-|20mm|25mm|30mm|40mm|105mm)", re.I)

def clean_weapons(raw: str) -> list[str]:
    """配用武器 → list of weapon names."""
    lines = _clean_lines(raw, "配用武器")
    text = "，".join(lines)
    text = re.split(r"(?:。|价格)", text)[0]
    items = re.split(r"[，,、；;]", text)
    out = []
    for it in items:
        it = it.strip(" 。;；")
        if not it: continue
        if WEAPON_PAT.search(it) or re.match(r"^[A-Z]+-?\d", it):
            out.append(it)
        elif re.match(r"^\d+\s?mm", it):  # 20mm炮 etc.
            out.append(it)
    return out[:12]


def clean_price(raw: str) -> tuple[str, float | None, int | None]:
    """价格 → (description, usd_million, year)."""
    lines = _clean_lines(raw, "价格")
    text = " ".join(lines)
    text = re.split(r"(?:。|现状)", text)[0].strip()
    if not text: return "", None, None
    # Patterns: "320万美元 (2002年币值)" / "300万美元（1998年币值）"
    m_amt = re.search(r"(\d+(?:\.\d+)?)\s*万美元", text)
    m_year = re.search(r"\((\d{4})年", text) or re.search(r"（(\d{4})年", text)
    amt = float(m_amt.group(1)) / 100 if m_amt else None   # convert 万 to million
    year = int(m_year.group(1)) if m_year else None
    return text, amt, year


STATUS_VOCAB = {
    "服役中": "服役中", "在役": "服役中", "正在服役": "服役中", "继续服役": "服役中",
    "已服役": "服役中", "现役": "服役中",
    "退役": "退役", "已退役": "退役", "停产": "退役",
    "研制中": "研制中", "在研": "研制中", "研制阶段": "研制中",
    "试验": "试验阶段", "试验阶段": "试验阶段", "试飞": "试验阶段",
    "生产中": "生产中", "出口": "出口", "升级中": "升级中",
}


def clean_status(raw: str) -> tuple[str, str]:
    """现状 → (description, canonical_status)."""
    lines = _clean_lines(raw, "现状")
    text = " ".join(lines)
    text = re.split(r"(?:。|技术特点|性能数据)", text)[0].strip()
    canon = ""
    for k, v in sorted(STATUS_VOCAB.items(), key=lambda x: -len(x[0])):
        if k in text:
            canon = v
            break
    return text, canon


def detect_orphan_status(norm_text: str) -> tuple[str, str]:
    """OCR sometimes drops '现' leaving lonely '状XX'. Search the page for any
    '\\n状<status_keyword>' (or 现状 / 状) pattern."""
    for k, v in sorted(STATUS_VOCAB.items(), key=lambda x: -len(x[0])):
        m = re.search(rf"(?:^|\n)(?:现)?状\s*({re.escape(k)})", norm_text, re.M)
        if m:
            return m.group(1), v
    return "", ""


# ────────────────────────────────────────────────────────────
#  Step 4: detect which radar this page belongs to
# ────────────────────────────────────────────────────────────

def detect_radar_id(page_text: str, radar_index: dict, alias_lookup: dict = None) -> list[str]:
    """Return canonical radar ids whose name OR alias appears in the top of the page."""
    head = page_text[:800]
    hits_with_pos = []
    # First check canonical ids
    for rid in sorted(radar_index, key=len, reverse=True):
        if len(rid) < 5: continue
        idx = head.find(rid)
        if idx >= 0:
            hits_with_pos.append((idx, rid))
    # Then check aliases (map to canonical id)
    if alias_lookup:
        for alias, canonical in sorted(alias_lookup.items(), key=lambda x: -len(x[0])):
            if len(alias) < 5: continue
            if canonical in radar_index and any(h[1] == canonical for h in hits_with_pos):
                continue   # already detected via canonical
            idx = head.find(alias)
            if idx >= 0:
                hits_with_pos.append((idx, canonical))
    # Sort by earliest position, dedup canonical ids
    hits_with_pos.sort()
    seen, out = set(), []
    for _, rid in hits_with_pos:
        if rid in seen: continue
        seen.add(rid); out.append(rid)
        if len(out) >= 3: break
    return out


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

    # Build radar→pages map (from pdf_results.json if present)
    pdf_results = ROOT / "extraction_results" / "pdf_results.json"
    radar_pages = defaultdict(set)
    if pdf_results.exists():
        with open(pdf_results, encoding="utf-8") as f:
            data = json.load(f)
        for t in data[0]["triples"]:
            if t["head_type"] in ("Radar", "RadarSystem"):
                for p in (t.get("pages") or []):
                    radar_pages[t["head"]].add(p)
    print(f"  pre-known radar→page mapping: {len(radar_pages)} radars")

    # Radar id index for cross-page detection
    radar_index = {e["id"]: e for e in entities if e["type"] in ("Radar", "RadarSystem")}
    # Alias → canonical id lookup
    alias_lookup = {}
    for e in entities:
        if e["type"] in ("Radar", "RadarSystem"):
            for alias in (e.get("aliases", []) or []):
                if alias and alias != e["id"]:
                    alias_lookup[alias] = e["id"]

    existing = {(e["head"], e["relation"], e["tail"]) for e in edges}

    # Counters
    parsed_pages = 0
    radar_assignments = Counter()
    field_hits = Counter()
    new_edges = []
    new_entities = []
    field_filled = Counter()
    samples = defaultdict(list)

    # Walk OCR pages
    for ocr_path in sorted(CACHE_DIR.glob("page_*.txt")):
        page_no = int(re.search(r"(\d+)", ocr_path.name).group(1))
        if not (39 <= page_no <= 510):
            continue
        text = ocr_path.read_text(encoding="utf-8")
        if len(text) < 200: continue
        norm = normalize_text(text)
        blocks = find_field_blocks(norm)
        if not blocks: continue
        parsed_pages += 1
        for f in blocks: field_hits[f] += 1

        # Which radar(s) does this page belong to?
        # Priority 1: pre-known mapping from pdf_results.json (authoritative)
        prelinked = [rid for rid, ps in radar_pages.items() if page_no in ps]
        radar = None
        if prelinked:
            # If multiple radars share this page, pick the one that appears in text;
            # otherwise pick the first (typically the page belongs to one entry).
            in_text = [r for r in prelinked if r in norm]
            radar = in_text[0] if in_text else prelinked[0]
        else:
            # Priority 2: text detection (now alias-aware)
            detected = detect_radar_id(norm, radar_index, alias_lookup)
            if detected:
                radar = detected[0]   # earliest in page
        if radar is None or radar not in ent_by_id: continue
        radar_assignments[radar] += 1
        r_ent = ent_by_id[radar]

        # ─── apply each field
        def add_edge(head, rel, tail, tail_type, conf, evidence):
            key = (head, rel, tail)
            if key in existing: return False
            existing.add(key)
            new_edges.append({
                "head": head, "head_type": r_ent["type"],
                "relation": rel, "tail": tail, "tail_type": tail_type,
                "confidence": conf, "evidence": evidence[:200],
                "source": "extract_std_field",
            })
            return True

        def ensure_entity(eid, etype, name=""):
            if eid in ent_by_id: return
            ent_by_id[eid] = {"id": eid, "type": etype,
                              "name_en": name or eid, "aliases": []}
            entities.append(ent_by_id[eid])
            new_entities.append(eid)

        # 体制
        if "体制" in blocks:
            techs = clean_tech_types(blocks["体制"])
            for t in techs:
                ensure_entity(t, "TechType")
                if add_edge(radar, "hasTechType", t, "TechType", 0.95, f"page {page_no} 体制"):
                    field_filled["hasTechType"] += 1
                    if len(samples["hasTechType"]) < 5:
                        samples["hasTechType"].append((radar, t))
            if techs and not r_ent.get("tech_types"):
                r_ent["tech_types"] = techs

        # 频段
        if "频段" in blocks:
            bands = clean_freq_bands(blocks["频段"])
            for b in bands:
                ensure_entity(b, "FrequencyBand")
                if add_edge(radar, "hasFrequencyBand", b, "FrequencyBand", 0.95, f"page {page_no} 频段"):
                    field_filled["hasFrequencyBand"] += 1
                    if len(samples["hasFrequencyBand"]) < 5:
                        samples["hasFrequencyBand"].append((radar, b))
            if bands and not r_ent.get("frequency_bands"):
                r_ent["frequency_bands"] = bands

        # 研制厂商
        if "研制厂商" in blocks:
            dev = clean_developer(blocks["研制厂商"])
            if dev and len(dev) > 2 and len(dev) < 120:
                ensure_entity(dev, "Manufacturer")
                if add_edge(radar, "developedBy", dev, "Manufacturer", 0.95, f"page {page_no} 研制厂商"):
                    field_filled["developedBy"] += 1
                    if len(samples["developedBy"]) < 5:
                        samples["developedBy"].append((radar, dev))
                if not r_ent.get("developer"):
                    r_ent["developer"] = dev

        # 研制时间
        if "研制时间" in blocks:
            desc, ymin, ymax = clean_period(blocks["研制时间"])
            if desc and not r_ent.get("rd_period_description"):
                r_ent["rd_period_description"] = desc
                field_filled["rd_period_description"] += 1
            if ymin and not r_ent.get("rd_year_min"):
                r_ent["rd_year_min"] = ymin
                field_filled["rd_year_min"] += 1
            if ymax and not r_ent.get("rd_year_max"):
                r_ent["rd_year_max"] = ymax
                field_filled["rd_year_max"] += 1

        # 装备时间
        if "装备时间" in blocks:
            desc, y = clean_deploy_year(blocks["装备时间"])
            if desc and not r_ent.get("deployment_year_description"):
                r_ent["deployment_year_description"] = desc
                field_filled["deployment_year_description"] += 1
            if y and not r_ent.get("year"):
                r_ent["year"] = y
                field_filled["year"] += 1
                if len(samples["year"]) < 5:
                    samples["year"].append((radar, y))

        # 装备机种
        if "装备机种" in blocks:
            plats = clean_platforms(blocks["装备机种"])
            for p in plats:
                ensure_entity(p, "Platform")
                if add_edge(radar, "deployedOn", p, "Platform", 0.95, f"page {page_no} 装备机种"):
                    field_filled["deployedOn"] += 1
                    if len(samples["deployedOn"]) < 5:
                        samples["deployedOn"].append((radar, p))

        # 配用武器
        if "配用武器" in blocks:
            ws = clean_weapons(blocks["配用武器"])
            for w in ws:
                ensure_entity(w, "Weapon")
                if add_edge(radar, "compatibleWith", w, "Weapon", 0.95, f"page {page_no} 配用武器"):
                    field_filled["compatibleWith"] += 1
                    if len(samples["compatibleWith"]) < 5:
                        samples["compatibleWith"].append((radar, w))

        # 价格
        if "价格" in blocks:
            desc, amt, py = clean_price(blocks["价格"])
            if desc and not r_ent.get("price_description"):
                r_ent["price_description"] = desc
                field_filled["price_description"] += 1
            if amt and not r_ent.get("price_usd_million"):
                r_ent["price_usd_million"] = amt
                field_filled["price_usd_million"] += 1
                if len(samples["price_usd_million"]) < 5:
                    samples["price_usd_million"].append((radar, amt))
            if py and not r_ent.get("price_year"):
                r_ent["price_year"] = py

        # 现状
        desc, canon = "", ""
        if "现状" in blocks:
            desc, canon = clean_status(blocks["现状"])
        if not canon:
            desc2, canon2 = detect_orphan_status(norm)
            if canon2:
                desc, canon = desc2, canon2
        if desc and not r_ent.get("status_description"):
            r_ent["status_description"] = desc
            field_filled["status_description"] += 1
        if canon and not r_ent.get("status"):
            r_ent["status"] = canon
            field_filled["status"] += 1

    # ─── save
    edge_data["edges"] = edges + new_edges
    edge_data["count"] = len(edge_data["edges"])
    ent_data["entities"] = entities
    ent_data["count"] = len(entities)
    with open(EDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(edge_data, f, ensure_ascii=False, indent=2)
    with open(ENT_PATH, "w", encoding="utf-8") as f:
        json.dump(ent_data, f, ensure_ascii=False, indent=2)

    # ─── report
    print(f"\nParsed {parsed_pages} pages with standard-field blocks")
    print(f"\nField-marker hit counts (pages with the marker):")
    for f, c in field_hits.most_common():
        print(f"  {f:15s} {c} pages")
    print(f"\nNew edges by relation:")
    for r, c in field_filled.most_common():
        print(f"  {r:30s} +{c}")
    print(f"\nNew entities created: {len(new_entities)}")
    print(f"Radars touched: {len(radar_assignments)}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Standard Field Block Extraction Report\n\n")
        f.write(f"Pages parsed: {parsed_pages}\n")
        f.write(f"Radars touched: {len(radar_assignments)}\n")
        f.write(f"New edges: {len(new_edges)}\n")
        f.write(f"New entities: {len(new_entities)}\n\n")
        f.write("## New edges/attrs by relation\n\n")
        for r, c in field_filled.most_common():
            f.write(f"- `{r}`: +{c}\n")
        f.write("\n## Sample fills\n\n")
        for r, items in samples.items():
            f.write(f"### {r}\n\n")
            for rid, v in items:
                f.write(f"- `{rid}` → {v}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
