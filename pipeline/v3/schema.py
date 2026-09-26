"""
v3 schema 编译器 —— lexicon/relations.json 是唯一事实源。

导出:
  EDGE_RELATIONS   24 个实体关系（→ 边）
  ATTR_RELATIONS   19 个字面量关系（→ 实体属性，永不进关系空间）
  TYPE_SIG         关系 → (head_types, tail_types)
  MULTI_VALUED     多值关系集合（单值关系冲突要记录）
  route_relations(heading)      章节标题 → 关系子集（ODKE+ 式动态本体子集）
  prompt_block(relations)       关系子集 → prompt 片段（自动生成，勿手写）
  resolve_alias(surface, kind)  保守别名解析（仅词表命中，不做模糊合并）
  normalize_freq_band / parse_measure  数值/单位归一
  NON_RADAR_ENTITIES            假雷达头黑名单（复用 extract_from_pdf）
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "extraction"))

from extract_from_pdf import NON_RADAR_ENTITIES, normalize_freq_band  # noqa: E402,F401

_LEX     = json.loads((ROOT / "lexicon" / "relations.json").read_text(encoding="utf-8"))
_ALIASES = json.loads((ROOT / "lexicon" / "entity_aliases.json").read_text(encoding="utf-8"))

RELATIONS      = _LEX["relations"]
EDGE_RELATIONS = {k: v for k, v in RELATIONS.items() if v.get("tail_kind") == "entity"}
ATTR_RELATIONS = {k: v for k, v in RELATIONS.items() if v.get("tail_kind") == "literal"}
TYPE_SIG       = {k: (set(v.get("head_types", [])), set(v.get("tail_types", [])))
                  for k, v in RELATIONS.items()}
MULTI_VALUED   = {k for k, v in RELATIONS.items() if v.get("is_multi_valued")}

# 字面量关系 → 属性名（hasRange → range_description 之类的摊平由此杜绝：
# 属性挂实体，snake_case 命名）
def attr_name(rel: str) -> str:
    s = re.sub(r"^has", "", rel)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# ─────────────────────────────────────────────────────────────
#  章节 → 关系子集路由（按 E1 语料的真实章节名设计）
# ─────────────────────────────────────────────────────────────

_CORE = ["operatedBy", "developedBy", "countryOfOrigin", "deployedOn",
         "hasFrequencyBand", "hasTechType", "hasFunction", "hasVariant",
         "derivedFrom", "replaces", "exportedTo", "usedIn"]

_SECTION_ROUTES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"operator|users|export|customer|使用国|用户", re.I),
     ["operatedBy", "exportedTo", "installedAt"]),
    (re.compile(r"variant|version|model|derivative|upgrade|改型|型号|衍生", re.I),
     ["hasVariant", "upgradeOf", "replaces", "replacedBy", "derivedFrom", "similarTo"]),
    (re.compile(r"develop|history|background|origin|program|研[发制]|历史", re.I),
     ["developedBy", "countryOfOrigin", "usedIn", "derivedFrom", "replaces"]),
    (re.compile(r"design|description|technical|characteristic|specification|"
                r"antenna|transmitter|receiver|signal|性能|技术|设计", re.I),
     ["hasTechType", "hasFrequencyBand", "hasMode", "hasFunction",
      "hasSubsystem", "hasComponent", "meetsStandard", "compatibleWith"]),
    (re.compile(r"deploy|platform|service|installation|site|location|部署|装备", re.I),
     ["deployedOn", "installedAt", "operatedBy", "usedIn"]),
]


def route_relations(heading: str | None, source_kind: str) -> list[str]:
    if source_kind == "rt":            # RT 卡：描述短而密，规格已走确定性层
        return ["hasTechType", "hasFrequencyBand", "hasMode", "hasFunction",
                "deployedOn", "operatedBy", "developedBy", "hasVariant",
                "derivedFrom", "replaces", "replacedBy", "installedAt"]
    if heading:
        for rx, rels in _SECTION_ROUTES:
            if rx.search(heading):
                return rels
    return _CORE


# ─────────────────────────────────────────────────────────────
#  prompt 片段自动生成
# ─────────────────────────────────────────────────────────────

def prompt_block(relations: list[str]) -> str:
    lines = []
    for r in relations:
        spec = RELATIONS.get(r)
        if not spec:
            continue
        heads = "/".join(sorted(TYPE_SIG[r][0])) or "Radar"
        tails = "/".join(sorted(TYPE_SIG[r][1])) or "Entity"
        lines.append(f"- {r}（{spec.get('zh_label', r)}）: {heads} → {tails}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  保守别名解析（词表命中才归一，绝不模糊合并——消歧已证伪）
# ─────────────────────────────────────────────────────────────

def _build_alias_index() -> dict:
    idx = {}
    for kind, key in [("Country", "countries"), ("RadarMode", "radar_modes"),
                      ("TechType", "tech_types"), ("Function", "functions"),
                      ("NavalVessel", "naval_vessel_classes")]:
        for canon, spec in (_ALIASES.get(key) or {}).items():
            if not isinstance(spec, dict):      # 个别类目是元数据键（如 valid 列表）
                continue
            surfaces = {canon, spec.get("zh", ""), spec.get("en", "")}
            surfaces.update(spec.get("aliases", []))
            for s in surfaces:
                if s:
                    idx[(kind, s.lower())] = canon
    # frequency_bands 是合法值枚举而非别名 dict：加 "X band"/"X-band" 变体
    for b in (_ALIASES.get("frequency_bands") or {}).get("valid", []):
        idx[("FrequencyBand", b.lower())]          = b
        idx[("FrequencyBand", f"{b.lower()} band")] = b
        idx[("FrequencyBand", f"{b.lower()}-band")] = b
    return idx


_ALIAS_IDX = _build_alias_index()


def resolve_alias(surface: str, kind: str) -> str:
    """命中词表返回规范名，否则原样返回（不猜）。"""
    return _ALIAS_IDX.get((kind, surface.strip().lower()), surface.strip())


# ─────────────────────────────────────────────────────────────
#  数值/单位归一（确定性层与属性抽取共用）
# ─────────────────────────────────────────────────────────────

_MEASURE_RX = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(km|kilometers?|nmi|nm|nautical miles?|mi|miles?|"
    r"GHz|MHz|kHz|Hz|MW|kW|W|[μµ]s|us|ms|r\.?p\.?m\.?|pps|[°º]|deg|degrees?|"
    r"m|ft|feet|kg|t|tons?)", re.I)

_TO_CANON = {  # (unit → (canonical_unit, factor))
    "km": ("km", 1), "kilometer": ("km", 1), "kilometers": ("km", 1),
    "nmi": ("km", 1.852), "nm": ("km", 1.852),   # 本语料 NM 恒为海里
    "nautical mile": ("km", 1.852), "nautical miles": ("km", 1.852),
    "mi": ("km", 1.609), "mile": ("km", 1.609), "miles": ("km", 1.609),
    "ghz": ("GHz", 1), "mhz": ("GHz", 0.001), "khz": ("GHz", 1e-6), "hz": ("GHz", 1e-9),
    "mw": ("kW", 1000), "kw": ("kW", 1), "w": ("kW", 0.001),
    "μs": ("us", 1), "us": ("us", 1), "ms": ("us", 1000),
    "rpm": ("rpm", 1), "pps": ("pps", 1),
    "°": ("deg", 1), "deg": ("deg", 1), "degree": ("deg", 1), "degrees": ("deg", 1),
    "m": ("m", 1), "ft": ("m", 0.3048), "feet": ("m", 0.3048),
    "kg": ("kg", 1), "t": ("kg", 1000), "ton": ("kg", 1000), "tons": ("kg", 1000),
}


_CYR = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "Yo",
    "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Shch",
    "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
}
RU_CANON = {
    "ЖУК": "Zhuk", "БАРС": "Bars", "ИРБИС": "Irbis", "МЕЧ": "Mech",
    "КОПЬЁ": "Kopyo", "КОПЬЕ": "Kopyo", "САПФИР": "Sapfir", "МОСКИТ": "Moskit",
    "ЗАСЛОН": "Zaslon", "ТОПАЗ": "Topaz", "ОСА": "Osa", "АРБАЛЕТ": "Arbalet",
    "АДЪЮТАНТ": "Adjutant", "ПЕРО": "Pero", "РЛПК": "RLPK", "ВИДИМОСТЬ": "Vidimost",
    "КОНТУР": "Kontur", "ЛУЧ": "Luch", "НИТБ": "NITB", "ОВОД": "Ovod",
    "ШМЕЛЬ": "Shmel", "ЮР": "YuR", "РП": "RP",
}


def has_cyrillic(s: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", s or ""))


def cyrillic_to_latin(s: str) -> str:
    """逐词转写；已知雷达名走 RU_CANON，未知词逐字母转。非西里尔部分原样保留。"""
    if not has_cyrillic(s):
        return s
    out = []
    for tok in re.findall(r"[А-Яа-яЁё]+|[^А-Яа-яЁё]+", s):
        if not has_cyrillic(tok):
            out.append(tok)
            continue
        up = tok.upper()
        if up in RU_CANON:
            out.append(RU_CANON[up])
            continue
        out.append("".join(_CYR.get(ch, _CYR.get(ch.upper(), ch)) for ch in tok))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def extract_designation(name: str) -> str | None:
    """抽取跨语言一致的型号编号（N011M / N035 …）作对齐锚点。"""
    m = re.search(r"\bN\d{2,3}[A-Z]?\b", name, re.I)
    return m.group().upper() if m else None


_NAVAL_RX = re.compile(
    r"destroyer|frigate|cruiser|corvette|carrier|battleship|submarine|"
    r"patrol boat|vessel|warship|-class|\bship\b|boat|舰|艇|护卫|驱逐|巡洋|航母|潜艇",
    re.I)
_AIR_RX = re.compile(
    r"\b(F|MiG|Su|Mi|AH|CH|UH|SH|OH|KC|E|P|B|A|C|T)-\d|fighter|helicopter|"
    r"aircraft|bomber|Boeing|Airbus|Tupolev|Sukhoi|UAV|drone|飞机|直升机|"
    r"战斗机|轰炸机|预警机|无人机", re.I)
_GROUND_RX = re.compile(
    r"vehicle|truck|chassis|TEL|launcher|tracked|wheeled|车|底盘|发射车", re.I)


def classify_platform(name: str) -> str:
    n = name.strip()
    if _NAVAL_RX.search(n):
        return "NavalVessel"
    if _GROUND_RX.search(n):
        return "GroundVehicle"
    if _AIR_RX.search(n):
        return "Aircraft"
    return "Platform"


def parse_measure(raw: str, unit_hint: str | None = None) -> dict | None:
    """'280 mi (450 km)' → {'value': 450, 'unit': 'km'}（括号内公制优先）。"""
    raw = re.sub(r"\s*on\w+=.*$", "", raw)               # RT 站 JS tooltip 垃圾
    # 空格/逗号千分位归一："2 900 MHz" → "2900 MHz"
    raw = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", raw)
    # 欧式小数逗号："1,5 µs" → "1.5 µs"（1-2 位小数才转，3 位视为千分位）
    raw = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", raw)
    # PRF 类字段：Hz 就是每秒脉冲数，不做频率换算
    if unit_hint == "pps":
        m = re.search(r"(\d[\d\.]*)\s*(?:Hz|pps)", raw, re.I)
        if m:
            return {"value": float(m.group(1)), "unit": "pps"}
    cands = []
    for m in _MEASURE_RX.finditer(raw):
        digits = m.group(1).replace(",", "").strip(".")
        if not re.search(r"\d", digits):          # 跳过没有数字的伪匹配
            continue
        num = float(digits)
        u = m.group(2).lower().replace("µ", "μ").replace("º", "°")
        u = u.replace(".", "") if u.startswith("r") else u   # r.p.m. → rpm
        if u in _TO_CANON:
            cu, f = _TO_CANON[u]
            cands.append({"value": round(num * f, 4), "unit": cu,
                          "metric": u in ("km", "kilometer", "kilometers", "m", "kg")})
    if not cands:
        return None
    best = cands[0]
    for c in reversed(cands):          # 括号内公制优先 = 最后出现的公制值
        if c["metric"]:
            best = c
            break
    best.pop("metric", None)
    return best
