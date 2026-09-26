"""
S2 确定性抽取层（零 LLM）：struct 块（wiki infobox + RT 规格表）→ 边 + 属性。

原则：
- 属性与关系在源头分流：数值/描述 → 属性（挂实体），实体值 → 边
- 每条产出 evidence = 原始 "key: value" 字符串，全程可溯源
- head = 页面主型号（infobox 属于页面主体，这里安全；叙述抽取不允许这样做）
- 未映射键保留为 unmapped 属性（可审计，不进检索渲染）

输入:  work/chunks.jsonl (kind=struct)
输出:  work/s2_edges.jsonl / work/s2_attrs.jsonl / work/s2_report.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))

import schema as S  # noqa: E402

WORK = ROOT / "pipeline" / "v3" / "work"

_BAND_RX = re.compile(
    r"\b(VHF|UHF|HF|Ku|Ka|mm|[LSCXKWEFGIJ](?:/[LSCXKWEFGIJ])?)\s*[- ]\s*band\b", re.I)
_YEAR_RX = re.compile(r"\b(19[3-9]\d|20[0-4]\d)\b")

# infobox/RT 规格键 → 处理器。键先做 normalize（小写、去括号注释、去空白差异）
_EDGE_KEYS = {
    "country of origin": "countryOfOrigin",
    "country":           "countryOfOrigin",
    "manufacturer":      "developedBy",
    "designer":          "developedBy",
    "developed by":      "developedBy",
    "builder":           "developedBy",
}
_ATTR_KEYS = {  # key → (属性名, 期望单位 or None)
    "introduced": ("service_entry", "year"), "in service": ("service_entry", "year"),
    "first built": ("service_entry", "year"),
    "range": ("range", "km"), "instrumented range": ("range", "km"),
    "maximum range": ("range", "km"), "detection range": ("range", "km"),
    "prf": ("prf", "pps"), "pulse repetition frequency": ("prf", "pps"),
    "pulse repetition time": ("prt", "us"),
    "pulsewidth": ("pulse_width", "us"), "pulse width": ("pulse_width", "us"),
    "peak power": ("peak_power", "kW"), "average power": ("avg_power", "kW"),
    "power": ("power", "kW"),
    "beamwidth": ("beamwidth", "deg"),
    "rpm": ("rotation", "rpm"), "antenna rotation": ("rotation", "rpm"),
    "accuracy": ("accuracy", None), "precision": ("accuracy", None),
    "altitude": ("altitude", "m"), "azimuth": ("azimuth", None),
    "elevation": ("elevation", None), "diameter": ("diameter", "m"),
    "height": ("height", "m"), "width": ("width", "m"), "weight": ("weight", "kg"),
    "no. built": ("units_built", None), "number built": ("units_built", None),
    "mtbcf": ("mtbf", None), "mtbf": ("mtbf", None),
    "range resolution": ("range_resolution", None),
    "hits per scan": ("hits_per_scan", None),
    "frequency": ("frequency", "GHz"),   # 数值部分；频段另出边
}
_RELATED_RX = re.compile(
    r"([A-Za-z0-9/\-\. ]{2,40}?)\s*\((?:the\s+)?(predecessor|successor)\)?|"
    r"([A-Za-z0-9/\-\. ]{2,40}?)\s+(predecessor|successor)\b", re.I)


def norm_key(k: str) -> str:
    k = re.sub(r"[\(\（][^)）]*[\)）]", "", k)          # 去括号注释 (τ) 等
    return re.sub(r"\s+", " ", k).strip().lower().rstrip(":")


def band_edges(head, raw, prov):
    out = []
    seen = set()
    for m in _BAND_RX.finditer(raw):
        band = S.resolve_alias(m.group(1).upper() if len(m.group(1)) <= 3
                               else m.group(1), "FrequencyBand")
        if band not in seen:
            seen.add(band)
            out.append(edge(head, "hasFrequencyBand", band, "FrequencyBand", raw, prov))
    return out


def edge(head, rel, tail, tail_type, raw, prov):
    return {"head": head, "head_type": "Radar", "relation": rel,
            "tail": tail, "tail_type": tail_type,
            "evidence": raw[:200], **prov, "tier": "v3_struct"}


def attr(head, name, raw, unit_hint, prov, unmapped=False):
    rec = {"entity": head, "attr": name, "value_raw": raw[:300],
           "evidence": raw[:200], **prov, "tier": "v3_struct",
           "unmapped": unmapped}
    if unit_hint == "year":
        m = _YEAR_RX.search(raw)
        if m:
            rec["value"] = int(m.group(1)); rec["unit"] = "year"
    elif unit_hint:
        pm = S.parse_measure(raw, unit_hint)
        if pm:
            rec["value"] = pm["value"]; rec["unit"] = pm["unit"]
    return rec


def process_struct(c: dict):
    head, prov = c["radar_hint"], {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                                   "source_kind": c["source_kind"]}
    edges, attrs, aliases = [], [], []
    for k_raw, v_raw in (c["struct"] or {}).items():
        k, v = norm_key(k_raw), str(v_raw).strip()
        if not v:
            continue
        raw = f"{k_raw}: {v}"

        if k in _EDGE_KEYS:
            rel = _EDGE_KEYS[k]
            for item in re.split(r"[;；]", v):
                item = re.sub(r"[\(\（][^)）]*[\)）]", "", item).strip(" .,")
                if not (2 <= len(item) <= 60):
                    continue
                tail = S.resolve_alias(item, "Country") if rel == "countryOfOrigin" else item
                ttype = "Country" if rel == "countryOfOrigin" else "Manufacturer"
                edges.append(edge(head, rel, tail, ttype, raw, prov))

        elif k == "type":
            for item in re.split(r"[;；,，]", v):
                item = item.strip()
                canon = S.resolve_alias(item, "TechType")
                if canon != item or item in ("AESA", "PESA"):
                    edges.append(edge(head, "hasTechType", canon, "TechType", raw, prov))
                elif item:
                    attrs.append(attr(head, "type_description", item, None, prov,
                                      unmapped=True))

        elif k == "related":
            for m in _RELATED_RX.finditer(v):
                name = (m.group(1) or m.group(3) or "").strip(" .,;")
                role = (m.group(2) or m.group(4) or "").lower()
                if not (2 <= len(name) <= 40):
                    continue
                rel = "replaces" if role == "predecessor" else "replacedBy"
                edges.append(edge(head, rel, name, "RadarSystem", raw, prov))

        elif k == "other names":
            aliases.extend(a.strip() for a in re.split(r"[;；,，]", v) if a.strip())

        elif k in _ATTR_KEYS:
            name, unit = _ATTR_KEYS[k]
            attrs.append(attr(head, name, v, unit, prov))
            if k == "frequency":
                edges.extend(band_edges(head, v, prov))

        else:
            attrs.append(attr(head, re.sub(r"[^\w]+", "_", k)[:40], v, None, prov,
                              unmapped=True))
    return edges, attrs, aliases


def main():
    edges_f = (WORK / "s2_edges.jsonl").open("w", encoding="utf-8")
    attrs_f = (WORK / "s2_attrs.jsonl").open("w", encoding="utf-8")
    alias_map, stats = {}, Counter()
    for line in (WORK / "chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["kind"] != "struct":
            continue
        edges, attrs, aliases = process_struct(c)
        for e in edges:
            edges_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            stats[f"edge:{e['relation']}"] += 1
        for a in attrs:
            attrs_f.write(json.dumps(a, ensure_ascii=False) + "\n")
            stats["attr:unmapped" if a["unmapped"] else f"attr:{a['attr']}"] += 1
        if aliases:
            alias_map.setdefault(c["radar_hint"], []).extend(aliases)
    edges_f.close(); attrs_f.close()
    (WORK / "s2_aliases.json").write_text(
        json.dumps(alias_map, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORK / "s2_report.json").write_text(
        json.dumps(dict(stats.most_common()), ensure_ascii=False, indent=1),
        encoding="utf-8")
    n_e = sum(v for k, v in stats.items() if k.startswith("edge:"))
    n_a = sum(v for k, v in stats.items() if k.startswith("attr:"))
    print(f"[S2] edges {n_e} + attrs {n_a} (unmapped {stats['attr:unmapped']}) "
          f"+ alias entities {len(alias_map)}")
    for k, v in stats.most_common(12):
        print(f"   {k:28} {v}")


if __name__ == "__main__":
    main()
