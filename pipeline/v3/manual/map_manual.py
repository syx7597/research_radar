"""
把手册视觉识别的结构化条目（manual/extract.jsonl）映射到 v3 schema，
产出高置信手册边/属性（tier=v3_manual，金标 93%），供 S5 融合。

结构化条目字段 → KG：
  tech_types    → hasTechType（体制）
  freq_bands    → hasFrequencyBand（频段）
  developer     → developedBy
  country       → countryOfOrigin（目录国别分组，权威）
  variants[].name      → hasVariant
  variants[].platforms → deployedOn（变体挂到主型号，简化：也给主型号）
  weapons       → compatibleWith（配用武器）
  performance   → 数值属性（frequency/range/scan）
  price/status/dev_period/service_entry → 实体属性

输出:  work/manual_edges.jsonl / work/manual_attrs.jsonl
运行:  python pipeline/v3/manual/map_manual.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"

TIER = "v3_manual"


def edge(head, rel, tail, ttype, page, ev):
    return {"head": head, "head_type": "Radar", "relation": rel, "tail": tail,
            "tail_type": ttype, "evidence": ev, "tier": TIER,
            "source_kind": "manual", "doc_id": f"manual_p{page}"}


def attr(ent, name, raw, page, value=None, unit=None):
    r = {"entity": ent, "attr": name, "value_raw": str(raw),
         "evidence": f"手册 p{page}: {name}={raw}", "doc_id": f"manual_p{page}",
         "tier": TIER, "unmapped": False}
    if value is not None:
        r["value"] = value
    if unit:
        r["unit"] = unit
    return r


def num_range_mid(s):
    nums = re.findall(r"\d+(?:\.\d+)?", str(s))
    if not nums:
        return None
    vals = [float(n) for n in nums]
    return round(sum(vals) / len(vals), 2)


def main():
    edges, attrs = [], []
    for line in (HERE / "extract.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        r, pg = d["radar"], d["printed_page"]
        for t in d.get("tech_types", []):
            edges.append(edge(r, "hasTechType", t, "TechType", pg, f"体制: {t}"))
        for b in d.get("freq_bands", []):
            edges.append(edge(r, "hasFrequencyBand", b, "FrequencyBand", pg, f"频段: {b}"))
        if d.get("developer"):
            edges.append(edge(r, "developedBy", d["developer"], "Manufacturer", pg,
                              "研制厂商: " + d["developer"]))
        if d.get("country"):
            edges.append(edge(r, "countryOfOrigin", d["country"], "Country", pg,
                              "目录国别: " + d["country"]))
        for v in d.get("variants", []):
            if v["name"] != r:
                edges.append(edge(r, "hasVariant", v["name"], "Radar", pg,
                                  "装备机种: " + v["name"]))
            for plat in v.get("platforms", []):
                edges.append(edge(v["name"], "deployedOn", plat, "Aircraft", pg,
                                  f"{v['name']} 装备 {plat}"))
        for w in d.get("weapons", []):
            edges.append(edge(r, "compatibleWith", w, "Weapon", pg, "配用武器: " + w))
        # 属性
        if d.get("price"):
            attrs.append(attr(r, "price", d["price"], pg))
        if d.get("status"):
            attrs.append(attr(r, "status", d["status"], pg))
        if d.get("service_entry"):
            attrs.append(attr(r, "service_entry", d["service_entry"], pg,
                              value=int(re.search(r"\d{4}", d["service_entry"]).group()),
                              unit="year"))
        if d.get("dev_period"):
            attrs.append(attr(r, "dev_period", d["dev_period"], pg))
        perf = d.get("performance", {})
        if perf.get("work_freq_ghz"):
            attrs.append(attr(r, "frequency", perf["work_freq_ghz"] + " GHz", pg,
                              value=num_range_mid(perf["work_freq_ghz"]), unit="GHz"))
        for k, v in (perf.get("range_km") or {}).items():
            attrs.append(attr(r, f"range[{k}]", v + " km", pg,
                              value=num_range_mid(v), unit="km"))
        if perf.get("scan_deg"):
            attrs.append(attr(r, "scan", perf["scan_deg"], pg))

    (WORK / "manual_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    (WORK / "manual_attrs.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in attrs), encoding="utf-8")
    print(f"[manual] {len(edges)} edges + {len(attrs)} attrs "
          f"from {sum(1 for _ in (HERE/'extract.jsonl').open(encoding='utf-8'))} radars")


if __name__ == "__main__":
    main()
