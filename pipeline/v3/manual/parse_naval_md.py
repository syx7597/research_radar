"""
解析《世界海用雷达手册》正文转录（data/v3/naval_manual/p*.md，204 窗口），
产出 v3 边 + 属性(tier=v3_manual 0.93)。复用 parse_body_md / map_appendix / S6。

窗口 md 每条目 `### 型号`：
  功能→hasFunction(复合按"和/与"拆+S6词表归一) / 体制→hasTechType / 频段→
  hasFrequencyBand / 研制厂商→developedBy / 国家(窗口分组标题)→countryOfOrigin
  装备舰型(泛化舰种)→ship_types 属性(不做泛 deployedOn) / 精确舰级由附录F给
  装备时间/现状/性能数值 → 属性；技术特点/装备情况 → description + 向量块
  型号"(见X)" → X 是别名(NATO代号↔本国型号)；无内容的纯指向只记别名
doc_id = naval_p{窗口起始页} → S5 按页分三段 domain(舰载/岸基/机载)。

输出: work/naval_body_edges.jsonl / naval_body_attrs.jsonl / naval_body_chunks.jsonl
      manual/naval_aliases.json（见X 别名）
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"
SRC = ROOT / "data" / "v3" / "naval_manual"
sys.path.insert(0, str(HERE))
import map_appendix as MA          # noqa: E402
import s6_canonicalize as S6       # noqa: E402
S = MA.S

EDGE_FIELDS = {
    "体制": ("hasTechType", "TechType"),
    "频段": ("hasFrequencyBand", "FrequencyBand"),
    "功能": ("hasFunction", "Function"),
}
ATTR_FIELDS = {
    "装备时间": ("service_entry", "year"), "研制时间": ("dev_period", "year"),
    "现状": ("status", None), "装备舰型": ("ship_types", None),
    "作用距离": ("range", "km"), "探测距离": ("range", "km"),
    "工作频率": ("frequency", "GHz"), "峰值功率": ("peak_power", "kW"),
    "平均功率": ("avg_power", "kW"), "发射机功率": ("tx_power", "kW"),
    "脉冲宽度": ("pulse_width", "us"), "重复频率": ("prf", "pps"),
    "天线增益": ("antenna_gain", None), "天线型式": ("antenna_type", None),
    "天线尺寸": ("antenna_size", "m"), "质量": ("weight", "kg"),
    "扫描范围": ("scan_range", None), "波束宽度": ("beamwidth", "deg"),
    "MTBF": ("mtbf", None), "跟踪目标": ("track_targets", None),
    "跟踪能力": ("track_targets", None), "作用高度": ("altitude", "m"),
    "分辨力": ("resolution", None), "冷却方式": ("cooling", None),
    "方位覆盖范围": ("azimuth_coverage", "deg"), "扫描速率": ("scan_rate", "rpm"),
}
NARRATIVE = {"技术特点", "装备情况", "研制情况", "研制、装备情况", "雷达简况"}


def norm(s):
    return re.sub(r"[\s\-_/\.]+", "", s.lower())


_FIELD_WORDS = (r"功能|体制|频段|研制厂商|研制|装备|现状|作用距离|探测距离|工作频率|"
                r"工作方式|天线|发射机|接收机|波束|扫描|重复频率|脉冲|处理机|冷却|质量|"
                r"尺寸|体积|天馈线|覆盖范围|方位|分辨力|MTBF|峰值功率|平均功率|数据|电源|"
                r"显示|噪声|增益|接口|输入功率|技术特点|性能")


def is_garbage(n):
    if re.fullmatch(r"[\d.,]+\s*(km|m|kg|GHz|MHz|kW|MW|°|dB|n\s*mile|海里)?", n, re.I):
        return True                       # 纯数值(误把性能数字当型号标题)
    if re.match(rf"^({_FIELD_WORDS})[\s：:]", n):
        return True                       # 规格块字段行被误当 ### 型号标题
    if not re.search(r"[A-Za-z0-9一-鿿]", n) or len(n) < 2:
        return True
    return bool(re.search(r"[：]", n)) or len(n) > 55 or n.count("/") >= 5


def load_alias():
    p = HERE / "ab_aliases.json"          # 复用机载附录A "见X"
    amap = {}
    if p.exists():
        for canon, variants in json.loads(p.read_text(encoding="utf-8")).items():
            for v in variants:
                amap[norm(v)] = canon
    return amap


ALIAS = load_alias()


def get_field(body, key):
    m = re.search(r"(?:^|\n)[\-*\s]*\*\*\s*" + re.escape(key)
                  + r"\s*\*\*[：:]\s*([^\n*]+)", body)
    return m.group(1).strip() if m else None


def split_multi(cell, funcs=False):
    pat = r"[、,，;；]" + (r"|和|与" if funcs else "")
    return [p.strip() for p in re.split(pat, cell) if p.strip()]


def resolve_one(idx, by_code, model):
    lat = S.cyrillic_to_latin(model)
    hit = idx.get(norm(lat)) or idx.get(norm(model))
    if hit:
        return hit
    code = S.extract_designation(lat)
    if code and code in by_code:
        return by_code[code]
    for v in MA.expand_slash(model):
        if idx.get(norm(v)):
            return idx.get(norm(v))
    return ALIAS.get(norm(model), model)


def main():
    idx, by_code = MA.load_radar_index()
    edges, attrs, chunks, seealso = [], [], [], {}
    files = sorted(SRC.glob("p*.md"))
    for f in files:
        m = re.match(r"p(\d+)", f.name)
        page = int(m.group(1)) if m else 0
        text = f.read_text(encoding="utf-8")
        country = None
        # 逐块：## 国家标题 更新 country；### 型号 为条目
        parts = re.split(r"(^#{2,3}\s+.+$)", text, flags=re.M)
        cur_title, cur_body = None, ""
        blocks = []
        for seg in parts:
            h = re.match(r"^(#{2,3})\s+(.+)$", seg.strip())
            if h:
                if cur_title is not None:
                    blocks.append((cur_title, cur_body))
                cur_title, cur_body = (h.group(1), h.group(2).strip()), ""
            else:
                cur_body += seg
        if cur_title is not None:
            blocks.append((cur_title, cur_body))

        for (level, title), body in blocks:
            # ## 级别是国家分组 / ### 级别是雷达条目
            if level == "##":
                c = re.sub(r"[\(（].*?[\)）]", "", re.sub(r"^\d+\.?\s*", "", title)).strip()
                if 2 <= len(c) <= 12 and re.search(r"[一-鿿]", c):
                    country = c
                continue
            raw_model = title
            see = re.search(r"[\(（]\s*见\s*([^\)）]{2,30})", raw_model)
            model = re.sub(r"[\(（][^\)）]*[\)）]", "", raw_model).strip()
            model = re.sub(r"[（(].*$", "", model).strip()
            if is_garbage(model) or len(model) < 2:
                continue
            has_content = bool(re.search(r"\*\*", body))
            if see:
                seealso.setdefault(see.group(1).strip(), []).append(model)
                if not has_content:
                    continue          # 纯指向，只记别名
            head = resolve_one(idx, by_code, ALIAS.get(norm(model), model))
            prov = {"tier": "v3_manual", "source_kind": "manual",
                    "doc_id": f"naval_p{page:04d}"}
            if country:
                edges.append({"head": head, "head_type": "Radar",
                              "relation": "countryOfOrigin", "tail": country,
                              "tail_type": "Country", "evidence": f"海用手册国别: {country}",
                              **prov})
            for key, (rel, ttype) in EDGE_FIELDS.items():
                val = get_field(body, key)
                if not val:
                    continue
                for item in split_multi(re.sub(r"[（(][^)）]*[)）]", "", val),
                                        funcs=(key == "功能")):
                    if not (1 <= len(item) <= 30):
                        continue
                    if rel in S6.REL_VOCAB_KEY:
                        for trel, canon in S6.canon_tail(rel, item):
                            edges.append({"head": head, "head_type": "Radar",
                                          "relation": trel, "tail": canon,
                                          "tail_type": ttype,
                                          "evidence": f"海用{key}: {item[:30]}", **prov})
                        if S6.canon_tail(rel, item):
                            continue
                    edges.append({"head": head, "head_type": "Radar", "relation": rel,
                                  "tail": item, "tail_type": ttype,
                                  "evidence": f"海用{key}: {item[:40]}", **prov})
            mk = get_field(body, "研制厂商")
            if mk:
                for m2 in split_multi(re.sub(r"[（(][^)）]*原研制.*?[)）]", "", mk)):
                    m2 = re.sub(r"[（(][^)）]*[)）]", "", m2).strip()
                    if 2 <= len(m2) <= 60:
                        edges.append({"head": head, "head_type": "Radar",
                                      "relation": "developedBy", "tail": m2,
                                      "tail_type": "Manufacturer",
                                      "evidence": f"海用研制厂商: {m2}", **prov})
            for key, (aname, unit) in ATTR_FIELDS.items():
                val = get_field(body, key)
                if not val:
                    continue
                rec = {"entity": head, "attr": aname, "value_raw": val[:200],
                       "evidence": f"海用 {key}: {val[:80]}", "tier": "v3_manual",
                       "doc_id": f"naval_p{page:04d}", "unmapped": False}
                if unit == "year":
                    ym = re.search(r"\d{4}", val)
                    if ym:
                        rec["value"] = int(ym.group()); rec["unit"] = "year"
                elif unit:
                    pm = S.parse_measure(val, unit)
                    if pm:
                        rec["value"] = pm["value"]; rec["unit"] = pm["unit"]
                attrs.append(rec)
            # 技术特点/装备情况 → chunk + description
            for sec in NARRATIVE:
                m3 = re.search(r"\*\*\s*" + sec + r"\s*\*\*(.*?)(?=\*\*[一-鿿]{2,6}\*\*|\Z)",
                               body, re.S)
                if not m3:
                    continue
                txt = re.sub(r"^[:：\s]+", "", m3.group(1)).strip()
                if len(txt) < 20:
                    continue
                chunks.append({"radar": head, "section": sec, "text": txt[:2000],
                               "doc_id": f"naval_p{page:04d}#{sec}"})
                if sec == "技术特点":
                    first = re.split(r"[。\n]", txt)[0][:120]
                    if first:
                        attrs.append({"entity": head, "attr": "description",
                                      "value_raw": first, "evidence": "海用技术特点",
                                      "tier": "v3_manual", "doc_id": f"naval_p{page:04d}",
                                      "unmapped": True})

    (WORK / "naval_body_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    (WORK / "naval_body_attrs.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in attrs), encoding="utf-8")
    (WORK / "naval_body_chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    (HERE / "naval_aliases.json").write_text(
        json.dumps(seealso, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    heads = set(e["head"] for e in edges)
    print(f"[naval-body] 雷达 {len(heads)} | 边 {len(edges)} "
          f"{dict(Counter(e['relation'] for e in edges))}")
    print(f"[naval-body] 属性 {len(attrs)}(数值 {sum(1 for a in attrs if 'value' in a)}) "
          f"| 叙述块 {len(chunks)} | 见X别名 {len(seealso)}")


if __name__ == "__main__":
    main()
