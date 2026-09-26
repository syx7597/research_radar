"""
解析视觉模型转录的正文（data/v3/雷达手册.md + 雷达手册 2.md，共 310 条），
产出 v3 边 + 实体属性（tier=v3_manual 0.93）。

每条 `## N. 国家 型号 类型`：
  规格块键 → 边：体制→hasTechType / 频段→hasFrequencyBand / 装备机种→deployedOn
             / 配用武器→compatibleWith / 现状→status / 研制厂商→developedBy
             / 国家→countryOfOrigin
  性能表键 → 工作方式→hasMode（走 S6 词表归一）；其余数值 → 实体属性（带单位）
  研制时间/装备时间/价格 → 属性
  技术特点/分系统 叙述 → description 属性（首句摘要）+ 完整段留给向量库(单独导出)
  参考文献 → 丢弃

保守写法归一：附录A "见X" 别名 + 去 "Radar System" 后缀，合并确定同型号写法，
保留真变体((V)1/H/J)。清转录垃圾节点(含全角冒号/超长斜杠串)。

输出:  work/manual_body_edges.jsonl / work/manual_body_attrs.jsonl
       work/manual_body_chunks.jsonl（叙述段，供向量库）
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"
sys.path.insert(0, str(HERE))
import map_appendix as MA  # noqa: E402
import s6_canonicalize as S6  # noqa: E402  复用类别 tail 归一（体制/工作方式→词表）
S = MA.S

BODY = [ROOT / "data" / "v3" / "雷达手册.md", ROOT / "data" / "v3" / "雷达手册 2.md"]

# 规格块 / 性能表字段 → (关系 或 属性名, 类型/单位)
EDGE_FIELDS = {
    "体制": ("hasTechType", "TechType"),
    "频段": ("hasFrequencyBand", "FrequencyBand"),
    "装备机种": ("deployedOn", "Aircraft"),
    "配用武器": ("compatibleWith", "Weapon"),
    "工作方式": ("hasMode", "RadarMode"),
}
ATTR_FIELDS = {   # 键 → (属性名, 期望单位/None)
    "研制时间": ("dev_period", "year"), "装备时间": ("service_entry", "year"),
    "价格": ("price", None), "现状": ("status", None),
    "作用距离": ("range", "km"), "探测距离": ("range", "km"),
    "工作频率": ("frequency", "GHz"), "峰值功率": ("peak_power", "kW"),
    "平均功率": ("avg_power", "kW"), "发射机功率": ("tx_power", "kW"),
    "输入功率": ("input_power", None), "功耗": ("power_draw", None),
    "重复频率": ("prf", "pps"), "脉冲宽度": ("pulse_width", "us"),
    "天线增益": ("antenna_gain", None), "天线型式": ("antenna_type", None),
    "天线尺寸": ("antenna_size", "m"), "天线直径": ("antenna_size", "m"),
    "质量": ("weight", "kg"), "体积": ("volume", None), "尺寸": ("size", None),
    "扫描范围": ("scan_range", None), "扫描方式": ("scan_type", None),
    "扫描速率": ("scan_rate", "rpm"), "天线转速": ("scan_rate", "rpm"),
    "波束宽度": ("beamwidth", "deg"), "天线波束宽度": ("beamwidth", "deg"),
    "MTBF": ("mtbf", None), "MTTR": ("mttr", None), "分辨力": ("resolution", None),
    "冷却方式": ("cooling", None), "冷却": ("cooling", None),
    "跟踪目标": ("track_targets", None), "跟踪能力": ("track_targets", None),
    "多目标处理": ("track_targets", None), "处理能力": ("track_targets", None),
    "LRU": ("lru_count", None), "ECCM": ("eccm", None),
    "接收通道": ("rx_channels", None), "工作高度": ("altitude", "m"),
    "工作温度": ("work_temp", None), "精度": ("accuracy", None),
    "方位覆盖范围": ("azimuth_coverage", "deg"), "扫描扇区": ("scan_sector", "deg"),
}
NARRATIVE_SECTIONS = {"技术特点", "分系统", "雷达简况", "研制、装备情况", "研制装备情况"}
DROP_SECTIONS = {"参考文献"}


def norm(s):
    return re.sub(r"[\s\-_/\.]+", "", s.lower())


def is_garbage_name(n):
    return bool(re.search(r"[：]", n) or len(n) > 55 or n.count("/") >= 4)


def load_alias_map():
    """附录A '见X' → {异名 norm: 规范名}；反向：异名指向规范。"""
    p = HERE / "ab_aliases.json"
    amap = {}
    if p.exists():
        for canon, variants in json.loads(p.read_text(encoding="utf-8")).items():
            for v in variants:
                amap[norm(v)] = canon
    return amap


ALIAS = load_alias_map()


_TYPE_SUFFIX = re.compile(
    r"\s*[一-鿿]*(雷达|系统|天线|吊舱|设备|电子系统)(系列)?\s*$")


def strip_type(name):
    """剥尾部中文类型描述：'AN/APG-77/77(V)多功能火控雷达' → 'AN/APG-77/77(V)'。
    纯中文名(无西式型号)剥后为空则保留原名。"""
    s = _TYPE_SUFFIX.sub("", name).strip()
    return s if s and re.search(r"[A-Za-z0-9]", s) else name.strip()


def canon_radar_name(raw):
    """剥类型后缀 + 去 'Radar System' + 附录A别名；不碰真变体。"""
    n = strip_type(re.sub(r"\s*Radar System\s*$", "", raw, flags=re.I).strip())
    return ALIAS.get(norm(n), n)


def split_multi(cell):
    return [p.strip() for p in re.split(r"[、,，;；]", cell) if p.strip()]


def parse_entries(text):
    for m in re.finditer(r"^## \d+\.\s*(.+?)$(.*?)(?=^## \d+\.|\Z)", text, re.M | re.S):
        yield m.group(1).strip(), m.group(2)


def get_field(body, key):
    m = re.search(r"\*\*" + re.escape(key) + r"\*\*[：:]\s*\*{0,2}([^\n*]+)", body)
    return m.group(1).strip() if m else None


def main():
    idx, by_code = MA.load_radar_index()

    def resolve_one(model):
        """对齐 KG 已有节点：俄语转写→编号锚点→斜杠展开取首个命中；否则原名建节点。"""
        lat = S.cyrillic_to_latin(model)
        hit = idx.get(norm(lat)) or idx.get(norm(model))
        if hit:
            return hit
        code = S.extract_designation(lat)
        if code and code in by_code:
            return by_code[code]
        for v in MA.expand_slash(model):        # AN/APG-77/77(V) → 首个命中 AN/APG-77
            h = idx.get(norm(v))
            if h:
                return h
        return MA.expand_slash(model)[0]        # 未命中：取主型号建骨架，不带类型后缀串

    edges, attrs, chunks = [], [], []
    seen_radar = set()
    for f in BODY:
        text = f.read_text(encoding="utf-8")
        for title, body in parse_entries(text):
            # 型号名：优先 **型号名称** 字段，回退标题去国家前缀
            mn = get_field(body, "型号名称") or title
            mn = re.sub(r"^\S+?\s+", "", mn) if re.match(r"^[一-鿿]{2,4}\s", mn) else mn
            radar = canon_radar_name(mn.strip("* "))
            if is_garbage_name(radar) or len(radar) < 2:
                continue
            head = resolve_one(radar)
            prov = {"tier": "v3_manual", "source_kind": "manual",
                    "doc_id": f"manual_body:{radar}"}
            seen_radar.add(head)

            country = get_field(body, "国家") or get_field(body, "国别")
            if country:
                country = re.sub(r"[（(].*?[)）]", "", country).strip()
                edges.append({"head": head, "head_type": "Radar",
                              "relation": "countryOfOrigin", "tail": country,
                              "tail_type": "Country", "evidence": f"手册国家: {country}",
                              **prov})
            for key, (rel, ttype) in EDGE_FIELDS.items():
                val = get_field(body, key)
                if not val:
                    continue
                for item in split_multi(re.sub(r"[（(][^)）]*[)）]", "", val)):
                    if not (2 <= len(item) <= 45):
                        continue
                    if rel in S6.REL_VOCAB_KEY:          # 体制/工作方式 → 词表归一
                        hits = S6.canon_tail(rel, item)
                        for trel, canon in hits:
                            edges.append({"head": head, "head_type": "Radar",
                                          "relation": trel, "tail": canon,
                                          "tail_type": ttype,
                                          "evidence": f"手册{key}: {item[:40]}", **prov})
                        if hits:
                            continue                     # 归不进的落到下面原文入库
                    tt = S.classify_platform(item) if rel == "deployedOn" else ttype
                    edges.append({"head": head, "head_type": "Radar",
                                  "relation": rel, "tail": item, "tail_type": tt,
                                  "evidence": f"手册{key}: {val[:60]}", **prov})
            # 研制厂商
            mk = get_field(body, "研制厂商")
            if mk:
                for m in split_multi(re.sub(r"[（(][^)）]*原研制.*?[)）]", "", mk)):
                    m = re.sub(r"[（(][^)）]*[)）]", "", m).strip()
                    if 2 <= len(m) <= 60:
                        edges.append({"head": head, "head_type": "Radar",
                                      "relation": "developedBy", "tail": m,
                                      "tail_type": "Manufacturer",
                                      "evidence": f"手册研制厂商: {m}", **prov})
            # 数值/文本属性
            for key, (aname, unit) in ATTR_FIELDS.items():
                val = get_field(body, key)
                if not val:
                    continue
                rec = {"entity": head, "attr": aname, "value_raw": val[:200],
                       "evidence": f"手册 {key}: {val[:80]}", "tier": "v3_manual",
                       "doc_id": f"manual_body:{radar}", "unmapped": False}
                if unit == "year":
                    ym = re.search(r"\d{4}", val)
                    if ym:
                        rec["value"] = int(ym.group()); rec["unit"] = "year"
                elif unit:
                    pm = S.parse_measure(val, unit)
                    if pm:
                        rec["value"] = pm["value"]; rec["unit"] = pm["unit"]
                attrs.append(rec)
            # 叙述段 → description 属性(首句) + chunk(向量库)
            for sm in re.finditer(r"^### (.+?)$(.*?)(?=^### |\Z)", body, re.M | re.S):
                sec, txt = sm.group(1).strip(), sm.group(2).strip()
                if sec in DROP_SECTIONS or not txt:
                    continue
                if sec in NARRATIVE_SECTIONS:
                    chunks.append({"radar": head, "section": sec, "text": txt,
                                   "doc_id": f"manual_body:{radar}#{sec}"})
                    if sec in ("技术特点", "雷达简况"):
                        first = re.split(r"[。\n]", txt)[0][:120]
                        if first:
                            attrs.append({"entity": head, "attr": "description",
                                          "value_raw": first, "evidence": f"手册{sec}",
                                          "tier": "v3_manual",
                                          "doc_id": f"manual_body:{radar}", "unmapped": True})

    (WORK / "manual_body_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    (WORK / "manual_body_attrs.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in attrs), encoding="utf-8")
    (WORK / "manual_body_chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    from collections import Counter
    print(f"[body] radars {len(seen_radar)} | edges {len(edges)} "
          f"{dict(Counter(e['relation'] for e in edges))}")
    print(f"[body] attrs {len(attrs)} (numeric {sum(1 for a in attrs if 'value' in a)}) "
          f"| narrative chunks {len(chunks)}")


if __name__ == "__main__":
    main()
