"""
解析《世界海用雷达手册》附录（data/v3/naval/*.md）→ v3 边(tier=v3_manual 0.93)。

附录F 平台索引 (国别|舰级|舰种|型号|功能|波段):
  型号 → deployedOn 舰级(NavalVessel) + hasFunction + hasFrequencyBand + countryOfOrigin
  舰种(泛化:潜艇/驱逐舰) → ship_type 属性(不做泛 deployedOn)
附录E 功能索引 (## 功能节; 国家|型号|页码): 型号 → hasFunction(节名归一) + countryOfOrigin
附录D 型号索引 (型号|页码): 型号"(见X)" → 别名(NATO↔本国)
附录C 缩略语 (英↔中): 术语参考(暂作别名字典,不直接建边)

复用 map_appendix 对齐(俄语转写/编号/斜杠/骨架)。见X 归并同机载。

输出: work/naval_appendix_edges.jsonl + work/naval_appendix_attrs.jsonl
      manual/naval_aliases.json(追加 D 的见X)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"
SRC = ROOT / "data" / "v3" / "naval"
sys.path.insert(0, str(HERE))
import map_appendix as MA          # noqa: E402
import s6_canonicalize as S6       # noqa: E402
S = MA.S


def norm(s):
    return re.sub(r"[\s\-_/\.]+", "", s.lower())


def clean_model(m):
    m = re.sub(r"[\(（]\s*见[^\)）]*[\)）]", "", m)     # 去(见X)
    m = re.sub(r"\*\*", "", m).strip()
    return m


def see_ref(m):
    r = re.search(r"[\(（]\s*见\s*([^\)）]{2,30})", m)
    return r.group(1).strip() if r else None


def edge(head, rel, tail, ttype, ev, tag):
    return {"head": head, "head_type": "Radar", "relation": rel, "tail": tail,
            "tail_type": ttype, "evidence": ev[:200], "tier": "v3_manual",
            "source_kind": "manual", "doc_id": f"naval_{tag}"}


def rows(text):
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("|") and s.endswith("|") and set(s) - set("|:- "):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not any(c in ("型号", "国别", "页码", "功能", "波段") for c in cells):
                yield cells


def hasfunc_edges(head, raw, tag, out):
    for item in re.split(r"[、,，;；]|和|与", re.sub(r"[（(][^)）]*[)）]", "", raw)):
        item = item.strip()
        if not (1 <= len(item) <= 30):
            continue
        hits = S6.canon_tail("hasFunction", item)
        if hits:
            for trel, canon in hits:
                out.append(edge(head, trel, canon, "Function", f"附录{tag}功能:{item}", tag))
        elif item not in ("多功能",):
            out.append(edge(head, "hasFunction", item, "Function", f"附录{tag}功能:{item}", tag))


def main():
    idx, by_code = MA.load_radar_index()

    def resolve(model):
        variants = MA.expand_slash(model)
        if len(variants) > 1:
            rs = [MA_resolve_one(idx, by_code, v) for v in variants]
            if any(not isn for _, isn in rs):
                return [(n, isn) for n, isn in rs if n]
        r = MA_resolve_one(idx, by_code, model)
        return [r] if r[0] else []

    def MA_resolve_one(idx, by_code, model):
        lat = S.cyrillic_to_latin(model)
        hit = idx.get(norm(lat)) or idx.get(norm(model))
        if hit:
            return hit, False
        code = S.extract_designation(lat)
        if code and code in by_code:
            return by_code[code], False
        c = MA.clean_model(lat)
        return (c, True) if c else (None, False)

    edges, attrs, seealso = [], [], {}
    text_all = "\n".join((SRC / n).read_text(encoding="utf-8")
                         for n in ("附录ab.md", "附录cde.md", "附录fgh.md")
                         if (SRC / n).exists())

    # 用大标题切分附录段
    segs = re.split(r"(?m)^#{0,3}\s*附录\s*([A-H])\b", text_all)
    sect = {}
    for i in range(1, len(segs), 2):
        sect[segs[i]] = segs[i + 1] if i + 1 < len(segs) else ""

    # 附录F 平台索引
    for cells in rows(sect.get("F", "")):
        if len(cells) < 6:
            continue
        country, shipclass, shiptype, model, func, band = cells[:6]
        if not model or model in ("型号",):
            continue
        for head, _ in resolve(clean_model(model)):
            if country and re.search(r"[一-鿿]", country):
                edges.append(edge(head, "countryOfOrigin", re.sub(r"\*\*", "", country),
                                  "Country", f"附录F国别:{country}", "F"))
            if shipclass and 2 <= len(shipclass) <= 40:
                edges.append(edge(head, "deployedOn", shipclass, "NavalVessel",
                                  f"附录F舰级:{shipclass}", "F"))
            if shiptype:
                attrs.append({"entity": head, "attr": "ship_type", "value_raw": shiptype,
                              "evidence": f"附录F舰种:{shiptype}", "tier": "v3_manual",
                              "doc_id": "naval_F", "unmapped": False})
            if func:
                hasfunc_edges(head, func, "F", edges)
            for b in re.split(r"[、,，/]", band):
                b = S.resolve_alias(b.strip(), "FrequencyBand")
                if b and len(b) <= 8:
                    edges.append(edge(head, "hasFrequencyBand", b, "FrequencyBand",
                                      f"附录F波段:{band}", "F"))

    # 附录E 功能索引：## 功能节 + 国家|型号|页码
    ecur_func = None
    for ln in sect.get("E", "").splitlines():
        hm = re.match(r"^##\s+(.+)$", ln.strip())
        if hm:
            ecur_func = hm.group(1).strip()
            continue
        s = ln.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        # 可能两组(国家|型号|页码)×2
        for g in range(0, len(cells) - 1, 3):
            country, model = cells[g], cells[g + 1] if g + 1 < len(cells) else ""
            model = clean_model(model)
            if not model or model in ("型号",) or len(model) < 2:
                continue
            if re.fullmatch(r"\d{1,4}", model):     # 页码列错位被当型号，跳过
                continue
            sr = see_ref(cells[g + 1])
            if sr:
                seealso.setdefault(sr, []).append(model)
            for head, _ in resolve(model):
                if ecur_func:
                    hasfunc_edges(head, ecur_func, "E", edges)
                cc = re.sub(r"\*\*", "", country).strip()
                if cc and re.search(r"[一-鿿]", cc) and 2 <= len(cc) <= 10:
                    edges.append(edge(head, "countryOfOrigin", cc, "Country",
                                      f"附录E国别:{cc}", "E"))

    # 附录D 型号索引：型号|页码 (两组) — 只取见X别名
    for cells in rows(sect.get("D", "")):
        for c in cells:
            sr = see_ref(c)
            if sr:
                seealso.setdefault(sr, []).append(clean_model(c))

    (WORK / "naval_appendix_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    (WORK / "naval_appendix_attrs.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in attrs), encoding="utf-8")
    # 合并 D/E 的见X 到 naval_aliases
    ap = HERE / "naval_aliases.json"
    existing = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else {}
    for k, v in seealso.items():
        existing.setdefault(k, [])
        existing[k] = sorted(set(existing[k]) | set(v))
    ap.write_text(json.dumps(existing, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"[naval-appx] 边 {len(edges)} {dict(Counter(e['relation'] for e in edges))}")
    print(f"[naval-appx] 属性 {len(attrs)} | 见X别名合并后 {len(existing)}")


if __name__ == "__main__":
    main()
