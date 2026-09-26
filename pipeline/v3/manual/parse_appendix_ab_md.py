"""
解析视觉模型转录的 data/v3/附录ab.md（附录A 型号索引 5 列 + 附录B 载机对照 2 列），
产出 v3 边（tier=v3_manual 0.93），并入 S5。复用 map_appendix 的对齐（俄语转写/
编号锚点/斜杠展开/骨架节点）。

附录A 每行: 型号 | 国别 | 研制生产机构 | 载机 | 备注(见X=别名)
  → countryOfOrigin, developedBy(拆多机构; "原研制厂商 X"→derivedFrom-历史厂商略),
    deployedOn(载机列), 别名(备注"见X" → 型号异名，写 alias 供归并)
附录B 每行: 载机 | 雷达列表  → 每个雷达 deployedOn 载机

输出:  work/manual_appendix_ab_edges.jsonl  +  manual/ab_aliases.json
运行:  python pipeline/v3/manual/parse_appendix_ab_md.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"
MD = ROOT / "data" / "v3" / "附录ab.md"
sys.path.insert(0, str(HERE))
import map_appendix as MA  # noqa: E402  复用 resolve/expand_slash/schema

S = MA.S


def split_multi(cell: str) -> list[str]:
    """按 、,，; 拆多值，去空白。"""
    parts = re.split(r"[、,，;；]", cell)
    return [p.strip() for p in parts if p.strip()]


def clean_maker(m: str) -> str:
    m = re.sub(r"[（(]\s*原研制厂商.*?[)）]", "", m)      # 去"(原研制厂商 …)"
    m = re.sub(r"[（(][^)）]*[)）]", "", m)               # 去其它括号注释(国别/简称)
    return re.sub(r"\s+", " ", m).strip(" ,，")


def clean_platform(p: str) -> str:
    p = re.sub(r"[（(][^)）]*[)）]", "", p)               # 去 (希腊)/(RC) 注释
    return re.sub(r"\s+", " ", p).strip()


def edge(head, rel, tail, ttype, page, ev):
    return {"head": head, "head_type": "Radar", "relation": rel, "tail": tail,
            "tail_type": ttype, "evidence": ev[:200], "tier": "v3_manual",
            "source_kind": "manual", "doc_id": f"manual_appendixA_p{page}"}


def parse_tables(text):
    """产出 (section, page, [cells...]) 数据行。"""
    section, page = None, 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# 附录 A"):
            section = "A"; continue
        if s.startswith("# 附录 B"):
            section = "B"; continue
        m = re.match(r"^##\s*页码\s*(\d+)", s)
        if m:
            page = int(m.group(1)); continue
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells or set("".join(cells)) <= set(": -"):   # 分隔行/表头
            continue
        if cells[0] in ("型号", "载机型号或名称"):
            continue
        yield section, page, cells


def main():
    idx, by_code = MA.load_radar_index()

    # 复刻 map_appendix.main 里的 resolve 闭包（俄语转写/编号锚点/斜杠展开/骨架）
    def resolve_one(model):
        lat = S.cyrillic_to_latin(model)
        hit = idx.get(MA.norm(lat)) or idx.get(MA.norm(model))
        if hit:
            return hit, False
        code = S.extract_designation(lat)
        if code and code in by_code:
            return by_code[code], False
        c = MA.clean_model(lat)
        return (c, True) if c else (None, False)

    def resolve_all(model):
        variants = MA.expand_slash(model)
        if len(variants) > 1:
            rs = [resolve_one(v) for v in variants]
            if any(not isn for _, isn in rs):
                return [(n, isn) for n, isn in rs if n]
        r = resolve_one(model)
        return [r] if r[0] else []

    edges, aliases = [], {}
    for section, page, cells in parse_tables(MD.read_text(encoding="utf-8")):
        if section == "A" and len(cells) >= 4:
            model, country, maker, carriers = cells[0], cells[1], cells[2], cells[3]
            note = cells[4] if len(cells) > 4 else ""
            for head, _ in resolve_all(model):
                if country:
                    edges.append(edge(head, "countryOfOrigin", country, "Country",
                                      page, f"附录A 国别: {country}"))
                for mk in split_multi(clean_maker(maker)):
                    if 2 <= len(mk) <= 60:
                        edges.append(edge(head, "developedBy", mk, "Manufacturer",
                                          page, f"附录A 研制机构: {mk}"))
                for pf in split_multi(carriers):
                    pf = clean_platform(pf)
                    if 2 <= len(pf) <= 40:
                        edges.append(edge(head, "deployedOn", pf,
                                          S.classify_platform(pf), page,
                                          f"附录A 载机: {pf}"))
                mref = re.search(r"见\s*([A-Za-z0-9/\-\(\) ]{2,30})", note)
                if mref:
                    aliases.setdefault(mref.group(1).strip(), []).append(model)
        elif section == "B" and len(cells) >= 2:
            carrier = clean_platform(cells[0])
            for rad in split_multi(cells[1]):
                for head, _ in resolve_all(rad):
                    if 2 <= len(carrier) <= 40:
                        edges.append(edge(head, "deployedOn", carrier,
                                          S.classify_platform(carrier), page,
                                          f"附录B 载机: {carrier}"))

    (WORK / "manual_appendix_ab_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    (HERE / "ab_aliases.json").write_text(
        json.dumps(aliases, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    rc = Counter(e["relation"] for e in edges)
    print(f"[appendixAB] {len(edges)} edges  {dict(rc)}  | 别名指向 {len(aliases)}")


if __name__ == "__main__":
    main()
