"""
把手册附录对照表映射为 v3 边（tier=v3_manual 0.93）。

附录C（研制机构/型号/国别）: 每型号 → developedBy maker + countryOfOrigin country
附录B（载机/型号）        : 每型号 → deployedOn platform      （待读入 appendix_b.jsonl）
附录E（频段/型号）        : 每型号 → hasFrequencyBand band    （待读入 appendix_e.jsonl）

只对**能对齐到 kg_v3 已有雷达**的型号建边（附录型号写法多样，避免造孤儿节点）；
对齐用放宽匹配（忽略大小写/空格/斜杠/radar 后缀）。未对齐的记入 report 供人工核。

输出:  work/manual_appendix_edges.jsonl（追加到 manual_edges 由 S5 读取）
运行:  python pipeline/v3/manual/map_appendix.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "pipeline" / "v3" / "work"
KGV3 = ROOT / "kg_v3"
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))
import schema as S  # noqa: E402


def norm(s):
    return re.sub(r"[\s\-_/\.]+", "", re.sub(r"(?i)\bradar\b", "", str(s))).lower()


def load_radar_index():
    ents = json.loads((KGV3 / "entities.json").read_text(encoding="utf-8"))
    idx, by_code = {}, {}
    for e in ents:
        if e["type"] in ("Radar", "WeaponSystem"):
            idx[norm(e["name"])] = e["name"]
            code = S.extract_designation(e["name"])
            if code:
                by_code[code] = e["name"]
    return idx, by_code


def edge(head, rel, tail, ttype, page, ev):
    return {"head": head, "head_type": "Radar", "relation": rel, "tail": tail,
            "tail_type": ttype, "evidence": ev, "tier": "v3_manual",
            "source_kind": "manual", "doc_id": f"manual_appendix_p{page}"}


MODEL_RX = re.compile(r"[A-Za-z0-9一-鿿]")
# 型号编号里的斜杠简写：AN/APY-1/2 → [AN/APY-1, AN/APY-2]；AN/APS-503/504 →
# [AN/APS-503, AN/APS-504]；AN/APG-67/67(V) → [AN/APG-67, AN/APG-67(V)]。
# 只展开最后一个 "-<段>" 之后的 "/<段>"，不动 "AN/" 前缀。
_SLASH_RX = re.compile(r"^(.*-)([A-Za-z0-9]+(?:\([^)]*\))?)((?:/[A-Za-z0-9]+(?:\([^)]*\))?)+)$")


def expand_slash(model):
    m = _SLASH_RX.match(model.strip())
    if not m:
        return [model]
    prefix, first, rest = m.group(1), m.group(2), m.group(3)
    out = [prefix + first]
    for seg in rest.split("/"):
        if seg:
            out.append(prefix + seg)
    return out


def clean_model(m):
    m = m.strip()
    # 过滤明显非雷达型号（纯厂商/描述词）
    if len(m) < 2 or len(m) > 45 or not MODEL_RX.search(m):
        return None
    return m


def main():
    idx, by_code = load_radar_index()
    edges, newnodes = [], 0

    def resolve_one(model):
        lat = S.cyrillic_to_latin(model)          # 俄语→拉丁（后续任何源同此路径）
        hit = idx.get(norm(lat)) or idx.get(norm(model))
        if hit:
            return hit, False
        code = S.extract_designation(lat)
        if code and code in by_code:              # 编号锚点：ЖУК(N010)→"N010 Zhuk"
            return by_code[code], False
        c = clean_model(lat)
        return (c, True) if c else (None, False)

    def resolve(model):
        """已有→规范名；未有→清理后原名（骨架节点）。斜杠简写(APY-1/2)先展开
        分别对齐,避免 norm 撞车(APY-1/2 vs APY-12)。返回 [(name,is_new)…]。"""
        variants = expand_slash(model)
        if len(variants) > 1:                     # 展开命中已有节点才采纳,否则当整体
            resolved = [resolve_one(v) for v in variants]
            if any(not isn for _, isn in resolved):
                return [(n, isn) for n, isn in resolved if n]
        r = resolve_one(model)
        return [r] if r[0] else []

    stats = {"aligned": 0, "new": 0}
    for cfile in sorted(HERE.glob("appendix_c*.jsonl")):
        for line in cfile.open(encoding="utf-8"):
            d = json.loads(line)
            for m in d["models"]:
                for head, is_new in resolve(m):
                    stats["new" if is_new else "aligned"] += 1
                    newnodes += is_new
                    edges.append(edge(head, "developedBy", d["maker"], "Manufacturer",
                                      d["page"], f"附录C 研制机构: {d['maker']}"))
                    if d.get("country"):
                        edges.append(edge(head, "countryOfOrigin", d["country"], "Country",
                                          d["page"], f"附录C 国别: {d['country']}"))

    for pat, rel, ttype, evlabel in [
            ("appendix_b*.jsonl", "deployedOn", "Aircraft", "附录B 载机"),
            ("appendix_e*.jsonl", "hasFrequencyBand", "FrequencyBand", "附录E 频段")]:
        for f in sorted(HERE.glob(pat)):
            for line in f.open(encoding="utf-8"):
                d = json.loads(line)
                for m in d["models"]:
                    for head, _ in resolve(m):
                        for tail in d["tails"]:
                            edges.append(edge(head, rel, tail, ttype, d["page"],
                                              f"{evlabel}: {tail}"))

    (WORK / "manual_appendix_edges.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    print(f"[appendix] {len(edges)} edges; aligned-to-existing {stats['aligned']} models, "
          f"new skeleton nodes {stats['new']}")


if __name__ == "__main__":
    main()
