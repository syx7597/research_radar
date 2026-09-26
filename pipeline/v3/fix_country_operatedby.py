# -*- coding: utf-8 -*-
"""KG 修复:海用手册附录F的"国别分章"= 操作国,被误标成 countryOfOrigin。
证据仅来自"附录F国别:X"且非印证的 countryOfOrigin 边 → 改标为 operatedBy。
真原产国(带 "Country of origin:" 证据 / wiki 源 / 印证)保持不动。

  python pipeline/v3/fix_country_operatedby.py            # dry-run(只报告)
  python pipeline/v3/fix_country_operatedby.py --apply    # 写回(先备份 edges.json.bak)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EDGES = ROOT / "kg_v3" / "edges.json"


def evlist(e):
    ev = e.get("evidence", [])
    return ev if isinstance(ev, list) else [ev]


def is_operator_pollution(e):
    """证据全部来自附录F国别表 且 非印证 → 操作国污染(非原产国)。"""
    if e["relation"] != "countryOfOrigin":
        return False
    if e.get("corroborated"):
        return False
    evs = [x for x in evlist(e) if x]
    if not evs:
        return False
    return all("附录F" in x or "附录 F" in x for x in evs)


def main():
    apply = "--apply" in sys.argv
    E = json.loads(EDGES.read_text(encoding="utf-8"))

    # 已有 operatedBy 对,用于去重
    existing_op = {(e["head"], e["tail"]) for e in E if e["relation"] == "operatedBy"}

    convert, keep_examples, converted_edges, dropped_dup = [], [], [], 0
    out = []
    for e in E:
        if is_operator_pollution(e):
            key = (e["head"], e["tail"])
            if key in existing_op:
                dropped_dup += 1          # 已有 operatedBy,丢弃重复
                continue
            e2 = dict(e)
            e2["relation"] = "operatedBy"
            e2["_relabeled_from"] = "countryOfOrigin"   # 留痕
            existing_op.add(key)
            out.append(e2)
            convert.append((e["head"], e["tail"]))
        else:
            out.append(e)
            if e["relation"] == "countryOfOrigin" and len(keep_examples) < 6:
                keep_examples.append((e["head"], e["tail"], evlist(e)[:1]))

    print(f"=== KG 修复 dry-run{' (将写回)' if apply else ''} ===")
    print(f"countryOfOrigin 总边: {sum(1 for e in E if e['relation']=='countryOfOrigin')}")
    print(f"判为操作国污染 → 改 operatedBy: {len(convert)}")
    print(f"其中与已有 operatedBy 重复而丢弃: {dropped_dup}")
    print(f"\n--- 改标示例(操作国) ---")
    for h, t in convert[:8]:
        print(f"   {h}  --operatedBy-->  {t}")
    print(f"\n--- 保留为 countryOfOrigin 的示例(真原产国,应带真实origin证据) ---")
    for h, t, ev in keep_examples:
        print(f"   {h} -> {t}   证据: {ev}")

    # 复查:改标后,还有多少雷达仍是多原产国(应大幅下降)
    from collections import defaultdict
    coo = defaultdict(int)
    for e in out:
        if e["relation"] == "countryOfOrigin":
            coo[e["head"]] += 1
    multi = sum(1 for v in coo.values() if v > 1)
    print(f"\n改标后仍 >1 原产国的雷达: {multi} (修复前 193)")

    if apply:
        EDGES.with_suffix(".json.bak").write_text(
            json.dumps(E, ensure_ascii=False, indent=1), encoding="utf-8")
        EDGES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[写回] edges.json 已更新;原文件备份 -> edges.json.bak")
    else:
        print(f"\n[dry-run] 未写回。确认无误后加 --apply")


if __name__ == "__main__":
    main()
