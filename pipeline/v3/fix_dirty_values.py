# -*- coding: utf-8 -*-
"""KG 层清洗 4 类脏值(人工核验发现;此前只在数据集生成器层清过,这里落到 kg_v3 让检索/RL环境也干净)。

  ① 字段粘连:head/tail 含全半角冒号或 <br>(如"变体：平台""规格 距离:50m")→ 丢弃该边(不瞎拆)
  ② hasFrequencyBand 书写不统一:归一(去空格/band后缀/中文残留, "K u"→"Ku")→ 原地改写
  ③ developedBy 混入非公司(型号/机型如 AN/AWG-10、Douglas F4D Skyray)→ 丢弃该边
  ④ 公司名未归一(子公司/拼写差异)→ 归并到已知母公司/修拼写 → 原地改写

  python pipeline/v3/fix_dirty_values.py           # dry-run
  python pipeline/v3/fix_dirty_values.py --apply    # 写回(备份 edges.json.bak2)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EDGES = ROOT / "kg_v3" / "edges.json"
ENTS = ROOT / "kg_v3" / "entities.json"

Nodes = json.loads(ENTS.read_text(encoding="utf-8"))
known_models = {n["name"] for n in Nodes
                if n.get("type") in ("Radar", "RadarSystem", "Aircraft",
                                      "NavalVessel", "GroundVehicle", "Platform")}

DIRTY = re.compile(r"[：:]|<br\s*/?>")
DESIG = re.compile(r"^(AN/|APG|APS|APQ|SPG|SPS|SPY|MPQ|TPS|TPY|MiG-|F-\d)", re.I)
AIRCRAFT_TOK = re.compile(r"(F4D|Skyray|Tornado|Mirage|MiG-|F-1[56])", re.I)
BAND_OK = re.compile(r"^(X|S|C|L|K|Ku|Ka|UHF|VHF|HF|EHF|SHF|P|W|D|E|F|G|H|I|J)([\s/-]|$)", re.I)

# ④ 已知母公司(子公司/部门归并到它);末尾拼写修正
PARENTS = ["Northrop Grumman", "Lockheed Martin", "Raytheon", "BAE Systems",
           "SELEX Galileo", "Selex Galileo", "Thales", "Westinghouse Electric",
           "Hughes Aircraft", "General Electric"]
COMPANY_FIX = {"Westinghouse Electrics": "Westinghouse Electric",
               "SELEX Galileo": "Selex Galileo"}

def canon_maker(c):
    c = c.strip()
    if "/" in c:                     # 联合研制(A / B)不动,避免丢共研方
        return c
    if c in COMPANY_FIX:
        c = COMPANY_FIX[c]
    for p in PARENTS:
        if c.lower().startswith(p.lower() + " "):   # 仅"母公司名+空格+部门"→母公司
            return COMPANY_FIX.get(p, p)
    return c

def norm_band(b):
    b = b.strip()
    b = re.sub(r"\([^)]*\)", "", b)                        # 去括注(IEEE/NATO designation)
    b = re.sub(r"\s*(band|波段)\s*", "", b, flags=re.I)   # 去 band/波段(任意位置)
    b = re.sub(r"[一-鿿]", "", b)                          # 去中文残留(低)
    b = re.sub(r"\bK\s+u\b", "Ku", b, flags=re.I)
    b = re.sub(r"\s*/\s*", "/", b)
    return re.sub(r"\s+", "", b).strip("/")

def is_company(tail):
    # 只按型号前缀/机型正则判"非公司"(不用 known_models: KG 有把真公司误标成 Radar 的)
    return not (DESIG.match(tail) or AIRCRAFT_TOK.search(tail))


def main():
    apply = "--apply" in sys.argv
    E = json.loads(EDGES.read_text(encoding="utf-8"))
    stats = Counter()
    ex = {"concat": [], "band": [], "noncompany": [], "company": []}
    out, seen = [], set()

    for e in E:
        h, r, t = e["head"], e["relation"], e["tail"].strip()
        # ① 字段粘连 → 丢
        if DIRTY.search(h) or DIRTY.search(t):
            stats["drop_concat"] += 1
            if len(ex["concat"]) < 5:
                ex["concat"].append(f"{h} -{r}-> {t}")
            continue
        # ③ developedBy 非公司 → 丢
        if r == "developedBy" and not is_company(t):
            stats["drop_noncompany"] += 1
            if len(ex["noncompany"]) < 5:
                ex["noncompany"].append(f"{h} -developedBy-> {t}")
            continue
        # ② 频段归一(保留真实频段, 只丢归一后变空的真垃圾; 不按 BAND_OK 删真数据)
        if r == "hasFrequencyBand":
            nt = norm_band(t)
            if not nt:
                stats["drop_badband"] += 1
                continue
            if nt != t:
                stats["norm_band"] += 1
                if len(ex["band"]) < 6:
                    ex["band"].append(f"{t!r} -> {nt!r}")
                t = nt
        # ④ 公司名归一
        if r == "developedBy":
            ct = canon_maker(t)
            if ct != t:
                stats["norm_company"] += 1
                if len(ex["company"]) < 6:
                    ex["company"].append(f"{t!r} -> {ct!r}")
                t = ct
        e2 = dict(e); e2["tail"] = t
        key = (h, r, t)
        if key in seen:            # 归一后可能与已有边重复 → 去重
            stats["dedup"] += 1
            continue
        seen.add(key)
        out.append(e2)

    print(f"=== KG 脏值清洗 dry-run{' (将写回)' if apply else ''} ===")
    print(f"原边 {len(E)} → 清洗后 {len(out)}")
    print(f"① 丢弃字段粘连: {stats['drop_concat']}")
    for x in ex["concat"]:
        print(f"     {x}")
    print(f"③ 丢弃非公司研制方: {stats['drop_noncompany']}")
    for x in ex["noncompany"]:
        print(f"     {x}")
    print(f"② 频段归一改写: {stats['norm_band']} (无效频段丢弃 {stats['drop_badband']})")
    for x in ex["band"]:
        print(f"     {x}")
    print(f"④ 公司名归一: {stats['norm_company']}")
    for x in ex["company"]:
        print(f"     {x}")
    print(f"归一后去重: {stats['dedup']}")

    if apply:
        EDGES.with_suffix(".json.bak2").write_text(
            json.dumps(E, ensure_ascii=False, indent=1), encoding="utf-8")
        EDGES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[写回] edges.json 已更新;备份 -> edges.json.bak2")
    else:
        print(f"\n[dry-run] 未写回。确认无误加 --apply")


if __name__ == "__main__":
    main()
