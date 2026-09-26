"""
S6 类别关系 tail 归一：hasFunction / hasTechType / hasMode。

把大模型抽出的自由文本 tail 合并到受控词表，让多个雷达共享同一功能/体制/模式
节点（提升连通密度）。归不进词表的真·独特描述 → 降级为雷达属性（不做成孤立边）。

策略（先纯规则，命中不了再 --llm 兜底）：
  1. 预处理 tail：小写、去 radar/antenna/system 等后缀、提括号缩写
  2. 别名多标签匹配（lexicon 词表 + 本模块 EXTRA_ALIASES 补充）
  3. 命中 → 替换为规范值（一个 tail 可拆多个）；未命中 → demote

输入:  work/s4_gated.jsonl
输出:  work/s6_canon.jsonl（归一后的类别边）
       work/s6_demoted.jsonl（降级为属性的）
       work/s6_report.md
运行:  python pipeline/v3/s6_canonicalize.py [--llm]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))

import schema as S      # noqa: E402
import llm_client as L  # noqa: E402

WORK = ROOT / "pipeline" / "v3" / "work"
_ALIASES = json.loads((ROOT / "lexicon" / "entity_aliases.json").read_text(encoding="utf-8"))

REL_VOCAB_KEY = {"hasFunction": "functions", "hasTechType": "tech_types",
                 "hasMode": "radar_modes"}

# 补充别名/规范值（数据驱动；验证有效后并入 lexicon/entity_aliases.json）
EXTRA_ALIASES = {
    "functions": {
        "air search": ["air search", "air surveillance", "aerial surveillance",
                       "air-search", "air search and surveillance", "air defence surveillance",
                       "对空搜索", "对空监视", "对空", "远程对空搜索", "低空搜索",
                       "远程对空", "对空警戒", "对空探测"],
        "surface search": ["surface search", "surface surveillance",
                           "sea search", "sea surveillance", "surface search and navigation",
                           "对海搜索", "对海监视", "对海", "水面搜索", "近程对海搜索",
                           "对海警戒", "海面搜索", "对海探测"],
        "early warning": ["early warning", "early-warning", "ballistic missile early warning",
                          "预警", "远程预警"],
        "surveillance": ["surveillance", "general surveillance", "battlefield surveillance",
                         "ground surveillance", "coastal surveillance", "监视", "警戒", "海岸监视"],
        "air traffic control": ["air traffic control", "atc", "airport surveillance",
                                "terminal approach", "precision approach", "ground controlled approach",
                                "飞机进场控制", "进场控制", "飞机进场测速", "着舰引导", "进场引导"],
        "navigation": ["navigation", "navigational", "导航", "近程导航", "导航和监视"],
        "weather": ["weather", "meteorological", "precipitation", "气象", "气象监控", "气象探测"],
        "air intercept": ["air intercept", "air-intercept", "intercept", "拦截", "截击"],
        "mortar/artillery locating": ["mortar locating", "artillery locating",
                                      "weapon locating", "shell tracking", "武器定位", "弹着点定位"],
        "target illumination": ["target illumination", "illuminator", "illumination",
                                "照射", "目标照射", "跟踪和照射", "跟踪与照射"],
        "radar altimeter": ["radar altimeter", "altimeter", "altimetry", "测高", "高度测量"],
        "ground attack": ["ground attack", "strike", "attack radar", "对地攻击"],
        "fire control": ["火控", "火力控制", "火炮控制", "武器控制", "跟踪与火控", "跟踪和火控"],
        "target designation": ["目标指示", "目标指引", "target designation", "指示"],
        "missile guidance": ["导弹制导", "制导", "missile guidance", "missile control"],
        "IFF": ["敌我识别", "identification friend or foe", "iff", "beacon interrogation"],
        "target tracking": ["跟踪", "目标跟踪", "target tracking", "tracking"],
        "communication": ["通信", "communication", "数据链", "数据传输"],
        "fire control": ["fire-control", "fire control", "weapon control", "gun laying"],
        "target tracking": ["target tracking", "tracking", "track a selected target"],
        "target acquisition": ["target acquisition", "acquisition"],
        "height finding": ["height finding", "height-finding", "altitude finding"],
        "target detection": ["detection", "target detection", "search and detection"],
        "counter-battery": ["counter-battery", "counter battery", "weapon locating",
                            "artillery locating", "counterbattery"],
        "missile guidance": ["missile guidance", "missile control", "missile tracking"],
        "IFF": ["identification friend or foe", "iff", "beacon interrogation"],
    },
    "tech_types": {
        "有源相控阵": ["active electronically scanned array", "aesa",
                    "active phased array", "active array"],
        "无源相控阵": ["passive electronically scanned array", "pesa",
                    "passive phased array"],
        "相控阵": ["phased array", "phased-array", "electronically scanned"],
        "脉冲多普勒": ["pulse-doppler", "pulse doppler", "pulsedoppler"],
        "合成孔径": ["synthetic aperture", "sar"],
        "单脉冲": ["monopulse", "mono-pulse"],
        "动目标指示": ["moving target indication", "mti"],
        "连续波": ["continuous wave", "cw"],
        "调频连续波": ["frequency-modulated continuous wave", "fmcw", "fm-cw"],
        "3D": ["3d", "three dimensional", "3-d", "3 d"],
        "2D": ["2d", "two dimensional", "2-d"],
        "secondary": ["secondary surveillance", "secondary radar", "ssr"],
        "over-the-horizon": ["over-the-horizon", "over the horizon", "oth"],
    },
    "radar_modes": {
        "边扫边跟": ["track-while-scan", "track while scan", "tws"],
        "地形跟随": ["terrain-following", "terrain following"],
        "地形回避": ["terrain-avoidance", "terrain avoidance"],
        "动目标指示": ["moving target indication", "mti"],
        "合成孔径": ["synthetic aperture", "sar"],
        "单目标跟踪": ["single target track", "stt", "target tracking"],
        "搜索": ["search"],
        "速度搜索": ["velocity search"],
        "空空": ["air-to-air", "air to air"],
        "空地": ["air-to-ground", "air to ground"],
        "圆锥扫描": ["conical scan", "conical-scan"],
    },
}

_SUFFIX_RX = re.compile(
    r"\b(radar|antenna|system|equipment|set|mode|capability|function|"
    r"radars|systems)\b", re.I)
_ABBR_RX = re.compile(r"\(([A-Za-z][A-Za-z0-9\-/]{1,7})\)")


def build_index() -> dict:
    """(rel, surface_lower) -> canonical。lexicon 词表 + EXTRA 合并。"""
    idx = defaultdict(dict)
    for rel, vkey in REL_VOCAB_KEY.items():
        for canon, spec in (_ALIASES.get(vkey) or {}).items():
            if not isinstance(spec, dict):
                continue
            for s in {canon, spec.get("zh", ""), spec.get("en", ""),
                      *spec.get("aliases", [])}:
                if s:
                    idx[rel][s.lower()] = canon
        for canon, surfs in (EXTRA_ALIASES.get(vkey) or {}).items():
            for s in surfs:
                idx[rel][s.lower()] = canon
    return idx


_IDX = build_index()


def _match_in_table(rel: str, raw: str, cleaned: str, abbrs: list) -> list[str]:
    hits, table = [], _IDX[rel]
    for cand in (raw, cleaned, *abbrs):          # 整串精确
        if cand in table and table[cand] not in hits:
            hits.append(table[cand])
    if hits:
        return hits
    for surface in sorted(table, key=len, reverse=True):   # 长别名优先
        if len(surface) < 3:
            continue
        if re.search(r"\b" + re.escape(surface) + r"\b", cleaned):
            if table[surface] not in hits:
                hits.append(table[surface])
    return hits


def canon_tail(rel: str, tail: str) -> list[tuple[str, str]]:
    """返回 [(目标关系, 规范值)] 多标签。先本类，未命中试另两类（掰回抽错的类）。
    空 = 归不进任何词表 → 降级为属性。"""
    raw = tail.strip().lower()
    abbrs = [a.lower() for a in _ABBR_RX.findall(tail)]
    cleaned = _ABBR_RX.sub(" ", raw)
    cleaned = _SUFFIX_RX.sub(" ", cleaned)
    cleaned = re.sub(r"[^\w\s\-/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    order = [rel] + [r for r in REL_VOCAB_KEY if r != rel]
    for target in order:
        hits = _match_in_table(target, raw, cleaned, abbrs)
        if hits:
            return [(target, c) for c in hits]
    return []


def main():
    use_llm = "--llm" in sys.argv
    gated = [json.loads(l) for l in (WORK / "s4_gated.jsonl").open(encoding="utf-8")]
    cat_edges = [e for e in gated if e["relation"] in REL_VOCAB_KEY]
    other = [e for e in gated if e["relation"] not in REL_VOCAB_KEY]

    canon_f = (WORK / "s6_canon.jsonl").open("w", encoding="utf-8")
    demo_f  = (WORK / "s6_demoted.jsonl").open("w", encoding="utf-8")
    for e in other:                       # 非类别边原样透传
        canon_f.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats = Counter()
    unmatched_by_rel = defaultdict(Counter)
    for e in cat_edges:
        hits = canon_tail(e["relation"], e["tail"])
        if hits:
            for target_rel, c in hits:
                ne = dict(e); ne["relation"] = target_rel
                ne["tail"] = c; ne["tail_raw"] = e["tail"]
                canon_f.write(json.dumps(ne, ensure_ascii=False) + "\n")
                if target_rel != e["relation"]:
                    stats[f"{e['relation']}->{target_rel}:reclassified"] += 1
            stats[f"{e['relation']}:matched"] += 1
            stats[f"{e['relation']}:expanded"] += len(hits)
        else:
            demo_f.write(json.dumps({
                "entity": e["head"], "attr": e["relation"].replace("has", "").lower()
                + "_description", "value_raw": e["tail"],
                "evidence": e.get("evidence", ""), "doc_id": e.get("doc_id"),
                "tier": e.get("tier"), "unmapped": True}, ensure_ascii=False) + "\n")
            stats[f"{e['relation']}:demoted"] += 1
            unmatched_by_rel[e["relation"]][e["tail"].strip()] += 1
    canon_f.close(); demo_f.close()

    lines = ["# S6 类别 tail 归一报告", ""]
    for rel in REL_VOCAB_KEY:
        m, d = stats[f"{rel}:matched"], stats[f"{rel}:demoted"]
        tot = m + d
        canon_uniq = len({json.loads(l)["tail"] for l in
                          (WORK / "s6_canon.jsonl").open(encoding="utf-8")
                          if json.loads(l)["relation"] == rel})
        lines.append(f"## {rel}: {tot} 边  →  命中 {m} ({m/max(tot,1):.0%}), "
                     f"降级 {d}; 归一后 unique tail = {canon_uniq}")
        lines.append("未命中高频(候选补词表):")
        for t, n in unmatched_by_rel[rel].most_common(12):
            lines.append(f"  {n:3}  {t[:55]}")
        lines.append("")
    (WORK / "s6_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:2]))
    for rel in REL_VOCAB_KEY:
        m, d = stats[f"{rel}:matched"], stats[f"{rel}:demoted"]
        print(f"[S6] {rel:14} matched {m} / demoted {d}  ({m/max(m+d,1):.0%})")


if __name__ == "__main__":
    main()
