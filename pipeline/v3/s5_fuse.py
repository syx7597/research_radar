"""
S5-S7 归一化 / 融合 / 产出。

- 全库唯一 junk filter（收编 generate_qa_500 那套规则）
- 保守规范化：head 仅精确匹配语料标题归一；tail 仅别名表命中归一（消歧已证伪）
- 类型由关系签名赋值（不信 LLM 的 type 意见）
- (head, relation, tail) 去重合并 provenance；多来源印证计数
- 单值关系多值 → conflicts.json 显式记录，不消解
- 置信度 = 金标分层先验（全量后需用新金标回归校准）

输入:  work/s2_edges.jsonl work/s4_gated.jsonl work/s2_attrs.jsonl work/s2_aliases.json
输出:  kg_v3/{edges,entities,conflicts}.json kg_v3/report.md
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))

import schema as S  # noqa: E402

WORK = ROOT / "pipeline" / "v3" / "work"
OUT  = ROOT / "kg_v3"

TIER_PRIOR = {"v3_wikidata": 0.92, "v3_manual": 0.93, "v3_struct": 0.90,
              "v3_llm_grounded": 0.85, "v3_llm_hinted": 0.75, "v3_judge_passed": 0.70,
              "v3_derived": 0.75}

GARBAGE = {"the same company", "which had previously", "the u", "the company",
           "no", "n/a", "none", "unknown", "various", "unnamed", "it", "this"}
NAME_OK = re.compile(r"^[A-Za-z0-9一-鿿][^\n\t]{1,59}$")

# 泛指词不构成 deployedOn/installedAt 的有效平台（试点抽检发现穿透词法门控）
GENERIC_PLATFORMS = {"aircraft", "ship", "ships", "shipborne", "vehicle",
                     "vehicles", "ground", "radar", "aircraft carriers",
                     "airborne", "naval vessels", "船", "舰艇", "飞机", "车辆"}
# usedIn 语义 = 参战冲突；尾巴长得像型号的一律拒（试点发现 usedIn AN/AWG-10 误用）
MODEL_LIKE = re.compile(r"^(AN/|[A-Z]{2,5}[-/ ]?\d)")


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/\.“”\"']+", "", s.lower())


_GENERIC_WORD = re.compile(r"\b(radar|system|systems|type|the|set|equipment|"
                           r"station|antenna)\b", re.I)


def _norm_relaxed(s: str) -> str:
    """放宽归一：额外去掉 radar/system/type 等通用词，用于谱系写法对齐。"""
    return _norm(_GENERIC_WORD.sub(" ", s))


_MAKER_SUFFIX = re.compile(
    r"\b(company|co|corp|corporation|inc|incorporated|ltd|limited|gmbh|llc|plc|"
    r"ag|sa|spa|pty|bv|ab|and|the|division|div|group|sector|systems?|electronics?|"
    r"technologies|technology|avionics?|aerospace|defen[cs]e|defense|navigation|"
    r"airborne|space|marine|naval|commercial|sensors?|integrated|solutions?|"
    r"radar|electronic|scanning)\b", re.I)


def maker_core(s: str) -> str:
    """厂商核心名：去括号注释/公司后缀/部门词 → 归一键。同核心=同厂商。
    只按逗号/分号分隔多厂商取主(不切 'and'，避免切碎部门名)。"""
    s = re.sub(r"[\(（][^)）]*[)）]", "", s)          # 去(英国分部)/(原研制…)
    s = re.split(r"[,，、;；]", s)[0]                 # 逗号分隔多厂商取主承包商
    s = _MAKER_SUFFIX.sub(" ", s)
    return re.sub(r"[^a-z0-9一-鿿]", "", s.lower())


def weg_aliases() -> dict:
    """WEG 条目 Alternative Designations → {系统名: [别名]}（NATO代号↔本国型号）。"""
    corpus = json.loads((ROOT / "radar_corpus" / "corpus.json").read_text(encoding="utf-8"))
    out = {}
    for d in corpus:
        if not d["id"].endswith("__weg"):
            continue
        m = re.search(r"Alternative Designations?:\s*([^\n]{2,90})", d.get("raw_text_en", ""))
        if not m:
            continue
        raw = m.group(1).strip()
        if raw.upper() in ("INA", "N/A", "SEE VARIANTS", "VARIANTS"):
            continue
        raw = re.split(r"\.\s|\bThe name\b|\bis\b|\bincludes\b", raw)[0]  # 砍掉句子尾巴
        als = [a.strip(" .,") for a in re.split(r"[,;/]| or ", raw) if len(a.strip(" .,")) >= 2]
        als = [a for a in als
               if 2 <= len(a) <= 25
               and not re.search(r"(?i)see |variant|manportable|date of|helicopt|"
                                 r"observation|radar and|system is", a)]
        if als:
            out[clean_system(d["en_title"])] = als
    return out


def clean_system(name: str) -> str:
    """WeaponSystem 名归一：去国家/类别前缀残留、括号注释、'X and Y' 取主型号。"""
    s = re.sub(r"[\(\（][^)）]*[\)）]", "", name)          # 去 (SA-11 FO)
    s = re.split(r"\s+and\s+", s)[0]                       # 'X and Y' → X
    # 去任意位置的国家/类别词（WEG 有的标题把类别放中间）
    s = re.sub(r"(?i)\b(Russian|Chinese|U\.S\.|British|French|German|Italian|"
               r"Swedish|European|Israeli|Iranian|SAM|MANPADS|SP AA)\b", " ", s)
    s = re.sub(r"(?i)\b(System|Air Defense|Gun/Missile|Launcher Vehicle|Vehicle)\b",
               " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,-/") or name.strip()


def _norm_platform(s: str) -> str:
    """平台写法归一：去括号国家注释/去 aircraft·class 等后缀，统一格式。
    只规整同一平台的不同写法，不跨平台合并。"""
    s = re.sub(r"[\(\（][^)）]*[\)）]", " ", s)          # 去 (希腊) 之类注释
    s = re.sub(r"\b(aircraft|helicopter|class|series|variant|"
               r"fighter|frigate|destroyer|cruiser|submarine|corvette|"
               r"carrier|patrol|vessel|ship|boat)\b", " ", s, flags=re.I)
    s = re.sub(r"[级型号舰艇船]+\s*$", "", s)            # 中文舰级后缀(Anzac级=Anzac)
    return re.sub(r"[\s\-_/\.]+", "", s.lower()).strip()


def is_junk(v: str) -> bool:
    v = v.strip()
    if re.fullmatch(r"\d{1,4}", v):                # 纯数字(页码残留当型号)
        return True
    if len(v) > 45 and v.count(" ") >= 5:          # 超长句子被当型号/实体
        return True
    return (not v or v.lower() in GARBAGE or not NAME_OK.match(v)
            or ("\n" in v) or len(v) > 60)


def load_title_canon() -> dict:
    """语料标题 + s2 别名 → 规范名映射（仅精确 norm 匹配）。wiki 名优先。"""
    corpus = json.loads((ROOT / "radar_corpus" / "corpus.json").read_text(encoding="utf-8"))
    canon = {}
    for d in corpus:                                   # rt/gs 先注册
        if d["id"].endswith(("__rt", "__gs")):
            canon.setdefault(_norm(d["en_title"]), d["en_title"])
    for d in corpus:                                   # wiki 覆盖（优先）
        if not d["id"].endswith(("__rt", "__gs")):
            canon[_norm(d["en_title"])] = d["en_title"]
    try:
        for name, aliases in json.loads((WORK / "s2_aliases.json")
                                         .read_text(encoding="utf-8")).items():
            for a in aliases:
                canon.setdefault(_norm(a), canon.get(_norm(name), name))
    except FileNotFoundError:
        pass
    # 放宽索引：去通用词后仍唯一的才收（避免歧义误合并）
    relaxed_ct = Counter(_norm_relaxed(v) for v in canon.values())
    relaxed = {}
    for k_norm, name in canon.items():
        rk = _norm_relaxed(name)
        if rk and relaxed_ct[rk] == 1:
            relaxed[rk] = name
    return canon, relaxed


def load_alias_merge():
    """见X 别名(手册权威异名) → 规范名。别名归一用。"""
    merge = {}
    for f in ("ab_aliases.json", "naval_aliases.json"):
        p = ROOT / "pipeline" / "v3" / "manual" / f
        if not p.exists():
            continue
        for canonical, variants in json.loads(p.read_text(encoding="utf-8")).items():
            for v in variants:
                merge[_norm(v)] = canonical.strip()
    return merge


def pick_canonical(group):
    """撞车组选最规范写法：连字符多 > AN/前缀 > 更长。"""
    return sorted(group, key=lambda x: (-x.count("-"),
                  -(1 if x.startswith("AN/") else 0), -len(x)))[0]


def main():
    OUT.mkdir(exist_ok=True)
    canon, canon_relaxed = load_title_canon()
    alias_merge = load_alias_merge()      # 见X 别名 → 规范名
    collide_merge = {}                    # 写法撞车：各写法 → 组内规范写法
    merged_aliases = defaultdict(set)     # 规范名 → 被并入的原写法(记录不丢)
    plat_canon = {}          # 平台写法归一：norm_platform -> 首个出现的规范写法
    stats = Counter()

    def canon_radar(name: str) -> str:
        """雷达名归一：精确/放宽 title → 撞车归并 → 见X别名归并。"""
        n = _norm(name)
        base = name
        if n in canon:
            base = canon[n]
        else:
            r = _norm_relaxed(name)
            if r in canon_relaxed:
                stats["lineage_relaxed_aligned"] += 1
                base = canon_relaxed[r]
        # 欠消歧归并：只做写法撞车(纯写法,绝对安全)。
        # 见X别名不合并节点——机载手册"见X"是系列条目指引(APQ-88见APQ-92),
        # 它们是不同型号;合并会把整个系列糊成一个。见X只作 aliases 记录(下方)。
        if base in collide_merge:
            merged_aliases[collide_merge[base]].add(base)
            return collide_merge[base]
        return base

    # s6_canon = 类别边已归一 + 非类别边透传（S6 未跑时回退到 s4_gated）
    llm_edges_file = "s6_canon.jsonl" if (WORK / "s6_canon.jsonl").exists() else "s4_gated.jsonl"
    raw_edges = []
    for f in ("s2_edges.jsonl", "s2b_wikidata_edges.jsonl", "manual_edges.jsonl",
              "manual_appendix_edges.jsonl", "manual_appendix_ab_edges.jsonl",
              "manual_body_edges.jsonl", "naval_body_edges.jsonl",
              "naval_appendix_edges.jsonl", llm_edges_file):
        p = WORK / f
        if p.exists():
            raw_edges.extend(json.loads(l) for l in p.open(encoding="utf-8"))

    # 写法撞车检测：收集雷达名(head + 雷达类型 tail)，同 norm 多写法 → 归并
    radar_names = set()
    for e in raw_edges:
        if e.get("head_type") in ("Radar", "RadarSystem"):
            radar_names.add(e["head"].strip())
        if e.get("tail_type") in ("Radar", "RadarSystem"):
            radar_names.add(e["tail"].strip())
    by_norm = defaultdict(list)
    for nm in radar_names:
        if nm and not is_junk(nm):
            by_norm[_norm(nm)].append(nm)
    for grp in by_norm.values():
        if len(grp) <= 1:
            continue
        if any("/" in g for g in grp):    # 斜杠型号(APY-1/2 vs APY-12)歧义，跳过
            continue
        c = pick_canonical(grp)
        for g in grp:
            if g != c:
                collide_merge[g] = c
    stats["collide_merged"] = len(collide_merge)

    # 厂商归一：同核心名的写法(Northrop Grumman Electronic Systems / Corporation …)
    # → 组内最短规范写法。厂商是 hub，写法碎片化是密度低的主因。
    maker_by_core = defaultdict(list)
    for e in raw_edges:
        if e["relation"] == "developedBy" and not is_junk(e["tail"]):
            maker_by_core[maker_core(e["tail"])].append(e["tail"].strip())
    maker_canon = {}
    for core, forms in maker_by_core.items():
        if not core or len(core) < 3:
            continue
        # 规范名选最简洁写法(Raytheon 而非 Raytheon Company Space and Airborne Systems)
        canonical = min(set(forms), key=lambda x: (len(x), x))
        for f in set(forms):
            maker_canon[f] = canonical
    stats["maker_forms_merged"] = sum(1 for f, c in maker_canon.items() if f != c)

    merged: dict[tuple, dict] = {}
    system_names: set[str] = set()          # norm 形式的 WeaponSystem 名
    for e in raw_edges:
        head, rel, tail = e["head"].strip(), e["relation"], e["tail"].strip()
        if rel not in S.EDGE_RELATIONS or is_junk(head) or is_junk(tail):
            stats["dropped_junk"] += 1
            continue
        # 试点抽检加的三条确定性规则
        if rel in ("deployedOn", "installedAt") and tail.lower() in GENERIC_PLATFORMS:
            stats["dropped_generic_platform"] += 1
            continue
        if rel == "usedIn" and MODEL_LIKE.match(tail):
            stats["dropped_usedin_model"] += 1
            continue
        if rel == "developedBy":
            as_country = S.resolve_alias(tail, "Country")
            if (S._ALIAS_IDX.get(("Country", tail.strip().lower()))
                    or as_country != tail.strip()):
                rel = "countryOfOrigin"          # 研制方填成国家 → 改挂国别
                tail = as_country
                stats["remap_devby_to_country"] += 1
            else:
                tail = maker_canon.get(tail, tail)   # 厂商写法归一到规范 hub
        if e.get("head_is_system"):               # WeaponSystem 名归一（在 key 之前）
            head = clean_system(head)
        else:
            head = canon_radar(head)
        tail_type = sorted(S.TYPE_SIG[rel][1])[0] if S.TYPE_SIG[rel][1] else "Entity"
        head = S.cyrillic_to_latin(head)          # 俄语雷达名转拉丁（对齐拉丁节点）
        if S.has_cyrillic(tail) and tail_type in ("RadarSystem", "Radar", "WeaponSystem"):
            tail = S.cyrillic_to_latin(tail)
        tail = S.resolve_alias(tail, tail_type)
        if rel in ("deployedOn", "installedAt"):  # 平台细分：载机/舰艇/车辆
            tail_type = S.classify_platform(tail)
        if rel == "partOfSystem":
            tail = clean_system(tail)             # 系统名归一
        elif tail_type in ("RadarSystem", "Radar"):
            tail = canon_radar(tail)              # 第2类：谱系放宽对齐
        elif tail_type in ("Platform", "NavalVessel", "AircraftPlatform",
                           "Aircraft", "GroundPlatform"):
            pk = _norm_platform(tail)             # 第3类：平台写法归一
            if pk:
                tail = plat_canon.setdefault(pk, tail)
        if _norm(head) == _norm(tail):
            stats["dropped_selfloop"] += 1
            continue

        key = (_norm(head), rel, _norm(tail))
        tier = e.get("tier", "v3_llm_grounded")
        head_type = "WeaponSystem" if e.get("head_is_system") else "Radar"
        if rel == "partOfSystem":
            system_names.add(_norm(tail))
        if e.get("head_is_system"):
            system_names.add(_norm(head))
        rec = merged.get(key)
        if rec is None:
            merged[key] = {
                "head": head, "relation": rel, "tail": tail,
                "head_type": head_type, "tail_type": tail_type,
                "tiers": [tier], "source_kinds": [e.get("source_kind", "?")],
                "doc_ids": [e.get("doc_id", "?")],
                "evidence": [e.get("evidence", "")[:200]],
            }
        else:
            rec["tiers"].append(tier)
            rec["source_kinds"].append(e.get("source_kind", "?"))
            rec["doc_ids"].append(e.get("doc_id", "?"))
            if len(rec["evidence"]) < 3:
                rec["evidence"].append(e.get("evidence", "")[:200])

    edges = []
    for rec in merged.values():
        tiers = rec.pop("tiers")
        kinds = sorted(set(rec.pop("source_kinds")))
        docs  = sorted(set(rec.pop("doc_ids")))
        rec["tier"] = max(tiers, key=lambda t: TIER_PRIOR.get(t, 0))
        rec["confidence"] = TIER_PRIOR.get(rec["tier"], 0.7)
        rec["source_kinds"] = kinds
        rec["doc_ids"] = docs
        rec["corroborated"] = len(kinds) >= 2 or len(docs) >= 2
        edges.append(rec)
        stats[f"rel:{rec['relation']}"] += 1
        stats[f"tier:{rec['tier']}"] += 1
        if rec["corroborated"]:
            stats["corroborated"] += 1

    # 派生 headquarteredIn：厂商总部国 = 它研制的雷达的多数国别(有据聚合，非灌水)
    maker_radars = defaultdict(list)
    radar_country = {}
    for rec in edges:
        if rec["relation"] == "developedBy":
            maker_radars[rec["tail"]].append(rec["head"])
        elif rec["relation"] == "countryOfOrigin":
            radar_country.setdefault(rec["head"], rec["tail"])
    for maker, rads in maker_radars.items():
        cc = Counter(radar_country[r] for r in rads if r in radar_country)
        if not cc:
            continue
        country, n = cc.most_common(1)[0]
        if n >= 2 and n >= 0.6 * sum(cc.values()):    # ≥2雷达且≥60%一致才派生
            edges.append({"head": maker, "head_type": "Manufacturer",
                          "relation": "headquarteredIn", "tail": country,
                          "tail_type": "Location",
                          "evidence": [f"派生: {maker} 研制的 {n} 部雷达国别为 {country}"],
                          "tier": "v3_derived", "confidence": 0.75,
                          "source_kinds": ["derived"], "doc_ids": ["derived"],
                          "corroborated": False})
            stats["derived_headquarteredIn"] += 1

    # 冲突：单值关系一头多尾
    conflicts = []
    by_hr = defaultdict(list)
    for rec in edges:
        if rec["relation"] not in S.MULTI_VALUED:
            by_hr[(rec["head"], rec["relation"])].append(rec)
    for (h, r), lst in by_hr.items():
        if len(lst) > 1:
            conflicts.append({"head": h, "relation": r,
                              "values": [{"tail": x["tail"], "tier": x["tier"],
                                          "doc_ids": x["doc_ids"]} for x in lst]})

    # 实体表：head 实体挂属性；tail 实体只记类型
    # 属性来源：S2 结构化属性 + S6 降级的独特功能/体制描述（不做成孤立边）
    attrs = defaultdict(list)
    for af in ("s2_attrs.jsonl", "s6_demoted.jsonl", "manual_attrs.jsonl",
               "manual_body_attrs.jsonl", "naval_body_attrs.jsonl",
               "naval_appendix_attrs.jsonl"):
        if not (WORK / af).exists():
            continue
        for l in (WORK / af).open(encoding="utf-8"):
            a = json.loads(l)
            name = canon.get(_norm(a["entity"]), a["entity"])
            attrs[name].append({k: a[k] for k in
                                ("attr", "value_raw", "value", "unit", "evidence",
                                 "doc_id", "tier", "unmapped") if k in a})
    try:
        alias_map = json.loads((WORK / "s2_aliases.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        alias_map = {}

    def etype(name, default):
        return "WeaponSystem" if _norm(name) in system_names else default

    # 雷达平台域推断：来源优先(手册/WEG) + deployedOn 平台类型补充
    dom_docs = defaultdict(set)      # radar → 该雷达所有边的 doc_id
    dom_plat = defaultdict(set)      # radar → deployedOn 的平台类型
    for rec in edges:
        for d in rec["doc_ids"]:
            dom_docs[rec["head"]].add(d)
        if rec["relation"] in ("deployedOn", "installedAt"):
            dom_plat[rec["head"]].add(rec["tail_type"])

    def radar_domain(name):
        docs = list(dom_docs.get(name, ()))
        # 世界海用雷达手册按 PDF 页分三段：33-520 舰载 / 521-576 岸基 / 577-642 机载
        nav_pages = [int(m.group(1)) for d in docs
                     if (m := re.search(r"naval_p0*(\d+)", d))]
        if nav_pages:
            p = min(nav_pages)
            return "airborne" if p >= 577 else "ground" if p >= 521 else "naval"
        joined = " ".join(docs)
        if "manual_body" in joined or "manual_appendix" in joined or "manual_p" in joined:
            return "airborne"                   # 机载雷达手册
        pts = dom_plat.get(name, set())
        if "NavalVessel" in pts:
            return "naval"
        if "Aircraft" in pts:
            return "airborne"
        if "GroundVehicle" in pts:
            return "ground"
        return "unknown"

    entities = {}
    for rec in edges:
        entities.setdefault(rec["head"], {"name": rec["head"],
                                          "type": etype(rec["head"], rec["head_type"]),
                                          "aliases": [], "attributes": []})
        entities.setdefault(rec["tail"], {"name": rec["tail"],
                                          "type": etype(rec["tail"], rec["tail_type"]),
                                          "aliases": [], "attributes": []})
    for name, lst in attrs.items():
        ent = entities.setdefault(name, {"name": name, "type": etype(name, "Radar"),
                                         "aliases": [], "attributes": []})
        ent["attributes"] = lst
    for name, al in alias_map.items():
        cname = canon.get(_norm(name), name)
        if cname in entities:
            entities[cname]["aliases"] = sorted(set(al))
    # WEG 条目的 Alternative Designations → WeaponSystem 实体别名（记录，不跨节点强并）
    for sysname, al in weg_aliases().items():
        if sysname in entities:
            entities[sysname]["aliases"] = sorted(set(entities[sysname]["aliases"]) | set(al))
    # 归并入的原写法 → 规范节点 aliases（撞车归并，不丢原名）
    for tgt, olds in merged_aliases.items():
        if tgt in entities:
            entities[tgt]["aliases"] = sorted(set(entities[tgt]["aliases"]) | olds)
    # 见X 别名 → 规范节点 aliases（只记录，不合并节点：机载见X是系列指引）
    for alias_norm, target in alias_merge.items():
        if target in entities and _norm(target) != alias_norm:
            entities[target]["aliases"] = sorted(
                set(entities[target].get("aliases", [])) | {alias_norm})
    # 雷达平台域标注（airborne/naval/ground/unknown）
    for ent in entities.values():
        if ent["type"] == "Radar":
            ent["domain"] = radar_domain(ent["name"])

    (OUT / "edges.json").write_text(
        json.dumps(edges, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "entities.json").write_text(
        json.dumps(list(entities.values()), ensure_ascii=False, indent=1),
        encoding="utf-8")
    (OUT / "conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=1), encoding="utf-8")

    n_attr = sum(len(v) for v in attrs.values())
    rels = {k[4:]: v for k, v in stats.items() if k.startswith("rel:")}
    tiers = {k[5:]: v for k, v in stats.items() if k.startswith("tier:")}
    lines = ["# KG v3 build report", "",
             f"- edges: **{len(edges)}**  (junk dropped {stats['dropped_junk']}, "
             f"self-loop {stats['dropped_selfloop']})",
             f"- entities: **{len(entities)}**, attributes: **{n_attr}**",
             f"- corroborated (≥2 sources/docs): **{stats['corroborated']}** "
             f"({stats['corroborated']/max(len(edges),1):.1%})",
             f"- conflicts (single-valued, multi-tail): **{len(conflicts)}**", "",
             "## edges by tier", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(tiers.items(), key=lambda x: -x[1])]
    lines += ["", "## edges by relation", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(rels.items(), key=lambda x: -x[1])]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[S5] edges {len(edges)} | entities {len(entities)} | attrs {n_attr} "
          f"| corroborated {stats['corroborated']} | conflicts {len(conflicts)}")


if __name__ == "__main__":
    main()
