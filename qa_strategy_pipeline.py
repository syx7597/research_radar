"""
End-to-end Strategy-Routed GraphRAG pipeline.

Flow:
  question
    -> QuestionRouter.route()                 (LLM-1: type + strategy)
    -> StrategyParser.parse()                 (LLM-2: extract structured args)
    -> StrategyExecutor.execute()             (KG queries, no LLM)
    -> ContextRenderer.render()               (format evidence per strategy)
    -> Answerer.answer()                      (LLM-3: final natural answer)

Strategies:
  lookup            BM25+Vector+RRF on the existing HybridRetriever
  exhaustive        kg.heads_with((rel, tail))   — full type-pool retrieval
  complement        kg.tails_of(head, rel)       — verify absence by enumerating presence
  path_plan         walk a relation chain        — multi-hop bridge / chain
  constrained_join  intersect heads from N constraints
  dual_subgraph     two parallel lookups, then compare
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json
import re
import time
import sys
import requests
from pathlib import Path
from collections import defaultdict
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lexicon import load_relations, build_alias_index, build_relation_lookup, render_triple_text
from qa_router import QuestionRouter, _call_llm as _router_llm  # reuse infrastructure


DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# LLM backend is a SWAPPABLE module (deployment portability / 保密边界): default DeepSeek,
# override via env to any OpenAI-compatible endpoint — e.g. a LOCAL Ollama model run
# entirely inside a secure boundary (no external API):
#   LLM_URL=http://localhost:11434/v1/chat/completions  LLM_MODEL=qwen2.5:7b  LLM_KEY=ollama
LLM_URL = os.getenv("LLM_URL", DEEPSEEK_URL)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_KEY = os.getenv("LLM_KEY", DEEPSEEK_KEY)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


def llm_call(messages: list, max_tokens: int = 512, temperature: float = 0.0,
             retries: int = 2) -> str:
    payload = {"model": LLM_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(LLM_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return f"[LLM_ERROR: {last_err}]"


# ─────────────────────────────────────────────────────────────
#  KG index
# ─────────────────────────────────────────────────────────────

class KGIndex:
    def __init__(self, triples):
        self.triples = triples
        self.by_relation_tail: dict = defaultdict(set)
        self.by_head_relation: dict = defaultdict(set)
        for t in triples:
            h, r, ta = t["head"], t["relation"], t["tail"]
            self.by_relation_tail[(r, ta)].add(h)
            self.by_head_relation[(h, r)].add(ta)

    def heads_with(self, relation: str, tail: str) -> set:
        # also try fuzzy alias match: if exact misses, try alias_index
        return set(self.by_relation_tail.get((relation, tail), set()))

    def tails_of(self, head: str, relation: str) -> set:
        return set(self.by_head_relation.get((head, relation), set()))

    def all_tails_for(self, relation: str) -> set:
        out = set()
        for (r, t), _ in self.by_relation_tail.items():
            if r == relation:
                out.add(t)
        return out


# ─────────────────────────────────────────────────────────────
#  Strategy argument parser
# ─────────────────────────────────────────────────────────────

PARSER_SYSTEM = """你是雷达知识图谱查询解析器。从问题中抽取结构化参数（用于图谱检索）。

可用关系列表（schema id (head_type -> tail_type)：中文标签 / 同义词）：
{relation_specs}

输出 JSON 格式（严格按此 schema，不要返回额外字段，不要 markdown 包裹）：
{{
  "primary_entity":   "主实体名（雷达型号、厂商、国家等），若无则空串",
  "secondary_entity": "次实体名（仅对比题用，否则空串）",
  "relation_chain":   ["关系schema id 列表，按问题中的语义顺序"],
  "constraints": [
    {{"relation": "关系schema id", "tail": "约束实体值"}}
  ],
  "forbidden_tail":   "否定题中的目标对象（如\\"是否被中国使用\\"中的\\"中国\\"），否则空串",
  "answer_target":    "count|enum|entity|yesno|literal"
}}

抽取规则：
- 关系名必须从可用列表里选 schema id；遇到中文同义词请映射到对应英文 id。
- **关系类型必须匹配**：constraint 的 tail 类型要和关系的 tail_type 一致。
  * Country 类型（美国/中国/俄罗斯/法国/...） → 用 operatedBy / countryOfOrigin / exportedTo / affiliatedTo（**绝不用 developedBy**，因为 developedBy 的 tail 是 Manufacturer 而不是 Country）。
  * Manufacturer 类型（Raytheon/Lockheed Martin/Northrop Grumman/...）→ 用 developedBy。
  * FrequencyBand 类型（S/X/L/Ku/...）→ 用 hasFrequencyBand，**频段值用单字母不带"波段"后缀**。
  * Platform/NavalVessel 类型 → 用 deployedOn。
- 中文表达消歧：
  * "X 国研制" / "X 国生产" / "X 国制造" → 当 X 是国家时用 countryOfOrigin（不是 developedBy）。
  * "X 国使用" / "X 国装备" → operatedBy。
  * "X 公司研制" → developedBy（X 是公司名）。
- 国家名保留中文标准形式（中国/美国/俄罗斯/...）。
- 雷达型号保留原文（AN/TPY-2、Raytheon 等不翻译）。
- "多少款/几种" → count；"列出/都有哪些" → enum；"是否/有没有" → yesno；属性查询 → literal/entity。
- agg/enum 题：constraints 至少 1 个；agg_count 时 answer_target=count。
- negation 题：constraints 通常 [(主实体所对应的关系)]，forbidden_tail 是被询问对象。
- attr_filter：constraints 列表 >= 2 个，answer_target=enum。
- 比较题（A 和 B）：primary=A, secondary=B, relation_chain 含被比较的关系。
- 桥接题：relation_chain 按链顺序列出，从 primary_entity 出发。

错例（不要这么做）：
- ❌ "美国研制的雷达" → constraints=[{{"relation":"developedBy","tail":"美国"}}]   # 错：美国是Country，developedBy要Manufacturer
- ✅ "美国研制的雷达" → constraints=[{{"relation":"countryOfOrigin","tail":"美国"}}]
- ❌ "S波段的雷达" → constraints=[{{"relation":"hasFrequencyBand","tail":"S波段"}}]
- ✅ "S波段的雷达" → constraints=[{{"relation":"hasFrequencyBand","tail":"S"}}]

只输出 JSON，不要解释。
"""

USER_TEMPLATE = """问题：{question}
已知题型：{qtype} （提示策略：{strategy}）

请输出 JSON。"""


def _build_relation_specs() -> str:
    rels = load_relations()["relations"]
    lines = []
    for rid, rec in rels.items():
        if not isinstance(rec, dict):  # skip comment/note entries like `_v2_new_relations_note`
            continue
        syns = "/".join([rec.get("zh_label", "")] + rec.get("zh_synonyms", []))
        head_t = "/".join(rec.get("head_types", []))
        tail_t = "/".join(rec.get("tail_types", []))
        lines.append(f"  {rid:18s} ({head_t} -> {tail_t}): {syns}")
    return "\n".join(lines)


_RELATION_SPECS_CACHE: Optional[str] = None
def get_relation_specs() -> str:
    global _RELATION_SPECS_CACHE
    if _RELATION_SPECS_CACHE is None:
        _RELATION_SPECS_CACHE = _build_relation_specs()
    return _RELATION_SPECS_CACHE


_ALL_TYPE_INDEX_CACHE = None
def _build_all_type_index() -> dict:
    """surface_form -> set of types (a value may belong to multiple types,
    e.g., '合成孔径' is both RadarMode and TechType)."""
    global _ALL_TYPE_INDEX_CACHE
    if _ALL_TYPE_INDEX_CACHE is not None:
        return _ALL_TYPE_INDEX_CACHE
    from lexicon import load_entity_aliases
    raw = load_entity_aliases()
    sec_to_type = {
        "countries": "Country", "radar_modes": "RadarMode", "tech_types": "TechType",
        "functions": "Function", "naval_vessel_classes": "NavalVessel",
    }
    out: dict = {}
    for section, ent_type in sec_to_type.items():
        for canonical, rec in raw.get(section, {}).items():
            for form in [canonical, rec.get("zh", ""), rec.get("en", "")] + list(rec.get("aliases", [])):
                if not form: continue
                out.setdefault(form, set()).add(ent_type)
                out.setdefault(form.lower(), set()).add(ent_type)
    _ALL_TYPE_INDEX_CACHE = out
    return out


def _infer_tail_types(tail_value: str, aliases: dict) -> set:
    """Return the SET of possible entity types for a tail value (multi-type aware)."""
    if not tail_value:
        return set()
    all_idx = _build_all_type_index()
    types = set(all_idx.get(tail_value, set()) | all_idx.get(tail_value.lower(), set()))
    if types:
        return types
    # fall back to single-type alias lookup
    hit = aliases.get(tail_value) or aliases.get(tail_value.lower())
    if hit and hit.get("type"):
        types.add(hit["type"])
    # try suffix-stripped variants
    for sfx in _SUFFIX_STRIPS:
        if tail_value.endswith(sfx) and len(tail_value) > len(sfx):
            stripped = tail_value[:-len(sfx)]
            types |= all_idx.get(stripped, set())
            shit = aliases.get(stripped)
            if shit and shit.get("type"):
                types.add(shit["type"])
    return types


def _suggest_compatible_relation(tail_type: str, current_relation: str,
                                  relations_lex: dict, kg: KGIndex) -> Optional[str]:
    """If current_relation's tail_type doesn't match the inferred tail_type,
    pick an alternative relation accepting tail_type. Prefers most-frequent."""
    cur_spec = relations_lex.get(current_relation, {})
    cur_tail_types = set(cur_spec.get("tail_types", []))
    if tail_type in cur_tail_types:
        return None  # already compatible

    candidates = []
    for rid, spec in relations_lex.items():
        if tail_type in (spec.get("tail_types", []) or []):
            # frequency in KG = num of edges with this relation
            freq = sum(1 for (r, _t), _ in kg.by_relation_tail.items() if r == rid)
            candidates.append((freq, rid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _validate_args(args: dict, aliases: dict, relations_lex: dict, kg: KGIndex) -> dict:
    """Post-parse: type-check each constraint. Only substitute when ALL inferred
    types for the tail are incompatible with the relation's expected tail_types
    AND the current (relation, tail) lookup yields zero hits.
    Multi-type tails (e.g., '合成孔径' is both RadarMode and TechType) are accepted
    if any inferred type matches."""
    if BILINGUAL_DISABLED:
        return args  # no type-aware relation substitution
    fixes_applied = []
    cons_list = args.get("constraints") or []
    for c in cons_list:
        rel = c.get("relation", "")
        tail = c.get("tail", "")
        if not (rel and tail):
            continue
        tail_types = _infer_tail_types(tail, aliases)
        if not tail_types:
            continue  # unknown → trust parser
        spec = relations_lex.get(rel, {})
        expected = set(spec.get("tail_types", []))
        if tail_types & expected:
            continue  # at least one inferred type is compatible
        # Type mismatch — but double-check: does the (rel, tail) pair actually
        # produce hits in the KG? If yes, the parser's intent is satisfiable; don't substitute.
        if kg.heads_with(rel, tail):
            continue
        # Try alias-resolved tails too
        any_hit = False
        for tc in _alias_resolve_tail(tail, aliases):
            if kg.heads_with(rel, tc):
                any_hit = True; break
        if any_hit:
            continue
        # Truly incompatible AND no hits: suggest replacement
        # Pick a relation whose tail_types overlaps with inferred and is most frequent
        best = None; best_freq = -1
        for rid, sp in relations_lex.items():
            tt = set(sp.get("tail_types", []))
            if tail_types & tt:
                freq = sum(1 for (r, _t), _ in kg.by_relation_tail.items() if r == rid)
                if freq > best_freq:
                    best_freq = freq; best = rid
        if best and best != rel:
            fixes_applied.append({
                "tail": tail, "tail_types": sorted(tail_types),
                "from": rel, "to": best,
                "reason": f"{rel} expects {sorted(expected)}, inferred {sorted(tail_types)}; no KG hits for original",
            })
            c["relation"] = best
    if fixes_applied:
        args["_type_fixes"] = fixes_applied
    return args


_RELATIONS_CACHE = None
def _get_relations_lex():
    global _RELATIONS_CACHE
    if _RELATIONS_CACHE is None:
        _RELATIONS_CACHE = load_relations()["relations"]
    return _RELATIONS_CACHE


def parse_args(question: str, qtype: str, strategy: str,
               aliases: dict | None = None, kg: KGIndex | None = None) -> dict:
    sys_p = PARSER_SYSTEM.format(relation_specs=get_relation_specs())
    user = USER_TEMPLATE.format(question=question, qtype=qtype, strategy=strategy)
    raw = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user",   "content": user}], max_tokens=400)
    # try to extract JSON from raw (LLM may wrap in markdown)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {"_parse_error": True, "raw": raw}
    try:
        obj = json.loads(m.group(0))
        obj["_raw"] = raw
    except json.JSONDecodeError as e:
        return {"_parse_error": True, "raw": raw, "err": str(e)}

    # post-parse type validation (only if we have access to alias + kg)
    if aliases is not None and kg is not None:
        obj = _validate_args(obj, aliases, _get_relations_lex(), kg)
    return obj


# ─────────────────────────────────────────────────────────────
#  Strategy executors (KG-side, no LLM)
# ─────────────────────────────────────────────────────────────

_SUFFIX_STRIPS = ["波段", "频段", "公司", "集团", "雷达", "型号", "导弹", "战斗机", "舰艇"]
_FREQ_BAND_NORMALIZE = {
    "S波段": "S", "X波段": "X", "L波段": "L", "C波段": "C",
    "Ku波段": "Ku", "KU波段": "Ku", "K波段": "K", "Ka波段": "Ka",
    "VHF波段": "VHF", "UHF波段": "UHF",
    "I波段": "I", "J波段": "J", "I/J波段": "I/J",
    "E波段": "E", "F波段": "F", "G波段": "G", "E/F波段": "E/F",
    "甚高频": "VHF", "特高频": "UHF",
}

# Module-level kill switch for the bilingual layer (used by the ablation
# experiment). When True, _alias_resolve_tail / equivalence-class fallback /
# type-validation substitution all become no-ops, so the pipeline relies purely
# on parser-extracted surface forms matching KG canonical forms verbatim.
BILINGUAL_DISABLED = False


def _alias_resolve_tail(tail_value: str, aliases: dict) -> list:
    """Given a parser-extracted tail, return all possible KG-canonical surface
    forms to query. Tries direct match, alias lookup, suffix-stripped variants."""
    if not tail_value:
        return []
    if BILINGUAL_DISABLED:
        return [tail_value]
    candidates = [tail_value]

    # 1. direct alias index lookup
    hit = aliases.get(tail_value) or aliases.get(tail_value.lower())
    if hit and hit.get("canonical") and hit["canonical"] != tail_value:
        candidates.append(hit["canonical"])

    # 2. frequency-band normalization map
    if tail_value in _FREQ_BAND_NORMALIZE:
        candidates.append(_FREQ_BAND_NORMALIZE[tail_value])

    # 3. strip common Chinese suffixes (波段/公司/雷达 etc.)
    for sfx in _SUFFIX_STRIPS:
        if tail_value.endswith(sfx) and len(tail_value) > len(sfx):
            stripped = tail_value[:-len(sfx)]
            candidates.append(stripped)
            # try alias on stripped form too
            shit = aliases.get(stripped) or aliases.get(stripped.lower())
            if shit and shit.get("canonical"):
                candidates.append(shit["canonical"])

    return list(dict.fromkeys(candidates))


def exec_complement(args: dict, kg: KGIndex, aliases: dict) -> dict:
    head = args.get("primary_entity", "")
    rels = args.get("relation_chain") or []
    rel = rels[0] if rels else (args.get("constraints") or [{}])[0].get("relation", "")
    forbidden = args.get("forbidden_tail", "")

    if not (head and rel):
        return {"head": head, "relation": rel, "known_tails": [], "note": "missing args"}

    known = sorted(kg.tails_of(head, rel))

    forbidden_canonical = []
    if forbidden:
        forbidden_canonical = _alias_resolve_tail(forbidden, aliases)
        # Also try direct match in known
    contains_forbidden = False
    if forbidden:
        for fc in forbidden_canonical:
            if fc in known:
                contains_forbidden = True
                break

    return {"head": head, "relation": rel, "known_tails": known,
            "forbidden": forbidden, "forbidden_tried": forbidden_canonical,
            "contains_forbidden": contains_forbidden,
            "is_empty": len(known) == 0}


def exec_path_plan(args: dict, kg: KGIndex, aliases: dict) -> dict:
    """Walk relation_chain from primary_entity. Returns final entity set + path trace."""
    head = args.get("primary_entity", "")
    chain = args.get("relation_chain") or []
    if not (head and chain):
        return {"path_trace": [], "final": [], "note": "missing args"}

    current = {head}
    trace = []
    for rel in chain:
        next_set = set()
        step_edges = []
        for h in current:
            tails = kg.tails_of(h, rel)
            for t in tails:
                step_edges.append((h, rel, t))
                next_set.add(t)
        trace.append({"relation": rel, "edges": step_edges, "in": sorted(current), "out": sorted(next_set)})
        if not next_set:
            return {"path_trace": trace, "final": [], "broken_at": rel}
        current = next_set
    return {"path_trace": trace, "final": sorted(current)}


# Semantically-equivalent relation classes for fallback when KG sub-graphs
# are disjoint due to extraction noise. Asking "X 国的雷达" should consider
# any of these relations pointing to a Country.
RELATION_EQUIV_CLASSES = [
    {"countryOfOrigin", "operatedBy"},   # both point Radar/RadarSystem -> Country
]


def _heads_with_union(kg: KGIndex, relation: str, tail_candidates: list) -> tuple[set, str]:
    """Try each tail candidate, return first non-empty result and the tail used."""
    for tc in tail_candidates:
        s = kg.heads_with(relation, tc)
        if s:
            return s, tc
    return set(), tail_candidates[0] if tail_candidates else ""


def _heads_for_constraint(c: dict, kg: KGIndex, aliases: dict) -> tuple[set, dict]:
    """Resolve a single constraint, with relation-equivalence fallback if empty.
    Returns (heads, info_dict)."""
    rel = c["relation"]
    tcs = _alias_resolve_tail(c["tail"], aliases)
    heads, used_tail = _heads_with_union(kg, rel, tcs)
    used_rel = rel
    fallback = None
    if not heads:
        # try equivalence class
        for cls in RELATION_EQUIV_CLASSES:
            if rel in cls:
                for alt in cls - {rel}:
                    h, t = _heads_with_union(kg, alt, tcs)
                    if h:
                        # union from alt
                        heads |= h
                        fallback = alt
                        used_rel = f"{rel}∪{alt}"
                        used_tail = t
                        break
                if heads:
                    break
    return heads, {"relation": used_rel, "tail": used_tail, "n": len(heads),
                   "sample": sorted(heads)[:5], "fallback_to": fallback}


def _resolve_constraint_with_equiv(c: dict, kg: KGIndex, aliases: dict) -> tuple[set, str]:
    """Return UNION over equivalence-class members for a constraint's relation.
    Used when per-constraint sets are individually non-empty but intersection is empty."""
    rel = c["relation"]
    tcs = _alias_resolve_tail(c["tail"], aliases)
    union = set()
    rels_used = [rel]
    for cls in RELATION_EQUIV_CLASSES:
        if rel in cls:
            for member in cls:
                for tc in tcs:
                    union |= kg.heads_with(member, tc)
                if member not in rels_used:
                    rels_used.append(member)
            return union, "∪".join(rels_used)
    # not in any equivalence class — just return the original
    s, info = _heads_for_constraint(c, kg, aliases)
    return s, info["relation"]


def exec_constrained_join(args: dict, kg: KGIndex, aliases: dict) -> dict:
    cons = args.get("constraints") or []
    if len(cons) < 2:
        return {"intersection": [], "note": f"need >=2 constraints, got {len(cons)}"}
    sets = []
    breakdown = []
    for c in cons:
        s, info = _heads_for_constraint(c, kg, aliases)
        sets.append(s)
        breakdown.append(info)
    inter = set.intersection(*sets) if sets else set()

    # Equivalence-class fallback: if intersection is empty but each constraint
    # individually has hits, try expanding constraints whose relation is in an
    # equivalence class (e.g., countryOfOrigin → countryOfOrigin ∪ operatedBy).
    fallback_used = False
    if not BILINGUAL_DISABLED and not inter and all(len(s) > 0 for s in sets):
        sets_v2 = []
        breakdown_v2 = []
        for c in cons:
            s2, used_rel = _resolve_constraint_with_equiv(c, kg, aliases)
            sets_v2.append(s2)
            breakdown_v2.append({"relation": used_rel, "tail": c["tail"], "n": len(s2),
                                 "sample": sorted(s2)[:5], "expanded_via_equiv": True})
        inter_v2 = set.intersection(*sets_v2) if sets_v2 else set()
        if inter_v2:
            inter = inter_v2
            breakdown = breakdown_v2
            fallback_used = True

    return {"intersection": sorted(inter), "n": len(inter),
            "breakdown": breakdown, "equiv_fallback_used": fallback_used}


def exec_exhaustive(args, kg, aliases):
    """Override prior version to use the equivalence-aware single-constraint resolver."""
    cons = args.get("constraints") or []
    if not cons:
        return {"heads": [], "note": "no constraints parsed"}
    s, info = _heads_for_constraint(cons[0], kg, aliases)
    return {"heads": sorted(s), "relation": info["relation"], "tail": info["tail"],
            "tail_tried": _alias_resolve_tail(cons[0]["tail"], aliases),
            "fallback_to": info.get("fallback_to"),
            "n": len(s)}


def exec_dual_subgraph(args: dict, kg: KGIndex, aliases: dict) -> dict:
    a = args.get("primary_entity", "")
    b = args.get("secondary_entity", "")
    rels = args.get("relation_chain") or []
    if not (a and b and rels):
        return {"a_tails": {}, "b_tails": {}, "note": "missing args"}
    a_tails = {r: sorted(kg.tails_of(a, r)) for r in rels}
    b_tails = {r: sorted(kg.tails_of(b, r)) for r in rels}
    same: dict = {}
    diff_a: dict = {}
    diff_b: dict = {}
    for r in rels:
        sa, sb = set(a_tails[r]), set(b_tails[r])
        same[r] = sorted(sa & sb)
        diff_a[r] = sorted(sa - sb)
        diff_b[r] = sorted(sb - sa)
    return {"entity_a": a, "entity_b": b, "a_tails": a_tails, "b_tails": b_tails,
            "same": same, "diff_a": diff_a, "diff_b": diff_b}


def exec_lookup(args: dict, kg: KGIndex, aliases: dict, retriever, question: str) -> dict:
    """Standard hybrid RAG retrieval, augmented with a direct (head, rel) KG
    lookup using parser-extracted args. The KG fallback covers the case where
    BM25/vector miss radar names with special characters (MM/SPQ-2, AN/X(V)Y)
    and ensures single_hop questions always have ground-truth evidence when KG
    actually contains the triple."""
    res = retriever.retrieve(question, top_k=8, use_graph_expansion=True, use_reranker=False)
    reranked = res.get("reranked", [])
    context = res.get("context", "")

    # KG fallback: if parser extracted (head, relation), append the ground-truth
    # triples to the context so the LLM has the canonical facts.
    head = args.get("primary_entity", "")
    rels = args.get("relation_chain") or []
    cons = args.get("constraints") or []
    target_rel = (rels[0] if rels else (cons[0]["relation"] if cons else ""))

    kg_facts = []
    if head and target_rel:
        tails = kg.tails_of(head, target_rel)
        for t in tails:
            kg_facts.append((head, target_rel, t))

    if kg_facts:
        kg_block_lines = ["\n【KG 直接查询补充】"]
        for h, r, t in kg_facts[:10]:
            kg_block_lines.append(f"  - {h} -[{r}]-> {t}")
        context = context + "\n" + "\n".join(kg_block_lines)

    return {"reranked": reranked, "context": context, "kg_fallback_facts": kg_facts}


# ─────────────────────────────────────────────────────────────
#  Context rendering
# ─────────────────────────────────────────────────────────────

def render_context(strategy: str, evidence: dict, args: dict) -> str:
    if strategy == "exhaustive":
        rel = evidence.get("relation", "?")
        tail = evidence.get("tail", "?")
        heads = evidence.get("heads", [])
        n = len(heads)
        sample = heads if n <= 60 else heads[:60] + [f"... 共 {n} 项"]
        return (f"【穷举检索结果】\n"
                f"约束：(?, {rel}, {tail})  共匹配 {n} 个实体\n"
                f"完整列表：\n" + "\n".join(f"  - {h}" for h in sample))

    if strategy == "complement":
        head = evidence.get("head", "?")
        rel = evidence.get("relation", "?")
        tails = evidence.get("known_tails", [])
        forbidden = evidence.get("forbidden", "")
        contains = evidence.get("contains_forbidden", False)
        is_empty = evidence.get("is_empty", False)
        lines = [f"【补全检验结果】",
                 f"主实体：{head}    关系：{rel}",
                 f"KG 中已知的所有 {rel} 取值：" + ("（无任何记录）" if is_empty else "")]
        for t in tails:
            lines.append(f"  - {t}")
        if forbidden:
            lines.append(f"\n被询问对象：{forbidden}")
            lines.append(f"该对象是否在已知集合中：{'是' if contains else '否'}")
        return "\n".join(lines)

    if strategy == "path_plan":
        trace = evidence.get("path_trace", [])
        final = evidence.get("final", [])
        lines = ["【路径执行轨迹】"]
        for i, step in enumerate(trace, 1):
            lines.append(f"步骤{i}：关系 {step['relation']}")
            edges = step.get("edges", [])
            for h, r, t in edges[:8]:
                lines.append(f"  {h} -[{r}]-> {t}")
            if len(edges) > 8:
                lines.append(f"  ...（共 {len(edges)} 条边）")
        lines.append(f"\n终态实体集（{len(final)}）：")
        for e in final[:30]:
            lines.append(f"  - {e}")
        return "\n".join(lines)

    if strategy == "constrained_join":
        inter = evidence.get("intersection", [])
        lines = ["【多约束交集结果】"]
        for b in evidence.get("breakdown", []):
            lines.append(f"约束 ({b['relation']}={b['tail']})：{b['n']} 个匹配，例如 {b['sample']}")
        lines.append(f"\n交集（{len(inter)}）：")
        for e in inter[:30]:
            lines.append(f"  - {e}")
        return "\n".join(lines)

    if strategy == "dual_subgraph":
        a, b = evidence.get("entity_a", "?"), evidence.get("entity_b", "?")
        lines = [f"【对比检索结果】实体A={a} vs 实体B={b}"]
        for r in evidence.get("a_tails", {}):
            lines.append(f"\n关系 {r}:")
            lines.append(f"  A 的取值：{evidence['a_tails'][r]}")
            lines.append(f"  B 的取值：{evidence['b_tails'][r]}")
            lines.append(f"  共同：{evidence['same'][r]}    A独有：{evidence['diff_a'][r]}    B独有：{evidence['diff_b'][r]}")
        return "\n".join(lines)

    if strategy == "lookup":
        return evidence.get("context", "")

    return f"[未知策略 {strategy}]"


# ─────────────────────────────────────────────────────────────
#  Final answer LLM
# ─────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """你是雷达知识图谱问答系统的回答生成器。基于给定的检索证据，用中文回答问题。

严格规则：
1. 答案必须基于证据；证据为空或不足时，明确说"未知"或"否（KG中无相关记录）"。
2. 不要编造不在证据里的实体名称。
3. 计数题：只输出一个整数 + 简短说明。
4. 列举题：列出所有匹配实体（基于证据的"完整列表"或"交集"）。
5. 是否题：明确"是"或"否"，再附 1 句证据。
6. 桥接题：给出最终目标实体；若证据展示了完整路径，也指明路径关键节点。
7. 回答尽量简洁，1-3 句。
"""


def answer_with_llm(question: str, strategy: str, context: str) -> str:
    user = (f"问题：{question}\n\n"
            f"使用的检索策略：{strategy}\n\n"
            f"检索证据：\n{context}\n\n"
            f"请给出答案：")
    # enum answers can be long (44+ entities); give 1200 tokens for safety
    return llm_call([{"role": "system", "content": ANSWER_SYSTEM},
                     {"role": "user",   "content": user}], max_tokens=1200)


# ─────────────────────────────────────────────────────────────
#  Pipeline orchestrator
# ─────────────────────────────────────────────────────────────

class StrategyPipeline:
    def __init__(self, triples_path: Optional[str] = None, lookup_retriever=None):
        if triples_path is None:
            triples_path = ROOT / "graphrag_index" / "merged_triples.json"
        with open(triples_path, encoding="utf-8") as f:
            self.triples = json.load(f)
        self.kg = KGIndex(self.triples)
        self.aliases = build_alias_index()
        self.router = QuestionRouter()
        self.lookup_retriever = lookup_retriever  # lazy: only built if needed

    def _ensure_lookup(self):
        if self.lookup_retriever is None:
            from graphrag_retriever import HybridRetriever
            self.lookup_retriever = HybridRetriever(self.triples, enable_reranker=False)

    def run(self, question: str, ablate_strategy: str | None = None) -> dict:
        """If ablate_strategy is provided and matches the router's pick, force
        the executor to use `lookup` instead. Used for per-strategy ablation."""
        # 1. route
        qtype, strategy, route_raw = self.router.route(question)
        original_strategy = strategy
        ablated = False
        if ablate_strategy and strategy == ablate_strategy:
            strategy = "lookup"
            ablated = True

        # 2. parse args (with type-validation against alias index + KG)
        args = parse_args(question, qtype, strategy, aliases=self.aliases, kg=self.kg)

        # 3. execute
        if strategy == "exhaustive":
            evidence = exec_exhaustive(args, self.kg, self.aliases)
        elif strategy == "complement":
            evidence = exec_complement(args, self.kg, self.aliases)
        elif strategy == "path_plan":
            evidence = exec_path_plan(args, self.kg, self.aliases)
        elif strategy == "constrained_join":
            evidence = exec_constrained_join(args, self.kg, self.aliases)
        elif strategy == "dual_subgraph":
            evidence = exec_dual_subgraph(args, self.kg, self.aliases)
        else:  # lookup
            self._ensure_lookup()
            evidence = exec_lookup(args, self.kg, self.aliases, self.lookup_retriever, question)

        # 4. render
        ctx = render_context(strategy, evidence, args)

        # 5. answer
        answer = answer_with_llm(question, strategy, ctx)

        return {
            "question": question,
            "qtype": qtype,
            "strategy": strategy,
            "original_strategy": original_strategy,
            "ablated": ablated,
            "args": args,
            "evidence": evidence,
            "context": ctx,
            "answer": answer,
        }


if __name__ == "__main__":
    pipe = StrategyPipeline()
    samples = [
        "美国一共使用了多少款雷达？",
        "AN/TPY-2 的研制公司属于哪个国家？",
        "AN/TPY-2 是否被中国使用？",
        "由 Raytheon 研制且部署在战斗机上的雷达？",
        "AN/TPY-2 和 AN/MPQ-65 是同一家公司研制的吗？",
    ]
    for q in samples:
        out = pipe.run(q)
        print(f"\n──────── {q}")
        print(f"  qtype={out['qtype']}  strategy={out['strategy']}")
        print(f"  args={out['args']}")
        print(f"  ANSWER: {out['answer']}")
