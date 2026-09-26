"""
Generate a 500-question stratified KGQA benchmark from the 5610-triple RadarKG.

Output: evaluation/qa_500.json  with per-question fields compatible with
        evaluation/validation_30.json (same scoring infrastructure).

Stratification target (~500 total):
  single_hop        80
  relation_inverse  50
  agg_count         50
  agg_enum          50
  two_hop_bridge    80
  three_hop_chain   30
  attr_filter       50
  negation          40
  set_compare       40
  unanswerable      20
  distractor        10

Garbage filtering removes:
  - self-loops (head == tail)
  - heads with non-ASCII-friendly characters in Radar/RadarSystem
  - tails with line breaks, sentence fragments, or excessive length
  - extraction artifacts ("the same company", "which had previously", ...)

All gold answers are derived from the cleaned KG so the validator can verify them.
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexicon import load_relations, load_entity_aliases, build_alias_index

random.seed(42)


# ─────────────────────────────────────────────────────────────
#  KG cleaning
# ─────────────────────────────────────────────────────────────

RADAR_NAME_OK = re.compile(r"^[A-Za-z0-9/\-\(\)\.\s]{2,60}$")
EXTRACTION_GARBAGE = {
    "the same company", "which had previously", "the U", "the company",
    "no", "n/a", "none", "unknown",
}


def is_junk_head(h: str, head_type: str) -> bool:
    if not h or len(h) > 60:
        return True
    if h.strip().lower() in EXTRACTION_GARBAGE:
        return True
    if head_type in ("Radar", "RadarSystem"):
        return not RADAR_NAME_OK.match(h.strip())
    return False


def is_junk_tail(tail: str, tail_type: str) -> bool:
    if not tail or len(tail) > 60:
        return True
    if tail.strip().lower() in EXTRACTION_GARBAGE:
        return True
    if "\n" in tail or "\t" in tail:
        return True
    if tail_type == "FrequencyBand" and len(tail) > 8:
        return True
    if tail_type != "Literal" and len(tail) > 25 and any(c in tail for c in ["（", "。", "，"]):
        return True
    return False


def load_clean_kg() -> tuple[list, dict]:
    with open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8") as f:
        all_triples = json.load(f)
    clean = []
    for t in all_triples:
        if t["head"] == t["tail"]:
            continue
        if is_junk_head(t["head"], t.get("head_type", "")):
            continue
        if is_junk_tail(t["tail"], t.get("tail_type", "")):
            continue
        clean.append(t)

    indices = {
        "by_rel": defaultdict(list),       # relation -> list of triples
        "by_rel_tail": defaultdict(set),   # (rel, tail) -> set of heads
        "by_head_rel": defaultdict(set),   # (head, rel) -> set of tails
        "ent_type": {},                    # entity name -> type
    }
    for t in clean:
        h, r, ta = t["head"], t["relation"], t["tail"]
        indices["by_rel"][r].append(t)
        indices["by_rel_tail"][(r, ta)].add(h)
        indices["by_head_rel"][(h, r)].add(ta)
        indices["ent_type"][h] = t.get("head_type", "")
        if t.get("tail_type"):
            indices["ent_type"][ta] = t.get("tail_type", "")
    return clean, indices


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

REL_LEX = load_relations()["relations"]
ENT_ALIAS = load_entity_aliases()
ALIAS_IDX = build_alias_index()


def canonicalize_tail(tail: str, idx: dict) -> str:
    """Map a tail to its canonical surface form (e.g., 'People's Republic of China'
    → '中国') so that gold counts align with the alias-aware retrieval. Falls back
    to the original tail if no alias is known."""
    if not tail:
        return tail
    hit = ALIAS_IDX.get(tail) or ALIAS_IDX.get(tail.lower())
    if hit and hit.get("canonical") and hit["canonical"] != tail:
        return hit["canonical"]
    return tail


def merge_alias_heads(rel: str, tail: str, idx: dict) -> set:
    """Return UNION of heads across all surface-form aliases of `tail` for relation
    `rel`. e.g., for tail='中国', also includes heads from tail='People's Republic of China'."""
    canonical = canonicalize_tail(tail, idx)
    aliases = {canonical, tail}
    # add all forms that map to the same canonical
    for surface, rec in ALIAS_IDX.items():
        if rec.get("canonical") == canonical:
            aliases.add(surface)
    union = set()
    for a in aliases:
        union |= idx["by_rel_tail"].get((rel, a), set())
    return union


def en_for_country(zh: str) -> str:
    rec = ENT_ALIAS["countries"].get(zh, {})
    return rec.get("en", zh)


def en_for_tail(tail: str, tail_type: str) -> str:
    if tail_type == "Country":
        return en_for_country(tail)
    if tail_type == "RadarMode":
        return ENT_ALIAS["radar_modes"].get(tail, {}).get("en", tail)
    if tail_type == "TechType":
        return ENT_ALIAS["tech_types"].get(tail, {}).get("en", tail)
    return tail  # English-primary


def render_triple_zh(rel: str, head: str, tail: str) -> str:
    tpl = REL_LEX.get(rel, {}).get("triple_text_zh", "{head} {rel} {tail}")
    return tpl.format(head=head, tail=tail)


def render_triple_en(rel: str, head: str, tail_en: str) -> str:
    tpl = REL_LEX.get(rel, {}).get("triple_text_en", "{head} {rel} {tail}")
    return tpl.format(head=head, tail=tail_en)


def pick_question_template_zh(rel: str, head_only: bool = True) -> str:
    """Pick a question template. If head_only, exclude templates that need {tail}."""
    tpls = REL_LEX.get(rel, {}).get("question_templates_zh", [])
    if head_only:
        tpls = [t for t in tpls if "{tail}" not in t]
    if not tpls:
        spec = REL_LEX.get(rel, {})
        zh = spec.get("zh_label", rel)
        return f"{{head}} 的{zh}是？"
    return random.choice(tpls)


# ─────────────────────────────────────────────────────────────
#  Question generators (one per type)
# ─────────────────────────────────────────────────────────────

def gen_single_hop(idx: dict, n: int = 80) -> list:
    """Pick (head, rel, tail) where head has a UNIQUE tail for this rel."""
    out = []
    rel_pool = [r for r in idx["by_rel"].keys()
                if r not in ("similarTo",) and len(idx["by_rel"][r]) >= 5]
    used = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        rel = random.choice(rel_pool)
        triple = random.choice(idx["by_rel"][rel])
        head, tail = triple["head"], triple["tail"]
        if (head, rel) in used:
            continue
        # require uniqueness for clean single-answer
        if len(idx["by_head_rel"][(head, rel)]) != 1:
            continue
        used.add((head, rel))
        spec = REL_LEX.get(rel, {})
        tpl_zh = pick_question_template_zh(rel)
        q_zh = tpl_zh.format(head=head)
        q_en = f"What is the {spec.get('en_label', rel)} of {head}?"
        tail_en = en_for_tail(tail, triple.get("tail_type", ""))
        out.append({
            "id": f"sh_{len(out)+1:03d}",
            "type": "single_hop",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_answer": tail,
            "gold_answer_en": tail_en,
            "gold_constraint": {"head": head, "relation": rel},
            "expected_strategy": "lookup",
        })
    return out


def gen_relation_inverse(idx: dict, n: int = 50) -> list:
    """For (rel, tail) with many heads, ask 'which radars X-rel tail?'.
    Uses canonical alias + merged head set to avoid surface-form fragmentation."""
    out = []
    candidates = []
    seen_canonical = set()
    for (rel, tail), heads in idx["by_rel_tail"].items():
        if len(heads) >= 3 and rel != "similarTo":
            tail_type = idx["ent_type"].get(tail, "")
            if tail_type in ("Country", "Manufacturer", "FrequencyBand", "Platform",
                             "TechType", "RadarMode", "Function"):
                canonical = canonicalize_tail(tail, idx)
                key = (rel, canonical)
                if key in seen_canonical:
                    continue
                seen_canonical.add(key)
                merged = merge_alias_heads(rel, canonical, idx)
                candidates.append((rel, canonical, len(merged)))
    random.shuffle(candidates)
    for rel, tail, _ in candidates:
        if len(out) >= n:
            break
        spec = REL_LEX.get(rel, {})
        zh_label = spec.get("zh_label", rel)
        q_zh = f"哪些雷达由 {tail} {zh_label}的关系？" if False else (
            "由 {tail} {label} 的雷达有哪些？".format(tail=tail, label=zh_label)
        )
        # cleaner per-relation phrasing
        if rel == "developedBy":
            q_zh = f"{tail} 研制了哪些雷达？"
            q_en = f"Which radars are developed by {tail}?"
        elif rel == "operatedBy":
            q_zh = f"{tail} 装备了哪些雷达？"
            q_en = f"Which radars are operated by {en_for_country(tail)}?"
        elif rel == "hasFrequencyBand":
            q_zh = f"工作在 {tail} 波段的雷达有哪些？"
            q_en = f"Which radars operate in the {tail} band?"
        elif rel == "deployedOn":
            q_zh = f"部署在{tail}上的雷达有哪些？"
            q_en = f"Which radars are deployed on {tail}?"
        elif rel == "hasTechType":
            q_zh = f"采用 {tail} 技术的雷达有哪些？"
            q_en = f"Which radars use {tail} technology?"
        elif rel == "hasMode":
            q_zh = f"支持 {tail} 模式的雷达有哪些？"
            q_en = f"Which radars support {tail} mode?"
        elif rel == "exportedTo":
            q_zh = f"出口到 {tail} 的雷达有哪些？"
            q_en = f"Which radars are exported to {en_for_country(tail)}?"
        elif rel == "countryOfOrigin":
            q_zh = f"{tail} 原产的雷达有哪些？"
            q_en = f"Which radars originate from {en_for_country(tail)}?"
        elif rel == "compatibleWith":
            q_zh = f"兼容 {tail} 武器的雷达有哪些？"
            q_en = f"Which radars are compatible with {tail}?"
        elif rel == "hasFunction":
            q_zh = f"具备 {tail} 功能的雷达有哪些？"
            q_en = f"Which radars have the {tail} function?"
        else:
            q_en = f"Which radars stand in {spec.get('en_label', rel)} relation with {tail}?"

        out.append({
            "id": f"ri_{len(out)+1:03d}",
            "type": "relation_inverse",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_constraint": {"relation": rel, "tail": tail},
            "gold_answer_size": len(merge_alias_heads(rel, tail, idx)),
            "expected_strategy": "exhaustive",
        })
    return out


def gen_agg_count(idx: dict, n: int = 50) -> list:
    out = []
    candidates = []
    seen_canonical = set()
    for (rel, tail), heads in idx["by_rel_tail"].items():
        if len(heads) >= 5 and rel != "similarTo":
            tail_type = idx["ent_type"].get(tail, "")
            if tail_type in ("Country", "Manufacturer", "FrequencyBand", "Platform",
                             "TechType", "RadarMode"):
                canonical = canonicalize_tail(tail, idx)
                key = (rel, canonical)
                if key in seen_canonical:
                    continue
                seen_canonical.add(key)
                merged_count = len(merge_alias_heads(rel, canonical, idx))
                if merged_count >= 5:
                    candidates.append((rel, canonical, merged_count))
    random.shuffle(candidates)
    for rel, tail, count in candidates:
        if len(out) >= n:
            break
        if rel == "developedBy":
            q_zh = f"{tail} 一共研制了多少款雷达？"
            q_en = f"How many radars has {tail} developed in total?"
        elif rel == "operatedBy":
            q_zh = f"{tail} 一共装备了多少款雷达？"
            q_en = f"How many radars are operated by {en_for_country(tail)} in total?"
        elif rel == "hasFrequencyBand":
            q_zh = f"工作在 {tail} 波段的雷达共有多少款？"
            q_en = f"How many radars operate in the {tail} band?"
        elif rel == "deployedOn":
            q_zh = f"部署在{tail}上的雷达共有多少款？"
            q_en = f"How many radars are deployed on {tail}?"
        elif rel == "hasTechType":
            q_zh = f"采用 {tail} 技术的雷达共有多少款？"
            q_en = f"How many radars use {tail} technology?"
        elif rel == "hasMode":
            q_zh = f"支持 {tail} 模式的雷达共有多少款？"
            q_en = f"How many radars support {tail} mode?"
        elif rel == "countryOfOrigin":
            q_zh = f"原产于 {tail} 的雷达共有多少款？"
            q_en = f"How many radars originate from {en_for_country(tail)}?"
        elif rel == "exportedTo":
            q_zh = f"出口到 {tail} 的雷达共有多少款？"
            q_en = f"How many radars are exported to {en_for_country(tail)}?"
        else:
            continue

        out.append({
            "id": f"ac_{len(out)+1:03d}",
            "type": "agg_count",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_answer": count,
            "gold_constraint": {"relation": rel, "tail": tail},
            "expected_strategy": "exhaustive",
        })
    return out


def gen_agg_enum(idx: dict, n: int = 50) -> list:
    out = []
    candidates = []
    seen_canonical = set()
    for (rel, tail), heads in idx["by_rel_tail"].items():
        # 5..50 for manageable enumeration
        if 5 <= len(heads) <= 50 and rel != "similarTo":
            tail_type = idx["ent_type"].get(tail, "")
            if tail_type in ("Country", "Manufacturer", "FrequencyBand", "Platform",
                             "TechType", "RadarMode"):
                canonical = canonicalize_tail(tail, idx)
                key = (rel, canonical)
                if key in seen_canonical:
                    continue
                seen_canonical.add(key)
                merged = merge_alias_heads(rel, canonical, idx)
                if 5 <= len(merged) <= 60:
                    candidates.append((rel, canonical, len(merged)))
    random.shuffle(candidates)
    for rel, tail, count in candidates:
        if len(out) >= n:
            break
        if rel == "developedBy":
            q_zh = f"请列出 {tail} 研制的所有雷达。"
            q_en = f"List all radars developed by {tail}."
        elif rel == "operatedBy":
            q_zh = f"请列出 {tail} 装备的所有雷达。"
            q_en = f"List all radars operated by {en_for_country(tail)}."
        elif rel == "hasFrequencyBand":
            q_zh = f"列出工作在 {tail} 波段的所有雷达。"
            q_en = f"List all radars operating in the {tail} band."
        elif rel == "deployedOn":
            q_zh = f"列出所有部署在{tail}上的雷达。"
            q_en = f"List all radars deployed on {tail}."
        elif rel == "hasTechType":
            q_zh = f"列出所有采用 {tail} 技术的雷达。"
            q_en = f"List all radars using {tail} technology."
        elif rel == "hasMode":
            q_zh = f"列出所有支持 {tail} 模式的雷达。"
            q_en = f"List all radars supporting {tail} mode."
        elif rel == "countryOfOrigin":
            q_zh = f"列出原产于 {tail} 的所有雷达。"
            q_en = f"List all radars originating from {en_for_country(tail)}."
        elif rel == "exportedTo":
            q_zh = f"列出出口到 {tail} 的所有雷达。"
            q_en = f"List all radars exported to {en_for_country(tail)}."
        else:
            continue

        gold_list = sorted(merge_alias_heads(rel, tail, idx))
        out.append({
            "id": f"ae_{len(out)+1:03d}",
            "type": "agg_enum",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_answer": gold_list,
            "gold_answer_size": len(gold_list),
            "gold_constraint": {"relation": rel, "tail": tail},
            "expected_strategy": "exhaustive",
        })
    return out


# Common 2-hop patterns we mine
TWO_HOP_PATTERNS = [
    ("developedBy",  "affiliatedTo",     "Manufacturer"),
    ("upgradeOf",    "developedBy",      "Radar"),
    ("upgradeOf",    "countryOfOrigin",  "Radar"),
    ("derivedFrom",  "developedBy",      "Radar"),
]

def gen_two_hop_bridge(idx: dict, n: int = 80) -> list:
    out = []
    chains = []
    for r1, r2, mid_type in TWO_HOP_PATTERNS:
        # head_type signature for r1 — must be Radar/RadarSystem to make the
        # natural-language question phrasing valid (e.g., "AN/TPY-2 的研制公司
        # 属于哪个国家" only makes sense if AN/TPY-2 is a radar)
        r1_head_types = set(REL_LEX.get(r1, {}).get("head_types", []))
        for (head, _r1), mids in idx["by_head_rel"].items():
            if _r1 != r1:
                continue
            head_type = idx["ent_type"].get(head, "")
            if r1_head_types and head_type not in r1_head_types:
                continue
            for m in mids:
                if idx["ent_type"].get(m, "") not in (mid_type, ""):
                    continue
                tails = idx["by_head_rel"].get((m, r2), set())
                for t in tails:
                    chains.append((head, r1, m, r2, t))
    random.shuffle(chains)
    seen = set()
    for h, r1, m, r2, t in chains:
        if len(out) >= n:
            break
        key = (h, r1, r2)
        if key in seen:
            continue
        seen.add(key)
        # skip noisy chains where same path produces multiple targets
        if r1 == "developedBy" and r2 == "affiliatedTo":
            q_zh = f"{h} 的研制公司属于哪个国家？"
            q_en = f"Which country does the company that developed {h} belong to?"
        elif r1 == "upgradeOf" and r2 == "developedBy":
            q_zh = f"{h} 升级自的型号由哪家公司研制？"
            q_en = f"Who developed the predecessor of {h}?"
        elif r1 == "upgradeOf" and r2 == "countryOfOrigin":
            q_zh = f"{h} 升级自的型号原产于哪个国家？"
            q_en = f"Where does the predecessor of {h} originate from?"
        elif r1 == "derivedFrom" and r2 == "developedBy":
            q_zh = f"{h} 衍生自的雷达由谁研制？"
            q_en = f"Who developed the radar from which {h} was derived?"
        else:
            continue

        out.append({
            "id": f"mh2_{len(out)+1:03d}",
            "type": "two_hop_bridge",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_answer": t,
            "gold_path": [f"{h} -[{r1}]-> {m}", f"{m} -[{r2}]-> {t}"],
            "expected_strategy": "path_plan",
        })
    return out


def gen_three_hop_chain(idx: dict, n: int = 30) -> list:
    """upgradeOf → developedBy → affiliatedTo"""
    out = []
    chains = []
    for (h, _r), mids1 in idx["by_head_rel"].items():
        if _r != "upgradeOf": continue
        for m1 in mids1:
            for m2 in idx["by_head_rel"].get((m1, "developedBy"), set()):
                for c in idx["by_head_rel"].get((m2, "affiliatedTo"), set()):
                    chains.append((h, m1, m2, c))
    random.shuffle(chains)
    seen = set()
    for h, m1, m2, c in chains:
        if len(out) >= n:
            break
        if h in seen: continue
        seen.add(h)
        out.append({
            "id": f"mh3_{len(out)+1:03d}",
            "type": "three_hop_chain",
            "question_zh": f"{h} 升级前型号的研制公司属于哪个国家？",
            "question_en": f"Which country does the developer of {h}'s predecessor belong to?",
            "gold_answer": c,
            "gold_path": [f"{h} -[upgradeOf]-> {m1}",
                          f"{m1} -[developedBy]-> {m2}",
                          f"{m2} -[affiliatedTo]-> {c}"],
            "expected_strategy": "path_plan",
        })
    return out


# Pairs of relations that often co-occur on the same head — good for attr_filter
ATTR_PAIRS = [
    ("operatedBy", "hasFrequencyBand"),
    ("operatedBy", "deployedOn"),
    ("operatedBy", "hasTechType"),
    ("developedBy", "deployedOn"),
    ("developedBy", "hasFrequencyBand"),
    ("countryOfOrigin", "hasMode"),
    ("hasFrequencyBand", "deployedOn"),
]

def gen_attr_filter(idx: dict, n: int = 50) -> list:
    out = []
    candidates = []
    seen_canonical = set()
    for r1, r2 in ATTR_PAIRS:
        # find (tail1, tail2) such that intersection of head sets is >=2
        tails1_raw = {canonicalize_tail(t, idx) for (r, t), s in idx["by_rel_tail"].items() if r == r1 and len(s) >= 5}
        tails2_raw = {canonicalize_tail(t, idx) for (r, t), s in idx["by_rel_tail"].items() if r == r2 and len(s) >= 5}
        for t1 in tails1_raw:
            s1 = merge_alias_heads(r1, t1, idx)
            if len(s1) < 5:
                continue
            for t2 in tails2_raw:
                key = (r1, t1, r2, t2)
                if key in seen_canonical:
                    continue
                s2 = merge_alias_heads(r2, t2, idx)
                inter = s1 & s2
                if 2 <= len(inter) <= 30:
                    seen_canonical.add(key)
                    candidates.append((r1, t1, r2, t2, inter))
    random.shuffle(candidates)
    seen = set()
    for r1, t1, r2, t2, inter in candidates:
        if len(out) >= n:
            break
        if (r1, t1, r2, t2) in seen: continue
        seen.add((r1, t1, r2, t2))

        def _phrase(rel, tail):
            if rel == "operatedBy":       return f"{tail} 装备的"
            if rel == "developedBy":      return f"{tail} 研制的"
            if rel == "hasFrequencyBand": return f"工作在 {tail} 波段的"
            if rel == "deployedOn":       return f"部署在{tail}上的"
            if rel == "hasTechType":      return f"采用 {tail} 技术的"
            if rel == "hasMode":          return f"支持 {tail} 模式的"
            if rel == "countryOfOrigin":  return f"原产于 {tail} 的"
            return f"{rel}={tail}"

        q_zh = f"{_phrase(r1, t1)}且{_phrase(r2, t2)}雷达有哪些？"
        q_en = f"Which radars satisfy both {r1}={t1} AND {r2}={t2}?"

        out.append({
            "id": f"af_{len(out)+1:03d}",
            "type": "attr_filter",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_constraints": [
                {"relation": r1, "tail": t1},
                {"relation": r2, "tail": t2},
            ],
            "gold_answer": sorted(inter),
            "gold_answer_size": len(inter),
            "expected_strategy": "constrained_join",
        })
    return out


def gen_negation(idx: dict, n: int = 40) -> list:
    """For (head, rel) with KNOWN tails T, pick forbidden NOT in T."""
    out = []
    samples = []
    rels_for_neg = ["operatedBy", "exportedTo", "developedBy", "deployedOn",
                    "hasFrequencyBand", "compatibleWith", "upgradeOf"]
    for rel in rels_for_neg:
        triples_for_rel = idx["by_rel"][rel]
        # pool of tails seen for this relation (used to pick a credible forbidden)
        all_tails = list({t["tail"] for t in triples_for_rel})
        if len(all_tails) < 5:
            continue
        # pick (head, rel) heads
        heads_with_rel = list({t["head"] for t in triples_for_rel})
        random.shuffle(heads_with_rel)
        for h in heads_with_rel:
            actual = idx["by_head_rel"][(h, rel)]
            # pick a forbidden tail not in actual
            attempts = 0
            while attempts < 5:
                attempts += 1
                forb = random.choice(all_tails)
                if forb not in actual:
                    samples.append((h, rel, forb))
                    break
    random.shuffle(samples)
    for h, rel, forb in samples:
        if len(out) >= n:
            break
        if rel == "operatedBy":
            q_zh = f"{h} 是否被 {forb} 装备使用？"
            q_en = f"Is {h} operated by {en_for_country(forb)}?"
        elif rel == "exportedTo":
            q_zh = f"{h} 是否出口到 {forb}？"
            q_en = f"Is {h} exported to {en_for_country(forb)}?"
        elif rel == "developedBy":
            q_zh = f"{h} 是由 {forb} 研制的吗？"
            q_en = f"Is {h} developed by {forb}?"
        elif rel == "deployedOn":
            q_zh = f"{h} 是否部署在{forb}上？"
            q_en = f"Is {h} deployed on {forb}?"
        elif rel == "hasFrequencyBand":
            q_zh = f"{h} 是否工作在 {forb} 波段？"
            q_en = f"Does {h} operate in the {forb} band?"
        elif rel == "compatibleWith":
            q_zh = f"{h} 是否兼容 {forb} 武器？"
            q_en = f"Is {h} compatible with the {forb} weapon?"
        elif rel == "upgradeOf":
            q_zh = f"{h} 是否升级自 {forb}？"
            q_en = f"Is {h} an upgrade of {forb}?"
        else:
            continue

        out.append({
            "id": f"neg_{len(out)+1:03d}",
            "type": "negation",
            "question_zh": q_zh,
            "question_en": q_en,
            "gold_answer": "否",
            "gold_evidence_check": {"head": h, "relation": rel, "tail_must_not_be": forb},
            "expected_strategy": "complement",
        })
    return out


def gen_set_compare(idx: dict, n: int = 40) -> list:
    """Pick 2 radars and ask same/different on a relation (developer / country)."""
    out = []
    rels = ["developedBy", "countryOfOrigin", "operatedBy"]
    for rel in rels:
        triples = idx["by_rel"][rel]
        heads = list({t["head"] for t in triples})
        random.shuffle(heads)
        for i in range(0, len(heads) - 1, 2):
            if len(out) >= n:
                break
            a, b = heads[i], heads[i+1]
            ta = sorted(idx["by_head_rel"][(a, rel)])
            tb = sorted(idx["by_head_rel"][(b, rel)])
            if not ta or not tb:
                continue
            same = (ta == tb) or bool(set(ta) & set(tb))
            zh_label = REL_LEX[rel].get("zh_label", rel)
            q_zh = f"{a} 和 {b} 的{zh_label}是否相同？"
            q_en = f"Do {a} and {b} have the same {REL_LEX[rel].get('en_label', rel)}?"
            out.append({
                "id": f"sc_{len(out)+1:03d}",
                "type": "set_compare",
                "question_zh": q_zh,
                "question_en": q_en,
                "gold_answer": "是" if same else "否",
                "gold_evidence_check": {"entity_a": a, "entity_b": b, "relation": rel,
                                         "a_tails": ta, "b_tails": tb,
                                         "same": same},
                "expected_strategy": "dual_subgraph",
            })
        if len(out) >= n:
            break
    return out


def gen_unanswerable(idx: dict, n: int = 20) -> list:
    """Real radar, but pick a relation it has NO triples for."""
    out = []
    radars = list({t["head"] for t in idx["by_rel"]["operatedBy"]})  # known radars
    random.shuffle(radars)
    target_rels = ["price", "hasMTBF", "hasPower", "hasECCM", "exportedTo", "compatibleWith"]
    for h in radars:
        if len(out) >= n:
            break
        for rel in target_rels:
            if (h, rel) in idx["by_head_rel"] and idx["by_head_rel"][(h, rel)]:
                continue  # has triples — not unanswerable
            zh_label = REL_LEX[rel].get("zh_label", rel)
            q_zh = f"{h} 的{zh_label}是？"
            q_en = f"What is the {REL_LEX[rel].get('en_label', rel)} of {h}?"
            out.append({
                "id": f"un_{len(out)+1:03d}",
                "type": "unanswerable",
                "question_zh": q_zh,
                "question_en": q_en,
                "gold_answer": "未知（KG中无相关记录）",
                "gold_evidence_check": {"head": h, "relation": rel, "expected": "no_triple"},
                "expected_strategy": "complement",
            })
            break
    return out


def gen_distractor(idx: dict, n: int = 10) -> list:
    """Find similar radar names (e.g., AN/SPY-1 vs AN/SPY-1A) and ask single_hop on the variant."""
    out = []
    radar_names = list({t["head"] for t in idx["by_rel"]["operatedBy"]})
    # group by prefix (first 7 chars or first dash-separated token)
    groups = defaultdict(list)
    for n_ in radar_names:
        n_ = n_.strip()
        m = re.match(r"^([A-Z]{2,4}/[A-Z]{2,4}-\d+)", n_)
        if m:
            groups[m.group(1)].append(n_)
    # pick groups with >= 2 variants
    families = [g for g in groups.values() if len(g) >= 2]
    random.shuffle(families)
    for fam in families:
        if len(out) >= n:
            break
        # pick one with a suffix variant
        fam.sort(key=lambda x: len(x))
        if len(fam) < 2: continue
        target = fam[-1]  # the longer/more specific one
        # find a single_hop fact on target
        for rel in ["operatedBy", "developedBy", "deployedOn", "hasFrequencyBand"]:
            tails = idx["by_head_rel"].get((target, rel), set())
            if len(tails) == 1:
                tail = next(iter(tails))
                spec = REL_LEX.get(rel, {})
                tpl_zh = pick_question_template_zh(rel)
                q_zh = tpl_zh.format(head=target) + f"（注意区分 {fam[0]}）"
                q_en = f"What is the {spec.get('en_label', rel)} of {target}? (distinct from {fam[0]})"
                out.append({
                    "id": f"dt_{len(out)+1:03d}",
                    "type": "distractor",
                    "question_zh": q_zh,
                    "question_en": q_en,
                    "gold_answer": tail,
                    "gold_constraint": {"head": target, "relation": rel},
                    "distractor_pair": [target, fam[0]],
                    "expected_strategy": "lookup",
                })
                break
    return out


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading and cleaning KG...")
    triples, idx = load_clean_kg()
    print(f"  {len(triples)} clean triples")

    questions = []
    print("\nGenerating questions per type:")
    for fn, target in [
        (gen_single_hop,        80),
        (gen_relation_inverse,  50),
        (gen_agg_count,         50),
        (gen_agg_enum,          50),
        (gen_two_hop_bridge,    80),
        (gen_three_hop_chain,   30),
        (gen_attr_filter,       50),
        (gen_negation,          40),
        (gen_set_compare,       40),
        (gen_unanswerable,      20),
        (gen_distractor,        10),
    ]:
        batch = fn(idx, target)
        questions.extend(batch)
        print(f"  {fn.__name__:25s} target={target:3d}  produced={len(batch)}")

    # final stats
    by_type = Counter(q["type"] for q in questions)
    print(f"\nTotal: {len(questions)}")
    for t, c in by_type.most_common():
        print(f"  {t:20s} {c}")

    out = ROOT / "evaluation" / "qa_500.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "version": "0.1-autogen",
            "description": f"500-question stratified KGQA bench, generated from {len(triples)} clean triples (random.seed=42).",
            "stats": dict(by_type),
            "questions": questions,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
