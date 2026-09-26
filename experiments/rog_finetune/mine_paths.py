#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine (question, relation-path-plan) training pairs from RadarKG structure for
fine-tuning a RoG-style path planner. INDEPENDENT of the 499 eval questions:
questions are sampled from KG triples with templated phrasings, and any question
that string-matches a 499-eval question is dropped.

Output plan format matches qa_rog_baseline.py exactly:
    START: <entity>
    <PATH>r1<SEP>r2</PATH>

Usage:
    python mine_paths.py --root . --out train_data.jsonl --seed 42
"""
import os, json, argparse, random, re
from collections import defaultdict

# --- relation -> (question phrasing builders). We use zh_label from relations.json. ---
# Forward 1-hop: ask the tail of (head, rel, ?)
FWD_TEMPLATES = [
    "{h}的{zh}是什么？",
    "{h}的{zh}是哪个？",
    "请问{h}的{zh}？",
]
# Inverse 1-hop (enumerate): ask which heads have (?, rel, tail)
INV_TEMPLATES = [
    "哪些雷达的{zh}是{t}？",
    "{t}对应的雷达有哪些？",
]
# Aggregation count/enum over inverse
AGG_TEMPLATES = [
    "{t}一共有多少款雷达？",
    "{t}的雷达共有几款？",
    "列出{t}的所有雷达。",
    "{t}都有哪些雷达？",
]
# Operator-specific natural inverse phrasings (more fluent than generic)
INV_NATURAL = {
    "operatedBy":     ["{t}使用了哪些雷达？", "{t}装备了哪些雷达？"],
    "developedBy":    ["{t}研制了哪些雷达？", "{t}生产了哪些雷达？"],
    "countryOfOrigin":["原产国是{t}的雷达有哪些？"],
    "hasFrequencyBand":["工作在{t}波段的雷达有哪些？"],
}
AGG_NATURAL = {
    "operatedBy":     ["{t}一共使用了多少款雷达？", "{t}装备了多少款雷达？"],
    "developedBy":    ["{t}一共研制了多少款雷达？", "{t}生产了多少型雷达？"],
    "countryOfOrigin":["{t}一共原产多少款雷达？"],
    "hasFrequencyBand":["工作在{t}波段的雷达共有多少款？"],
}

# Relations usable for forward single-hop questions (head is a specific radar)
FWD_RELS = ["developedBy", "countryOfOrigin", "hasFrequencyBand", "deployedOn",
            "operatedBy", "hasMode", "hasFunction", "hasTechType"]
# Relations usable for inverse enumeration / aggregation (tail is an aggregator)
INV_RELS = ["operatedBy", "developedBy", "countryOfOrigin", "hasFrequencyBand"]
# 2-hop bridges: (r1 from radar) then (r2 from the middle entity)
BRIDGES = [
    ("developedBy", "affiliatedTo"),   # radar -> manufacturer -> country
]
BRIDGE_TEMPLATES = [
    "{h}的{zh1}的{zh2}是什么？",
    "{h}的{zh1}属于哪个{zh2tail}？",
]


def load(root):
    triples = json.load(open(os.path.join(root, "graphrag_index", "merged_triples.json"), encoding="utf-8"))
    relations = json.load(open(os.path.join(root, "lexicon", "relations.json"), encoding="utf-8"))["relations"]
    return triples, relations


def zh_of(relations, rid):
    rec = relations.get(rid, {})
    return rec.get("zh_label") or rid


def build_indexes(triples):
    by_hr = defaultdict(set)   # (head, rel) -> tails
    by_rt = defaultdict(set)   # (rel, tail) -> heads
    by_h = defaultdict(list)   # head -> [(rel, tail)]
    radars = set()
    for t in triples:
        h, r, ta = t["head"], t["relation"], t["tail"]
        ht = t.get("head_type", "")
        by_hr[(h, r)].add(ta)
        by_rt[(r, ta)].add(h)
        by_h[h].append((r, ta))
        if ht in ("Radar", "RadarSystem"):
            radars.add(h)
    return by_hr, by_rt, by_h, radars


def load_eval_questions(root):
    """The 499 eval questions — to exclude from training (no contamination)."""
    p = os.path.join(root, "results", "qa500_3way_full.json")
    if not os.path.exists(p):
        return set()
    data = json.load(open(p, encoding="utf-8"))["per_question"]
    return set(q["question"].strip() for q in data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="train_data.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    triples, relations = load(args.root)
    by_hr, by_rt, by_h, radars = build_indexes(triples)
    eval_qs = load_eval_questions(args.root)
    print(f"[*] KG: {len(triples)} triples, {len(radars)} radar entities, {len(eval_qs)} eval questions to exclude")

    examples = []
    seen_q = set()

    def add(question, start, paths, category):
        q = question.strip()
        if q in eval_qs or q in seen_q:
            return False
        seen_q.add(q)
        plan_lines = [f"START: {start}"]
        for p in paths:
            plan_lines.append("<PATH>" + "<SEP>".join(p) + "</PATH>")
        examples.append({"question": q, "plan": "\n".join(plan_lines), "category": category})
        return True

    # ---- 1. forward single-hop ----
    fwd_target = 70
    tries = 0
    fwd_pairs = [(h, r) for (h, r) in by_hr.keys() if r in FWD_RELS and h in radars]
    random.shuffle(fwd_pairs)
    n = 0
    for (h, r) in fwd_pairs:
        if n >= fwd_target:
            break
        zh = zh_of(relations, r)
        tmpl = random.choice(FWD_TEMPLATES)
        if add(tmpl.format(h=h, zh=zh), h, [[r]], "single_hop_fwd"):
            n += 1
    print(f"[*] forward single-hop: {n}")

    # ---- 2. inverse enumeration ----
    inv_target = 55
    inv_pairs = [(r, t) for (r, t) in by_rt.keys() if r in INV_RELS and len(by_rt[(r, t)]) >= 3]
    random.shuffle(inv_pairs)
    n = 0
    for (r, t) in inv_pairs:
        if n >= inv_target:
            break
        zh = zh_of(relations, r)
        tmpls = INV_NATURAL.get(r, []) + INV_TEMPLATES
        tmpl = random.choice(tmpls)
        if add(tmpl.format(t=t, zh=zh), t, [[r + "^-1"]], "relation_inverse"):
            n += 1
    print(f"[*] inverse enumeration: {n}")

    # ---- 3. aggregation count/enum ----
    agg_target = 65
    random.shuffle(inv_pairs)
    n = 0
    for (r, t) in inv_pairs:
        if n >= agg_target:
            break
        zh = zh_of(relations, r)
        tmpls = AGG_NATURAL.get(r, []) + AGG_TEMPLATES
        tmpl = random.choice(tmpls)
        if add(tmpl.format(t=t, zh=zh), t, [[r + "^-1"]], "aggregation"):
            n += 1
    print(f"[*] aggregation: {n}")

    # ---- 4. two-hop bridges ----
    bridge_target = 50
    n = 0
    for (r1, r2) in BRIDGES:
        zh1 = zh_of(relations, r1)
        zh2 = zh_of(relations, r2)
        zh2tail = "国家" if r2 == "affiliatedTo" else zh2
        bridge_heads = [h for h in radars if by_hr.get((h, r1))]
        random.shuffle(bridge_heads)
        for h in bridge_heads:
            if n >= bridge_target:
                break
            mids = by_hr[(h, r1)]
            # require the middle entity to actually have r2
            if not any(by_hr.get((m, r2)) for m in mids):
                continue
            tmpl = random.choice(BRIDGE_TEMPLATES)
            q = tmpl.format(h=h, zh1=zh1, zh2=zh2, zh2tail=zh2tail)
            if add(q, h, [[r1, r2]], "two_hop_bridge"):
                n += 1
    print(f"[*] two-hop bridge: {n}")

    random.shuffle(examples)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    cats = Counter(e["category"] for e in examples)
    print(f"\n[+] wrote {len(examples)} examples to {args.out}")
    print(f"[+] by category: {dict(cats)}")
    print(f"\n=== 8 random samples ===")
    for ex in random.sample(examples, min(8, len(examples))):
        print(f"\n[{ex['category']}] Q: {ex['question']}")
        print(f"  PLAN: {ex['plan'].replace(chr(10), ' || ')}")


if __name__ == "__main__":
    main()
