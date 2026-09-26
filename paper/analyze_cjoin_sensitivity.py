# -*- coding: utf-8 -*-
"""Deterministic sensitivity of the constrained-join equivalence-class fallback.

For each attr_filter question (constraints stated explicitly as "rel=val AND
rel=val" in the question text), recompute the answer set two ways from the raw
KG triples: (a) STRICT intersection of the stated relations; (b) with the
countryOfOrigin<->operatedBy union fallback applied when the strict
intersection is empty. Compare both to the gold set. No API, no pipeline ---
this isolates the executor-level effect of the fallback on attr_filter.
"""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
TRIP = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
QS = json.load(open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8"))["questions"]

# index: (relation, tail) -> set(heads)
heads = defaultdict(set)
for t in TRIP:
    heads[(t["relation"], str(t["tail"]))].add(str(t["head"]))

EQUIV = {"countryOfOrigin", "operatedBy"}

def strict_set(constraints):
    sets = [heads.get((r, v), set()) for r, v in constraints]
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out

def union_fallback_set(constraints):
    # union members of the equivalence class for each constraint relation
    sets = []
    for r, v in constraints:
        if r in EQUIV:
            u = set()
            for m in EQUIV:
                u |= heads.get((m, v), set())
            sets.append(u)
        else:
            sets.append(heads.get((r, v), set()))
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out

def parse_constraints(qtext):
    # questions look like: "... satisfy both REL=VAL AND REL=VAL?"  (or Chinese variant)
    m = re.search(r"(?:both\s+)?(.*?=.*?)\s*[?？]?$", qtext)
    body = qtext
    # take the part after 'both'/'同时满足' if present
    for marker in ["both ", "同时满足", "满足"]:
        if marker in body:
            body = body.split(marker, 1)[1]
            break
    body = body.rstrip("?？ 的雷达which radars ")
    pairs = re.split(r"\s+AND\s+|\s*且\s*|\s*并且\s*", body)
    cons = []
    for p in pairs:
        if "=" in p:
            r, v = p.split("=", 1)
            cons.append((r.strip(), v.strip().rstrip("?？的 ")))
    return cons

rows = []
fire = 0
n_eval = 0
strict_correct = fb_correct = 0
for q in QS:
    if q["type"] != "attr_filter":
        continue
    cons = parse_constraints(q.get("question_en") or q["question_zh"])
    if len(cons) < 2:
        cons = parse_constraints(q["question_zh"])
    gold = set(map(str, q["gold_answer"])) if isinstance(q["gold_answer"], list) else set()
    n_eval += 1
    ss = strict_set(cons)
    fired = (len(ss) == 0)          # fallback only fires when strict intersection empty
    fb = union_fallback_set(cons) if fired else ss
    if fired:
        fire += 1
    # exact-set correctness
    strict_correct += (ss == gold)
    fb_correct += (fb == gold)
    rows.append({"id": q.get("id"), "cons": cons, "gold_n": len(gold),
                 "strict_n": len(ss), "fired": fired, "fb_n": len(fb),
                 "strict_ok": ss == gold, "fb_ok": fb == gold})

print(f"attr_filter questions analyzed: {n_eval}")
print(f"parsed >=2 constraints: {sum(1 for r in rows if len(r['cons'])>=2)}/{n_eval}")
print(f"fallback FIRES (strict intersection empty): {fire}/{n_eval}")
print(f"executor exact-set match vs gold:  strict={strict_correct}/{n_eval}  with-fallback={fb_correct}/{n_eval}")
print(f"=> fallback changes executor accuracy by {(fb_correct-strict_correct)/n_eval*100:+.1f} pp (executor-level, exact-set)")
# show the cases where fallback fired
print("\ncases where fallback fired:")
for r in rows:
    if r["fired"]:
        print(f"  {r['id']}: cons={r['cons']} gold_n={r['gold_n']} strict_n=0 fb_n={r['fb_n']} fb_ok={r['fb_ok']}")
