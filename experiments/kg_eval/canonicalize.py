"""
Entity resolution / canonicalization for the radar KG (deterministic, auditable)
================================================================================
Adapts EDC's "Canonicalize" to military nomenclature. Two deterministic, auditable
steps (no LLM in the trust path):
  1) NOISE removal: drop triples whose tail is a truncated/stopword fragment
     ("the", "the U", "theu", 1-char, ...).
  2) VARIANT merge (conservative): merge surface form B into shorter A iff
     norm(B) startswith norm(A), len(norm(A))>=5, and the remainder begins with a
     clear separator (space ( / , ) — version/family/spelling variants like
     "APQ-159(V)5"->"APQ-159", "Lockheed Martin Maritime Sensors..."->"Lockheed Martin",
     "AN/APG-63 radar family"->"AN/APG-63". Different models (AN/APG-63 vs -68) are NOT
     merged (remainder would start with a digit, blocked).

Measures the payoff:
  - functional-relation contradictions before vs after (developedBy/countryOfOrigin...)
  - agg_count exact-match accuracy before vs after (the dedup ceiling on §4 counting)

Run:  python experiments/kg_eval/canonicalize.py
"""
import re, json
from pathlib import Path
from collections import defaultdict
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qa_strategy_pipeline import KGIndex

T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
FUNCTIONAL = {"developedBy", "countryOfOrigin", "decade", "manufacturedBy"}
STOPWORDS = {"the", "a", "an", "of", "to", "and", "u", "theu", "thu", "der", "die", "le"}


def norm(s): return re.sub(r"\s+", "", str(s)).strip().lower()


def is_noise(tail):
    t = str(tail).strip()
    nt = norm(t)
    if nt in STOPWORDS or len(nt) <= 1:
        return True
    if re.fullmatch(r"(the|a|an)\b.{0,3}", t.lower() or ""):    # "the", "the U", "the"
        return True
    return False


def build_canon_map(entities):
    """Conservative variant merge. Returns surface -> canonical."""
    uniq = sorted(set(entities), key=lambda s: (len(norm(s)), norm(s)))
    canon = {}
    block = defaultdict(list)            # blocking key (first 4 chars) -> chosen canonicals
    for e in uniq:
        ne = norm(e)
        if not ne:
            canon[e] = e; continue
        key = ne[:4]
        matched = None
        for c in block[key]:
            nc = norm(c)
            if len(nc) >= 5 and ne.startswith(nc) and len(ne) > len(nc):
                rem = ne[len(nc):]
                if rem[:1] in " (/,-":     # clear separator -> variant, not a different model
                    # avoid merging "-<digit>" that forms a different number token
                    if not (rem[:1] == "-" and rem[1:2].isdigit()):
                        matched = c; break
        if matched:
            canon[e] = matched
        else:
            canon[e] = e; block[key].append(e)
    return canon


def main():
    ents = [t["head"] for t in T] + [t["tail"] for t in T if t.get("tail_type") != "Literal"]
    canon = build_canon_map(ents)
    n_clusters = len({canon[e] for e in canon})
    merged = sum(1 for e in canon if canon[e] != e)
    print(f"实体表面形: {len({norm(e) for e in canon})} 个 -> 规范实体 {n_clusters} 个 "
          f"(合并了 {merged} 个变体)")

    # canonicalized triples (drop noise tails; remap head/tail; dedup)
    cano_T, dropped = [], 0
    seen = set()
    for t in T:
        if is_noise(t["tail"]):
            dropped += 1; continue
        h = canon.get(t["head"], t["head"])
        ta = canon.get(t["tail"], t["tail"]) if t.get("tail_type") != "Literal" else t["tail"]
        key = (h, t["relation"], norm(ta))
        if key in seen:
            continue
        seen.add(key)
        nt = dict(t); nt["head"] = h; nt["tail"] = ta
        cano_T.append(nt)
    print(f"清噪: 删 {dropped} 条噪声三元组; 去重后三元组 {len(T)} -> {len(cano_T)}")

    # (1) functional-relation contradictions before vs after
    def contradictions(triples):
        hr = defaultdict(set)
        for t in triples:
            if t["relation"] in FUNCTIONAL:
                hr[(t["head"], t["relation"])].add(norm(t["tail"]))
        return sum(1 for v in hr.values() if len(v) > 1)
    c0, c1 = contradictions(T), contradictions(cano_T)
    print(f"\n功能性关系冲突: {c0} -> {c1}  (减少 {c0-c1}, {(c0-c1)/max(c0,1):.0%})")

    # (2) agg_count exact accuracy before vs after
    KG0 = KGIndex(T); KG1 = KGIndex(cano_T)
    qs = [q for q in json.load(open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8"))["questions"]
          if q["type"] == "agg_count" and q.get("gold_constraint") and isinstance(q.get("gold_answer"), int)]
    ok0 = ok1 = 0; err0 = err1 = 0
    for q in qs:
        rel, tail = q["gold_constraint"]["relation"], q["gold_constraint"]["tail"]
        gold = q["gold_answer"]
        ctail = canon.get(tail, tail)
        n0 = len(KG0.heads_with(rel, tail))
        n1 = len(KG1.heads_with(rel, ctail))
        ok0 += (n0 == gold); ok1 += (n1 == gold)
        err0 += abs(n0 - gold); err1 += abs(n1 - gold)
    m = len(qs)
    print(f"\nagg_count (n={m}):")
    print(f"  精确率(计数==金标):  {ok0}/{m}={ok0/m:.0%}  ->  {ok1}/{m}={ok1/m:.0%}")
    print(f"  平均计数误差:        {err0/m:.2f}  ->  {err1/m:.2f}")

    # show a few merges (auditability)
    print("\n合并示例(可审计):")
    shown = 0
    for e in sorted(canon, key=lambda x: len(norm(x))):
        if canon[e] != e and shown < 8:
            print(f"    {e!r}  ->  {canon[e]!r}")
            shown += 1


if __name__ == "__main__":
    main()
