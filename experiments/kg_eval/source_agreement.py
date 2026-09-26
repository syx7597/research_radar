"""
(I) Multi-source agreement / corroboration — Knowledge-Vault-style trust signal
================================================================================
No human, no LLM. Idea (Dong et al., Knowledge Vault, KDD'14): a fact asserted by
SEVERAL INDEPENDENT sources is more likely true. We group the raw per-source
extractions into INDEPENDENT ORIGINS and, for each fact, count how many origins
assert it.

Independent origins (NOT extraction methods — origin of the underlying evidence):
  wikipedia      = method_a (rule) + method_b + method_c (LLM)   [same text origin]
  manual         = pdf_results (domain manuals)
  wikidata       = wikidata_results (structured KB)
  globalsecurity = globalsecurity_results (independent web source)

Reports:
  1. corroboration distribution (1 / 2 / >=3 independent origins);
  2. VALIDATION that agreement predicts quality: string-grounding rate vs #origins
     (if multi-source facts ground better, agreement is a real quality signal);
  3. value-consistency on functional relations across origins (do sources agree on
     the single value?).

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/source_agreement.py
"""
import re, json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
ER = ROOT / "extraction_results"
FUNCTIONAL = {"developedby", "countryoforigin", "manufacturedby", "affiliatedto"}

ORIGINS = {
    "wikipedia": ["method_a_results.json", "method_b_results.json", "method_c_results.json"],
    "manual": ["pdf_results.json"],
    "wikidata": ["wikidata_results.json"],
    "globalsecurity": ["globalsecurity_results.json"],
}


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def iter_triples(obj):
    """yield (head,relation,tail) from the various per-source layouts."""
    recs = obj if isinstance(obj, list) else [obj]
    for r in recs:
        if not isinstance(r, dict):
            continue
        if "triples" in r and isinstance(r["triples"], list):
            for t in r["triples"]:
                if isinstance(t, dict):
                    yield t
        elif {"head", "relation", "tail"} <= set(r):
            yield r


def get(t, *names):
    for n in names:
        if t.get(n) not in (None, ""):
            return t[n]
    return ""


def load_origin(files):
    facts = set()           # (nh, nr, nt)
    hv = defaultdict(set)   # (nh, nr) -> {nt}  for functional value-consistency
    for fn in files:
        p = ER / fn
        if not p.exists():
            continue
        for t in iter_triples(json.load(open(p, encoding="utf-8"))):
            h = get(t, "head", "subject", "h"); r = get(t, "relation", "predicate", "rel")
            ta = get(t, "tail", "object", "t")
            if not (h and r and ta):
                continue
            nh, nr, nt = norm(h), norm(r), norm(ta)
            facts.add((nh, nr, nt)); hv[(nh, nr)].add(nt)
    return facts, hv


def main():
    origin_facts, origin_hv = {}, {}
    for name, files in ORIGINS.items():
        origin_facts[name], origin_hv[name] = load_origin(files)
        print(f"  origin {name:<14} {len(origin_facts[name])} facts")

    # fact -> set of origins
    fact_origins = defaultdict(set)
    for name, facts in origin_facts.items():
        for f in facts:
            fact_origins[f].add(name)
    universe = len(fact_origins)
    dist = Counter(len(o) for o in fact_origins.values())
    corro = sum(v for k, v in dist.items() if k >= 2)
    print(f"\n=== 多源印证(独立来源) ===")
    print(f"事实总数(并集): {universe}")
    print(f"  仅 1 来源: {dist[1]} ({dist[1]/universe:.0%})")
    print(f"  2 来源:   {dist[2]} ({dist[2]/universe:.0%})")
    print(f"  >=3 来源: {sum(v for k,v in dist.items() if k>=3)}")
    print(f"  被>=2独立来源印证: {corro} ({corro/universe:.0%})")

    # VALIDATION: does corroboration predict grounding quality?
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    text_by_head = {norm(d["en_title"]): norm((d.get("raw_text_en") or "")) for d in corpus}
    bucket = {1: [0, 0], 2: [0, 0], 3: [0, 0]}   # nbr_origins -> [grounded, total]
    for (nh, nr, nt), origins in fact_origins.items():
        txt = text_by_head.get(nh)
        if not txt or len(nt) < 3:
            continue
        b = min(len(origins), 3)
        bucket[b][1] += 1
        if nt in txt:
            bucket[b][0] += 1
    print(f"\n=== 验证:印证度 vs 接地率(印证越多,接地是否越高) ===")
    for b in (1, 2, 3):
        g, n = bucket[b]
        lab = {1: "1 来源", 2: "2 来源", 3: ">=3 来源"}[b]
        if n:
            print(f"  {lab:<8} 接地 {g}/{n} = {g/n:.0%}")

    # functional value-consistency across origins
    print(f"\n=== 功能性关系跨源取值一致性 ===")
    hv_all = defaultdict(lambda: defaultdict(set))   # (nh,nr) -> origin -> {nt}
    for name in ORIGINS:
        for (nh, nr), vals in origin_hv[name].items():
            if nr in FUNCTIONAL:
                hv_all[(nh, nr)][name] = vals
    multi = {k: v for k, v in hv_all.items() if len(v) >= 2}
    agree = 0
    for (nh, nr), per in multi.items():
        sets = list(per.values())
        inter = set.intersection(*sets)
        if inter:
            agree += 1
    if multi:
        print(f"  >=2 来源都给了值的 (head,功能关系): {len(multi)} 个")
        print(f"  其中取值一致(交集非空): {agree} ({agree/len(multi):.0%})")
    else:
        print("  (跨源同时覆盖同一 head+功能关系的样本很少)")

    print(f"\n读法: 若'印证度↑→接地率↑',则多源印证是可信度的真实信号,可作无需人工的质量维度;"
          f"\n被>=2独立来源印证的事实可标为高可信层(Knowledge-Vault 思路)。")


if __name__ == "__main__":
    main()
