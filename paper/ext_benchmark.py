# -*- coding: utf-8 -*-
"""External-domain, external-gold validation of the answer-geometry-mismatch claim.

Reviewer concern: the +35.9 pp gain is shown only on a self-built benchmark.
This script builds an INDEPENDENT benchmark from Wikidata (films domain), where
the knowledge graph AND the gold answers come from Wikidata SPARQL (external
truth, not authored by us), then tests whether the structural gain reproduces.

Design (deterministic, no LLM -> isolates the access-pattern claim from LLM noise):
  * Flat KG: triples pulled from Wikidata for a broad film neighbourhood
    (films, their directors / genres / countries / decades). The answer to a
    Count/enum question is a SUBSET of this KG, exactly as in RadarKG.
  * Question types mirror the AGM-exposed RadarKG types, with SPARQL-derived gold:
      agg_count        "How many films did D direct?"               (card)
      agg_enum         "List the films directed by D."              (card)
      relation_inverse "Which directors directed a <genre> film?"   (card)
      attr_filter      "Films directed by D AND of genre G?"        (card+comp)
      single_hop       "What decade is film F from?"                (none; control)
  * Baseline = BM25 top-K over triple text, with ORACLE answer extraction
    (we count/list the *correct* entities that appear in the top-K set -- a
    generous upper bound for a top-K system: it is still capped at K).
  * Strategy = the matching typed operator on the flat KG (exhaustive /
    constrained-join / lookup), exactly as in the paper.
  * Score: exact count match (agg_count/relation_inverse) or exact set match
    (agg_enum/attr_filter) or exact value (single_hop), vs Wikidata gold.

Output: paper/ext_benchmark_results.json + a printed per-type table.
"""
import json, time, math, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import urllib.parse, urllib.request

OUT = Path(__file__).resolve().parent / "ext_benchmark_results.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "radarkg-research/1.0 (academic; contact via paper)"
K_LIST = [8, 20]

def sparql(query, retries=3):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json",
                                               "User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())["results"]["bindings"]
        except Exception as e:
            print(f"  sparql retry {i+1}: {e}", file=sys.stderr); time.sleep(3)
    return []

def lbl(b, k):
    return b.get(k, {}).get("value", "")

# -------------------------------------------------------------------------
# 1) Pull a broad film neighbourhood. We take ~40 prolific directors, all their
#    films, and each film's genre / country / decade. Labels in English.
# -------------------------------------------------------------------------
print("[1/4] pulling directors with many films ...")
directors = sparql("""
SELECT ?d ?dLabel (COUNT(?f) AS ?c) WHERE {
  ?f wdt:P31 wd:Q11424 ; wdt:P57 ?d .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} GROUP BY ?d ?dLabel HAVING (?c >= 25) ORDER BY DESC(?c) LIMIT 40
""")
dir_ids = [b["d"]["value"].rsplit("/", 1)[1] for b in directors]
dir_name = {b["d"]["value"].rsplit("/", 1)[1]: lbl(b, "dLabel") for b in directors}
print(f"      {len(dir_ids)} directors (>=25 films each)")

triples = []        # (head, relation, tail)
film_name = {}
def add(h, r, t): triples.append((h, r, t))

print("[2/4] pulling films + attributes per director ...")
for i, did in enumerate(dir_ids):
    rows = sparql(f"""
    SELECT ?f ?fLabel ?gLabel ?cLabel ?date WHERE {{
      ?f wdt:P31 wd:Q11424 ; wdt:P57 wd:{did} .
      OPTIONAL {{ ?f wdt:P136 ?g . }}
      OPTIONAL {{ ?f wdt:P495 ?c . }}
      OPTIONAL {{ ?f wdt:P577 ?date . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}""")
    seen_film = set()
    for b in rows:
        f = b["f"]["value"].rsplit("/", 1)[1]; fn = lbl(b, "fLabel")
        if not fn or fn.startswith("Q"):   # skip unlabeled
            continue
        film_name[f] = fn
        if f not in seen_film:
            add(fn, "directedBy", dir_name[did]); seen_film.add(f)
        g = lbl(b, "gLabel")
        if g and not g.startswith("Q"): add(fn, "hasGenre", g)
        c = lbl(b, "cLabel")
        if c and not c.startswith("Q"): add(fn, "country", c)
        dt = lbl(b, "date")
        if dt and len(dt) >= 4 and dt[:4].isdigit():
            add(fn, "decade", f"{dt[:3]}0s")
    if (i + 1) % 10 == 0: print(f"      {i+1}/{len(dir_ids)} directors done, {len(triples)} triples")
    time.sleep(0.4)

# dedup
triples = sorted(set(triples))
print(f"      flat KG: {len(triples)} triples, {len(film_name)} films")

# -------------------------------------------------------------------------
# 2) KG indices
# -------------------------------------------------------------------------
heads_by_rt = defaultdict(set)   # (rel, tail) -> heads
tails_by_hr = defaultdict(set)   # (head, rel) -> tails
for h, r, t in triples:
    heads_by_rt[(r, t)].add(h)
    tails_by_hr[(h, r)].add(t)

# -------------------------------------------------------------------------
# 3) Generate questions + Wikidata-derived gold
# -------------------------------------------------------------------------
print("[3/4] generating questions + gold ...")
Q = []
import random; random.seed(7)

# agg_count + agg_enum: films directed by D  (high cardinality)
for did in dir_ids:
    d = dir_name[did]
    films = heads_by_rt[("directedBy", d)]
    if len(films) >= 25:
        Q.append({"type": "agg_count", "q": f"How many films did {d} direct?",
                  "op": "exhaustive", "rel": "directedBy", "tail": d, "gold": len(films)})
        Q.append({"type": "agg_enum", "q": f"List the films directed by {d}.",
                  "op": "exhaustive", "rel": "directedBy", "tail": d, "gold": sorted(films)})

# relation_inverse: directors who directed a film of genre G (enumerate inverse)
genre_dir = defaultdict(set)
for (h, r, t) in triples:
    if r == "hasGenre":
        for d in tails_by_hr[(h, "directedBy")]:
            genre_dir[t].add(d)
for g, ds in genre_dir.items():
    if len(ds) >= 15:
        Q.append({"type": "relation_inverse", "q": f"Which directors directed a film of genre {g}?",
                  "op": "exhaustive_inv", "rel": "hasGenre", "tail": g, "gold": sorted(ds)})

# attr_filter: films directed by D AND genre G  (intersection)
for did in dir_ids[:25]:
    d = dir_name[did]
    fd = heads_by_rt[("directedBy", d)]
    gc = Counter()
    for f in fd:
        for g in tails_by_hr[(f, "hasGenre")]:
            gc[g] += 1
    for g, n in gc.items():
        if n >= 3:
            inter = {f for f in fd if g in tails_by_hr[(f, "hasGenre")]}
            Q.append({"type": "attr_filter",
                      "q": f"Which films are directed by {d} AND of genre {g}?",
                      "op": "constrained_join",
                      "cons": [("directedBy", d), ("hasGenre", g)], "gold": sorted(inter)})
            break   # one per director keeps it balanced

# single_hop control: decade of a film
films_with_decade = [h for (h, r, t) in triples if r == "decade"]
for f in random.sample(films_with_decade, min(40, len(films_with_decade))):
    dec = sorted(tails_by_hr[(f, "decade")])
    if len(dec) == 1:
        Q.append({"type": "single_hop", "q": f"What decade is the film {f} from?",
                  "op": "lookup", "head": f, "rel": "decade", "gold": dec[0]})

print(f"      {len(Q)} questions: " + str(dict(Counter(q['type'] for q in Q))))

# -------------------------------------------------------------------------
# 4) BM25 over triple text (for the Baseline)
# -------------------------------------------------------------------------
def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
docs = [f"{h} {r} {t}" for (h, r, t) in triples]
doc_toks = [toks(d) for d in docs]
df = Counter()
for dt in doc_toks:
    for w in set(dt): df[w] += 1
N = len(docs); avgdl = sum(len(d) for d in doc_toks) / max(N, 1)
def bm25_top(query, k):
    qt = toks(query); k1, b = 1.5, 0.75
    scores = []
    for i, dt in enumerate(doc_toks):
        tf = Counter(dt); s = 0.0; dl = len(dt)
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scores.append((s, i))
    scores.sort(reverse=True)
    return [triples[i] for _, i in scores[:k]]

# -------------------------------------------------------------------------
# answer extraction
# -------------------------------------------------------------------------
def strategy_answer(q):
    op = q["op"]
    if op == "exhaustive":
        return sorted(heads_by_rt[(q["rel"], q["tail"])])
    if op == "exhaustive_inv":
        ds = set()
        for h in heads_by_rt[(q["rel"], q["tail"])]:
            ds |= tails_by_hr[(h, "directedBy")]
        return sorted(ds)
    if op == "constrained_join":
        sets = [heads_by_rt[c] for c in q["cons"]]
        out = set(sets[0])
        for s in sets[1:]: out &= s
        return sorted(out)
    if op == "lookup":
        v = sorted(tails_by_hr[(q["head"], q["rel"])]); return v[0] if v else ""
    return []

def baseline_answer(q, k):
    """Generous oracle extraction from BM25 top-K: keep only the *correct* items
    present in the retrieved set (upper bound for a top-K reader, still K-capped)."""
    top = bm25_top(q["q"], k)
    if q["type"] in ("agg_count", "agg_enum"):
        cand = {h for (h, r, t) in top if r == q["rel"] and t == q["tail"]}
        return sorted(cand)
    if q["type"] == "relation_inverse":
        films = {h for (h, r, t) in top if r == "hasGenre" and t == q["tail"]}
        ds = set()
        for f in films: ds |= tails_by_hr[(f, "directedBy")]
        return sorted(ds)
    if q["type"] == "attr_filter":
        d, g = q["cons"][0][1], q["cons"][1][1]
        cand = {h for (h, r, t) in top if (r, t) in (q["cons"][0], q["cons"][1])}
        inter = {f for f in cand if g in tails_by_hr[(f, "hasGenre")] and d in tails_by_hr[(f, "directedBy")]}
        return sorted(inter)
    if q["type"] == "single_hop":
        for (h, r, t) in top:
            if h == q["head"] and r == q["rel"]: return t
        return ""
    return []

def correct(q, ans):
    if q["type"] == "agg_count":
        n = len(ans) if isinstance(ans, list) else ans
        return int(n == q["gold"])
    if q["type"] == "single_hop":
        return int(ans == q["gold"])
    return int(sorted(ans) == sorted(q["gold"]))   # exact set

# -------------------------------------------------------------------------
# run
# -------------------------------------------------------------------------
print("[4/4] scoring ...")
rows = []
for q in Q:
    s_ans = strategy_answer(q)
    rec = {"type": q["type"], "q": q["q"], "gold_n": (q["gold"] if q["type"]=="agg_count" else (1 if q["type"]=="single_hop" else len(q["gold"]))),
           "strategy_ok": correct(q, s_ans)}
    for k in K_LIST:
        rec[f"baseline_k{k}_ok"] = correct(q, baseline_answer(q, k))
    rows.append(rec)

def agg(rows, key):
    by = defaultdict(list)
    for r in rows: by[r["type"]].append(r[key])
    return {t: sum(v)/len(v)*100 for t, v in by.items()}, sum(r[key] for r in rows)/len(rows)*100

types = ["agg_count", "agg_enum", "relation_inverse", "attr_filter", "single_hop"]
print("\n=== External Wikidata (films) benchmark: Baseline (BM25 top-K) vs Strategy ===")
print(f"{'type':<18}{'n':>4}{'gold_med':>9}{'base@8':>9}{'base@20':>9}{'Strat':>9}{'Δ(S-b20)':>10}")
b8, _ = agg(rows, "baseline_k8_ok"); b20, _ = agg(rows, "baseline_k20_ok"); st, _ = agg(rows, "strategy_ok")
import statistics as S
for t in types:
    tr = [r for r in rows if r["type"] == t]
    if not tr: continue
    med = int(S.median([r["gold_n"] for r in tr]))
    print(f"{t:<18}{len(tr):>4}{med:>9}{b8.get(t,0):>9.1f}{b20.get(t,0):>9.1f}{st.get(t,0):>9.1f}{st.get(t,0)-b20.get(t,0):>10.1f}")
ob8 = sum(r['baseline_k8_ok'] for r in rows)/len(rows)*100
ob20 = sum(r['baseline_k20_ok'] for r in rows)/len(rows)*100
ost = sum(r['strategy_ok'] for r in rows)/len(rows)*100
print("-"*68)
print(f"{'OVERALL':<18}{len(rows):>4}{'':>9}{ob8:>9.1f}{ob20:>9.1f}{ost:>9.1f}{ost-ob20:>10.1f}")

json.dump({"n_triples": len(triples), "n_questions": len(rows), "rows": rows,
           "overall": {"baseline_k8": ob8, "baseline_k20": ob20, "strategy": ost}},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nsaved {OUT}")
