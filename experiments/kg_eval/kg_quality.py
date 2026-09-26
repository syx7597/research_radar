"""
KG quality evaluation — automatable parts (consistency + evidence grounding)
============================================================================
Implements the standard KG-quality dimensions (accuracy / consistency /
completeness; Zaveri et al.; Gao et al. VLDB'19 sampling-based accuracy) for the
radar KG, doing the parts that need NO human and NO external source:

  (A) CONSISTENCY (syntactic accuracy):
      - type-signature violations (head_type/tail_type vs the relation schema)
      - self-loops (head == tail)
      - functional-relation contradictions (>1 distinct tail for a single-valued
        relation, e.g. developedBy / countryOfOrigin)
      - malformed / noise tails (stopword-only, truncated)
  (B) EVIDENCE GROUNDING (faithfulness of LLM/PDF extractions to their cited text):
      for extraction-sourced triples, does the cited evidence actually contain the
      asserted tail?  -> a hallucination/grounding proxy, reported PER SOURCE.
  (C) STRATIFIED SAMPLE for the manual accuracy annotation (the gold precision the
      student fills in), stratified by source, written to annotation_sheet.csv.

Run:  python experiments/kg_eval/kg_quality.py
"""
import json, re, random, csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
RELS = json.load(open(ROOT / "lexicon" / "relations.json", encoding="utf-8"))["relations"]
SIG = {rid: (set(rec.get("head_types", [])), set(rec.get("tail_types", [])))
       for rid, rec in RELS.items() if isinstance(rec, dict) and rec.get("head_types")}

# sources whose `evidence` is QUOTED SOURCE TEXT with a literal tail (grounding-checkable)
QUOTED_TEXT_SRC = {"pdf_narrative_llm", "pdf_spec_block", "llm_few_shot", "llm_zero_shot"}
# evidence is a PAGE POINTER (checkable against the PDF, not by string-match here)
POINTER_SRC = {"extract_std_field", "extract_perf_data"}
# evidence is a DERIVATION NOTE (an inference, low-confidence BY DESIGN — not a quote)
NOTE_SRC = {"enrich_func_llm", "enrich_compat_llm", "enrich_llm", "enrich_similar_llm",
            "rule_facts", "enrich_rule", "v2_attribute", "rule_pattern"}
# type-name aliases (the KG uses Radar / RadarSystem interchangeably)
TYPE_ALIAS = {"RadarSystem": "Radar", "Radar": "RadarSystem"}
FUNCTIONAL = {"developedBy", "countryOfOrigin", "decade", "manufacturedBy"}
STOPWORDS = {"the", "a", "an", "of", "in", "to", "and", "der", "die", "le", "la", "el"}


def norm(s): return re.sub(r"\s+", "", str(s)).strip().lower()


def consistency():
    n = len(T)
    type_viol, selfloop, malformed = [], [], []
    def type_ok(val, allowed):
        return val in allowed or TYPE_ALIAS.get(val) in allowed
    for t in T:
        r = t["relation"]
        if r in SIG:
            ht, tt = SIG[r]
            if not (type_ok(t.get("head_type"), ht) and type_ok(t.get("tail_type"), tt)):
                type_viol.append(t)
        if norm(t["head"]) == norm(t["tail"]):
            selfloop.append(t)
        tl = str(t["tail"]).strip()
        if tl.lower() in STOPWORDS or (len(tl) <= 1) or re.fullmatch(r"(the|a|an)\b.*", tl.lower() or "") and len(tl) <= 5:
            malformed.append(t)
    # functional contradictions
    hr = defaultdict(set)
    for t in T:
        if t["relation"] in FUNCTIONAL:
            hr[(t["head"], t["relation"])].add(norm(t["tail"]))
    contra = {k: v for k, v in hr.items() if len(v) > 1}

    print("=" * 66)
    print("(A) CONSISTENCY (syntactic accuracy) — automatable, no external source")
    print("=" * 66)
    print(f"  total triples: {n}")
    print(f"  type-signature violations : {len(type_viol):>5}  ({len(type_viol)/n:.2%})  "
          f"[Radar/RadarSystem normalised; {len(SIG)} typed relations]")
    print(f"     (by relation: {dict(Counter(t['relation'] for t in type_viol).most_common(5))})")
    print(f"  self-loops (head==tail)   : {len(selfloop):>5}  ({len(selfloop)/n:.2%})")
    print(f"  malformed/noise tails     : {len(malformed):>5}  ({len(malformed)/n:.2%})")
    print(f"  functional-relation contradictions: {len(contra)} heads have >1 tail "
          f"on a single-valued relation (potential errors)")
    for t in type_viol[:3]:
        print(f"     type-viol e.g.: ({t['head_type']}) {t['head']} -[{t['relation']}]-> "
              f"{t['tail']} ({t['tail_type']})")
    for (h, r), v in list(contra.items())[:3]:
        print(f"     contradiction e.g.: {h} -[{r}]-> {sorted(v)}")
    return type_viol, contra


def evidence_grounding():
    by_src_tot = Counter(); by_src_ok = Counter()
    kind_count = Counter()
    for t in T:
        s = str(t.get("source"))
        if s in QUOTED_TEXT_SRC:
            by_src_tot[s] += 1
            if norm(t["tail"]) in norm(t.get("evidence")):
                by_src_ok[s] += 1
        elif s in POINTER_SRC:
            kind_count["page-pointer (checkable vs PDF)"] += 1
        elif s in NOTE_SRC:
            kind_count["derivation note (inference, low-conf by design)"] += 1
        else:
            kind_count["other"] += 1
    print("\n" + "=" * 66)
    print("(B) EVIDENCE GROUNDING — for sources whose evidence is QUOTED SOURCE TEXT:")
    print("    does the cited text literally contain the asserted tail? (lower bound)")
    print("=" * 66)
    print(f"  {'source':<20}{'grounded':>10}{'total':>8}{'rate':>8}")
    tot = ok = 0
    for s in sorted(by_src_tot, key=lambda x: -by_src_tot[x]):
        r = by_src_ok[s] / by_src_tot[s] if by_src_tot[s] else 0
        print(f"  {s:<20}{by_src_ok[s]:>10}{by_src_tot[s]:>8}{r:>8.1%}")
        tot += by_src_tot[s]; ok += by_src_ok[s]
    print(f"  {'— quoted-text overall —':<20}{ok:>10}{tot:>8}{ok/tot:>8.1%}")
    print("\n  evidence TYPE of the remaining sources (lexical tail-match N/A for these):")
    for k, c in kind_count.most_common():
        print(f"    {c:>6}  {k}")


def stratified_sample(n_per_source=15, seed=0):
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for t in T:
        by_src[str(t.get("source"))].append(t)
    sample = []
    for s, ts in sorted(by_src.items(), key=lambda x: -len(x[1])):
        rng.shuffle(ts)
        sample += ts[:n_per_source]
    out = Path(__file__).resolve().parent / "annotation_sheet.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "head", "relation", "tail", "head_type", "tail_type",
                    "source", "confidence", "evidence", "correct(1/0)", "note"])
        for i, t in enumerate(sample):
            w.writerow([i, t["head"], t["relation"], t["tail"], t.get("head_type"),
                        t.get("tail_type"), t.get("source"), t.get("confidence"),
                        str(t.get("evidence"))[:160], "", ""])
    print("\n" + "=" * 66)
    print("(C) STRATIFIED SAMPLE for MANUAL accuracy annotation")
    print("=" * 66)
    print(f"  wrote {len(sample)} triples (~{n_per_source}/source) -> {out.name}")
    print("  -> annotate the 'correct(1/0)' column by hand; precision = mean, with a")
    print("     Wald/bootstrap 95% CI, reported overall and per source (Gao et al. VLDB'19).")


if __name__ == "__main__":
    consistency()
    evidence_grounding()
    stratified_sample()
