"""
Build ANNOTATABLE sheets — separating verifiable extracted facts from inferences.
================================================================================
The first sheet wasted slots on triples whose `evidence` is just a tag / page-pointer
/ derivation note, which a human cannot judge from the cell alone. This version:

  - categorises every triple by what its evidence actually IS, and
  - emits TWO sheets:
      annotation_facts.csv      — verifiable extracted facts (evidence is real text,
                                  a Wikidata/Wikipedia lookup, a manual page, or an
                                  attribute value you can check). Annotate truth.
      annotation_inference.csv  — rule/LLM-INFERRED enrichment (no original source).
                                  Judge plausibility, or skip — these are low-confidence
                                  by design and are reported separately, not as precision.
  - each row carries the FULL evidence text + a per-category "verify_hint", and the
    label column accepts 1 (correct) / 0 (wrong) / ? (cannot verify).

Run:  python experiments/kg_eval/make_annotation_sheet.py
"""
import json, csv, random
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
T = json.load(open(HERE.parents[1] / "graphrag_index" / "merged_triples.json", encoding="utf-8"))

# what the evidence actually is, per source
CAT = {}
for s in ["pdf_narrative_llm", "stage_c_llm", "pdf_spec_block", "llm_few_shot",
          "llm_zero_shot", "rule_pattern", "globalsecurity", "manual_expert"]:
    CAT[s] = "TEXT"          # evidence is real source text
for s in ["wikidata", "enrich_wiki", "wikidata_curated", "wikidata_curated_merger"]:
    CAT[s] = "WIKI"          # verify by Wikidata/Wikipedia lookup
for s in ["extract_std_field", "extract_perf_data"]:
    CAT[s] = "POINTER"       # evidence is a manual page number
for s in ["v2_attribute", "normalize_attrs"]:
    CAT[s] = "ATTR"          # an attribute value (verify the parameter by knowledge/source)
for s in ["enrich_func_llm", "rule_facts", "enrich_rule", "enrich_llm",
          "enrich_similar_llm", "enrich_chain", "enrich_compat_llm"]:
    CAT[s] = "INFERRED"      # derived / inferred — no original source

HINT = {
    "TEXT": "evidence 是原文片段：核对它是否支持该事实(再辅以常识)",
    "WIKI": "在 Wikidata/维基百科查该型号，核对该字段",
    "POINTER": "evidence 是手册页码：翻该页核对，或按常识/检索判断",
    "ATTR": "这是属性值：按领域常识/检索核对该型号此参数(频率/国别等)",
    "INFERRED": "推断/富化值，无原始出处：判断该推断是否合理，或填 ? 跳过",
}
# relations worth sampling for the precision number (factual, checkable)
KEY_REL = {"developedBy", "countryOfOrigin", "operatedBy", "deployedOn",
           "hasFrequencyBand", "hasTechType", "manufacturedBy", "decade",
           "upgradeOf", "compatibleWith", "frequency_GHz", "range_km"}


def write_sheet(path, rows, title):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([f"# {title}  —  label correct: 1=对 / 0=错 / ?=无法核实"])
        w.writerow(["id", "head", "relation", "tail", "source", "category",
                    "verify_hint", "evidence", "correct(1/0/?)", "note"])
        for i, t in enumerate(rows):
            cat = CAT.get(str(t.get("source")), "INFERRED")
            w.writerow([i, t["head"], t["relation"], t["tail"], t.get("source"), cat,
                        HINT[cat], str(t.get("evidence")), "", ""])


def main():
    rng = random.Random(0)
    facts_pool = defaultdict(list); infer_pool = []
    for t in T:
        cat = CAT.get(str(t.get("source")), "INFERRED")
        if cat == "INFERRED":
            infer_pool.append(t)
        else:
            # for the precision sample, prefer key factual relations
            key = (cat, t["relation"] in KEY_REL)
            facts_pool[key].append(t)

    # facts sheet: stratified across (category × is-key-relation), ~160 total
    facts = []
    for (cat, iskey), ts in facts_pool.items():
        rng.shuffle(ts)
        take = 22 if iskey else 8        # weight toward checkable key relations
        facts += ts[:take]
    rng.shuffle(facts)
    write_sheet(HERE / "annotation_facts.csv", facts, "可核实抽取事实(算精度)")

    # inference sheet: ~50
    rng.shuffle(infer_pool)
    write_sheet(HERE / "annotation_inference.csv", infer_pool[:50], "推断/富化(判合理性，单列报告)")

    bycat = defaultdict(int)
    for t in facts:
        bycat[CAT.get(str(t.get("source")), "INFERRED")] += 1
    print(f"annotation_facts.csv     : {len(facts)} 条可核实事实，按类别: {dict(bycat)}")
    print(f"annotation_inference.csv : 50 条推断/富化(单独判合理性)")
    print("\n标注：correct 列填 1/0/?；facts 表算精度，inference 表单列报告。")
    print("(旧 annotation_sheet.csv 作废，用这两份)")


if __name__ == "__main__":
    main()
