"""
Build the GOLD human-annotation sheet (Gao et al., VLDB'19 stratified sampling)
==============================================================================
Produces a sheet a human can actually fill: stratified by source (so we get
per-source precision AND a stratified overall estimate with a CI), every row
carrying the BEST available evidence to judge from:
    quoted evidence text  >  corpus context window for the head  >  manual page pointer.
Rows are sorted by radar so you read each model's facts together (cluster-read).

Label column `correct`: 1 = 正确 / 0 = 错误 / ? = 无法核实 (leave blank = not done).
Partial fills are fine — score_gold.py only scores filled rows and widens the CI.

Output: gold_annotation.csv (UTF-8-BOM, opens in Excel) + gold_weights.json (source
sizes for the stratified estimator, frozen at sampling time).

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/make_gold_sheet.py
"""
import re, csv, json, random
from pathlib import Path
from collections import defaultdict, Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
REL = json.load(open(ROOT / "lexicon" / "relations.json", encoding="utf-8"))["relations"]
corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))

N_PER_SOURCE = 12          # target per stratum
SEED = 20

# verifiable extraction sources -> category (inferred/enrich tiers excluded from the
# precision sheet; they are low-confidence by design and reported separately)
CAT = {}
for s in ["pdf_narrative_llm", "stage_c_llm", "pdf_spec_block", "llm_few_shot",
          "llm_zero_shot", "rule_pattern", "globalsecurity", "manual_expert",
          "schema_expand_llm"]:
    CAT[s] = "TEXT"
for s in ["wikidata", "wikidata_curated", "wikidata_curated_merger", "enrich_wiki"]:
    CAT[s] = "WIKI"
for s in ["extract_std_field", "extract_perf_data"]:
    CAT[s] = "POINTER"
for s in ["v2_attribute", "normalize_attrs"]:
    CAT[s] = "ATTR"

HINT = {
    "TEXT": "对照下方原文片段判断",
    "WIKI": "可查 Wikidata/维基百科核对",
    "POINTER": "下方含手册页码+语料片段；必要时查原手册",
    "ATTR": "数值/参数：对照下方语料或原始规格",
}


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def claim_zh(t):
    r = REL.get(t["relation"], {})
    tpl = r.get("answer_template_zh") or r.get("triple_text_zh")
    if tpl:
        try:
            return tpl.format(head=t["head"], tail=t["tail"])
        except Exception:
            pass
    return f'{t["head"]} 的 {t["relation"]} 是 {t["tail"]}'


def rel_zh(rel):
    return REL.get(rel, {}).get("zh_label", rel)


# corpus context by head
TX = {}
for d in corpus:
    TX[norm(d["en_title"])] = d


def context_for(t):
    """best fillable evidence: quoted text, else corpus window, else pointer."""
    ev = str(t.get("evidence") or "").strip()
    bad = any(k in ev.lower() for k in ("v2 attribute", "inferred", "enrich", "derived",
                                        "normalized", "wikidata "))
    pieces = []
    if ev and not bad and (len(ev) > 18 or CAT.get(t["source"]) == "TEXT"):
        pieces.append(f"[证据] {ev}")
    elif ev:
        pieces.append(f"[来源标记] {ev}")
    d = TX.get(norm(t["head"]))
    if d:
        txt = (d.get("raw_text_zh") or "") + " ||EN|| " + (d.get("raw_text_en") or "")
        low = txt.lower()
        win = ""
        for tok in [str(t["tail"]).lower()] + str(t["tail"]).lower().split():
            if len(tok) >= 3:
                i = low.find(tok)
                if i >= 0:
                    win = re.sub(r"\s+", " ", txt[max(0, i - 160):i + 200]).strip()
                    break
        if not win:
            ib = d.get("infobox") or d.get("known_facts") or ""
            win = re.sub(r"\s+", " ", (str(ib) or txt)[:240]).strip()
        if win:
            pieces.append(f"[语料] {win}")
    if not pieces:
        pieces.append("[无内置证据，需查原手册/外部检索]")
    return "  ".join(pieces)[:500]


def main():
    rng = random.Random(SEED)
    # group verifiable triples by source
    by_src = defaultdict(list)
    src_size = Counter()
    for t in T:
        s = t.get("source")
        if s in CAT and str(t.get("tail")).strip():
            by_src[s].append(t)
        if s in CAT:
            src_size[s] += 1

    rows = []
    for s, lst in by_src.items():
        rng.shuffle(lst)
        for t in lst[:N_PER_SOURCE]:
            rows.append(t)
    rng.shuffle(rows)
    rows.sort(key=lambda t: norm(t["head"]))     # cluster-read by radar

    out = HERE / "gold_annotation.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "雷达(head)", "关系", "值(tail)", "论断(中文)", "来源", "类别",
                    "证据/语境(据此判断)", "怎么核", "correct(1对/0错/?无法核)", "备注"])
        for i, t in enumerate(rows, 1):
            cat = CAT[t["source"]]
            w.writerow([i, t["head"], rel_zh(t["relation"]), t["tail"], claim_zh(t),
                        t["source"], cat, context_for(t), HINT[cat], "", ""])

    json.dump({"src_size": dict(src_size), "n_sampled_per_source":
               {s: min(N_PER_SOURCE, len(by_src[s])) for s in by_src},
               "seed": SEED, "total_rows": len(rows)},
              open(HERE / "gold_weights.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"生成 {out.name}: {len(rows)} 行,按雷达排序,每行带可判断的证据/语境。")
    print("各来源抽样:")
    for s in sorted(by_src, key=lambda x: -src_size[x]):
        print(f"  {s:<20} 抽 {min(N_PER_SOURCE,len(by_src[s])):>2} / 全库 {src_size[s]:>5}  [{CAT[s]}]")
    print("\n填法: 在 correct 列填 1(正确)/0(错误)/?(无法核实),可只填一部分。")
    print("填完跑: python experiments/kg_eval/score_gold.py")


if __name__ == "__main__":
    main()
