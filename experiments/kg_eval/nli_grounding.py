"""
(J) NLI / textual-entailment grounding — semantic faithfulness (upgrades string match)
======================================================================================
The string-match grounding (RESULTS §B) has two error modes:
  - FALSE NEGATIVE: tail is paraphrased / translated / normalised -> not found, but the
    text DOES support the fact;
  - FALSE POSITIVE: tail string appears, but NOT in a way that asserts the relation
    (e.g. "Raytheon" appears as a competitor, not the developer).
We replace it with an NLI-style judge: given the SOURCE TEXT (premise) and the triple
rendered as a natural-language CLAIM (hypothesis), decide entailed / not_supported /
unclear. The judge reads the source and checks SUPPORT (reading comprehension), not
world knowledge -- this is the standard mitigation for the "LLM judging an LLM-built KG"
circularity (it is not recalling facts, it is checking text-support); for a fully
cross-model check, point LLM_URL at a different backend.

Reports NLI-support rate vs string-grounding rate on the SAME stratified sample, the
2x2 agreement, and examples of each correction (string-miss/NLI-support ; string-hit/
NLI-reject).

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/nli_grounding.py
"""
import os, re, sys, json, random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for i, ln in enumerate((ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()):
    if "api.deepseek.com" in ln:
        for nx in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()[i:i+4]:
            m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nx)
            if m:
                os.environ["DEEPSEEK_API_KEY"] = m.group(1)
from qa_strategy_pipeline import llm_call

REL = json.load(open(ROOT / "lexicon" / "relations.json", encoding="utf-8"))["relations"]
# sources whose `evidence` is genuine quoted/prose text (so the citation can be audited);
# excludes field-tag / page-pointer / derivation-note sources (v2_attribute, extract_*,
# enrich_*, rule_facts, wikidata) where 'is it faithful to a cited text' is not the question.
TEXT_SOURCES = {"pdf_narrative_llm", "pdf_spec_block", "stage_c_llm", "llm_few_shot",
                "llm_zero_shot", "schema_expand_llm", "rule_pattern", "globalsecurity"}
BAD_EV = ("inferred", "enrich", "derived", "page ", "页", "v2 attribute", "wikidata",
          "推断", "派生", "normalized")


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def jparse(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def claim_of(t):
    r = REL.get(t["relation"], {})
    tpl = r.get("answer_template_zh") or r.get("triple_text_zh")
    if tpl:
        try:
            return tpl.format(head=t["head"], tail=t["tail"])
        except Exception:
            pass
    return f'{t["head"]} 的 {t["relation"]} 是 {t["tail"]}'


def premise_for(t, text_by_head):
    """premise = the cited evidence (audit the citation); augment short evidence with a
    corpus window if the head is in the corpus. Return None if no usable premise."""
    ev = str(t.get("evidence") or "").strip()
    bad = any(k in ev.lower() for k in BAD_EV)
    base = "" if bad else ev
    txt = text_by_head.get(norm(t["head"]), "")
    if len(base) < 30 and txt:                      # augment with corpus window around tail
        low = txt.lower()
        for tok in [str(t["tail"]).lower()] + str(t["tail"]).lower().split():
            if len(tok) >= 4:
                i = low.find(tok)
                if i >= 0:
                    win = re.sub(r"\s+", " ", txt[max(0, i-240):i+320]).strip()
                    base = (base + " " + win).strip()
                    break
        else:
            if len(base) < 12:
                base = re.sub(r"\s+", " ", txt[:900]).strip()
    return base[:800] if len(base) >= 12 else None


def nli(premise, claim):
    sys_p = ("你是事实校验员(NLI)。只依据【原文】判断【论断】是否被原文支持，不要用你自己的世界知识。"
             '输出JSON: {"verdict":"entailed|not_supported|unclear","why":"≤15字"}。'
             "entailed=原文明确支持(含同义/翻译/改写);not_supported=原文未提及或相矛盾;unclear=信息不足。只输出JSON。")
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"【原文】\n{premise}\n\n【论断】{claim}"}], max_tokens=120)
    v = jparse(out).get("verdict", "unclear")
    return v if v in ("entailed", "not_supported", "unclear") else "unclear"


def main():
    rng = random.Random(11)
    T = json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    text_by_head = {norm(d["en_title"]): (d.get("raw_text_en") or "") for d in corpus}

    # stratified sample over text-derived sources
    by_src = defaultdict(list)
    for t in T:
        if t.get("source") in TEXT_SOURCES and str(t.get("tail")).strip():
            by_src[t["source"]].append(t)
    sample = []
    for src, lst in by_src.items():
        rng.shuffle(lst)
        sample += lst[:24]
    rng.shuffle(sample)
    print(f"分层抽样候选 {len(sample)} 条(引文型来源 {len(by_src)} 类);NLI 判定模型见 LLM_MODEL。\n")

    cells = Counter()    # (string_hit, nli_verdict)
    str_g = nli_e = n = 0
    rec_examples, fp_examples = [], []
    for k, t in enumerate(sample, 1):
        prem = premise_for(t, text_by_head)
        if not prem:                       # no auditable citation -> out of scope, skip
            continue
        n += 1
        nt = norm(t["tail"])
        s_hit = len(nt) >= 3 and nt in norm(prem)
        v = nli(prem, claim_of(t))
        cells[(s_hit, v)] += 1
        str_g += s_hit; nli_e += (v == "entailed")
        if (not s_hit) and v == "entailed" and len(rec_examples) < 4:
            rec_examples.append((t, prem))
        if s_hit and v == "not_supported" and len(fp_examples) < 4:
            fp_examples.append((t, prem))
        if k % 30 == 0:
            print(f"  {k}/{len(sample)} (有效 {n}) ...", flush=True)

    print("\n" + "=" * 60)
    print(f"字符串接地率:  {str_g}/{n} = {str_g/n:.0%}")
    print(f"NLI 支持率:    {nli_e}/{n} = {nli_e/n:.0%}")
    print(f"  (NLI: entailed {nli_e}, not_supported {sum(v for (s,vv),v in cells.items() if vv=='not_supported')}, "
          f"unclear {sum(v for (s,vv),v in cells.items() if vv=='unclear')})")
    print("\n2x2(字符串命中 × NLI):")
    print(f"  命中 & entailed     {cells[(True,'entailed')]:>3}   (真接地)")
    print(f"  未命中 & entailed   {cells[(False,'entailed')]:>3}   <- NLI 找回(改写/翻译,字符串漏判)")
    print(f"  命中 & not_support  {cells[(True,'not_supported')]:>3}   <- NLI 揪出(字面在但关系不成立,字符串误判)")
    print(f"  未命中 & not_support{cells[(False,'not_supported')]:>3}   (确未支持)")
    if rec_examples:
        print("\nNLI 找回示例(字符串漏判,改写/翻译):")
        for t, prem in rec_examples:
            print(f"    {t['head']} -[{t['relation']}]-> {t['tail']}  | 引文: {prem[:60]}")
    if fp_examples:
        print("\nNLI 揪出示例(字符串误判为接地):")
        for t, prem in fp_examples:
            print(f"    {t['head']} -[{t['relation']}]-> {t['tail']}  | 引文: {prem[:60]}")
    print("\n读法: NLI 语义判定同时修正字符串法的漏判(改写/翻译)与误判(字面在但不表达该关系),"
          "\n是比字符串接地更硬的忠实度指标。")


if __name__ == "__main__":
    main()
