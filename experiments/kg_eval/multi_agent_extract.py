"""
Multi-agent extraction pipeline + the honest "where does the critic earn its place?" test
==========================================================================================
Documents the extraction as a 3-role multi-agent pipeline and tests the critic
HONESTLY — not on clean Wikipedia (where it's already known to be neutral, 86%->82%
in extract_pipeline.py) but as a function of INPUT NOISE, the realistic deployment
condition (scanned/classified manuals, whose real grounding was only 77% vs 87% for
clean prose, see RESULTS_kg_quality.md §B).

Roles:
  1. Extractor      — pulls triples (the workhorse).
  2. Critic         — drops triples the source text doesn't support (faithfulness gate).
                      This is the gate already used (lexical form) in extract_newrels.py.
  3. Schema-Proposer— proposes relations the schema can't hold (validated POSITIVE,
                      realised in schema_expand.py / extract_newrels.py: +583 edges).

Controlled ablation (this file): SAME documents, vary only input noise.
  clean text   -> extractor -> {raw, +critic}
  noised text  -> extractor -> {raw, +critic}      (OCR-style degradation of the SAME doc)
Grounding is always measured against the CLEAN reference text (ground truth), so a
garbled/hallucinated tail from noisy input is correctly counted as ungrounded.
Hypothesis: critic lift (kept_grounding - raw_grounding) is ~0 on clean input but
POSITIVE on noisy input -> the critic is justified exactly in the deployment regime.

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/multi_agent_extract.py
"""
import os, re, sys, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for i, ln in enumerate((ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()):
    if "api.deepseek.com" in ln:
        for nx in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()[i:i+4]:
            m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nx)
            if m:
                os.environ["DEEPSEEK_API_KEY"] = m.group(1)
from qa_strategy_pipeline import llm_call

RELS = ("developedBy,countryOfOrigin,operatedBy,deployedOn,hasFrequencyBand,hasTechType,"
        "hasMode,hasFunction,upgradeOf,compatibleWith,manufacturedBy,hasSubsystem")


def norm(s):
    return re.sub(r"\s+", "", str(s)).strip().lower()


def jparse(s, default):
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else default
    except Exception:
        return default


# ---- Role 1: Extractor ----
def extractor(title, text):
    sys_p = (f"你是雷达知识抽取器。从文本抽取关于该雷达的三元组，优先用关系: {RELS}。"
             '输出JSON: {"triples":[{"head":..,"relation":..,"tail":..}]}，只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"雷达型号: {title}\n文本: {text[:1700]}"}], max_tokens=900)
    o = jparse(out, {})
    return [t for t in o.get("triples", []) if isinstance(t, dict) and t.get("relation") and t.get("tail")]


# ---- Role 2: Critic (faithfulness gate) ----
def critic(text, triples):
    if not triples:
        return []
    listing = "\n".join(f"{i}. {t.get('head','?')} -[{t['relation']}]-> {t['tail']}"
                        for i, t in enumerate(triples))
    sys_p = ("你是抽取校验员。对每条三元组判断原文是否真的支持该事实(值是否在文本中、是否被讹误)。"
             '只输出JSON数组 [{"i":序号,"keep":true/false}]，对无原文支持/明显讹误判 keep=false。只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"原文:\n{text[:1700]}\n\n三元组:\n{listing}"}], max_tokens=500)
    v = jparse(out, [])
    keep = {x["i"] for x in v if isinstance(x, dict) and x.get("keep")} if v else set(range(len(triples)))
    return [t for i, t in enumerate(triples) if i in keep]


# ---- OCR-style degradation (emulates scanned-manual artifacts: truncation/dropout) ----
def ocr_noise(text, p, rng):
    res = []
    for w in text.split():
        r = rng.random()
        if len(w) > 3 and r < p * 0.35:                 # truncate tail ("the U"->"theu" style)
            w = w[:len(w) - rng.randint(1, 2)]
        elif len(w) > 3 and r < p * 0.65:               # delete an interior char
            i = rng.randint(1, len(w) - 2); w = w[:i] + w[i + 1:]
        elif len(w) > 4 and r < p * 0.80:               # OCR substitution
            i = rng.randint(0, len(w) - 1); w = w[:i] + rng.choice("uiltco0") + w[i + 1:]
        res.append(w)
    return " ".join(res)


def grounding(triples, ref_text_n):
    """fraction of triples whose tail appears in the CLEAN reference text."""
    if not triples:
        return 0.0, 0
    g = sum(1 for t in triples if norm(t["tail"]) and norm(t["tail"]) in ref_text_n)
    return g / len(triples), len(triples)


def main():
    rng = random.Random(7)
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    docs = [d for d in corpus if len(d.get("raw_text_en") or "") > 900]
    rng.shuffle(docs); docs = docs[:30]
    P = 0.18  # noise level (moderate scan degradation)
    print(f"样本 {len(docs)} 篇；同文档 clean vs noised(p={P});接地一律对**干净原文**判(真值)。\n")

    acc = {"clean_raw": [], "clean_crit": [], "noisy_raw": [], "noisy_crit": []}
    cnt = {k: 0 for k in acc}
    for k, d in enumerate(docs, 1):
        clean = d.get("raw_text_en") or ""
        ref_n = norm(clean)
        noisy = ocr_noise(clean, P, rng)
        try:
            craw = extractor(d["en_title"], clean)
            ccrit = critic(clean, craw)
            nraw = extractor(d["en_title"], noisy)
            ncrit = critic(noisy, nraw)
        except Exception as e:
            print(f"  [{k}] err {e}", flush=True); continue
        for tag, tr in [("clean_raw", craw), ("clean_crit", ccrit), ("noisy_raw", nraw), ("noisy_crit", ncrit)]:
            g, n = grounding(tr, ref_n)
            if n:
                acc[tag].append(g); cnt[tag] += n
        if k % 10 == 0:
            print(f"  {k}/{len(docs)} ...", flush=True)

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    print("\n" + "=" * 64)
    print("接地率(对干净原文;按文档平均)  —  Critic 在干净 vs 噪声输入上的增益")
    print("=" * 64)
    cr, cc = mean(acc["clean_raw"]), mean(acc["clean_crit"])
    nr, nc = mean(acc["noisy_raw"]), mean(acc["noisy_crit"])
    print(f"  干净输入:  Extractor {cr:.0%}  ->  +Critic {cc:.0%}   (增益 {cc-cr:+.1%})  "
          f"产量 {cnt['clean_raw']}->{cnt['clean_crit']}")
    print(f"  噪声输入:  Extractor {nr:.0%}  ->  +Critic {nc:.0%}   (增益 {nc-nr:+.1%})  "
          f"产量 {cnt['noisy_raw']}->{cnt['noisy_crit']}")
    print("\n读法(诚实):")
    print(f"  - 噪声把抽取忠实度从 {cr:.0%} 拉到 {nr:.0%}(模拟扫描手册);")
    print(f"  - Critic 在干净输入增益 {cc-cr:+.1%}、在噪声输入增益 {nc-nr:+.1%}。")
    if (nc - nr) - (cc - cr) > 0.02:
        print("  => Critic 的价值随输入噪声上升:在贴近涉密手册的脏文本上才真正发挥作用(有条件正面)。")
    else:
        print("  => Critic 增益不随噪声显著上升:作为忠实度门控保证可审计性,精度中性(如实写)。")


if __name__ == "__main__":
    main()
