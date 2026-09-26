"""
Validate an improved extraction pipeline (multi-agent + controlled schema expansion)
====================================================================================
Compares, on the SAME radar source texts and the SAME LLM (control), two pipelines:
  - SINGLE-PASS (baseline = the project's current method): one extraction call.
  - MULTI-AGENT: extractor -> critic/validator (rejects unfaithful or schema-invalid
    triples) ; the extractor may also PROPOSE new relations (controlled schema
    expansion) into a separate pool (NOT injected into the typed core).
Metrics per pipeline: yield (triples/doc), grounding (tail appears in source text =
faithfulness), schema-validity. Plus: critic rejection rate, proposed new relations.

Goal: does the multi-agent critic raise faithfulness/precision over single-pass?
(Validate before any rebuild — small sample, reported honestly.)

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/extract_pipeline.py
"""
import os, re, sys, json, random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# DeepSeek key into env (control model for both pipelines)
for i, ln in enumerate((ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()):
    if "api.deepseek.com" in ln:
        for nx in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()[i:i+4]:
            m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nx)
            if m:
                os.environ["DEEPSEEK_API_KEY"] = m.group(1)
from qa_strategy_pipeline import llm_call

KNOWN = {"developedBy", "countryOfOrigin", "operatedBy", "deployedOn", "hasFrequencyBand",
         "hasTechType", "hasMode", "hasFunction", "upgradeOf", "compatibleWith", "manufacturedBy",
         "hasSubsystem", "hasComponent", "derivedFrom", "meetsStandard"}
RELS = ("developedBy,countryOfOrigin,operatedBy,deployedOn,hasFrequencyBand,hasTechType,"
        "hasMode,hasFunction,upgradeOf,compatibleWith,manufacturedBy,hasSubsystem")


def norm(s): return re.sub(r"\s+", "", str(s)).strip().lower()
def jparse(s, default):
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else default
    except Exception:
        return default


def extractor(title, text, allow_new=True):
    extra = ("。若有重要事实上述关系装不下，可在 new_relations 提议新关系(不要硬塞)。"
             if allow_new else "。")
    sys_p = (f"你是雷达知识抽取器。从文本抽取关于该雷达的三元组，优先用关系: {RELS}{extra} "
             '输出JSON: {"triples":[{"head":..,"relation":..,"tail":..}],"new_relations":[{"relation":..,"example":..}]}，只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"雷达型号: {title}\n文本: {text[:1600]}"}], max_tokens=900)
    o = jparse(out, {})
    tr = [t for t in o.get("triples", []) if isinstance(t, dict) and t.get("relation") and t.get("tail")]
    return tr, o.get("new_relations", [])


def critic(text, triples):
    """Validator agent: keep only triples faithful to text and schema-reasonable."""
    if not triples:
        return [], 0
    listing = "\n".join(f"{i}. {t.get('head','?')} -[{t['relation']}]-> {t['tail']}" for i, t in enumerate(triples))
    sys_p = ("你是抽取校验员。对每条三元组判断：(a)原文是否支持该事实 (b)关系类型是否合理。"
             '只输出JSON数组 [{"i":序号,"keep":true/false}]，对无原文支持或明显错误的判 keep=false。只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"原文:\n{text[:1600]}\n\n三元组:\n{listing}"}], max_tokens=500)
    verdict = jparse(out, [])
    keep_idx = {v["i"] for v in verdict if isinstance(v, dict) and v.get("keep")} if verdict else set(range(len(triples)))
    kept = [t for i, t in enumerate(triples) if i in keep_idx]
    return kept, len(triples) - len(kept)


def measure(tag, doc_triples, docs):
    tot = gr = vd = 0
    for tr, d in zip(doc_triples, docs):
        tx = norm(d.get("raw_text_en") or "")
        for t in tr:
            tot += 1
            if t["relation"] in KNOWN: vd += 1
            if norm(t["tail"]) and norm(t["tail"]) in tx: gr += 1
    n = len(docs)
    print(f"  {tag:<22} 抽取率 {tot/n:.1f} 条/篇 | 接地率 {gr/max(tot,1):.0%} | "
          f"合法关系率 {vd/max(tot,1):.0%}  (共 {tot} 条)")
    return tot, gr


def main():
    rng = random.Random(1)
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    docs = [d for d in corpus if len(d.get("raw_text_en") or "") > 600]
    rng.shuffle(docs); docs = docs[:25]
    print(f"样本: {len(docs)} 篇雷达源文本；模型 DeepSeek(两条流水线同模型，只比方法)\n")

    sp, ma = [], []
    new_rel_pool = Counter(); rej_total = 0; rej_examples = []
    for k, d in enumerate(docs, 1):
        # single-pass baseline
        sp_tr, _ = extractor(d["en_title"], d.get("raw_text_en") or "", allow_new=False)
        sp.append(sp_tr)
        # multi-agent: extractor (+propose new) -> critic
        cand, newr = extractor(d["en_title"], d.get("raw_text_en") or "", allow_new=True)
        kept, nrej = critic(d.get("raw_text_en") or "", cand)
        ma.append(kept); rej_total += nrej
        for r in newr:
            if isinstance(r, dict) and r.get("relation") and r["relation"] not in KNOWN:
                new_rel_pool[r["relation"]] += 1
        if nrej and len(rej_examples) < 5:
            dropped = [t for t in cand if t not in kept][:1]
            if dropped:
                rej_examples.append((d["en_title"][:20], dropped[0]))
        print(f"  [{k}/{len(docs)}] {d['en_title'][:22]:<22} single={len(sp_tr)} "
              f"multi={len(kept)}(毙{nrej})", flush=True)

    print("\n" + "=" * 60)
    print("结果对比(同文本同模型)")
    print("=" * 60)
    measure("单次抽取(旧方法)", sp, docs)
    measure("多智能体(抽取+批判)", ma, docs)
    print(f"\n  批判 agent 共毙掉 {rej_total} 条(无原文支持/不合理)")
    if rej_examples:
        print("  毙掉例:")
        for title, t in rej_examples:
            print(f"    [{title}] {t.get('head','?')}-[{t['relation']}]->{t['tail']}")
    if new_rel_pool:
        print(f"\n  受控 schema 扩展——提议的新关系(候选，待人工审核加入):")
        for r, c in new_rel_pool.most_common(10):
            print(f"    {r}  (x{c})")
    print("\n读法: 若多智能体接地率/合法率↑(精度提升)、抽取率略降(毙掉幻觉)，则方法有效。")


if __name__ == "__main__":
    main()
