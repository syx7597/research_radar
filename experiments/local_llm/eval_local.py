"""
Local vs cloud LLM — extraction (A) and runtime QA (B), measured.
=================================================================
A) EXTRACTION: give both models the SAME radar source texts (corpus.json raw_text_en)
   with the same prompt; compare extraction RATE (triples/doc), GROUNDING (tail appears
   verbatim in source text = faithfulness/anti-hallucination), and schema validity.
B) RUNTIME QA: run agg_count/agg_enum questions through routing+parsing (the LLM's job)
   + the deterministic operator; score the answer set vs gold. Measures whether a local
   model routes/parses well enough. Both run with local Ollama and cloud DeepSeek.

Honest: these are modest samples (compute-bound on a local 14B). Reported as-is.
Run (background):  python experiments/local_llm/eval_local.py
"""
import os, re, sys, json, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def deepseek_key():
    txt = (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(txt):
        if "api.deepseek.com" in ln:
            for nx in txt[i:i+4]:
                m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nx)
                if m:
                    return m.group(1)
    return "none"

import qa_strategy_pipeline as qsp
import qa_router as qr
from qa_strategy_pipeline import KGIndex, parse_args, exec_exhaustive, exec_complement, exec_constrained_join
from qa_router import QuestionRouter
from lexicon import build_alias_index

LOCAL = ("http://localhost:11434/v1/chat/completions", os.getenv("LOCAL_MODEL", "qwen2.5:72b"), "ollama")
CLOUD = ("https://api.deepseek.com/v1/chat/completions", "deepseek-chat", deepseek_key())

def set_backend(b):
    qsp.LLM_URL, qsp.LLM_MODEL, qsp.LLM_KEY = b
    qr.LLM_URL, qr.LLM_MODEL, qr.LLM_KEY = b
    qsp.LLM_TIMEOUT = qr.LLM_TIMEOUT = 180

def norm(s): return re.sub(r"\s+", "", str(s)).strip().lower()
def f1(p, g):
    if not g and not p: return 1.0
    if not g or not p: return 0.0
    pn, gn = {norm(x) for x in p}, {norm(x) for x in g}
    tp = len(pn & gn)
    return 0.0 if not tp else 2*tp/len(pn)*tp/len(gn) / (tp/len(pn)+tp/len(gn))

KG = KGIndex(json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8")))
AL = build_alias_index()
KNOWN_RELS = {"developedBy", "countryOfOrigin", "operatedBy", "deployedOn", "hasFrequencyBand",
              "hasTechType", "hasMode", "hasFunction", "upgradeOf", "compatibleWith", "manufacturedBy"}
RELS_PROMPT = ("developedBy(研制方),countryOfOrigin(原产国),operatedBy(使用国),deployedOn(部署平台),"
               "hasFrequencyBand(频段,单字母如S/X),hasTechType(技术体制),hasMode(工作模式),"
               "hasFunction(功能),upgradeOf(前代型号),compatibleWith(兼容武器)")


# ---------- A: extraction ----------
def extract(title, text):
    msg = [{"role": "system", "content":
            f"你是雷达知识抽取器。从文本中抽取关于该雷达的三元组。只用这些关系:{RELS_PROMPT}。"
            f"主体一般是该雷达型号。严格输出 JSON 数组 [{{\"head\":..,\"relation\":..,\"tail\":..}}]，只输出JSON。"},
           {"role": "user", "content": f"雷达型号: {title}\n文本: {text[:1600]}"}]
    out = qsp.llm_call(msg, max_tokens=800)
    m = re.search(r"\[.*\]", out, re.DOTALL)
    try:
        arr = json.loads(m.group(0)) if m else []
        return [t for t in arr if isinstance(t, dict) and t.get("relation") and t.get("tail")]
    except Exception:
        return []

def run_A(docs):
    print("=" * 64); print("(A) EXTRACTION — local vs cloud, same texts"); print("=" * 64)
    for b in (LOCAL, CLOUD):
        set_backend(b)
        tot = gr = vd = 0; ndoc = 0; ex = None
        for d in docs:
            tr = extract(d["en_title"], d.get("raw_text_en") or "")
            tx = norm(d.get("raw_text_en") or "")
            for t in tr:
                tot += 1
                if t["relation"] in KNOWN_RELS: vd += 1
                if norm(t["tail"]) and norm(t["tail"]) in tx: gr += 1
            if ex is None and tr:
                ex = (d["en_title"], tr[:4])
            ndoc += 1
            print(f"    [{b[1]}] {d['en_title'][:24]:<24} -> {len(tr)} triples", flush=True)
        print(f"  >> {b[1]:<14} 抽取率 {tot/ndoc:.1f} 条/篇 | 接地率 {gr/max(tot,1):.0%} | "
              f"合法关系率 {vd/max(tot,1):.0%}  (共 {tot} 条/{ndoc} 篇)")
        if ex:
            print(f"     例[{ex[0]}]: " + "; ".join(f"{t.get('head','?')}-{t['relation']}->{t['tail']}" for t in ex[1]))
        print()


# ---------- B: runtime QA ----------
def exec_answer(strategy, args):
    if strategy in ("exhaustive", "relation_inverse"):
        return set(exec_exhaustive(args, KG, AL)["heads"])
    if strategy == "complement":
        return set(exec_complement(args, KG, AL).get("known_tails", []))
    if strategy == "constrained_join":
        return set(exec_constrained_join(args, KG, AL).get("intersection", []))
    cons = args.get("constraints") or []; head = args.get("primary_entity", "")
    rels = args.get("relation_chain") or []
    if cons and cons[0].get("tail") is not None:
        return set(KG.heads_with(cons[0]["relation"], cons[0]["tail"]))
    rel = rels[0] if rels else ""
    return set(KG.tails_of(head, rel)) if (head and rel) else set()

def run_B(qs):
    print("=" * 64); print("(B) RUNTIME QA — routing+parsing by the model, operators offline"); print("=" * 64)
    router = QuestionRouter()
    for b in (LOCAL, CLOUD):
        set_backend(b)
        acc = defaultdict(list); route_ok = 0
        for q in qs:
            try:
                qt, strat, _ = router.route(q["question_zh"])
                if strat == q["expected_strategy"]:
                    route_ok += 1
                args = parse_args(q["question_zh"], qt, strat, aliases=AL, kg=KG)
                ans = exec_answer(strat, args)
            except Exception:
                ans = set()
            if q["type"] == "agg_count":
                s = 1.0 if (isinstance(q["gold_answer"], int) and len(ans) == q["gold_answer"]) else 0.0
            else:
                s = f1(ans, set(q["gold_answer"]))
            acc[q["type"]].append(s)
            print(f"    [{b[1]}] {q['type']:<10} route={strat:<12} score={s:.2f}", flush=True)
        alls = [s for v in acc.values() for s in v]
        print(f"  >> {b[1]:<14} 路由正确率 {route_ok/len(qs):.0%} | "
              + " | ".join(f"{t} {sum(v)/len(v):.2f}" for t, v in acc.items())
              + f" | 总 {sum(alls)/len(alls):.2f}")
        print()


def main():
    rng = random.Random(0)
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    docs = [d for d in corpus if len(d.get("raw_text_en") or "") > 600]
    rng.shuffle(docs)
    run_A(docs[:10])

    qdata = json.load(open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8"))["questions"]
    pool = defaultdict(list)
    for q in qdata:
        if q["type"] in ("agg_count", "agg_enum") and q.get("gold_constraint"):
            if q["type"] == "agg_count" and not isinstance(q.get("gold_answer"), int):
                continue
            pool[q["type"]].append(q)
    qs = []
    for t in ("agg_count", "agg_enum"):
        rng.shuffle(pool[t]); qs += pool[t][:15]
    run_B(qs)


if __name__ == "__main__":
    main()
