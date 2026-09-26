"""
Controlled schema expansion — collect candidate NEW relations across the corpus.
================================================================================
Runs the extractor (allowed to propose new relations) over a corpus sample, then
aggregates the proposed relations by frequency with example triples — a REVIEW LIST.
The human keeps the good ones and adds them to the 28-relation schema (controlled
expansion); nothing is auto-injected into the typed core.

This is the validated-useful method (extract_pipeline.py showed multi-agent critic
doesn't help, but schema expansion surfaces real missing relations).

Run (background):  PYTHONIOENCODING=utf-8 python experiments/kg_eval/schema_expand.py
"""
import os, re, sys, json, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for i, ln in enumerate((ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()):
    if "api.deepseek.com" in ln:
        for nx in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines()[i:i+4]:
            m = re.search(r'api[_]?key\s*[=:]\s*"?([A-Za-z0-9\-]{16,})"?', nx)
            if m:
                os.environ["DEEPSEEK_API_KEY"] = m.group(1)
from qa_strategy_pipeline import llm_call

KNOWN = {"developedBy", "countryOfOrigin", "operatedBy", "deployedOn", "hasFrequencyBand",
         "hasTechType", "hasMode", "hasFunction", "upgradeOf", "compatibleWith", "manufacturedBy",
         "hasSubsystem", "hasComponent", "derivedFrom", "meetsStandard", "similarTo", "affiliatedTo"}
RELS = ("developedBy,countryOfOrigin,operatedBy,deployedOn,hasFrequencyBand,hasTechType,"
        "hasMode,hasFunction,upgradeOf,compatibleWith,manufacturedBy,hasSubsystem,hasComponent")


def jparse(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def propose(title, text):
    sys_p = (f"你是雷达知识抽取器。优先用关系: {RELS}。"
             "若文本中有重要事实这些关系装不下，在 new_relations 里提议新关系(给关系名+一条例三元组)。"
             '输出JSON: {"new_relations":[{"relation":..,"head":..,"tail":..}]}，只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"雷达型号: {title}\n文本: {text[:1600]}"}], max_tokens=500)
    return jparse(out).get("new_relations", [])


def main():
    rng = random.Random(2)
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    docs = [d for d in corpus if len(d.get("raw_text_en") or "") > 600]
    rng.shuffle(docs); docs = docs[:60]
    print(f"在 {len(docs)} 篇上收集提议新关系...\n")
    freq = Counter(); examples = defaultdict(list)
    for k, d in enumerate(docs, 1):
        for r in propose(d["en_title"], d.get("raw_text_en") or ""):
            if not isinstance(r, dict):
                continue
            rel = str(r.get("relation", "")).strip()
            if rel and rel not in KNOWN:
                freq[rel] += 1
                if len(examples[rel]) < 2 and r.get("head") and r.get("tail"):
                    examples[rel].append(f"{r['head']}-[{rel}]->{r['tail']}")
        if k % 15 == 0:
            print(f"  {k}/{len(docs)} ...", flush=True)

    print("\n" + "=" * 60)
    print("候选新关系(按出现频次;人工审核后择优加入 schema)")
    print("=" * 60)
    for rel, c in freq.most_common(25):
        ex = "  |  ".join(examples[rel])
        print(f"  {c:>3}x  {rel:<22} {ex}")
    print(f"\n共 {len(freq)} 个候选新关系。建议保留频次≥2、语义清晰、可类型化的;"
          "\n弃一次性/含糊/可并入现有关系的。保留的加进 lexicon/relations.json 再补抽。")


if __name__ == "__main__":
    main()
