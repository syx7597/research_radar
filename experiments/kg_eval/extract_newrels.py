"""
Stage D — populate the newly-adopted relations (controlled schema expansion).
=============================================================================
Targeted extraction over the FULL corpus for ONLY the 9 new schema relations
(validated by schema_expand.py). Every emitted triple is:
  - GROUNDED: its tail must appear in the source text (faithfulness gate, same
    lexical standard as the 86% grounding eval) — ungrounded proposals dropped;
  - PROVENANCED: source="schema_expand_llm", evidence=quoted snippet, confidence
    by tier (structural 0.8 / categorical 0.7 / numeric low-conf 0.55);
  - LINKED: head/entity-tail snapped to an existing KG radar entity when one
    matches (so new edges attach to the graph, raising connectivity).

Writes extraction_results/schema_expand_results.json (pipeline record), backs up
and merges into graphrag_index/merged_triples.json, and reports coverage gain.

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/extract_newrels.py
"""
import os, re, sys, json, time
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

KG_PATH = ROOT / "graphrag_index" / "merged_triples.json"
OUT_PATH = ROOT / "extraction_results" / "schema_expand_results.json"

# new relation -> (tail_type, confidence)
STRUCT = {"hasVariant": "Radar", "replaces": "Radar", "replacedBy": "Radar", "installedAt": "Location"}
ATTR = {"hasAntennaType": 0.7, "hasTransportMode": 0.7,
        "hasMaxTrackTargets": 0.55, "hasPulseWidth": 0.55, "hasAccuracy": 0.55}
CONF = {**{k: 0.8 for k in STRUCT}, **ATTR}
ALL_RELS = list(CONF)


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def jparse(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


PROMPT = (
    "你是雷达知识抽取器。只抽取下列 9 种关系的事实，且**值必须在原文中明确出现**（不得推断/编造）。\n"
    "结构关系(值为另一型号或地名):\n"
    "  hasVariant=该雷达的变体/子型号;  replaces=该雷达取代了哪款旧型号;\n"
    "  replacedBy=该雷达被哪款新型号取代;  installedAt=固定部署的地名/基地。\n"
    "属性(值为文本/数值，照抄原文):\n"
    "  hasAntennaType=天线类型;  hasTransportMode=机动/运输方式;\n"
    "  hasMaxTrackTargets=最大可跟踪目标数;  hasPulseWidth=脉冲宽度;  hasAccuracy=测量精度。\n"
    "把同义关系归并到上面(successorOf->replacedBy, predecessor->replaces, mountedOn 站点->installedAt)。\n"
    '只输出JSON: {"hasVariant":[..],"replaces":[..],"installedAt":[..],"hasAntennaType":[".."],...}，'
    "每个关系给一个值的数组，没有就给空数组。只输出JSON。"
)


def extract(title, text):
    out = llm_call([{"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"雷达型号: {title}\n原文:\n{text[:2600]}"}],
                   max_tokens=600)
    o = jparse(out)
    res = {}
    for r in ALL_RELS:
        v = o.get(r, [])
        if isinstance(v, str):
            v = [v]
        res[r] = [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
    return res


def grounded(tail, text_n, rel):
    """tail must appear in source text (lexical faithfulness gate)."""
    nt = norm(tail)
    if rel in ATTR and rel in ("hasMaxTrackTargets", "hasPulseWidth", "hasAccuracy"):
        nums = re.findall(r"\d[\d,\.]*", str(tail))
        return any(n.replace(",", "") in text_n for n in nums) if nums else False
    return len(nt) >= 3 and nt in text_n


def snippet(tail, text):
    """short quoted evidence window around the tail."""
    low = text.lower()
    for tok in [str(tail)] + str(tail).split():
        i = low.find(tok.lower())
        if i >= 0:
            return re.sub(r"\s+", " ", text[max(0, i-40):i+len(tok)+40]).strip()
    return f"quoted: {tail}"


def main():
    corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))
    T = json.load(open(KG_PATH, encoding="utf-8"))
    # canonical radar-entity map (norm -> exact existing head string)
    radar_heads = {}
    for t in T:
        if t.get("head_type") in ("Radar", "RadarSystem"):
            radar_heads.setdefault(norm(t["head"]), t["head"])
        if t.get("tail_type") in ("Radar", "RadarSystem"):
            radar_heads.setdefault(norm(t["tail"]), t["tail"])

    def snap(name):
        return radar_heads.get(norm(name), name)

    existing = {(t["head"], t["relation"], norm(t["tail"])) for t in T}
    docs = [d for d in corpus if len(d.get("raw_text_en") or "") > 400]
    print(f"在 {len(docs)} 篇语料上补抽 9 个新关系...", flush=True)

    new_triples, seen = [], set()
    proposed = grounded_n = 0
    per_rel = Counter()
    t0 = time.time()
    for k, d in enumerate(docs, 1):
        title, text = d["en_title"], (d.get("raw_text_en") or "")
        text_n = norm(text)
        head = snap(title)
        try:
            res = extract(title, text)
        except Exception as e:
            if k % 25 == 0:
                print(f"  [{k}] extract err {e}", flush=True)
            continue
        for rel, vals in res.items():
            for v in vals:
                proposed += 1
                if not grounded(v, text_n, rel):
                    continue
                grounded_n += 1
                if rel in STRUCT:
                    tail = snap(v) if STRUCT[rel] != "Location" else v
                    ttype = STRUCT[rel]
                    if rel in ("hasVariant", "replaces", "replacedBy") and norm(tail) == norm(head):
                        continue  # no self-edge
                else:
                    tail, ttype = v, "Literal"
                key = (head, rel, norm(tail))
                if key in existing or key in seen:
                    continue
                seen.add(key)
                new_triples.append({
                    "head": head, "head_type": "RadarSystem", "relation": rel,
                    "tail": tail, "tail_type": ttype, "confidence": CONF[rel],
                    "evidence": snippet(v, text)[:160], "source": "schema_expand_llm",
                })
                per_rel[rel] += 1
        if k % 25 == 0:
            print(f"  {k}/{len(docs)}  新边 {len(new_triples)}  ({time.time()-t0:.0f}s)", flush=True)

    # save pipeline record
    OUT_PATH.parent.mkdir(exist_ok=True)
    json.dump(new_triples, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # backup + merge into live KG
    bak = KG_PATH.with_suffix(".pre_schema_expand.bak.json")
    if not bak.exists():
        json.dump(T, open(bak, "w", encoding="utf-8"), ensure_ascii=False)
    merged = T + new_triples
    json.dump(merged, open(KG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # coverage report
    def avg_degree(triples):
        deg = defaultdict(set)
        for t in triples:
            if t.get("tail_type") not in ("Literal", None):
                deg[t["head"]].add((t["relation"], t["tail"]))
        return sum(len(v) for v in deg.values()) / max(len(deg), 1), len(deg)

    d0, n0 = avg_degree(T)
    d1, n1 = avg_degree(merged)
    radars_gained = len({t["head"] for t in new_triples})
    new_locs = len({t["tail"] for t in new_triples if t["tail_type"] == "Location"})
    print("\n" + "=" * 60)
    print(f"提议 {proposed} 条 -> 接地 {grounded_n} 条({grounded_n/max(proposed,1):.0%}) "
          f"-> 去重入库 {len(new_triples)} 条新边")
    print("按关系:")
    for r in ALL_RELS:
        print(f"  {r:<20} {per_rel[r]:>4}")
    print(f"\nKG: {len(T)} -> {len(merged)} 三元组 (+{len(new_triples)})")
    print(f"获得新边的雷达实体: {radars_gained} 个;  新增地点(installedAt)节点: {new_locs} 个")
    print(f"实体平均度(仅实体边): {d0:.2f}({n0}节点) -> {d1:.2f}({n1}节点)")
    print(f"\n备份: {bak.name};  新边记录: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
