"""
Layer 3 — LLM-GROUNDED repair of garbled v2_attribute descriptions (feasible subset).
=====================================================================================
Feasibility (checked): only ~30 of 330 no-sibling garbled `*_description` triples have
their head in the corpus, i.e. a clean source text to re-extract from (the rest are
manual-OCR with no recoverable source and are left confidence-downgraded by
clean_v2attr.py). For those 30: read the source article, ask the LLM for the clean
value of that attribute, and REPLACE the garbled tail ONLY if the new value is grounded
in the source (its key token appears in the text). Ungrounded -> left as-is.

This repairs (not just drops) and, being source-grounded, can also fix wrong values —
but its reach is bounded by source availability, reported honestly.

Run:  PYTHONIOENCODING=utf-8 python experiments/kg_eval/repair_v2attr.py [--apply]
"""
import os, re, sys, json
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

KG = ROOT / "graphrag_index" / "merged_triples.json"
T = json.load(open(KG, encoding="utf-8"))
corpus = json.load(open(ROOT / "radar_corpus" / "corpus.json", encoding="utf-8"))

SIBMAP = {"range_description": "range_km", "frequency_description": "frequency_GHz",
          "weight_description": "weight_kg", "deployment_year_description": "year",
          "peak_power_description": "peak_power_kW", "lru_description": "lru_count",
          "mtbf_description": "mtbf_hours", "rd_period_description": "rd_year_min",
          "antenna_gain_description": "antenna_gain_dB"}
ATTR = {"range_description": "最大作用/探测距离", "scan_range_description": "扫描范围",
        "frequency_description": "工作频率", "status_description": "服役状态",
        "price_description": "单价/造价", "pulse_width_description": "脉冲宽度",
        "receiver_description": "接收机关键指标", "azimuth_description": "方位覆盖范围",
        "beam_width_description": "波束宽度", "antenna_type_description": "天线类型",
        "peak_power_description": "峰值功率", "elevation_description": "俯仰覆盖范围",
        "prf_description": "脉冲重复频率", "mtbf_description": "平均无故障时间"}


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def garbled(v):
    v = str(v)
    return len(v) > 40 or (re.search(r"\d{4}\s*年", v) and re.search(r"工作方式|模式|波形|探测|通道", v)) \
        or len(re.findall(r"\d[\d,\.]*", v)) >= 4


TX = {norm(d["en_title"]): d for d in corpus}
sib = {(t["head"], t["relation"]) for t in T
       if t.get("source") == "v2_attribute" and re.fullmatch(r"-?\d[\d.]*", str(t["tail"]).strip())}


def attr_name(r):
    return ATTR.get(r, r.replace("_description", ""))


def extract_clean(title, text, attr, garbled_val):
    sys_p = ("你是雷达参数抽取器。根据【原文】给出该雷达的『" + attr + "』的**简洁规范值**"
             "(只给值,如 '148 km'、'X 波段'、'现役';原文没有就输出 '无')。不要照抄乱码。"
             '输出JSON: {"value":".."}。只输出JSON。')
    out = llm_call([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"雷达: {title}\n(旧的乱码值: {garbled_val[:60]})\n原文:\n{text[:2200]}"}],
                   max_tokens=80)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    try:
        return str(json.loads(m.group(0)).get("value", "")).strip() if m else ""
    except Exception:
        return ""


def grounded(val, text_n):
    toks = re.findall(r"\d[\d,\.]*|[A-Za-z一-鿿]{2,}", str(val))
    return any(norm(tok) in text_n for tok in toks if len(tok) >= 2)


def main(apply=False):
    targets = []
    for t in T:
        if (t.get("source") in ("v2_attribute",) and t["relation"].endswith("_description")
                and garbled(t["tail"]) and (t["head"], SIBMAP.get(t["relation"], "_")) not in sib
                and norm(t["head"]) in TX):
            targets.append(t)
    print(f"可接地重抽的乱码描述: {len(targets)} 条\n")

    repaired = failed = 0
    for k, t in enumerate(targets, 1):
        d = TX[norm(t["head"])]
        text = (d.get("raw_text_zh") or "") + "\n" + (d.get("raw_text_en") or "")
        val = extract_clean(t["head"], text, attr_name(t["relation"]), str(t["tail"]))
        if val and val not in ("无", "None") and grounded(val, norm(text)):
            print(f"  ✓ {t['head']} {t['relation']}: 乱码→ '{val}'")
            t["tail"] = val
            t["confidence"] = 0.7
            t["source"] = "v2_attribute_repaired"
            t["evidence"] = f"regrounded from source: {val}"
            t.pop("quality_flag", None)
            repaired += 1
        else:
            failed += 1
        if k % 10 == 0:
            print(f"  ...{k}/{len(targets)}", flush=True)

    print(f"\n修复 {repaired} 条(接地通过) / 未修复 {failed} 条(原文无据,保持降权)")
    if apply and repaired:
        bak = KG.with_suffix(".pre_v2repair.bak.json")
        if not bak.exists():
            json.dump(json.load(open(KG, encoding="utf-8")), open(bak, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(T, open(KG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"已写入 KG(备份 {bak.name})。")
    elif not apply:
        print("(dry-run; 加 --apply 入库)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
