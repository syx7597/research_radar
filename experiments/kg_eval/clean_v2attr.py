"""
Clean the v2_attribute tier (deterministic, auditable) — validate BEFORE applying.
==================================================================================
Diagnosis (characterize.): the numeric fields are clean; the damage is in
`*_description` OCR fragments (9%) and bad head entities (3%). So we clean, safely:

  A. HEAD HYGIENE   — strip trailing/leading punctuation on heads; if the cleaned head
     matches a real entity, remap; if the head is a non-entity even after stripping
     (orphan/garbage), drop its attribute triples.
  B. REDUNDANT GARBLED DESCRIPTIONS — drop a garbled `*_description` triple ONLY when a
     clean normalised sibling exists for the same head (e.g. range_description garbled
     but range_km present) -> zero information loss, removes the error-prone copy.
  C. FLOAT ARTIFACTS — round float tails (0.42000000000000004 -> 0.42).

No LLM. Garbled descriptions WITHOUT a clean sibling are only FLAGGED (not dropped),
because there we'd lose the sole record — left for review.

Validation: precision on the annotated gold rows (gold_annotation.csv) BEFORE vs AFTER
cleaning — does it remove the wrong ones while keeping the correct ones?

Run (dry-run + validate):  python experiments/kg_eval/clean_v2attr.py
Apply (backup + write KG):  python experiments/kg_eval/clean_v2attr.py --apply
"""
import re, sys, json, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "graphrag_index" / "merged_triples.json"
T = json.load(open(KG, encoding="utf-8"))

SIBLING = {
    "range_description": "range_km", "frequency_description": "frequency_GHz",
    "weight_description": "weight_kg", "deployment_year_description": "year",
    "peak_power_description": "peak_power_kW", "lru_description": "lru_count",
    "mtbf_description": "mtbf_hours", "rd_period_description": "rd_year_min",
    "antenna_gain_description": "antenna_gain_dB",
}


def norm(s):
    return re.sub(r"[\s\-_/().]+", "", str(s)).strip().lower()


def strip_head(h):
    return re.sub(r"^[\s，,、]+|[\s，,、]+$", "", str(h))


def garbled_desc(v):
    v = str(v)
    if len(v) > 40:
        return True
    if re.search(r"\d{4}\s*年", v) and re.search(r"工作方式|模式|波形|探测|通道", v):
        return True
    if len(re.findall(r"\d[\d,\.]*", v)) >= 4:          # many number fragments = OCR table dump
        return True
    return False


def main(apply=False):
    ent_heads = set()
    for t in T:
        if t.get("source") != "v2_attribute" and t.get("head_type") not in ("Literal", None):
            ent_heads.add(t["head"])
    ent_all = ent_heads | {t["tail"] for t in T if t.get("tail_type") not in ("Literal", None)}
    # clean-sibling presence: head -> set of relations with a parseable number
    has_sibling = defaultdict(set)
    for t in T:
        if t.get("source") == "v2_attribute" and re.fullmatch(r"-?\d[\d.]*", str(t["tail"]).strip()):
            has_sibling[t["head"]].add(t["relation"])

    cnt = defaultdict(int)
    kept, removed = [], []
    for t in T:
        if t.get("source") != "v2_attribute":
            kept.append(t); continue
        h, r, v = t["head"], t["relation"], str(t["tail"]).strip()
        sh = strip_head(h)
        # A. head hygiene
        if sh != h and sh in ent_heads:
            t = dict(t); t["head"] = sh; h = sh; cnt["head_remap"] += 1
        elif (h not in ent_all) and (sh not in ent_all):
            cnt["orphan_drop"] += 1; removed.append((t, "orphan_head")); continue
        # B. redundant garbled description
        if r.endswith("_description") and garbled_desc(v):
            sib = SIBLING.get(r)
            if sib and sib in has_sibling.get(h, ()):
                cnt["redundant_desc_drop"] += 1; removed.append((t, "garbled+sibling")); continue
            cnt["garbled_desc_flag"] += 1        # no sibling -> keep but DOWNGRADE confidence
            t = dict(t); t["confidence"] = min(float(t.get("confidence", 0.5)), 0.3)
            t["quality_flag"] = "garbled_ocr_unverifiable"
        # C. float artifact
        if re.fullmatch(r"-?\d+\.\d{6,}", v):
            t = dict(t); t["tail"] = str(round(float(v), 3)); cnt["float_round"] += 1
        kept.append(t)

    print("清洗动作(确定性):")
    print(f"  头标点修正(remap)      {cnt['head_remap']}")
    print(f"  孤儿/劣质头 删除        {cnt['orphan_drop']}")
    print(f"  乱码描述(有干净兄弟)删除 {cnt['redundant_desc_drop']}")
    print(f"  乱码描述(无兄弟)仅标记   {cnt['garbled_desc_flag']}")
    print(f"  浮点精度修正            {cnt['float_round']}")
    tot_removed = cnt["orphan_drop"] + cnt["redundant_desc_drop"]
    print(f"  => 删除 {tot_removed} 条,KG {len(T)} -> {len(kept)}")

    # VALIDATE against gold annotations
    validate(removed)

    if apply:
        bak = KG.with_suffix(".pre_v2clean.bak.json")
        if not bak.exists():
            json.dump(T, open(bak, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(kept, open(KG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n已写入清洗后 KG(备份 {bak.name})。重跑 score_gold.py 看新准确率。")
    else:
        print("\n(dry-run;确认无误后加 --apply 入库)")


def validate(removed):
    sheet = ROOT / "experiments" / "kg_eval" / "gold_annotation.csv"
    if not sheet.exists():
        return
    rows = None
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            rows = list(csv.DictReader(open(sheet, encoding=enc))); break
        except Exception:
            continue

    def col(r, *k):
        for c in r:
            if any(x in (c or "") for x in k):
                return r[c]
        return ""
    rm_ht = {(norm(t["head"]), norm(t["tail"])) for t, _ in removed}
    b_ok = b_bad = a_ok = a_bad = 0
    for r in rows:
        if "v2_attribute" not in (col(r, "来源", "source") or ""):
            continue
        lab = (col(r, "correct") or "").strip()
        if lab not in ("1", "0"):
            continue
        b_ok += lab == "1"; b_bad += lab == "0"
        key = (norm(col(r, "雷达", "head")), norm(col(r, "值", "tail")))
        if key in rm_ht:
            continue                     # removed by cleaner
        a_ok += lab == "1"; a_bad += lab == "0"
    print("\n验证(gold 中 v2_attribute 行,清洗前 vs 后):")
    print(f"  清洗前 精度 {b_ok}/{b_ok+b_bad} = {b_ok/max(b_ok+b_bad,1):.0%}  (对{b_ok}/错{b_bad})")
    print(f"  清洗后 精度 {a_ok}/{a_ok+a_bad} = {a_ok/max(a_ok+a_bad,1):.0%}  (对{a_ok}/错{a_bad};"
          f"移除了 {(b_ok+b_bad)-(a_ok+a_bad)} 条)")
    if (b_ok+b_bad) and (a_ok+a_bad):
        print(f"  -> 移除的是否都是错的: 删对 {b_ok-a_ok} / 删错 {b_bad-a_bad}"
              f"  (理想: 删对=0)")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
