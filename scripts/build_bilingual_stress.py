"""
Build a 'bilingual stress test' by substituting English aliases into the
otherwise Chinese qa_500 questions. This isolates the bilingual layer's value:
on this stress set, the parser may extract the English form (USA, AESA, SAR)
which won't match the KG's Chinese canonical (美国, 有源相控阵, 合成孔径) without
alias resolution.

Output: evaluation/bilingual_stress_30.json
"""

import json
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)


# Substitutions for entity surface forms that have well-known English/abbrev variants
SUBSTITUTIONS = {
    # countries
    "美国":     "USA",
    "中国":     "China",
    "俄罗斯":   "Russia",
    "法国":     "France",
    "英国":     "UK",
    "日本":     "Japan",
    "德国":     "Germany",
    "以色列":   "Israel",
    "意大利":   "Italy",
    "印度":     "India",
    # tech_types
    "有源相控阵": "AESA",
    "无源相控阵": "PESA",
    "合成孔径":   "SAR",
    "动目标指示": "MTI",
    "脉冲多普勒": "PD",
    # frequency band — already letter codes, but with "波段" suffix
    "S 波段": "S band",
    "X 波段": "X band",
    # radar mode
    "边扫描边跟踪": "TWS",
    "边扫边跟":     "TWS",
}


def transform_question(q_zh: str) -> tuple[str, list]:
    """Apply substitutions to a Chinese question. Returns (new_zh, applied_subs)."""
    new = q_zh
    applied = []
    for zh, en in SUBSTITUTIONS.items():
        if zh in new:
            new = new.replace(zh, en)
            applied.append((zh, en))
    return new, applied


def main():
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]

    # Find questions whose Chinese text contains at least one substitutable entity.
    # Stratify: keep at least 3 per relevant type so we can compare per-type.
    candidates_by_type = {}
    for q in all_qs:
        new, subs = transform_question(q["question_zh"])
        if not subs:
            continue
        candidates_by_type.setdefault(q["type"], []).append((q, new, subs))

    # Sample ~30 with stratification across types that benefit from bilingual layer
    target = {
        "agg_count":         5,
        "agg_enum":          5,
        "attr_filter":       5,
        "relation_inverse":  5,
        "negation":          5,
        "two_hop_bridge":    3,
        "single_hop":        2,
    }

    stress_questions = []
    for t, n in target.items():
        pool = candidates_by_type.get(t, [])
        if not pool:
            continue
        sample = random.sample(pool, min(n, len(pool)))
        for q, new_zh, subs in sample:
            new_q = dict(q)
            new_q["id"]                  = "bs_" + q["id"]
            new_q["question_zh_original"] = q["question_zh"]
            new_q["question_zh"]         = new_zh
            new_q["bilingual_subs"]      = subs
            stress_questions.append(new_q)

    print(f"Built {len(stress_questions)} bilingual stress questions")
    from collections import Counter
    by_t = Counter(q["type"] for q in stress_questions)
    for t, c in by_t.most_common():
        print(f"  {t:20s} {c}")

    # Show 5 samples
    print("\nSample transformations:")
    for q in stress_questions[:5]:
        print(f"  [{q['id']}]")
        print(f"    orig: {q['question_zh_original']}")
        print(f"    new : {q['question_zh']}")
        print(f"    subs: {q['bilingual_subs']}")

    out = ROOT / "evaluation" / "bilingual_stress_30.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "version": "0.1-bilingual-stress",
            "description": "Stress test built by substituting English/abbrev aliases (USA/AESA/SAR/...) into Chinese qa_500 questions. Tests whether the bilingual alias layer recovers KG canonical forms.",
            "stats": dict(by_t),
            "questions": stress_questions,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
