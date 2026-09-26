"""
Build an OOD bilingual stress set by injecting aggressive English/abbreviation
surface forms into Chinese questions sampled from qa_500.

This goes beyond the auto-generator: we use less-common abbreviations and
multi-form substitutions to ensure the bilingual alias layer actually fires.
"""
import json
import re
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# Aggressive substitution dictionary (canonical Chinese → OOD English/abbrev)
SUBS = {
    # Countries — use less obvious English forms
    "美国":           ["USA", "U.S.", "United States"],
    "俄罗斯":         ["Russia", "USSR", "Russian Federation"],
    "中国":           ["China", "PRC"],
    "英国":           ["UK", "Britain", "United Kingdom"],
    "法国":           ["France"],
    "德国":           ["Germany", "FRG"],
    "意大利":         ["Italy"],
    "以色列":         ["Israel"],
    "日本":           ["Japan"],
    "瑞典":           ["Sweden"],
    "印度":           ["India"],
    "土耳其":         ["Turkey"],
    # Tech types
    "脉冲多普勒":     ["PD", "pulse Doppler", "Pulse-Doppler"],
    "合成孔径":       ["SAR", "synthetic aperture"],
    "逆合成孔径":     ["ISAR"],
    "动目标指示":     ["MTI", "moving target indication"],
    "地面动目标指示": ["GMTI", "Ground MTI"],
    "有源相控阵":     ["AESA", "active phased array"],
    "无源相控阵":     ["PESA", "passive phased array"],
    "相控阵":         ["phased array"],
    "调频连续波":     ["FMCW"],
    "连续波":         ["CW"],
    "单脉冲":         ["monopulse"],
    # Modes
    "搜索":           ["search mode"],
    "跟踪":           ["track mode"],
    "边搜索边跟踪":   ["TWS", "track-while-scan"],
    "边扫边跟":       ["TWS"],
    "地形回避":       ["TA", "terrain avoidance"],
    "地形跟随":       ["TF", "terrain following"],
    "地形测绘":       ["ground mapping"],
    "气象探测":       ["weather mode"],
    "导航":           ["navigation"],
    # Freq bands - use 'X band' format
    "X 波段":         ["X band", "X-band"],
    "S 波段":         ["S band", "S-band"],
    "L 波段":         ["L band", "L-band"],
    "C 波段":         ["C band", "C-band"],
    "Ku 波段":        ["Ku band", "Ku-band"],
    "Ka 波段":        ["Ka band", "Ka-band"],
    # Companies
    "雷神":           ["Raytheon"],
    "诺斯罗普·格鲁曼": ["Northrop Grumman", "NGC"],
    "洛克希德马丁":   ["Lockheed Martin", "LMT"],
}

# Reverse map: surface form → canonical (for tracking)
def main():
    random.seed(0)
    with open(ROOT / "evaluation" / "qa_500.json", encoding="utf-8") as f:
        all_qs = json.load(f)["questions"]

    # Group by gold_constraint tail or by question type
    candidate_qs = []
    for q in all_qs:
        text = q.get("question_zh", "")
        gc = q.get("gold_constraint", {})
        gold_tail = gc.get("tail", "")
        gold_head = gc.get("head", "")

        # Find which substitutable surface forms appear in this question
        sub_targets = []
        for canon, english_forms in SUBS.items():
            if canon in text:
                sub_targets.append((canon, english_forms[0]))   # use the first variant

        if sub_targets:
            candidate_qs.append((q, sub_targets))

    print(f"Questions with substitutable content: {len(candidate_qs)}")
    # Distribution by type
    type_counts = Counter(q["type"] for q, _ in candidate_qs)
    print("By type:")
    for t, c in type_counts.most_common():
        print(f"  {t:25s} {c}")

    # Stratified sample: target ~5-8 per type, total ~35
    target_per_type = {
        "agg_count":       6,
        "agg_enum":        6,
        "relation_inverse":6,
        "attr_filter":     5,
        "single_hop":      5,
        "two_hop_bridge":  4,
        "negation":        3,
    }
    by_type = defaultdict(list)
    for item in candidate_qs:
        by_type[item[0]["type"]].append(item)

    stress = []
    for t, target in target_per_type.items():
        pool = by_type.get(t, [])
        sample = random.sample(pool, min(target, len(pool)))
        for q, subs in sample:
            new_text = q["question_zh"]
            applied = []
            for canon, eng in subs[:2]:    # max 2 substitutions per question
                new_text = new_text.replace(canon, eng)
                applied.append([canon, eng])
            stress.append({
                "id":   f"ood_{q['id']}",
                "type": q["type"],
                "question_zh":          new_text,
                "question_zh_original": q["question_zh"],
                "bilingual_subs":       applied,
                "gold_answer":          q.get("gold_answer"),
                "gold_constraint":      q.get("gold_constraint"),
                "expected_strategy":    q.get("expected_strategy"),
            })

    print(f"\nGenerated {len(stress)} OOD stress questions")
    out = ROOT / "evaluation" / "bilingual_stress_OOD.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "version": "ood-v1",
            "description": "Hand-crafted OOD bilingual stress set; substitutions are aggressive (less common English/abbrev forms).",
            "questions": stress,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")

    # Show samples
    print("\nSample 5 transformations:")
    for q in random.sample(stress, 5):
        print(f"\n  [{q['id']}]")
        print(f"    orig: {q['question_zh_original']}")
        print(f"    new : {q['question_zh']}")
        print(f"    subs: {q['bilingual_subs']}")


if __name__ == "__main__":
    main()
