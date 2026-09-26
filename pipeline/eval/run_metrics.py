"""
计算三元组抽取的 Precision / Recall / F1
运行: python run_metrics.py

标注文件: extraction_results/annotation_template_labeled.json
（把 annotation_template_labeled.json 重命名或复制成 annotation_template.json 也可以）

我们使用的标签体系:
  correct  → 完全正确
  partial  → 部分正确（按 0.5 分计入 correct）
  wrong    → 错误（实体/关系/方向 等各类错误统一归入此类）

Recall 的分母 = 模型输出正确数 + 人工补充的漏召数
因为我们没有做全量人工标注，human_added 暂为 0，
Recall 当前反映的是"已知正例中的召回率"，是保守估计。
"""

import json
from pathlib import Path
from collections import defaultdict

ANN_PATH = Path("extraction_results/annotation_template_labeled.json")
if not ANN_PATH.exists():
    # 兼容旧文件名
    ANN_PATH = Path("extraction_results/annotation_template.json")

print(f"读取标注文件: {ANN_PATH}")
with open(ANN_PATH, encoding="utf-8") as f:
    annotations = json.load(f)

# ── 统计 ──────────────────────────────────────────────────
total       = 0
correct_full = 0   # label=correct
correct_half = 0   # label=partial，按 0.5 计
wrong       = 0
unannotated = 0

per_rel = defaultdict(lambda: {"total": 0, "correct": 0, "partial": 0, "wrong": 0})

for item in annotations:
    for t in item.get("llm_fewshot_triples_to_annotate", []):
        label    = t.get("label", "")
        relation = t.get("relation", "unknown")
        total   += 1
        per_rel[relation]["total"] += 1

        if label == "correct":
            correct_full += 1
            per_rel[relation]["correct"] += 1
        elif label == "partial":
            correct_half += 1
            per_rel[relation]["partial"] += 1
        elif label == "wrong":
            wrong += 1
            per_rel[relation]["wrong"] += 1
        else:
            unannotated += 1

human_added = sum(
    len(item.get("human_added_triples", []))
    for item in annotations
)

annotated = total - unannotated

# ── 计算指标（两种口径）─────────────────────────────────────
# 口径1：严格 Precision（只有 correct 算对）
precision_strict = correct_full / annotated if annotated else 0

# 口径2：宽松 Precision（partial 按 0.5 分）
precision_loose  = (correct_full + correct_half * 0.5) / annotated if annotated else 0

# Recall：分母 = 正确召回 + 漏召（human_added）
recall_den_strict = correct_full + human_added
recall_strict     = correct_full / recall_den_strict if recall_den_strict else 0

recall_den_loose  = correct_full + correct_half * 0.5 + human_added
recall_loose      = (correct_full + correct_half * 0.5) / recall_den_loose if recall_den_loose else 0

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0

# ── 输出 ──────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  方法C（少样本LLM）三元组抽取评测结果")
print("═" * 60)
print(f"  标注总数:        {annotated} 条（未标注: {unannotated} 条）")
print(f"  correct:         {correct_full} 条  ({correct_full/annotated:.1%})")
print(f"  partial:         {correct_half} 条  ({correct_half/annotated:.1%})")
print(f"  wrong:           {wrong} 条  ({wrong/annotated:.1%})")
print(f"  人工补充漏召:    {human_added} 条")

print(f"\n  ── 严格口径（只有 correct 算对）──")
print(f"  Precision:  {precision_strict:.3f}")
print(f"  Recall:     {recall_strict:.3f}  [!] 无全量人工标注，保守估计")
print(f"  F1:         {f1(precision_strict, recall_strict):.3f}")

print(f"\n  ── 宽松口径（partial 按 0.5 分）──")
print(f"  Precision:  {precision_loose:.3f}")
print(f"  Recall:     {recall_loose:.3f}")
print(f"  F1:         {f1(precision_loose, recall_loose):.3f}")

print(f"\n  ── 分关系类型 Precision（严格口径）──")
print(f"  {'关系类型':<30} {'Prec':>6}  {'详情'}")
print(f"  {'-'*55}")
for rel, s in sorted(per_rel.items(), key=lambda x: -x[1]["total"]):
    p = s["correct"] / s["total"] if s["total"] else 0
    bar = "█" * int(p * 20)
    detail = f"correct={s['correct']} partial={s['partial']} wrong={s['wrong']} total={s['total']}"
    print(f"  {rel:<30} {p:>5.1%}  {detail}")

print("═" * 60)

# ── 保存结果供论文引用 ─────────────────────────────────────
import datetime
result = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "method": "method_c_few_shot",
    "annotated": annotated,
    "correct": correct_full,
    "partial": correct_half,
    "wrong": wrong,
    "human_added": human_added,
    "strict": {
        "precision": round(precision_strict, 4),
        "recall":    round(recall_strict, 4),
        "f1":        round(f1(precision_strict, recall_strict), 4),
    },
    "loose": {
        "precision": round(precision_loose, 4),
        "recall":    round(recall_loose, 4),
        "f1":        round(f1(precision_loose, recall_loose), 4),
    },
    "per_relation": {
        rel: {
            "precision": round(s["correct"]/s["total"], 4) if s["total"] else 0,
            **s
        }
        for rel, s in per_rel.items()
    }
}

out_path = Path("extraction_results/metrics_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 {out_path}")
