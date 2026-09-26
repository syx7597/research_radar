"""
交互式标注工具
用法: python annotate.py
  - 逐条显示三元组 + 原文证据 + 规则方法参考结果
  - 按 1=correct  2=wrong  3=partial  s=skip  q=保存退出
  - 支持断点续标：已有 label 的自动跳过
  - 完成后直接运行 compute_metrics() 即可出 P/R/F1
"""

import json, sys, os
from pathlib import Path

ANN_PATH = Path("extraction_results/annotation_template.json")

if not ANN_PATH.exists():
    # 兼容直接在本文件所在目录运行
    ANN_PATH = Path("annotation_template.json")
if not ANN_PATH.exists():
    print("❌ 找不到 annotation_template.json，请把本脚本放到项目根目录运行")
    sys.exit(1)

with open(ANN_PATH, encoding="utf-8") as f:
    records = json.load(f)

LABEL_MAP = {"1": "correct", "2": "wrong", "3": "partial"}

def save():
    with open(ANN_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def stats():
    done = sum(
        1 for r in records
        for t in r["llm_fewshot_triples_to_annotate"]
        if t["label"] in ("correct", "wrong", "partial")
    )
    total = sum(len(r["llm_fewshot_triples_to_annotate"]) for r in records)
    correct = sum(
        1 for r in records
        for t in r["llm_fewshot_triples_to_annotate"]
        if t["label"] == "correct"
    )
    return done, total, correct

def run():
    done, total, _ = stats()
    print(f"\n{'='*60}")
    print(f"  雷达知识图谱三元组标注工具")
    print(f"  已标注: {done}/{total}  剩余: {total-done}")
    print(f"  1=correct  2=wrong  3=partial  s=skip  q=保存退出")
    print(f"{'='*60}\n")

    count = 0
    for rec_idx, record in enumerate(records):
        radar = record["en_title"]
        triples = record["llm_fewshot_triples_to_annotate"]
        rule_ref = record.get("rule_triples_for_reference", [])

        for t_idx, triple in enumerate(triples):
            if triple["label"] in ("correct", "wrong", "partial"):
                continue  # 已标注，跳过

            count += 1
            done_now, total_now, correct_now = stats()
            print(f"\n[{done_now+1}/{total_now}]  雷达: {radar}")
            print(f"  三元组: {triple['head']}  --[{triple['relation']}]-->  {triple['tail']}")
            print(f"  证据  : {triple.get('evidence', '（无）')}")

            # 显示规则方法对同一关系的结果作为参考
            matching_rules = [
                r for r in rule_ref
                if r.get("relation") == triple["relation"]
            ]
            if matching_rules:
                rule_tails = ", ".join(r["tail"] for r in matching_rules)
                print(f"  规则参考({triple['relation']}): {rule_tails}")

            while True:
                key = input("  标注 (1/2/3/s/q): ").strip().lower()
                if key == "q":
                    save()
                    done_f, total_f, correct_f = stats()
                    print(f"\n已保存。进度: {done_f}/{total_f}，正确率参考: {correct_f/max(done_f,1):.1%}")
                    return
                elif key == "s":
                    break
                elif key in LABEL_MAP:
                    triple["label"] = LABEL_MAP[key]
                    # 顺手记录标注人和时间
                    import datetime
                    triple["annotated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    if count % 10 == 0:
                        save()
                        print("  [自动保存]")
                    break
                else:
                    print("  请输入 1/2/3/s/q")

    save()
    done_f, total_f, correct_f = stats()
    print(f"\n✅ 标注完成！共 {done_f} 条，预估精确率: {correct_f/max(done_f,1):.1%}")
    print("下一步: 在 triple_extraction_v2.py 里调用 compute_metrics() 计算 P/R/F1")

if __name__ == "__main__":
    run()
