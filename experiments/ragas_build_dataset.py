# -*- coding: utf-8 -*-
"""构建 RAGAS 评测数据集：抽样 qa_500 → 跑 v3 检索 → (question, contexts, gold)。
不依赖 ragas，先把上下文备好。输出 experiments/ragas_eval_input.json。"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "v3"))
sys.path.insert(0, str(ROOT / "pipeline" / "v3" / "manual"))
import retrieve_v3 as R  # noqa: E402

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40


def triple_str(h):
    return f"{h['head']} {h['relation']} {h['tail']}"


def main():
    qa = json.loads((ROOT / "evaluation" / "qa_500.json").read_text(encoding="utf-8"))["questions"]
    qa = [q for q in qa if q.get("gold_answer") not in (None, "")]  # 只留标量金标题
    # 按 type 分层抽样
    from collections import defaultdict
    byt = defaultdict(list)
    for q in qa:
        byt[q["type"]].append(q)
    rng = random.Random(42)
    per = max(1, N // len(byt))
    sample = []
    for t, items in byt.items():
        sample += rng.sample(items, min(per, len(items)))
    sample = sample[:N]

    r = R.V3Retriever()
    out = []
    for i, q in enumerate(sample):
        res = r.retrieve(q["question_zh"], top_k=8)
        ctx = [triple_str(h) for h in res["triples"]]
        ctx += [c["text"] for c in res.get("narrative", [])]
        out.append({
            "id": q["id"], "type": q["type"],
            "question": q["question_zh"],
            "contexts": ctx,
            "ground_truth": str(q["gold_answer"]),
        })
        print(f"[{i+1}/{len(sample)}] {q['type']:12} ctx={len(ctx):2} | {q['question_zh'][:30]}")

    (ROOT / "experiments" / "ragas_eval_input.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(out)} 条 -> experiments/ragas_eval_input.json")


if __name__ == "__main__":
    main()
