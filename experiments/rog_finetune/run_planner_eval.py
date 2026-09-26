"""Generic planner-condition eval: take a plans file (start+paths per question),
run the SAME walker + DeepSeek reasoner + scorer, save per-id scores.

  python run_planner_eval.py --plans results/rog_qwen_zs_plans.json \
                             --outjson results/rog_qwen_zs_eval.json --label qwen_zs
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa_strategy_pipeline import KGIndex, llm_call
from qa_rog_baseline import walk_path, render_rog_context, ROG_ANSWER_SYSTEM
from scorer import score_question

QA_PATH = ROOT / "evaluation" / "qa_500.json"
KG_PATH = ROOT / "graphrag_index" / "merged_triples.json"


def conv_path(path_strs):
    out = []
    for tok in path_strs:
        tok = tok.strip()
        if tok.endswith("^-1"):
            out.append((tok[:-3].strip(), True))
        else:
            out.append((tok, False))
    return out


def planner_answer(plan_rec, kg):
    start = plan_rec.get("start", "")
    paths = [conv_path(p) for p in plan_rec.get("paths", [])]
    walks = [walk_path(start, p, kg) for p in paths] if (start and paths) else []
    ctx = render_rog_context({"start": start, "paths": paths}, walks)
    user = f"问题：{plan_rec['question']}\n\n路径遍历证据：\n{ctx}\n\n请给出答案："
    return llm_call([{"role": "system", "content": ROG_ANSWER_SYSTEM},
                     {"role": "user", "content": user}], max_tokens=1200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--outjson", required=True)
    ap.add_argument("--label", default="cond")
    args = ap.parse_args()

    qa = {q["id"]: q for q in json.load(open(QA_PATH, encoding="utf-8"))["questions"]}
    plans = {p["id"]: p for p in json.load(open(ROOT / args.plans, encoding="utf-8"))}
    kg = KGIndex(json.load(open(KG_PATH, encoding="utf-8")))

    outp = ROOT / args.outjson
    results, done = ([], set())
    if outp.exists():
        results = json.load(open(outp, encoding="utf-8"))
        done = {r["id"] for r in results}

    todo = [qid for qid in qa if qid not in done and qid in plans]
    print(f"[*] {args.label}: {len(qa)} q, {len(done)} done, {len(todo)} to do")
    t0 = time.time()
    for i, qid in enumerate(todo, 1):
        q = qa[qid]
        try:
            ans = planner_answer(plans[qid], kg)
            score = score_question(q, ans, kg)
        except Exception as e:
            ans, score = f"[ERR {e}]", {"correct": False, "error": str(e)}
        results.append({"id": qid, "type": q["type"],
                        "answer": ans[:300] if isinstance(ans, str) else "", "score": score})
        if i % 25 == 0 or i == len(todo):
            json.dump(results, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  {i}/{len(todo)}  {(time.time()-t0)/60:.1f}m")
    json.dump(results, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    by_t = defaultdict(list)
    for r in results:
        by_t[r["type"]].append(bool(r["score"].get("correct")))
    print(f"\n=== {args.label} per-type ===")
    tot = [0, 0]
    for t in sorted(by_t):
        c, n = sum(by_t[t]), len(by_t[t])
        tot[0] += c; tot[1] += n
        print(f"  {t:18s} {c/n*100:5.1f}  (n={n})")
    print(f"  {'OVERALL':18s} {tot[0]/tot[1]*100:5.1f}  (n={tot[1]})")
    print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
