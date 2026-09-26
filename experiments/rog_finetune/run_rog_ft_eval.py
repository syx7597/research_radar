"""Local eval for RoG-FT: take the fine-tuned planner's pre-computed plans
(results/rog_ft_plans.json), run the SAME walker + DeepSeek reasoner + scorer as
the zero-shot RoG condition, and emit a 4-way per-type table
(Baseline / RoG-zeroshot / RoG-FT / Strategy).

Baseline/RoG-zeroshot/Strategy scores are reused verbatim from
results/qa500_3way_full.json (computed earlier with the same scorer); only RoG-FT
is newly computed here. Reasoner = DeepSeek (same model as all other conditions).

  python experiments/rog_finetune/run_rog_ft_eval.py
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from qa_strategy_pipeline import KGIndex, llm_call
from qa_rog_baseline import walk_path, render_rog_context, ROG_ANSWER_SYSTEM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import score_question

PLANS_PATH = ROOT / "results" / "rog_ft_plans.json"
QA_PATH = ROOT / "evaluation" / "qa_500.json"
FULL_PATH = ROOT / "results" / "qa500_3way_full.json"
KG_PATH = ROOT / "graphrag_index" / "merged_triples.json"
OUT_JSON = ROOT / "results" / "rog_ft_eval.json"
OUT_SUMMARY = ROOT / "results" / "rog_ft_4way_summary.md"


def conv_path(path_strs):
    out = []
    for tok in path_strs:
        tok = tok.strip()
        if tok.endswith("^-1"):
            out.append((tok[:-3].strip(), True))
        else:
            out.append((tok, False))
    return out


def rog_ft_answer(plan_rec, kg):
    start = plan_rec.get("start", "")
    paths = [conv_path(p) for p in plan_rec.get("paths", [])]
    walks = []
    if start and paths:
        for p in paths:
            walks.append(walk_path(start, p, kg))
    render_plan = {"start": start, "paths": paths}
    ctx = render_rog_context(render_plan, walks)
    user = f"问题：{plan_rec['question']}\n\n路径遍历证据：\n{ctx}\n\n请给出答案："
    answer = llm_call([{"role": "system", "content": ROG_ANSWER_SYSTEM},
                       {"role": "user", "content": user}], max_tokens=1200)
    return answer


def load_resume():
    if OUT_JSON.exists():
        data = json.load(open(OUT_JSON, encoding="utf-8"))
        return data, {r["id"] for r in data}
    return [], set()


def main():
    qa = {q["id"]: q for q in json.load(open(QA_PATH, encoding="utf-8"))["questions"]}
    plans = {p["id"]: p for p in json.load(open(PLANS_PATH, encoding="utf-8"))}
    full = {r["id"]: r for r in json.load(open(FULL_PATH, encoding="utf-8"))["per_question"]}
    triples = json.load(open(KG_PATH, encoding="utf-8"))
    kg = KGIndex(triples)

    results, done = load_resume()
    todo = [qid for qid in qa if qid not in done]
    print(f"[*] {len(qa)} questions, {len(done)} done, {len(todo)} to do")

    t0 = time.time()
    for i, qid in enumerate(todo, 1):
        q = qa[qid]
        plan_rec = plans[qid]
        try:
            ans = rog_ft_answer(plan_rec, kg)
            score = score_question(q, ans, kg)
        except Exception as e:
            ans = f"[ERR {e}]"
            score = {"correct": False, "error": str(e)}
        results.append({"id": qid, "type": q["type"], "question": q["question_zh"],
                        "rog_ft_answer": ans[:300] if isinstance(ans, str) else "",
                        "score": score})
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  {i}/{len(todo)}  elapsed={el/60:.1f}m eta={eta/60:.1f}m")
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- aggregate 4-way ----
    ft_by_id = {r["id"]: bool(r["score"].get("correct")) for r in results}
    by_t = defaultdict(lambda: {"b": [], "r": [], "ft": [], "s": []})
    for qid, r in full.items():
        t = r["type"]
        by_t[t]["b"].append(bool(r["baseline"]["score"].get("correct")))
        by_t[t]["r"].append(bool(r["rog_style"]["score"].get("correct")))
        by_t[t]["s"].append(bool(r["strategy"]["score"].get("correct")))
        by_t[t]["ft"].append(ft_by_id.get(qid, False))

    lines = ["# RoG-FT 4-way Results (Baseline / RoG-zeroshot / RoG-FT / Strategy)", ""]
    lines.append("| Type | n | Baseline | RoG-zs | RoG-FT | Strategy | Δ FT-zs | Δ S-FT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    tot = defaultdict(int)
    for t in sorted(by_t):
        d = by_t[t]
        n = len(d["b"])
        b, r, ft, s = sum(d["b"]), sum(d["r"]), sum(d["ft"]), sum(d["s"])
        tot["n"] += n; tot["b"] += b; tot["r"] += r; tot["ft"] += ft; tot["s"] += s
        lines.append(f"| {t} | {n} | {b/n*100:.1f} | {r/n*100:.1f} | {ft/n*100:.1f} | "
                     f"{s/n*100:.1f} | {(ft-r)/n*100:+.1f} | {(s-ft)/n*100:+.1f} |")
    n = tot["n"]
    lines.append(f"| **OVERALL** | **{n}** | **{tot['b']/n*100:.1f}** | **{tot['r']/n*100:.1f}** | "
                 f"**{tot['ft']/n*100:.1f}** | **{tot['s']/n*100:.1f}** | "
                 f"**{(tot['ft']-tot['r'])/n*100:+.1f}** | **{(tot['s']-tot['ft'])/n*100:+.1f}** |")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
