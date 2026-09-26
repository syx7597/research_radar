"""
端到端问答系统 v2
运行: python qa_pipeline.py

修复：强制离线模式，避免每次启动时联网检查模型更新
"""

# ── 必须在所有 import 之前设置，阻止 huggingface 联网 ──────
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import json, requests
from pathlib import Path
import sys
# Moved to pipeline/eval/ in 2026-06-22 restructure: put repo root on sys.path so
# the root-level serving modules graphrag_retriever and reranker resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graphrag_retriever import load_triples_merged, HybridRetriever
from reranker import CrossEncoderReranker

# ══════════════════════════════════════════════════════════
#  配置——只需改这里
# ══════════════════════════════════════════════════════════

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TOP_K        = 8

# ══════════════════════════════════════════════════════════
#  初始化
# ══════════════════════════════════════════════════════════

print("正在加载索引...")
triples = load_triples_merged(
    "extraction_results/method_c_results.json",
    "extraction_results/method_a_results.json",
    prefer_source="llm_few_shot"
)
retriever = HybridRetriever(triples, enable_reranker=False)
reranker  = CrossEncoderReranker()
print("[OK] 加载完成\n")


# ══════════════════════════════════════════════════════════
#  LLM 调用
# ══════════════════════════════════════════════════════════

def call_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 512,
                  "temperature": 0.1},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM调用失败: {e}]"


SYSTEM_PROMPT = """你是雷达装备领域的专业知识库助手。
请严格根据提供的知识图谱证据回答问题。
要求：
1. 答案必须有证据支撑，不能推测或编造
2. 如果证据不足，明确说"根据现有知识库无法确定"
3. 回答简洁准确，直接给结论，再简要引用关键证据"""

MULTIHOP_SYSTEM_PROMPT = """你是雷达装备领域的专业知识库助手。
请严格根据提供的知识图谱证据，按步骤推理后回答问题。
要求：
1. 先分步列出推理链（每步引用一条证据）
2. 再给出最终结论
3. 若某步缺乏证据，明确说明"此步无法从知识库确定"
4. 不得推测或捏造"""

DECOMPOSE_PROMPT = """请将以下复杂问题拆解为2-3个独立的简单子问题，每个子问题只询问一个事实。
输出格式：每行一个子问题，不加序号或其他符号。

复杂问题：{question}

子问题："""

import re as _re


def _detect_question_type(question: str) -> str:
    """检测问题类型：comparison / multi_hop / single_hop。"""
    comp_patterns = [r"哪个更", r"哪种更", r"都.*吗", r"分别", r"比较", r"两者", r"哪个.*高", r"哪个.*低"]
    multi_patterns = [r"研制.*的.*雷达.*装备", r"装备了.*的.*哪", r"用.*雷达.*国家"]
    for p in comp_patterns:
        if _re.search(p, question):
            return "comparison"
    for p in multi_patterns:
        if _re.search(p, question):
            return "multi_hop"
    return "single_hop"


def decompose_query(question: str) -> list[str]:
    """
    将比较/多跳问题拆解为简单子问题列表。
    对于单跳问题直接返回 [question]。
    """
    qtype = _detect_question_type(question)
    if qtype == "single_hop":
        return [question]

    prompt = DECOMPOSE_PROMPT.format(question=question)
    try:
        raw = call_llm(prompt)
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip() and "？" in l]
        if lines:
            return lines[:3]  # 最多3个子问题
    except Exception:
        pass
    return [question]  # 降级：不拆解


# ══════════════════════════════════════════════════════════
#  主问答函数
# ══════════════════════════════════════════════════════════

def ask(question: str, top_k: int = TOP_K,
        use_reranker: bool = True, verbose: bool = False,
        use_decompose: bool = True) -> dict:
    """
    执行问答。对比较/多跳问题自动拆解为子问题后合并上下文。
    """
    qtype = _detect_question_type(question)

    # ── 子问题拆解（仅针对比较问题；多跳问题由图扩展处理） ──────
    if use_decompose and qtype == "comparison":
        sub_questions = decompose_query(question)
    else:
        sub_questions = [question]

    # ── 为每个子问题检索上下文 ────────────────────────────
    all_reranked: list[dict] = []
    all_graph_expanded: list[dict] = []

    for sq in sub_questions:
        result = retriever.retrieve(sq, top_k=top_k * 2, use_graph_expansion=True)
        reranked_sq = (
            reranker.rerank(sq, result["fused"], top_k=top_k)
            if use_reranker else result["fused"][:top_k]
        )
        all_reranked.extend(reranked_sq)
        all_graph_expanded.extend(result.get("graph_expanded", []))

    # 去重（保留最先出现的）
    seen_keys: set[tuple] = set()
    reranked: list[dict] = []
    for t in all_reranked:
        key = (t.get("head",""), t.get("relation",""), t.get("tail",""))
        if key not in seen_keys:
            seen_keys.add(key)
            reranked.append(t)
    reranked = reranked[:top_k]

    context = retriever._build_context(reranked, all_graph_expanded)

    if verbose:
        print(f"[问题类型: {qtype}]")
        if len(sub_questions) > 1:
            print(f"[子问题拆解]: {sub_questions}")

    # ── 选择 prompt 模板 ──────────────────────────────────
    if qtype == "multi_hop" and len(sub_questions) > 1:
        prompt = f"{MULTIHOP_SYSTEM_PROMPT}\n\n{context}\n\n问题：{question}\n推理与答案："
    else:
        prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n问题：{question}\n答案："

    answer = call_llm(prompt)

    result_last = retriever.retrieve(question, top_k=1, use_graph_expansion=False)
    return {
        "question":     question,
        "answer":       answer,
        "context":      context,
        "question_type": qtype,
        "sub_questions": sub_questions if len(sub_questions) > 1 else None,
        "retrieval": {
            "bm25":     len(result_last.get("bm25_results", [])),
            "vector":   len(result_last.get("vector_results", [])),
            "fused":    len(result_last.get("fused", [])),
            "reranked": len(reranked),
        },
    }


# ══════════════════════════════════════════════════════════
#  批量评测
# ══════════════════════════════════════════════════════════

def batch_evaluate(qa_path: str = "evaluation/qa_dataset.json",
                   top_k: int = 5, use_reranker: bool = True,
                   save_results: bool = True) -> dict:
    with open(qa_path, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    hits_by_type = {}
    all_results  = []

    for q in questions:
        qtype   = q.get("type", "unknown")
        answers = [q["answer"]] if isinstance(q["answer"], str) else list(q["answer"])
        answers += q.get("answer_aliases", [])
        out     = ask(q["question"], top_k=top_k, use_reranker=use_reranker)
        ctx_lower = out["context"].lower()
        hit     = any(str(a).lower() in ctx_lower for a in answers if a)
        hits_by_type.setdefault(qtype, []).append(hit)
        all_results.append({"id": q.get("id",""), "type": qtype,
                             "question": q["question"], "answer": q["answer"],
                             "hit": hit, "generated": out["answer"]})
        print(f"  [{'[OK]' if hit else '[X]'}][{qtype}] {q['question'][:45]}...")

    summary = {}
    total_h = total_q = 0
    for qt, hits in hits_by_type.items():
        r = sum(hits) / len(hits)
        summary[qt] = {"recall": round(r, 3), "count": len(hits)}
        total_h += sum(hits); total_q += len(hits)
    summary["overall"] = {"recall": round(total_h / max(total_q,1), 3), "count": total_q}

    print(f"\n{'='*45}")
    for qt, s in summary.items():
        print(f"  {qt:<15} Recall@{top_k}: {s['recall']:.1%}  ({s['count']} 条)")
    print(f"{'='*45}")

    if save_results:
        out_path = Path("evaluation/eval_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": all_results},
                      f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {out_path}")

    return {"summary": summary, "details": all_results}


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_questions = [
        "AN/TPY-2的工作频段是什么？",
        "Raytheon研制了哪些雷达系统？",
        "AN/APY-9雷达部署在哪种飞机上？",
        "有哪些雷达工作在X波段？",
    ]

    # for q in test_questions:
    #     print(f"\n{'─'*55}")
    #     print(f"问：{q}")
    #     result = ask(q, verbose=False)
    #     print(f"答：{result['answer']}")
    #     r = result['retrieval']
    #     print(f"   [BM25={r['bm25']} 向量={r['vector']} 融合={r['fused']} 精排={r['reranked']}]")

    # 跑完整评测时取消注释：
    batch_evaluate("evaluation/qa_dataset.json", top_k=5)
