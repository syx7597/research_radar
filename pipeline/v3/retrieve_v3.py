"""
v3 检索入口 —— 把检索器迁到 kg_v3。

复用 graphrag_retriever 的标准 4 层检索(BM25/向量/RRF/图扩展);v3 三元组用独立
索引(kg_v3/retrieval_index/),不动 v2 冻结索引(实验仍可复现)。叙述向量库
(NarrativeRetriever radar=)可选接入,供图命中雷达后拉其技术特点段。

  python pipeline/v3/retrieve_v3.py --build         # 建 v3 三元组索引
  python pipeline/v3/retrieve_v3.py --query "F-22 的雷达是谁研制的"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "v3" / "manual"))

from graphrag_retriever import HybridRetriever  # noqa: E402
import narrative_index as NI                      # noqa: E402

KGV3 = ROOT / "kg_v3"
V3_INDEX = KGV3 / "retrieval_index"


def load_kg_v3_triples():
    """kg_v3/edges.json → 检索器三元组格式(evidence list→str, 保留 tier/印证)。"""
    edges = json.loads((KGV3 / "edges.json").read_text(encoding="utf-8"))
    out = []
    for e in edges:
        ev = e.get("evidence", "")
        if isinstance(ev, list):
            ev = ev[0] if ev else ""
        out.append({
            "head": e["head"], "head_type": e.get("head_type", ""),
            "relation": e["relation"], "tail": e["tail"],
            "tail_type": e.get("tail_type", ""),
            "confidence": e.get("confidence", 0.7),
            "corroborated": e.get("corroborated", False),
            "tier": e.get("tier", ""), "evidence": ev,
        })
    return out


class V3Retriever:
    def __init__(self, enable_reranker=False):
        self.triples = load_kg_v3_triples()
        self.hybrid = HybridRetriever(self.triples, index_dir=V3_INDEX,
                                      enable_reranker=enable_reranker)
        try:
            self.narrative = NI.NarrativeRetriever()
        except Exception:
            self.narrative = None

    def retrieve(self, query, top_k=10, with_narrative=True, n_narr=3):
        """v3 混合检索(BM25+向量+RRF+图扩展) + 图文联动叙述召回。

        返回 {triples: [...], narrative: [...], entity_hits: [...]}。
        图检索命中的雷达 → 自动补召回它们的技术特点/分系统叙述段(手册最值钱的
        技术描述),这样"原理/结构"类问题也能答,而不只是结构化事实。"""
        res = self.hybrid.retrieve(query, rrf_top_k=top_k * 3,
                                   use_graph_expansion=True)
        triples = (res.get("reranked") or res.get("fused") or [])[:top_k]
        entity_hits = res.get("entity_hits", []) or []
        narrative = []
        if with_narrative and self.narrative and entity_hits:
            # 先在命中雷达内定向召回;不足再全局补
            for radar in entity_hits[:3]:
                narrative += self.narrative.retrieve(query, top_k=2, radar=radar)
            if len(narrative) < n_narr:
                narrative += self.narrative.retrieve(query, top_k=n_narr)
            seen, uniq = set(), []
            for c in narrative:
                k = (c["radar"], c["section"])
                if k not in seen:
                    seen.add(k); uniq.append(c)
            narrative = uniq[:n_narr]
        return {"triples": triples, "narrative": narrative,
                "entity_hits": entity_hits}


def main():
    if "--build" in sys.argv:
        r = V3Retriever()
        print(f"[v3检索] 索引 {len(r.triples)} 三元组 -> {V3_INDEX}")
        return
    if "--query" in sys.argv:
        q = sys.argv[sys.argv.index("--query") + 1]
        r = V3Retriever()
        print(f"\n=== v3 检索 : {q} ===")
        res = r.retrieve(q, top_k=6)
        print("── KG 结构事实 ──")
        for h in res["triples"]:
            print(f"  [conf={h.get('confidence',0):.2f} {h.get('tier','')}] "
                  f"{h['head']} -{h['relation']}-> {h['tail']}")
        if res["narrative"]:
            print("── 叙述技术细节(图文联动) ──")
            for c in res["narrative"]:
                print(f"  [{c['radar']}/{c['section']}] {c['text'][:80]}…")
    else:
        print("用法: --build | --query <问题>")


if __name__ == "__main__":
    main()
