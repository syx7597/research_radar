"""
Cross-Encoder 重排模块 v1.0
雷达知识图谱检索链路 - 精排层

在 GraphRAG 检索链路中的位置：
  BM25 + 向量检索（粗排）→ RRF 融合 → [本模块] Cross-Encoder 重排（精排）→ LLM

原理：
  Bi-Encoder（向量检索）：query 和文档分开 encode，速度快但精度低
  Cross-Encoder（本模块）：query 和文档拼接后一起 encode，精度高但慢
  → 先用 Bi-Encoder 粗筛到 10-20 条，再用 Cross-Encoder 精排到 5 条

推荐模型（按效果排序）：
  BAAI/bge-reranker-base         中英混合，适合雷达数据（推荐，约 280MB）
  BAAI/bge-reranker-large        效果更好，约 560MB
  cross-encoder/ms-marco-MiniLM-L-6-v2  英文专用，更小更快

安装：
  pip install sentence-transformers

用法（直接使用）：
  from reranker import CrossEncoderReranker
  reranker = CrossEncoderReranker()
  reranked = reranker.rerank(query="AN/SPY-1的频段", candidates=triples, top_k=5)

用法（集成到 HybridRetriever）：
  from graphrag_retriever import HybridRetriever
  retriever = HybridRetriever(triples, enable_reranker=True)
  result    = retriever.retrieve("AN/SPY-1的频段", top_k=5)
  # result["reranked"] 是精排后的结果
  # result["context"]  已经基于精排结果生成
"""

import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  三元组 → 用于重排的文本
# ═══════════════════════════════════════════════════════

def triple_to_rerank_text(t: dict) -> str:
    """
    把三元组转成 Cross-Encoder 的输入文本。
    格式比 embedding 用的更丰富：包含实体类型和原文证据，
    让 Cross-Encoder 有足够信息做精细评分。
    """
    head      = t.get("head", "")
    head_type = t.get("head_type", "")
    relation  = t.get("relation", "")
    tail      = t.get("tail", "")
    tail_type = t.get("tail_type", "")
    evidence  = t.get("evidence", "")

    REL_ZH = {
        "deployedOn":       "部署于",
        "developedBy":      "由…研制",
        "operatedBy":       "装备于",
        "exportedTo":       "出口至",
        "upgradeOf":        "是…的升级版",
        "derivedFrom":      "衍生自",
        "competitorOf":     "竞争型号为",
        "coDeployedWith":   "与…同平台配套",
        "hasFrequencyBand": "工作频段为",
        "hasFunction":      "具备功能",
        "affiliatedTo":     "隶属于",
    }
    rel_zh = REL_ZH.get(relation, relation)

    parts = [f"{head}（{head_type}）{rel_zh} {tail}（{tail_type}）"]
    if evidence:
        parts.append(f"原文依据：{evidence[:150]}")
    return "。".join(parts)


# ═══════════════════════════════════════════════════════
#  Cross-Encoder 重排器
# ═══════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    基于 Cross-Encoder 的精排器。

    参数：
        model_name      模型名（Hugging Face Hub）
        batch_size      批量打分大小，内存不够时调小
        max_length      输入截断长度（token 数）
        score_threshold 低于此分数的三元组直接过滤（None = 不过滤）
        cache_dir       模型缓存目录（默认 ~/.cache/huggingface）
    """

    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-base",
                 batch_size: int = 32,
                 max_length: int = 512,
                 score_threshold: Optional[float] = None,
                 cache_dir: Optional[str] = None):

        self.model_name      = model_name
        self.batch_size      = batch_size
        self.max_length      = max_length
        self.score_threshold = score_threshold
        self._model          = None   # 懒加载

        self._cache_dir = cache_dir

        log.info(f"CrossEncoderReranker 初始化（模型：{model_name}，懒加载）")

    @property
    def model(self):
        """懒加载：第一次 rerank 时才载入模型，避免影响启动速度。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info(f"加载 Cross-Encoder 模型：{self.model_name}")
            t0 = time.time()
            kwargs = {}
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                **kwargs
            )
            log.info(f"Cross-Encoder 加载完成，耗时 {time.time()-t0:.1f}s")
        return self._model

    def rerank(self,
               query: str,
               candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        对候选三元组列表做精排。

        参数：
            query       用户查询
            candidates  待精排的三元组列表（来自 RRF 融合结果）
            top_k       返回前 K 条

        返回：
            按 rerank_score 降序排列的三元组列表，每条新增：
              rerank_score   Cross-Encoder 打分（越高越相关）
              rerank_rank    精排后排名（从 1 开始）
        """
        if not candidates:
            return []

        # 构造 (query, document) 对
        pairs = [
            (query, triple_to_rerank_text(t))
            for t in candidates
        ]

        t0 = time.time()
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        elapsed = time.time() - t0
        log.debug(
            f"Cross-Encoder 打分：{len(pairs)} 条，耗时 {elapsed*1000:.0f}ms"
        )

        # 把分数附加到三元组
        scored = []
        for t, score in zip(candidates, scores):
            t = dict(t)
            t["rerank_score"] = float(score)
            scored.append(t)

        # 按分数降序排列
        scored.sort(key=lambda x: -x["rerank_score"])

        # 阈值过滤
        if self.score_threshold is not None:
            scored = [t for t in scored if t["rerank_score"] >= self.score_threshold]

        # 取 top_k，附上排名
        result = []
        for rank, t in enumerate(scored[:top_k], start=1):
            t["rerank_rank"] = rank
            result.append(t)

        return result

    def rerank_with_analysis(self,
                              query: str,
                              candidates: list[dict],
                              top_k: int = 5) -> dict:
        """
        精排 + 顺序变化分析（调试/论文写作用）。

        返回：
            reranked        精排后结果
            order_changes   每条三元组的排名变化（RRF排名 → 精排排名）
            promoted        精排后排名上升最多的三元组（体现精排价值）
            demoted         精排后排名下降最多的三元组
        """
        reranked = self.rerank(query, candidates, top_k=len(candidates))

        # 计算顺序变化
        rrf_rank  = {
            (t.get("head",""), t.get("relation",""), t.get("tail","")): i+1
            for i, t in enumerate(candidates)
        }
        changes = []
        for t in reranked[:top_k]:
            key       = (t.get("head",""), t.get("relation",""), t.get("tail",""))
            old_rank  = rrf_rank.get(key, len(candidates))
            new_rank  = t["rerank_rank"]
            delta     = old_rank - new_rank   # 正数 = 上升，负数 = 下降
            changes.append({
                "triple":   f"{t.get('head','')} --[{t.get('relation','')}]--> {t.get('tail','')}",
                "rrf_rank":    old_rank,
                "rerank_rank": new_rank,
                "delta":       delta,
                "rerank_score": t["rerank_score"],
            })

        promoted = sorted(changes, key=lambda x: -x["delta"])[:3]
        demoted  = sorted(changes, key=lambda x:  x["delta"])[:3]

        return {
            "reranked":      reranked[:top_k],
            "order_changes": changes,
            "promoted":      promoted,
            "demoted":       demoted,
        }


# ═══════════════════════════════════════════════════════
#  消融实验对比工具
# ═══════════════════════════════════════════════════════

def compare_rrf_vs_reranker(retriever,
                             reranker: CrossEncoderReranker,
                             queries: list[str],
                             ground_truth: dict[str, list[str]],
                             top_k: int = 5) -> dict:
    """
    对比 RRF 融合 vs Cross-Encoder 精排的命中率。
    用于消融实验：证明精排层有贡献。

    参数：
        retriever     HybridRetriever 实例
        reranker      CrossEncoderReranker 实例
        queries       查询列表
        ground_truth  {query: [正确答案列表]}
        top_k         评测 Recall@K

    返回：
        {
          "rrf_recall":     RRF 融合的 Recall@K,
          "rerank_recall":  Cross-Encoder 精排的 Recall@K,
          "delta":          提升幅度,
          "per_query":      每条查询的详细结果
        }
    """
    rrf_hits    = 0
    rerank_hits = 0
    per_query   = []

    for query in queries:
        answers = ground_truth.get(query, [])
        if not answers:
            continue

        # 粗排
        result     = retriever.retrieve(query, top_k=top_k * 3,
                                         use_graph_expansion=False)
        fused      = result["fused"]

        # 精排
        reranked   = reranker.rerank(query, fused, top_k=top_k)

        def hit(triples: list[dict]) -> bool:
            ctx = " ".join(
                f"{t.get('head','')} {t.get('tail','')}"
                for t in triples
            ).lower()
            return any(str(a).lower() in ctx for a in answers)

        rrf_hit    = hit(fused[:top_k])
        rerank_hit = hit(reranked)

        rrf_hits    += int(rrf_hit)
        rerank_hits += int(rerank_hit)

        per_query.append({
            "query":       query,
            "rrf_hit":     rrf_hit,
            "rerank_hit":  rerank_hit,
            # 精排改变了结果的 case（最有分析价值）
            "changed":     rrf_hit != rerank_hit,
        })

    n = len(per_query)
    rrf_recall    = rrf_hits    / n if n else 0
    rerank_recall = rerank_hits / n if n else 0

    return {
        "rrf_recall":    round(rrf_recall, 3),
        "rerank_recall": round(rerank_recall, 3),
        "delta":         round(rerank_recall - rrf_recall, 3),
        "n_queries":     n,
        "per_query":     per_query,
        # 精排改变结果的 case（changed=True）是最有价值的分析对象
        "changed_cases": [q for q in per_query if q["changed"]],
    }
