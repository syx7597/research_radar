"""
GraphRAG 检索模块 v1.0
雷达知识图谱 - 混合检索链路

架构：
  层1  BM25 关键词检索
  层2  FAISS 向量检索
  层3  RRF 融合（倒数排名融合）
  层4  图扩展（NetworkX 内存图 / Neo4j 可选）

安装依赖：
  pip install faiss-cpu rank-bm25 sentence-transformers networkx

快速开始：
  from graphrag_retriever import HybridRetriever, load_triples_from_results
  triples   = load_triples_from_results("extraction_results/method_c_results.json")
  retriever = HybridRetriever(triples)
  result    = retriever.retrieve("AN/SPY-1的工作频段是什么？")
  print(result["context"])
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INDEX_DIR = Path("graphrag_index")
INDEX_DIR.mkdir(exist_ok=True)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def load_triples_from_results(results_path: str) -> list[dict]:
    """从 method_x_results.json 中展平所有三元组。"""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件: {results_path}")
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    triples = []
    for record in results:
        for t in record.get("triples", []):
            t = dict(t)
            t.setdefault("radar_id", record.get("radar_id", ""))
            t.setdefault("en_title", record.get("en_title", ""))
            triples.append(t)
    log.info(f"加载三元组: {len(triples)} 条（来自 {len(results)} 个雷达）")
    return triples


def load_triples_merged(*result_paths: str,
                         prefer_source: str = "llm_few_shot") -> list[dict]:
    """合并多路抽取结果，去重时优先保留 prefer_source 来源的高置信度条目。"""
    all_triples: list[dict] = []
    for path in result_paths:
        try:
            all_triples.extend(load_triples_from_results(path))
        except FileNotFoundError as e:
            log.warning(str(e))

    best: dict[tuple, dict] = {}
    for t in all_triples:
        key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
        existing = best.get(key)
        if existing is None:
            best[key] = t
        else:
            prefer_new = (
                t.get("source") == prefer_source and
                existing.get("source") != prefer_source
            )
            higher_conf = (
                t.get("source") == existing.get("source") and
                t.get("confidence", 0) > existing.get("confidence", 0)
            )
            if prefer_new or higher_conf:
                best[key] = t

    merged = list(best.values())
    log.info(f"合并后: {len(merged)} 条（去重前 {len(all_triples)} 条）")
    return merged


def triple_to_text(t: dict) -> str:
    """三元组 → 可 embedding 的自然语言句子。"""
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
    rel  = REL_ZH.get(t.get("relation", ""), t.get("relation", ""))
    base = f"{t.get('head','')} {rel} {t.get('tail','')}"
    ev   = t.get("evidence", "")
    return f"{base}。{ev[:100]}" if ev else base


# ═══════════════════════════════════════════════════════
#  层1：BM25 关键词检索
# ═══════════════════════════════════════════════════════

class BM25Retriever:

    def __init__(self, triples: list[dict]):
        from rank_bm25 import BM25Okapi
        self.triples = triples
        tokenized    = [self._tokenize(triple_to_text(t)) for t in triples]
        self.bm25    = BM25Okapi(tokenized)
        log.info(f"BM25 索引构建完成: {len(triples)} 条")

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.findall(r'[a-zA-Z0-9/\-\.]+|[\u4e00-\u9fff]', text)
        stop   = {"的","了","在","是","和","与","为","于","由","其",
                  "a","an","the","is","are","was","were","of","in",
                  "on","by","to","for","and","or"}
        return [t.lower() for t in tokens if t.lower() not in stop]

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores  = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                t = dict(self.triples[idx])
                t["bm25_score"] = float(scores[idx])
                results.append(t)
        return results


# ═══════════════════════════════════════════════════════
#  层2：FAISS 向量检索
# ═══════════════════════════════════════════════════════

class VectorRetriever:

    def __init__(self, triples: list[dict],
                 model_name: str = DEFAULT_EMBED_MODEL,
                 index_dir: Path = INDEX_DIR):
        self.triples    = triples
        self.model_name = model_name
        self._model     = None

        index_dir.mkdir(exist_ok=True)
        self._index_path = index_dir / "faiss.index"
        self._meta_path  = index_dir / "faiss_meta.pkl"

        if self._index_path.exists() and self._meta_path.exists():
            self._load_index()
        else:
            self._build_index()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info(f"加载 embedding 模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_index(self):
        import faiss
        texts = [triple_to_text(t) for t in self.triples]
        log.info(f"构建 FAISS 索引: {len(texts)} 条...")
        t0 = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim, n = embeddings.shape[1], len(embeddings)

        if n < 2000:
            self.index = faiss.IndexFlatIP(dim)
        else:
            nlist      = min(int(n ** 0.5), 256)
            quantizer  = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = max(nlist // 4, 1)

        self.index.add(embeddings)
        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.triples, f)

        log.info(f"FAISS 索引完成，dim={dim}，耗时 {time.time()-t0:.1f}s")

    def _load_index(self):
        import faiss
        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            cached = pickle.load(f)
        if len(cached) != len(self.triples):
            log.warning(f"三元组数量变化，重建索引")
            self._build_index()
        else:
            log.info(f"FAISS 索引加载完成: {self.index.ntotal} 条向量")

    def rebuild(self):
        for p in [self._index_path, self._meta_path]:
            if p.exists():
                p.unlink()
        self._build_index()

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            t = dict(self.triples[idx])
            t["vector_score"] = float(score)
            results.append(t)
        return results


# ═══════════════════════════════════════════════════════
#  层3：RRF 融合
# ═══════════════════════════════════════════════════════

def infer_answer_type(query: str) -> Optional[str]:
    """
    从查询中推断期望的答案实体类型（tail_type）。
    用于 type-aware 检索评分加权。
    返回 RELATION_TYPES 中的 tail_type 值，或 None（无法推断时）。
    """
    patterns: list[tuple[str, str]] = [
        # 制造商类问题
        (r"由哪家公司|研制方|制造商|谁研制|哪个公司", "Manufacturer"),
        # 国家类问题
        (r"哪些国家|哪个国家|装备于|出口到|服役于|由哪国", "Country"),
        # 频段类问题
        (r"频段|工作频率|波段|GHz|MHz|什么频", "FrequencyBand"),
        # 平台类问题
        (r"哪种平台|哪类平台|部署在|装备在|搭载于|哪艘|哪架", "Platform"),
        # 雷达系统类问题（升级/竞争/衍生）
        (r"升级版|升级自|前身|后继型|竞争型号|衍生自|哪个型号", "RadarSystem"),
        # 功能类问题
        (r"什么功能|哪些功能|用于|具备", "FunctionDomain"),
    ]
    import re
    for pat, etype in patterns:
        if re.search(pat, query):
            return etype
    return None


def rrf_fusion(lists: list[list[dict]], k: int = 60,
               query: Optional[str] = None,
               type_boost: float = 0.30) -> list[dict]:
    """
    倒数排名融合：score(d) = Σ 1/(k + rank_i(d))

    当 query 不为 None 时，启用 type-aware 加权：
    若三元组的 tail_type 与查询期望类型匹配，分数乘以 (1 + type_boost)。
    使用乘法而非加法，避免类型匹配压倒 BM25/vector 相关性分数。
    """
    def key(t: dict) -> tuple:
        return (t.get("head",""), t.get("relation",""), t.get("tail",""))

    # 推断期望答案类型（用于 type-aware 加权）
    expected_type: Optional[str] = None
    if query:
        expected_type = infer_answer_type(query)

    scores: dict[tuple, dict] = {}
    for ranked_list in lists:
        for rank, t in enumerate(ranked_list):
            k_ = key(t)
            if k_ not in scores:
                scores[k_] = {"triple": t, "rrf": 0.0}
            scores[k_]["rrf"] += 1.0 / (k + rank + 1)

    # type-aware 加权：tail_type 匹配时乘以 (1 + type_boost)，保持相对排序稳定
    if expected_type:
        for item in scores.values():
            tail_type = item["triple"].get("tail_type", "")
            if tail_type == expected_type or (
                expected_type == "Platform" and
                tail_type in ("NavalVessel", "AircraftPlatform", "GroundPlatform")
            ):
                item["rrf"] *= (1.0 + type_boost)

    merged = sorted(scores.values(), key=lambda x: -x["rrf"])
    result = []
    for item in merged:
        t = dict(item["triple"])
        t["rrf_score"] = round(item["rrf"], 6)
        t.pop("bm25_score",   None)
        t.pop("vector_score", None)
        result.append(t)
    return result


# ═══════════════════════════════════════════════════════
#  层4：图扩展
# ═══════════════════════════════════════════════════════

class InMemoryGraphExpander:
    """NetworkX 内存图扩展，不依赖 Neo4j。"""

    def __init__(self, triples: list[dict], min_confidence: float = 0.5):
        import networkx as nx
        self.G = nx.MultiDiGraph()
        for t in triples:
            if t.get("confidence", 0.5) < min_confidence:
                continue
            self.G.add_edge(
                t.get("head",""), t.get("tail",""),
                relation   = t.get("relation",""),
                confidence = t.get("confidence", 0.5),
                evidence   = t.get("evidence",""),
                head_type  = t.get("head_type",""),
                tail_type  = t.get("tail_type",""),
            )
        log.info(
            f"内存图: {self.G.number_of_nodes()} 节点, "
            f"{self.G.number_of_edges()} 边"
        )

    def expand(self, entity_names: list[str], hops: int = 2) -> list[dict]:
        visited: set[tuple] = set()
        result:  list[dict] = []
        frontier = set(entity_names)

        for _ in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                if node not in self.G:
                    continue
                for _, nbr, data in self.G.out_edges(node, data=True):
                    ek = (node, data["relation"], nbr)
                    if ek not in visited:
                        visited.add(ek)
                        result.append({
                            "head": node, "head_type": data.get("head_type",""),
                            "relation": data["relation"],
                            "tail": nbr, "tail_type": data.get("tail_type",""),
                            "confidence": data.get("confidence", 0.5),
                            "evidence": data.get("evidence",""),
                            "source": "graph_expansion",
                        })
                    nxt.add(nbr)
                for pred, _, data in self.G.in_edges(node, data=True):
                    ek = (pred, data["relation"], node)
                    if ek not in visited:
                        visited.add(ek)
                        result.append({
                            "head": pred, "head_type": data.get("head_type",""),
                            "relation": data["relation"],
                            "tail": node, "tail_type": data.get("tail_type",""),
                            "confidence": data.get("confidence", 0.5),
                            "evidence": data.get("evidence",""),
                            "source": "graph_expansion",
                        })
                    nxt.add(pred)
            frontier = nxt - set(entity_names)

        return result


class Neo4jGraphExpander:
    """Neo4j 图扩展（需要运行中的 Neo4j 实例）。"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with self.driver.session() as s:
            s.run("RETURN 1")
        log.info("Neo4j 连接成功")

    def expand(self, entity_names: list[str],
               hops: int = 2, min_confidence: float = 0.5) -> list[dict]:
        query = """
        UNWIND $names AS name
        MATCH (n {name: name})
        CALL apoc.path.subgraphNodes(n, {maxLevel: $hops}) YIELD node
        MATCH (node)-[r]->(m)
        WHERE coalesce(r.confidence, 0.5) >= $min_conf
        RETURN node.name AS head, node.type AS head_type,
               type(r) AS relation,
               m.name AS tail, m.type AS tail_type,
               r.confidence AS confidence, r.evidence AS evidence
        LIMIT 200
        """
        with self.driver.session() as session:
            rows = [dict(r) for r in session.run(
                query, names=entity_names, hops=hops, min_conf=min_confidence
            )]
        for r in rows:
            r["source"] = "graph_expansion_neo4j"
        return rows

    def close(self):
        self.driver.close()


# ═══════════════════════════════════════════════════════
#  主检索器
# ═══════════════════════════════════════════════════════

class HybridRetriever:
    """
    完整检索链路：BM25 + 向量 + RRF 融合 + 图扩展 + Cross-Encoder 精排。

    检索流程：
      ① BM25 召回 Top-30
      ② 向量检索 召回 Top-30
      ③ RRF 融合 → Top-10（粗排结果）
      ④ 图扩展：从 Top-5 实体出发 1-2 跳扩展
      ⑤ Cross-Encoder 重排 → Top-5（精排结果，可选）

    参数：
      triples          所有三元组
      model_name       bi-encoder 模型（FAISS 用）
      neo4j_config     dict(uri, user, password)，不传则用 NetworkX 内存图
      graph_hops       图扩展跳数
      min_graph_conf   图扩展置信度过滤阈值
      enable_reranker  是否开启 Cross-Encoder 精排（默认 False，懒加载）
      reranker_model   Cross-Encoder 模型名
    """

    def __init__(self,
                 triples: list[dict],
                 model_name: str = DEFAULT_EMBED_MODEL,
                 neo4j_config: Optional[dict] = None,
                 graph_hops: int = 2,
                 min_graph_conf: float = 0.5,
                 enable_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-base",
                 index_dir: Path = INDEX_DIR):

        self.triples    = triples
        self.graph_hops = graph_hops

        log.info("初始化 BM25...")
        self.bm25 = BM25Retriever(triples)

        log.info("初始化向量检索...")
        self.vector = VectorRetriever(triples, model_name, index_dir=index_dir)

        if neo4j_config:
            try:
                self.graph_expander = Neo4jGraphExpander(**neo4j_config)
            except Exception as e:
                log.warning(f"Neo4j 不可用 ({e})，降级为内存图")
                self.graph_expander = InMemoryGraphExpander(triples, min_graph_conf)
        else:
            log.info("使用 NetworkX 内存图扩展")
            self.graph_expander = InMemoryGraphExpander(triples, min_graph_conf)

        # Cross-Encoder 精排（懒加载：初始化时不载入模型，第一次 retrieve 时才载入）
        self.reranker = None
        if enable_reranker:
            from reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(model_name=reranker_model)
            log.info(f"Cross-Encoder 已启用（{reranker_model}，懒加载）")

        log.info("HybridRetriever 初始化完成 [OK]")

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 bm25_top_k: int = 30,
                 vec_top_k: int = 30,
                 rrf_top_k: int = 10,
                 use_graph_expansion: bool = True,
                 use_reranker: Optional[bool] = None,
                 graph_hops: Optional[int] = None) -> dict:
        """
        执行完整检索链路。

        参数：
          top_k           最终返回条数（精排后）
          bm25_top_k      BM25 粗召回数量
          vec_top_k       向量粗召回数量
          rrf_top_k       RRF 融合后保留数量（送给 Cross-Encoder 的候选集大小）
          use_graph_expansion  是否做图扩展
          use_reranker    是否使用 Cross-Encoder（None = 跟随初始化设置）

        返回字段：
          query           原始查询
          bm25_results    BM25 粗召回
          vector_results  向量粗召回
          fused           RRF 融合后（粗排结果）
          graph_expanded  图扩展补充
          reranked        Cross-Encoder 精排后（若未启用则等于 fused[:top_k]）
          context         组装好的 LLM 输入上下文（基于 reranked）
          entity_hits     精排后命中的实体列表
          pipeline_used   实际使用的链路描述（方便日志和调试）
        """
        # ── 步骤1：BM25 + 向量并行粗召回 ──────────────────────
        bm25_res   = self.bm25.retrieve(query, top_k=bm25_top_k)
        vector_res = self.vector.retrieve(query, top_k=vec_top_k)

        # ── 步骤2：RRF 融合（粗排，启用 type-aware 加权） ────────
        fused = rrf_fusion([bm25_res, vector_res], query=query)[:rrf_top_k]

        # ── 步骤3：图扩展 ──────────────────────────────────────
        graph_res: list[dict] = []
        if use_graph_expansion and self.graph_expander:
            seeds = list(
                {t["head"] for t in fused[:5] if t.get("head")} |
                {t["tail"] for t in fused[:5] if t.get("tail")}
            )
            hops  = graph_hops or self.graph_hops
            try:
                graph_res = self.graph_expander.expand(seeds, hops=hops)
                log.debug(f"图扩展: {len(seeds)} 实体 → {len(graph_res)} 条")
            except Exception as e:
                log.warning(f"图扩展失败: {e}")

        # 图扩展结果合并到候选池（去重后送给精排）
        # 注意：graph_res 不参与 RRF，而是作为补充候选直接拼接
        all_candidates = fused + [
            t for t in graph_res
            if (t.get("head",""), t.get("relation",""), t.get("tail",""))
            not in {(c["head"], c["relation"], c["tail"]) for c in fused}
        ]

        # ── 步骤4：Cross-Encoder 精排（可选） ─────────────────
        do_rerank = use_reranker if use_reranker is not None else (self.reranker is not None)

        if do_rerank and self.reranker:
            reranked = self.reranker.rerank(query, all_candidates, top_k=top_k)
            pipeline_used = "BM25 + Vector → RRF → GraphExpand → CrossEncoder"
        else:
            # 无精排时直接取 RRF top_k
            reranked = [dict(t) for t in all_candidates[:top_k]]
            for i, t in enumerate(reranked, start=1):
                t["rerank_rank"] = i
            pipeline_used = "BM25 + Vector → RRF → GraphExpand"

        # ── 组装输出 ───────────────────────────────────────────
        return {
            "query":          query,
            "bm25_results":   bm25_res[:top_k],
            "vector_results": vector_res[:top_k],
            "fused":          fused,
            "graph_expanded": graph_res,
            "reranked":       reranked,          # ← 精排后结果（最终答案依据）
            "context":        self._build_context(reranked, graph_res),
            "entity_hits":    list({t["head"] for t in reranked}),
            "pipeline_used":  pipeline_used,
        }

    def _build_context(self,
                       reranked: list[dict],
                       graph_expanded: list[dict],
                       max_triples: int = 20) -> str:
        """
        基于精排结果（reranked）组装 LLM 上下文。
        graph_expanded 中额外的补充条目附在后面。
        """
        lines = ["【检索到的图谱证据】\n"]
        seen:  set[tuple] = set()
        count = 0

        # 精排结果优先
        primary = reranked
        # 图扩展中精排未覆盖的条目作为补充
        reranked_keys = {
            (t.get("head",""), t.get("relation",""), t.get("tail",""))
            for t in reranked
        }
        supplementary = [
            t for t in graph_expanded
            if (t.get("head",""), t.get("relation",""), t.get("tail",""))
            not in reranked_keys
        ]

        for t in primary + supplementary:
            if count >= max_triples:
                break
            key = (t.get("head",""), t.get("relation",""), t.get("tail",""))
            if key in seen:
                continue
            seen.add(key)

            # 优先显示精排分数，无则用置信度
            score    = t.get("rerank_score", t.get("rrf_score", t.get("confidence", 0.0)))
            src_tag  = " [图扩展]" if "graph" in t.get("source","") else ""
            evidence = t.get("evidence","")

            line = (
                f"[{count+1:02d}] "
                f"({t.get('head_type','?')}) {t.get('head','')} "
                f"--[{t.get('relation','')}]--> "
                f"({t.get('tail_type','?')}) {t.get('tail','')}"
                f"  [相关度:{score:.3f}]{src_tag}"
            )
            if evidence:
                line += f"\n      证据: {evidence[:120]}"
            lines.append(line)
            count += 1

        if count == 0:
            lines.append("  （未检索到相关图谱证据）")
        return "\n".join(lines)

    def explain(self, result: dict) -> str:
        """打印检索过程完整摘要，调试用。"""
        reranked = result.get("reranked", [])
        score_str = ""
        if reranked and "rerank_score" in reranked[0]:
            top3 = [(t.get("head",""), t.get("tail",""), t["rerank_score"])
                    for t in reranked[:3]]
            score_str = "\n  精排Top3: " + " | ".join(
                f"{h}→{t}({s:.3f})" for h, t, s in top3
            )

        return "\n".join([
            f"\n{'='*60}",
            f"  查询: {result['query']}",
            f"  链路: {result.get('pipeline_used','')}",
            f"{'='*60}",
            f"  BM25 召回:    {len(result['bm25_results'])} 条",
            f"  向量召回:     {len(result['vector_results'])} 条",
            f"  RRF融合后:    {len(result['fused'])} 条",
            f"  图扩展:       {len(result['graph_expanded'])} 条",
            f"  精排后:       {len(reranked)} 条" + score_str,
            "",
            result["context"],
            "=" * 60,
        ])


# ═══════════════════════════════════════════════════════
#  消融实验
# ═══════════════════════════════════════════════════════

def run_ablation(triples: list[dict],
                 qa_path: str,
                 top_k: int = 5,
                 model_name: str = DEFAULT_EMBED_MODEL,
                 deepseek_key: Optional[str] = None) -> dict:
    """
    六种配置的 Recall@K 对比：
      BM25-only / Vector-only / BM25+Vector / Full(+Graph) /
      LLM-only(无KG) / NaiveRAG(原文分块BM25)

    qa_path 格式（evaluation/qa_dataset.json）：
    {
      "questions": [
        {
          "id": "sq_001",
          "type": "single_hop",        # single_hop / multi_hop / comparison
          "question": "...",
          "answer": "AN/SPY-1",        # 字符串或字符串列表
          "answer_aliases": [...]      # 可选，额外合法答案
        }
      ]
    }
    """
    qa_path_ = Path(qa_path)
    if not qa_path_.exists():
        raise FileNotFoundError(f"找不到QA文件: {qa_path}")

    with open(qa_path_, encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])

    log.info(f"消融实验: {len(questions)} 条问题，top_k={top_k}")

    # 只初始化一次（共用）
    bm25   = BM25Retriever(triples)
    vector = VectorRetriever(triples, model_name)
    graph  = InMemoryGraphExpander(triples)

    # ── NaiveRAG：从原始语料构建原文分块索引 ─────────────────
    corpus_path = Path("radar_corpus/corpus.json")
    naive_bm25  = None
    naive_chunks: list[str] = []
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        for item in corpus:
            text = item.get("raw_text_en", "") or item.get("raw_text_zh", "")
            title = item.get("en_title", "")
            # 每500字符分一块
            for i in range(0, max(1, len(text)), 500):
                chunk = f"{title}: {text[i:i+500]}"
                naive_chunks.append(chunk)
        if naive_chunks:
            from rank_bm25 import BM25Okapi
            tokenized = [
                [tok.lower() for tok in c.replace(",","").replace(".","").split()
                 if len(tok) > 1]
                for c in naive_chunks
            ]
            naive_bm25 = BM25Okapi(tokenized)
            log.info(f"NaiveRAG BM25: {len(naive_chunks)} 个文本块")

    configs = [
        {"name": "BM25-only",            "use_bm25": True,  "use_vec": False, "use_graph": False, "mode": "kg"},
        {"name": "Vector-only",           "use_bm25": False, "use_vec": True,  "use_graph": False, "mode": "kg"},
        {"name": "BM25+Vector(RRF)",      "use_bm25": True,  "use_vec": True,  "use_graph": False, "mode": "kg"},
        {"name": "Full(BM25+Vec+Graph)",  "use_bm25": True,  "use_vec": True,  "use_graph": True,  "mode": "kg"},
        {"name": "NaiveRAG(原文BM25)",    "use_bm25": False, "use_vec": False, "use_graph": False, "mode": "naive"},
        {"name": "LLM-only(无KG)",        "use_bm25": False, "use_vec": False, "use_graph": False, "mode": "llm_only"},
    ]

    def _llm_answer(question: str, key: str) -> str:
        """调用 DeepSeek 纯零样本回答（无任何上下文）。"""
        try:
            import requests as _req
            resp = _req.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": question}],
                      "max_tokens": 256,
                      "temperature": 0.1},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"LLM-only 调用失败: {e}")
            return ""

    def eval_config(cfg: dict) -> dict:
        per_type: dict[str, list[bool]] = {}
        for q in questions:
            qtype = q.get("type", "unknown")
            answer = q.get("answer", "")
            aliases = q.get("answer_aliases", [])

            all_ans = [answer] if isinstance(answer, str) else list(answer)
            for a in aliases:
                all_ans.extend(a if isinstance(a, list) else [a])

            if cfg["mode"] == "naive":
                # NaiveRAG：对原文分块做 BM25 检索
                if naive_bm25 and naive_chunks:
                    import numpy as _np
                    qtoks = [tok.lower() for tok in q["question"].split() if len(tok) > 1]
                    scores = naive_bm25.get_scores(qtoks)
                    top_idx = _np.argsort(scores)[::-1][:top_k * 3]
                    ctx = " ".join(naive_chunks[i] for i in top_idx if scores[i] > 0).lower()
                else:
                    ctx = ""
                hit = any(str(a).lower() in ctx for a in all_ans if a)

            elif cfg["mode"] == "llm_only":
                # LLM-only：不提供任何检索上下文
                if not deepseek_key:
                    hit = False
                else:
                    ans_text = _llm_answer(q["question"], deepseek_key).lower()
                    hit = any(str(a).lower() in ans_text for a in all_ans if a)

            else:
                # KG 模式
                lists = []
                if cfg["use_bm25"]:
                    lists.append(bm25.retrieve(q["question"], top_k=top_k * 3))
                if cfg["use_vec"]:
                    lists.append(vector.retrieve(q["question"], top_k=top_k * 3))

                fused     = rrf_fusion(lists)[:top_k] if lists else []
                graph_res = []
                if cfg["use_graph"]:
                    seeds = list(
                        {t["head"] for t in fused[:5] if t.get("head")} |
                        {t["tail"] for t in fused[:5] if t.get("tail")}
                    )
                    graph_res = graph.expand(seeds, hops=2)

                ctx = " ".join(
                    f"{t.get('head','')} {t.get('tail','')}"
                    for t in fused + graph_res
                ).lower()
                hit = any(str(a).lower() in ctx for a in all_ans if a)

            per_type.setdefault(qtype, [])
            per_type[qtype].append(hit)

        out: dict[str, dict] = {}
        total_h = total_q = 0
        for qt, hits in per_type.items():
            r = sum(hits) / len(hits) if hits else 0
            out[qt] = {"recall": round(r, 3), "count": len(hits)}
            total_h += sum(hits)
            total_q += len(hits)
        out["overall"] = {
            "recall": round(total_h / total_q, 3) if total_q else 0,
            "count":  total_q,
        }
        return out

    all_results = {}
    print(f"\n{'='*72}")
    print(f"  消融实验  (top_k={top_k}, {len(questions)} 条问题)")
    print(f"{'='*72}")
    print(f"  {'配置':<30} {'单跳':>7} {'多跳':>7} {'对比':>7} {'整体':>7}")
    print(f"  {'-'*65}")

    def fmt(v):
        return f"{v:.1%}" if isinstance(v, float) else "  -  "

    for cfg in configs:
        if cfg["mode"] == "llm_only" and not deepseek_key:
            print(f"  {cfg['name']:<30} {'(需要deepseek_key)':>30}")
            all_results[cfg["name"]] = {"skipped": True}
            continue
        s = eval_config(cfg)
        all_results[cfg["name"]] = s
        print(
            f"  {cfg['name']:<30}"
            f" {fmt(s.get('single_hop',{}).get('recall','-')):>7}"
            f" {fmt(s.get('multi_hop', {}).get('recall','-')):>7}"
            f" {fmt(s.get('comparison',{}).get('recall','-')):>7}"
            f" {fmt(s.get('overall',  {}).get('recall','-')):>7}"
        )

    print(f"{'='*72}\n")

    out_path = INDEX_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"消融实验结果已保存: {out_path}")
    return all_results


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    result_files = [
        "extraction_results/method_c_results.json",
        "extraction_results/method_a_results.json",
    ]
    existing = [f for f in result_files if Path(f).exists()]
    if not existing:
        print("❌ 找不到抽取结果，请先运行 triple_extraction.py")
        sys.exit(1)

    triples = (
        load_triples_merged(*existing, prefer_source="llm_few_shot")
        if len(existing) > 1
        else load_triples_from_results(existing[0])
    )

    retriever = HybridRetriever(triples)

    for q in [
        "AN/SPY-1的工作频段是什么？",
        "装备了相控阵雷达的美国驱逐舰有哪些？",
        "Raytheon研制了哪些雷达系统？",
        "X波段舰载雷达有哪些型号？",
    ]:
        result = retriever.retrieve(q, top_k=8)
        print(retriever.explain(result))
