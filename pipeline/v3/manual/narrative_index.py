"""
手册叙述块（技术特点/分系统/研制装备情况）向量索引 —— GraphRAG 文本侧。

结构化知识进 KG（边/属性），叙述知识进这里的文本向量库；检索时 KG 命中雷达
→ 图扩展 + 本库按 radar/section 召回对应叙述段，混合检索。用与三元组索引
同一个 embedding 模型（bge-small-zh），可被 graphrag_retriever 融合。

  python narrative_index.py --build          # 建索引
  python narrative_index.py --query "F-22 的雷达用什么技术"

产物: kg_v3/narrative.faiss + kg_v3/narrative_meta.pkl
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "pipeline" / "v3" / "work"
OUT = ROOT / "kg_v3"
CHUNK_FILES = [WORK / "manual_body_chunks.jsonl", WORK / "naval_body_chunks.jsonl"]
IDX = OUT / "narrative.faiss"
META = OUT / "narrative_meta.pkl"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_chunks():
    out = []
    for f in CHUNK_FILES:
        if f.exists():
            out.extend(json.loads(l) for l in f.open(encoding="utf-8"))
    return out


def build():
    import faiss
    chunks = load_chunks()
    # embedding 文本：雷达名 + 章节 + 正文（雷达名入向量，利于按型号召回）
    texts = [f"{c['radar']} {c['section']}：{c['text']}" for c in chunks]
    emb = model().encode(texts, batch_size=64, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    OUT.mkdir(exist_ok=True)
    faiss.write_index(index, str(IDX))
    META.write_bytes(pickle.dumps(chunks))
    print(f"[narrative] 索引 {len(chunks)} 叙述块, dim={emb.shape[1]} -> {IDX.name}")


class NarrativeRetriever:
    """可被 graphrag_retriever.HybridRetriever 作为文本侧一路融合。"""

    def __init__(self):
        import faiss
        self.index = faiss.read_index(str(IDX))
        self.meta = pickle.loads(META.read_bytes())

    def retrieve(self, query, top_k=5, radar=None):
        q = model().encode([query], normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
        scores, idxs = self.index.search(q, top_k * 4 if radar else top_k)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            c = dict(self.meta[i])
            if radar and c["radar"] != radar:      # 图命中雷达后可按型号过滤
                continue
            c["score"] = float(s)
            out.append(c)
            if len(out) >= top_k:
                break
        return out


def main():
    if "--build" in sys.argv:
        build()
    elif "--query" in sys.argv:
        q = sys.argv[sys.argv.index("--query") + 1]
        for c in NarrativeRetriever().retrieve(q, top_k=4):
            print(f"[{c['score']:.3f}] {c['radar']} / {c['section']}: {c['text'][:90]}…")
    else:
        print("用法: --build | --query <问题>")


if __name__ == "__main__":
    main()
