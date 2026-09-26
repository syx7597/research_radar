"""
Rebuild FAISS + BM25 indexes from the current merged_triples.json
WITHOUT regenerating triples from extraction_results (which would overwrite v2 data).
"""
import os
import sys
import json
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag_retriever import HybridRetriever

triples_path = ROOT / "graphrag_index" / "merged_triples.json"
print(f"Loading triples from {triples_path}…")
with open(triples_path, encoding="utf-8") as f:
    triples = json.load(f)
print(f"  Loaded {len(triples)} triples")

# Delete stale indexes
for old in [ROOT / "graphrag_index" / "faiss.index",
            ROOT / "graphrag_index" / "faiss_meta.pkl"]:
    if old.exists():
        old.unlink()
        print(f"  Removed stale {old.name}")

print("\nBuilding HybridRetriever (FAISS + BM25)…")
retriever = HybridRetriever(triples, enable_reranker=False)
print("\n[DONE] Index rebuilt on v2 KG.")
print("  faiss.index, faiss_meta.pkl regenerated in graphrag_index/")

# Quick smoke test
print("\n=== Smoke test ===")
for q in ["AN/APG-77 的工作频段", "Raytheon 研制的雷达"]:
    result = retriever.retrieve(q, top_k=3, use_graph_expansion=False)
    print(f"\nQ: {q}")
    for t in result["fused"][:3]:
        print(f"  {t.get('head')} --[{t.get('relation')}]--> {t.get('tail')}")
