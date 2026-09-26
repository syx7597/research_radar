"""
合并三元组 + 实体规范化 + 建立检索索引
运行: python build_index.py

这是整个系统的"数据准备"入口，必须在 qa_pipeline.py 之前运行。
规范化后的三元组同时用于:
  1. 构建 FAISS + BM25 检索索引（qa_pipeline.py 使用）
  2. 导出到 Neo4j（export_to_neo4j.py 使用同一份文件）
"""

import os, json
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

from pathlib import Path
import sys
# Moved to pipeline/index/ in 2026-06-22 restructure: put repo root on sys.path
# so the root-level serving module graphrag_retriever resolves. entity_normalizer
# lives alongside this file (pipeline/index/) and resolves as a sibling.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from graphrag_retriever import load_triples_merged, HybridRetriever
from entity_normalizer import normalize_triples


def quality_filter(triples: list[dict]) -> list[dict]:
    """
    应用关系级置信度过滤，剔除低质量三元组。
    - derivedFrom: 要求 confidence >= 0.75（历史精度仅8%，严格过滤）
    - upgradeOf:   要求 confidence >= 0.65
    - competitorOf: 全部过滤（由 kg_enrichment.py 推断）
    """
    filtered = []
    thresholds = {
        "derivedFrom":    0.75,
        "upgradeOf":      0.65,
        "competitorOf":   1.01,  # 永远过滤（规则推断，不来自手册）
        "compatibleWith": 0.70,  # 武器兼容：要求较高置信度
        "hasMode":        0.60,
        "countryOfOrigin":0.70,
    }
    for t in triples:
        rel   = t.get("relation", "")
        conf  = t.get("confidence", 0.5)
        floor = thresholds.get(rel, 0.0)
        if conf >= floor:
            filtered.append(t)
    return filtered

# ══════════════════════════════════════════════════════════
#  Step 1：合并多路抽取结果
# ══════════════════════════════════════════════════════════

print("=" * 55)
print("Step 1: 合并三元组")
print("=" * 55)

sources = []
for path in [
    "extraction_results/method_a_results.json",          # 规则抽取（补全）
    "extraction_results/method_b_results.json",          # 零样本LLM
    "extraction_results/method_c_results.json",          # 少样本LLM（优先）
    "extraction_results/globalsecurity_results.json",    # GlobalSecurity爬取
    "extraction_results/wikidata_results.json",          # Wikidata SPARQL
    "extraction_results/manual_results.json",            # Word手册（如果有）
    "extraction_results/pdf_results.json",               # PDF手册（如果有）
]:
    if Path(path).exists():
        sources.append(path)
        print(f"  找到: {path}")
    else:
        print(f"  跳过: {path}（不存在）")

triples_raw = load_triples_merged(*sources, prefer_source="llm_few_shot")
print(f"合并后（规范化前）: {len(triples_raw)} 条\n")

# ══════════════════════════════════════════════════════════
#  Step 2：实体规范化
# ══════════════════════════════════════════════════════════

print("=" * 55)
print("Step 2: 实体规范化")
print("=" * 55)

triples_normalized = normalize_triples(triples_raw)
triples = quality_filter(triples_normalized)
removed = len(triples_normalized) - len(triples)
print(f"质量过滤：移除 {removed} 条低质量三元组（主要为低置信度derivedFrom/upgradeOf/competitorOf）")
print()

# 统计规范化后的关系分布
from collections import Counter
rel_dist = Counter(t["relation"] for t in triples)
print("规范化后关系分布:")
for rel, cnt in rel_dist.most_common():
    print(f"  {rel}: {cnt}")

# ══════════════════════════════════════════════════════════
#  Step 3：保存规范化后的三元组（供 Neo4j 导出使用）
# ══════════════════════════════════════════════════════════

print()
print("=" * 55)
print("Step 3: 保存规范化三元组")
print("=" * 55)

out_dir = Path("graphrag_index")
out_dir.mkdir(exist_ok=True)
normalized_path = out_dir / "merged_triples.json"

with open(normalized_path, "w", encoding="utf-8") as f:
    json.dump(triples, f, ensure_ascii=False, indent=2)
print(f"已保存到: {normalized_path}")
print(f"共 {len(triples)} 条三元组（规范化 + 去重后）")
print("  [OK] export_to_neo4j.py will read this file")

# ══════════════════════════════════════════════════════════
#  Step 4：构建检索索引
# ══════════════════════════════════════════════════════════

print()
print("=" * 55)
print("Step 4: 构建 FAISS + BM25 检索索引")
print("=" * 55)
print("首次运行需要加载 embedding 模型，请稍候...\n")

# 删除旧索引，强制重建（因为三元组已更新）
for old in [out_dir / "faiss.index", out_dir / "faiss_meta.pkl"]:
    if old.exists():
        old.unlink()
        print(f"  删除旧索引: {old.name}")

retriever = HybridRetriever(triples, enable_reranker=False)
print("\n[DONE] Index built, files saved in graphrag_index/")
print("   faiss.index     — 向量索引")
print("   faiss_meta.pkl  — 索引元数据")
print("   merged_triples.json — 规范化后的三元组（Neo4j 也用这个）")

# ══════════════════════════════════════════════════════════
#  Step 5：快速冒烟测试
# ══════════════════════════════════════════════════════════

print()
print("=" * 55)
print("Step 5: 快速冒烟测试")
print("=" * 55)

test_cases = [
    ("AN/TPY-2的工作频段",    "hasFrequencyBand", "X"),
    ("Raytheon研制的雷达",    "developedBy",      "Raytheon"),
    ("RTX研制的雷达",         "developedBy",      "Raytheon"),   # 测试规范化是否生效
    ("部署在驱逐舰上的雷达",  "deployedOn",       None),
]

for query, expected_rel, expected_tail in test_cases:
    result = retriever.retrieve(query, top_k=3, use_graph_expansion=False)
    top3 = result["fused"][:3]
    print(f"\n查询: {query}")
    for t in top3:
        mark = ""
        if expected_tail and t.get("tail") == expected_tail:
            mark = " [OK]"
        print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}{mark}")

print()
print("=" * 55)
print("全部完成！下一步：")
print("  运行问答系统: python qa_pipeline.py")
print("  导出图数据库: python export_to_neo4j.py")
print("=" * 55)
