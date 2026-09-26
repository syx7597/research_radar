"""
多来源三元组合并脚本

将以下来源的抽取结果合并成统一的 KG 文件：
  1. extraction_results/method_c_results.json     (Wikipedia LLM抽取，现有)
  2. extraction_results/method_a_results.json     (Wikipedia规则抽取，现有)
  3. extraction_results/manual_results.json       (Word手册，extract_from_manual.py)
  4. extraction_results/pdf_results.json          (PDF手册，extract_from_pdf.py)
  5. extraction_results/wikidata_results.json     (Wikidata SPARQL，import_wikidata.py)
  6. extraction_results/globalsecurity_results.json (GlobalSecurity.org，scrape_globalsecurity.py)

合并策略：
  - 相同 (head, relation, tail) 的三元组去重，保留置信度最高的
  - 来源优先级：手册 > Wikidata > GlobalSecurity > Wikipedia LLM > Wikipedia规则
  - 实体名称标准化（大小写、连字符等）
  - 可选：运行 kg_enrichment.py 补充推断三元组

输出：
  graphrag_index/merged_triples.json  (覆盖原文件)
  graphrag_index/merged_triples_enriched.json (运行enrichment后)

用法：
    python merge_sources.py                    # 合并，不重建索引
    python merge_sources.py --rebuild-index   # 合并 + 重建FAISS索引
    python merge_sources.py --enrich          # 合并 + enrichment推断
    python merge_sources.py --stats           # 只显示各来源统计
"""

import json
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════
#  来源定义（优先级从低到高排列，高优先级覆盖低优先级）
# ══════════════════════════════════════════════════════════

SOURCES = [
    # (文件路径, 来源标签, 优先级)
    ("extraction_results/method_a_results.json",       "method_a",      1),
    ("extraction_results/method_b_results.json",       "method_b",      2),
    ("extraction_results/method_c_results.json",       "wikipedia_llm", 3),
    ("extraction_results/globalsecurity_results.json", "globalsecurity", 4),
    ("extraction_results/wikidata_results.json",       "wikidata",       5),
    ("extraction_results/manual_results.json",         "manual_docx",    6),
    ("extraction_results/pdf_results.json",            "manual_pdf",     7),
]

OUTPUT_MERGED = Path("graphrag_index/merged_triples.json")

# 有效关系类型白名单
VALID_RELATIONS = {
    # 原有关系
    "developedBy", "operatedBy", "deployedOn", "hasFrequencyBand",
    "hasFunction", "affiliatedTo", "exportedTo", "upgradeOf",
    "derivedFrom", "competitorOf",
    # 新增关系
    "compatibleWith",   # Radar → Weapon（可制导的导弹/武器）
    "hasMode",          # Radar → RadarMode（工作模式：SAR/GMTI/TWS/STT/FireControl等）
    "countryOfOrigin",  # Radar → Country（研制来源国，区别于operatedBy）
}

# 频段标准化
FREQ_NORMALIZE = {
    "x-band": "X", "x band": "X",
    "s-band": "S", "s band": "S",
    "l-band": "L", "l band": "L",
    "c-band": "C", "c band": "C",
    "ku-band": "Ku", "ku band": "Ku",
    "ka-band": "Ka", "ka band": "Ka",
    "uhf": "UHF", "vhf": "VHF",
    "p-band": "P", "p band": "P",
    "w-band": "W", "w band": "W",
    "millimeter wave": "Ka", "mmwave": "Ka",
}


# ══════════════════════════════════════════════════════════
#  实体名称标准化
# ══════════════════════════════════════════════════════════

_COUNTRY_ALIASES = {
    "usa": "United States", "u.s.": "United States", "u.s.a.": "United States",
    "us": "United States", "america": "United States",
    "uk": "United Kingdom", "britain": "United Kingdom", "great britain": "United Kingdom",
    "prc": "China", "people's republic of china": "China",
    "ussr": "Russia", "soviet union": "Russia",
    "dprk": "North Korea", "north korea": "North Korea",
    "rok": "South Korea", "south korea": "South Korea",
}


def normalize_entity(name: str, relation: str = "", tail_type: str = "") -> str:
    """标准化实体名称。"""
    if not name:
        return name

    s = name.strip()

    # 频段标准化
    if relation == "hasFrequencyBand":
        key = s.lower().strip()
        return FREQ_NORMALIZE.get(key, s.upper() if len(s) <= 3 else s)

    # 国家名称标准化
    if tail_type in ("Country",) or relation in ("operatedBy", "affiliatedTo", "exportedTo", "countryOfOrigin"):
        key = s.lower()
        if key in _COUNTRY_ALIASES:
            return _COUNTRY_ALIASES[key]

    # 武器型号：保留原始大小写，去除多余空格
    if relation == "compatibleWith" or tail_type == "Weapon":
        return re.sub(r'\s+', ' ', s)

    # 雷达模式：统一大写缩写
    if relation == "hasMode" or tail_type == "RadarMode":
        mode_map = {
            "sar": "SAR", "gmti": "GMTI", "tws": "TWS", "stt": "STT",
            "lprf": "LPRF", "mprf": "MPRF", "hiprf": "HIPRF",
            "fire control": "FireControl", "fire-control": "FireControl",
            "terrain avoidance": "TerrainAvoidance",
            "ground mapping": "GroundMapping",
            "weather": "Weather", "navigation": "Navigation",
            "search": "Search", "track": "Track",
        }
        return mode_map.get(s.lower(), s)

    # AN/XXX 命名统一大写
    m = re.match(r'an/([a-z]{3}-\d+[a-z]?)', s, re.IGNORECASE)
    if m:
        return "AN/" + m.group(1).upper()

    return s


def normalize_triple(t: dict) -> Optional[dict]:
    """
    标准化单条三元组：
    - 过滤无效关系
    - 标准化实体名称
    - 过滤过短/过长的实体
    - 返回 None 表示过滤掉
    """
    head = normalize_entity(t.get("head", "").strip())
    tail = normalize_entity(
        t.get("tail", "").strip(),
        relation=t.get("relation", ""),
        tail_type=t.get("tail_type", "")
    )
    relation = t.get("relation", "").strip()

    # 基本有效性检查
    if not head or not tail or not relation:
        return None
    if len(head) < 2 or len(tail) < 2:
        return None
    if len(head) > 200 or len(tail) > 200:
        return None
    if relation not in VALID_RELATIONS:
        return None
    # 自环检测（同一实体）
    if head.lower() == tail.lower():
        return None

    return {**t, "head": head, "tail": tail, "relation": relation}


# ══════════════════════════════════════════════════════════
#  加载各来源
# ══════════════════════════════════════════════════════════

def load_source(path: str, source_tag: str) -> list[dict]:
    """
    加载单个来源文件，处理不同格式。
    支持：
      - 列表格式（每个元素含 triples 字段）
      - 扁平三元组列表
    """
    p = Path(path)
    if not p.exists():
        log.debug(f"跳过（不存在）: {path}")
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"加载失败 {path}: {e}")
        return []

    triples = []

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            # 检查是否是按雷达分组的格式
            if "triples" in data[0]:
                for item in data:
                    for t in item.get("triples", []):
                        t["_source"] = source_tag
                        triples.append(t)
            # 检查是否是扁平三元组格式
            elif "head" in data[0] and "relation" in data[0]:
                for t in data:
                    t["_source"] = source_tag
                    triples.append(t)
    elif isinstance(data, dict):
        # 可能是 {radar_id: {triples: [...]}} 格式
        for v in data.values():
            if isinstance(v, dict) and "triples" in v:
                for t in v["triples"]:
                    t["_source"] = source_tag
                    triples.append(t)
            elif isinstance(v, list):
                for t in v:
                    t["_source"] = source_tag
                    triples.append(t)

    log.info(f"  {source_tag}: {len(triples)} 条三元组 ← {path}")
    return triples


# ══════════════════════════════════════════════════════════
#  合并与去重
# ══════════════════════════════════════════════════════════

def merge_all_sources() -> list[dict]:
    """
    合并所有来源，去重，标准化，返回最终三元组列表。
    """
    log.info("=== 加载各来源 ===")
    all_raw = []
    for path, tag, priority in SOURCES:
        raw = load_source(path, tag)
        # 记录优先级
        for t in raw:
            t["_priority"] = priority
        all_raw.extend(raw)

    log.info(f"\n原始三元组总数: {len(all_raw)}")

    # 标准化
    normalized = []
    skipped = 0
    for t in all_raw:
        n = normalize_triple(t)
        if n:
            normalized.append(n)
        else:
            skipped += 1

    log.info(f"标准化后: {len(normalized)} 条（过滤 {skipped} 条无效）")

    # 去重：相同 (head, relation, tail) 保留优先级最高的
    best: dict[tuple, dict] = {}
    for t in normalized:
        key = (t["head"].lower(), t["relation"], t["tail"].lower())
        existing = best.get(key)
        if existing is None:
            best[key] = t
        else:
            # 高优先级来源覆盖低优先级，同优先级取高置信度
            if (t.get("_priority", 0) > existing.get("_priority", 0) or
                (t.get("_priority", 0) == existing.get("_priority", 0) and
                 t.get("confidence", 0) > existing.get("confidence", 0))):
                best[key] = t

    merged = list(best.values())
    log.info(f"去重后: {len(merged)} 条三元组")

    # 清理内部字段
    for t in merged:
        t.pop("_source", None)
        t.pop("_priority", None)

    return merged


# ══════════════════════════════════════════════════════════
#  统计报告
# ══════════════════════════════════════════════════════════

def print_stats(triples: list[dict]):
    """打印三元组统计信息。"""
    print(f"\n{'='*60}")
    print(f"  知识图谱统计")
    print(f"{'='*60}")

    # 基本统计
    entities = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    heads = {t["head"] for t in triples}

    print(f"  三元组总数: {len(triples)}")
    print(f"  实体总数:   {len(entities)}")
    print(f"  雷达实体:   {len(heads)} (以头实体计)")

    # 关系分布
    print(f"\n  关系分布:")
    rel_dist = Counter(t["relation"] for t in triples)
    for rel, cnt in rel_dist.most_common():
        bar = "█" * (cnt // max(1, max(rel_dist.values()) // 30))
        print(f"    {rel:<20s} {cnt:>5d}  {bar}")

    # 来源分布
    print(f"\n  来源分布:")
    src_dist = Counter(t.get("source", "unknown") for t in triples)
    for src, cnt in src_dist.most_common():
        print(f"    {src:<30s} {cnt:>5d}")

    # 置信度分布
    conf_bins = {"≥0.95": 0, "0.85-0.94": 0, "0.70-0.84": 0, "<0.70": 0}
    for t in triples:
        c = t.get("confidence", 0.8)
        if c >= 0.95:
            conf_bins["≥0.95"] += 1
        elif c >= 0.85:
            conf_bins["0.85-0.94"] += 1
        elif c >= 0.70:
            conf_bins["0.70-0.84"] += 1
        else:
            conf_bins["<0.70"] += 1

    print(f"\n  置信度分布:")
    for k, v in conf_bins.items():
        print(f"    {k:<12s} {v:>5d}")

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════

def main(rebuild_index: bool = False, enrich: bool = False, stats_only: bool = False):

    if stats_only:
        # 只统计各来源
        log.info("各来源统计：")
        for path, tag, priority in SOURCES:
            raw = load_source(path, tag)
            if raw:
                rel_dist = Counter(t.get("relation", "?") for t in raw)
                log.info(f"  {tag} ({priority}级): {len(raw)} 条")
                for r, c in rel_dist.most_common(5):
                    log.info(f"    {r}: {c}")
        return

    # 合并
    merged = merge_all_sources()

    # 保存
    OUTPUT_MERGED.parent.mkdir(exist_ok=True)
    with open(OUTPUT_MERGED, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    log.info(f"\n保存到: {OUTPUT_MERGED}")

    # 打印统计
    print_stats(merged)

    # 可选：知识图谱增强（推断三元组）
    if enrich:
        log.info("运行 kg_enrichment.py 推断三元组...")
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "kg_enrichment.py"], check=True)
        except Exception as e:
            log.error(f"enrichment 失败: {e}")

    # 可选：重建检索索引
    if rebuild_index:
        log.info("重建 FAISS + BM25 检索索引...")
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "build_index.py"], check=True)
        except Exception as e:
            log.error(f"索引重建失败: {e}")

    log.info("合并完成！")
    log.info("\n推荐后续步骤:")
    log.info("  1. python kg_enrichment.py         # 推断 competitorOf、coDeployedWith 等")
    log.info("  2. python build_index.py            # 重建检索索引")
    log.info("  3. python datasets/loader.py        # 验证新KG可正常加载")
    log.info("  4. python run_wn18rr.py             # （可选）在新KG上重跑OntoCom")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多来源三元组合并")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="合并后重建 FAISS 检索索引")
    parser.add_argument("--enrich", action="store_true",
                        help="合并后运行 kg_enrichment.py 推断三元组")
    parser.add_argument("--stats", action="store_true",
                        help="只显示各来源统计，不合并")
    args = parser.parse_args()

    main(
        rebuild_index=args.rebuild_index,
        enrich=args.enrich,
        stats_only=args.stats,
    )
