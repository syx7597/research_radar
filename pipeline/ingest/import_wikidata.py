"""
Wikidata SPARQL 雷达知识图谱导入脚本

从 Wikidata 公共 SPARQL 端点查询雷达系统实体及其结构化属性，
直接生成与现有 KG 兼容的三元组，无需 LLM。

查询覆盖：
  - 雷达系统实例（Q5962774 及子类）
  - 制造商 (P176) → developedBy
  - 运营商 (P137) → operatedBy
  - 来源国 (P17)  → operatedBy / affiliatedTo
  - 所属组织 (P749/P355) → affiliatedTo
  - 改进自 (P144/P6501) → upgradeOf / derivedFrom
  - 频段属性（如存在）→ hasFrequencyBand

用法：
    python import_wikidata.py
    python import_wikidata.py --limit 5000
    python import_wikidata.py --lang zh  # 尝试获取中文标签

结果保存到：extraction_results/wikidata_results.json
"""

import json
import time
import re
import logging
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PATH = Path("extraction_results/wikidata_results.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "RadarKG-Research-Bot/1.0 (academic research; contact: research@example.com)",
    "Accept": "application/sparql-results+json",
}

# 关系映射：Wikidata property → KG relation
PROPERTY_TO_RELATION = {
    "P176":  "developedBy",    # manufacturer
    "P137":  "operatedBy",     # operator
    "P17":   "operatedBy",     # country (of operator / origin)
    "P149":  "affiliatedTo",   # architectural style (skip)
    "P749":  "affiliatedTo",   # parent organization
    "P355":  "affiliatedTo",   # subsidiary (reverse)
    "P144":  "upgradeOf",      # based on
    "P6501": "derivedFrom",    # derivative work
    "P571":  None,             # inception date (skip)
    "P856":  None,             # official website (skip)
}

# 频段字符串标准化
FREQ_BAND_MAP = {
    "x band": "X", "x-band": "X",
    "s band": "S", "s-band": "S",
    "l band": "L", "l-band": "L",
    "c band": "C", "c-band": "C",
    "ku band": "Ku", "ku-band": "Ku",
    "ka band": "Ka", "ka-band": "Ka",
    "uhf": "UHF", "vhf": "VHF",
    "p band": "P", "p-band": "P",
    "w band": "W", "w-band": "W",
    "millimeter": "Ka",
}


# ══════════════════════════════════════════════════════════
#  SPARQL 查询
# ══════════════════════════════════════════════════════════

def sparql_query(query: str, retries: int = 3) -> Optional[dict]:
    """向 Wikidata SPARQL 端点发起查询。"""
    for attempt in range(retries):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=60
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                log.warning(f"  速率限制，等待 {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.warning(f"  超时 (attempt {attempt+1}/{retries})")
            time.sleep(5)
        except Exception as e:
            log.warning(f"  查询失败: {e} (attempt {attempt+1}/{retries})")
            time.sleep(3)
    return None


# Wikidata 雷达相关类型 QID：
#   Q126177905 = "radar model"      (最精准，345个实体)
#   Q47528     = "radar"            (技术概念，少量装备实体)
#   Q22812603  = "fire-control radar"
#   Q1517980   = "airborne radar"
# 用 UNION 合并多个类型以最大化覆盖
RADAR_TYPES_QUERY = """
SELECT DISTINCT ?radar ?radarLabel ?radarLabelZh
WHERE {
  {
    ?radar wdt:P31 wd:Q126177905 .  # radar model
  } UNION {
    ?radar wdt:P31 wd:Q22812603 .   # fire-control radar
  } UNION {
    ?radar wdt:P31 wd:Q1517980 .    # airborne radar
  } UNION {
    ?radar wdt:P31/wdt:P279 wd:Q126177905 .  # subclass of radar model
  }
  FILTER NOT EXISTS { ?radar wdt:P31 wd:Q4167410 }   # 排除消歧义页
  FILTER NOT EXISTS { ?radar wdt:P31 wd:Q13406463 }  # 排除列表文章
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
    ?radar rdfs:label ?radarLabel .
  }
  OPTIONAL {
    ?radar rdfs:label ?radarLabelZh .
    FILTER(LANG(?radarLabelZh) = "zh")
  }
}
LIMIT %(limit)d
OFFSET %(offset)d
"""

RADAR_PROPERTIES_QUERY = """
SELECT DISTINCT
  ?radar ?radarLabel
  ?manufacturer ?manufacturerLabel
  ?operator ?operatorLabel
  ?originCountry ?originCountryLabel
  ?basedOn ?basedOnLabel
  ?subclassOf ?subclassOfLabel
WHERE {
  VALUES ?radar { %(ids)s }
  OPTIONAL { ?radar wdt:P176 ?manufacturer }      # manufacturer
  OPTIONAL { ?radar wdt:P137 ?operator }           # operator
  OPTIONAL { ?radar wdt:P495 ?originCountry }      # country of origin
  OPTIONAL { ?radar wdt:P144 ?basedOn }            # based on (predecessor)
  OPTIONAL { ?radar wdt:P279 ?subclassOf }         # subclass of (parent type)
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
    ?radar rdfs:label ?radarLabel .
    ?manufacturer rdfs:label ?manufacturerLabel .
    ?operator rdfs:label ?operatorLabel .
    ?originCountry rdfs:label ?originCountryLabel .
    ?basedOn rdfs:label ?basedOnLabel .
    ?subclassOf rdfs:label ?subclassOfLabel .
  }
}
"""

MANUFACTURER_COUNTRY_QUERY = """
SELECT DISTINCT ?mfr ?mfrLabel ?country ?countryLabel
WHERE {
  VALUES ?mfr { %(ids)s }
  { ?mfr wdt:P17 ?country }
  UNION
  { ?mfr wdt:P749/wdt:P17 ?country }
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
    ?mfr rdfs:label ?mfrLabel .
    ?country rdfs:label ?countryLabel .
  }
}
"""


def _label(binding: dict, key: str) -> str:
    """安全地从 SPARQL binding 提取字符串值。"""
    val = binding.get(key, {})
    return val.get("value", "").strip() if isinstance(val, dict) else ""


def _qid(binding: dict, key: str) -> str:
    """从 Wikidata URI 提取 QID，如 wd:Q12345。"""
    uri = _label(binding, key)
    m = re.search(r'Q\d+$', uri)
    return f"wd:{m.group()}" if m else ""


# ══════════════════════════════════════════════════════════
#  Step 1：获取雷达实体列表
# ══════════════════════════════════════════════════════════

def fetch_radar_entities(limit: int = 3000) -> list[dict]:
    """分页获取所有雷达实体。"""
    entities = []
    page_size = 500
    offset = 0

    log.info(f"查询雷达实体（最多 {limit} 个）...")
    while offset < limit:
        query = RADAR_TYPES_QUERY % {
            "limit": min(page_size, limit - offset),
            "offset": offset,
        }
        result = sparql_query(query)
        if not result:
            break

        bindings = result.get("results", {}).get("bindings", [])
        if not bindings:
            break

        for b in bindings:
            qid = _qid(b, "radar")
            name_en = _label(b, "radarLabel")
            name_zh = _label(b, "radarLabelZh")
            if qid and name_en:
                entities.append({
                    "qid": qid,
                    "name_en": name_en,
                    "name_zh": name_zh or name_en,
                })

        log.info(f"  offset={offset}: +{len(bindings)} 实体 (累计 {len(entities)})")
        offset += len(bindings)
        if len(bindings) < page_size:
            break
        time.sleep(1.0)  # Wikidata 速率限制

    log.info(f"共获取 {len(entities)} 个雷达实体")
    return entities


# ══════════════════════════════════════════════════════════
#  Step 2：获取每个雷达的属性（批量查询）
# ══════════════════════════════════════════════════════════

def fetch_radar_properties(entities: list[dict],
                           batch_size: int = 50) -> list[dict]:
    """批量查询雷达的制造商、运营商、来源国、基于型号等属性。"""
    all_props = []
    total = len(entities)

    for i in range(0, total, batch_size):
        batch = entities[i: i + batch_size]
        ids_str = " ".join(e["qid"] for e in batch)
        query = RADAR_PROPERTIES_QUERY % {"ids": ids_str}

        result = sparql_query(query)
        if not result:
            log.warning(f"  批次 {i//batch_size+1} 查询失败，跳过")
            continue

        bindings = result.get("results", {}).get("bindings", [])
        all_props.extend(bindings)
        log.info(f"  批次 {i//batch_size+1}/{(total-1)//batch_size+1}: "
                 f"+{len(bindings)} 属性记录")
        time.sleep(0.8)

    return all_props


# ══════════════════════════════════════════════════════════
#  Step 3：获取制造商所属国家（affiliatedTo）
# ══════════════════════════════════════════════════════════

def fetch_manufacturer_countries(manufacturer_qids: set[str],
                                  batch_size: int = 50) -> dict[str, str]:
    """查询制造商所属国家，用于 affiliatedTo 三元组。"""
    if not manufacturer_qids:
        return {}

    mfr_country = {}
    qids = list(manufacturer_qids)

    for i in range(0, len(qids), batch_size):
        batch = qids[i: i + batch_size]
        ids_str = " ".join(batch)
        query = MANUFACTURER_COUNTRY_QUERY % {"ids": ids_str}

        result = sparql_query(query)
        if not result:
            continue

        for b in result.get("results", {}).get("bindings", []):
            mfr_label = _label(b, "mfrLabel")
            country_label = _label(b, "countryLabel")
            if mfr_label and country_label:
                mfr_country[mfr_label] = country_label

        time.sleep(0.5)

    log.info(f"  获取 {len(mfr_country)} 个制造商-国家对应关系")
    return mfr_country


# ══════════════════════════════════════════════════════════
#  Step 4：转换为 KG 三元组格式
# ══════════════════════════════════════════════════════════

def _entity_type_for_relation(relation: str, tail_label: str) -> str:
    type_map = {
        "developedBy":    "Manufacturer",
        "operatedBy":     "Country",
        "affiliatedTo":   "Country",
        "upgradeOf":      "Radar",
        "derivedFrom":    "Radar",
    }
    # 制造商 affiliatedTo 的是 Country
    if relation == "affiliatedTo" and any(
        x in tail_label for x in ["United", "China", "Russia", "France", "UK", "Germany"]
    ):
        return "Country"
    return type_map.get(relation, "Entity")


def convert_to_triples(
    entities: list[dict],
    prop_bindings: list[dict],
    mfr_country: dict[str, str],
) -> list[dict]:
    """
    将 SPARQL 查询结果转换为 KG 三元组列表。
    按雷达型号分组，便于与现有数据合并。
    """
    # 建立 QID → 实体信息 索引
    qid_to_entity = {e["qid"]: e for e in entities}

    # 按雷达 QID 分组属性
    radar_props: dict[str, list[dict]] = {}
    for b in prop_bindings:
        qid = _qid(b, "radar")
        if qid:
            radar_props.setdefault(qid, []).append(b)

    results = []
    for qid, entity in qid_to_entity.items():
        radar_name = entity["name_en"]
        triples = []

        for b in radar_props.get(qid, []):
            for prop_key, (label_key, relation) in [
                ("manufacturer",   ("manufacturerLabel",   "developedBy")),
                ("operator",       ("operatorLabel",       "operatedBy")),
                ("originCountry",  ("originCountryLabel",  "operatedBy")),  # P495 → operatedBy
                ("basedOn",        ("basedOnLabel",        "upgradeOf")),
            ]:
                tail = _label(b, label_key)
                if not tail or tail == radar_name:
                    continue

                tail_type = _entity_type_for_relation(relation, tail)
                triple = {
                    "head":       radar_name,
                    "head_type":  "Radar",
                    "relation":   relation,
                    "tail":       tail,
                    "tail_type":  tail_type,
                    "confidence": 0.90,
                    "evidence":   f"Wikidata {qid}",
                    "source":     "wikidata",
                    "source_file": f"wikidata:{qid}",
                }
                triples.append(triple)

                # 如果是制造商，再加 affiliatedTo
                if relation == "developedBy" and tail in mfr_country:
                    country = mfr_country[tail]
                    triples.append({
                        "head":       tail,
                        "head_type":  "Manufacturer",
                        "relation":   "affiliatedTo",
                        "tail":       country,
                        "tail_type":  "Country",
                        "confidence": 0.90,
                        "evidence":   f"Wikidata manufacturer country",
                        "source":     "wikidata",
                        "source_file": f"wikidata:{qid}",
                    })

        # 去重
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        if unique_triples:
            results.append({
                "radar_id":     qid.replace("wd:", "").lower(),
                "en_title":     radar_name,
                "zh_title":     entity.get("name_zh", ""),
                "wikidata_qid": qid,
                "triple_count": len(unique_triples),
                "triples":      unique_triples,
            })

    return results


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def run_wikidata_import(limit: int = 3000) -> list[dict]:
    """
    完整 Wikidata 导入流程。

    Args:
        limit: 最多获取的雷达实体数量

    Returns:
        按雷达分组的三元组列表
    """
    log.info("=== Wikidata 雷达知识图谱导入 ===")

    # Step 1
    entities = fetch_radar_entities(limit=limit)
    if not entities:
        log.error("未获取到任何雷达实体，请检查网络连接")
        return []

    # Step 2
    log.info(f"\n查询 {len(entities)} 个实体的属性...")
    prop_bindings = fetch_radar_properties(entities)

    # Step 3：收集所有制造商 QID，批量查询其国家
    mfr_qids = set()
    for b in prop_bindings:
        qid = _qid(b, "manufacturer")
        if qid:
            mfr_qids.add(qid)
    log.info(f"\n查询 {len(mfr_qids)} 个制造商的所属国家...")
    mfr_country = fetch_manufacturer_countries(mfr_qids)

    # Step 4
    log.info("\n转换为三元组...")
    results = convert_to_triples(entities, prop_bindings, mfr_country)

    # 统计
    total_triples = sum(r["triple_count"] for r in results)
    log.info(f"\n导入完成:")
    log.info(f"  雷达实体: {len(entities)}")
    log.info(f"  有三元组的实体: {len(results)}")
    log.info(f"  三元组总数: {total_triples}")

    from collections import Counter
    all_triples = [t for r in results for t in r["triples"]]
    rel_dist = Counter(t["relation"] for t in all_triples)
    log.info("  关系分布:")
    for rel, cnt in rel_dist.most_common():
        log.info(f"    {rel}: {cnt}")

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"\n保存到: {OUTPUT_PATH}")

    return results


# ══════════════════════════════════════════════════════════
#  快速测试（单个雷达）
# ══════════════════════════════════════════════════════════

def test_single(radar_name: str = "AN/TPY-2"):
    """
    测试单个已知雷达的 Wikidata 数据（调试用）。
    """
    # 先搜索实体
    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item ?label "{radar_name}"@en .
      ?item wdt:P31/wdt:P279* wd:Q5962774 .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }} LIMIT 5
    """
    result = sparql_query(query)
    if not result:
        print(f"未找到 {radar_name}")
        return

    bindings = result.get("results", {}).get("bindings", [])
    print(f"找到 {len(bindings)} 个结果:")
    for b in bindings:
        print(f"  {_label(b, 'itemLabel')} — {_label(b, 'item')}")


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikidata 雷达知识图谱导入")
    parser.add_argument("--limit", type=int, default=3000,
                        help="最多获取的雷达实体数量（默认 3000）")
    parser.add_argument("--test", type=str, default=None,
                        help="测试单个雷达名称（如 --test 'AN/TPY-2'）")
    args = parser.parse_args()

    if args.test:
        test_single(args.test)
    else:
        run_wikidata_import(limit=args.limit)

    print(f"\n下一步：python merge_sources.py")
