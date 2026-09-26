"""
知识图谱增强模块
功能：
  1. CompetitorInferenceModule — 从图结构推断 competitorOf 关系
     （同频段 + 同平台类型 + 不同国家 → 竞争型号）
  2. CoDeploymentInferenceModule — 从 deployedOn 三元组推断 coDeployedWith 关系
     （同一平台 + 不同雷达 → 同平台配套）

使用方式：
    from kg_enrichment import enrich_triples
    enriched = enrich_triples(triples)
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
#  平台类型分组（用于判断是否"同类平台"）
# ═══════════════════════════════════════════════════════

# 将平台关键词归入大类（NavalVessel / AircraftPlatform / GroundPlatform）
PLATFORM_TYPE_HINTS: dict[str, str] = {
    "destroyer":        "NavalVessel",
    "frigate":          "NavalVessel",
    "cruiser":          "NavalVessel",
    "carrier":          "NavalVessel",
    "corvette":         "NavalVessel",
    "submarine":        "NavalVessel",
    "naval":            "NavalVessel",
    "ship":             "NavalVessel",
    "舰":               "NavalVessel",
    "驱逐":             "NavalVessel",
    "护卫":             "NavalVessel",
    "巡洋":             "NavalVessel",
    "fighter":          "AircraftPlatform",
    "aircraft":         "AircraftPlatform",
    "bomber":           "AircraftPlatform",
    "helicopter":       "AircraftPlatform",
    "uav":              "AircraftPlatform",
    "airborne":         "AircraftPlatform",
    "aew":              "AircraftPlatform",
    "awacs":            "AircraftPlatform",
    "预警机":            "AircraftPlatform",
    "战斗机":            "AircraftPlatform",
    "direct":           "GroundPlatform",
    "ground":           "GroundPlatform",
    "mobile":           "GroundPlatform",
    "fixed":            "GroundPlatform",
    "vehicle":          "GroundPlatform",
    "地面":             "GroundPlatform",
    "固定":             "GroundPlatform",
    "车载":             "GroundPlatform",
}


def _platform_class(platform_name: str) -> Optional[str]:
    """将平台名称映射到大类（NavalVessel / AircraftPlatform / GroundPlatform）。"""
    lower = platform_name.lower()
    for kw, cls in PLATFORM_TYPE_HINTS.items():
        if kw in lower:
            return cls
    return None


# ═══════════════════════════════════════════════════════
#  1. CompetitorInferenceModule
# ═══════════════════════════════════════════════════════

class CompetitorInferenceModule:
    """
    从知识图谱结构推断 competitorOf 关系。

    推断规则（AND条件）：
      ① 两型号拥有至少一个共同的 hasFrequencyBand
      ② 两型号的 deployedOn 平台属于同一大类（NavalVessel / Aircraft / Ground）
      ③ 两型号的 operatedBy 国家不同（跨国竞争）
      ④ 两型号 head 不同（排除自比较）

    输出三元组置信度 = 0.60（推断级别，低于直接抽取）
    """

    def infer(self, triples: list[dict],
              min_confidence: float = 0.50) -> list[dict]:
        """
        给定三元组列表，返回推断出的 competitorOf 三元组。
        不修改原始列表，返回新增的三元组。
        """
        # 过滤低置信度三元组
        t_conf = [t for t in triples if t.get("confidence", 0) >= min_confidence]

        # 构建辅助索引
        radar_bands:    dict[str, set[str]] = defaultdict(set)   # radar → {band}
        radar_countries: dict[str, set[str]] = defaultdict(set)  # radar → {country}
        radar_plat_cls: dict[str, set[str]] = defaultdict(set)   # radar → {platform_class}

        for t in t_conf:
            rel  = t.get("relation", "")
            head = t.get("head", "")
            tail = t.get("tail", "")
            if not head or not tail:
                continue
            if rel == "hasFrequencyBand":
                radar_bands[head].add(tail)
            elif rel == "operatedBy":
                radar_countries[head].add(tail)
            elif rel == "deployedOn":
                cls = _platform_class(tail) or t.get("tail_type", "")
                if cls:
                    radar_plat_cls[head].add(cls)

        # 只考虑同时具有频段 + 平台 + 国家信息的雷达
        candidates = [
            r for r in radar_bands
            if radar_bands[r] and radar_countries[r] and radar_plat_cls[r]
        ]

        inferred: list[dict] = []
        seen: set[frozenset] = set()

        for i, r1 in enumerate(candidates):
            for r2 in candidates[i + 1:]:
                # 条件①：共同频段
                shared_bands = radar_bands[r1] & radar_bands[r2]
                if not shared_bands:
                    continue
                # 条件②：同类平台
                shared_cls = radar_plat_cls[r1] & radar_plat_cls[r2]
                if not shared_cls:
                    continue
                # 条件③：不同国家
                if radar_countries[r1] & radar_countries[r2]:
                    continue  # 同一国家的型号不视为竞争
                # 条件④：不同雷达名
                if r1.lower() == r2.lower():
                    continue

                pair = frozenset([r1, r2])
                if pair in seen:
                    continue
                seen.add(pair)

                evidence = (
                    f"inferred: shared band={list(shared_bands)[0]}, "
                    f"platform_class={list(shared_cls)[0]}, "
                    f"different countries: {list(radar_countries[r1])[0]} vs "
                    f"{list(radar_countries[r2])[0]}"
                )
                for head, tail in [(r1, r2), (r2, r1)]:
                    inferred.append({
                        "head":       head,
                        "head_type":  "RadarSystem",
                        "relation":   "competitorOf",
                        "tail":       tail,
                        "tail_type":  "RadarSystem",
                        "confidence": 0.60,
                        "evidence":   evidence,
                        "source":     "inferred_competitor",
                    })

        log.info(f"CompetitorInference: 推断 {len(inferred) // 2} 对竞争关系 "
                 f"（{len(inferred)} 条有向三元组）")
        return inferred


# ═══════════════════════════════════════════════════════
#  2. CoDeploymentInferenceModule
# ═══════════════════════════════════════════════════════

class CoDeploymentInferenceModule:
    """
    从 deployedOn 三元组推断 coDeployedWith 关系。

    规则：若 R1 和 R2 都 deployedOn 同一平台 P，则 R1 coDeployedWith R2。
    置信度 = min(conf_R1_P, conf_R2_P) — 继承最弱的那条证据
    """

    def infer(self, triples: list[dict],
              min_confidence: float = 0.65) -> list[dict]:
        platform_radars: dict[str, list[dict]] = defaultdict(list)

        for t in triples:
            if t.get("relation") == "deployedOn":
                conf = t.get("confidence", 0.5)
                if conf >= min_confidence:
                    platform_radars[t.get("tail", "")].append(t)

        inferred: list[dict] = []
        seen: set[frozenset] = set()

        for platform, ts in platform_radars.items():
            if len(ts) < 2:
                continue
            for i, t1 in enumerate(ts):
                for t2 in ts[i + 1:]:
                    r1 = t1.get("head", "")
                    r2 = t2.get("head", "")
                    if not r1 or not r2 or r1.lower() == r2.lower():
                        continue
                    pair = frozenset([r1, r2])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    conf = round(min(
                        t1.get("confidence", 0.5),
                        t2.get("confidence", 0.5)
                    ), 2)
                    evidence = f"inferred: both deployed on {platform}"
                    for head, tail in [(r1, r2), (r2, r1)]:
                        inferred.append({
                            "head":       head,
                            "head_type":  "RadarSystem",
                            "relation":   "coDeployedWith",
                            "tail":       tail,
                            "tail_type":  "RadarSystem",
                            "confidence": conf,
                            "evidence":   evidence,
                            "source":     "inferred_codeployment",
                        })

        log.info(f"CoDeploymentInference: 推断 {len(inferred) // 2} 对同平台配套关系")
        return inferred


# ═══════════════════════════════════════════════════════
#  公共入口
# ═══════════════════════════════════════════════════════

def enrich_triples(triples: list[dict],
                   infer_competitors: bool = True,
                   infer_codeployment: bool = True,
                   save_path: Optional[str] = None) -> list[dict]:
    """
    对三元组列表执行结构推断增强，返回合并后的完整列表。

    参数：
      triples           原始三元组列表
      infer_competitors 是否推断 competitorOf
      infer_codeployment 是否推断 coDeployedWith
      save_path         若不为 None，将增强后的三元组保存到该路径

    返回：
      原始三元组 + 推断三元组（已去重）
    """
    enriched = list(triples)
    existing_keys = {
        (t.get("head",""), t.get("relation",""), t.get("tail",""))
        for t in enriched
    }

    new_triples: list[dict] = []

    if infer_competitors:
        comp_module = CompetitorInferenceModule()
        new_triples.extend(comp_module.infer(triples))

    if infer_codeployment:
        codepl_module = CoDeploymentInferenceModule()
        new_triples.extend(codepl_module.infer(triples))

    # 去重（不覆盖已有高精度三元组）
    added = 0
    for t in new_triples:
        key = (t.get("head",""), t.get("relation",""), t.get("tail",""))
        if key not in existing_keys:
            existing_keys.add(key)
            enriched.append(t)
            added += 1

    log.info(f"KG 增强完成: 新增 {added} 条推断三元组，总计 {len(enriched)} 条")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        log.info(f"增强后三元组已保存: {out}")

    return enriched


# ═══════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    merged_path = Path("graphrag_index/merged_triples.json")
    if not merged_path.exists():
        print(f"找不到 {merged_path}，请先运行 build_index.py")
        raise SystemExit(1)

    with open(merged_path, encoding="utf-8") as f:
        triples = json.load(f)

    print(f"原始三元组: {len(triples)} 条")

    enriched = enrich_triples(
        triples,
        save_path="graphrag_index/merged_triples_enriched.json"
    )

    from collections import Counter
    rel_dist = Counter(t["relation"] for t in enriched)
    print("\n增强后关系分布:")
    for rel, cnt in rel_dist.most_common():
        src_inferred = sum(
            1 for t in enriched
            if t["relation"] == rel and "inferred" in t.get("source","")
        )
        note = f"  （其中推断: {src_inferred}）" if src_inferred else ""
        print(f"  {rel}: {cnt}{note}")
