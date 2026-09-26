"""
实体规范化模块
在三元组写入 Neo4j 之前调用，统一实体名称和消除歧义

使用方法:
  from entity_normalizer import normalize_triples
  triples = normalize_triples(triples)
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════
#  1. 制造商名称规范化字典
#     格式: "原始写法（或别名）": "规范名称"
# ══════════════════════════════════════════════════════════

COMPANY_ALIASES = {
    # Bell 系列
    "Bell Labs":                        "Bell Laboratories",
    "Bell Telephone Laboratories":      "Bell Laboratories",
    "Bell Lab":                         "Bell Laboratories",

    # General Electric
    "GE":                               "General Electric",
    "G.E.":                             "General Electric",

    # Hughes
    "Hughes":                           "Hughes Aircraft Company",
    "Hughes Aircraft":                  "Hughes Aircraft Company",

    # Raytheon 系列（Raytheon 先后改名）
    "Raytheon Technologies":            "Raytheon",
    "RTX":                              "Raytheon",
    "RTX Corporation":                  "Raytheon",
    "Raytheon Company":                 "Raytheon",
    "Raytheon Missile Systems":         "Raytheon",

    # BAE Systems 子公司统一到母公司
    "BAE Systems Maritime":             "BAE Systems",
    "BAE Systems Integrated System Technologies": "BAE Systems",
    "BAE Systems Electronics":          "BAE Systems",

    # Lockheed Martin
    "Lockheed":                         "Lockheed Martin",
    "Lockheed Corporation":             "Lockheed Martin",
    "Martin Marietta":                  "Lockheed Martin",

    # Northrop Grumman
    "Northrop":                         "Northrop Grumman",
    "Grumman":                          "Northrop Grumman",
    "Westinghouse":                     "Northrop Grumman",
    "Westinghouse Electric":            "Northrop Grumman",
    "Westinghouse (Northrop Grumman)":  "Northrop Grumman",

    # L3Harris / Harris
    "Harris":                           "L3Harris",
    "L3 Technologies":                  "L3Harris",

    # Thales 子公司
    "Thales Group":                     "Thales",
    "Thales Nederland":                 "Thales",
    "Thomson-CSF":                      "Thales",

    # Leonardo / Finmeccanica / Selex
    "Selex ES":                         "Leonardo",
    "SELEX Sistemi Integrati":          "Leonardo",
    "Finmeccanica":                     "Leonardo",
    "Leonardo S.p.A.":                  "Leonardo",
    "Selex":                            "Leonardo",

    # 以色列 IAI / Elta
    "IAI":                              "Israel Aerospace Industries",
    "Israel Aerospace":                 "Israel Aerospace Industries",
    "Elta Systems":                     "Israel Aerospace Industries",
    "ELTA":                             "Israel Aerospace Industries",
    "Elta":                             "Israel Aerospace Industries",

    # ITT
    "ITT-Gilfillan":                    "ITT Gilfillan",
    "ITT":                              "ITT Gilfillan",

    # Japan Radio
    "Japan Radio Company":              "Japan Radio",
    "JRC":                              "Japan Radio",

    # 印度 BEL
    "Bharat Electronics Limited":       "Bharat Electronics",
    "BEL":                              "Bharat Electronics",

    # 中国
    "CETC":                             "中国电子科技集团",
    "中国电子科技集团公司":              "中国电子科技集团",

    # 噪声实体（抽取错误的，直接标记为删除）
    "British":                          "__DELETE__",
    "China":                            "__DELETE__",
    "Germany":                          "__DELETE__",
    "Electronics":                      "__DELETE__",
    "Japan after World War II":         "__DELETE__",
}


# ══════════════════════════════════════════════════════════
#  2. 国家名称规范化字典（统一中文）
# ══════════════════════════════════════════════════════════

COUNTRY_ALIASES = {
    # 英文 → 中文
    "United States":        "美国",
    "United States of America": "美国",
    "USA":                  "美国",
    "US":                   "美国",
    "America":              "美国",
    "American":             "美国",
    "Russia":               "俄罗斯",
    "Russian":              "俄罗斯",
    "Soviet Union":         "俄罗斯",    # 苏联归并到俄罗斯（可按需改为独立节点）
    "USSR":                 "俄罗斯",
    "China":                "中国",
    "Chinese":              "中国",
    "PRC":                  "中国",
    "Japan":                "日本",
    "Japanese":             "日本",
    "United Kingdom":       "英国",
    "UK":                   "英国",
    "Britain":              "英国",
    "British":              "英国",
    "France":               "法国",
    "French":               "法国",
    "Germany":              "德国",
    "German":               "德国",
    "Italy":                "意大利",
    "Italian":              "意大利",
    "Israel":               "以色列",
    "Israeli":              "以色列",
    "India":                "印度",
    "Indian":               "印度",
    "Sweden":               "瑞典",
    "Swedish":              "瑞典",
    "Netherlands":          "荷兰",
    "Dutch":                "荷兰",
    "South Korea":          "韩国",
    "Republic of Korea":    "韩国",
    "Korea":                "韩国",
    "Norway":               "挪威",
    "Norwegian":            "挪威",
    "Spain":                "西班牙",
    "Spanish":              "西班牙",
    "Australia":            "澳大利亚",
    "Australian":           "澳大利亚",
    "Canada":               "加拿大",
    "Canadian":             "加拿大",
    "Singapore":            "新加坡",
    "Taiwan":               "台湾",
    "Pakistan":             "巴基斯坦",
    "Iran":                 "伊朗",
    "Ukraine":              "乌克兰",
    "Azerbaijan":           "阿塞拜疆",
    "Finland":              "芬兰",
    "Poland":               "波兰",
    "Greece":               "希腊",
    "Turkey":               "土耳其",
    "Brazil":               "巴西",
    "Egypt":                "埃及",
    "Saudi Arabia":         "沙特阿拉伯",
}


# ══════════════════════════════════════════════════════════
#  3. 频段规范化（统一大写字母）
# ══════════════════════════════════════════════════════════

FREQUENCY_ALIASES = {
    "x":     "X",
    "x-band": "X",
    "X-band": "X",
    "X band": "X",
    "X_Band": "X",
    "s":     "S",
    "s-band": "S",
    "S-band": "S",
    "S band": "S",
    "l":     "L",
    "l-band": "L",
    "L-band": "L",
    "L band": "L",
    "c":     "C",
    "c-band": "C",
    "C-band": "C",
    "C band": "C",
    "ku":    "Ku",
    "KU":    "Ku",
    "Ku-band": "Ku",
    "ka":    "Ka",
    "KA":    "Ka",
    "Ka":    "Ka",
    "Ka-band": "Ka",
    "uhf":   "UHF",
    "UHF-band": "UHF",
    "vhf":   "VHF",
    "VHF-band": "VHF",
    "p":     "P",
    "P-band": "P",
    "w":     "W",
    "W-band": "W",
    "mmw":   "mmW",
    "millimeter": "mmW",
    "e":     "E",
    "f":     "F",
    "g":     "G",
    "h":     "H",
    "i":     "I",
    "j":     "J",
    "k":     "K",
}


# ══════════════════════════════════════════════════════════
#  4. 中文平台类别 → 规范英文（可选，统一语言）
#     注意：方法A提取的是类别，方法C提取的是具体型号
#     这里只对纯类别名做规范，具体型号保持不变
# ══════════════════════════════════════════════════════════

PLATFORM_ZH_TO_EN = {
    "驱逐舰":       "destroyer",
    "护卫舰":       "frigate",
    "巡洋舰":       "cruiser",
    "航空母舰":     "aircraft carrier",
    "潜艇":         "submarine",
    "舰载（通用）": "naval vessel",
    "直升机":       "helicopter",
    "战斗机":       "fighter aircraft",
    "预警机":       "airborne early warning aircraft",
    "无人机":       "unmanned aerial vehicle",
    "固定站":       "fixed installation",
    "车载机动":     "mobile ground vehicle",
}


# ══════════════════════════════════════════════════════════
#  规范化函数
# ══════════════════════════════════════════════════════════

def normalize_entity(name: str, entity_type: str, relation: str) -> Optional[str]:
    """
    对单个实体名称做规范化。
    返回 None 表示这条三元组应该被删除（噪声实体）。
    """
    name = name.strip()
    if not name:
        return None

    # 按关系类型选择规范化策略
    if relation in ("developedBy",) or entity_type in ("Manufacturer", "Company"):
        result = COMPANY_ALIASES.get(name, name)
        if result == "__DELETE__":
            return None
        return result

    if relation in ("operatedBy", "exportedTo", "affiliatedTo") or entity_type == "Country":
        return COUNTRY_ALIASES.get(name, name)

    if relation == "hasFrequencyBand" or entity_type == "FrequencyBand":
        return FREQUENCY_ALIASES.get(name, name.upper() if len(name) <= 3 else name)

    if relation == "deployedOn" or entity_type in ("Platform", "NavalVessel",
                                                    "AircraftPlatform", "GroundPlatform"):
        # 中文类别名 → 英文（可选，如果你想统一成英文）
        # 注释掉下面这行则保留中文
        # return PLATFORM_ZH_TO_EN.get(name, name)
        return name  # 目前保持原样，中英文都保留

    return name


def normalize_triple(t: dict) -> Optional[dict]:
    """
    对单条三元组做规范化。
    返回 None 表示应该丢弃这条三元组。
    """
    t = dict(t)

    # 规范化 head
    new_head = normalize_entity(
        t.get("head", ""),
        t.get("head_type", ""),
        ""  # head 通常是雷达名，不需要关系上下文
    )
    if new_head is None:
        return None
    t["head"] = new_head

    # 规范化 tail
    new_tail = normalize_entity(
        t.get("tail", ""),
        t.get("tail_type", ""),
        t.get("relation", "")
    )
    if new_tail is None:
        return None
    t["tail"] = new_tail

    # 过滤掉明显的噪声（tail 超长说明抽取错误）
    if len(t["tail"]) > 80:
        return None
    if len(t["head"]) > 80:
        return None

    return t


def normalize_triples(triples: list[dict]) -> list[dict]:
    """
    对三元组列表做完整规范化，包括：
      1. 实体名称规范化
      2. 去除噪声三元组
      3. 去重（规范化后可能产生新的重复）
    """
    normalized = []
    deleted = 0
    for t in triples:
        result = normalize_triple(t)
        if result is None:
            deleted += 1
        else:
            normalized.append(result)

    # 规范化后去重（保留置信度最高的）
    best: dict[tuple, dict] = {}
    for t in normalized:
        key = (t.get("head",""), t.get("relation",""), t.get("tail",""))
        if key not in best or t.get("confidence",0) > best[key].get("confidence",0):
            best[key] = t

    result_list = list(best.values())

    print(f"规范化结果: {len(triples)} → {len(result_list)} 条"
          f"（删除噪声 {deleted} 条，去重减少 {len(normalized)-len(result_list)} 条）")
    return result_list


# ══════════════════════════════════════════════════════════
#  调试工具：查看规范化前后对比
# ══════════════════════════════════════════════════════════

def show_normalization_diff(triples: list[dict]):
    """打印规范化前后有变化的三元组，用于检查规范化效果。"""
    changes = []
    for t in triples:
        result = normalize_triple(dict(t))
        if result is None:
            changes.append(("DELETE", t.get("head",""), t.get("relation",""), t.get("tail","")))
        elif result["head"] != t.get("head","") or result["tail"] != t.get("tail",""):
            changes.append(("CHANGE",
                            f"{t['head']} → {result['head']}",
                            t["relation"],
                            f"{t['tail']} → {result['tail']}"))

    print(f"共 {len(changes)} 条有变化（含删除）:")
    deletes = [c for c in changes if c[0] == "DELETE"]
    modifies = [c for c in changes if c[0] == "CHANGE"]
    print(f"  删除: {len(deletes)} 条")
    print(f"  修改: {len(modifies)} 条")
    print()
    print("修改样本（前30条）:")
    for _, head, rel, tail in modifies[:30]:
        print(f"  [{rel}]  {head}  ||  {tail}")
    print()
    print("删除样本（前10条）:")
    for _, head, rel, tail in deletes[:10]:
        print(f"  [{rel}]  {head} → {tail}")
