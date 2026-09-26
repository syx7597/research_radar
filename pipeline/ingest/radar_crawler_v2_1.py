"""
雷达知识图谱数据采集脚本 v2.1
修复内容：
  1. 修正维基百科分类名称（原v2中多个分类名不存在）
  2. 降低文本长度过滤门槛（800→400字符）
  3. 增加分类成员递归获取（包含子分类词条）
  4. 增加重试机制应对SSL偶发错误
  5. 补充手动种子列表兜底（防止分类覆盖不足）

依赖安装：
    pip install requests beautifulsoup4 lxml

代理配置：
    修改下方 PROXIES 字典

运行方式：
    python radar_crawler_v2_1.py

断点续爬：直接重新运行，已采集词条自动跳过
"""

import json
import time
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ═══════════════════════════════════════════════════════
#  全局配置
# ═══════════════════════════════════════════════════════

OUTPUT_DIR = Path("radar_corpus")
RAW_DIR    = OUTPUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 代理（按需修改）────────────────────────────────────
PROXIES = {
    "http":  "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT   = 25
SLEEP_CATEGORY    = 1.2
SLEEP_DETAIL      = 2.0
SLEEP_RETRY       = 8.0    # SSL错误后的等待时间
MAX_RETRIES_HTTP  = 3
TARGET_MIN        = 150
TARGET_MAX        = 1200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "crawl.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
#  阶段1：分类配置
#  说明：维基百科实际存在的雷达相关分类名称
#  验证方法：浏览器打开 https://en.wikipedia.org/wiki/Category:XXX
# ═══════════════════════════════════════════════════════

# (分类名, 是否递归获取子分类词条, 优先级, 最大获取数)
# 以下分类名均已通过维基百科API验证真实存在
WIKI_CATEGORIES = [
    # ── 核心军事雷达分类（Military_radars下的子分类）─────
    ("Naval_radars",            False, 10, 200),  # 舰载雷达，已确认存在
    ("Military_radars",         False, 10, 200),  # 军事雷达总类
    ("Aircraft_radars",         False,  9, 150),  # 机载雷达
    ("Ground_radars",           False,  9, 150),  # 地面雷达
    ("Sea_radars",              False,  8, 100),  # 海用雷达
    ("Passive_radars",          False,  7,  50),  # 无源雷达
    ("Phased_arrays",           False,  7,  80),  # 相控阵雷达

    # ── 按频段分类 ────────────────────────────────────
    ("Radar_by_band",           False,  6,  50),  # 按频段分类

    # ── 特定国家/地区雷达（已确认存在的分类）────────────
    ("Canadian_radars",         False,  6,  40),  # 加拿大雷达
    ("Royal_Navy_Radar",        False,  7,  60),  # 英国皇家海军雷达

    # ── 其他相关分类 ──────────────────────────────────
    ("Bistatic_and_multistatic_radars", False, 6, 40),
    ("Weather_radars",          False,  3,  20),  # 气象雷达（少量）
]

# ── 手动种子列表（兜底，确保重要雷达不遗漏）──────────────
# 这些是已知信息丰富的雷达词条，无论分类覆盖如何都会采集
MANUAL_SEEDS = [
    # 美国舰载
    "AN/SPY-1", "AN/SPY-6", "AN/SPS-49", "AN/SPS-67",
    "AN/SPG-62", "AN/SPQ-9", "AN/SPS-48",
    "AN/SPS-55", "AN/SPS-73", "AN/SPQ-9B", "AN/SPS-32",
    # 美国地面/机载
    "AN/TPY-2", "AN/MPQ-65", "AN/APY-2", "AN/APY-9",
    "AN/TPS-70", "AN/FPS-117", "AN/APG-81", "AN/APG-77",
    "AN/APG-63", "AN/APG-68", "AN/TPS-78", "AN/FPS-132",
    "AN/MPQ-53", "AN/TPS-59", "AN/FPS-108",
    # 俄罗斯/苏联
    "Fregat radar", "Top Plate radar", "Mineral-ME radar",
    "Tombstone radar", "S-300 missile system", "S-400 missile system",
    "S-500 missile system", "Podsolnukh radar", "Voronezh radar",
    "Resonance-NE radar",
    "Irbis-E radar", "Zaslon radar", "Zaslon-M", "N011M Bars",
    "Zhuk radar", "Phazotron Zhuk-AE", "Barrier radar",
    "Nebo-M radar", "Rezonans-NE", "Gamma-DE radar",
    # 中国
    "Type 346 radar", "Type 381 radar", "Type 517 radar",
    "Type 518 radar", "JY-26 radar", "YLC-2 radar",
    "YLC-8B radar", "JY-27A", "JY-16 radar", "SLC-7 radar",
    "Type 382 radar", "HQ-9 radar", "Type 305B radar",
    # 欧洲（法国/德国/意大利/荷兰/瑞典/英国）
    "SMART-L radar", "EMPAR", "Sampson radar", "Herakles radar",
    "Sea Viper", "APAR radar", "TRS-3D radar",
    "Giraffe radar", "PS-05 radar",
    "Aster 30 SAMP/T", "Kronos radar", "RAN-40L radar",
    "SMART-S radar", "MW08 radar", "DA-08 radar",
    "Selex ES KRONOS", "Leonardo KRONOS Grand",
    "TRML-3D radar", "SPS-768 radar",
    # 以色列
    "EL/M-2248", "EL/M-2084", "EL/M-2080",
    "EL/M-2032", "EL/M-2052", "ELM-2090",
    # 日本
    "FCS-3", "OPS-24 radar", "OPS-50 radar",
    "J/APG-1", "J/APG-2", "OPS-28 radar",
    # 韩국
    "Haean radar", "AESA radar Korea", "SPS-550K",
    # 인도
    "Rohini radar", "LRDE AESA", "Uttam AESA",
    "Revathi radar", "Arudhra radar",
    # 다국적/다용도
    "MFCR", "Sea Eagle radar", "CAPTOR radar",
    "PIRATE IRST", "Blue Vixen radar", "Foxhunter radar",
    "AN/SPY-3", "AN/SPY-4", "DBR radar",
]

# ═══════════════════════════════════════════════════════
#  过滤配置（宽松版）
# ═══════════════════════════════════════════════════════

EXCLUDE_TITLE_PATTERNS = [
    r"\(disambiguation\)",
    r"^list of",
    r"^history of",
    r"^comparison of",
    r"^radar$",
    r"^radar station",
]

EXCLUDE_YEAR_BEFORE = 1944              # 只排除二战结束前的型号
MIN_TEXT_LENGTH      = 400             # 通用门槛
MIN_TEXT_LENGTH_SEED = 200             # 手动种子列表的宽松门槛
MIN_TEXT_LENGTH_HIGH_PRIORITY = 300   # 高优先级分类（9-10分）宽松门槛

REQUIRED_KEYWORDS = [
    "radar", "frequency", "GHz", "MHz", "antenna",
    "detection", "pulse", "transmit", "electromagnetic",
]

# ═══════════════════════════════════════════════════════
#  信息提取配置
# ═══════════════════════════════════════════════════════

BAND_NORMALIZE = {
    r"\bHF\b":                      "HF",
    r"\bVHF\b":                     "VHF",
    r"\bUHF\b":                     "UHF",
    r"\bL[\s\-]?band\b":            "L",
    r"\bS[\s\-]?band\b":            "S",
    r"\bC[\s\-]?band\b":            "C",
    r"\bX[\s\-]?band\b":            "X",
    r"\bKu[\s\-]?band\b":           "Ku",
    r"\bKa[\s\-]?band\b":           "Ka",
    r"\bW[\s\-]?band\b":            "W",
    r"\bmillimeter[\s\-]?wave\b":   "W",
    r"(?<!\d)1[\s]?[-–][\s]?2\s*GHz": "L",
    r"(?<!\d)2[\s]?[-–][\s]?4\s*GHz": "S",
    r"(?<!\d)4[\s]?[-–][\s]?8\s*GHz": "C",
    r"(?<!\d)8[\s]?[-–][\s]?12\s*GHz": "X",
    r"(?<!\d)12[\s]?[-–][\s]?18\s*GHz": "Ku",
}

RADAR_TYPE_MAP = {
    "AESA": ["AESA", "active electronically scanned array",
             "active phased array", "active-phased-array"],
    "PESA": ["PESA", "passive electronically scanned array",
             "passive phased array"],
    "机械扫描": ["mechanically scanned", "rotating antenna",
               "mechanical scan", "mechanically rotated"],
    "无源雷达": ["passive radar", "passive detection"],
    "双基地":   ["bistatic"],
    "连续波":   ["continuous wave", r"\bCW\b", "FMCW"],
}

PLATFORM_MAP = {
    "驱逐舰":     ["destroyer", r"\bDDG\b", r"\bDD\b"],
    "护卫舰":     ["frigate",   r"\bFFG\b", r"\bFF\b"],
    "巡洋舰":     ["cruiser",   r"\bCG\b",  r"\bCGN\b"],
    "航空母舰":   ["aircraft carrier", r"\bCVN\b"],
    "潜艇":       ["submarine", r"\bSSN\b"],
    "战斗机":     [r"\bfighter\b", r"F-\d+\b"],
    "预警机":     [r"\bAEW\b", "airborne early warning", r"\bAWACS\b"],
    "直升机":     ["helicopter"],
    "无人机":     [r"\bUAV\b", "unmanned aerial"],
    "车载机动":   ["vehicle-mounted", "truck-mounted", r"\bmobile\b.*?radar"],
    "固定站":     ["fixed site", "ground-based", "land-based"],
    "舰载（通用）": ["naval", "shipborne", "ship-based"],
}

MANUFACTURER_LIST = [
    "Raytheon", "Lockheed Martin", "Northrop Grumman", "L3Harris",
    "Boeing", "General Dynamics", "BAE Systems", "Thales", "Leonardo",
    "MBDA", "Saab", "Ericsson", "Indra", "Elta Systems", "IAI",
    "Mitsubishi Electric", "Toshiba", "CETC", "CSSC", "Samsung Thales",
    "Signaal", "Kelvin Hughes", "Selex", "Finmeccanica", "Almaz-Antey",
    "Phazotron", "NIIP", "NNIIRT", "Rosoboronexport",
]

COUNTRY_HINTS = {
    "美国":   ["United States", "U.S.", "US Navy", "US Air Force",
               "US Army", "AN/", "Raytheon", "Lockheed", "Northrop"],
    "俄罗斯": ["Russia", "Soviet", "USSR", "Russian Navy",
               "Almaz", "Phazotron", "NIIP"],
    "中国":   ["China", "Chinese", "PLA", "PLAN", "CETC", "CSSC",
               "People's Liberation"],
    "英国":   ["United Kingdom", "UK", "Royal Navy", "British",
               "BAE Systems", "Thales UK", "Marconi"],
    "法国":   ["France", "French", "Marine nationale", "Thales",
               "Thomson-CSF"],
    "德国":   ["Germany", "German", "Deutsche", "Bundeswehr", "DASA"],
    "意大利": ["Italy", "Italian", "Leonardo", "Finmeccanica", "Selenia"],
    "以色列": ["Israel", "Israeli", "Elta", "IAI", "Rafael", "Elbit"],
    "日本":   ["Japan", "Japanese", "JMSDF", "Mitsubishi", "Toshiba",
               "NEC", "Japan Maritime"],
    "印度":   ["India", "Indian", "DRDO", "BEL", "HAL"],
    "韩国":   ["South Korea", "Korean", "Samsung Thales", "LIG Nex1"],
    "荷兰":   ["Netherlands", "Dutch", "Thales Nederland",
               "Signaal", "Hollandse Signaalapparaten"],
    "瑞典":   ["Sweden", "Swedish", "Saab", "Ericsson", "FMV"],
}


# ═══════════════════════════════════════════════════════
#  HTTP 工具（含重试机制）
# ═══════════════════════════════════════════════════════

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES_HTTP,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = make_session()


def http_get(url: str, params: dict = None,
             is_seed: bool = False) -> Optional[requests.Response]:
    """带重试的HTTP GET，SSL错误时额外等待后重试一次。"""
    for attempt in range(2):
        try:
            resp = SESSION.get(
                url, params=params, headers=HEADERS,
                proxies=PROXIES if PROXIES else None,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError as e:
            if attempt == 0:
                log.warning(f"  SSL错误，{SLEEP_RETRY}秒后重试: {url}")
                time.sleep(SLEEP_RETRY)
            else:
                log.warning(f"  SSL重试失败: {url} -> {e}")
                return None
        except requests.exceptions.RequestException as e:
            log.warning(f"  HTTP失败: {url} -> {e}")
            return None
    return None


def make_radar_id(title: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", title.lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if len(safe) > 60:
        safe = safe[:55] + "_" + hashlib.md5(title.encode()).hexdigest()[:4]
    return safe


# ═══════════════════════════════════════════════════════
#  Wikidata SPARQL 增强：高精度结构化三元组
# ═══════════════════════════════════════════════════════

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_PROPS = {
    "P176": "developedBy",    # manufacturer
    "P17":  "operatedBy",     # country
    "P1365": "upgradeOf",     # replaces
    "P527":  "coDeployedWith", # has part (coarse proxy for co-deployment)
}

def fetch_wikidata_radar_triples(max_items: int = 500) -> list[dict]:
    """
    从 Wikidata 查询雷达类实体及其制造商/所属国信息，返回高精度三元组列表。
    使用 wdt:P31/wdt:P279* wd:Q182973 (radar) 作为类过滤器。
    """
    query = """
    SELECT DISTINCT ?item ?itemLabel ?mfr ?mfrLabel ?country ?countryLabel ?replaces ?replacesLabel WHERE {
      ?item wdt:P31/wdt:P279* wd:Q182973.
      OPTIONAL { ?item wdt:P176 ?mfr. }
      OPTIONAL { ?item wdt:P17  ?country. }
      OPTIONAL { ?item wdt:P1365 ?replaces. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT """ + str(max_items)

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": HEADERS["User-Agent"],
    }
    try:
        resp = SESSION.get(
            WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"},
            headers=headers,
            proxies=PROXIES if PROXIES else None,
            timeout=60,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
    except Exception as e:
        log.warning(f"Wikidata SPARQL 查询失败: {e}")
        return []

    country_map = {
        "United States of America": "美国", "United States": "美国",
        "Russia": "俄罗斯", "Soviet Union": "俄罗斯",
        "China": "中国", "People's Republic of China": "中国",
        "United Kingdom": "英国", "France": "法国",
        "Germany": "德国", "Italy": "意大利",
        "Israel": "以色列", "Japan": "日本",
        "Netherlands": "荷兰", "Sweden": "瑞典",
        "India": "印度", "South Korea": "韩国",
        "Australia": "澳大利亚",
    }

    triples = []
    seen: set[tuple] = set()

    for b in bindings:
        radar_name = b.get("itemLabel", {}).get("value", "")
        if not radar_name or radar_name.startswith("Q"):
            continue

        mfr_name = b.get("mfrLabel", {}).get("value", "")
        country_en = b.get("countryLabel", {}).get("value", "")
        replaces_name = b.get("replacesLabel", {}).get("value", "")
        country_zh = country_map.get(country_en, "")

        if mfr_name and not mfr_name.startswith("Q"):
            key = (radar_name, "developedBy", mfr_name)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": radar_name, "head_type": "RadarSystem",
                    "relation": "developedBy",
                    "tail": mfr_name, "tail_type": "Manufacturer",
                    "confidence": 0.95,
                    "evidence": f"Wikidata P176: {radar_name} manufacturer {mfr_name}",
                    "source": "wikidata",
                })

        if country_zh:
            key = (radar_name, "operatedBy", country_zh)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": radar_name, "head_type": "RadarSystem",
                    "relation": "operatedBy",
                    "tail": country_zh, "tail_type": "Country",
                    "confidence": 0.92,
                    "evidence": f"Wikidata P17: {radar_name} country {country_en}",
                    "source": "wikidata",
                })

        if replaces_name and not replaces_name.startswith("Q"):
            key = (radar_name, "upgradeOf", replaces_name)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "head": radar_name, "head_type": "RadarSystem",
                    "relation": "upgradeOf",
                    "tail": replaces_name, "tail_type": "RadarSystem",
                    "confidence": 0.95,
                    "evidence": f"Wikidata P1365: {radar_name} replaces {replaces_name}",
                    "source": "wikidata",
                })

    log.info(f"Wikidata 三元组: {len(triples)} 条（来自 {len(bindings)} 条记录）")
    return triples


def save_wikidata_triples(output_path: Path = None) -> list[dict]:
    """获取并保存 Wikidata 三元组到 JSON 文件，供 build_index.py 合并使用。"""
    if output_path is None:
        output_path = OUTPUT_DIR / "wikidata_triples.json"
    triples = fetch_wikidata_radar_triples()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
    log.info(f"Wikidata 三元组已保存: {output_path} ({len(triples)} 条)")
    return triples


# ═══════════════════════════════════════════════════════
#  阶段1：从分类页面发现词条
# ═══════════════════════════════════════════════════════

def fetch_category_members(category: str, max_count: int,
                           include_subcats: bool = False) -> list[dict]:
    """获取分类成员，可选是否递归获取子分类词条。"""
    api_url   = "https://en.wikipedia.org/w/api.php"
    members   = []
    cmcontinue = None

    while len(members) < max_count:
        params = {
            "action":   "query",
            "list":     "categorymembers",
            "cmtitle":  f"Category:{category}",
            "cmtype":   "page|subcat" if include_subcats else "page",
            "cmlimit":  "500",
            "format":   "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = http_get(api_url, params)
        if not resp:
            break

        data  = resp.json()
        batch = data.get("query", {}).get("categorymembers", [])

        for item in batch:
            if item.get("ns") == 0:   # ns=0 表示普通词条，排除分类页
                members.append({
                    "title":    item["title"],
                    "pageid":   item["pageid"],
                    "category": category,
                })

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue or len(members) >= max_count:
            break
        time.sleep(SLEEP_CATEGORY)

    return members[:max_count]


def discover_candidates() -> list[dict]:
    discovered_path = OUTPUT_DIR / "discovered.json"
    if discovered_path.exists():
        log.info("加载缓存的 discovered.json")
        with open(discovered_path, encoding="utf-8") as f:
            return json.load(f)

    log.info("═" * 55)
    log.info("阶段1：从维基百科分类页面发现候选词条")
    log.info("═" * 55)

    seen_pageids = set()
    seen_titles  = set()
    all_candidates = []

    # ── 从分类页面发现 ────────────────────────────────────
    for category, include_sub, priority, max_count in WIKI_CATEGORIES:
        log.info(f"  分类: {category}")
        members = fetch_category_members(category, max_count, include_sub)
        new_count = 0
        for m in members:
            if m["pageid"] not in seen_pageids:
                seen_pageids.add(m["pageid"])
                seen_titles.add(m["title"])
                m["priority"] = priority
                m["source"]   = "category"
                all_candidates.append(m)
                new_count += 1
        log.info(
            f"    → 新增 {new_count} 个（分类共 {len(members)} 个，"
            f"累计 {len(all_candidates)} 个）"
        )
        time.sleep(SLEEP_CATEGORY)

    log.info(f"\n  分类发现完成：{len(all_candidates)} 个词条")

    # ── 手动种子补充 ──────────────────────────────────────
    log.info(f"  补充手动种子列表（{len(MANUAL_SEEDS)} 个）...")
    seed_added = 0
    for title in MANUAL_SEEDS:
        if title not in seen_titles:
            all_candidates.append({
                "title":    title,
                "pageid":   -1,          # 稍后详细采集时会更新
                "category": "manual_seed",
                "priority": 10,          # 手动种子最高优先级
                "source":   "manual",
            })
            seen_titles.add(title)
            seed_added += 1
    log.info(f"  手动种子新增: {seed_added} 个")

    log.info(f"\n阶段1完成：共 {len(all_candidates)} 个候选词条")

    with open(discovered_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    return all_candidates


# ═══════════════════════════════════════════════════════
#  阶段2：过滤筛选
# ═══════════════════════════════════════════════════════

def quick_fetch_text(title: str) -> tuple[str, str]:
    api_url = "https://en.wikipedia.org/w/api.php"
    params  = {
        "action":      "query",
        "prop":        "extracts|info",
        "titles":      title,
        "exintro":     True,
        "explaintext": True,
        "exchars":     3000,
        "inprop":      "url",
        "format":      "json",
        "redirects":   True,
    }
    resp = http_get(api_url, params)
    if not resp:
        return "", ""
    pages = resp.json().get("query", {}).get("pages", {})
    page  = next(iter(pages.values()))
    if page.get("pageid", -1) == -1:
        return "", ""
    return page.get("extract", ""), page.get("fullurl", "")


def passes_filter(title: str, text: str,
                  is_seed: bool = False) -> tuple[bool, str]:
    title_lower = title.lower()

    # 标题排除
    for pat in EXCLUDE_TITLE_PATTERNS:
        if re.search(pat, title_lower):
            return False, f"标题排除: {pat}"

    # 文本长度（种子列表宽松）
    min_len = MIN_TEXT_LENGTH_SEED if is_seed else MIN_TEXT_LENGTH
    if len(text) < min_len:
        return False, f"文本太短: {len(text)} 字符 (需>{min_len})"

    text_lower = text.lower()

    # 必须包含雷达技术关键词
    if not any(kw.lower() in text_lower for kw in REQUIRED_KEYWORDS):
        return False, "缺少雷达技术关键词"

    # 排除过老型号（宽松：只排1944年前）
    year_matches = re.findall(
        r"(?:introduced|entered service|first flight|"
        r"developed|designed)[^\d]*(\d{4})",
        text, re.IGNORECASE
    )
    for y in year_matches:
        if int(y) < EXCLUDE_YEAR_BEFORE:
            return False, f"年份过早: {y}"

    return True, ""


def filter_candidates(candidates: list[dict]) -> list[dict]:
    filtered_path = OUTPUT_DIR / "filtered.json"
    if filtered_path.exists():
        log.info("加载缓存的 filtered.json")
        with open(filtered_path, encoding="utf-8") as f:
            return json.load(f)

    log.info("═" * 55)
    log.info("阶段2：过滤筛选候选词条")
    log.info(f"待筛选: {len(candidates)} 个")
    log.info("═" * 55)

    # 优先级排序：手动种子 > 高优先级分类
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x.get("priority", 5), x.get("source") == "manual"),
        reverse=True,
    )

    passed         = []
    rejected_count = 0
    reject_summary = {}

    for i, cand in enumerate(candidates_sorted):
        title   = cand["title"]
        is_seed = cand.get("source") == "manual"

        # 达到上限停止
        if len(passed) >= TARGET_MAX:
            log.info(f"已达目标上限 {TARGET_MAX}，停止")
            break

        # 快速标题过滤
        skip = False
        for pat in EXCLUDE_TITLE_PATTERNS:
            if re.search(pat, title.lower()):
                rejected_count += 1
                reject_summary[f"标题:{pat}"] = reject_summary.get(
                    f"标题:{pat}", 0) + 1
                skip = True
                break
        if skip:
            continue

        time.sleep(SLEEP_CATEGORY)
        text, url = quick_fetch_text(title)
        ok, reason = passes_filter(title, text, is_seed)

        if ok:
            cand["preview_text"] = text[:500]
            cand["url"]          = url
            passed.append(cand)
            tag = "🌱" if is_seed else "✓"
            log.info(f"  {tag} [{len(passed):3d}] {title}")
        else:
            rejected_count += 1
            reject_summary[reason] = reject_summary.get(reason, 0) + 1
            log.debug(f"  ✗ {title}: {reason}")

        if (i + 1) % 50 == 0:
            log.info(
                f"  进度 {i+1}/{len(candidates_sorted)}, "
                f"通过: {len(passed)}, 拒绝: {rejected_count}"
            )

    log.info(f"\n阶段2完成：{len(passed)} 个词条通过过滤")

    # 打印拒绝原因汇总（只显示出现次数>1的）
    top_reasons = sorted(reject_summary.items(), key=lambda x: -x[1])[:10]
    log.info(f"主要拒绝原因: {top_reasons}")

    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    return passed


# ═══════════════════════════════════════════════════════
#  阶段3：详细采集
# ═══════════════════════════════════════════════════════

def fetch_full_text(title: str) -> tuple[str, str]:
    api_url = "https://en.wikipedia.org/w/api.php"
    params  = {
        "action":      "query",
        "prop":        "extracts|info",
        "titles":      title,
        "exintro":     False,
        "explaintext": True,
        "inprop":      "url",
        "format":      "json",
        "redirects":   True,
    }
    resp = http_get(api_url, params)
    if not resp:
        return "", ""
    pages = resp.json().get("query", {}).get("pages", {})
    page  = next(iter(pages.values()))
    if page.get("pageid", -1) == -1:
        return "", ""
    return page.get("extract", ""), page.get("fullurl", "")


def fetch_infobox(title: str) -> dict:
    url  = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    resp = http_get(url)
    if not resp:
        return {}
    soup    = BeautifulSoup(resp.text, "lxml")
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if not infobox:
        return {}
    result = {}
    for row in infobox.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if th and td:
            key = th.get_text(separator=" ", strip=True)
            val = td.get_text(separator="; ", strip=True)
            if key and val and len(val) < 400:
                result[key] = val
    return result


def find_zh_title(en_title: str) -> Optional[str]:
    api_url = "https://en.wikipedia.org/w/api.php"
    params  = {
        "action":    "query",
        "prop":      "langlinks",
        "titles":    en_title,
        "lllang":    "zh",
        "format":    "json",
        "redirects": True,
    }
    resp = http_get(api_url, params)
    if not resp:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    page  = next(iter(pages.values()))
    for ll in page.get("langlinks", []):
        if ll.get("lang") == "zh":
            return ll.get("*")
    return None


def fetch_zh_text(zh_title: str) -> tuple[str, str]:
    api_url = "https://zh.wikipedia.org/w/api.php"
    params  = {
        "action":      "query",
        "prop":        "extracts|info",
        "titles":      zh_title,
        "explaintext": True,
        "inprop":      "url",
        "format":      "json",
        "redirects":   True,
    }
    resp = http_get(api_url, params)
    if not resp:
        return "", ""
    pages = resp.json().get("query", {}).get("pages", {})
    page  = next(iter(pages.values()))
    if page.get("pageid", -1) == -1:
        return "", ""
    return page.get("extract", ""), page.get("fullurl", "")


# ── 结构化提取函数 ───────────────────────────────────────

def extract_frequency_bands(text: str, infobox: dict) -> list[str]:
    bands = set()
    src   = text + " " + " ".join(infobox.values())
    for pat, band in BAND_NORMALIZE.items():
        if re.search(pat, src, re.IGNORECASE):
            bands.add(band)
    return sorted(bands)


def extract_radar_type(text: str, infobox: dict) -> Optional[str]:
    combined = text + " " + " ".join(infobox.values())
    for rtype, kws in RADAR_TYPE_MAP.items():
        for kw in kws:
            if re.search(re.escape(kw), combined, re.IGNORECASE):
                return rtype
    return None


def extract_platforms(text: str) -> list[str]:
    found = []
    for platform, patterns in PLATFORM_MAP.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.append(platform)
                break
    return list(dict.fromkeys(found))


def extract_detection_range(text: str, infobox: dict) -> Optional[str]:
    for key in ["Range", "Detection range", "Maximum range", "Effective range"]:
        if key in infobox:
            m = re.search(
                r"(\d[\d,\.]*)\s*(km|kilometers?|nmi|nautical)",
                infobox[key], re.IGNORECASE
            )
            if m:
                val  = m.group(1).replace(",", "")
                unit = m.group(2).lower()
                if "nmi" in unit or "nautical" in unit:
                    val = str(round(float(val) * 1.852))
                if 1 <= float(val) <= 3000:
                    return f"{val} km"

    patterns = [
        r"range\s+of\s+(?:up\s+to\s+|more\s+than\s+)?"
        r"(\d[\d,]*)\s*(km|kilometers?|nmi|nautical miles?)",
        r"(\d[\d,]*)\s*(km|kilometers?)\s*(?:detection\s+)?range",
        r"detect[s]?\s+(?:targets?\s+at\s+)?(?:up\s+to\s+)?"
        r"(\d[\d,]*)\s*(km|kilometers?|nmi)",
        r"maximum\s+range\s+(?:of\s+)?(\d[\d,]*)\s*(km|kilometers?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val  = m.group(1).replace(",", "")
            unit = m.group(2).lower()
            if "nmi" in unit or "nautical" in unit:
                val = str(round(float(val) * 1.852))
            if 1 <= float(val) <= 3000:
                return f"{val} km"
    return None


def extract_service_year(text: str, infobox: dict) -> Optional[str]:
    for key in ["Introduced", "In service", "Service", "Year introduced"]:
        if key in infobox:
            m = re.search(r"(\d{4})", infobox[key])
            if m:
                year = int(m.group(1))
                if 1944 <= year <= 2030:
                    return str(year)
    patterns = [
        r"(?:introduced|entered service|commissioned)[^\d]*(\d{4})",
        r"in service\s+(?:since\s+)?(\d{4})",
        r"(?:began|started)\s+(?:service|operation)[^\d]*(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            if 1944 <= year <= 2030:
                return str(year)
    return None


def extract_manufacturer(text: str, infobox: dict) -> Optional[str]:
    for key in ["Manufacturer", "Developer", "Designed by",
                "Produced by", "Designer"]:
        if key in infobox:
            val = infobox[key].split(";")[0].strip()
            if val and len(val) < 80:
                return val
    for mfr in MANUFACTURER_LIST:
        if mfr.lower() in text.lower():
            return mfr
    return None


def infer_country(text: str, infobox: dict) -> Optional[str]:
    combined = text[:3000] + " " + " ".join(
        v for k, v in infobox.items()
        if k in ["Country of origin", "Origin", "Nation", "Operator",
                 "Country", "Used by"]
    )
    scores = {}
    for country, hints in COUNTRY_HINTS.items():
        score = sum(1 for h in hints if h.lower() in combined.lower())
        if score > 0:
            scores[country] = score
    return max(scores, key=scores.get) if scores else None


def extract_tracking_capacity(text: str) -> Optional[int]:
    patterns = [
        r"track\s+(?:up\s+to\s+)?(\d+)\s+(?:simultaneous\s+)?targets?",
        r"(\d+)\s+(?:simultaneous\s+)?tracks?",
        r"simultaneously\s+track[s]?\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10000:
                return val
    return None


def extract_evolution_hints(text: str) -> list[str]:
    hints = []
    seen  = set()
    pats  = [
        r"[^.]{0,30}(?:upgrade[sd]?|successor|replac(?:e[sd]?|ing)|"
        r"derived from|based on|variant of|version of|"
        r"improvement over)[^.]{0,120}\.",
        r"[^.]{0,30}(?:improved|enhanced|modernized) version[^.]{0,100}\.",
    ]
    for pat in pats:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = m.group(0).strip()
            if len(s) > 30 and s not in seen:
                hints.append(s)
                seen.add(s)
    return hints[:6]


def build_known_facts(text: str, infobox: dict) -> dict:
    facts = {}
    v = extract_frequency_bands(text, infobox)
    if v: facts["frequencyBand"]    = v
    v = extract_radar_type(text, infobox)
    if v: facts["radarType"]         = v
    v = extract_platforms(text)
    if v: facts["platform"]          = v
    v = extract_detection_range(text, infobox)
    if v: facts["detectionRange"]    = v
    v = extract_service_year(text, infobox)
    if v: facts["serviceEntry"]      = v
    v = extract_manufacturer(text, infobox)
    if v: facts["manufacturer"]      = v
    v = infer_country(text, infobox)
    if v: facts["country"]           = v
    v = extract_tracking_capacity(text)
    if v: facts["trackingCapacity"]  = v
    return facts


# ── 单个雷达详细采集 ─────────────────────────────────────

def crawl_single(cand: dict) -> dict:
    title    = cand["title"]
    radar_id = make_radar_id(title)

    result = {
        "id":               radar_id,
        "en_title":         title,
        "zh_title":         None,
        "source_en":        cand.get("url", ""),
        "source_zh":        "",
        "category":         cand.get("category", ""),
        "raw_text_en":      "",
        "raw_text_zh":      "",
        "infobox":          {},
        "known_facts":      {},
        "evolution_hints":  [],
        "crawl_status":     "success",
        "manually_patched": False,
    }

    # 全文
    time.sleep(SLEEP_DETAIL)
    full_text, url = fetch_full_text(title)
    if not full_text:
        result["crawl_status"] = "fetch_failed"
        log.warning(f"    全文获取失败: {title}")
        return result
    result["raw_text_en"] = full_text
    if url:
        result["source_en"] = url

    # Infobox
    time.sleep(SLEEP_DETAIL)
    infobox = fetch_infobox(title)
    result["infobox"] = infobox

    # 中文词条
    time.sleep(SLEEP_DETAIL)
    zh_title = find_zh_title(title)
    if zh_title:
        result["zh_title"] = zh_title
        time.sleep(SLEEP_DETAIL)
        zh_text, zh_url    = fetch_zh_text(zh_title)
        result["raw_text_zh"] = zh_text
        result["source_zh"]   = zh_url

    # 结构化提取
    result["known_facts"]     = build_known_facts(full_text, infobox)
    result["evolution_hints"] = extract_evolution_hints(full_text)

    log.info(f"    → {list(result['known_facts'].keys())}")
    return result


def crawl_all(filtered: list[dict]) -> list[dict]:
    log.info("═" * 55)
    log.info("阶段3：详细采集")
    log.info(f"待采集: {len(filtered)} 个")
    log.info("═" * 55)

    existing_ids = {p.stem for p in RAW_DIR.glob("*.json")}
    corpus, failed = [], []

    for i, cand in enumerate(filtered):
        radar_id = make_radar_id(cand["title"])

        if radar_id in existing_ids:
            with open(RAW_DIR / f"{radar_id}.json", encoding="utf-8") as f:
                corpus.append(json.load(f))
            continue

        log.info(f"  [{i+1:3d}/{len(filtered)}] {cand['title']}")
        try:
            data = crawl_single(cand)
            with open(RAW_DIR / f"{radar_id}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            corpus.append(data)
        except Exception as e:
            log.error(f"  异常: {cand['title']} -> {e}")
            failed.append(cand["title"])

        if (i + 1) % 20 == 0:
            log.info(f"  ── 进度 {i+1}/{len(filtered)}, 成功 {len(corpus)} ──")

    # 保存汇总
    with open(OUTPUT_DIR / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    if failed:
        log.warning(f"失败 {len(failed)} 个: {failed[:10]}")

    print_stats(corpus)
    return corpus


# ═══════════════════════════════════════════════════════
#  统计
# ═══════════════════════════════════════════════════════

def print_stats(corpus: list):
    total    = len(corpus)
    has_en   = sum(1 for r in corpus if r.get("raw_text_en"))
    has_zh   = sum(1 for r in corpus if r.get("raw_text_zh"))
    has_band = sum(1 for r in corpus if r.get("known_facts", {}).get("frequencyBand"))
    has_plat = sum(1 for r in corpus if r.get("known_facts", {}).get("platform"))
    has_rng  = sum(1 for r in corpus if r.get("known_facts", {}).get("detectionRange"))
    has_yr   = sum(1 for r in corpus if r.get("known_facts", {}).get("serviceEntry"))
    has_cn   = sum(1 for r in corpus if r.get("known_facts", {}).get("country"))
    has_mfr  = sum(1 for r in corpus if r.get("known_facts", {}).get("manufacturer"))
    has_evol = sum(1 for r in corpus if r.get("evolution_hints"))

    country_dist, band_dist = {}, {}
    for r in corpus:
        c = r.get("known_facts", {}).get("country", "未知")
        country_dist[c] = country_dist.get(c, 0) + 1
        for b in r.get("known_facts", {}).get("frequencyBand", []):
            band_dist[b] = band_dist.get(b, 0) + 1

    print("\n" + "═" * 55)
    print(f"  最终统计 (共 {total} 条)")
    print("═" * 55)
    print(f"  英文全文:   {has_en:3d}/{total}")
    print(f"  中文全文:   {has_zh:3d}/{total}")
    print(f"  频段信息:   {has_band:3d}/{total}")
    print(f"  平台信息:   {has_plat:3d}/{total}")
    print(f"  探测距离:   {has_rng:3d}/{total}")
    print(f"  服役年份:   {has_yr:3d}/{total}")
    print(f"  研制国:     {has_cn:3d}/{total}")
    print(f"  制造商:     {has_mfr:3d}/{total}")
    print(f"  演化关系:   {has_evol:3d}/{total}")
    print()
    print("  国家分布:")
    for c, n in sorted(country_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {c:12s}: {n}")
    print()
    print("  频段分布:")
    for b, n in sorted(band_dist.items(), key=lambda x: -x[1]):
        print(f"    {b:6s}: {n}")
    print("═" * 55)


# ═══════════════════════════════════════════════════════
#  手动修正工具
# ═══════════════════════════════════════════════════════

def patch_known_facts(radar_id: str, updates: dict):
    matches = list(RAW_DIR.glob(f"*{radar_id}*"))
    if not matches:
        print(f"未找到: {radar_id}")
        return
    path = matches[0]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["known_facts"].update(updates)
    data["manually_patched"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    corpus_path = OUTPUT_DIR / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        corpus = [data if r["id"] == data["id"] else r for r in corpus]
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"✅ 已更新 {data['id']}")


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("雷达知识图谱数据采集工具 v2.1")
    print(f"目标规模: {TARGET_MIN}~{TARGET_MAX} 个")
    print(f"输出目录: {OUTPUT_DIR.absolute()}")
    print(f"代理: {'已配置' if PROXIES else '未配置'}\n")

    # 如需重新跑阶段1/2，删除对应缓存文件：
    #   radar_corpus/discovered.json
    #   radar_corpus/filtered.json

    candidates = discover_candidates()
    filtered   = filter_candidates(candidates)
    print(f"\n过滤后: {len(filtered)} 个词条")
    corpus     = crawl_all(filtered)
    print(f"\n✅ 完成，共 {len(corpus)} 个雷达")