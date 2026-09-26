"""
GlobalSecurity.org 雷达数据采集脚本

GlobalSecurity.org 是最大的开源军事装备数据库之一，
包含约 800+ 雷达系统的规格页面，覆盖：
  - 研制商/生产商
  - 装备国家和军种
  - 部署平台
  - 工作频段和功能
  - 型号谱系（改进型/衍生型）

采集策略：
  1. 从已知雷达系统名称构建目标 URL（匹配现有 KG 实体）
  2. 爬取雷达系统索引页获取更多链接
  3. 对每个页面用 LLM 提取三元组

注意：
  - 遵守 robots.txt，请求间隔 ≥ 2 秒
  - 仅用于学术研究，不用于商业目的
  - GlobalSecurity.org 不限制教育/研究用途的访问

用法：
    python scrape_globalsecurity.py              # 爬取索引页 + 提取
    python scrape_globalsecurity.py --from-kg    # 以现有KG实体为起点
    python scrape_globalsecurity.py --limit 50   # 限制页面数量

结果保存到：extraction_results/globalsecurity_results.json
"""

import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from collections import Counter

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PATH  = Path("extraction_results/globalsecurity_results.json")
CACHE_DIR    = Path("radar_corpus/globalsecurity_cache")
OUTPUT_PATH.parent.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLM_CONFIG = {
    "provider":         "deepseek",
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "deepseek_model":   "deepseek-chat",
}

SLEEP_BETWEEN = 2.5   # GlobalSecurity.org 请求间隔（秒）
MAX_TEXT_CHARS = 6000  # 每页最多送给LLM的字符数
MAX_PAGES = 200        # 最多爬取页面数

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; +https://research.example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# GlobalSecurity.org 雷达系统索引页
INDEX_URLS = [
    # 机载雷达
    "https://www.globalsecurity.org/military/systems/aircraft/systems/",
    # 舰载雷达
    "https://www.globalsecurity.org/military/systems/ship/systems/",
    # 地面雷达
    "https://www.globalsecurity.org/military/systems/ground/radar.htm",
    # 导弹防御相关雷达
    "https://www.globalsecurity.org/space/systems/",
    # 通用武器系统
    "https://www.globalsecurity.org/military/systems/",
]

# AN/XXX 命名格式的已知雷达（作为种子，可扩展）
KNOWN_RADAR_SEEDS = [
    # 机载
    "an-apg-63", "an-apg-65", "an-apg-68", "an-apg-71", "an-apg-73",
    "an-apg-77", "an-apg-79", "an-apg-81", "an-apg-83",
    "an-apy-1", "an-apy-2", "an-apy-6", "an-apy-9",
    "an-aps-137", "an-aps-143", "an-aps-145",
    # 地面
    "an-tpy-2", "an-fpq-16", "an-mpr-t", "an-tps-59", "an-tps-70",
    "an-tps-75", "an-tps-77", "an-tps-80",
    # 舰载
    "an-spg-62", "an-spy-1", "an-spy-6", "an-spq-9",
    "an-sps-48", "an-sps-49", "an-sps-73",
    # 欧洲/其他
    "sampson-radar", "smart-l", "herakles-radar",
    "1l119-nebo-svuq", "rezonans-ne", "resonance-ne",
    # 俄系
    "s-400", "s-500", "pantsir",
]


# ══════════════════════════════════════════════════════════
#  Step 1：网页抓取
# ══════════════════════════════════════════════════════════

def fetch_url(url: str, use_cache: bool = True) -> Optional[str]:
    """
    获取网页 HTML，支持本地磁盘缓存（避免重复请求）。
    """
    cache_key = re.sub(r'[^\w]', '_', url)[:120] + ".html"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
        if use_cache:
            cache_path.write_text(html, encoding="utf-8")
        time.sleep(SLEEP_BETWEEN)
        return html
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            log.debug(f"404: {url}")
        else:
            log.warning(f"HTTP {e.response.status_code}: {url}")
        return None
    except Exception as e:
        log.warning(f"请求失败 {url}: {e}")
        return None


def extract_page_text(html: str) -> tuple[str, str]:
    """
    从 GlobalSecurity.org 页面提取正文文字和标题。
    返回 (title, text)
    """
    soup = BeautifulSoup(html, "html.parser")

    # 提取标题
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.text.strip()
        # 去掉 " - GlobalSecurity.org" 后缀
        title = re.sub(r'\s*[-|]\s*GlobalSecurity\.org.*$', '', title).strip()

    # 提取 h1/h2
    for h in soup.find_all(["h1", "h2"])[:3]:
        t = h.get_text(" ", strip=True)
        if t and len(t) < 100:
            title = t
            break

    # 移除导航、广告、脚本
    for tag in soup.find_all(["script", "style", "nav", "header",
                               "footer", "aside", "form", "noscript"]):
        tag.decompose()

    # 尝试定位主要内容区域
    main = (soup.find("div", id="content") or
            soup.find("div", class_="content") or
            soup.find("div", id="main") or
            soup.find("article") or
            soup.find("body"))

    if main:
        # 转为文字，保留段落换行
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    # 清理多余空行
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = "\n".join(lines)

    # 截断（避免超长页面消耗大量 LLM Token）
    return title, text[:MAX_TEXT_CHARS]


def extract_radar_links_from_index(html: str, base_url: str) -> list[str]:
    """
    从索引页提取雷达系统的链接。
    GlobalSecurity.org 索引页通常列出 AN/XXX 等型号链接。
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto"):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # 只保留 globalsecurity.org 的页面
        if "globalsecurity.org" not in parsed.netloc:
            continue
        # 排除图片、PDF
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|mp4)$', parsed.path, re.I):
            continue
        # 只保留 /military/ 或 /space/ 路径下的具体页面
        if not re.search(r'/(military|space)/', parsed.path):
            continue
        # 排除列表/索引页（路径以 / 结尾，认为是目录）
        if parsed.path.endswith("/") and len(parsed.path) > 30:
            continue

        links.add(full_url)

    return sorted(links)


def build_url_from_name(name: str) -> list[str]:
    """
    根据雷达型号名称生成可能的 GlobalSecurity.org URL。
    """
    slug = name.lower().replace(" ", "-").replace("/", "-").replace("_", "-")
    slug = re.sub(r'-+', '-', slug).strip("-")

    candidates = [
        f"https://www.globalsecurity.org/military/systems/aircraft/systems/{slug}.htm",
        f"https://www.globalsecurity.org/military/systems/ship/systems/{slug}.htm",
        f"https://www.globalsecurity.org/military/systems/ground/{slug}.htm",
        f"https://www.globalsecurity.org/military/systems/aircraft/systems/{slug}-radar.htm",
        f"https://www.globalsecurity.org/military/systems/{slug}.htm",
    ]
    return candidates


# ══════════════════════════════════════════════════════════
#  Step 2：LLM 三元组抽取
# ══════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "") -> Optional[str]:
    import requests as req
    key = LLM_CONFIG["deepseek_api_key"]
    if not key:
        log.error("未设置 DEEPSEEK_API_KEY")
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = req.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": LLM_CONFIG["deepseek_model"],
                  "messages": messages,
                  "max_tokens": 2000,
                  "temperature": 0.1},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"LLM 调用失败: {e}")
        return None


SYSTEM_PROMPT = """你是军事雷达领域的知识图谱专家。
从给定的英文网页文本中抽取关于雷达系统的三元组，只输出 JSON 数组。"""

FEW_SHOT = """
示例输入（英文页面）：
"AN/APG-79 is an active electronically scanned array (AESA) radar manufactured by Raytheon.
It is used by the US Navy on F/A-18E/F Super Hornets. The APG-79 is an upgrade of the AN/APG-73.
It provides air-to-air and air-to-ground capabilities."

示例输出：
[
  {"head": "AN/APG-79", "relation": "developedBy", "tail": "Raytheon", "head_type": "Radar", "tail_type": "Manufacturer", "confidence": 0.98, "evidence": "manufactured by Raytheon"},
  {"head": "AN/APG-79", "relation": "operatedBy", "tail": "United States", "head_type": "Radar", "tail_type": "Country", "confidence": 0.95, "evidence": "used by the US Navy"},
  {"head": "AN/APG-79", "relation": "deployedOn", "tail": "F/A-18E/F Super Hornet", "head_type": "Radar", "tail_type": "Platform", "confidence": 0.98, "evidence": "on F/A-18E/F Super Hornets"},
  {"head": "AN/APG-79", "relation": "upgradeOf", "tail": "AN/APG-73", "head_type": "Radar", "tail_type": "Radar", "confidence": 0.98, "evidence": "an upgrade of the AN/APG-73"},
  {"head": "AN/APG-79", "relation": "hasFunction", "tail": "air-to-air combat", "head_type": "Radar", "tail_type": "Function", "confidence": 0.90, "evidence": "air-to-air capabilities"},
  {"head": "AN/APG-79", "relation": "hasFunction", "tail": "air-to-ground attack", "head_type": "Radar", "tail_type": "Function", "confidence": 0.90, "evidence": "air-to-ground capabilities"}
]
"""

RELATION_GUIDE = """
Relations to extract:
- developedBy   : manufacturer / developer
- operatedBy    : country or military branch operating this radar
- deployedOn    : platform (ship, aircraft, vehicle, fixed site)
- hasFrequencyBand: frequency band (X/S/L/C/Ku/Ka/UHF/VHF/P only)
- hasFunction   : radar function (air search, fire control, navigation, EW, etc.)
- affiliatedTo  : company/org belongs to country
- exportedTo    : export destination country
- upgradeOf     : this radar improves upon another (tail = predecessor)
- derivedFrom   : this radar is derived from another design
- competitorOf  : competing radar system

Rules:
- Only use the relations above, no new ones
- Only extract what's explicitly stated, do not infer
- For operatedBy, extract the country name (e.g. "United States"), not the branch
- For deployedOn, use platform model name (e.g. "F/A-18E/F Super Hornet")
- Frequency band: use single letters only (X, S, L, C, Ku, Ka, UHF, VHF, P)
- Omit numerical specs (range, power, weight) — no matching relation type
"""


def extract_triples_from_page(text: str, radar_name: str, url: str) -> list[dict]:
    """对单个网页文本进行三元组抽取。"""
    prompt = f"""{FEW_SHOT}

{RELATION_GUIDE}

Extract triples about radar system: {radar_name}
If no relevant relations found, return [].
Output JSON array only.

Page text:
{text[:MAX_TEXT_CHARS]}"""

    raw = call_llm(prompt, system=SYSTEM_PROMPT)
    if not raw:
        return []

    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

    try:
        triples = json.loads(raw)
        if not isinstance(triples, list):
            return []
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                triples = json.loads(m.group())
            except Exception:
                return []
        else:
            return []

    valid = []
    for t in triples:
        if not all(k in t for k in ("head", "relation", "tail")):
            continue
        t.setdefault("head_type", "Radar")
        t.setdefault("tail_type", "Entity")
        t.setdefault("confidence", 0.85)
        t.setdefault("evidence", "")
        t["source"] = "globalsecurity"
        t["source_file"] = url
        valid.append(t)
    return valid


# ══════════════════════════════════════════════════════════
#  Step 3：去重
# ══════════════════════════════════════════════════════════

def deduplicate(triples: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for t in triples:
        key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
        if key not in best or t.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = t
    return list(best.values())


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def scrape_from_kg_entities(kg_path: str = "graphrag_index/merged_triples.json",
                             limit: int = MAX_PAGES) -> list[dict]:
    """
    以现有KG中的雷达实体名称为起点，在 GlobalSecurity.org 上搜索对应页面。
    这是最精准的方式，直接补充现有KG的知识。
    """
    # 加载现有KG，提取所有雷达实体名称
    try:
        triples = json.load(open(kg_path, encoding="utf-8"))
        radar_names = set()
        for t in triples:
            if t.get("head_type") in ("Radar", "RadarSystem"):
                radar_names.add(t["head"])
            if t.get("tail_type") in ("Radar", "RadarSystem"):
                radar_names.add(t["tail"])
        log.info(f"从现有KG提取 {len(radar_names)} 个雷达实体")
    except Exception as e:
        log.warning(f"无法加载现有KG: {e}，使用内置种子列表")
        radar_names = set(KNOWN_RADAR_SEEDS)

    return _scrape_name_list(list(radar_names)[:limit])


def scrape_from_seeds(limit: int = MAX_PAGES) -> list[dict]:
    """使用内置种子雷达名称列表进行爬取。"""
    return _scrape_name_list(KNOWN_RADAR_SEEDS[:limit])


def scrape_from_index(limit: int = MAX_PAGES) -> list[dict]:
    """
    从索引页发现雷达链接后逐页爬取。
    发现链接数量更多，但精确度稍低。
    """
    all_links = set()
    for idx_url in INDEX_URLS[:2]:  # 只爬前两个索引页，避免过度爬取
        log.info(f"爬取索引页: {idx_url}")
        html = fetch_url(idx_url)
        if html:
            links = extract_radar_links_from_index(html, idx_url)
            all_links.update(links)
            log.info(f"  发现 {len(links)} 个链接（累计 {len(all_links)}）")

    log.info(f"总共发现 {len(all_links)} 个候选页面")
    return _scrape_url_list(list(all_links)[:limit])


def _scrape_name_list(names: list[str]) -> list[dict]:
    """根据名称列表爬取对应页面。"""
    results = []
    processed = set()

    for name in names:
        if name in processed:
            continue
        processed.add(name)

        # 构造可能的 URL
        urls = build_url_from_name(name)
        html = None
        used_url = None

        for url in urls:
            html = fetch_url(url)
            if html and len(html) > 500:
                used_url = url
                break

        if not html:
            log.debug(f"  {name}: 未找到对应页面")
            continue

        result = _process_page(html, used_url, hint_name=name)
        if result and result["triples"]:
            results.append(result)
            log.info(f"  {name}: {result['triple_count']} 条三元组")

    return results


def _scrape_url_list(urls: list[str]) -> list[dict]:
    """从 URL 列表逐页爬取。"""
    results = []
    for url in urls:
        html = fetch_url(url)
        if not html or len(html) < 500:
            continue
        result = _process_page(html, url)
        if result and result["triples"]:
            results.append(result)

    return results


def _process_page(html: str, url: str, hint_name: str = "") -> Optional[dict]:
    """处理单个网页：提取文字 → LLM 抽取三元组。"""
    title, text = extract_page_text(html)

    # 用页面标题或 hint 确定雷达名称
    radar_name = hint_name or title

    # 尝试从标题提取雷达型号
    if not hint_name:
        m = re.search(r'AN/[A-Z]{3}-\d+[A-Z]?', title)
        if m:
            radar_name = m.group()

    if not text.strip() or len(text) < 200:
        return None

    triples = extract_triples_from_page(text, radar_name, url)
    triples = deduplicate(triples)

    return {
        "radar_id":    re.sub(r'[^\w]', '_', radar_name.lower()),
        "en_title":    radar_name,
        "source_url":  url,
        "triple_count": len(triples),
        "triples":     triples,
    }


def run_globalsecurity_scrape(mode: str = "from_kg",
                               limit: int = MAX_PAGES) -> list[dict]:
    """
    主入口：完整 GlobalSecurity.org 爬取流程。

    Args:
        mode: "from_kg"（以现有KG为起点）| "seeds"（内置种子列表）| "index"（从索引页发现）
        limit: 最多处理页面数

    Returns:
        按雷达分组的三元组列表
    """
    log.info(f"=== GlobalSecurity.org 爬取（mode={mode}, limit={limit}）===")

    if mode == "from_kg":
        results = scrape_from_kg_entities(limit=limit)
    elif mode == "seeds":
        results = scrape_from_seeds(limit=limit)
    elif mode == "index":
        results = scrape_from_index(limit=limit)
    else:
        raise ValueError(f"未知 mode: {mode}")

    total = sum(r["triple_count"] for r in results)
    log.info(f"\n爬取完成: {len(results)} 个雷达, {total} 条三元组")

    rel_dist = Counter(t["relation"]
                       for r in results for t in r["triples"])
    log.info("关系分布:")
    for rel, cnt in rel_dist.most_common():
        log.info(f"  {rel}: {cnt}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"保存到: {OUTPUT_PATH}")

    return results


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GlobalSecurity.org 雷达数据采集")
    parser.add_argument("--mode", type=str, default="from_kg",
                        choices=["from_kg", "seeds", "index"],
                        help="爬取模式：from_kg/seeds/index（默认 from_kg）")
    parser.add_argument("--limit", type=int, default=100,
                        help="最多处理页面数（默认 100）")
    args = parser.parse_args()

    run_globalsecurity_scrape(mode=args.mode, limit=args.limit)
    print("\n下一步：python merge_sources.py")
