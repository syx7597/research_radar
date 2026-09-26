"""
E4: 非 Wikipedia 开源语料采集 —— Radartutorial.eu + GlobalSecurity.org。

范围（用户裁决 B）：两站全量雷达页，不限于语料已有的 250 型号。
只采原文与规格表，**不做 LLM 抽取**（抽取统一留给 v3 流水线的接地门控）。

  - Radartutorial.eu  "Card index of radar sets"（GFDL + CC BY-SA 3.0，注明来源）
      索引 19.kartei/ka03.en.html → 每卡: h1/h4 型号名 + 双列规格表 + 描述段落
  - GlobalSecurity.org 雷达系统页（复用 scrape_globalsecurity 的缓存与解析，
      去掉 6000 字符截断；索引发现 + AN/XXX 种子）

产出：追加写入 radar_corpus/corpus.json（同 schema；id 带 __rt / __gs 后缀；
     幂等——已存在的 id 跳过）。HTML 缓存在 radar_corpus/*_cache/，重跑免抓。

运行: python pipeline/ingest/scrape_aux_sources.py [--rt-only|--gs-only] [--limit N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "ingest"))

import scrape_globalsecurity as gs  # noqa: E402

gs.MAX_TEXT_CHARS = 10**9           # 语料要全文，去掉给 LLM 准备的截断

CORPUS_PATH = ROOT / "radar_corpus" / "corpus.json"
RT_CACHE    = ROOT / "radar_corpus" / "radartutorial_cache"
RT_CACHE.mkdir(parents=True, exist_ok=True)

RT_BASE   = "https://www.radartutorial.eu/19.kartei/"
RT_INDEX  = RT_BASE + "ka03.en.html"
RT_SLEEP  = 0.8
MIN_TEXT  = 250      # 正文短于此且规格 <3 键 → 视为空卡，跳过

RT_CATEGORY = {
    "01.oth":      "Ground_radars",   # over-the-horizon
    "02.surv":     "Ground_radars",   # air surveillance
    "03.atc":      "Ground_radars",   # ATC
    "04.battle":   "Ground_radars",   # battlefield
    "05.erly":     "Ground_radars",   # early warning
    "06.missile":  "Ground_radars",   # air defense / SAM
    "07.naval":    "Naval_radars",
    "08.airborne": "Aircraft_radars",
    "09.anti":     "Ground_radars",
    "10.weather":  "Weather_radars",
    "13.labs":     "Military_radars",
}

HEADERS = {"User-Agent": gs.HEADERS["User-Agent"]}


def slugify(name: str) -> str:
    s = name.lower().replace("/", "_").replace(" ", "_")
    return re.sub(r"[^\w\-]+", "", re.sub(r"_+", "_", s)).strip("_") or "unnamed"


def fetch_cached(url: str, cache_dir: Path, sleep: float) -> str:
    key = re.sub(r"[^\w]", "_", url)[:120] + ".html"
    p = cache_dir / key
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        p.write_text(resp.text, encoding="utf-8")
        time.sleep(sleep)
        return resp.text
    except Exception as e:
        print(f"  !! fetch {url}: {e}")
        time.sleep(sleep)
        return ""


# ─────────────────────────────────────────────────────────────
#  Radartutorial
# ─────────────────────────────────────────────────────────────

def rt_card_links() -> list[str]:
    html = fetch_cached(RT_INDEX, RT_CACHE, RT_SLEEP)
    hrefs = set(re.findall(r'href="((?:\d\d\.\w+/)?karte\d+\.en\.html)"', html))
    return sorted(urljoin(RT_INDEX, h) for h in hrefs)


def rt_parse_card(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    for junk in soup.find_all(["script", "style", "nav"]):
        junk.decompose()

    # 型号名：<title> "AN/FPS-85 - Radartutorial" 最可靠；h4 兜底
    # （h1 是栏目标题 "Card Index of Radar Sets - ..."，不能用）
    name = ""
    tt = soup.find("title")
    if tt:
        name = re.sub(r"\s*[-–—]\s*Radartutorial.*$", "",
                      tt.get_text(" ", strip=True)).strip()
    if not name or name.lower().startswith("card index"):
        h4 = soup.find("h4")
        name = h4.get_text(" ", strip=True) if h4 else ""
    if not name:
        return None

    def _clean(cell) -> str:
        for sup in cell.find_all("sup"):
            sup.decompose()
        t = cell.get_text(" ", strip=True).replace("\xa0", " ")
        t = re.sub(r"\s*\[\d+\]", "", t)          # 残留的 [1] 脚注
        return re.sub(r"\s+", " ", t).strip()

    # 双列规格表 → infobox
    specs = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) != 2:
            continue
        k = _clean(cells[0]).rstrip(":").strip()
        v = _clean(cells[1])
        if k and v and len(k) < 60 and len(v) < 400:
            specs[k] = v

    # 描述段落（排除表格内的 p）
    paras = []
    for p in soup.find_all("p"):
        if p.find_parent("table"):
            continue
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if t and len(t) > 40:
            paras.append(t)
    text = "\n".join(paras)

    if len(text) < MIN_TEXT and len(specs) < 3:
        return None

    subdir = url.rsplit("/", 2)[-2]
    return {
        "id":            slugify(name) + "__rt",
        "en_title":      name,
        "zh_title":      None,
        "source_en":     url,
        "source_zh":     None,
        "category":      RT_CATEGORY.get(subdir, "Military_radars"),
        "raw_text_en":   text,
        "raw_text_zh":   "",
        "infobox":       specs,
        "known_facts":   {},
        "evolution_hints": [],
        "crawl_status":  "success",
        "manually_patched": False,
    }


def scrape_radartutorial(limit: int | None) -> list[dict]:
    links = rt_card_links()
    if limit:
        links = links[:limit]
    print(f"[RT] {len(links)} card links")
    docs = []
    for i, url in enumerate(links):
        html = fetch_cached(url, RT_CACHE, RT_SLEEP)
        if not html:
            continue
        doc = rt_parse_card(html, url)
        if doc:
            docs.append(doc)
        if (i + 1) % 100 == 0:
            print(f"[RT] {i+1}/{len(links)} fetched, {len(docs)} kept")
    print(f"[RT] done: {len(docs)} cards kept")
    return docs


# ─────────────────────────────────────────────────────────────
#  GlobalSecurity
# ─────────────────────────────────────────────────────────────

# 预过滤：URL slug 像雷达 / 传感器的才抓正文
GS_SLUG_RX = re.compile(
    r"(radar|an-[a-z]{2,3}-?\d|spy|sps-|spg-|spq-|tps-|fps-|mpq-|apg-|apy-|"
    r"aps-|apq-|asr-|arsr|nebo|voronezh|daryal|dnepr|don-2|sampson|smart-l|"
    r"herakles|giraffe|erieye|sostar|jstars|awacs|aew|otr|oth-)", re.I)


def gs_looks_like_radar(title: str, text: str) -> bool:
    if re.search(r"radar", title, re.I):
        return True
    return len(re.findall(r"\bradar\b", text, re.I)) >= 3


def scrape_globalsec(limit: int | None) -> list[dict]:
    candidates = set()
    for idx_url in gs.INDEX_URLS:
        html = gs.fetch_url(idx_url)
        if html:
            for u in gs.extract_radar_links_from_index(html, idx_url):
                if GS_SLUG_RX.search(u.rsplit("/", 1)[-1]):
                    candidates.add(u)
    for seed in gs.KNOWN_RADAR_SEEDS:
        candidates.update(gs.build_url_from_name(seed))
    candidates = sorted(candidates)
    if limit:
        candidates = candidates[:limit]
    print(f"[GS] {len(candidates)} candidate urls")

    docs = []
    for i, url in enumerate(candidates):
        if url.rstrip("/").endswith(("systems", "military", "space")):
            continue                      # 目录索引页不是文档
        html = gs.fetch_url(url)          # 2.5s 限速 + 磁盘缓存，404 返回 None
        if not html:
            continue
        # 标题必须取 <title>：GS 页面的 h1 恒为栏目名 "Military"，
        # gs.extract_page_text 的 h1/h2 覆盖逻辑会让所有页撞成同一标题
        _, text = gs.extract_page_text(html)
        m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"\s*[-|]\s*GlobalSecurity\.org.*$", "",
                       m.group(1).strip()) if m else ""
        if not title or title.lower() in ("military", "space") \
                or len(text) < MIN_TEXT or not gs_looks_like_radar(title, text):
            continue
        docs.append({
            "id":            slugify(title) + "__gs",
            "en_title":      title,
            "zh_title":      None,
            "source_en":     url,
            "source_zh":     None,
            "category":      "Military_radars",
            "raw_text_en":   text,
            "raw_text_zh":   "",
            "infobox":       {},
            "known_facts":   {},
            "evolution_hints": [],
            "crawl_status":  "success",
            "manually_patched": False,
        })
        if (i + 1) % 50 == 0:
            print(f"[GS] {i+1}/{len(candidates)} fetched, {len(docs)} kept")
    print(f"[GS] done: {len(docs)} pages kept")
    return docs


# ─────────────────────────────────────────────────────────────
#  合并入 corpus.json（幂等）
# ─────────────────────────────────────────────────────────────

def merge(new_docs: list[dict]) -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    seen_ids  = {d["id"] for d in corpus}
    seen_urls = {d.get("source_en") for d in corpus}
    added = 0
    for d in new_docs:
        if d["id"] in seen_ids or d["source_en"] in seen_urls:
            continue
        corpus.append(d)
        seen_ids.add(d["id"])
        seen_urls.add(d["source_en"])
        added += 1
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    total_en = sum(len(x.get("raw_text_en") or "") for x in corpus)
    print(f"\n==== E4 merge ====\nadded {added} docs -> corpus {len(corpus)} docs, "
          f"EN {total_en:,} chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-only", action="store_true")
    ap.add_argument("--gs-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    docs = []
    if not args.gs_only:
        docs += scrape_radartutorial(args.limit)
    if not args.rt_only:
        docs += scrape_globalsec(args.limit)
    merge(docs)


if __name__ == "__main__":
    main()
