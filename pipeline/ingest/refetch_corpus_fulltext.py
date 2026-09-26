"""
E1: 存量语料全文重抓（不改准入、不加条目，只提升每篇完整度）。

针对 radar_corpus/corpus.json 的 250 篇：
  1. raw_text_en  ← Wikipedia 全文（旧值只有导语；TextExtracts API 即使
                    exintro=False 也只回导语，故改为解析页面 HTML 正文，
                    章节标题保留为 "== Heading ==" 行）
  2. raw_text_zh  ← zh 维基全文（zh_title 缺失的先经 langlinks 反查）
  3. infobox      ← 干净重解析：剔除 <sup> 引用角标 / <style>，仅 <br>/<li> 作分隔
  4. crawl_status / source_en 同步更新；known_facts / evolution_hints 不动

每篇文档 EN 只需一次 HTML 请求（正文 + infobox 同一响应解析）。
写回 corpus.json 原格式（旧状态在 git tag kg-v2-freeze，可 diff / 回滚）。
每 25 篇写一次 checkpoint（corpus.json.refetch.tmp），中断可续跑。

运行: python pipeline/ingest/refetch_corpus_fulltext.py
"""

import json
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline" / "ingest"))

from radar_crawler_v2_1 import http_get, find_zh_title  # noqa: E402

CORPUS_PATH = ROOT / "radar_corpus" / "corpus.json"
TMP_PATH    = ROOT / "radar_corpus" / "corpus.json.refetch.tmp"
SLEEP       = 0.4          # 礼貌限速：每请求间隔
CHECKPOINT  = 25

# 这些章节是引用/导航，不是内容，跳过
SKIP_SECTIONS = {
    "references", "external links", "see also", "further reading",
    "notes", "bibliography", "sources", "citations", "gallery",
    "参考文献", "外部链接", "参见", "延伸阅读", "注释", "参考资料", "图集",
}


# ─────────────────────────────────────────────────────────────
#  HTML 页面抓取 + 正文/infobox 解析
# ─────────────────────────────────────────────────────────────

def fetch_page_soup(title: str, lang: str = "en"):
    url  = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    resp = http_get(url)          # requests 默认跟随重定向
    if not resp or resp.status_code != 200:
        return None, ""
    return BeautifulSoup(resp.text, "lxml"), resp.url


def extract_article_text(soup) -> str:
    """从渲染 HTML 提取全文纯文本，章节标题保留为 == Heading == 行。"""
    content = soup.select_one("div.mw-parser-output")
    if not content:
        return ""

    # 全局剔除：引用角标 / 样式 / 脚本 / 编辑按钮 / 坐标等页面装饰
    for sel in ("sup.reference", "style", "script", "span.mw-editsection",
                "sup.noprint", "span#coordinates", "div.hatnote"):
        for node in content.select(sel):
            node.decompose()

    lines, skipping = [], False
    for el in content.find_all(["h2", "h3", "h4", "p", "ul", "ol", "dl"],
                               recursive=True):
        # 跳过表格/导航框/信息框内部的元素（正文只取顶层叙述与列表）
        if el.find_parent(["table", "figure"]) is not None:
            continue
        if el.find_parent("div", class_=re.compile(
                r"navbox|reflist|refbegin|toc|thumb|sidebar|metadata")):
            continue

        if el.name in ("h2", "h3", "h4"):
            heading = el.get_text(" ", strip=True)
            skipping = heading.strip().lower() in SKIP_SECTIONS
            if not skipping and heading:
                marker = "=" * (int(el.name[1]))
                lines.append(f"\n{marker} {heading} {marker}")
            continue
        if skipping:
            continue

        if el.name in ("ul", "ol"):
            # 只取直接子 li，避免嵌套重复
            for li in el.find_all("li", recursive=False):
                t = re.sub(r"\s+", " ", li.get_text(" ", strip=True))
                if t:
                    lines.append(f"- {t}")
        elif el.name == "dl":
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if t:
                lines.append(t)
        else:  # p
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if t:
                lines.append(t)

    return "\n".join(lines).strip()


def parse_infobox_clean(soup) -> dict:
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if not infobox:
        return {}
    for junk in infobox.find_all(["sup", "style", "script"]):
        junk.decompose()
    for hidden in infobox.find_all(attrs={"style": re.compile(r"display\s*:\s*none")}):
        hidden.decompose()

    result = {}
    for row in infobox.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if not (th and td):
            continue
        key = th.get_text(" ", strip=True)
        for br in td.find_all("br"):
            br.replace_with("\x1f")
        lis = td.find_all("li")
        if lis:
            items = [li.get_text(" ", strip=True) for li in lis]
        else:
            items = td.get_text(" ", strip=True).split("\x1f")
        vals = [re.sub(r"\s+", " ", v).strip(" ;,") for v in items]
        vals = [v for v in vals if v]
        val  = "; ".join(vals)
        if key and val and len(val) < 400:
            result[key] = val
    return result


# ─────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────

def refetch(doc: dict) -> dict:
    """就地更新一篇文档；任一步失败保留旧值。"""
    title = doc["en_title"]

    soup, final_url = fetch_page_soup(title, "en")
    time.sleep(SLEEP)
    if soup:
        text_en = extract_article_text(soup)
        if text_en and len(text_en) >= len(doc.get("raw_text_en") or ""):
            doc["raw_text_en"] = text_en
            doc["source_en"]   = final_url
            doc["crawl_status"] = "success"
        infobox = parse_infobox_clean(soup)
        if infobox:
            doc["infobox"] = infobox
    elif not (doc.get("raw_text_en") or ""):
        doc["crawl_status"] = "fetch_failed"

    zh_title = doc.get("zh_title") or find_zh_title(title)
    time.sleep(SLEEP)
    if zh_title:
        doc["zh_title"] = zh_title
        zsoup, zh_url = fetch_page_soup(zh_title, "zh")
        time.sleep(SLEEP)
        if zsoup:
            text_zh = extract_article_text(zsoup)
            if text_zh and len(text_zh) >= len(doc.get("raw_text_zh") or ""):
                doc["raw_text_zh"] = text_zh
                doc["source_zh"]   = zh_url
    return doc


def main():
    docs = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    done = {}
    if TMP_PATH.exists():
        done = {d["id"]: d for d in json.loads(TMP_PATH.read_text(encoding="utf-8"))}
        print(f"[resume] checkpoint 已有 {len(done)} 篇，跳过")

    before_en = sum(len(d.get("raw_text_en") or "") for d in docs)
    out, failed = [], []
    for i, doc in enumerate(docs):
        if doc["id"] in done:
            out.append(done[doc["id"]])
            continue
        try:
            out.append(refetch(doc))
        except Exception as e:            # 单篇失败不拖垮全局，保留旧值
            print(f"  !! {doc['en_title']}: {e}")
            failed.append(doc["en_title"])
            out.append(doc)
        n_en = len(out[-1].get("raw_text_en") or "")
        print(f"[{i+1:3}/{len(docs)}] {doc['en_title'][:44]:46} EN {n_en:6}  "
              f"ZH {len(out[-1].get('raw_text_zh') or ''):6}  "
              f"ibx {len(out[-1].get('infobox') or {}):2}")
        if (i + 1) % CHECKPOINT == 0:
            TMP_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    CORPUS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    TMP_PATH.unlink(missing_ok=True)

    after_en = sum(len(d.get("raw_text_en") or "") for d in out)
    n_zh  = sum(1 for d in out if d.get("raw_text_zh"))
    n_ibx = sum(1 for d in out if d.get("infobox"))
    print("\n==== E1 summary ====")
    print(f"EN chars : {before_en:,} -> {after_en:,} ({after_en/max(before_en,1):.1f}x)")
    print(f"ZH docs  : {n_zh}/{len(out)}")
    print(f"infobox  : {n_ibx}/{len(out)}")
    if failed:
        print(f"failed   : {len(failed)} {failed}")


if __name__ == "__main__":
    main()
