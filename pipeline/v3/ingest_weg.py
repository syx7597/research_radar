"""
把 WEG 装备指南（data/AD1117055.pdf, TRADOC Worldwide Equipment Guide Vol.2
Airspace & Air Defense, 2011, 公开发布）的雷达/防空条目转成语料文档。

WEG 条目主体多是 SAM/防空系统（SA-6 等），雷达是内嵌的关联雷达（STRAIGHT
FLUSH 等）。因此这些文档标 source_kind=weg，S3 对 weg 只用实体侦察结果作 head
（不把系统名强当雷达），避免把导弹系统污染成雷达节点。

产出：追加写入 radar_corpus/corpus.json（id 带 __weg 后缀，幂等）。
运行：python pipeline/v3/ingest_weg.py
"""

import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent.parent
PDF = ROOT / "data" / "AD1117055.pdf"
CORPUS = ROOT / "radar_corpus" / "corpus.json"

TITLE_RX = re.compile(r"^(.{6,90}?)\s+_{3,}\s*$", re.M)
RADAR_KW = re.compile(r"radar|air defen|SAM|surveillance|missile system|"
                      r"AAA|early warning|SPAAG|MANPADS|gun/missile", re.I)
COUNTRY_RX = re.compile(
    r"^(Russian|Chinese|U\.S\.|US|British|French|German|Italian|Swedish|"
    r"European|Israeli|Iranian|Indian|Ukrainian|Czech|Polish|Dutch|"
    r"North Korean|South Korean|Japanese|Spanish)\s+", re.I)


def slugify(s):
    return re.sub(r"[^\w]+", "_", s.lower()).strip("_")[:50] or "weg"


def extract_entries():
    doc = fitz.open(PDF)
    # 记录每页起始字符偏移，用于给条目定位页码
    text, page_start = "", []
    for i in range(doc.page_count):
        page_start.append((len(text), i + 1))
        text += doc[i].get_text() + "\n"

    def page_of(pos):
        p = 1
        for off, pn in page_start:
            if off <= pos:
                p = pn
            else:
                break
        return p

    matches = list(TITLE_RX.finditer(text))
    entries = []
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not RADAR_KW.search(title):
            continue
        if len(body) < 150:
            continue
        # 型号：去掉国家/类别前缀，取尾部型号串
        model = COUNTRY_RX.sub("", title)
        model = re.sub(r"^.*\b(System|Vehicle|Radar|Launcher|Gun|SAM|MANPADS)\b\s*",
                       "", model).strip() or title
        entries.append({"title": title, "model": model[:60],
                        "page": page_of(m.start()),
                        "text": (title + "\n" + body)})
    return entries


def main():
    entries = extract_entries()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    seen = {d["id"] for d in corpus}
    added = 0
    for e in entries:
        did = "weg_" + slugify(e["model"]) + "__weg"
        if did in seen:
            continue
        corpus.append({
            "id": did, "en_title": e["model"], "zh_title": None,
            "source_en": f"AD1117055.pdf (TRADOC WEG 2011) p.{e['page']}",
            "source_zh": None, "category": "Air_defense",
            "raw_text_en": e["text"], "raw_text_zh": "",
            "infobox": {}, "known_facts": {}, "evolution_hints": [],
            "crawl_status": "success", "manually_patched": False})
        seen.add(did)
        added += 1
    CORPUS.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[WEG] radar/air-defense entries: {len(entries)}, added {added} docs "
          f"-> corpus {len(corpus)} total")


if __name__ == "__main__":
    main()
