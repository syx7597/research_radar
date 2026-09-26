"""
按雷达型号把手册正文切成图片：每个雷达一个文件夹，含它的页（允许重叠——
一页有两个雷达时，边界页同时进两个文件夹）。供外部识图工具批量识别。

输入:  toc_radars.json = [{"radar","page","country"}]（目录 XX1-XX5 提取）
输出:  manual_split/<NNN_slug>/p{印刷页}.png
逻辑:  按 page 排序；雷达 i 的页范围 = [page_i, page_{i+1}]（闭区间→边界页重叠），
       同国别最后一个雷达到国别结束页；单雷达上限 MAX_PAGES 页防跑飞。

  python split_by_radar.py            # 全量切分
  python split_by_radar.py --limit 5  # 只切前 5 个（验证）
"""

import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PDF = ROOT / "manuals" / "机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf"
OUT = ROOT / "pipeline" / "v3" / "manual" / "manual_split"
OFFSET = 38          # PDF 页 = 印刷页 + 38
DPI = 150
MAX_PAGES = 9        # 单雷达最多页数（防目录缺项导致范围过大）
APPENDIX_PAGE = 473  # 正文结束（附录起始印刷页）


def slug(s):
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return re.sub(r"\s+", "_", s).strip("_")[:48] or "radar"


def render_flat():
    """把正文 1-471 印刷页平铺渲染到 manual_body_pages/（不分文件夹，供批量喂图）。"""
    flat = ROOT / "pipeline" / "v3" / "manual" / "manual_body_pages"
    flat.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    n = 0
    for pp in range(1, APPENDIX_PAGE):          # 正文印刷页 1..472
        pdfp = pp + OFFSET
        if not (1 <= pdfp <= doc.page_count):
            continue
        fp = flat / f"p{pp:04d}.png"
        if not fp.exists():
            doc[pdfp - 1].get_pixmap(dpi=DPI).save(str(fp))
        n += 1
    print(f"[flat] {n} body page-images -> {flat}")


def render_range(lo, hi, outname):
    out = ROOT / "pipeline" / "v3" / "manual" / outname
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    n = 0
    for pp in range(lo, hi + 1):
        pdfp = pp + OFFSET
        if not (1 <= pdfp <= doc.page_count):
            continue
        fp = out / f"p{pp:04d}.png"
        if not fp.exists():
            doc[pdfp - 1].get_pixmap(dpi=DPI).save(str(fp))
        n += 1
    print(f"[range] {n} page-images ({lo}-{hi}) -> {out}")


def main():
    if "--flat" in sys.argv:
        render_flat()
        return
    if "--appendix-ab" in sys.argv:                # 附录A(型号索引473-493)+B(载机494-510)
        render_range(473, 510, "manual_appendix_ab_pages")
        return
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    toc = []
    for f in sorted(HERE.glob("toc_radars*.json")):
        toc.extend(json.loads(f.read_text(encoding="utf-8")))
    toc = [t for t in toc if isinstance(t.get("page"), int)]
    toc.sort(key=lambda t: t["page"])
    if limit:
        toc = toc[:limit]

    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    n_folder = n_img = 0
    for i, t in enumerate(toc):
        start = t["page"]
        nxt = toc[i + 1]["page"] if i + 1 < len(toc) else APPENDIX_PAGE
        end = min(max(nxt, start), start + MAX_PAGES)   # 闭区间到下一雷达页（重叠）
        folder = OUT / f"{start:04d}_{slug(t['radar'])}"
        folder.mkdir(exist_ok=True)
        (folder / "meta.json").write_text(
            json.dumps({"radar": t["radar"], "country": t.get("country"),
                        "printed_pages": list(range(start, end + 1))},
                       ensure_ascii=False), encoding="utf-8")
        for pp in range(start, end + 1):
            pdfp = pp + OFFSET
            if not (1 <= pdfp <= doc.page_count):
                continue
            fp = folder / f"p{pp:04d}.png"
            if not fp.exists():
                doc[pdfp - 1].get_pixmap(dpi=DPI).save(str(fp))
            n_img += 1
        n_folder += 1
    print(f"[split] {n_folder} radar folders, {n_img} page-images -> {OUT}")


if __name__ == "__main__":
    main()
