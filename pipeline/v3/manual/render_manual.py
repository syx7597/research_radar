"""
把扫描版《机载雷达手册（第4版）》按页渲染成 PNG，供对话里视觉识别。

页码换算：PDF 页 = 印刷页 + 38（印刷页1 = PDF页39）。
用印刷页码传参，脚本自动转 PDF 页。

  python render_manual.py 146 153      # 渲染印刷页 146-153（APG-66 条目）
  python render_manual.py --appendix   # 渲染附录 B/C/E（对照表）
输出到 pipeline/v3/manual/pages/p{印刷页}.png
"""

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
PDF = ROOT / "manuals" / "机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf"
OUT = ROOT / "pipeline" / "v3" / "manual" / "pages"
OFFSET = 38          # PDF页 = 印刷页 + 38
DPI = 150


def render(printed_lo: int, printed_hi: int):
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for pp in range(printed_lo, printed_hi + 1):
        pdfp = pp + OFFSET
        if not (1 <= pdfp <= doc.page_count):
            continue
        pix = doc[pdfp - 1].get_pixmap(dpi=DPI)
        fp = OUT / f"p{pp:04d}.png"
        pix.save(str(fp))
        print(f"printed p{pp} (pdf {pdfp}) -> {fp.name}  {pix.width}x{pix.height}")


def main():
    if "--appendix" in sys.argv:
        for lo, hi in [(494, 522), (523, 524)]:   # 附录B载机 / C研制方 / E频段
            render(lo, hi)
        return
    if len(sys.argv) >= 3:
        render(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 2:
        render(int(sys.argv[1]), int(sys.argv[1]))
    else:
        print("用法: render_manual.py <起印刷页> [<止印刷页>] | --appendix")


if __name__ == "__main__":
    main()
