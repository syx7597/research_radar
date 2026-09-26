"""
OCR the body of the radar handbook PDF (pages 39-510).

Skips pages already cached in data/v2/ocr_cache/page_XXXX.txt.
Reuses the singleton PaddleOCR via scripts.stage_c_extract.get_ocr().
"""
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data" / "v2" / "ocr_cache"
PDF_PATH = ROOT / "manuals" / "机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf"

BODY_START, BODY_END = 39, 510   # inclusive


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    have = set()
    for f in CACHE_DIR.glob("page_*.txt"):
        m = re.search(r"page_(\d+)\.txt", f.name)
        if m: have.add(int(m.group(1)))

    missing = [p for p in range(BODY_START, BODY_END + 1) if p not in have]
    print(f"Body pages {BODY_START}-{BODY_END}: {BODY_END - BODY_START + 1}")
    print(f"  already OCR'd: {len(have & set(range(BODY_START, BODY_END + 1)))}")
    print(f"  to OCR:        {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    import pdfplumber, numpy as np
    from paddleocr import PaddleOCR
    print("Loading PaddleOCR…")
    ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
    print("Opening PDF…")
    pdf = pdfplumber.open(PDF_PATH)
    print(f"PDF pages: {len(pdf.pages)}")

    for i, page_no in enumerate(missing):
        page_idx = page_no - 1  # 0-based
        if page_idx >= len(pdf.pages):
            print(f"  skip {page_no}: beyond PDF length")
            continue
        out = CACHE_DIR / f"page_{page_no:04d}.txt"
        try:
            img = np.array(pdf.pages[page_idx].to_image(resolution=200).original.convert("RGB"))
            result = ocr.ocr(img, cls=False)
            text = "\n".join(line[1][0] for line in result[0]) if result and result[0] else ""
            out.write_text(text, encoding="utf-8")
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(missing)}] page {page_no} done ({len(text)} chars)")
        except Exception as e:
            print(f"  ! page {page_no} failed: {e}")

    pdf.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
