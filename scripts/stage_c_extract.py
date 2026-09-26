"""
v2 Stage C: PDF narrative re-extraction targeting v2 schema relations.

Pipeline:
  1. Build radar → pages map from existing extraction_results/pdf_results.json
  2. Pick top N narrative-rich radars
  3. OCR their pages (cache to data/v2/ocr_cache/)
  4. Run focused LLM extraction targeting:
       - hasSubsystem  (transmitter, receiver, antenna, signal_processor, cooling, power_supply)
       - hasComponent  (TWT, MMIC, T/R module, magnetron, klystron, phase shifter, ...)
       - meetsStandard (MIL-STD-1553, ARINC-429, ARINC-664, AFDX, ...)
       - usedIn        (Conflict entities — wars, deployments to operations)
  5. Validate, dedupe, append to v2 KG

Resume-friendly: OCR cache survives restarts; LLM responses cached to JSON.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data" / "v2" / "ocr_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE_PATH = ROOT / "data" / "v2" / "stage_c_llm_cache.json"

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Knobs (override via CLI: --max-radars N, --all)
import argparse
TARGET_N_RADARS = 120          # default pilot scope; overridden by CLI
PDF_PATH = ROOT / "manuals" / "机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf"


def _parse_cli():
    p = argparse.ArgumentParser(description="Stage C v2-schema extraction from PDF")
    p.add_argument("--max-radars", type=int, default=TARGET_N_RADARS,
                   help="Max radars to process (default 120)")
    p.add_argument("--all", action="store_true",
                   help="Process ALL radars with narrative content (overrides --max-radars)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────
#  OCR with disk cache
# ────────────────────────────────────────────────────────────

_OCR = None

def get_ocr():
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
    return _OCR


def ocr_page(page_idx: int, pdf, dpi: int = 200) -> str:
    """OCR one page with disk cache. page_idx is 0-based."""
    cache_file = CACHE_DIR / f"page_{page_idx+1:04d}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    import numpy as np
    img = np.array(pdf.pages[page_idx].to_image(resolution=dpi).original.convert("RGB"))
    result = get_ocr().ocr(img, cls=False)
    text = "\n".join(line[1][0] for line in result[0]) if result and result[0] else ""
    cache_file.write_text(text, encoding="utf-8")
    return text


# ────────────────────────────────────────────────────────────
#  LLM call with response cache
# ────────────────────────────────────────────────────────────

def _load_llm_cache() -> dict:
    if LLM_CACHE_PATH.exists():
        with open(LLM_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_llm_cache(cache: dict):
    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


_LLM_CACHE = _load_llm_cache()


def call_llm(messages, max_tokens: int = 800, temperature: float = 0.0) -> str:
    """DeepSeek call with cache keyed by prompt content."""
    import hashlib, requests
    key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    payload = {"model": "deepseek-chat", "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            _LLM_CACHE[key] = txt
            if len(_LLM_CACHE) % 10 == 0:
                _save_llm_cache(_LLM_CACHE)
            return txt
        except Exception as e:
            if attempt == 2:
                return f"[LLM_ERROR: {e}]"
            time.sleep(1.5 * (attempt + 1))
    return ""


# ────────────────────────────────────────────────────────────
#  Extraction prompt — v2 schema specific
# ────────────────────────────────────────────────────────────

EXTRACTOR_SYSTEM = """你是雷达装备知识图谱专家，从雷达条目文本中抽取四类 v2 schema 的关系：

1. hasSubsystem  : 雷达包含的功能子系统
   有效 tail 值（标准化后）：transmitter / receiver / antenna / signal_processor / display_control / power_supply / cooling
   触发线索：发射机/接收机/天线/信号处理机/显控/电源/冷却

2. hasComponent  : 子系统/雷达内部的物理部件
   触发线索：行波管(TWT)/磁控管/速调管/MMIC/T/R 模块/移相器/巴克码相位调制器/嵌入式热传感器
   tail 用通用化的部件名称英文（TWT/MMIC/T/R-module/magnetron/klystron/phase-shifter/...)；如果是中文常用术语保留中文

3. meetsStandard : 雷达/子系统遵循的接口或标准
   触发线索：MIL-STD-XXXX / ARINC-XXX / RS-XXX / AFDX / 1553 / 1760
   tail 用标准号（MIL-STD-1553 / ARINC-429 等）

4. usedIn        : 雷达在哪场战争/冲突中实战使用
   触发线索：海湾战争 / 越战 / 伊拉克战争 / 阿富汗战争 / 利比亚 / 福克兰群岛 / 马岛战争 / Gulf War / Vietnam War / ...
   tail 用战争名称（"海湾战争 1991" / "Vietnam War" 等）；只在文本明确说"实战使用/参战/用于X战争"时抽取，不要把研发时期当 usedIn

输出严格 JSON 数组：
[{"relation":"...","tail":"...","confidence":0.0-1.0,"evidence":"原文片段（30 字内）"}, ...]
不要输出其他文字。如无可抽取返回 []
"""

USER_TPL = """雷达型号：{radar}

文本：
{text}

请抽取 hasSubsystem / hasComponent / meetsStandard / usedIn 四类。"""

VALID_RELATIONS = {"hasSubsystem", "hasComponent", "meetsStandard", "usedIn"}
VALID_SUBSYSTEMS = {"transmitter", "receiver", "antenna", "signal_processor",
                     "display_control", "power_supply", "cooling"}


def llm_extract(radar: str, text: str) -> list:
    """Run LLM extraction. Returns list of {relation, tail, confidence, evidence} dicts."""
    if len(text.strip()) < 100:
        return []
    text = text[:3500]   # cap
    raw = call_llm(
        [{"role": "system", "content": EXTRACTOR_SYSTEM},
         {"role": "user",   "content": USER_TPL.format(radar=radar, text=text)}],
        max_tokens=900,
    )
    if raw.startswith("[LLM_ERROR"):
        return []
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        rows = json.loads(m.group(0)) if m else json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    valid = []
    for r in rows:
        if not isinstance(r, dict): continue
        rel = r.get("relation", ""); tail = r.get("tail", "").strip()
        if rel not in VALID_RELATIONS or not tail: continue
        if rel == "hasSubsystem" and tail not in VALID_SUBSYSTEMS:
            # try to normalize
            tl = tail.lower()
            for v in VALID_SUBSYSTEMS:
                if v in tl or v.split("_")[0] in tl:
                    tail = v; break
            else:
                continue
        valid.append({
            "relation":   rel,
            "tail":       tail,
            "confidence": float(r.get("confidence", 0.7)),
            "evidence":   r.get("evidence", "")[:200],
        })
    return valid


# ────────────────────────────────────────────────────────────
#  Driver
# ────────────────────────────────────────────────────────────

def main():
    args = _parse_cli()
    target_n = 10**9 if args.all else args.max_radars
    print(f"[config] target_n_radars = {target_n if target_n < 10**9 else 'ALL'}")
    print(f"Loading existing PDF extraction…")
    with open(ROOT / "extraction_results" / "pdf_results.json", encoding="utf-8") as f:
        pdf_results = json.load(f)[0]

    # Build radar → pages map (only narrative-extracted triples)
    radar_pages: dict[str, set[int]] = defaultdict(set)
    radar_narrative_n: dict[str, int] = defaultdict(int)
    for t in pdf_results["triples"]:
        if t.get("source") != "pdf_narrative_llm": continue
        for p in t.get("pages", []):
            radar_pages[t["head"]].add(p)
        radar_narrative_n[t["head"]] += 1

    # Score and pick top N radars by narrative content
    ranked = sorted(radar_pages.keys(),
                    key=lambda r: -(radar_narrative_n[r] * 10 + len(radar_pages[r])))
    target_radars = ranked[:target_n]
    pages_to_ocr = sorted({p for r in target_radars for p in radar_pages[r]})
    print(f"Target: {len(target_radars)} radars, {len(pages_to_ocr)} pages to OCR")

    # Open PDF once for OCR
    print("Opening PDF…")
    import pdfplumber
    pdf = pdfplumber.open(PDF_PATH)
    print(f"  {len(pdf.pages)} pages total")

    # OCR all required pages (cached)
    print(f"\nOCR phase ({len(pages_to_ocr)} pages, ~{len(pages_to_ocr)*6/60:.1f} min if cold cache)…")
    page_texts: dict[int, str] = {}
    t0 = time.time()
    for i, page_num in enumerate(pages_to_ocr, 1):
        page_texts[page_num] = ocr_page(page_num - 1, pdf)
        if i % 20 == 0 or i == len(pages_to_ocr):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  OCR {i:>4d}/{len(pages_to_ocr)}  elapsed={elapsed:.0f}s  rate={rate:.1f}/s")
    pdf.close()

    # LLM extraction per radar
    print(f"\nLLM extraction phase…")
    all_extractions: list[dict] = []
    radar_yield = Counter()
    t0 = time.time()
    for i, radar in enumerate(target_radars, 1):
        page_set = sorted(radar_pages[radar])
        if not page_set: continue
        # join texts of those pages
        joined = "\n\n".join(page_texts.get(p, "") for p in page_set)
        if len(joined.strip()) < 200: continue
        rows = llm_extract(radar, joined)
        if rows:
            radar_yield[radar] = len(rows)
            for r in rows:
                all_extractions.append({
                    "head": radar, "head_type": "Radar",
                    "relation":   r["relation"],
                    "tail":       r["tail"],
                    "tail_type":  {"hasSubsystem":"Subsystem","hasComponent":"Component",
                                    "meetsStandard":"Standard","usedIn":"Conflict"}[r["relation"]],
                    "confidence": r["confidence"],
                    "evidence":   r["evidence"],
                    "source":     "stage_c_llm",
                    "source_pages": page_set,
                })
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  LLM {i:>4d}/{len(target_radars)}  yield_so_far={len(all_extractions)}  elapsed={elapsed:.0f}s")
    _save_llm_cache(_LLM_CACHE)

    # Save raw extractions
    out_raw = ROOT / "data" / "v2" / "stage_c_raw_extractions.json"
    with open(out_raw, "w", encoding="utf-8") as f:
        json.dump({"count": len(all_extractions), "extractions": all_extractions},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_raw}")
    print(f"Total raw extractions: {len(all_extractions)}")
    rel_counts = Counter(e["relation"] for e in all_extractions)
    for r, c in rel_counts.most_common():
        print(f"  {r:18s} {c}")


if __name__ == "__main__":
    main()
