"""
《世界海用雷达手册》自动视觉转录 —— Qwen3-VL Plus (dashscope OpenAI 兼容模式)。

**多页滑动窗口**转录：一次调用喂连续 WINDOW 页、窗口间重叠 OVERLAP 页，
让 Qwen-VL 在窗口内把跨页雷达整合完整（一页一调用会把跨页雷达切成两半）。
每窗口存 data/v3/naval_manual/p{start}-{end}.md；解析时按雷达型号(### 标题)
合并重叠页的重复雷达。断点续跑、限速、重试。

页码(PDF 页): 15-32 目录 / 33-642 正文 / 643-727 附录。默认只转正文。

  python naval_transcribe.py                 # 正文 33-642
  python naval_transcribe.py --pages 33 44   # 指定范围(先小范围验证)
  python naval_transcribe.py --appendix      # 附录 643-727
  python naval_transcribe.py --window 4 --overlap 1
key/base_url 从 apikey.txt 的 dashscope 组读取；模型默认 qwen3-vl-plus(可 --model 改)。
"""

import argparse
import base64
import re
import sys
import time
from pathlib import Path

import fitz
import requests

ROOT = Path(__file__).resolve().parents[3]
PDF = ROOT / "manuals" / "世界海用雷达手册.pdf"
OUT = ROOT / "data" / "v3" / "naval_manual"
DPI = 150
SLEEP = 1.0
# 模型链：欠费(Arrearage)/不可用 → 自动切下一个(用户指定顺序)
MODEL_CHAIN = ["qwen3.6-plus", "qwen3-vl-plus", "qwen-vl-ocr",
               "qwen3.5-ocr", "qwen-vl-plus"]
_cur_model = [0]        # 当前模型索引(可变,跨窗口保持)

PROMPT = """你是一名严谨的文档转录员。下面给你的是一本中文《世界海用雷达手册》的**连续多页**扫描图片（按顺序排列）。
请把这些页面的全部内容一字不漏地转录成结构化 Markdown。你的唯一任务是忠实转录，尽可能提取识别出来每个雷达的全部信息。输出时，按照每个雷达型号来输出，其中参考文献项不需要输出。
**关键**：如果一个雷达的内容跨越了两页（在一页底部开始、下一页顶部继续），请把它**整合成一个完整的雷达条目**，不要拆成两条。按型号用 `### 型号名` 分节。
- 双栏页面先转左栏规格块再转右栏叙述，不要左右交错。
- 规格块用键值列表（体制/频段/研制厂商/装备平台/现状等，以页面实际出现为准）。
- 表格转成 Markdown 表格，保留原有行列。
- 型号、数字、单位、厂商名照原样，不翻译、不推断、不补全。
- 只输出转录内容，不要解释或总结。若这些页面均无雷达内容，输出：（本批无雷达条目）"""


def load_dashscope():
    """从 apikey.txt 取 base_url 含 dashscope 的组 (base_url + apikey/api_key)。"""
    groups, cur = [], {}
    for ln in (ROOT / "apikey.txt").read_text(encoding="utf-8").splitlines():
        mu = re.match(r'\s*"?base_url"?\s*[=:]\s*"?([^\s",]+)', ln)
        if mu:
            if cur:
                groups.append(cur)
            cur = {"base_url": mu.group(1)}
            continue
        mk = re.match(r'\s*"?api_?key"?\s*[=:]\s*"?(sk-[A-Za-z0-9._\-]+)', ln)
        if mk and cur:
            cur["key"] = mk.group(1)
    if cur:
        groups.append(cur)
    # 阿里云百炼 workspace key(sk-ws-)走公共 dashscope 兼容 endpoint；
    # apikey.txt 里的专属 maas endpoint 公网 404，忽略之。
    PUBLIC = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    for g in groups:
        bu = g.get("base_url", "")
        if g.get("key") and ("aliyuncs" in bu or "dashscope" in bu
                             or g["key"].startswith("sk-ws")):
            return PUBLIC, g["key"]
    raise RuntimeError("apikey.txt 未找到阿里云 base_url + key 组")


def page_to_datauri(doc, pdf_page):
    pix = doc[pdf_page - 1].get_pixmap(dpi=DPI)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    return f"data:image/png;base64,{b64}"


def transcribe(url, key, datauris):
    """按模型链发请求。欠费/模型不可用 → 切下一个模型；网络错误 → 重试当前模型。"""
    content = [{"type": "image_url", "image_url": {"url": u}} for u in datauris]
    content.append({"type": "text", "text": PROMPT})
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    while _cur_model[0] < len(MODEL_CHAIN):
        model = MODEL_CHAIN[_cur_model[0]]
        payload = {"model": model, "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": 8000}
        for attempt in range(3):
            try:
                # 阿里云国内 endpoint 直连，绕过系统代理(代理连不上 cn-beijing.maas)
                r = requests.post(url, headers=headers, json=payload, timeout=300,
                                  proxies={"http": None, "https": None})
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"], model
                body = r.text
                if "Arrearage" in body or r.status_code in (401, 403):
                    print(f"    [{model}] 欠费/拒绝 → 切下一个模型", flush=True)
                    break                       # 换模型
                if r.status_code == 400:        # 模型不支持该请求 → 换模型
                    print(f"    [{model}] 400: {body[:120]} → 切下一个模型", flush=True)
                    break
                r.raise_for_status()
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"    [{model}] 网络失败3次: {e}", flush=True)
                    break
                time.sleep(3 * (attempt + 1))
        _cur_model[0] += 1                      # 当前模型不行，切下一个
    raise RuntimeError("模型链全部不可用(全欠费或不支持)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--window", type=int, default=4, help="一次调用页数")
    ap.add_argument("--overlap", type=int, default=1, help="窗口重叠页数")
    ap.add_argument("--render-only", nargs=2, type=int, metavar=("LO", "HI"),
                    help="只渲染页图到 naval_appendix_pages/(供其它视觉模型),不调API")
    args = ap.parse_args()

    if args.render_only:
        rlo, rhi = args.render_only
        rout = ROOT / "data" / "v3" / "naval_appendix_pages"
        rout.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(PDF)
        for pp in range(rlo, rhi + 1):
            fp = rout / f"p{pp:04d}.png"
            if not fp.exists():
                doc[pp - 1].get_pixmap(dpi=DPI).save(str(fp))
        print(f"[naval] 渲染 {rhi-rlo+1} 页 ({rlo}-{rhi}) -> {rout}")
        return

    lo, hi = (643, 727) if args.appendix else (args.pages or (33, 642))
    step = max(args.window - args.overlap, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    url, key = load_dashscope()
    doc = fitz.open(PDF)
    print(f"[naval] 转录 PDF 页 {lo}-{hi}  window={args.window} overlap={args.overlap} "
          f"模型链 {MODEL_CHAIN}", flush=True)

    done = n = 0
    start, prev_end = lo, lo - 1
    while start <= hi:
        end = min(start + args.window - 1, hi)
        if end <= prev_end:          # 末尾窗口无新页 → 停(不产生冗余单页窗口)
            break
        prev_end = end
        fp = OUT / f"p{start:04d}-{end:04d}.md"
        if fp.exists() and fp.stat().st_size > 0:
            done += 1
            start += step
            continue
        try:
            uris = [page_to_datauri(doc, pp) for pp in range(start, end + 1)]
            md, used_model = transcribe(url, key, uris)
        except Exception as e:
            print(f"  !! p{start}-{end}: {e}", flush=True)
            break                               # 模型链全欠费 → 停(可续跑)
        fp.write_text(f"<!-- pages {start}-{end} model={used_model} -->\n\n{md}",
                      encoding="utf-8")
        n += 1
        print(f"  p{start}-{end}: window {n} done [{used_model}] ({done} pre-existing)",
              flush=True)
        start += step
        time.sleep(SLEEP)
    print(f"[naval] 完成: 本次 {n} 窗口, 已有 {done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
