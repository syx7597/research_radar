"""
PDF雷达手册三元组抽取脚本（PaddleOCR版）
支持：扫描版PDF、技术规格书、作战手册、装备年鉴（.pdf 格式）

依赖：
    pip install pymupdf paddlepaddle==2.6.2 paddleocr==2.9.1

使用方法：
    1. 把 PDF 文件放到 manuals/ 目录
    2. 设置 DEEPSEEK_API_KEY 环境变量（或在 LLM_CONFIG 里填入）
    3. python extract_from_pdf.py
    4. python extract_from_pdf.py --pages 50-150   # 只处理指定页段
    5. python extract_from_pdf.py --scan           # 扫描模式：输出OCR结果到日志，不调用LLM

结果保存到：extraction_results/pdf_results.json
"""

import json
import re
import os
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════

MANUAL_DIR  = Path("manuals")
OUTPUT_PATH = Path("extraction_results/pdf_results.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True)
MANUAL_DIR.mkdir(exist_ok=True)

LLM_CONFIG = {
    "provider":         "deepseek",
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "deepseek_model":   "deepseek-chat",
}

MAX_CHUNK_CHARS  = 2500
OVERLAP_CHARS    = 300
MIN_PAGE_CHARS   = 30    # 跳过几乎空白的页面
MAX_PAGES        = 400   # 最多处理前N页
OCR_SCALE        = 1.5   # 图像放大倍数（越大越准，越慢）
OCR_CONF_THRESH  = 0.5   # 置信度过滤阈值

# 雷达型号正则（用于从OCR文本中识别当前章节讨论的雷达）
RADAR_NAME_PATTERNS = [
    r'AN/[A-Z]{3}-\d+[A-Z]?\w*',          # AN/APG-77, AN/TPY-2(V)3
    r'[A-Z]{2,5}-\d{2,4}[A-Z]?(?:\(\w+\))?',  # APG-77, EL/M-2054, PS-05
    r'[A-Z]{1,4}/[A-Z]{1,4}-\d{1,4}[A-Z]?',   # EL/M-2054
    r'(?:APAR|ISAR|AESA|PESA|MFCR|LPAR)\b',
]
RADAR_REGEX = re.compile('|'.join(RADAR_NAME_PATTERNS))

# ══════════════════════════════════════════════════════════
#  Step 1：PaddleOCR 读取 PDF
# ══════════════════════════════════════════════════════════

_ocr_instance = None

def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        log.info("初始化 PaddleOCR（首次运行会下载模型，请稍候）...")
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='ch', show_log=True)
    return _ocr_instance


def ocr_page(page, scale: float = OCR_SCALE) -> list[tuple[str, float]]:
    """
    对单页做OCR，返回 [(文字, 置信度), ...] 列表。
    若 PaddlePaddle 抛出 'could not create a primitive'（内存/OneDNN分配失败），
    自动降低缩放比例重试一次；仍失败则返回空列表并记录警告。
    """
    import fitz
    import numpy as np

    def _run(s: float):
        mat = fitz.Matrix(s, s)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        result = get_ocr().ocr(img, cls=True)
        return [(line[1][0], line[1][1]) for line in result[0]] if result and result[0] else []

    try:
        return _run(scale)
    except RuntimeError as e:
        if "primitive" in str(e).lower() or "memory" in str(e).lower():
            fallback = max(0.8, scale - 0.5)
            log.warning(f"OCR内存错误，降低缩放至{fallback}重试: {e}")
            try:
                return _run(fallback)
            except RuntimeError as e2:
                log.error(f"OCR重试仍失败，跳过本页: {e2}")
                return []
        raise


def read_pdf_paddleocr(path: Path,
                        max_pages: int = MAX_PAGES,
                        page_range: Optional[tuple[int, int]] = None) -> dict:
    """
    用 PyMuPDF + PaddleOCR 读取扫描版 PDF。
    page_range: (start, end) 从1开始的页码，None = 全部处理。
    返回与旧接口兼容的字典。
    """
    import fitz
    doc = fitz.open(str(path))
    total_pages = len(doc)

    if page_range:
        start_idx = max(0, page_range[0] - 1)
        end_idx   = min(total_pages, page_range[1])
    else:
        start_idx, end_idx = 0, min(total_pages, max_pages)

    log.info(f"PDF 共 {total_pages} 页，处理 {start_idx+1}–{end_idx} 页")

    page_texts = []       # [(page_num, text), ...]
    page_radars = []      # [(page_num, radar_name_or_None), ...]

    for i in range(start_idx, end_idx):
        try:
            lines_conf = ocr_page(doc[i])
        except Exception as e:
            log.error(f"页{i+1} OCR完全失败，跳过: {e}")
            lines_conf = []
        lines = [t for t, c in lines_conf if c >= OCR_CONF_THRESH]
        text  = "\n".join(lines)

        if len(text.strip()) >= MIN_PAGE_CHARS:
            page_texts.append((i + 1, text))

        # 从每页检测雷达名
        radars_on_page = RADAR_REGEX.findall(text)
        radar = radars_on_page[0] if radars_on_page else None
        page_radars.append((i + 1, radar))

        if (i - start_idx) % 20 == 0:
            log.info(f"  OCR 进度: {i+1}/{end_idx}  {'[' + radar + ']' if radar else ''}")

    doc.close()
    full_text = "\n\n".join(t for _, t in page_texts)
    return {
        "source":     "paddleocr",
        "num_pages":  end_idx - start_idx,
        "paragraphs": [t for _, t in page_texts],
        "page_texts": page_texts,     # (page_num, text)
        "page_radars": page_radars,   # (page_num, radar_name)
        "tables":     [],
        "full_text":  full_text,
    }


# ══════════════════════════════════════════════════════════
#  Step 2：章节感知分块
# ══════════════════════════════════════════════════════════

def split_with_radar_context(page_texts: list[tuple[int, str]],
                              max_chars: int = MAX_CHUNK_CHARS,
                              overlap: int = OVERLAP_CHARS
                              ) -> list[dict]:
    """
    按页分块，附带当前页面追踪到的雷达名作为上下文。
    返回 [{"text": ..., "radar_name": ..., "pages": ...}, ...]
    """
    chunks = []
    current_text  = ""
    current_radar = "Unknown"
    current_pages = []

    for page_num, text in page_texts:
        # 从本页更新雷达名
        found = RADAR_REGEX.findall(text)
        if found:
            current_radar = found[0]

        if len(current_text) + len(text) + 2 > max_chars:
            if current_text.strip():
                chunks.append({
                    "text":       current_text.strip(),
                    "radar_name": current_radar,
                    "pages":      list(current_pages),
                })
            # 保留部分重叠
            current_text  = current_text[-overlap:] + "\n" + text if current_text else text
            current_pages = [page_num]
        else:
            current_text  = current_text + "\n" + text if current_text else text
            current_pages.append(page_num)

    if current_text.strip():
        chunks.append({
            "text":       current_text.strip(),
            "radar_name": current_radar,
            "pages":      list(current_pages),
        })

    return chunks


# ══════════════════════════════════════════════════════════
#  Step 3：LLM 调用
# ══════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "") -> Optional[str]:
    import requests
    key = LLM_CONFIG["deepseek_api_key"]
    if not key:
        log.error("DEEPSEEK_API_KEY 未设置")
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": LLM_CONFIG["deepseek_model"],
                  "messages": messages,
                  "max_tokens": 2000,
                  "temperature": 0.1},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"LLM 调用失败: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  Step 4：Prompt
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是雷达装备领域的知识图谱专家。
请从给定的雷达技术文本中抽取三元组，只输出 JSON 数组，不要任何其他文字。
文本可能是中文扫描手册的OCR结果，雷达型号名保留英文原文（如AN/APG-77, EL/M-2054）。"""

FEW_SHOT = """示例（文本来自中文手册OCR）：
输入："AN/APG-77 是洛克希德·马丁公司与诺斯罗普·格鲁曼公司联合研制的X波段有源相控阵火控雷达，
      美国研制，装备于F-22A战斗机，由美国空军使用。具备SAR和GMTI工作模式，
      可制导AIM-120C和AIM-9X导弹。"

输出：
[
  {"head": "AN/APG-77", "relation": "developedBy",      "tail": "洛克希德·马丁",   "head_type": "Radar", "tail_type": "Manufacturer", "confidence": 0.95, "evidence": "洛克希德·马丁公司与诺斯罗普·格鲁曼公司联合研制"},
  {"head": "AN/APG-77", "relation": "developedBy",      "tail": "诺斯罗普·格鲁曼", "head_type": "Radar", "tail_type": "Manufacturer", "confidence": 0.95, "evidence": "洛克希德·马丁公司与诺斯罗普·格鲁曼公司联合研制"},
  {"head": "AN/APG-77", "relation": "countryOfOrigin",  "tail": "United States",   "head_type": "Radar", "tail_type": "Country",      "confidence": 0.97, "evidence": "美国研制"},
  {"head": "AN/APG-77", "relation": "hasFrequencyBand", "tail": "X",               "head_type": "Radar", "tail_type": "FrequencyBand","confidence": 0.98, "evidence": "X波段"},
  {"head": "AN/APG-77", "relation": "hasFunction",      "tail": "FireControl",     "head_type": "Radar", "tail_type": "Function",     "confidence": 0.97, "evidence": "火控雷达"},
  {"head": "AN/APG-77", "relation": "deployedOn",       "tail": "F-22A",           "head_type": "Radar", "tail_type": "Platform",     "confidence": 0.98, "evidence": "装备于F-22A战斗机"},
  {"head": "AN/APG-77", "relation": "operatedBy",       "tail": "United States",   "head_type": "Radar", "tail_type": "Country",      "confidence": 0.95, "evidence": "美国空军使用"},
  {"head": "AN/APG-77", "relation": "hasMode",          "tail": "SAR",             "head_type": "Radar", "tail_type": "RadarMode",    "confidence": 0.95, "evidence": "SAR工作模式"},
  {"head": "AN/APG-77", "relation": "hasMode",          "tail": "GMTI",            "head_type": "Radar", "tail_type": "RadarMode",    "confidence": 0.95, "evidence": "GMTI工作模式"},
  {"head": "AN/APG-77", "relation": "compatibleWith",   "tail": "AIM-120C",        "head_type": "Radar", "tail_type": "Weapon",       "confidence": 0.95, "evidence": "可制导AIM-120C"},
  {"head": "AN/APG-77", "relation": "compatibleWith",   "tail": "AIM-9X",          "head_type": "Radar", "tail_type": "Weapon",       "confidence": 0.95, "evidence": "可制导AIM-9X"}
]
"""

RELATION_GUIDE = """
可用关系（只使用这些）：
- developedBy     : 研制/生产厂商（tail_type=Manufacturer）
- operatedBy      : 装备国家/军事部门（tail_type=Country）
- countryOfOrigin : 雷达的研制来源国（tail_type=Country，区别于operatedBy）
- deployedOn      : 安装平台——舰艇/飞机/车辆（tail_type=Platform）
- hasFrequencyBand: 工作频段，只写字母：X/S/L/C/Ku/Ka/UHF/VHF/P（tail_type=FrequencyBand）
- hasFunction     : 粗粒度功能——对空搜索/火控/导航/预警（tail_type=Function）
- hasMode         : 工作模式——SAR/GMTI/TWS/STT/FireControl/TerrainAvoidance/GroundMapping/Navigation/Search/LPRF/MPRF/HIPRF（tail_type=RadarMode）
- compatibleWith  : 可制导/配合使用的武器或导弹型号（tail_type=Weapon，如R-77/AIM-120/Meteor）
- affiliatedTo    : 公司/机构所属国家或集团（tail_type=Country）
- exportedTo      : 出口目的国（tail_type=Country）
- upgradeOf       : 改进自某型号（tail_type=Radar）
- derivedFrom     : 派生自某基础型（tail_type=Radar）

注意：
- 雷达型号名保留英文（AN/APG-77），制造商/国家/武器型号保留原文
- countryOfOrigin 只用于明确说明"某国研制/来源国"，和operatedBy（使用国）不同
- hasMode 和 hasFunction 不重复抽取：功能(hasFunction)=粗分类，模式(hasMode)=具体工作模式
- compatibleWith 的 tail 只写武器/导弹型号，如"R-77"，不写"使用R-77"
- 只抽取明确提到的关系，不要推断；纯数值参数不抽取
- OCR可能有识别错误，对模糊内容给低置信度（<0.7）
"""


def extract_triples_from_chunk(chunk_text: str,
                                radar_name: str,
                                source_file: str,
                                pages: list[int]) -> list[dict]:
    prompt = f"""{FEW_SHOT}

{RELATION_GUIDE}

当前章节讨论的雷达（可能有多个，以文本为准）：【{radar_name}】
页码范围：{pages[0] if pages else '?'}–{pages[-1] if pages else '?'}

请从以下文本中抽取所有雷达装备三元组。如无可抽取内容，返回 []。
只输出 JSON 数组。

文本：
{chunk_text}"""

    raw = call_llm(prompt, system=SYSTEM_PROMPT)
    if not raw:
        return []

    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '',          raw, flags=re.MULTILINE)

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
        t.setdefault("confidence", 0.8)
        t.setdefault("evidence", "")
        t["source"]      = "pdf_paddleocr"
        t["source_file"] = source_file
        t["pages"]       = pages
        valid.append(t)
    return valid


# ══════════════════════════════════════════════════════════
#  Step 4.5：结构感知词条解析（新增）
#  针对《机载雷达手册》等百科词典式PDF：
#    - 规格块（频段/厂商/机种/武器）→ 直接正则解析，confidence≈0.99
#    - ■技术特点等叙述段 → 只送LLM补充，prompt更聚焦
# ══════════════════════════════════════════════════════════

# 国家页眉（页面右上角分类标签）→ 英文实体名
COUNTRY_HEADER_RE = re.compile(
    r'(美\s*国|俄\s*罗\s*斯|英\s*国|法\s*国|德\s*国|以\s*色\s*列|瑞\s*典|'
    r'荷\s*兰|意\s*大\s*利|日\s*本|中\s*国|印\s*度|澳\s*大\s*利\s*亚|'
    r'乌\s*克\s*兰|南\s*非|加\s*拿\s*大|西\s*班\s*牙|挪\s*威|土\s*耳\s*其)'
)
COUNTRY_ZH_TO_EN = {
    "美国": "United States", "俄罗斯": "Russia",
    "英国": "United Kingdom", "法国": "France",
    "德国": "Germany", "以色列": "Israel",
    "瑞典": "Sweden", "荷兰": "Netherlands",
    "意大利": "Italy", "日本": "Japan",
    "中国": "China", "印度": "India",
    "澳大利亚": "Australia", "乌克兰": "Ukraine",
    "南非": "South Africa", "加拿大": "Canada",
    "西班牙": "Spain", "挪威": "Norway", "土耳其": "Turkey",
}

# 词条标题行：雷达型号 + 中文功能描述 + "雷达"（用于精准追踪当前词条主体）
# 允许全角括号（OCR常见混用），不要求行尾严格结束
# 例："AN/APQ-126(V)/158 导航与攻击雷达"
#     "AN/APG-70/70(V）多功能火控雷达（附AN/APQ-180/180（V)）"
ENTRY_TITLE_RE = re.compile(
    r'^([A-Z][^\n\u4e00-\u9fa5]{3,40}?)\s*[\u4e00-\u9fa5].{1,30}雷达',
    re.MULTILINE
)

# 规格块行识别（字段名内部允许OCR空格，字段名与值之间允许0-4个空格）
SPEC_LINE_RE = re.compile(
    r'^(体[\s]*制|频[\s]*段|研[\s]*制[\s]*厂[\s]*商|装[\s]*备[\s]*机[\s]*种|'
    r'配[\s]*用[\s]*武[\s]*器|研[\s]*制[\s]*时[\s]*间|装[\s]*备[\s]*时[\s]*间|'
    r'现[\s]*状|价[\s]*格)\s{0,4}([^\s].+)',
    re.MULTILINE
)

# 性能数据区的属性字段（数值/规格参数，作为实体属性保存）
PERF_ATTR_RE = re.compile(
    r'^(工[\s]*作[\s]*频[\s]*率|作[\s]*用[\s]*距[\s]*离|峰[\s]*值[\s]*功[\s]*率|'
    r'天[\s]*线[\s]*增[\s]*益|质[\s]*量|重[\s]*量|MTBF|ECCM|冷[\s]*却[\s]*方[\s]*式|'
    r'输[\s]*入[\s]*功[\s]*率|LRU)\s{0,4}([^\s].+)',
    re.MULTILINE
)

# 规格字段 → (KG关系, tail_type)  用于实体属性三元组
ATTR_FIELD_MAP = {
    "研制时间": ("researchedIn",  "Literal"),
    "装备时间": ("deployedIn",    "Literal"),
    "价格":     ("price",         "Literal"),
    "现状":     ("status",        "Literal"),
    "工作频率": ("hasFrequency",  "Literal"),
    "作用距离": ("hasRange",      "Literal"),
    "峰值功率": ("hasPeakPower",  "Literal"),
    "天线增益": ("hasAntennaGain","Literal"),
    "质量":     ("hasWeight",     "Literal"),
    "重量":     ("hasWeight",     "Literal"),
    "MTBF":     ("hasMTBF",       "Literal"),
    "ECCM":     ("hasECCM",       "Literal"),
    "冷却方式": ("hasCooling",    "Literal"),
    "输入功率": ("hasPower",      "Literal"),
    "LRU":      ("hasLRUCount",   "Literal"),
}

# 性能数据区的工作方式字段（可能在■性能数据 section 内）
PERF_MODE_RE = re.compile(
    r'^工[\s]*作[\s]*方[\s]*式\s{0,4}([^\s].+)',
    re.MULTILINE
)

# 章节标题：■开头 或 直接是常见章节名（■可能被OCR漏识别）
SECTION_HEADER_RE = re.compile(
    r'^(?:■\s*)?(?:技术特点|性能数据|研制[、，及]*试验[、，及]*装备情况|装备情况|分系统|参考文献)',
    re.MULTILINE
)


def normalize_freq_band(raw: str) -> list[str]:
    """
    去掉括号内注释，保持原文频段值，不做NATO→IEEE映射。
      'Ku(J)'         → ['Ku']
      'X/Ku/L(故我识别)' → ['X/Ku/L']
      'I/J'           → ['I/J']
      'IJ'            → ['IJ']
    """
    cleaned = re.sub(r'[\(\（][^)）]{1,30}[\)\）]', '', raw).strip()
    return [cleaned] if cleaned else [raw.strip()]


def extract_country_from_page(text: str) -> Optional[str]:
    """从页面前200字符（页眉区域）提取国家分类标签。"""
    m = COUNTRY_HEADER_RE.search(text[:200])
    if not m:
        return None
    zh = re.sub(r'\s+', '', m.group(1))
    return COUNTRY_ZH_TO_EN.get(zh)


def parse_spec_block(page_text: str, radar_name: str,
                     country: Optional[str], source_file: str,
                     page_num: int) -> list[dict]:
    """
    解析规格块键值对，直接生成高置信度三元组（不经LLM）。
    只处理■章节标题之前的区域。
    """
    section_m = SECTION_HEADER_RE.search(page_text)
    spec_region = page_text[:section_m.start()] if section_m else page_text

    triples = []

    if country:
        triples.append({
            "head": radar_name, "relation": "countryOfOrigin", "tail": country,
            "head_type": "Radar", "tail_type": "Country",
            "confidence": 0.95, "evidence": f"页眉国家分类: {country}",
            "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
        })

    for m in SPEC_LINE_RE.finditer(spec_region):
        field_raw = re.sub(r'\s+', '', m.group(1))
        value_raw = m.group(2).strip()

        if field_raw == "体制":
            # 按"，"或","拆分各技术体制词（如"脉冲波形, 单脉冲"）
            # 过滤OCR行截断碎片：纯方位/地形词（通常是多行OCR截断的片段）
            _TECH_FRAGMENT_RE = re.compile(r'^(地面|合成|高分|动目|宽带)$')
            for tech in re.split(r'[,，]', value_raw):
                tech = tech.strip()
                if len(tech) >= 3 and not _TECH_FRAGMENT_RE.match(tech):
                    triples.append({
                        "head": radar_name, "relation": "hasTechType", "tail": tech,
                        "head_type": "Radar", "tail_type": "TechType",
                        "confidence": 0.97, "evidence": f"体制: {value_raw}",
                        "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                    })

        elif field_raw == "频段":
            for band in normalize_freq_band(value_raw):
                triples.append({
                    "head": radar_name, "relation": "hasFrequencyBand", "tail": band,
                    "head_type": "Radar", "tail_type": "FrequencyBand",
                    "confidence": 0.99, "evidence": f"频段: {value_raw}",
                    "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                })

        elif field_raw == "研制厂商":
            # 去掉括号注释（如"原研制厂商 TI"），保留主要厂商
            mfr_clean = re.sub(r'[\(\（][^)）]*[\)\）]', '', value_raw).strip()
            # 只按"/"分割（如"Raytheon/Northrop"）；避免中文"与/和"误切英文公司名
            for mfr in re.split(r'(?<=[a-zA-Z])\s*/\s*(?=[A-Z])', mfr_clean):
                mfr = mfr.strip()
                if len(mfr) >= 2:
                    triples.append({
                        "head": radar_name, "relation": "developedBy", "tail": mfr,
                        "head_type": "Radar", "tail_type": "Manufacturer",
                        "confidence": 0.98, "evidence": f"研制厂商: {value_raw}",
                        "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                    })

        elif field_raw == "装备机种":
            # 处理"型号前缀: 平台1, 平台2"格式，可能多条（APQ-126: A-7D; APQ-158: MH-53J/M）
            # 先按分号或换行拆分子条目
            sub_entries = re.split(r'[;\n]', value_raw)
            for entry in sub_entries:
                # 去掉"型号前缀:"
                plat_raw = re.sub(r'^[A-Z][A-Z0-9\-\(\)/]{2,}\s*[:：]\s*', '', entry.strip())
                for plat in re.split(r'[,，、]', plat_raw):
                    # 去掉括号内国家注释（如"A-7E/7H(希腊)"→"A-7E/7H"）
                    plat = re.sub(r'[\(\（][^)）]*[\)\）]', '', plat).strip()
                    if len(plat) >= 2:
                        triples.append({
                            "head": radar_name, "relation": "deployedOn", "tail": plat,
                            "head_type": "Radar", "tail_type": "Platform",
                            "confidence": 0.97, "evidence": f"装备机种: {value_raw}",
                            "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                        })

        elif field_raw == "配用武器":
            for wpn in re.split(r'[,，、]', value_raw):
                wpn = wpn.strip()
                # 只取以大写字母开头的武器型号（如AGM-12），过滤通用词（机炮、火箭）
                wpn_id = re.match(r'([A-Z][A-Z0-9\-]+(?:/[A-Z0-9]+)*)', wpn)
                if wpn_id:
                    triples.append({
                        "head": radar_name, "relation": "compatibleWith", "tail": wpn_id.group(1),
                        "head_type": "Radar", "tail_type": "Weapon",
                        "confidence": 0.96, "evidence": f"配用武器: {value_raw}",
                        "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                    })

        elif field_raw in ("研制时间", "装备时间", "价格", "现状"):
            # 作为实体属性（Literal）保存，不做进一步解析
            relation, tail_type = ATTR_FIELD_MAP[field_raw]
            # 清理价格/时间的括号注释，保留主要值
            val = re.sub(r'[\(\（][^)）]{1,30}[\)\）]', '', value_raw).strip()
            if val:
                triples.append({
                    "head": radar_name, "relation": relation, "tail": val,
                    "head_type": "Radar", "tail_type": tail_type,
                    "confidence": 0.95, "evidence": f"{field_raw}: {value_raw}",
                    "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                })

    # ── 性能数据区：属性字段（工作频率/距离/功率等）→ Literal 属性 ──
    for m in PERF_ATTR_RE.finditer(page_text):
        field_raw = re.sub(r'\s+', '', m.group(1))
        value_raw = m.group(2).strip()
        info = ATTR_FIELD_MAP.get(field_raw)
        if not info:
            continue
        # 过滤表格列标题行：值形如"宽/cm"、"/kg"、"高/cm"（中文字+单位，无数字）
        if re.match(r'^[\u4e00-\u9fa5]{0,3}/[a-zA-Z]+$', value_raw) or value_raw.startswith('/'):
            continue
        # 过滤过短的无意义值
        if len(value_raw.strip()) < 2:
            continue
        # 去掉"空空："/"空地："类别前缀，保留主要数值
        value_clean = re.sub(r'^(空空|空地|地面|信标)[：:]\s*', '', value_raw)
        relation, tail_type = info
        triples.append({
            "head": radar_name, "relation": relation, "tail": value_clean,
            "head_type": "Radar", "tail_type": tail_type,
            "confidence": 0.95, "evidence": f"{field_raw}: {value_raw}",
            "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
        })

    # ── 性能数据区：提取工作方式 → hasMode ────────────────
    # 工作方式的值可能跨多行（OCR折行），向后多抓 3 行做拼接
    for m in PERF_MODE_RE.finditer(page_text):
        pos = m.end()
        first_line = m.group(1).strip()
        # 继续抓后续行，直到遇到新字段行或章节标题
        extra_lines = []
        for line in page_text[pos:pos + 300].splitlines():
            line = line.strip()
            if not line:
                break
            # 遇到新的字段名行则停止
            if SPEC_LINE_RE.match(line) or PERF_MODE_RE.match(line) or SECTION_HEADER_RE.match(line):
                break
            # 遇到带括号注释的说明段则停止
            if re.match(r'^[\(\（\[【\d①②③]', line):
                break
            extra_lines.append(line)
        modes_raw = first_line + '，'.join(extra_lines)
        # 去掉括号内补充说明（如"宽/窄波束两种"）
        modes_clean = re.sub(r'[\(\（][^)）]{1,30}[\)\）]', '', modes_raw)
        for mode in re.split(r'[,，、；;]', modes_clean):
            # 去掉"空空："/"空地："等分类前缀标签（保留后面的具体模式名）
            mode = re.sub(r'^(空空|空地|前视|后向)[：:]\s*', '', mode.strip()).strip()
            if 2 <= len(mode) <= 15 and not re.match(r'^\d', mode):
                triples.append({
                    "head": radar_name, "relation": "hasMode", "tail": mode,
                    "head_type": "Radar", "tail_type": "RadarMode",
                    "confidence": 0.97, "evidence": f"工作方式: {modes_raw[:80]}",
                    "source": "pdf_spec_block", "source_file": source_file, "pages": [page_num],
                })

    return triples


def extract_narrative_text(page_text: str) -> str:
    """提取■章节标题之后的叙述文本。"""
    m = SECTION_HEADER_RE.search(page_text)
    return page_text[m.start():] if m else ""


_NARRATIVE_SYSTEM = """你是雷达装备知识图谱专家。
从雷达技术文本中补充抽取规格表未涵盖的关系三元组。只输出JSON数组，不输出其他文字。"""

_NARRATIVE_USER_TPL = """雷达型号：{radar_name}
规格表已提取：{known_facts}

请从以下文本中抽取规格表未覆盖的关系，重点关注：
【技术特点段】
- upgradeOf   : 明确说"改进自/改进型"的型号（tail_type=Radar）
- derivedFrom : 明确说"派生自/基于"的型号（tail_type=Radar）
- hasMode     : 叙述中提到的具体工作模式（tail_type=RadarMode）；注意性能数据的"工作方式"字段已由规则提取，不要重复

【装备情况段】（如有"装备情况"章节）
- operatedBy  : 装备/使用该雷达的国家或军种，如"美空军"→tail="美国"（tail_type=Country）
- exportedTo  : 出口/出售给的国家，如"出售给希腊"→tail="希腊"（tail_type=Country）

规则：
- tail值用中文原文国家名（如"希腊"，不要写"Greece"）
- 不要重复规格表已有的信息
- 不确定的内容给低置信度（<0.7）

输出格式：[{{"head":"...","relation":"...","tail":"...","head_type":"Radar","tail_type":"...","confidence":0.9,"evidence":"..."}}]
无可抽取则返回 []

文本：
{narrative_text}"""


def extract_from_narrative_llm(narrative: str, radar_name: str,
                                spec_triples: list[dict],
                                source_file: str, pages: list[int]) -> list[dict]:
    """LLM补充抽取：只处理规格块未覆盖的关系，prompt更聚焦。"""
    if len(narrative.strip()) < 50:
        return []

    known = ", ".join(f"{t['relation']}={t['tail']}" for t in spec_triples[:8])
    prompt = _NARRATIVE_USER_TPL.format(
        radar_name=radar_name,
        known_facts=known or "无",
        narrative_text=narrative[:2000],
    )

    raw = call_llm(prompt, system=_NARRATIVE_SYSTEM)
    if not raw:
        return []

    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)

    try:
        triples = json.loads(raw)
        if not isinstance(triples, list):
            return []
    except json.JSONDecodeError:
        hit = re.search(r'\[.*\]', raw, re.DOTALL)
        if hit:
            try:
                triples = json.loads(hit.group())
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
        t.setdefault("confidence", 0.8)
        t.setdefault("evidence", "")
        t["source"]      = "pdf_narrative_llm"
        t["source_file"] = source_file
        t["pages"]       = pages
        valid.append(t)
    return valid


# ══════════════════════════════════════════════════════════
#  Step 5：俄罗斯雷达名归一化
#  PaddleOCR 把西里尔字母按形近拉丁字母识别，需要映射回正确英文名
# ══════════════════════════════════════════════════════════

# OCR变形 → 正确英文/NATO名称
# 规律：К→K, Ж→X/K, Щ→III/Ш, П→II, Ф→Φ/F, Б→B, Э→E, Ю→10
RUSSIAN_RADAR_NAME_MAP: dict[str, str] = {
    # Zhuk (Жук/ЖУК) 系列 — Fazotron
    "KYK":          "Zhuk",
    "XYK":          "Zhuk",
    "KYK-A":        "Zhuk-A",
    "KYK-A3":       "Zhuk-AE",
    "KYK-ME":       "Zhuk-ME",
    "KYK-M3":       "Zhuk-ME",
    "KYK-MC3":      "Zhuk-MSE",
    "XYK-MC3":      "Zhuk-MSE",
    "KYK-MΦ3":      "Zhuk-MFE",
    "XYK-MΦ3":      "Zhuk-MFE",
    "KYK-Φ":        "Zhuk-F",
    "XYK-Φ":        "Zhuk-F",
    "KYK-8-II":     "Zhuk-8-II",
    "XYK-8-II":     "Zhuk-8-II",
    "KYK-27":       "Zhuk-27",
    "XYK-27":       "Zhuk-27",
    "N010":         "N010 Zhuk",
    "N010M":        "N010M Zhuk-M",
    # RLPK (РЛПК) 系列火控综合体 — MiG-29/Su-27配套
    # OCR规律: Р→P, Л→JI/II, П→II, К→K → РЛПК→PIIIK/PJIK
    "PIIIK-27":     "RLPK-27",  "PIIIK-29":     "RLPK-29",
    "PJIIK-27":     "RLPK-27",  "PJIIK-29":     "RLPK-29",
    "PJIK-29":      "RLPK-29",  "N019":         "N019 Sapfir-29",
    "N019M":        "N019M Topaz",
    # RP / RLS (РП/РЛС) — radar sight/station designations
    "PII-35":       "RLS-35",   "PII-11":       "RP-11",
    "PI-25":        "RP-25",    "PII-25":       "RP-25",
    "PII-22":       "RP-22",    "PII-5":        "RP-5",
    # SVU
    "CBV-16":       "SVU-16",   "CBY-16":       "SVU-16",
    "CBY-16Φ":      "SVU-16F",
    # Kopyo (Копьё) 系列
    "KOIIbE":       "Kopyo",    "KOIIbE-A":     "Kopyo-A",
    "KOIIbE-21":    "Kopyo-21", "KOIIЬЕ":       "Kopyo",
    "KOПЬЕ":        "Kopyo",    "KOПЬЕ-A":      "Kopyo-A",
    # Bars (Барс) 系列 — NIIP
    "BAPС":         "Bars",     "N011":         "N011 Bars",
    "N011M":        "N011M Bars",
    # Irbis (Ирбис)
    "N035":         "N035 Irbis-E",
    # Zaslon (Заслон)
    "CACЛOH":       "Zaslon",   "3ACЛOH":       "Zaslon",
    "SACЛOH":       "Zaslon",   "3ACЛOH-M":     "Zaslon-M",
    "3АЧЛОН":       "Zaslon",
    # Luch (Луч) — Л→JI, у→y, ч→4
    "JIy4":         "Luch",     "JIY4":         "Luch",
    "JIy4-C":       "Luch-S",
    # IOR/Zhur
    "IOP-40":       "Zhur-40",  "IOP":          "Zhur",
    # Others
    "OCA":          "Osa",
    "B004":         "V004",
    "KHT-23":       "KNT-23",
}

# 简单西里尔→拉丁字符级映射（用于未收录型号的模糊匹配）
_CYR_TO_LAT = str.maketrans({
    'К': 'K', 'к': 'k', 'Ж': 'Zh', 'ж': 'zh',
    'З': 'Z', 'з': 'z', 'Б': 'B', 'б': 'b',
    'П': 'P', 'п': 'p', 'Г': 'G', 'г': 'g',
    'Д': 'D', 'д': 'd', 'Щ': 'Shch', 'щ': 'shch',
    'Ш': 'Sh', 'ш': 'sh', 'Ч': 'Ch', 'ч': 'ch',
    'Ц': 'Ts', 'ц': 'ts', 'Х': 'Kh', 'х': 'kh',
    'Ф': 'F', 'ф': 'f', 'Э': 'E', 'э': 'e',
    'Ю': 'Yu', 'ю': 'yu', 'Я': 'Ya', 'я': 'ya',
    'А': 'A', 'а': 'a', 'В': 'V', 'в': 'v',
    'Е': 'E', 'е': 'e', 'И': 'I', 'и': 'i',
    'Й': 'Y', 'й': 'y', 'Л': 'L', 'л': 'l',
    'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n',
    'О': 'O', 'о': 'o', 'Р': 'R', 'р': 'r',
    'С': 'S', 'с': 's', 'Т': 'T', 'т': 't',
    'У': 'U', 'у': 'u', 'Ы': 'Y', 'ы': 'y',
    'Ь': '', 'ь': '', 'Ъ': '', 'ъ': '',
})

_HAS_CYRILLIC = re.compile(r'[А-яЁё]')

# 非雷达实体黑名单：LLM 有时把这些误标为 Radar head
# 命中则整条三元组被过滤掉（返回 None）
NON_RADAR_ENTITIES: set[str] = {
    # 数据总线 / 接口标准
    "ARINC-429", "ARINC-664", "MIL-STD-1553", "MIL-STD-1760",
    "RS-422", "RS-485", "CAN", "AFDX",
    # 通用缩写 / 非型号词
    "TWT", "AESA", "PESA", "LPI", "PRF", "SAR", "GMTI",
    "IFF", "ECM", "ESM", "RWR",
    # 组织机构（偶尔被误抽为雷达名）
    "NATO", "USAF", "USN", "RAF",
}

def normalize_russian_radar_name(name: str) -> str:
    """
    归一化俄罗斯雷达名：
    1. 精确匹配已知映射表
    2. 如果包含西里尔字母，做字符级转写
    3. 否则原样返回
    """
    stripped = name.strip()

    # 精确映射
    if stripped in RUSSIAN_RADAR_NAME_MAP:
        return RUSSIAN_RADAR_NAME_MAP[stripped]

    # 前缀匹配（处理带版本号后缀的变体，如 "KYK-A3(V)1"）
    for ocr_form, proper in RUSSIAN_RADAR_NAME_MAP.items():
        if stripped.upper().startswith(ocr_form.upper()) and len(stripped) > len(ocr_form):
            suffix = stripped[len(ocr_form):]
            return proper + suffix

    # 含西里尔字母 → 字符级转写
    if _HAS_CYRILLIC.search(stripped):
        transliterated = stripped.translate(_CYR_TO_LAT)
        log.debug(f"  西里尔转写: {stripped!r} -> {transliterated!r}")
        return transliterated

    return stripped


def normalize_triple_names(triples: list[dict]) -> list[dict]:
    """对所有三元组的 head/tail 做俄罗斯雷达名归一化，并过滤非雷达实体。"""
    radar_types = {"Radar", "RadarSystem"}
    normalized = []
    skipped = 0
    for t in triples:
        t = dict(t)
        # 过滤：head 是非雷达实体（数据总线、接口标准、通用缩写等）
        if t.get("head", "") in NON_RADAR_ENTITIES:
            log.debug(f"  过滤黑名单实体 head={t['head']!r}: {t['head']} --[{t['relation']}]--> {t['tail']}")
            skipped += 1
            continue
        if t.get("head_type", "") in radar_types:
            orig = t["head"]
            t["head"] = normalize_russian_radar_name(orig)
            if t["head"] != orig:
                log.debug(f"  归一化 head: {orig!r} -> {t['head']!r}")
        if t.get("tail_type", "") in radar_types:
            orig = t["tail"]
            t["tail"] = normalize_russian_radar_name(orig)
            if t["tail"] != orig:
                log.debug(f"  归一化 tail: {orig!r} -> {t['tail']!r}")
        normalized.append(t)
    if skipped:
        log.info(f"  黑名单过滤：移除 {skipped} 条非雷达实体三元组")
    return normalized


# ══════════════════════════════════════════════════════════
#  Step 6：去重
# ══════════════════════════════════════════════════════════

def deduplicate(triples: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for t in triples:
        key = (t.get("head", "").lower(),
               t.get("relation", ""),
               t.get("tail", "").lower())
        if key not in best or t.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = t
    return list(best.values())


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def scan_pdf(pdf_path: Path, page_range=None):
    """纯OCR扫描模式：输出OCR文本，不调用LLM。用于调试。"""
    log.info(f"扫描模式: {pdf_path.name}")
    doc_data = read_pdf_paddleocr(pdf_path, page_range=page_range)
    for page_num, text in doc_data["page_texts"][:20]:
        print(f"\n=== Page {page_num} ===")
        safe = text[:500].encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(safe.encode(os.device_encoding(1) or "utf-8", errors="replace").decode(os.device_encoding(1) or "utf-8", errors="replace"))


CHECKPOINT_DIR = Path("extraction_results")


def _checkpoint_path(pdf_path: Path) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{pdf_path.stem}.json"


def _save_checkpoint(ckpt_path: Path, last_page: int, current_radar: Optional[str],
                     current_country: Optional[str], triples: list[dict]) -> None:
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "last_page":       last_page,
            "current_radar":   current_radar,
            "current_country": current_country,
            "triples":         triples,
        }, f, ensure_ascii=False)
    tmp.replace(ckpt_path)


def _load_checkpoint(ckpt_path: Path) -> Optional[dict]:
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"加载断点失败，将从头开始: {e}")
        return None


def process_pdf(pdf_path: Path, page_range=None) -> dict:
    """
    结构感知版PDF处理管线（支持断点续跑）：
      1. 规格块（频段/厂商/机种/武器）→ 正则直接解析，confidence≈0.97-0.99
      2. ■叙述段（技术特点等）→ 聚焦LLM补充 operatedBy/upgradeOf/hasMode 等
      3. 每页处理完后保存断点，中断后可从上次位置继续
    """
    import fitz

    log.info(f"处理 PDF（结构感知模式）: {pdf_path.name}")
    ckpt_path = _checkpoint_path(pdf_path)

    # 尝试加载断点
    ckpt = _load_checkpoint(ckpt_path)
    if ckpt:
        last_completed = ckpt["last_page"]
        current_radar   = ckpt["current_radar"]
        current_country = ckpt["current_country"]
        all_triples     = ckpt["triples"]
        log.info(f"  发现断点：已完成至第 {last_completed} 页，加载 {len(all_triples)} 条三元组，继续...")
    else:
        last_completed  = 0
        current_radar   = None
        current_country = None
        all_triples     = []

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if page_range:
        start_idx = max(0, page_range[0] - 1)
        end_idx   = min(total_pages, page_range[1])
    else:
        start_idx, end_idx = 0, min(total_pages, MAX_PAGES)

    log.info(f"PDF 共 {total_pages} 页，处理 {start_idx+1}–{end_idx} 页")

    processed_any = False
    for i in range(start_idx, end_idx):
        page_num = i + 1

        # 跳过已完成的页
        if page_num <= last_completed:
            continue

        # OCR 当前页
        try:
            lines_conf = ocr_page(doc[i])
        except Exception as e:
            log.error(f"页{page_num} OCR完全失败，跳过: {e}")
            lines_conf = []

        lines = [t for t, c in lines_conf if c >= OCR_CONF_THRESH]
        text  = "\n".join(lines)

        if (i - start_idx) % 20 == 0:
            log.info(f"  OCR 进度: {page_num}/{end_idx}")

        if len(text.strip()) < MIN_PAGE_CHARS:
            _save_checkpoint(ckpt_path, page_num, current_radar, current_country, all_triples)
            continue

        # 更新国家上下文
        page_country = extract_country_from_page(text)
        if page_country:
            current_country = page_country

        # 优先用词条标题行更新当前雷达
        title_m = ENTRY_TITLE_RE.search(text)
        if title_m:
            raw_name = title_m.group(1).strip()
            current_radar = raw_name.replace('（', '(').replace('）', ')')
            log.info(f"  词条标题检测: [{current_radar}] (页{page_num})")
        elif current_radar is None:
            found = RADAR_REGEX.findall(text)
            if found:
                current_radar = found[0]

        if current_radar:
            # ── 规格块：正则直接解析 ──────────────────────────────
            spec_triples = parse_spec_block(
                text, current_radar, current_country, pdf_path.name, page_num
            )

            # ── 叙述段：聚焦LLM补充 ──────────────────────────────
            narrative = extract_narrative_text(text)
            narr_triples = extract_from_narrative_llm(
                narrative, current_radar, spec_triples, pdf_path.name, [page_num]
            )

            page_total = len(spec_triples) + len(narr_triples)
            if page_total:
                log.info(f"  页{page_num:3d} [{current_radar}]: "
                         f"规格块{len(spec_triples)}条 + 叙述LLM{len(narr_triples)}条")

            all_triples.extend(spec_triples)
            all_triples.extend(narr_triples)
            time.sleep(0.2)

        # 每页完成后保存断点
        _save_checkpoint(ckpt_path, page_num, current_radar, current_country, all_triples)
        processed_any = True

    doc.close()

    if not all_triples:
        log.warning("未抽取到任何三元组（OCR失败或全图页面）")
        return {"file": pdf_path.name, "triple_count": 0, "triples": [], "error": "empty_text"}

    normalized = normalize_triple_names(all_triples)
    deduped    = deduplicate(normalized)

    rel_dist   = Counter(t["relation"] for t in deduped)
    radar_dist = Counter(t.get("head", "?") for t in deduped)
    src_dist   = Counter(t.get("source", "?") for t in deduped)

    log.info(f"去重后: {len(deduped)} 条三元组，涉及 {len(radar_dist)} 个雷达型号")
    for rel, cnt in rel_dist.most_common():
        log.info(f"  {rel}: {cnt}")
    log.info(f"来源分布: { {k: v for k, v in src_dist.items()} }")

    # 全部完成后删除断点文件
    if ckpt_path.exists():
        ckpt_path.unlink()
        log.info(f"  断点文件已清除: {ckpt_path.name}")

    return {
        "file":          pdf_path.name,
        "triple_count":  len(deduped),
        "radar_count":   len(radar_dist),
        "triples":       deduped,
        "relation_dist": dict(rel_dist),
        "source_dist":   dict(src_dist),
    }


def run_pdf_extraction(file: Optional[str] = None,
                       page_range: Optional[tuple[int,int]] = None,
                       scan_only: bool = False):
    if file:
        paths = [MANUAL_DIR / file] if (MANUAL_DIR / file).exists() else [Path(file)]
    else:
        paths = sorted(MANUAL_DIR.glob("*.pdf"))

    if not paths:
        log.warning(f"{MANUAL_DIR}/ 下没有 PDF 文件")
        return []

    log.info(f"找到 {len(paths)} 个 PDF")
    results = []

    for path in paths:
        try:
            if scan_only:
                scan_pdf(path, page_range=page_range)
            else:
                r = process_pdf(path, page_range=page_range)
                results.append(r)
        except Exception as e:
            log.error(f"处理 {path.name} 失败: {e}", exc_info=True)

    if not scan_only and results:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        total = sum(r.get("triple_count", 0) for r in results)
        log.info(f"\n完成: {len(results)} 个文件, {total} 条三元组 -> {OUTPUT_PATH}")

    return results


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF雷达手册三元组抽取（PaddleOCR版）")
    parser.add_argument("--file",  type=str, default=None,
                        help="只处理指定文件（如 --file AN_TPY2.pdf）")
    parser.add_argument("--pages", type=str, default=None,
                        help="页码范围，如 --pages 50-150（从1开始）")
    parser.add_argument("--scan",  action="store_true",
                        help="只输出OCR文本，不调用LLM")
    args = parser.parse_args()

    page_range = None
    if args.pages:
        m = re.match(r'(\d+)-(\d+)', args.pages)
        if m:
            page_range = (int(m.group(1)), int(m.group(2)))
            log.info(f"只处理第 {page_range[0]}–{page_range[1]} 页")

    run_pdf_extraction(file=args.file, page_range=page_range, scan_only=args.scan)
    print("\n下一步：")
    print("  python merge_sources.py   # 合并所有来源")
    print("  python build_index.py     # 重建索引")
