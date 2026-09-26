"""
雷达手册（Word 文档）三元组抽取脚本
支持 .docx 格式，自动处理正文段落、表格、章节标题

运行前安装: pip install python-docx

使用方法:
  1. 把 .docx 手册文件放到 manuals/ 目录
  2. 修改下方 LLM_CONFIG 填入 API Key
  3. python extract_from_manual.py

结果保存到: extraction_results/manual_results.json
可直接与 method_c_results.json 合并后重建索引
"""

import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════

MANUAL_DIR  = Path("manuals")          # 存放 .docx 文件的目录
OUTPUT_PATH = Path("extraction_results/manual_results.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True)
MANUAL_DIR.mkdir(exist_ok=True)

LLM_CONFIG = {
    "provider":         "deepseek",
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", "填入你的key"),
    "deepseek_model":   "deepseek-chat",
}

# 每次送给 LLM 的最大字符数（手册段落可能很长，需要分块）
MAX_CHUNK_CHARS = 2000
# 块与块之间的重叠字符数（避免跨块实体被截断）
OVERLAP_CHARS   = 200


# ══════════════════════════════════════════════════════════
#  Step 1：读取 Word 文档
# ══════════════════════════════════════════════════════════

def read_docx(path: Path) -> dict:
    """
    读取 .docx 文件，返回结构化内容：
      title       文件名（作为雷达ID的基础）
      paragraphs  正文段落列表
      tables      表格列表（每张表是二维列表）
      headings    标题列表（带层级）
    """
    from docx import Document
    doc = Document(str(path))

    paragraphs = []
    headings   = []
    tables_text = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name.lower()
        if "heading" in style:
            # 提取标题层级（Heading 1/2/3）
            level = 1
            for i in range(1, 5):
                if str(i) in style:
                    level = i
                    break
            headings.append({"level": level, "text": text})
            paragraphs.append(f"【标题{level}】{text}")
        else:
            paragraphs.append(text)

    # 处理表格——转成文字描述
    for table_idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            table_text = f"【表格{table_idx+1}】\n" + "\n".join(rows)
            tables_text.append(table_text)
            paragraphs.append(table_text)

    return {
        "title":      path.stem,      # 文件名去掉扩展名作为标识
        "file_path":  str(path),
        "paragraphs": paragraphs,
        "headings":   headings,
        "tables":     tables_text,
        "full_text":  "\n".join(paragraphs),
    }


# ══════════════════════════════════════════════════════════
#  Step 2：文本分块
# ══════════════════════════════════════════════════════════

def split_into_chunks(text: str,
                      max_chars: int = MAX_CHUNK_CHARS,
                      overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    按段落边界分块，保留重叠避免实体被截断。
    手册文本通常是中文，按段落换行分割。
    """
    # 按双换行或章节标题分段
    paragraphs = re.split(r'\n{2,}|(?=【标题[1-4]】)', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
                # 保留尾部作为下一块的重叠
                current = current[-overlap:] + "\n" + para
            else:
                # 单段落超长，强制截断
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars])
                current = ""
        else:
            current = current + "\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ══════════════════════════════════════════════════════════
#  Step 3：LLM 调用（复用 triple_extraction_v2.py 的逻辑）
# ══════════════════════════════════════════════════════════

def call_llm(prompt: str, system: str = "") -> Optional[str]:
    import requests
    if LLM_CONFIG["provider"] == "deepseek":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {LLM_CONFIG['deepseek_api_key']}",
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
    return None


# ══════════════════════════════════════════════════════════
#  Step 4：三元组抽取 Prompt（中文手册专用）
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是雷达装备领域的知识图谱专家。
请从给定的雷达技术手册文本中抽取三元组，只输出 JSON 数组，不要任何其他文字。"""

FEW_SHOT_EXAMPLES = """
示例输入：
"某型雷达由中国电子科技集团公司第十四研究所研制，工作在S波段，装备于052D型驱逐舰，于2014年入役。该雷达是346A型雷达的改进版本，具备对空、对海搜索功能。"

示例输出：
[
  {"head": "某型雷达", "head_type": "Radar", "relation": "developedBy", "tail": "中国电子科技集团公司第十四研究所", "tail_type": "Company", "confidence": 0.98, "evidence": "由中国电子科技集团公司第十四研究所研制"},
  {"head": "某型雷达", "head_type": "Radar", "relation": "hasFrequencyBand", "tail": "S", "tail_type": "FrequencyBand", "confidence": 0.98, "evidence": "工作在S波段"},
  {"head": "某型雷达", "head_type": "Radar", "relation": "deployedOn", "tail": "052D型驱逐舰", "tail_type": "Platform", "confidence": 0.98, "evidence": "装备于052D型驱逐舰"},
  {"head": "某型雷达", "head_type": "Radar", "relation": "upgradeOf", "tail": "346A型雷达", "tail_type": "Radar", "confidence": 0.95, "evidence": "是346A型雷达的改进版本"},
  {"head": "某型雷达", "head_type": "Radar", "relation": "hasFunction", "tail": "对空搜索", "tail_type": "Function", "confidence": 0.90, "evidence": "具备对空、对海搜索功能"},
  {"head": "某型雷达", "head_type": "Radar", "relation": "hasFunction", "tail": "对海搜索", "tail_type": "Function", "confidence": 0.90, "evidence": "具备对空、对海搜索功能"}
]
"""

RELATION_GUIDE = """
可用关系类型（只使用这些，不要自创）：
- developedBy    : 雷达由某公司/机构研制或生产
- operatedBy     : 雷达装备于某国军队/部门
- deployedOn     : 雷达安装在某平台（舰艇/飞机/车辆/阵地）
- hasFrequencyBand: 雷达的工作频段（值用字母：L/S/C/X/Ku/Ka/UHF/VHF/P）
- hasFunction    : 雷达具备的功能（对空搜索/火控/导航/预警等）
- affiliatedTo   : 机构/公司隶属于某个国家
- exportedTo     : 雷达出口到某国
- upgradeOf      : 是某型雷达的改进/升级版本
- derivedFrom    : 从某型雷达派生出来（注意：tail是原型，head是新型号）
- competitorOf   : 竞争对手型号
"""


def extract_triples_from_chunk(chunk_text: str,
                                radar_name: str,
                                source_file: str) -> list[dict]:
    """对单个文本块进行三元组抽取。"""
    user_prompt = f"""{FEW_SHOT_EXAMPLES}

{RELATION_GUIDE}

现在请从以下文本中抽取与【{radar_name}】相关的三元组。
如果文本中没有可抽取的关系，返回空数组 []。
只输出 JSON 数组，不要任何其他内容。

文本：
{chunk_text}"""

    raw = call_llm(user_prompt, system=SYSTEM_PROMPT)
    if not raw:
        return []

    # 清理并解析 JSON
    raw = raw.strip()
    # 去掉可能的代码块标记
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

    try:
        triples = json.loads(raw)
        if not isinstance(triples, list):
            return []
    except json.JSONDecodeError:
        # 尝试提取 JSON 数组
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                triples = json.loads(match.group())
            except:
                return []
        else:
            return []

    # 验证和补全字段
    valid = []
    for t in triples:
        if not all(k in t for k in ("head", "relation", "tail")):
            continue
        t.setdefault("head_type", "Radar")
        t.setdefault("tail_type", "Entity")
        t.setdefault("confidence", 0.8)
        t.setdefault("evidence", "")
        t["source"] = "manual_few_shot"
        t["source_file"] = source_file
        valid.append(t)

    return valid


# ══════════════════════════════════════════════════════════
#  Step 5：去重合并
# ══════════════════════════════════════════════════════════

def deduplicate(triples: list[dict]) -> list[dict]:
    """去除完全重复的三元组，保留置信度最高的。"""
    best: dict[tuple, dict] = {}
    for t in triples:
        key = (t.get("head",""), t.get("relation",""), t.get("tail",""))
        if key not in best or t.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = t
    return list(best.values())


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def process_manual(docx_path: Path) -> dict:
    """处理单个 Word 手册，返回抽取结果。"""
    log.info(f"处理: {docx_path.name}")

    # 读取文档
    doc = read_docx(docx_path)
    radar_name = doc["title"]
    full_text  = doc["full_text"]

    log.info(f"  文档长度: {len(full_text)} 字符")

    # 尝试从标题自动识别雷达名称
    if doc["headings"]:
        top_heading = doc["headings"][0]["text"]
        # 如果一级标题看起来像雷达型号，用它替换文件名
        if len(top_heading) < 30:
            radar_name = top_heading
            log.info(f"  识别雷达名称: {radar_name}")

    # 分块
    chunks = split_into_chunks(full_text)
    log.info(f"  分成 {len(chunks)} 个文本块")

    # 逐块抽取
    all_triples = []
    for i, chunk in enumerate(chunks):
        log.info(f"  处理块 {i+1}/{len(chunks)} ({len(chunk)} 字符)")
        triples = extract_triples_from_chunk(chunk, radar_name, docx_path.name)
        all_triples.extend(triples)
        log.info(f"    抽取到 {len(triples)} 条三元组")
        time.sleep(0.5)  # 避免 API 限流

    # 去重
    triples_dedup = deduplicate(all_triples)
    log.info(f"  去重后: {len(triples_dedup)} 条三元组")

    return {
        "radar_id":   radar_name.lower().replace(" ", "_").replace("/", "_"),
        "en_title":   radar_name,
        "source_file": docx_path.name,
        "triples":    triples_dedup,
    }


def run_manual_extraction():
    """批量处理 manuals/ 目录下所有 .docx 文件。"""
    docx_files = list(MANUAL_DIR.glob("*.docx"))
    if not docx_files:
        log.warning(f"在 {MANUAL_DIR} 目录下没有找到 .docx 文件")
        log.info("请把 Word 手册放到 manuals/ 目录下，文件名建议用雷达型号命名")
        return []

    log.info(f"找到 {len(docx_files)} 个 Word 文件: {[f.name for f in docx_files]}")

    results = []
    for path in docx_files:
        try:
            result = process_manual(path)
            results.append(result)
        except Exception as e:
            log.error(f"处理 {path.name} 失败: {e}")

    # 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总统计
    total = sum(len(r["triples"]) for r in results)
    log.info(f"\n抽取完成: {len(results)} 个手册, {total} 条三元组")
    log.info(f"结果保存到: {OUTPUT_PATH}")

    # 关系分布
    from collections import Counter
    rel_dist = Counter(t["relation"] for r in results for t in r["triples"])
    log.info("关系分布:")
    for rel, cnt in rel_dist.most_common():
        log.info(f"  {rel}: {cnt}")

    return results


# ══════════════════════════════════════════════════════════
#  合并到现有知识库
# ══════════════════════════════════════════════════════════

def merge_with_existing():
    """
    把手册抽取结果合并进现有知识库，重建索引。
    运行 manual extraction 完成后调用。
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from graphrag_retriever import load_triples_merged, HybridRetriever

    # 合并手册结果 + 原有结果
    sources = []
    for path in [
        str(OUTPUT_PATH),                                    # 手册
        "extraction_results/method_c_results.json",          # 少样本LLM
        "extraction_results/method_a_results.json",          # 规则
    ]:
        if Path(path).exists():
            sources.append(path)

    if len(sources) < 2:
        log.warning("找不到足够的三元组文件，跳过合并")
        return

    log.info(f"合并 {len(sources)} 个来源...")
    triples = load_triples_merged(*sources, prefer_source="manual_few_shot")
    log.info(f"合并后: {len(triples)} 条三元组")

    log.info("重建检索索引（需要几分钟）...")
    retriever = HybridRetriever(triples, enable_reranker=False)
    log.info("索引重建完成，可直接运行 qa_pipeline.py 使用新知识库")


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="雷达手册三元组抽取")
    parser.add_argument("--merge", action="store_true",
                        help="抽取完成后自动合并并重建索引")
    parser.add_argument("--file", type=str, default=None,
                        help="只处理指定文件（如 --file 某型雷达手册.docx）")
    args = parser.parse_args()

    if args.file:
        # 处理单个文件
        path = MANUAL_DIR / args.file
        if not path.exists():
            path = Path(args.file)
        if path.exists():
            result = process_manual(path)
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
            log.info(f"抽取完成: {len(result['triples'])} 条三元组")
        else:
            log.error(f"找不到文件: {args.file}")
    else:
        # 处理所有文件
        run_manual_extraction()

    if args.merge:
        merge_with_existing()
    else:
        print("\n提示: 运行以下命令将手册结果合并到知识库并重建索引:")
        print("  python extract_from_manual.py --merge")
