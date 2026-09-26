"""
QA 测试集自动生成器
从知识图谱三元组自动生成多种类型的问答对，目标生成 150+ 条高质量测试题。

问题类型：
  single_hop   - 单跳事实问题（直接从三元组生成）
  multi_hop    - 多跳推理问题（需要两步三元组链）
  comparison   - 比较型问题（同一属性的多个实体对比）
  aggregation  - 聚合型问题（"有哪些X-band雷达由Raytheon研制？"）
  negation     - 否定/不可知问题（答案不在KG中，测试拒绝幻觉能力）

运行方式：
    python generate_qa.py
    python generate_qa.py --min_conf 0.90 --out evaluation/qa_dataset_v2.json
"""

import json
import re
import argparse
import logging
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

DEFAULT_TRIPLES_PATH  = Path("graphrag_index/merged_triples.json")
DEFAULT_QA_OUT        = Path("evaluation/qa_dataset_v2.json")
DEFAULT_MIN_CONF      = 0.88

# 问题模板：(关系, 中文问题模板, answer_field)
SINGLE_HOP_TEMPLATES: list[tuple[str, str]] = [
    ("developedBy",      "{head}是由哪家公司研制的？"),
    ("operatedBy",       "{head}被哪个国家装备使用？"),
    ("hasFrequencyBand", "{head}的工作频段是什么？"),
    ("deployedOn",       "{head}部署在哪种平台上？"),
    ("exportedTo",       "{head}已出口到哪些国家？"),
    ("upgradeOf",        "{head}是哪个型号的升级版？"),
    ("affiliatedTo",     "{tail}是哪个国家的军工企业？"),  # tail→affiliatedTo→Country
]

# 多跳模板：(关系链路, 问题模板, 中间变量名)
MULTI_HOP_TEMPLATES: list[tuple[list[str], str]] = [
    (["developedBy", "affiliatedTo"],
     "研制{head}的公司来自哪个国家？"),
    (["deployedOn", "operatedBy"],
     "装备了{tail_r1}的国家是哪个？"),  # {tail_r1} = deployedOn的tail
    (["operatedBy", "developedBy"],
     "{tail_r1}装备了哪些型号的雷达，这些雷达由哪个公司研制？"),
    (["upgradeOf", "developedBy"],
     "{head}的前身型号是由哪家公司研制的？"),
]

# 聚合问题模板
AGGREGATION_TEMPLATES: list[tuple[str, str]] = [
    ("hasFrequencyBand", "有哪些{tail}波段的雷达系统？"),
    ("developedBy",      "{tail}研制了哪些雷达型号？"),
    ("operatedBy",       "{tail}装备了哪些雷达系统？"),
    ("deployedOn",       "部署在{tail}上的雷达有哪些型号？"),
]

# 否定/不可知问题（固定生成，不依赖三元组）
NEGATION_QUESTIONS = [
    {
        "id": "neg_001", "type": "negation",
        "question": "AN/SPY-1雷达工作在Ka波段吗？",
        "answer": "否",
        "answer_aliases": ["不是", "No", "不工作在Ka波段"],
        "note": "AN/SPY-1工作在S波段，不是Ka波段",
    },
    {
        "id": "neg_002", "type": "negation",
        "question": "Type 346雷达是由Raytheon公司研制的吗？",
        "answer": "否",
        "answer_aliases": ["不是", "No", "不是Raytheon研制"],
        "note": "Type 346由中国CETC研制",
    },
    {
        "id": "neg_003", "type": "negation",
        "question": "SMART-L雷达的制造商是哪个美国公司？",
        "answer": "根据现有知识库无法确定",
        "answer_aliases": ["无法确定", "不是美国公司", "Thales Nederland是荷兰公司"],
        "note": "SMART-L由荷兰Thales Nederland制造，非美国公司",
    },
    {
        "id": "neg_004", "type": "negation",
        "question": "AN/TPY-2雷达被部署在驱逐舰上吗？",
        "answer": "否",
        "answer_aliases": ["不是", "No", "AN/TPY-2是地基雷达"],
        "note": "AN/TPY-2是THAAD地基雷达，不部署在舰艇上",
    },
    {
        "id": "neg_005", "type": "negation",
        "question": "目前哪款中国雷达装备在美国海军驱逐舰上？",
        "answer": "根据现有知识库无法确定",
        "answer_aliases": ["无法确定", "没有", "不存在"],
        "note": "没有中国雷达装备在美国海军舰艇上",
    },
]


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def _normalize_id(text: str) -> str:
    return re.sub(r"[^\w]", "_", text.lower())[:30]


def _deduplicate_qa(qa_list: list[dict]) -> list[dict]:
    """按问题文本去重。"""
    seen: set[str] = set()
    result: list[dict] = []
    for q in qa_list:
        key = q["question"].strip()
        if key not in seen:
            seen.add(key)
            result.append(q)
    return result


# ═══════════════════════════════════════════════════════
#  生成器
# ═══════════════════════════════════════════════════════

class QAGenerator:

    def __init__(self, triples: list[dict], min_conf: float = DEFAULT_MIN_CONF):
        self.triples  = [t for t in triples if t.get("confidence", 0) >= min_conf]
        self.min_conf = min_conf

        # 构建关系索引
        self._rel_index: dict[str, list[dict]] = defaultdict(list)
        for t in self.triples:
            self._rel_index[t.get("relation","")].append(t)

        # 实体→关系→尾实体索引（用于多跳）
        self._head_rel_tail: dict[tuple, list[str]] = defaultdict(list)
        for t in self.triples:
            key = (t.get("head",""), t.get("relation",""))
            self._head_rel_tail[key].append(t.get("tail",""))

        log.info(f"QAGenerator 初始化: {len(self.triples)} 条高置信度三元组")

    # ── 单跳问题 ──────────────────────────────────────────

    def generate_single_hop(self, max_per_template: int = 8) -> list[dict]:
        questions: list[dict] = []
        q_id = 0

        for relation, template in SINGLE_HOP_TEMPLATES:
            triples = self._rel_index.get(relation, [])
            random.shuffle(triples)

            # affiliatedTo 特殊处理：以 tail(Manufacturer) 为 head
            if relation == "affiliatedTo":
                for t in triples[:max_per_template]:
                    head = t.get("tail","")   # Manufacturer
                    tail = t.get("tail","")
                    source_head = t.get("head","")  # actual manufacturer
                    if not source_head:
                        continue
                    question = template.format(tail=source_head)
                    answer   = t.get("tail","")  # Country
                    # fix: template uses {tail} = Country, but we want manufacturer→country
                    question = f"{source_head}是哪个国家的军工企业？"
                    answer   = t.get("tail","")
                    if not question or not answer:
                        continue
                    q_id += 1
                    questions.append({
                        "id":             f"sq_{q_id:03d}",
                        "type":           "single_hop",
                        "question":       question,
                        "answer":         answer,
                        "answer_aliases": [],
                        "source_triple":  f"{source_head} affiliatedTo {answer}",
                    })
            else:
                for t in triples[:max_per_template]:
                    head = t.get("head","")
                    tail = t.get("tail","")
                    if not head or not tail:
                        continue
                    question = template.format(head=head, tail=tail)
                    q_id += 1
                    questions.append({
                        "id":             f"sq_{q_id:03d}",
                        "type":           "single_hop",
                        "question":       question,
                        "answer":         tail,
                        "answer_aliases": [],
                        "source_triple":  f"{head} {relation} {tail}",
                    })

        log.info(f"单跳问题生成: {len(questions)} 条")
        return questions

    # ── 多跳问题 ──────────────────────────────────────────

    def generate_multi_hop(self, max_questions: int = 40) -> list[dict]:
        questions: list[dict] = []
        q_id = 0

        # 链路1: RadarSystem→developedBy→Manufacturer→affiliatedTo→Country
        for t1 in self._rel_index.get("developedBy", []):
            if q_id >= max_questions:
                break
            radar = t1.get("head","")
            mfr   = t1.get("tail","")
            countries = self._head_rel_tail.get((mfr, "affiliatedTo"), [])
            if not countries:
                continue
            country = countries[0]
            question = f"研制{radar}的公司来自哪个国家？"
            q_id += 1
            questions.append({
                "id":             f"mq_{q_id:03d}",
                "type":           "multi_hop",
                "question":       question,
                "answer":         country,
                "answer_aliases": [],
                "hop_chain":      f"{radar}→developedBy→{mfr}→affiliatedTo→{country}",
            })

        # 链路2: RadarSystem→deployedOn→Platform; Platform→operatedBy→Country (via radar)
        # 实际路径：Radar→deployedOn→Platform；找同一Radar的operatedBy→Country
        for t1 in self._rel_index.get("deployedOn", []):
            if q_id >= max_questions:
                break
            radar    = t1.get("head","")
            platform = t1.get("tail","")
            countries = self._head_rel_tail.get((radar, "operatedBy"), [])
            if not countries:
                continue
            country = countries[0]
            question = f"{radar}部署在{platform}上，该雷达属于哪个国家的海军装备？"
            q_id += 1
            questions.append({
                "id":             f"mq_{q_id:03d}",
                "type":           "multi_hop",
                "question":       question,
                "answer":         country,
                "answer_aliases": [],
                "hop_chain":      f"{radar}→deployedOn→{platform}; {radar}→operatedBy→{country}",
            })

        # 链路3: RadarSystem→upgradeOf→OldRadar；找OldRadar→developedBy→Manufacturer
        for t1 in self._rel_index.get("upgradeOf", []):
            if q_id >= max_questions:
                break
            new_radar  = t1.get("head","")
            old_radar  = t1.get("tail","")
            mfrs = self._head_rel_tail.get((old_radar, "developedBy"), [])
            if not mfrs:
                mfrs = self._head_rel_tail.get((new_radar, "developedBy"), [])
            if not mfrs:
                continue
            mfr = mfrs[0]
            question = f"{new_radar}的前身型号{old_radar}是由哪家公司研制的？"
            q_id += 1
            questions.append({
                "id":             f"mq_{q_id:03d}",
                "type":           "multi_hop",
                "question":       question,
                "answer":         mfr,
                "answer_aliases": [],
                "hop_chain":      f"{new_radar}→upgradeOf→{old_radar}→developedBy→{mfr}",
            })

        log.info(f"多跳问题生成: {len(questions)} 条")
        return questions

    # ── 比较型问题 ────────────────────────────────────────

    def generate_comparison(self, max_questions: int = 25) -> list[dict]:
        questions: list[dict] = []
        q_id = 0

        # 同国家、不同频段比较
        country_radars: dict[str, list[str]] = defaultdict(list)
        radar_band: dict[str, set[str]] = defaultdict(set)
        for t in self._rel_index.get("operatedBy", []):
            country_radars[t["tail"]].append(t["head"])
        for t in self._rel_index.get("hasFrequencyBand", []):
            radar_band[t["head"]].add(t["tail"])

        for country, radars in country_radars.items():
            if q_id >= max_questions:
                break
            radars_with_band = [r for r in radars if radar_band.get(r)]
            if len(radars_with_band) < 2:
                continue
            r1, r2 = radars_with_band[0], radars_with_band[1]
            b1 = ", ".join(sorted(radar_band[r1]))
            b2 = ", ".join(sorted(radar_band[r2]))
            question = f"{r1}和{r2}分别工作在什么频段？"
            q_id += 1
            questions.append({
                "id":             f"cq_{q_id:03d}",
                "type":           "comparison",
                "question":       question,
                "answer":         f"{r1}工作在{b1}波段；{r2}工作在{b2}波段",
                "answer_aliases": [b1, b2, r1, r2],
                "comparison_entities": [r1, r2],
            })

        # 同制造商不同雷达比较
        mfr_radars: dict[str, list[str]] = defaultdict(list)
        for t in self._rel_index.get("developedBy", []):
            mfr_radars[t["tail"]].append(t["head"])

        for mfr, radars in mfr_radars.items():
            if q_id >= max_questions:
                break
            if len(radars) < 2:
                continue
            r1, r2 = radars[0], radars[1]
            question = f"{mfr}同时研制了{r1}和{r2}吗？这两款雷达分别部署在哪类平台上？"
            plat1 = self._head_rel_tail.get((r1, "deployedOn"), ["未知"])[0]
            plat2 = self._head_rel_tail.get((r2, "deployedOn"), ["未知"])[0]
            if plat1 == "未知" or plat2 == "未知":
                continue
            q_id += 1
            questions.append({
                "id":             f"cq_{q_id:03d}",
                "type":           "comparison",
                "question":       question,
                "answer":         f"是，{r1}部署在{plat1}；{r2}部署在{plat2}",
                "answer_aliases": [plat1, plat2, mfr],
                "comparison_entities": [r1, r2],
            })

        log.info(f"比较型问题生成: {len(questions)} 条")
        return questions

    # ── 聚合型问题 ────────────────────────────────────────

    def generate_aggregation(self, max_per_template: int = 5) -> list[dict]:
        questions: list[dict] = []
        q_id = 0

        for relation, template in AGGREGATION_TEMPLATES:
            # 按 tail 分组
            tail_groups: dict[str, list[str]] = defaultdict(list)
            for t in self._rel_index.get(relation, []):
                tail_groups[t["tail"]].append(t["head"])

            # 取拥有最多头实体的 tail 值
            sorted_tails = sorted(
                tail_groups.items(), key=lambda x: -len(x[1])
            )[:max_per_template]

            for tail, heads in sorted_tails:
                if len(heads) < 2:
                    continue
                question = template.format(tail=tail)
                q_id += 1
                questions.append({
                    "id":             f"aq_{q_id:03d}",
                    "type":           "aggregation",
                    "question":       question,
                    "answer":         heads[:5],   # 列表形式
                    "answer_aliases": heads,
                    "expected_count": len(heads),
                })

        log.info(f"聚合型问题生成: {len(questions)} 条")
        return questions

    # ── 质量过滤 ──────────────────────────────────────────

    def quality_filter(self, questions: list[dict],
                        existing_qa_path: Optional[str] = None) -> list[dict]:
        """
        过滤质量差的问题：
        1. 问题或答案为空
        2. 答案过短（< 1 个字符）
        3. 与已有 QA 集重复（若提供）
        """
        existing_questions: set[str] = set()
        if existing_qa_path and Path(existing_qa_path).exists():
            with open(existing_qa_path, encoding="utf-8") as f:
                existing = json.load(f).get("questions", [])
            existing_questions = {q["question"].strip() for q in existing}

        filtered: list[dict] = []
        for q in questions:
            question = q.get("question","").strip()
            answer   = q.get("answer","")
            if not question or (isinstance(answer, str) and not answer):
                continue
            if question in existing_questions:
                continue
            filtered.append(q)

        log.info(f"质量过滤: {len(questions)} → {len(filtered)} 条")
        return filtered


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def generate_benchmark(
    triples_path: str = str(DEFAULT_TRIPLES_PATH),
    output_path:  str = str(DEFAULT_QA_OUT),
    min_conf:     float = DEFAULT_MIN_CONF,
    existing_qa:  Optional[str] = "evaluation/qa_dataset.json",
    seed:         int = 42,
) -> dict:
    """生成完整的 QA 基准数据集。"""
    random.seed(seed)

    if not Path(triples_path).exists():
        print(f"找不到三元组文件: {triples_path}")
        print("请先运行 build_index.py")
        raise SystemExit(1)

    with open(triples_path, encoding="utf-8") as f:
        triples = json.load(f)

    gen = QAGenerator(triples, min_conf=min_conf)

    # 生成各类型问题
    single_hop   = gen.generate_single_hop(max_per_template=10)
    multi_hop    = gen.generate_multi_hop(max_questions=40)
    comparison   = gen.generate_comparison(max_questions=25)
    aggregation  = gen.generate_aggregation(max_per_template=5)
    negation     = NEGATION_QUESTIONS

    all_questions = single_hop + multi_hop + comparison + aggregation + negation
    all_questions = _deduplicate_qa(all_questions)
    all_questions = gen.quality_filter(all_questions, existing_qa)

    # 重新编号
    for i, q in enumerate(all_questions, start=1):
        if not q.get("id") or q["id"].startswith("sq_") or q["id"].startswith("mq_"):
            prefix = {"single_hop": "sq", "multi_hop": "mq",
                      "comparison": "cq", "aggregation": "aq",
                      "negation": "nq"}.get(q.get("type",""), "q")
            q["id"] = f"{prefix}_{i:03d}"

    output = {
        "version": "2.0",
        "description": "雷达装备领域KGQA基准数据集（自动生成 + 人工审核）",
        "stats": {
            "total":       len(all_questions),
            "single_hop":  sum(1 for q in all_questions if q["type"] == "single_hop"),
            "multi_hop":   sum(1 for q in all_questions if q["type"] == "multi_hop"),
            "comparison":  sum(1 for q in all_questions if q["type"] == "comparison"),
            "aggregation": sum(1 for q in all_questions if q["type"] == "aggregation"),
            "negation":    sum(1 for q in all_questions if q["type"] == "negation"),
        },
        "questions": all_questions,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  QA 基准生成完成: {output_path}")
    print(f"{'='*55}")
    for k, v in output["stats"].items():
        print(f"  {k:<15}: {v}")
    print(f"{'='*55}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA 测试集自动生成器")
    parser.add_argument("--triples", default=str(DEFAULT_TRIPLES_PATH),
                        help="三元组 JSON 路径")
    parser.add_argument("--out",     default=str(DEFAULT_QA_OUT),
                        help="输出 QA JSON 路径")
    parser.add_argument("--min_conf", type=float, default=DEFAULT_MIN_CONF,
                        help="最低置信度阈值（默认 0.88）")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    generate_benchmark(
        triples_path = args.triples,
        output_path  = args.out,
        min_conf     = args.min_conf,
        seed         = args.seed,
    )
