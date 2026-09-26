"""
RichEvaluator — 多维度评估框架 v2
支持指标：
  - Recall@K      (向后兼容)
  - MRR           (Mean Reciprocal Rank)
  - NDCG@K        (分级相关度，使用标注标签 correct=1.0/partial=0.5/wrong=0.0)
  - Exact Match   (EM，规范化字符串匹配)
  - F1-token      (词级别 F1，参考中文QA标准)
  - BERTScore     (语义相似度，使用 hfl/chinese-roberta-wwm-ext)

运行方式：
    python evaluation_v2.py
    python evaluation_v2.py --qa evaluation/qa_dataset.json --results evaluation/eval_results.json
    python evaluation_v2.py --rich  # 包含 BERTScore（较慢）
"""

import json
import re
import os
import math
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_QA_PATH      = Path("evaluation/qa_dataset.json")
DEFAULT_RESULTS_PATH = Path("evaluation/eval_results.json")
DEFAULT_ANNOTATED    = Path("extraction_results/annotation_v2_labeled.json")


# ═══════════════════════════════════════════════════════
#  文本规范化工具
# ═══════════════════════════════════════════════════════

_ZH_PARTICLES = {"的","了","吗","呢","啊","哦","嗯","是","在","有","也","都","和","与","或","但","因","为"}
_EN_ARTICLES  = {"a","an","the","is","are","was","were","of","in","on","at","by","for"}

def normalize_text(text: str) -> str:
    """规范化文本：小写、去标点、去虚词（用于 EM 和 F1 计算）。"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[，。？！；：""''「」【】（）、,.?!;:\"'()\[\]{}]", " ", text)
    tokens = re.findall(r'[a-z0-9/\-\.]+|[\u4e00-\u9fff]', text)
    filtered = [t for t in tokens
                if t not in _ZH_PARTICLES and t not in _EN_ARTICLES and len(t) > 0]
    return " ".join(filtered)


def tokenize(text: str) -> list[str]:
    """简单分词（支持中英混合）。"""
    return normalize_text(text).split()


# ═══════════════════════════════════════════════════════
#  基础指标函数
# ═══════════════════════════════════════════════════════

def exact_match(prediction: str, references: list[str]) -> float:
    """Exact Match：规范化后是否完全匹配任一参考答案。"""
    pred_norm = normalize_text(prediction)
    return float(any(pred_norm == normalize_text(r) for r in references if r))


def f1_token(prediction: str, references: list[str]) -> float:
    """词级别 F1（取多个参考答案中的最高值）。"""
    pred_tokens = set(tokenize(prediction))
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ref in references:
        if not ref:
            continue
        ref_tokens = set(tokenize(ref))
        if not ref_tokens:
            continue
        tp = len(pred_tokens & ref_tokens)
        if tp == 0:
            continue
        precision = tp / len(pred_tokens)
        recall    = tp / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def recall_at_k(context: str, references: list[str]) -> bool:
    """Recall@K：答案是否出现在上下文中（字符串匹配）。"""
    ctx_lower = context.lower()
    return any(str(r).lower() in ctx_lower for r in references if r)


def reciprocal_rank(ranked_items: list[dict], references: list[str]) -> float:
    """MRR：第一个命中位置的倒数排名。"""
    for rank, t in enumerate(ranked_items, start=1):
        text = f"{t.get('head','')} {t.get('tail','')}".lower()
        if any(str(r).lower() in text for r in references if r):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_items: list[dict],
              references: list[str],
              labeled_triples: Optional[dict] = None,
              k: int = 5) -> float:
    """
    NDCG@K：分级相关度评分。
    若提供 labeled_triples（标注数据字典），使用标注分级（correct=1.0/partial=0.5/wrong=0.0）。
    否则使用二值相关度（命中=1/否=0）。
    """
    def relevance(t: dict, rank: int) -> float:
        if rank > k:
            return 0.0
        head = t.get("head","")
        tail = t.get("tail","")
        text = f"{head} {tail}".lower()

        # 优先使用标注分级
        if labeled_triples:
            triple_key = f"{head}|{t.get('relation','')}|{tail}"
            label = labeled_triples.get(triple_key, "")
            if label == "correct":
                return 1.0
            if label == "partial":
                return 0.5
            if label == "wrong":
                return 0.0

        # 回退到字符串匹配（binary）
        return 1.0 if any(str(r).lower() in text for r in references if r) else 0.0

    dcg  = sum(relevance(t, i+1) / math.log2(i + 2) for i, t in enumerate(ranked_items[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len([r for r in references if r]))))
    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════
#  BERTScore
# ═══════════════════════════════════════════════════════

def bertscore_batch(predictions: list[str], references: list[str],
                    model_name: str = "hfl/chinese-roberta-wwm-ext") -> list[float]:
    """
    批量计算 BERTScore F1（使用指定的中文模型）。
    返回每对 (prediction, reference) 的 F1 值列表。
    """
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score(
            predictions, references,
            lang="zh",
            model_type=model_name,
            verbose=False,
            device="cpu",
        )
        return F1.tolist()
    except ImportError:
        log.warning("bert_score 未安装，跳过 BERTScore 计算。pip install bert-score")
        return [0.0] * len(predictions)
    except Exception as e:
        log.warning(f"BERTScore 计算失败: {e}")
        return [0.0] * len(predictions)


# ═══════════════════════════════════════════════════════
#  RichEvaluator 主类
# ═══════════════════════════════════════════════════════

class RichEvaluator:
    """
    多维度评估器。

    输入：
      qa_path      — QA 数据集（包含 questions）
      results_path — 系统输出（包含 details，每条含 generated/context）
      annotated_path — 人工标注文件（可选，用于 NDCG 分级相关度）
    """

    def __init__(self,
                 qa_path:        str = str(DEFAULT_QA_PATH),
                 results_path:   str = str(DEFAULT_RESULTS_PATH),
                 annotated_path: Optional[str] = None):

        with open(qa_path, encoding="utf-8") as f:
            qa_data = json.load(f)
        self.questions: list[dict] = qa_data.get("questions", [])

        self.results: list[dict] = []
        if Path(results_path).exists():
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            self.results = data.get("details", [])

        # 建立 question → result 的映射
        self._result_map: dict[str, dict] = {
            r.get("question",""): r for r in self.results
        }

        # 标注数据：(head|relation|tail) → label
        self._labeled_triples: dict[str, str] = {}
        if annotated_path and Path(annotated_path).exists():
            with open(annotated_path, encoding="utf-8") as f:
                ann_data = json.load(f)
            for radar in ann_data:
                for t in radar.get("llm_fewshot_triples_to_annotate", []):
                    if t.get("label"):
                        key = f"{t.get('head','')}|{t.get('relation','')}|{t.get('tail','')}"
                        self._labeled_triples[key] = t["label"]

        log.info(f"RichEvaluator: {len(self.questions)} 个问题, "
                 f"{len(self.results)} 条系统结果, "
                 f"{len(self._labeled_triples)} 条标注三元组")

    def _get_references(self, q: dict) -> list[str]:
        answer  = q.get("answer","")
        aliases = q.get("answer_aliases", [])
        refs = ([answer] if isinstance(answer, str) else list(answer))
        refs += [a for a in aliases if isinstance(a, str)]
        # 展开列表中的列表
        flat_refs = []
        for r in refs:
            if isinstance(r, list):
                flat_refs.extend(r)
            else:
                flat_refs.append(str(r))
        return [r for r in flat_refs if r]

    def evaluate(self,
                 top_k: int = 5,
                 compute_bertscore: bool = False) -> dict:
        """
        执行完整评估，返回指标字典。
        """
        metrics_by_type: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        predictions_for_bertscore: list[str] = []
        references_for_bertscore:  list[str] = []

        for q in self.questions:
            qtype = q.get("type", "unknown")
            refs  = self._get_references(q)
            result = self._result_map.get(q.get("question",""), {})

            generated = result.get("generated", result.get("answer", ""))
            context   = result.get("context", "")

            # 从 result 中提取 ranked triples（若有）
            ranked = result.get("reranked", result.get("fused", []))
            if isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
                ranked_triples = ranked
            else:
                ranked_triples = []

            # ── 各项指标计算 ──────────────────────────────
            r_at_k = float(recall_at_k(context, refs))
            mrr    = reciprocal_rank(ranked_triples, refs) if ranked_triples else r_at_k
            ndcg   = ndcg_at_k(ranked_triples, refs, self._labeled_triples, k=top_k) \
                     if ranked_triples else r_at_k
            em     = exact_match(generated, refs) if generated else 0.0
            f1     = f1_token(generated, refs) if generated else 0.0

            for metric, val in [
                ("recall_at_k", r_at_k),
                ("mrr",         mrr),
                ("ndcg",        ndcg),
                ("em",          em),
                ("f1_token",    f1),
            ]:
                metrics_by_type[qtype][metric].append(val)
                metrics_by_type["overall"][metric].append(val)

            if compute_bertscore and generated and refs:
                predictions_for_bertscore.append(generated)
                references_for_bertscore.append(refs[0])

        # BERTScore（批量）
        if compute_bertscore and predictions_for_bertscore:
            log.info("计算 BERTScore...")
            bs_scores = bertscore_batch(
                predictions_for_bertscore,
                references_for_bertscore,
            )
            # 分配回对应 question
            bs_idx = 0
            for q in self.questions:
                result = self._result_map.get(q.get("question",""), {})
                generated = result.get("generated", result.get("answer", ""))
                refs = self._get_references(q)
                qtype = q.get("type", "unknown")
                if generated and refs:
                    val = bs_scores[bs_idx] if bs_idx < len(bs_scores) else 0.0
                    metrics_by_type[qtype]["bertscore"].append(val)
                    metrics_by_type["overall"]["bertscore"].append(val)
                    bs_idx += 1

        # ── 汇总 ──────────────────────────────────────────
        summary: dict[str, dict] = {}
        for qtype, metric_lists in metrics_by_type.items():
            summary[qtype] = {
                metric: round(sum(vals) / len(vals), 4) if vals else 0.0
                for metric, vals in metric_lists.items()
            }
            summary[qtype]["count"] = len(next(iter(metric_lists.values()), []))

        return summary

    def print_report(self, summary: dict, top_k: int = 5):
        """打印格式化评估报告。"""
        metrics = ["recall_at_k", "mrr", "ndcg", "em", "f1_token"]
        if any("bertscore" in v for v in summary.values()):
            metrics.append("bertscore")

        header = f"{'类型':<15}" + "".join(f"{m:>12}" for m in metrics) + f"{'样本数':>8}"
        print(f"\n{'='*80}")
        print(f"  评估报告 (top_k={top_k})")
        print(f"{'='*80}")
        print(f"  {header}")
        print(f"  {'-'*78}")

        order = sorted(summary.keys(), key=lambda x: (x != "overall", x))
        for qtype in order:
            vals = summary[qtype]
            row = f"  {qtype:<15}"
            for m in metrics:
                v = vals.get(m, 0.0)
                row += f"{v:>12.1%}"
            row += f"{vals.get('count', 0):>8}"
            print(row)

        print(f"{'='*80}")

    def save_report(self, summary: dict,
                    output_path: str = "evaluation/eval_results_v2.json"):
        """保存评估结果到 JSON 文件。"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info(f"评估报告已保存: {output_path}")


# ═══════════════════════════════════════════════════════
#  全流程评估（集成到 qa_pipeline.py 后运行）
# ═══════════════════════════════════════════════════════

def run_full_evaluation(
    qa_path:         str = str(DEFAULT_QA_PATH),
    top_k:           int = 5,
    use_reranker:    bool = True,
    compute_bertscore: bool = False,
    annotated_path:  Optional[str] = None,
    save_path:       str = "evaluation/eval_results_v2.json",
):
    """
    运行 qa_pipeline.batch_evaluate() 获取系统输出，
    然后用 RichEvaluator 计算全套指标。
    """
    # Step 1: 运行系统
    log.info("运行 QA pipeline 评估...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from qa_pipeline import batch_evaluate
        results = batch_evaluate(qa_path, top_k=top_k, use_reranker=use_reranker)
    except Exception as e:
        log.error(f"QA pipeline 运行失败: {e}")
        raise

    # 保存 pipeline 结果（供 RichEvaluator 读取）
    tmp_results = Path("evaluation/eval_results_tmp.json")
    with open(tmp_results, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Step 2: 多维度评估
    evaluator = RichEvaluator(
        qa_path        = qa_path,
        results_path   = str(tmp_results),
        annotated_path = annotated_path,
    )
    summary = evaluator.evaluate(top_k=top_k, compute_bertscore=compute_bertscore)
    evaluator.print_report(summary, top_k=top_k)
    evaluator.save_report(summary, save_path)

    return summary


# ═══════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RichEvaluator — 多维度评估框架")
    parser.add_argument("--qa",          default=str(DEFAULT_QA_PATH))
    parser.add_argument("--results",     default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--annotated",   default=None,
                        help="标注文件路径（用于 NDCG 分级相关度）")
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--rich",        action="store_true",
                        help="计算 BERTScore（较慢）")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="重新运行 QA pipeline 再评估")
    parser.add_argument("--save",        default="evaluation/eval_results_v2.json")
    args = parser.parse_args()

    if args.run_pipeline:
        run_full_evaluation(
            qa_path           = args.qa,
            top_k             = args.top_k,
            compute_bertscore = args.rich,
            annotated_path    = args.annotated,
            save_path         = args.save,
        )
    else:
        # 仅读取已有结果文件进行评估
        evaluator = RichEvaluator(
            qa_path        = args.qa,
            results_path   = args.results,
            annotated_path = args.annotated,
        )
        summary = evaluator.evaluate(top_k=args.top_k, compute_bertscore=args.rich)
        evaluator.print_report(summary, top_k=args.top_k)
        evaluator.save_report(summary, args.save)
