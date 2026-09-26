"""
领域自适应双编码器微调
使用雷达领域QA对对 BAAI/bge-small-zh-v1.5 进行对比学习微调。

方法：
  - 正例对：(问题, 正确答案三元组文本)
  - 难负例：对每个问题，用 BM25 检索的 Top-N 中排除正例的三元组
  - 损失函数：MultipleNegativesRankingLoss（in-batch negatives）

数据来源：
  - evaluation/qa_dataset.json（QA问题 + 答案）
  - graphrag_index/merged_triples.json（知识图谱三元组）

运行方式：
    pip install sentence-transformers rank-bm25
    python finetune_embeddings.py
    python finetune_embeddings.py --epochs 5 --batch_size 16 --out models/radar_bge

评估：
    python finetune_embeddings.py --eval --model models/radar_bge
"""

import json
import os
import argparse
import logging
import random
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── 离线模式（如果模型已缓存本地）─────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE",       "1")

BASE_MODEL      = "BAAI/bge-small-zh-v1.5"
DEFAULT_OUT_DIR = Path("models/radar_bge")
DEFAULT_QA_PATH = Path("evaluation/qa_dataset.json")
DEFAULT_TRIPLES = Path("graphrag_index/merged_triples.json")


# ═══════════════════════════════════════════════════════
#  三元组转文本（与 graphrag_retriever.py 中保持一致）
# ═══════════════════════════════════════════════════════

REL_ZH = {
    "deployedOn":       "部署于",
    "developedBy":      "由…研制",
    "operatedBy":       "装备于",
    "exportedTo":       "出口至",
    "upgradeOf":        "是…的升级版",
    "derivedFrom":      "衍生自",
    "competitorOf":     "竞争型号为",
    "coDeployedWith":   "与…同平台配套",
    "hasFrequencyBand": "工作频段为",
    "hasFunction":      "具备功能",
    "affiliatedTo":     "隶属于",
}

def triple_to_text(t: dict) -> str:
    rel  = REL_ZH.get(t.get("relation", ""), t.get("relation", ""))
    base = f"{t.get('head','')} {rel} {t.get('tail','')}"
    ev   = t.get("evidence", "")
    return f"{base}。{ev[:100]}" if ev else base


# ═══════════════════════════════════════════════════════
#  构建训练数据对
# ═══════════════════════════════════════════════════════

def build_training_pairs(
    qa_path: str,
    triples_path: str,
    hard_negatives_k: int = 10,
    seed: int = 42,
) -> list[tuple[str, str, list[str]]]:
    """
    为每个 QA 对构建训练三元组 (query, positive, [hard_negatives])。

    正例匹配策略：
      - 先用答案字符串在三元组文本中做精确匹配
      - 若无精确匹配，用 BM25 检索 Top-1 作为近似正例

    难负例：BM25 Top-K 中排除正例后的前 hard_negatives_k 条
    """
    random.seed(seed)

    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = qa_data.get("questions", [])

    with open(triples_path, encoding="utf-8") as f:
        triples = json.load(f)

    triple_texts = [triple_to_text(t) for t in triples]

    # 构建 BM25 索引
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("请安装 rank_bm25: pip install rank-bm25")
        raise

    import re
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r'[a-zA-Z0-9/\-\.]+|[\u4e00-\u9fff]', text)
        stop = {"的","了","在","是","和","与","为","于","由","a","an","the","is","of","in","on","by"}
        return [t.lower() for t in tokens if t.lower() not in stop]

    bm25 = BM25Okapi([tokenize(t) for t in triple_texts])
    log.info(f"BM25 索引: {len(triple_texts)} 条三元组")

    training_data: list[tuple[str, str, list[str]]] = []
    skipped = 0

    for q in questions:
        question = q.get("question", "")
        answer   = q.get("answer", "")
        aliases  = q.get("answer_aliases", [])

        all_answers = ([answer] if isinstance(answer, str) else list(answer))
        all_answers += [a for a in aliases if isinstance(a, str)]
        all_answers = [a.lower() for a in all_answers if a]

        if not all_answers:
            skipped += 1
            continue

        # 找正例三元组
        positive_idx: Optional[int] = None

        # 精确匹配：答案出现在三元组文本中
        import numpy as np
        scores = bm25.get_scores(tokenize(question))
        top_indices = np.argsort(scores)[::-1][:50]

        for idx in top_indices:
            text_lower = triple_texts[idx].lower()
            if any(a in text_lower for a in all_answers):
                positive_idx = int(idx)
                break

        if positive_idx is None:
            skipped += 1
            continue

        positive_text = triple_texts[positive_idx]

        # 难负例：top-K BM25 中排除正例
        hard_neg_indices = [
            int(i) for i in top_indices
            if int(i) != positive_idx
        ][:hard_negatives_k]
        hard_negatives = [triple_texts[i] for i in hard_neg_indices]

        training_data.append((question, positive_text, hard_negatives))

    log.info(f"训练对构建: {len(training_data)} 条（跳过 {skipped} 条无法匹配的问题）")
    return training_data


# ═══════════════════════════════════════════════════════
#  数据增强：问题改写
# ═══════════════════════════════════════════════════════

def augment_with_paraphrase(
    training_data: list[tuple[str, str, list[str]]],
    augment_ratio: float = 0.5,
) -> list[tuple[str, str, list[str]]]:
    """
    简单规则式问题改写（不依赖LLM，快速增强）。
    augment_ratio: 对多少比例的训练对进行增强。
    """
    PARAPHRASE_RULES = [
        # (原模式, 改写模式)
        (r"^(.+)是由(.+)研制的？$",     r"\2研制了\1吗？"),
        (r"^(.+)的工作频段是什么？$",    r"\1使用哪个频段？"),
        (r"^(.+)部署在哪种平台上？$",    r"\1搭载在哪类舰艇/飞机/地面平台？"),
        (r"^(.+)被哪个国家装备使用？$",  r"哪国海军/空军装备了\1？"),
        (r"^研制(.+)的公司来自哪个国家", r"\1的研制方是哪个国家的企业？"),
    ]

    import re
    augmented = list(training_data)
    n_aug = int(len(training_data) * augment_ratio)
    selected = random.sample(training_data, min(n_aug, len(training_data)))

    for query, positive, hard_negs in selected:
        for pattern, replacement in PARAPHRASE_RULES:
            new_query = re.sub(pattern, replacement, query)
            if new_query != query:
                augmented.append((new_query, positive, hard_negs))
                break

    log.info(f"数据增强: {len(training_data)} → {len(augmented)} 条")
    return augmented


# ═══════════════════════════════════════════════════════
#  模型微调
# ═══════════════════════════════════════════════════════

def finetune(
    qa_path:          str   = str(DEFAULT_QA_PATH),
    triples_path:     str   = str(DEFAULT_TRIPLES),
    output_dir:       str   = str(DEFAULT_OUT_DIR),
    base_model:       str   = BASE_MODEL,
    epochs:           int   = 3,
    batch_size:       int   = 16,
    warmup_steps:     int   = 50,
    eval_split:       float = 0.15,
    augment:          bool  = True,
    seed:             int   = 42,
) -> str:
    """
    微调双编码器，返回输出模型路径。
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from sentence_transformers.evaluation import InformationRetrievalEvaluator
        from torch.utils.data import DataLoader
    except ImportError:
        print("请安装 sentence-transformers: pip install sentence-transformers")
        raise

    random.seed(seed)
    output_dir_ = Path(output_dir)
    output_dir_.mkdir(parents=True, exist_ok=True)

    # ── 构建训练数据 ─────────────────────────────────────
    log.info("构建训练对...")
    training_data = build_training_pairs(qa_path, triples_path, seed=seed)

    if not training_data:
        print("没有可用的训练对，请先扩充 QA 数据集（运行 generate_qa.py）")
        raise SystemExit(1)

    if augment:
        training_data = augment_with_paraphrase(training_data)

    # ── 切分训练/验证集 ──────────────────────────────────
    random.shuffle(training_data)
    n_val   = max(1, int(len(training_data) * eval_split))
    val_data  = training_data[:n_val]
    train_data = training_data[n_val:]
    log.info(f"训练集: {len(train_data)} 条，验证集: {len(val_data)} 条")

    # ── 构建 InputExample（MultipleNegativesRankingLoss格式）─
    train_examples = [
        InputExample(texts=[query, positive])
        for query, positive, _ in train_data
    ]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # ── 加载模型 ─────────────────────────────────────────
    log.info(f"加载基础模型: {base_model}")
    model = SentenceTransformer(base_model)

    # ── 损失函数 ─────────────────────────────────────────
    loss = losses.MultipleNegativesRankingLoss(model)

    # ── 评估器（IR Evaluator）────────────────────────────
    queries    = {str(i): q  for i, (q, p, _) in enumerate(val_data)}
    corpus     = {str(i): p  for i, (q, p, _) in enumerate(val_data)}
    relevant   = {str(i): {str(i)} for i in range(len(val_data))}

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant,
        name="radar_val",
        show_progress_bar=False,
    )

    # ── 开始微调 ─────────────────────────────────────────
    log.info(f"开始微调: epochs={epochs}, batch_size={batch_size}")
    model.fit(
        train_objectives=[(train_loader, loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir_),
        evaluation_steps=max(10, len(train_loader) // 2),
        show_progress_bar=True,
        save_best_model=True,
    )

    log.info(f"微调完成，模型已保存至: {output_dir_}")
    return str(output_dir_)


# ═══════════════════════════════════════════════════════
#  评估：冻结 vs 微调后的 Recall@K 对比
# ═══════════════════════════════════════════════════════

def evaluate_models(
    qa_path:      str,
    triples_path: str,
    finetuned_model: str,
    base_model:      str = BASE_MODEL,
    top_k:           int = 5,
) -> dict:
    """对比冻结基础模型 vs 微调后模型在 QA 上的 Recall@K。"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
    except ImportError:
        print("请安装 sentence-transformers 和 faiss-cpu")
        raise

    with open(qa_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = qa_data.get("questions", [])

    with open(triples_path, encoding="utf-8") as f:
        triples = json.load(f)

    triple_texts = [triple_to_text(t) for t in triples]

    def compute_recall(model_name_or_path: str) -> float:
        model = SentenceTransformer(model_name_or_path)
        corpus_emb = model.encode(
            triple_texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        index = faiss.IndexFlatIP(corpus_emb.shape[1])
        index.add(corpus_emb)

        hits = 0
        for q in questions:
            question = q.get("question","")
            answer   = q.get("answer","")
            aliases  = q.get("answer_aliases",[])
            all_ans  = ([answer] if isinstance(answer,str) else list(answer))
            all_ans += [a for a in aliases if isinstance(a,str)]
            all_ans  = [a.lower() for a in all_ans if a]

            q_emb = model.encode(
                [question], normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            _, indices = index.search(q_emb, top_k)
            ctx = " ".join(triple_texts[i] for i in indices[0] if i >= 0).lower()
            if any(a in ctx for a in all_ans):
                hits += 1

        return hits / len(questions) if questions else 0.0

    print(f"\n{'='*55}")
    print(f"  双编码器 Recall@{top_k} 对比")
    print(f"{'='*55}")
    base_recall = compute_recall(base_model)
    print(f"  冻结基础模型 ({base_model}): {base_recall:.1%}")
    ft_recall = compute_recall(finetuned_model)
    print(f"  微调后模型   ({finetuned_model}): {ft_recall:.1%}")
    delta = ft_recall - base_recall
    print(f"  提升:        {delta:+.1%}")
    print(f"{'='*55}")

    return {"base": base_recall, "finetuned": ft_recall, "delta": delta}


# ═══════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双编码器领域自适应微调")
    parser.add_argument("--qa",         default=str(DEFAULT_QA_PATH))
    parser.add_argument("--triples",    default=str(DEFAULT_TRIPLES))
    parser.add_argument("--out",        default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--eval",       action="store_true",
                        help="仅评估（不微调），需要指定 --model")
    parser.add_argument("--model",      default=str(DEFAULT_OUT_DIR),
                        help="待评估的微调模型路径（--eval 模式使用）")
    args = parser.parse_args()

    if args.eval:
        evaluate_models(
            qa_path          = args.qa,
            triples_path     = args.triples,
            finetuned_model  = args.model,
            base_model       = args.base_model,
        )
    else:
        out = finetune(
            qa_path      = args.qa,
            triples_path = args.triples,
            output_dir   = args.out,
            base_model   = args.base_model,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            augment      = not args.no_augment,
        )
        # 自动评估
        evaluate_models(
            qa_path         = args.qa,
            triples_path    = args.triples,
            finetuned_model = out,
            base_model      = args.base_model,
        )
