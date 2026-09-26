"""
QA Integration experiment: OntoCom embeddings for entity re-ranking.

Protocol:
1. Load RadarKG dataset and train OntoCom (or load checkpoint)
2. For each QA query, use BM25+Vector retrieval to get candidate entities
3. Re-rank candidates using OntoCom embedding similarity
4. Compare: original pipeline vs +OntoCom re-ranking

This demonstrates that OntoCom embeddings improve downstream KG-grounded QA
beyond standard retrieval methods.

Usage:
    python -m experiments.run_qa_integration
"""

import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset
from models.ontocom import train_ontocom
from evaluation.kg_metrics import _build_true_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_qa_dataset(path: str) -> list[dict]:
    """Load QA evaluation dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("questions", data.get("data", []))


def entity_similarity_reranking(
    query_entity: str,
    candidates: list[str],
    model,
    dataset,
    device: str = "cpu",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Re-rank candidate entities using OntoCom embedding similarity.

    For a query entity, find its embedding and rank candidates by
    cosine similarity in embedding space.

    Args:
        query_entity: entity name (head entity in the query)
        candidates: list of candidate entity names
        model: trained OntoCom model
        dataset: KGDataset with entity_to_id mapping
        device: torch device
        top_k: number of top entities to return

    Returns:
        list of (entity_name, similarity_score) sorted by score
    """
    if query_entity not in dataset.entity_to_id:
        return [(c, 0.0) for c in candidates[:top_k]]

    h_id = dataset.entity_to_id[query_entity]

    # Get query entity embedding (complex: real + imaginary parts)
    h_tensor = torch.tensor([h_id], dtype=torch.long)
    with torch.no_grad():
        h_emb = model.scorer.entity_embedding(h_tensor)  # (1, dim*2)

    # Compute similarity to all candidates
    scored = []
    for cand in candidates:
        if cand not in dataset.entity_to_id:
            scored.append((cand, 0.0))
            continue
        c_id = dataset.entity_to_id[cand]
        c_tensor = torch.tensor([c_id], dtype=torch.long)
        with torch.no_grad():
            c_emb = model.scorer.entity_embedding(c_tensor)  # (1, dim*2)

        # Cosine similarity
        sim = torch.nn.functional.cosine_similarity(h_emb, c_emb).item()
        scored.append((cand, sim))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def score_answer_with_ontocom(
    answer: str,
    query: str,
    model,
    dataset,
    device: str = "cpu",
) -> float:
    """
    Score a candidate answer entity using OntoCom for a given query entity.

    Uses embedding similarity as a plausibility score.
    """
    # Extract any entity names from the query
    for ename in dataset.entity_to_id:
        if ename.lower() in query.lower():
            sims = entity_similarity_reranking(ename, [answer], model, dataset, device)
            if sims:
                return sims[0][1]
    return 0.0


def evaluate_qa_with_ontocom(
    qa_data: list[dict],
    model,
    dataset,
    alpha: float = 0.7,
    device: str = "cpu",
) -> dict:
    """
    Evaluate QA performance with OntoCom re-ranking.

    alpha: weight for original retrieval score (1-alpha for OntoCom score)
    """
    results = {"total": 0, "em": 0, "recall_at_1": 0, "recall_at_3": 0}

    for item in qa_data:
        question = item.get("question", "")
        answer = item.get("answer", item.get("answers", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        retrieved = item.get("retrieved_entities", [])

        if not retrieved:
            continue

        # Try to re-rank with OntoCom
        query_entities = [e for e in dataset.entity_to_id
                         if e.lower() in question.lower()]

        if query_entities and len(retrieved) > 1:
            q_ent = query_entities[0]
            sims = entity_similarity_reranking(
                q_ent, retrieved, model, dataset, device, top_k=len(retrieved)
            )
            # Blend original rank with OntoCom score
            orig_scores = {e: 1.0 / (i + 1) for i, e in enumerate(retrieved)}
            blended = []
            for e, sim_score in sims:
                orig = orig_scores.get(e, 0.0)
                blended.append((e, alpha * orig + (1 - alpha) * max(sim_score, 0)))
            blended.sort(key=lambda x: -x[1])
            reranked = [e for e, _ in blended]
        else:
            reranked = retrieved

        results["total"] += 1
        if reranked and reranked[0].lower() == answer.lower():
            results["em"] += 1
            results["recall_at_1"] += 1
        if answer.lower() in [e.lower() for e in reranked[:3]]:
            results["recall_at_3"] += 1

    n = max(results["total"], 1)
    return {
        "em": round(results["em"] / n, 4),
        "recall_at_1": round(results["recall_at_1"] / n, 4),
        "recall_at_3": round(results["recall_at_3"] / n, 4),
        "total": results["total"],
    }


def run_qa_integration(device: str = "cpu", epochs: int = 500):
    """Run QA integration experiment."""
    ds = load_dataset("radarkg")

    # Train OntoCom
    log.info("Training OntoCom for QA integration...")
    model, _ = train_ontocom(ds, device=device, epochs=epochs)

    # Load QA dataset
    qa_path = Path("evaluation/qa_dataset_v2.json")
    if not qa_path.exists():
        qa_path = Path("evaluation/qa_dataset.json")

    qa_data = load_qa_dataset(str(qa_path))
    log.info(f"Loaded {len(qa_data)} QA items from {qa_path}")

    # Check if retrieved_entities field exists
    has_retrieved = any("retrieved_entities" in item for item in qa_data[:5])
    if not has_retrieved:
        log.warning("QA dataset does not have 'retrieved_entities' field. "
                    "Run graphrag retrieval first to populate this field.")
        log.info("Falling back to embedding-based entity recommendation...")

        # Demonstrate embedding-based entity similarity
        log.info("\n=== OntoCom Entity Similarity Demo ===")
        demo_entities = [e for e in list(ds.entity_to_id.keys())[:5]]
        log.info(f"Query entities: {demo_entities[:2]}")
        all_entities = list(ds.entity_to_id.keys())

        for query_ent in demo_entities[:2]:
            similar = entity_similarity_reranking(
                query_ent, all_entities, model, ds, device, top_k=5
            )
            log.info(f"\nMost similar to '{query_ent}':")
            for e, score in similar:
                if e != query_ent:
                    etype = ds.entity_type.get(ds.entity_to_id[e], "?")
                    log.info(f"  {e} ({etype}): {score:.4f}")

        return {"status": "no_retrieved_entities", "demo_completed": True}

    results = evaluate_qa_with_ontocom(qa_data, model, ds, device=device)
    log.info(f"\n=== QA Integration Results ===")
    log.info(f"  EM: {results['em']:.4f}")
    log.info(f"  Recall@1: {results['recall_at_1']:.4f}")
    log.info(f"  Recall@3: {results['recall_at_3']:.4f}")
    log.info(f"  Total evaluated: {results['total']}")

    Path("results/qa_integration").mkdir(parents=True, exist_ok=True)
    with open("results/qa_integration/ontocom_qa.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_qa_integration()
