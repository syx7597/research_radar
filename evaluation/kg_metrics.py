"""
Filtered link prediction evaluation: MRR, Hits@1, Hits@3, Hits@10.

Implements the standard filtered setting where known true triples are
removed from the ranking (so a correct prediction that happens to be
another true fact is not penalized).
"""

import torch
import logging
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)


def _build_true_set(train, valid, test) -> set[tuple[int, int, int]]:
    """Build a set of all known true triples for filtered evaluation."""
    true_set = set()
    for split in [train, valid, test]:
        for h, r, t in split.tolist():
            true_set.add((h, r, t))
    return true_set


@torch.no_grad()
def evaluate_link_prediction(
    score_fn,
    test_triples: torch.LongTensor,
    num_entities: int,
    true_triples: set[tuple[int, int, int]],
    batch_size: int = 256,
    device: str = "cpu",
    relation_filter: Optional[set[int]] = None,
) -> dict:
    """
    Filtered link prediction evaluation.

    Args:
        score_fn: callable(heads, relations, tails) -> scores tensor.
                  Each input is a 1-D LongTensor; output is a 1-D float tensor.
        test_triples: (N, 3) tensor of [h, r, t] test triples.
        num_entities: total number of entities in the KG.
        true_triples: set of all known (h, r, t) tuples for filtering.
        batch_size: number of test triples to process at once.
        device: torch device.
        relation_filter: if provided, only evaluate on these relation IDs.

    Returns:
        dict with keys: mrr, hits_at_1, hits_at_3, hits_at_10,
                        per_relation (dict[int, same metrics])
    """
    all_entities = torch.arange(num_entities, device=device)
    ranks_head = []
    ranks_tail = []
    per_rel: dict[int, dict[str, list]] = defaultdict(lambda: {"head": [], "tail": []})

    test_triples = test_triples.to(device)
    n = len(test_triples)

    for start in range(0, n, batch_size):
        batch = test_triples[start:start + batch_size]

        for idx in range(len(batch)):
            h, r, t = batch[idx].tolist()

            if relation_filter and r not in relation_filter:
                continue

            # --- Tail prediction: (h, r, ?) ---
            heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
            rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
            scores_tail = score_fn(heads, rels, all_entities)

            # Filter: set scores of known true triples (except the target) to -inf
            for e in range(num_entities):
                if e != t and (h, r, e) in true_triples:
                    scores_tail[e] = float("-inf")

            rank_t = (scores_tail >= scores_tail[t]).sum().item()
            ranks_tail.append(rank_t)
            per_rel[r]["tail"].append(rank_t)

            # --- Head prediction: (?, r, t) ---
            tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
            rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
            scores_head = score_fn(all_entities, rels, tails)

            for e in range(num_entities):
                if e != h and (e, r, t) in true_triples:
                    scores_head[e] = float("-inf")

            rank_h = (scores_head >= scores_head[h]).sum().item()
            ranks_head.append(rank_h)
            per_rel[r]["head"].append(rank_h)

        if (start + batch_size) % (batch_size * 10) == 0 or start + batch_size >= n:
            done = min(start + batch_size, n)
            log.info(f"Evaluated {done}/{n} test triples")

    all_ranks = ranks_head + ranks_tail
    metrics = _compute_metrics(all_ranks)

    # Per-relation metrics
    per_relation = {}
    for rid, sides in per_rel.items():
        rel_ranks = sides["head"] + sides["tail"]
        per_relation[rid] = _compute_metrics(rel_ranks)

    metrics["per_relation"] = per_relation
    return metrics


def _compute_metrics(ranks: list[int]) -> dict:
    """Compute MRR, Hits@1, Hits@3, Hits@10 from a list of ranks (1-based)."""
    if not ranks:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0, "count": 0}
    ranks_t = torch.tensor(ranks, dtype=torch.float)
    mrr = (1.0 / ranks_t).mean().item()
    h1 = (ranks_t <= 1).float().mean().item()
    h3 = (ranks_t <= 3).float().mean().item()
    h10 = (ranks_t <= 10).float().mean().item()
    return {
        "mrr": round(mrr, 4),
        "hits_at_1": round(h1, 4),
        "hits_at_3": round(h3, 4),
        "hits_at_10": round(h10, 4),
        "count": len(ranks),
    }


def type_violation_rate(
    predictions: list[tuple[int, int, int]],
    entity_type: dict[int, str],
    relation_domain: dict[int, set[str]],
    relation_range: dict[int, set[str]],
) -> dict:
    """
    Compute the fraction of predicted triples that violate type constraints.

    Args:
        predictions: list of (h, r, t) predicted triples.
        entity_type: entity_id -> type_name.
        relation_domain: rel_id -> set of valid head types.
        relation_range: rel_id -> set of valid tail types.

    Returns:
        dict with violation_rate, head_violations, tail_violations, total.
    """
    head_violations = 0
    tail_violations = 0
    total = len(predictions)

    for h, r, t in predictions:
        h_type = entity_type.get(h, "unknown")
        t_type = entity_type.get(t, "unknown")

        if r in relation_domain and h_type != "unknown":
            if h_type not in relation_domain[r]:
                head_violations += 1

        if r in relation_range and t_type != "unknown":
            if t_type not in relation_range[r]:
                tail_violations += 1

    return {
        "violation_rate": round((head_violations + tail_violations) / max(total * 2, 1), 4),
        "head_violations": head_violations,
        "tail_violations": tail_violations,
        "total_predictions": total,
    }
