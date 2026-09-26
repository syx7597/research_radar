"""
Structural Pattern Recovery experiment.

Protocol:
1. Remove all instances of a target relation from the KG
2. Train the model on remaining triples
3. Score all candidate triples for the removed relation
4. Evaluate: Precision@K, Recall@K against ground truth

This is a novel evaluation task that specifically tests SAPC's
ability to learn structural patterns (e.g., shared-band => competitor).

Usage:
    python -m experiments.run_structural_recovery --dataset radarkg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

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


def remove_relation(dataset, target_relation: str):
    """
    Create a modified dataset with all triples of target_relation removed.
    Returns: (modified_dataset_copy, removed_triples)
    """
    from datasets.loader import KGDataset
    from copy import deepcopy

    rid = dataset.relation_to_id.get(target_relation)
    if rid is None:
        raise ValueError(f"Relation '{target_relation}' not found. "
                         f"Available: {list(dataset.relation_to_id.keys())}")

    def filter_split(triples, rid):
        mask = triples[:, 1] != rid
        return triples[mask], triples[~mask]

    train_filtered, train_removed = filter_split(dataset.train, rid)
    valid_filtered, valid_removed = filter_split(dataset.valid, rid)
    test_filtered, test_removed = filter_split(dataset.test, rid)

    all_removed = torch.cat([train_removed, valid_removed, test_removed])

    # Create modified dataset
    modified = KGDataset(
        name=f"{dataset.name}_no_{target_relation}",
        train=train_filtered,
        valid=valid_filtered,
        test=test_filtered,
        entity_to_id=dataset.entity_to_id,
        relation_to_id=dataset.relation_to_id,
        id_to_entity=dataset.id_to_entity,
        id_to_relation=dataset.id_to_relation,
        entity_type=dataset.entity_type,
        type_to_entities=dataset.type_to_entities,
        relation_domain=dataset.relation_domain,
        relation_range=dataset.relation_range,
        type_hierarchy=dataset.type_hierarchy,
    )

    log.info(f"Removed {len(all_removed)} '{target_relation}' triples "
             f"(train: {len(train_removed)}, valid: {len(valid_removed)}, "
             f"test: {len(test_removed)})")

    return modified, all_removed


def evaluate_recovery(score_fn, removed_triples, dataset, k_values=(5, 10, 20)):
    """
    Evaluate how well the model can recover removed triples.

    For each removed triple (h, r, t), rank all valid tail candidates
    and check if the ground truth tail is in top-K.
    """
    rid = removed_triples[0, 1].item()

    # Get valid candidate entities for this relation type
    range_types = dataset.relation_range.get(rid, set())
    candidates = []
    for t in range_types:
        candidates.extend(dataset.type_to_entities.get(t, []))
    if not candidates:
        candidates = list(range(dataset.num_entities))
    candidates = list(set(candidates))

    ground_truth = set()
    for h, r, t in removed_triples.tolist():
        ground_truth.add((h, t))

    # For each unique head in removed triples, rank all candidates
    heads = set(h.item() for h in removed_triples[:, 0])
    all_predictions = {}

    for h in heads:
        h_tensor = torch.full((len(candidates),), h, dtype=torch.long)
        r_tensor = torch.full((len(candidates),), rid, dtype=torch.long)
        t_tensor = torch.tensor(candidates, dtype=torch.long)

        with torch.no_grad():
            scores = score_fn(h_tensor, r_tensor, t_tensor)

        # Sort by score (descending)
        sorted_indices = scores.argsort(descending=True)
        ranked_candidates = [candidates[i] for i in sorted_indices.tolist()]
        all_predictions[h] = ranked_candidates

    # Compute Precision@K and Recall@K
    results = {}
    for k in k_values:
        tp = 0
        total_predicted = 0
        total_relevant = len(ground_truth)

        for h, ranked in all_predictions.items():
            top_k = set(ranked[:k])
            true_tails = {t for (hh, t) in ground_truth if hh == h}
            tp += len(top_k & true_tails)
            total_predicted += k

        precision = tp / max(total_predicted, 1)
        recall = tp / max(total_relevant, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        results[f"precision_at_{k}"] = round(precision, 4)
        results[f"recall_at_{k}"] = round(recall, 4)
        results[f"f1_at_{k}"] = round(f1, 4)

    results["num_removed"] = len(removed_triples)
    results["num_heads"] = len(heads)
    results["num_candidates"] = len(candidates)

    return results


def run_recovery(dataset_name: str = "radarkg",
                 target_relation: str = "operatedBy",
                 device: str = "cpu",
                 epochs: int = 500):
    """Run structural pattern recovery experiment."""
    ds = load_dataset(dataset_name)

    # Find the best target relation (one with structural patterns)
    log.info(f"Dataset: {dataset_name}")
    log.info(f"Relations: {list(ds.relation_to_id.keys())}")

    modified_ds, removed = remove_relation(ds, target_relation)
    log.info(modified_ds.summary())

    if len(removed) < 3:
        log.warning(f"Only {len(removed)} removed triples — results may be noisy")

    # Train OntoCom (with SAPC) on modified dataset
    log.info("Training OntoCom (with SAPC)...")
    model_sapc, _ = train_ontocom(
        modified_ds, device=device, epochs=epochs,
        use_tcns=True, use_ore=True, use_sapc=True,
    )

    # Train OntoCom (without SAPC) for comparison
    log.info("Training OntoCom (without SAPC)...")
    model_no_sapc, _ = train_ontocom(
        modified_ds, device=device, epochs=epochs,
        use_tcns=True, use_ore=True, use_sapc=False,
    )

    # Train plain RotatE backbone
    log.info("Training RotatE backbone...")
    model_rotate, _ = train_ontocom(
        modified_ds, device=device, epochs=epochs,
        use_tcns=False, use_ore=False, use_sapc=False,
    )

    # Evaluate recovery
    results = {}
    for name, model in [("OntoCom (full)", model_sapc),
                         ("OntoCom (no SAPC)", model_no_sapc),
                         ("RotatE backbone", model_rotate)]:
        score_fn = model.get_score_fn(device)
        metrics = evaluate_recovery(score_fn, removed, ds)
        results[name] = metrics
        log.info(f"{name}: {metrics}")

    # Print results
    print(f"\n{'='*70}")
    print(f"  STRUCTURAL RECOVERY: {target_relation} on {dataset_name}")
    print(f"  ({len(removed)} triples removed)")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'P@5':>8s} {'R@5':>8s} {'P@10':>8s} {'R@10':>8s} {'R@20':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in results.items():
        print(f"  {name:<25s} "
              f"{m.get('precision_at_5', 0):>8.4f} {m.get('recall_at_5', 0):>8.4f} "
              f"{m.get('precision_at_10', 0):>8.4f} {m.get('recall_at_10', 0):>8.4f} "
              f"{m.get('recall_at_20', 0):>8.4f}")
    print(f"{'='*70}\n")

    # Save
    out_dir = Path("results/structural_recovery")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"recovery_{dataset_name}_{target_relation}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "target_relation": target_relation,
                   "results": results}, f, ensure_ascii=False, indent=2)
    log.info(f"Results saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="radarkg")
    parser.add_argument("--relation", type=str, default="operatedBy")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    run_recovery(args.dataset, args.relation, args.device, args.epochs)
