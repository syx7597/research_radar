"""
Analysis experiments for OntoCom paper:
  1. Type violation rate analysis
  2. Low-resource scaling experiment
  3. Per-relation breakdown visualization
"""

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset
from models.ontocom import train_ontocom
from evaluation.kg_metrics import (
    evaluate_link_prediction, _build_true_set, type_violation_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results/analysis")


def type_violation_analysis(dataset_name: str = "radarkg",
                            device: str = "cpu", epochs: int = 500,
                            top_k: int = 10):
    """
    For each model's top-K predictions per test query,
    count how many violate type constraints.
    """
    ds = load_dataset(dataset_name)
    true_set = _build_true_set(ds.train, ds.valid, ds.test)

    models_to_eval = [
        ("RotatE backbone", {"use_tcns": False, "use_ore": False, "use_sapc": False}),
        ("+ TCNS", {"use_tcns": True, "use_ore": False, "use_sapc": False}),
        ("OntoCom (full)", {"use_tcns": True, "use_ore": True, "use_sapc": True}),
    ]

    results = {}
    for name, flags in models_to_eval:
        log.info(f"Training {name}...")
        model, _ = train_ontocom(ds, device=device, epochs=epochs,
                                 use_tcns=flags.get("use_tcns", True),
                                 use_ore=flags.get("use_ore", True),
                                 use_sapc=flags.get("use_sapc", True))
        score_fn = model.get_score_fn(device)

        # Collect top-K predictions for each test triple
        predictions = []
        all_entities = torch.arange(ds.num_entities)

        for i in range(len(ds.test)):
            h, r, t = ds.test[i].tolist()

            # Tail prediction
            heads = torch.full((ds.num_entities,), h, dtype=torch.long)
            rels = torch.full((ds.num_entities,), r, dtype=torch.long)
            scores = score_fn(heads, rels, all_entities)

            # Filter known true triples
            for e in range(ds.num_entities):
                if e != t and (h, r, e) in true_set:
                    scores[e] = float("-inf")

            top_indices = scores.topk(top_k).indices.tolist()
            for pred_t in top_indices:
                predictions.append((h, r, pred_t))

        # Compute violation rate
        vr = type_violation_rate(
            predictions, ds.entity_type,
            ds.relation_domain, ds.relation_range,
        )
        results[name] = vr
        log.info(f"  {name}: violation_rate={vr['violation_rate']:.4f} "
                 f"({vr['head_violations']}h + {vr['tail_violations']}t "
                 f"/ {vr['total_predictions']} predictions)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TYPE VIOLATION ANALYSIS: {dataset_name} (top-{top_k})")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'ViolRate':>10s} {'HeadViol':>10s} {'TailViol':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name, vr in results.items():
        print(f"  {name:<25s} {vr['violation_rate']:>10.2%} "
              f"{vr['head_violations']:>10d} {vr['tail_violations']:>10d}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"type_violations_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def scaling_experiment(dataset_name: str = "radarkg",
                       fractions: list[float] = None,
                       device: str = "cpu", epochs: int = 500):
    """
    Train on varying fractions of training data.
    Shows how OntoCom degrades more gracefully than baselines.
    """
    fractions = fractions or [0.2, 0.4, 0.6, 0.8, 1.0]
    ds = load_dataset(dataset_name)
    true_set = _build_true_set(ds.train, ds.valid, ds.test)
    n = len(ds.train)

    models_to_eval = [
        ("RotatE backbone", {"use_tcns": False, "use_ore": False, "use_sapc": False}),
        ("OntoCom (full)", {"use_tcns": True, "use_ore": True, "use_sapc": True}),
    ]

    results = {name: [] for name, _ in models_to_eval}

    for frac in fractions:
        # Subsample training data
        k = max(int(n * frac), 10)
        perm = torch.randperm(n)[:k]
        from datasets.loader import KGDataset
        sub_ds = KGDataset(
            name=f"{dataset_name}_{frac:.0%}",
            train=ds.train[perm],
            valid=ds.valid, test=ds.test,
            entity_to_id=ds.entity_to_id,
            relation_to_id=ds.relation_to_id,
            id_to_entity=ds.id_to_entity,
            id_to_relation=ds.id_to_relation,
            entity_type=ds.entity_type,
            type_to_entities=ds.type_to_entities,
            relation_domain=ds.relation_domain,
            relation_range=ds.relation_range,
            type_hierarchy=ds.type_hierarchy,
        )

        for name, flags in models_to_eval:
            log.info(f"Scaling: {name} @ {frac:.0%} ({k} triples)")
            model, _ = train_ontocom(sub_ds, device=device, epochs=epochs,
                                     use_tcns=flags.get("use_tcns", True),
                                     use_ore=flags.get("use_ore", True),
                                     use_sapc=flags.get("use_sapc", True))
            score_fn = model.get_score_fn(device)
            metrics = evaluate_link_prediction(
                score_fn, ds.test, ds.num_entities, true_set, device=device,
            )
            results[name].append({
                "fraction": frac,
                "num_triples": k,
                "mrr": metrics["mrr"],
                "hits_at_10": metrics["hits_at_10"],
            })
            log.info(f"  MRR={metrics['mrr']:.4f} H@10={metrics['hits_at_10']:.4f}")

    # Print summary
    print(f"\n{'='*65}")
    print(f"  LOW-RESOURCE SCALING: {dataset_name}")
    print(f"{'='*65}")
    for name, data_list in results.items():
        print(f"\n  {name}:")
        print(f"    {'Fraction':>10s} {'Triples':>10s} {'MRR':>8s} {'H@10':>8s}")
        for d in data_list:
            print(f"    {d['fraction']:>10.0%} {d['num_triples']:>10d} "
                  f"{d['mrr']:>8.4f} {d['hits_at_10']:>8.4f}")
    print(f"{'='*65}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"scaling_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="type_violation",
                        choices=["type_violation", "scaling"])
    parser.add_argument("--dataset", type=str, default="radarkg")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    if args.experiment == "type_violation":
        type_violation_analysis(args.dataset, args.device, args.epochs)
    else:
        scaling_experiment(args.dataset, device=args.device, epochs=args.epochs)
