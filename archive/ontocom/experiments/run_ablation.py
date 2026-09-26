"""
Ablation study for OntoCom components.

Runs multiple OntoCom variants with different components enabled/disabled:
  - RotatE backbone only (no ontology components)
  - + TCNS only
  - + ORE only
  - + SAPC only
  - + TCNS + ORE
  - + TCNS + ORE + SAPC (full OntoCom)

Usage:
    python -m experiments.run_ablation --dataset radarkg
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
from evaluation.kg_metrics import evaluate_link_prediction, _build_true_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ABLATION_VARIANTS = [
    {"name": "RotatE (backbone)",    "tcns": False, "ore": False, "sapc": False},
    {"name": "+ TCNS",               "tcns": True,  "ore": False, "sapc": False},
    {"name": "+ ORE",                "tcns": False, "ore": True,  "sapc": False},
    {"name": "+ SAPC",               "tcns": False, "ore": False, "sapc": True},
    {"name": "+ TCNS + ORE",         "tcns": True,  "ore": True,  "sapc": False},
    {"name": "OntoCom (full)",       "tcns": True,  "ore": True,  "sapc": True},
]

RESULTS_DIR = Path("results/ablation")


def run_ablation(dataset_name: str = "radarkg", device: str = "cpu",
                 epochs: int = 1000, embedding_dim: int = 128):
    """Run ablation study on a dataset."""
    ds = load_dataset(dataset_name)
    true_set = _build_true_set(ds.train, ds.valid, ds.test)

    results = []
    for variant in ABLATION_VARIANTS:
        log.info(f"\n{'='*55}")
        log.info(f"  Ablation: {variant['name']}")
        log.info(f"{'='*55}")

        model, train_result = train_ontocom(
            ds, device=device, epochs=epochs, embedding_dim=embedding_dim,
            use_tcns=variant["tcns"], use_ore=variant["ore"], use_sapc=variant["sapc"],
        )

        score_fn = model.get_score_fn(device)
        metrics = evaluate_link_prediction(
            score_fn, ds.test, ds.num_entities, true_set, device=device,
        )

        variant_result = {
            "name": variant["name"],
            "tcns": variant["tcns"],
            "ore": variant["ore"],
            "sapc": variant["sapc"],
            "mrr": metrics["mrr"],
            "hits_at_1": metrics["hits_at_1"],
            "hits_at_3": metrics["hits_at_3"],
            "hits_at_10": metrics["hits_at_10"],
            "training_time": train_result["training_time"],
            "final_loss": train_result["final_loss"],
        }
        results.append(variant_result)

        print(f"\n  {variant['name']:<25s} MRR={metrics['mrr']:.4f}  "
              f"H@1={metrics['hits_at_1']:.4f}  H@3={metrics['hits_at_3']:.4f}  "
              f"H@10={metrics['hits_at_10']:.4f}")

    # Summary table
    print(f"\n{'='*75}")
    print(f"  ABLATION STUDY: {dataset_name}")
    print(f"{'='*75}")
    print(f"  {'Variant':<25s} {'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['name']:<25s} {r['mrr']:>8.4f} {r['hits_at_1']:>8.4f} "
              f"{r['hits_at_3']:>8.4f} {r['hits_at_10']:>8.4f}")
    print(f"{'='*75}\n")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ablation_{dataset_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "variants": results}, f,
                  ensure_ascii=False, indent=2)
    log.info(f"Results saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OntoCom ablation study")
    parser.add_argument("--dataset", type=str, default="radarkg")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    run_ablation(args.dataset, args.device, args.epochs, args.dim)
