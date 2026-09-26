"""
Run link prediction experiments for all baselines on all datasets.

Usage:
    python -m experiments.run_link_prediction --model rotate --dataset radarkg
    python -m experiments.run_link_prediction --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset, DATASET_REGISTRY
from models.baselines import train_baseline, make_score_fn, MODEL_CONFIGS
from evaluation.kg_metrics import evaluate_link_prediction, _build_true_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results/link_prediction")


def run_single(model_name: str, dataset_name: str, device: str = "cpu",
               epochs: int = None, save: bool = True) -> dict:
    """Train a model and evaluate link prediction on one dataset."""
    log.info(f"=== {model_name.upper()} on {dataset_name} ===")

    # Load dataset
    ds = load_dataset(dataset_name)
    log.info(ds.summary())

    # Train
    result = train_baseline(model_name, ds, device=device, epochs=epochs,
                            save_dir="checkpoints")

    # Evaluate
    model = result["model"]
    score_fn = make_score_fn(model, device=device)
    true_set = _build_true_set(ds.train, ds.valid, ds.test)

    log.info(f"Evaluating on {len(ds.test)} test triples (filtered setting)...")
    t0 = time.time()
    metrics = evaluate_link_prediction(
        score_fn=score_fn,
        test_triples=ds.test,
        num_entities=ds.num_entities,
        true_triples=true_set,
        batch_size=256,
        device=device,
    )
    eval_time = time.time() - t0
    log.info(f"Evaluation done in {eval_time:.1f}s")

    # Print results
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} on {dataset_name}")
    print(f"{'='*55}")
    print(f"  MRR:      {metrics['mrr']:.4f}")
    print(f"  Hits@1:   {metrics['hits_at_1']:.4f}")
    print(f"  Hits@3:   {metrics['hits_at_3']:.4f}")
    print(f"  Hits@10:  {metrics['hits_at_10']:.4f}")
    print(f"  Count:    {metrics['count']}")
    print(f"  Train:    {result['training_time']}s | Eval: {eval_time:.1f}s")
    print(f"{'='*55}\n")

    # Per-relation breakdown
    if ds.id_to_relation:
        print("  Per-relation MRR:")
        for rid, rm in sorted(metrics.get("per_relation", {}).items(),
                               key=lambda x: -x[1]["mrr"]):
            rname = ds.id_to_relation.get(rid, str(rid))[:40]
            print(f"    {rname:<40s} MRR={rm['mrr']:.4f}  (n={rm['count']})")
        print()

    # Save results
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "model": model_name,
            "dataset": dataset_name,
            "metrics": {k: v for k, v in metrics.items() if k != "per_relation"},
            "per_relation": {
                ds.id_to_relation.get(rid, str(rid)): rm
                for rid, rm in metrics.get("per_relation", {}).items()
            },
            "training_time": result["training_time"],
            "eval_time": round(eval_time, 1),
            "final_loss": result["final_loss"],
        }
        out_path = RESULTS_DIR / f"{model_name}_{dataset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        log.info(f"Results saved: {out_path}")

    return metrics


def run_all(device: str = "cpu", datasets: list[str] = None,
            models: list[str] = None, epochs: int = None):
    """Run all model x dataset combinations."""
    datasets = datasets or ["radarkg"]  # Start with RadarKG for fast iteration
    models = models or list(MODEL_CONFIGS.keys())

    all_results = {}
    for ds_name in datasets:
        for model_name in models:
            try:
                metrics = run_single(model_name, ds_name, device=device,
                                     epochs=epochs)
                all_results[f"{model_name}_{ds_name}"] = metrics
            except Exception as e:
                log.error(f"FAILED: {model_name} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    print(f"\n{'='*75}")
    print(f"  SUMMARY: Link Prediction Results")
    print(f"{'='*75}")
    print(f"  {'Model':<12s} {'Dataset':<12s} {'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key, m in all_results.items():
        parts = key.split("_", 1)
        print(f"  {parts[0]:<12s} {parts[1]:<12s} {m['mrr']:>8.4f} "
              f"{m['hits_at_1']:>8.4f} {m['hits_at_3']:>8.4f} {m['hits_at_10']:>8.4f}")
    print(f"{'='*75}\n")

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    summary = {k: {sk: sv for sk, sv in v.items() if sk != "per_relation"}
               for k, v in all_results.items()}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link prediction experiments")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--dataset", type=str, default="radarkg",
                        help=f"Dataset name: {list(DATASET_REGISTRY.keys())}")
    parser.add_argument("--all", action="store_true",
                        help="Run all models on all datasets")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    args = parser.parse_args()

    if args.all:
        run_all(device=args.device, epochs=args.epochs)
    elif args.model:
        run_single(args.model, args.dataset, device=args.device, epochs=args.epochs)
    else:
        # Default: run all models on RadarKG
        run_all(device=args.device, datasets=["radarkg"], epochs=args.epochs)
