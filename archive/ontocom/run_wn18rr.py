"""Run OntoCom on WN18RR and save results."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from datasets.loader import load_dataset
from models.ontocom import train_ontocom
from evaluation.kg_metrics import evaluate_link_prediction, _build_true_set

ds = load_dataset("wn18rr")
log.info(f"WN18RR: {ds.num_entities} entities, {ds.num_relations} rels, "
         f"{len(ds.train)} train / {len(ds.valid)} valid / {len(ds.test)} test triples")

model, result = train_ontocom(
    ds, device="cpu", epochs=200,
    embedding_dim=64,
    num_neg=64,
    batch_size=512,
    lr=1e-3,
    path_min_support=100,
    lambda_cluster=0.005,
    lambda_hierarchy=0.002,
    lambda_path=0.05,
    p_typed_start=0.5,
    p_typed_end=0.9,
    ore_frequency=50,   # WN18RR has 170 batches/epoch; compute ORE ~3x per epoch
)

log.info(f"Training time: {result['training_time']}s, final_loss: {result['final_loss']:.4f}")

score_fn = model.get_score_fn(device="cpu")

log.info("Building true set for filtered evaluation...")
true_set = _build_true_set(ds.train, ds.valid, ds.test)

log.info(f"Evaluating on {len(ds.test)} test triples...")
metrics = evaluate_link_prediction(
    score_fn=score_fn,
    test_triples=ds.test,
    num_entities=ds.num_entities,
    true_triples=true_set,
    batch_size=256,
    device="cpu",
)

out = {
    "model": "ontocom",
    "dataset": "wn18rr",
    "training_time": result["training_time"],
    **{k: v for k, v in metrics.items() if k != "per_relation"},
    "per_relation": metrics.get("per_relation", {}),
}

Path("results/link_prediction").mkdir(parents=True, exist_ok=True)
with open("results/link_prediction/ontocom_wn18rr.json", "w") as f:
    json.dump(out, f, indent=2)

log.info(f"Results saved. MRR={metrics['mrr']:.4f} "
         f"H@1={metrics['hits_at_1']:.4f} "
         f"H@10={metrics['hits_at_10']:.4f}")
