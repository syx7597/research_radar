"""
Baseline KGE models via PyKEEN for OntoCom comparison.

Supported models: TransE, RotatE, CompGCN, TuckER, ComplEx, HAKE.
Each model is configured with consistent hyperparameters and trained
on the provided KGDataset, then evaluated using filtered link prediction.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from pykeen.models import (
    TransE,
    RotatE,
    TuckER,
    ComplEx,
    DistMult,
)
from pykeen.training import SLCWATrainingLoop
from pykeen.losses import MarginRankingLoss, NSSALoss, BCEWithLogitsLoss
from pykeen.sampling import BasicNegativeSampler
from pykeen.triples import TriplesFactory

log = logging.getLogger(__name__)

# Model registry with default hyperparameters
MODEL_CONFIGS = {
    "transe": {
        "cls": TransE,
        "kwargs": {"embedding_dim": 256},
        "loss": MarginRankingLoss,
        "loss_kwargs": {"margin": 5.0},
        "lr": 1e-3,
        "epochs": 500,
        "neg_per_pos": 256,
    },
    "rotate": {
        "cls": RotatE,
        "kwargs": {"embedding_dim": 256},
        "loss": NSSALoss,
        "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
        "lr": 1e-3,
        "epochs": 500,
        "neg_per_pos": 256,
    },
    "tucker": {
        "cls": TuckER,
        "kwargs": {"embedding_dim": 200, "relation_dim": 200,
                   "dropout_0": 0.3, "dropout_1": 0.4, "dropout_2": 0.5},
        "loss": NSSALoss,
        "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
        "lr": 5e-4,
        "epochs": 500,
        "neg_per_pos": 128,
    },
    "complex": {
        "cls": ComplEx,
        "kwargs": {"embedding_dim": 256,
                   "regularizer": "LpRegularizer",
                   "regularizer_kwargs": {"p": 3, "weight": 0.001}},
        "loss": NSSALoss,
        "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
        "lr": 1e-3,
        "epochs": 500,
        "neg_per_pos": 256,
    },
    "distmult": {
        "cls": DistMult,
        "kwargs": {"embedding_dim": 256},
        "loss": NSSALoss,
        "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
        "lr": 1e-3,
        "epochs": 500,
        "neg_per_pos": 256,
    },
}

# Small-dataset overrides (for RadarKG with <2000 triples)
SMALL_DATASET_OVERRIDES = {
    "embedding_dim": 128,
    "epochs": 1000,
    "neg_per_pos": 64,
    "lr": 5e-4,
}

# CPU-feasible settings for large public datasets (WN18RR, FB15k-237)
LARGE_DATASET_CPU_OVERRIDES = {
    "embedding_dim": 64,
    "epochs": 200,
    "neg_per_pos": 64,
    "lr": 1e-3,
    "batch_size": 512,
}


def _make_triples_factory(triples: torch.LongTensor,
                          entity_to_id: dict, relation_to_id: dict) -> TriplesFactory:
    """Create a PyKEEN TriplesFactory from pre-mapped triples."""
    return TriplesFactory(
        mapped_triples=triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


def train_baseline(
    model_name: str,
    dataset,
    device: str = "cpu",
    epochs: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Train a baseline KGE model on a KGDataset.

    Args:
        model_name: one of MODEL_CONFIGS keys.
        dataset: KGDataset object from datasets.loader.
        device: "cpu" or "cuda".
        epochs: override default epoch count.
        embedding_dim: override default embedding dimension.
        save_dir: if set, save model checkpoint here.

    Returns:
        dict with model, training_time, config.
    """
    key = model_name.lower()
    if key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[key].copy()
    is_small = len(dataset.train) < 2000
    is_large_cpu = len(dataset.train) >= 50000  # WN18RR, FB15k-237 on CPU

    # Apply small-dataset overrides
    if is_small:
        log.info(f"Small dataset detected ({len(dataset.train)} triples), applying overrides")
        if "embedding_dim" in config["kwargs"]:
            config["kwargs"]["embedding_dim"] = SMALL_DATASET_OVERRIDES["embedding_dim"]
        if "relation_dim" in config["kwargs"]:
            config["kwargs"]["relation_dim"] = SMALL_DATASET_OVERRIDES["embedding_dim"]
        config["epochs"] = SMALL_DATASET_OVERRIDES["epochs"]
        config["neg_per_pos"] = SMALL_DATASET_OVERRIDES["neg_per_pos"]
        config["lr"] = SMALL_DATASET_OVERRIDES["lr"]
    elif is_large_cpu and device == "cpu":
        log.info(f"Large dataset on CPU ({len(dataset.train)} triples), applying CPU overrides")
        if "embedding_dim" in config["kwargs"]:
            config["kwargs"]["embedding_dim"] = LARGE_DATASET_CPU_OVERRIDES["embedding_dim"]
        if "relation_dim" in config["kwargs"]:
            config["kwargs"]["relation_dim"] = LARGE_DATASET_CPU_OVERRIDES["embedding_dim"]
        config["epochs"] = LARGE_DATASET_CPU_OVERRIDES["epochs"]
        config["neg_per_pos"] = LARGE_DATASET_CPU_OVERRIDES["neg_per_pos"]
        config["lr"] = LARGE_DATASET_CPU_OVERRIDES["lr"]

    # Apply user overrides
    if epochs is not None:
        config["epochs"] = epochs
    if embedding_dim is not None:
        config["kwargs"]["embedding_dim"] = embedding_dim
        if "relation_dim" in config["kwargs"]:
            config["kwargs"]["relation_dim"] = embedding_dim

    # Create TriplesFactory
    tf_train = _make_triples_factory(dataset.train, dataset.entity_to_id, dataset.relation_to_id)

    # Build model
    loss_fn = config["loss"](**config.get("loss_kwargs", {}))
    model = config["cls"](
        triples_factory=tf_train,
        loss=loss_fn,
        **config["kwargs"],
    ).to(device)

    log.info(f"Model: {model_name} | Params: {sum(p.numel() for p in model.parameters()):,} | "
             f"Dim: {config['kwargs'].get('embedding_dim', '?')} | "
             f"Epochs: {config['epochs']} | Device: {device}")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    sampler = BasicNegativeSampler(
        mapped_triples=tf_train.mapped_triples,
        num_negs_per_pos=config["neg_per_pos"],
    )
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=tf_train,
        optimizer=optimizer,
        negative_sampler=sampler,
    )

    t0 = time.time()
    losses = training_loop.train(
        triples_factory=tf_train,
        num_epochs=config["epochs"],
        batch_size=64 if is_small else (512 if is_large_cpu else 256),
        use_tqdm=True,
        use_tqdm_batch=False,
    )
    train_time = time.time() - t0

    log.info(f"Training complete in {train_time:.1f}s | Final loss: {losses[-1]:.4f}")

    # Save checkpoint
    if save_dir:
        save_path = Path(save_dir) / f"{model_name}_{dataset.name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {k: str(v) if not isinstance(v, (int, float, str, dict, list)) else v
                       for k, v in config.items()},
            "dataset": dataset.name,
            "model_name": model_name,
        }, save_path)
        log.info(f"Checkpoint saved: {save_path}")

    return {
        "model": model,
        "model_name": model_name,
        "dataset_name": dataset.name,
        "training_time": round(train_time, 1),
        "final_loss": round(losses[-1], 4),
        "config": config,
        "triples_factory": tf_train,
    }


def make_score_fn(model, device="cpu"):
    """
    Create a score function compatible with kg_metrics.evaluate_link_prediction.

    Returns callable(heads, relations, tails) -> scores.
    """
    model.eval()

    def score_fn(heads, relations, tails):
        h = heads.to(device)
        r = relations.to(device)
        t = tails.to(device)
        # PyKEEN models expect (batch, 3) input for score_hrt
        triples = torch.stack([h, r, t], dim=1)
        return model.score_hrt(triples).squeeze(-1)

    return score_fn
