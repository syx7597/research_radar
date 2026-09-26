"""
OntoCom: Ontology-Constrained Knowledge Graph Completion.

Combines RotatE backbone with three novel components:
  1. TCNS (Type-Constrained Negative Sampling)
  2. ORE  (Ontology-Regularized Embedding)
  3. SAPC (Structure-Aware Path Completion)

Joint loss: L = L_base + lambda_cluster * L_cluster
                      + lambda_hierarchy * L_hierarchy
                      + lambda_path * L_path
"""

import logging
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.type_constraint import TypeIndex, TypeConstrainedSampler, OntologyRegularizer
from models.path_miner import mine_path_templates, PathAuxLoss

log = logging.getLogger(__name__)


class RotatEScorer(nn.Module):
    """
    RotatE scoring function: h ∘ r ≈ t in complex space.
    Each entity/relation is represented as a complex vector.
    Relations are unit-modulus rotations.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 gamma: float = 9.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)

        # Entity embeddings (real + imaginary parts)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim * 2)
        # Relation phases (rotation angles)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.entity_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.relation_embedding.weight,
                         -math.pi, math.pi)

    def forward(self, h_idx: torch.LongTensor, r_idx: torch.LongTensor,
                t_idx: torch.LongTensor) -> torch.Tensor:
        """Score triples. Higher score = more plausible."""
        h = self.entity_embedding(h_idx)
        t = self.entity_embedding(t_idx)
        r_phase = self.relation_embedding(r_idx)

        dim = self.embedding_dim
        h_re, h_im = h[..., :dim], h[..., dim:]
        t_re, t_im = t[..., :dim], t[..., dim:]
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # RotatE: h ∘ r - t
        score_re = h_re * r_re - h_im * r_im - t_re
        score_im = h_re * r_im + h_im * r_re - t_im

        # L1 distance
        score = torch.sqrt(score_re ** 2 + score_im ** 2 + 1e-9)
        score = self.gamma - score.sum(dim=-1)
        return score

    def get_entity_embeddings(self) -> torch.Tensor:
        """Return entity embedding weight matrix."""
        return self.entity_embedding.weight


class OntoCom(nn.Module):
    """
    Full OntoCom model: RotatE + TCNS + ORE + SAPC.
    """

    def __init__(self, dataset,
                 embedding_dim: int = 128,
                 gamma: float = 9.0,
                 # TCNS params
                 num_neg: int = 64,
                 p_typed_start: float = 0.5,
                 p_typed_end: float = 0.9,
                 # ORE params
                 lambda_cluster: float = 0.01,
                 lambda_hierarchy: float = 0.005,
                 # SAPC params
                 lambda_path: float = 0.1,
                 path_min_support: int = 5,
                 path_margin: float = 3.0,
                 # Training
                 total_epochs: int = 500,
                 lr: float = 1e-3,
                 batch_size: int = 256,
                 adversarial_temperature: float = 1.0,
                 # Ablation flags
                 use_tcns: bool = True,
                 use_ore: bool = True,
                 use_sapc: bool = True):
        super().__init__()

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.lambda_cluster = lambda_cluster
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_path = lambda_path
        self.total_epochs = total_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.adversarial_temperature = adversarial_temperature
        self.use_tcns = use_tcns
        self.use_ore = use_ore
        self.use_sapc = use_sapc

        # RotatE backbone
        self.scorer = RotatEScorer(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            embedding_dim=embedding_dim,
            gamma=gamma,
        )

        # Type index
        self.type_index = TypeIndex(
            entity_type=dataset.entity_type,
            type_to_entities=dataset.type_to_entities,
            relation_domain=dataset.relation_domain,
            relation_range=dataset.relation_range,
            type_hierarchy=dataset.type_hierarchy,
        )

        # TCNS
        self.sampler = TypeConstrainedSampler(
            type_index=self.type_index,
            num_neg=num_neg,
            p_typed_start=p_typed_start if use_tcns else 0.0,
            p_typed_end=p_typed_end if use_tcns else 0.0,
            total_epochs=total_epochs,
        )

        # ORE
        self.ont_reg = OntologyRegularizer(self.type_index) if use_ore else None

        # SAPC — mine path templates
        if use_sapc:
            templates = mine_path_templates(
                dataset.train,
                min_support=path_min_support,
                max_templates=50,
            )
            self.path_loss_fn = PathAuxLoss(templates, margin=path_margin) if templates else None
        else:
            self.path_loss_fn = None

        log.info(f"OntoCom initialized: dim={embedding_dim}, "
                 f"TCNS={use_tcns}, ORE={use_ore}, SAPC={use_sapc}")
        log.info(f"  Entities: {dataset.num_entities}, Relations: {dataset.num_relations}, "
                 f"Types: {dataset.num_types}")

    def score_hrt(self, triples: torch.LongTensor) -> torch.Tensor:
        """Score triples (B, 3) -> (B,)."""
        return self.scorer(triples[:, 0], triples[:, 1], triples[:, 2])

    def score(self, h: torch.LongTensor, r: torch.LongTensor,
              t: torch.LongTensor) -> torch.Tensor:
        """Score individual h, r, t tensors."""
        return self.scorer(h, r, t)

    def _nssa_loss(self, pos_scores: torch.Tensor,
                   neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Negative Sampling Self-Adversarial loss (from RotatE paper).

        L = -log σ(γ - d(h∘r, t)) - Σ_i p_i · log σ(d(h_i'∘r, t_i') - γ)

        where p_i = softmax(α · d(h_i'∘r, t_i')) are adversarial weights.
        """
        # pos_scores: (B,), neg_scores: (B, num_neg)
        pos_loss = -F.logsigmoid(pos_scores).mean()

        # Adversarial weights (detached — no gradient through weights)
        with torch.no_grad():
            neg_weights = F.softmax(
                self.adversarial_temperature * neg_scores, dim=-1
            )

        neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1).mean()

        return pos_loss + neg_loss

    def train_model(self, device: str = "cpu", verbose: bool = True,
                    ore_frequency: int = 20) -> dict:
        """
        Full training loop with TCNS + ORE + SAPC.

        Args:
            ore_frequency: compute ORE loss every N batches (default 10) to reduce overhead
                           on large datasets.

        Returns:
            dict with losses, training_time, etc.
        """
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_triples = self.dataset.train.to(device)
        n = len(train_triples)

        history = {
            "base_loss": [], "cluster_loss": [], "hierarchy_loss": [],
            "path_loss": [], "total_loss": [],
        }

        t0 = time.time()

        for epoch in range(self.total_epochs):
            self.sampler.set_epoch(epoch)

            # Shuffle training data
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            epoch_losses = {"base": 0, "cluster": 0, "hierarchy": 0, "path": 0}
            n_batches = 0

            batch_num = 0
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                pos_triples = train_triples[batch_idx]  # (B, 3)

                # Positive scores
                pos_scores = self.score_hrt(pos_triples)  # (B,)

                # Negative sampling (TCNS or uniform)
                neg_triples = self.sampler.sample(pos_triples.cpu()).to(device)  # (B, num_neg, 3)
                B, K, _ = neg_triples.shape
                neg_flat = neg_triples.reshape(-1, 3)
                neg_scores = self.score_hrt(neg_flat).reshape(B, K)

                # Base loss
                base_loss = self._nssa_loss(pos_scores, neg_scores)

                # ORE losses — computed every ore_frequency batches to reduce overhead
                if self.ont_reg is not None and (batch_num % ore_frequency == 0):
                    entity_emb = self.scorer.get_entity_embeddings()
                    ore_loss, ore_detail = self.ont_reg(
                        entity_emb,
                        lambda_cluster=self.lambda_cluster,
                        lambda_hierarchy=self.lambda_hierarchy,
                    )
                else:
                    ore_loss = torch.tensor(0.0, device=device)
                    ore_detail = {"cluster": 0.0, "hierarchy": 0.0}

                # SAPC loss
                if self.path_loss_fn is not None:
                    path_loss = self.path_loss_fn(
                        lambda h, r, t: self.score(h, r, t),
                        batch_size=min(32, self.batch_size),
                        device=device,
                    )
                    path_loss = self.lambda_path * path_loss
                else:
                    path_loss = torch.tensor(0.0, device=device)

                # Joint loss
                total_loss = base_loss + ore_loss + path_loss

                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_losses["base"] += base_loss.item()
                epoch_losses["cluster"] += ore_detail["cluster"]
                epoch_losses["hierarchy"] += ore_detail["hierarchy"]
                epoch_losses["path"] += path_loss.item()
                n_batches += 1
                batch_num += 1

            # Record epoch averages
            for key in epoch_losses:
                epoch_losses[key] /= max(n_batches, 1)
            history["base_loss"].append(epoch_losses["base"])
            history["cluster_loss"].append(epoch_losses["cluster"])
            history["hierarchy_loss"].append(epoch_losses["hierarchy"])
            history["path_loss"].append(epoch_losses["path"])
            history["total_loss"].append(epoch_loss / max(n_batches, 1))

            if verbose and (epoch + 1) % 50 == 0:
                elapsed = time.time() - t0
                p_typed = self.sampler.p_typed
                log.info(
                    f"Epoch {epoch+1}/{self.total_epochs} | "
                    f"Loss: {epoch_loss/n_batches:.4f} "
                    f"(base={epoch_losses['base']:.4f} "
                    f"cluster={epoch_losses['cluster']:.4f} "
                    f"path={epoch_losses['path']:.4f}) | "
                    f"p_typed={p_typed:.2f} | {elapsed:.0f}s"
                )

        train_time = time.time() - t0
        log.info(f"Training complete in {train_time:.1f}s")

        return {
            "training_time": round(train_time, 1),
            "final_loss": history["total_loss"][-1],
            "history": history,
        }

    def get_score_fn(self, device: str = "cpu"):
        """Return a score function for evaluation."""
        self.eval()

        def score_fn(heads, relations, tails):
            h = heads.to(device)
            r = relations.to(device)
            t = tails.to(device)
            with torch.no_grad():
                return self.score(h, r, t)

        return score_fn


def train_ontocom(dataset, device: str = "cpu",
                  embedding_dim: int = 128,
                  epochs: int = 500,
                  use_tcns: bool = True,
                  use_ore: bool = True,
                  use_sapc: bool = True,
                  ore_frequency: int = 20,
                  **kwargs) -> tuple:
    """
    Convenience function to train OntoCom on a dataset.

    Args:
        ore_frequency: compute ORE loss every N batches (default 20).
                       Increase for large datasets to reduce per-epoch overhead.

    Returns:
        (model, training_result)
    """
    is_small = len(dataset.train) < 2000

    # Default values (can be overridden by kwargs)
    defaults = dict(
        num_neg=64 if is_small else 256,
        batch_size=64 if is_small else 256,
        lr=5e-4 if is_small else 1e-3,
        path_min_support=3 if is_small else 10,
        lambda_cluster=0.001 if is_small else 0.01,
        lambda_hierarchy=0.0005 if is_small else 0.005,
        lambda_path=0.05 if is_small else 0.1,
        p_typed_start=0.3 if is_small else 0.5,
        p_typed_end=0.7 if is_small else 0.9,
    )
    # kwargs override defaults
    defaults.update(kwargs)

    model = OntoCom(
        dataset=dataset,
        embedding_dim=embedding_dim,
        total_epochs=epochs,
        use_tcns=use_tcns,
        use_ore=use_ore,
        use_sapc=use_sapc,
        **defaults,
    )

    result = model.train_model(device=device, ore_frequency=ore_frequency)
    return model, result
