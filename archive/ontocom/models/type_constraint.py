"""
Type-Constrained Negative Sampling (TCNS) and Ontology-Regularized Embedding (ORE).

TCNS: For each relation, sample negative entities only from the correct type set
      (domain types for head corruption, range types for tail corruption).
      Uses curriculum scheduling: gradually increases type-constrained ratio.

ORE:  Two regularization losses that shape embedding geometry:
      L_cluster:   entities of same type cluster together
      L_hierarchy: parent type centroid = mean of child centroids
"""

import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class TypeIndex:
    """
    Index mapping relations to their domain/range entity types,
    and types to their entity sets.
    """

    def __init__(self, entity_type: dict[int, str],
                 type_to_entities: dict[str, list[int]],
                 relation_domain: dict[int, set[str]],
                 relation_range: dict[int, set[str]],
                 type_hierarchy: Optional[dict[str, list[str]]] = None):
        self.entity_type = entity_type
        self.type_to_entities = type_to_entities
        self.relation_domain = relation_domain
        self.relation_range = relation_range
        self.type_hierarchy = type_hierarchy or {}

        # Pre-compute entity lists per relation for fast sampling
        self._domain_entities: dict[int, list[int]] = {}
        self._range_entities: dict[int, list[int]] = {}

        for rid, types in relation_domain.items():
            entities = []
            for t in types:
                entities.extend(type_to_entities.get(t, []))
            self._domain_entities[rid] = entities if entities else list(entity_type.keys())

        for rid, types in relation_range.items():
            entities = []
            for t in types:
                entities.extend(type_to_entities.get(t, []))
            self._range_entities[rid] = entities if entities else list(entity_type.keys())

        self.all_entities = list(entity_type.keys())
        self.num_entities = len(self.all_entities)

        # Pre-compute numpy arrays for vectorized sampling
        self._domain_arr: dict[int, np.ndarray] = {
            rid: np.array(ents, dtype=np.int64)
            for rid, ents in self._domain_entities.items()
        }
        self._range_arr: dict[int, np.ndarray] = {
            rid: np.array(ents, dtype=np.int64)
            for rid, ents in self._range_entities.items()
        }
        self._all_arr = np.arange(self.num_entities, dtype=np.int64)

    def get_domain_entities(self, relation_id: int) -> list[int]:
        """Get entities valid as heads for this relation."""
        return self._domain_entities.get(relation_id, self.all_entities)

    def get_range_entities(self, relation_id: int) -> list[int]:
        """Get entities valid as tails for this relation."""
        return self._range_entities.get(relation_id, self.all_entities)

    def get_domain_arr(self, relation_id: int) -> np.ndarray:
        return self._domain_arr.get(relation_id, self._all_arr)

    def get_range_arr(self, relation_id: int) -> np.ndarray:
        return self._range_arr.get(relation_id, self._all_arr)


class TypeConstrainedSampler:
    """
    Negative sampler with curriculum-scheduled type constraints.

    At epoch 0, p_typed fraction of negatives are type-constrained.
    By the final epoch, this increases to p_typed_max.
    Remaining negatives are sampled uniformly (to preserve some type boundary learning).
    """

    def __init__(self, type_index: TypeIndex,
                 num_neg: int = 64,
                 p_typed_start: float = 0.5,
                 p_typed_end: float = 0.9,
                 total_epochs: int = 500):
        self.type_index = type_index
        self.num_neg = num_neg
        self.p_typed_start = p_typed_start
        self.p_typed_end = p_typed_end
        self.total_epochs = total_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    @property
    def p_typed(self) -> float:
        """Current fraction of type-constrained negatives (curriculum)."""
        progress = min(self._current_epoch / max(self.total_epochs - 1, 1), 1.0)
        return self.p_typed_start + progress * (self.p_typed_end - self.p_typed_start)

    def _effective_p(self, pool_size: int) -> float:
        """
        Adaptively reduce typed fraction for small type pools.
        If the pool has fewer entities than num_neg/2, sampling with replacement
        produces many duplicates — reduce typed ratio to maintain diversity.
        """
        p = self.p_typed
        if pool_size <= 2:
            return 0.0  # Too small to be useful
        # Scale down when pool_size < num_neg (avoid excessive duplicates)
        if pool_size < self.num_neg:
            diversity_factor = min(pool_size / self.num_neg, 1.0)
            p = p * diversity_factor
        return p

    def sample(self, triples: torch.LongTensor) -> torch.LongTensor:
        """
        Vectorized negative sampling with type constraints.

        Processes each unique relation in the batch together, avoiding
        per-triple Python loops. Groups triples by relation, samples
        from type pools using numpy for speed, then reconstructs output.

        Args:
            triples: (B, 3) tensor of [h, r, t].

        Returns:
            neg_triples: (B, num_neg, 3) tensor of corrupted triples.
                         First half corrupt tail, second half corrupt head.
        """
        batch_size = triples.size(0)
        half = self.num_neg // 2
        num_ent = self.type_index.num_entities

        # Build output array using numpy (much faster than torch for indexing)
        triples_np = triples.numpy()
        neg_np = np.tile(triples_np[:, np.newaxis, :], (1, self.num_neg, 1))  # (B, K, 3)

        # Process each unique relation as a batch
        relations = triples_np[:, 1]
        unique_rels = np.unique(relations)

        for r in unique_rels:
            mask = relations == r
            indices = np.where(mask)[0]
            n = len(indices)

            # --- Corrupt tails ---
            range_arr = self.type_index.get_range_arr(int(r))
            p_tail = self._effective_p(len(range_arr))
            n_typed = int(half * p_tail)
            n_uniform = half - n_typed

            if n_typed > 0 and len(range_arr) > 1:
                # Sample from type pool for all n triples at once
                pool_idx = np.random.randint(0, len(range_arr), size=(n, n_typed))
                neg_np[np.ix_(indices, np.arange(n_typed), [2])] = range_arr[pool_idx][:, :, np.newaxis]

            if n_uniform > 0:
                uniform = np.random.randint(0, num_ent, size=(n, n_uniform))
                neg_np[np.ix_(indices, np.arange(n_typed, half), [2])] = uniform[:, :, np.newaxis]

            # --- Corrupt heads ---
            domain_arr = self.type_index.get_domain_arr(int(r))
            p_head = self._effective_p(len(domain_arr))
            n_typed_h = int(half * p_head)
            n_uniform_h = half - n_typed_h

            if n_typed_h > 0 and len(domain_arr) > 1:
                pool_idx_h = np.random.randint(0, len(domain_arr), size=(n, n_typed_h))
                neg_np[np.ix_(indices, np.arange(half, half + n_typed_h), [0])] = \
                    domain_arr[pool_idx_h][:, :, np.newaxis]

            if n_uniform_h > 0:
                uniform_h = np.random.randint(0, num_ent, size=(n, n_uniform_h))
                neg_np[np.ix_(indices, np.arange(half + n_typed_h, self.num_neg), [0])] = \
                    uniform_h[:, :, np.newaxis]

        return torch.from_numpy(neg_np)


class OntologyRegularizer(nn.Module):
    """
    Ontology-based regularization losses for entity embeddings.

    L_cluster:   Σ_T Σ_{e∈T} || embed(e) - centroid(T) ||²
    L_hierarchy: Σ_{parent,child} || centroid(child) - centroid(parent) ||² / radius(parent)²
    """

    def __init__(self, type_index: TypeIndex):
        super().__init__()
        self.type_index = type_index

        # Pre-compute type membership as tensors for efficient loss computation
        self.type_entity_ids: dict[str, torch.LongTensor] = {}
        for tname, eids in type_index.type_to_entities.items():
            if len(eids) >= 2:  # Need at least 2 entities for clustering
                self.type_entity_ids[tname] = torch.tensor(eids, dtype=torch.long)

    def cluster_loss(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        L_cluster: entities of same type should cluster around their centroid.
        """
        loss = torch.tensor(0.0, device=entity_embeddings.device)
        count = 0

        for tname, eids in self.type_entity_ids.items():
            eids_dev = eids.to(entity_embeddings.device)
            embeds = entity_embeddings[eids_dev]  # (n_type, dim)
            centroid = embeds.mean(dim=0, keepdim=True)  # (1, dim)
            diffs = embeds - centroid  # (n_type, dim)
            loss = loss + (diffs ** 2).sum()
            count += len(eids)

        return loss / max(count, 1)

    def hierarchy_loss(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        L_hierarchy: child centroids should be close to parent centroid,
        scaled by parent cluster radius.
        """
        hierarchy = self.type_index.type_hierarchy
        if not hierarchy:
            return torch.tensor(0.0, device=entity_embeddings.device)

        loss = torch.tensor(0.0, device=entity_embeddings.device)
        count = 0

        for parent, children in hierarchy.items():
            if parent not in self.type_entity_ids:
                continue

            parent_ids = self.type_entity_ids[parent].to(entity_embeddings.device)
            parent_embeds = entity_embeddings[parent_ids]
            parent_centroid = parent_embeds.mean(dim=0)
            parent_radius_sq = ((parent_embeds - parent_centroid.unsqueeze(0)) ** 2).sum() / len(parent_ids)
            parent_radius_sq = parent_radius_sq.clamp(min=1e-6)

            for child in children:
                if child not in self.type_entity_ids:
                    continue
                child_ids = self.type_entity_ids[child].to(entity_embeddings.device)
                child_embeds = entity_embeddings[child_ids]
                child_centroid = child_embeds.mean(dim=0)

                dist_sq = ((child_centroid - parent_centroid) ** 2).sum()
                loss = loss + dist_sq / parent_radius_sq
                count += 1

        return loss / max(count, 1)

    def forward(self, entity_embeddings: torch.Tensor,
                lambda_cluster: float = 0.01,
                lambda_hierarchy: float = 0.005) -> tuple[torch.Tensor, dict]:
        """
        Compute combined ontology regularization loss.

        Returns:
            (total_loss, {"cluster": cluster_val, "hierarchy": hierarchy_val})
        """
        l_cluster = self.cluster_loss(entity_embeddings)
        l_hierarchy = self.hierarchy_loss(entity_embeddings)
        total = lambda_cluster * l_cluster + lambda_hierarchy * l_hierarchy
        return total, {
            "cluster": l_cluster.item(),
            "hierarchy": l_hierarchy.item(),
        }
