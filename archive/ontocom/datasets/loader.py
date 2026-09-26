"""
Unified dataset loader for OntoCom experiments.
Loads FB15k-237, WN18RR (via PyKEEN) and RadarKG (from local files).
Each dataset is returned as a standardized KGDataset object with type info.

Usage:
    from datasets.loader import load_dataset
    ds = load_dataset("fb15k237")  # or "wn18rr" or "radarkg"
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class KGDataset:
    """Standardized KG dataset with type information."""
    name: str
    # Core data: each is (N, 3) tensor of [head_id, relation_id, tail_id]
    train: torch.LongTensor
    valid: torch.LongTensor
    test: torch.LongTensor
    # Mappings
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    id_to_entity: dict[int, str] = field(default_factory=dict)
    id_to_relation: dict[int, str] = field(default_factory=dict)
    # Type information
    entity_type: dict[int, str] = field(default_factory=dict)       # entity_id -> type_name
    type_to_entities: dict[str, list[int]] = field(default_factory=dict)  # type -> [entity_ids]
    relation_domain: dict[int, set[str]] = field(default_factory=dict)    # rel_id -> {head_types}
    relation_range: dict[int, set[str]] = field(default_factory=dict)     # rel_id -> {tail_types}
    # Type hierarchy (parent -> [children])
    type_hierarchy: dict[str, list[str]] = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)

    @property
    def num_triples(self) -> int:
        return len(self.train) + len(self.valid) + len(self.test)

    @property
    def num_types(self) -> int:
        return len(self.type_to_entities)

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  Entities:  {self.num_entities}",
            f"  Relations: {self.num_relations}",
            f"  Types:     {self.num_types}",
            f"  Train:     {len(self.train)}",
            f"  Valid:     {len(self.valid)}",
            f"  Test:      {len(self.test)}",
            f"  Total:     {self.num_triples}",
        ]
        if self.type_hierarchy:
            lines.append(f"  Hierarchy: {self.type_hierarchy}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
#  FB15k-237 loader
# ═══════════════════════════════════════════════════════

def _infer_fb15k237_types(entity_to_id: dict, relation_to_id: dict,
                          train_triples: torch.LongTensor) -> tuple[dict, dict]:
    """
    Infer entity types from Freebase relation structure.
    Relations have form: /domain/type/property — we use the domain/type prefix.
    For each entity, the most frequent type from its head/tail occurrences wins.
    """
    id_to_rel = {v: k for k, v in relation_to_id.items()}
    entity_type_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for h, r, t in train_triples.tolist():
        rel_name = id_to_rel[r]
        parts = rel_name.strip("/").split("/")
        if len(parts) >= 2:
            head_type = f"{parts[0]}/{parts[1]}"
            entity_type_votes[h][head_type] += 1
            # Try to infer tail type from the property name or the range
            if len(parts) >= 3:
                # For relations like /people/person/nationality, tail is often a location
                # We use a simple heuristic: the head type is most reliable
                pass

    entity_type = {}
    for eid, votes in entity_type_votes.items():
        entity_type[eid] = max(votes, key=votes.get)

    # Assign "unknown" to entities without type info
    for eid in range(len(entity_to_id)):
        if eid not in entity_type:
            entity_type[eid] = "unknown"

    # Build type_to_entities index
    type_to_entities: dict[str, list[int]] = defaultdict(list)
    for eid, etype in entity_type.items():
        type_to_entities[etype].append(eid)

    return entity_type, dict(type_to_entities)


def _compute_relation_types(train_triples: torch.LongTensor,
                            entity_type: dict[int, str]) -> tuple[dict, dict]:
    """Compute domain (head types) and range (tail types) for each relation."""
    relation_domain: dict[int, set[str]] = defaultdict(set)
    relation_range: dict[int, set[str]] = defaultdict(set)
    for h, r, t in train_triples.tolist():
        relation_domain[r].add(entity_type.get(h, "unknown"))
        relation_range[r].add(entity_type.get(t, "unknown"))
    return dict(relation_domain), dict(relation_range)


def load_fb15k237() -> KGDataset:
    """Load FB15k-237 via PyKEEN with inferred type information."""
    from pykeen.datasets import FB15k237
    ds = FB15k237()

    entity_to_id = dict(ds.training.entity_to_id)
    relation_to_id = dict(ds.training.relation_to_id)
    train = ds.training.mapped_triples
    valid = ds.validation.mapped_triples if ds.validation else torch.zeros(0, 3, dtype=torch.long)
    test = ds.testing.mapped_triples if ds.testing else torch.zeros(0, 3, dtype=torch.long)

    entity_type, type_to_entities = _infer_fb15k237_types(entity_to_id, relation_to_id, train)
    relation_domain, relation_range = _compute_relation_types(train, entity_type)

    return KGDataset(
        name="FB15k-237",
        train=train, valid=valid, test=test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity={v: k for k, v in entity_to_id.items()},
        id_to_relation={v: k for k, v in relation_to_id.items()},
        entity_type=entity_type,
        type_to_entities=type_to_entities,
        relation_domain=relation_domain,
        relation_range=relation_range,
    )


# ═══════════════════════════════════════════════════════
#  WN18RR loader
# ═══════════════════════════════════════════════════════

def _infer_wn18rr_types(entity_to_id: dict) -> tuple[dict, dict]:
    """
    Infer WN18RR entity types from WordNet lexname categories.
    Entity names are synset offset IDs (e.g., '02084071'). We look up each
    offset in WordNet and use its lexname (e.g., 'noun.animal', 'verb.motion')
    as the entity type. This gives ~45 meaningful semantic categories.
    """
    import warnings
    warnings.filterwarnings("ignore", message="No WordNet synset found")

    try:
        from nltk.corpus import wordnet as wn
        use_wordnet = True
    except ImportError:
        log.warning("nltk not available; WN18RR types will be 'unknown'")
        use_wordnet = False

    entity_type = {}
    for name, eid in entity_to_id.items():
        if not use_wordnet:
            entity_type[eid] = "unknown"
            continue
        offset = int(name)
        matched = False
        for pos in ['n', 'v', 'a', 'r', 's']:
            try:
                syn = wn.synset_from_pos_and_offset(pos, offset)
                entity_type[eid] = syn.lexname()
                matched = True
                break
            except Exception:
                continue
        if not matched:
            entity_type[eid] = "unknown"

    type_to_entities: dict[str, list[int]] = defaultdict(list)
    for eid, etype in entity_type.items():
        type_to_entities[etype].append(eid)

    return entity_type, dict(type_to_entities)


def load_wn18rr() -> KGDataset:
    """Load WN18RR via PyKEEN with inferred type information."""
    from pykeen.datasets import WN18RR
    ds = WN18RR()

    entity_to_id = dict(ds.training.entity_to_id)
    relation_to_id = dict(ds.training.relation_to_id)
    train = ds.training.mapped_triples
    valid = ds.validation.mapped_triples if ds.validation else torch.zeros(0, 3, dtype=torch.long)
    test = ds.testing.mapped_triples if ds.testing else torch.zeros(0, 3, dtype=torch.long)

    entity_type, type_to_entities = _infer_wn18rr_types(entity_to_id)
    relation_domain, relation_range = _compute_relation_types(train, entity_type)

    return KGDataset(
        name="WN18RR",
        train=train, valid=valid, test=test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity={v: k for k, v in entity_to_id.items()},
        id_to_relation={v: k for k, v in relation_to_id.items()},
        entity_type=entity_type,
        type_to_entities=type_to_entities,
        relation_domain=relation_domain,
        relation_range=relation_range,
    )


# ═══════════════════════════════════════════════════════
#  RadarKG loader
# ═══════════════════════════════════════════════════════

def load_radarkg(triples_path: str = "graphrag_index/merged_triples.json",
                 split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                 seed: int = 42) -> KGDataset:
    """
    Load RadarKG from merged_triples.json with explicit type information.
    Creates train/valid/test splits.
    """
    path = Path(triples_path)
    if not path.exists():
        raise FileNotFoundError(f"RadarKG triples not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_triples = json.load(f)

    # Build entity and relation vocabularies
    entity_set: set[str] = set()
    relation_set: set[str] = set()
    entity_type_raw: dict[str, str] = {}

    for t in raw_triples:
        h, r, tail = t["head"], t["relation"], t["tail"]
        entity_set.add(h)
        entity_set.add(tail)
        relation_set.add(r)
        # Record type info from the triple metadata
        if t.get("head_type"):
            entity_type_raw[h] = t["head_type"]
        if t.get("tail_type"):
            entity_type_raw[tail] = t["tail_type"]

    entity_to_id = {e: i for i, e in enumerate(sorted(entity_set))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relation_set))}

    # Convert to tensor
    mapped = []
    for t in raw_triples:
        h_id = entity_to_id[t["head"]]
        r_id = relation_to_id[t["relation"]]
        t_id = entity_to_id[t["tail"]]
        mapped.append([h_id, r_id, t_id])

    all_triples = torch.tensor(mapped, dtype=torch.long)

    # Shuffle and split
    rng = random.Random(seed)
    indices = list(range(len(all_triples)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * split_ratios[0])
    n_valid = int(n * split_ratios[1])

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    train = all_triples[train_idx]
    valid = all_triples[valid_idx]
    test = all_triples[test_idx]

    # Entity types (from explicit metadata)
    entity_type: dict[int, str] = {}
    for name, eid in entity_to_id.items():
        entity_type[eid] = entity_type_raw.get(name, "unknown")

    type_to_entities: dict[str, list[int]] = defaultdict(list)
    for eid, etype in entity_type.items():
        type_to_entities[etype].append(eid)

    relation_domain, relation_range = _compute_relation_types(train, entity_type)

    # RadarKG type hierarchy
    type_hierarchy = {
        "Platform": ["NavalVessel", "AircraftPlatform", "GroundPlatform"],
        "Weapon":   ["Missile", "Bomb", "Torpedo"],
    }

    return KGDataset(
        name="RadarKG",
        train=train, valid=valid, test=test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity={v: k for k, v in entity_to_id.items()},
        id_to_relation={v: k for k, v in relation_to_id.items()},
        entity_type=entity_type,
        type_to_entities=dict(type_to_entities),
        relation_domain=relation_domain,
        relation_range=relation_range,
        type_hierarchy=type_hierarchy,
    )


# ═══════════════════════════════════════════════════════
#  Unified entry point
# ═══════════════════════════════════════════════════════

DATASET_REGISTRY = {
    "fb15k237": load_fb15k237,
    "fb15k-237": load_fb15k237,
    "wn18rr": load_wn18rr,
    "radarkg": load_radarkg,
}


def load_dataset(name: str, **kwargs) -> KGDataset:
    """Load a dataset by name. See DATASET_REGISTRY for available names."""
    key = name.lower().replace(" ", "").replace("_", "")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key](**kwargs)


# ═══════════════════════════════════════════════════════
#  CLI: verify all datasets load correctly
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for name in ["radarkg", "fb15k237", "wn18rr"]:
        print(f"\nLoading {name}...")
        try:
            ds = load_dataset(name)
            print(ds.summary())
            # Show type distribution
            from collections import Counter
            type_counts = Counter(ds.entity_type.values())
            print(f"  Type distribution (top 10):")
            for t, c in type_counts.most_common(10):
                print(f"    {t}: {c}")
            # Show relation domain/range
            print(f"  Relation type constraints (first 5):")
            for rid in list(ds.relation_domain.keys())[:5]:
                rname = ds.id_to_relation.get(rid, str(rid))
                domain = ds.relation_domain.get(rid, set())
                range_ = ds.relation_range.get(rid, set())
                dom_str = ",".join(sorted(domain)[:3])
                rng_str = ",".join(sorted(range_)[:3])
                print(f"    {rname[:50]}: {dom_str} -> {rng_str}")
        except Exception as e:
            print(f"  ERROR: {e}")
