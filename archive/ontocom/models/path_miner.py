"""
Structure-Aware Path Completion (SAPC) — path template mining.

Automatically discovers frequent 2-hop paths in the training KG:
   (h, r1, m) + (m, r2, t) => (h, r_target, t)

These path templates provide auxiliary training signal: the model
learns to score (h, r_target, t) consistently with the 2-hop path
evidence, without needing explicit path-based inference at test time.
"""

import logging
from collections import defaultdict

import torch

log = logging.getLogger(__name__)


def mine_path_templates(
    train_triples: torch.LongTensor,
    min_support: int = 5,
    max_templates: int = 50,
) -> list[dict]:
    """
    Mine frequent 2-hop path templates from training triples.

    For each triple (h, r_target, t), check if there exists an intermediary m
    such that (h, r1, m) and (m, r2, t) are both in the training set.
    Count co-occurrence of (r1, r2) -> r_target patterns.

    Args:
        train_triples: (N, 3) tensor of [h, r, t].
        min_support: minimum number of instances for a path template.
        max_templates: maximum number of templates to return.

    Returns:
        List of dicts: {r1, r2, r_target, support, instances}
    """
    # Build adjacency: head -> [(rel, tail)]
    head_to_rt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    # Build reverse adjacency: tail -> [(rel, head)]
    tail_to_rh: dict[int, list[tuple[int, int]]] = defaultdict(list)

    triples_list = train_triples.tolist()
    triple_set = set()

    for h, r, t in triples_list:
        head_to_rt[h].append((r, t))
        tail_to_rh[t].append((r, h))
        triple_set.add((h, r, t))

    # Count path patterns: (r1, r2) -> r_target -> count
    pattern_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    pattern_instances: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)

    for h, r_target, t in triples_list:
        # Find intermediaries: entities m such that (h, r1, m) and (m, r2, t) exist
        for r1, m in head_to_rt[h]:
            if m == h or m == t:
                continue
            for r2, t2 in head_to_rt[m]:
                if t2 == t and r1 != r_target and r2 != r_target:
                    key = (r1, r2, r_target)
                    pattern_counts[key] += 1
                    if len(pattern_instances[key]) < 20:  # Keep some examples
                        pattern_instances[key].append((h, m, t))

    # Filter by support and sort
    templates = []
    for (r1, r2, r_target), count in pattern_counts.items():
        if count >= min_support:
            templates.append({
                "r1": r1,
                "r2": r2,
                "r_target": r_target,
                "support": count,
                "instances": pattern_instances[(r1, r2, r_target)],
            })

    templates.sort(key=lambda x: -x["support"])
    templates = templates[:max_templates]

    log.info(f"Mined {len(templates)} path templates (min_support={min_support})")
    for tmpl in templates[:5]:
        log.info(f"  ({tmpl['r1']}, {tmpl['r2']}) => {tmpl['r_target']}  "
                 f"support={tmpl['support']}")

    return templates


class PathAuxLoss(torch.nn.Module):
    """
    Auxiliary loss for SAPC: for each path template (r1, r2) => r_target,
    encourage score(h, r_target, t) to be high when path evidence exists.

    L_path = Σ_templates Σ_{(h,m,t) ∈ instances}
             max(0, γ - score(h, r_target, t))

    This is a margin-based loss that pushes the target relation score above
    a threshold γ for triples with supporting path evidence.
    """

    def __init__(self, templates: list[dict], margin: float = 3.0):
        super().__init__()
        self.templates = templates
        self.margin = margin

        # Pre-compute training pairs: list of (h, r_target, t) from path instances
        self.path_triples: list[tuple[int, int, int]] = []
        for tmpl in templates:
            r_target = tmpl["r_target"]
            for h, m, t in tmpl["instances"]:
                self.path_triples.append((h, r_target, t))

        log.info(f"SAPC: {len(self.path_triples)} path-derived training triples "
                 f"from {len(templates)} templates")

    def sample_batch(self, batch_size: int) -> torch.LongTensor:
        """Sample a batch of path-derived triples for auxiliary training."""
        if not self.path_triples:
            return torch.zeros(0, 3, dtype=torch.long)

        import random
        indices = random.choices(range(len(self.path_triples)), k=batch_size)
        batch = [self.path_triples[i] for i in indices]
        return torch.tensor(batch, dtype=torch.long)

    def forward(self, score_fn, batch_size: int = 32, device: str = "cpu") -> torch.Tensor:
        """
        Compute path auxiliary loss.

        Args:
            score_fn: callable(h, r, t) -> score (higher = more plausible)
            batch_size: number of path triples to sample per call
            device: torch device

        Returns:
            Scalar loss tensor.
        """
        if not self.path_triples:
            return torch.tensor(0.0, device=device)

        batch = self.sample_batch(batch_size).to(device)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        scores = score_fn(h, r, t)
        # Margin loss: push scores above margin
        loss = torch.relu(self.margin - scores).mean()
        return loss
