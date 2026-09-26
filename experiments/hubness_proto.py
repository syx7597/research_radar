"""
Phase 0C: Hubness-Correction Feasibility Check

Hypothesis: KGE raw scores exhibit strong hubness (a small set of entities
dominates top-K predictions across queries, regardless of correctness).
Subtracting a log-hubness penalty at inference should improve filtered MRR
without any retraining.

Method:
  1. Estimate entity hubness on TRAINING triples:
     For each (r, direction), rank the KGE's full score over all entities for
     every training query. Count how often each entity appears in top-K.
  2. At test time, apply penalty:
     score_final(h, r, t) = score_KGE(h, r, t) - lambda * log(1 + h_r^dir(t))
  3. Sweep lambda on validation; report test MRR with best lambda.

Gate: >=2% relative filtered MRR improvement over baseline.

Run from /tmp:
    cd /tmp && PYTHONPATH="" python path/to/hubness_proto.py
"""

import sys
import os
import json
import time
import random
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
_original_path = sys.path.copy()

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(_PROJECT_DIR) / "results" / "hubness_proto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(_PROJECT_DIR) / "cache" / "fb15k237"


# ══════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════

def compute_mrr(ranks: list[int]) -> dict:
    r = np.asarray(ranks, dtype=float)
    return {
        "mrr": float((1.0 / r).mean()),
        "hits_at_1": float((r <= 1).mean()),
        "hits_at_3": float((r <= 3).mean()),
        "hits_at_10": float((r <= 10).mean()),
        "n": int(len(r)),
    }


# ══════════════════════════════════════════════════════════
#  Hubness estimation
# ══════════════════════════════════════════════════════════

def estimate_hubness(model, ds, device, top_k_for_hubness: int = 10,
                     max_train_queries: int = 2000,
                     relation_conditional: bool = True):
    """
    For each (r, direction), count how often each entity appears in the
    model's top-K predictions across a sample of training queries.

    Returns:
        h_tail[(r, t)] — times t appears in top-K over (h, r, ?) for this r
        h_head[(r, h)] — times h appears in top-K over (?, r, t) for this r
        Also global variants (not relation-conditional).
    """
    num_entities = model.num_entities
    all_entities = torch.arange(num_entities, device=device)
    train_triples = ds.training.mapped_triples.tolist()

    # Subsample to keep estimation cheap
    if len(train_triples) > max_train_queries:
        rng = random.Random(42)
        train_triples = rng.sample(train_triples, max_train_queries)

    hub_tail: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(num_entities, dtype=np.int64))
    hub_head: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(num_entities, dtype=np.int64))
    hub_tail_global = np.zeros(num_entities, dtype=np.int64)
    hub_head_global = np.zeros(num_entities, dtype=np.int64)

    t0 = time.time()
    log.info(f"Estimating hubness on {len(train_triples)} training queries (top-{top_k_for_hubness})")
    for idx, (h, r, t) in enumerate(train_triples):
        if idx % 500 == 0:
            log.info(f"  Hubness {idx}/{len(train_triples)}  ({time.time()-t0:.0f}s)")

        # Tail direction: (h, r, ?)
        heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
        rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
        trps = torch.stack([heads, rels, all_entities], dim=1)
        scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
        top_ids = np.argpartition(-scores, top_k_for_hubness)[:top_k_for_hubness]
        hub_tail[r][top_ids] += 1
        hub_tail_global[top_ids] += 1

        # Head direction: (?, r, t)
        tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
        rels2 = torch.full((num_entities,), r, dtype=torch.long, device=device)
        trps = torch.stack([all_entities, rels2, tails], dim=1)
        scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
        top_ids = np.argpartition(-scores, top_k_for_hubness)[:top_k_for_hubness]
        hub_head[r][top_ids] += 1
        hub_head_global[top_ids] += 1

    log.info(f"Hubness estimation done in {time.time()-t0:.0f}s")

    # Report distribution
    gt = hub_tail_global
    log.info(f"Global tail hubness stats: max={gt.max()}, "
             f"p99={np.percentile(gt[gt>0], 99):.0f}, "
             f"median={np.median(gt[gt>0]):.0f}, "
             f"nonzero entities={int((gt > 0).sum())}/{num_entities}")

    return {
        "tail": dict(hub_tail),
        "head": dict(hub_head),
        "tail_global": hub_tail_global,
        "head_global": hub_head_global,
    }


# ══════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════

def evaluate_with_hubness(model, ds, hubness, lambdas, n_queries, device,
                          split: str = "test", use_relation_conditional: bool = True):
    """
    Evaluate filtered MRR for each lambda in lambdas. Returns dict[lambda -> metrics].
    Also computes baseline (lambda=0).
    """
    num_entities = model.num_entities
    all_entities = torch.arange(num_entities, device=device)

    triples_src = {"test": ds.testing, "valid": ds.validation}[split].mapped_triples
    rng = random.Random(42 if split == "test" else 123)
    indices = rng.sample(range(len(triples_src)), min(n_queries, len(triples_src)))
    triples = triples_src[indices]

    # True triple set for filtering
    true_set = set()
    for spl in [ds.training, ds.validation, ds.testing]:
        for h, r, t in spl.mapped_triples.tolist():
            true_set.add((h, r, t))

    # For each query-direction we'll cache the raw filtered scores and compute
    # corrected ranks for all lambda values in one pass.
    log.info(f"[{split}] Scoring {len(triples)} queries × 2 directions = {len(triples)*2}")

    results = {lam: {"ranks": []} for lam in lambdas}

    for idx, (h, r, t) in enumerate(triples.tolist()):
        if idx % 100 == 0:
            log.info(f"  [{split}] {idx}/{len(triples)}")

        # --- Tail: (h, r, ?) ---
        heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
        rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
        trps = torch.stack([heads, rels, all_entities], dim=1)
        scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
        # Filter known triples
        for e in range(num_entities):
            if e != t and (h, r, e) in true_set:
                scores[e] = -np.inf

        # Get hubness vector for this (r, direction)
        if use_relation_conditional and r in hubness["tail"]:
            h_vec = hubness["tail"][r]
        else:
            h_vec = hubness["tail_global"]
        penalty = np.log1p(h_vec.astype(np.float64))

        for lam in lambdas:
            adj = scores - lam * penalty
            rank = int((adj >= adj[t]).sum())
            results[lam]["ranks"].append(rank)

        # --- Head: (?, r, t) ---
        tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
        rels2 = torch.full((num_entities,), r, dtype=torch.long, device=device)
        trps = torch.stack([all_entities, rels2, tails], dim=1)
        scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
        for e in range(num_entities):
            if e != h and (e, r, t) in true_set:
                scores[e] = -np.inf

        if use_relation_conditional and r in hubness["head"]:
            h_vec = hubness["head"][r]
        else:
            h_vec = hubness["head_global"]
        penalty = np.log1p(h_vec.astype(np.float64))

        for lam in lambdas:
            adj = scores - lam * penalty
            rank = int((adj >= adj[h]).sum())
            results[lam]["ranks"].append(rank)

    # Compute metrics
    out = {}
    for lam, d in results.items():
        m = compute_mrr(d["ranks"])
        out[lam] = m
    return out


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Phase 0C: Hubness-Correction Feasibility")
    log.info("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Load cached RotatE + FB15k-237
    log.info("\n--- Loading cached RotatE on FB15k-237 ---")
    from pykeen.datasets import FB15k237
    from pykeen.models import RotatE
    ds = FB15k237()
    ckpt = CACHE_DIR / "rotate_fb15k237_proto.pt"
    if not ckpt.exists():
        log.error(f"Missing checkpoint {ckpt}")
        sys.exit(1)
    model = RotatE(triples_factory=ds.training, embedding_dim=64).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Estimate hubness from training triples
    log.info("\n--- Estimating hubness ---")
    hubness = estimate_hubness(model, ds, device,
                               top_k_for_hubness=10,
                               max_train_queries=2000,
                               relation_conditional=True)

    # Cache hubness tensors
    np.savez(RESULTS_DIR / "hubness.npz",
             tail_global=hubness["tail_global"],
             head_global=hubness["head_global"])

    # Show top hub entities for illustration
    top_hub_idx = np.argsort(-hubness["tail_global"])[:10]
    log.info(f"Top-10 tail hub entities (by global top-10 frequency across training queries):")
    for i, eid in enumerate(top_hub_idx):
        log.info(f"  #{i+1}: entity_id={eid}  hub_count={hubness['tail_global'][eid]}")

    # Sweep lambda on validation
    log.info("\n--- Sweep lambda on validation ---")
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    val_results = evaluate_with_hubness(model, ds, hubness, lambdas,
                                        n_queries=200, device=device,
                                        split="valid",
                                        use_relation_conditional=True)
    log.info("Validation results (relation-conditional hubness):")
    for lam in lambdas:
        m = val_results[lam]
        log.info(f"  lambda={lam:.2f}  MRR={m['mrr']:.4f}  H@1={m['hits_at_1']:.4f}  "
                 f"H@10={m['hits_at_10']:.4f}")

    baseline_val = val_results[0.0]["mrr"]
    best_lam = max(lambdas, key=lambda l: val_results[l]["mrr"])
    best_val_mrr = val_results[best_lam]["mrr"]
    log.info(f"\nBest lambda on val: {best_lam}  (MRR {best_val_mrr:.4f} vs baseline {baseline_val:.4f})")

    # Evaluate on test with best lambda (and a few neighbors for robustness)
    log.info("\n--- Evaluating on test ---")
    test_lambdas = sorted(set([0.0, best_lam, max(0.0, best_lam - 0.1), best_lam + 0.1]))
    test_results = evaluate_with_hubness(model, ds, hubness, test_lambdas,
                                         n_queries=500, device=device,
                                         split="test",
                                         use_relation_conditional=True)

    # Also test global (not relation-conditional) for ablation
    log.info("\n--- Ablation: global (not relation-conditional) hubness ---")
    test_results_global = evaluate_with_hubness(model, ds, hubness, [0.0, best_lam],
                                                n_queries=500, device=device,
                                                split="test",
                                                use_relation_conditional=False)

    log.info("Test results (relation-conditional hubness):")
    for lam in test_lambdas:
        m = test_results[lam]
        log.info(f"  lambda={lam:.2f}  MRR={m['mrr']:.4f}  H@1={m['hits_at_1']:.4f}  "
                 f"H@10={m['hits_at_10']:.4f}")
    log.info("Test results (global hubness, for ablation):")
    for lam in [0.0, best_lam]:
        m = test_results_global[lam]
        log.info(f"  lambda={lam:.2f}  MRR={m['mrr']:.4f}  H@1={m['hits_at_1']:.4f}  "
                 f"H@10={m['hits_at_10']:.4f}")

    # Save
    out = {
        "best_lambda_on_val": best_lam,
        "val": {str(k): v for k, v in val_results.items()},
        "test_rel_cond": {str(k): v for k, v in test_results.items()},
        "test_global": {str(k): v for k, v in test_results_global.items()},
    }
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Verdict
    baseline_test = test_results[0.0]["mrr"]
    corrected_test = test_results[best_lam]["mrr"]
    rel_improvement = (corrected_test - baseline_test) / max(baseline_test, 1e-6) * 100

    print("\n" + "=" * 60)
    print("  Phase 0C Hubness-Correction Results")
    print("=" * 60)
    print(f"  Baseline MRR (RotatE, no correction):   {baseline_test:.4f}")
    print(f"  Corrected MRR (lambda={best_lam}, rel-cond):  {corrected_test:.4f}")
    global_test = test_results_global[best_lam]["mrr"]
    print(f"  Corrected MRR (lambda={best_lam}, global):    {global_test:.4f}")
    print(f"  Relative improvement: {rel_improvement:+.2f}%")
    print()

    if rel_improvement >= 2.0:
        print(f"  [OK] CLEAR SIGNAL (>={rel_improvement:.1f}% MRR improvement)")
        print("  -> Proceed: full hubness-correction implementation across backbones/datasets")
    elif rel_improvement >= 0.5:
        print(f"  [WARN] MODEST SIGNAL ({rel_improvement:.1f}% MRR improvement)")
        print("  -> Worth exploring further; try different top-K, more hubness queries, or per-type pools")
    elif rel_improvement >= 0.0:
        print(f"  [WEAK] MARGINAL ({rel_improvement:.1f}% MRR improvement)")
        print("  -> Hubness correction does not meaningfully help")
    else:
        print(f"  [FAIL] NEGATIVE ({rel_improvement:.1f}% MRR change)")
        print("  -> Hubness penalty hurts; reconsider approach")

    print("=" * 60)


if __name__ == "__main__":
    main()
