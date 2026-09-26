"""
Phase 0B: Retrieval-Augmented Selective Prediction — Feasibility Check

Hypothesis: the CER retrieval margin — too weak to rerank — is strong enough
to separate confidently-correct from confidently-wrong KGC predictions,
enabling reliable abstention under the Open-World Assumption.

Metric: Area Under Risk-Coverage curve (AURC; lower = better) and coverage
at fixed risk thresholds.

Compares three confidence signals on FB15k-237 (500 test queries, bidirectional):
  1. KGE-only:  margin = score(top1) - score(top2) from RotatE
  2. Retrieval-only:  log s(top1) - log s_pool  (CER margin at top-1)
  3. Fused:  z(KGE-margin) + α · z(retrieval-margin), α swept on validation

Gate: fused AURC improves over KGE-only AURC by ≥10% (relative).

Reuses cached RotatE + FAISS index built by cer_prototype.py.
Run from /tmp to avoid datasets/ import conflict:
    cd /tmp && PYTHONPATH="" python path/to/confidence_proto.py
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

RESULTS_DIR = Path(_PROJECT_DIR) / "results" / "confidence_proto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(_PROJECT_DIR) / "cache" / "fb15k237"


# ══════════════════════════════════════════════════════════
#  Selective prediction metrics
# ══════════════════════════════════════════════════════════

def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray):
    """
    Compute the risk-coverage curve.

    Sort by confidence desc; for each prefix of length k, risk = 1 - mean(correct[:k]).
    Returns (coverages, risks) in order of increasing coverage.
    """
    assert len(confidence) == len(correct)
    n = len(correct)
    order = np.argsort(-confidence, kind="stable")
    c_sorted = correct[order].astype(float)

    cum = np.cumsum(c_sorted)
    ks = np.arange(1, n + 1)
    coverages = ks / n
    risks = 1.0 - cum / ks
    return coverages, risks


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area Under Risk-Coverage curve; lower is better."""
    cov, risk = risk_coverage_curve(confidence, correct)
    # integrate risk over coverage
    return float(np.trapz(risk, cov))


def coverage_at_risk(confidence, correct, target_risk: float) -> float:
    """Maximum coverage such that risk <= target_risk. 0 if not achievable."""
    cov, risk = risk_coverage_curve(confidence, correct)
    mask = risk <= target_risk
    if not mask.any():
        return 0.0
    return float(cov[mask].max())


def optimal_aurc(correct: np.ndarray) -> float:
    """Lower bound on AURC: perfect ranking of errors to the end."""
    c_sorted = np.sort(correct.astype(float))[::-1]  # all correct first
    n = len(c_sorted)
    cum = np.cumsum(c_sorted)
    ks = np.arange(1, n + 1)
    risks = 1.0 - cum / ks
    coverages = ks / n
    return float(np.trapz(risks, coverages))


def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-8)


# ══════════════════════════════════════════════════════════
#  Main feasibility check
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Phase 0B: Retrieval-Augmented Selective Prediction")
    log.info("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ─── Load cached RotatE + dataset ───
    log.info("\n--- Loading cached RotatE + FB15k-237 ---")
    from pykeen.datasets import FB15k237
    from pykeen.models import RotatE
    ds = FB15k237()
    ckpt = CACHE_DIR / "rotate_fb15k237_proto.pt"
    if not ckpt.exists():
        log.error(f"Missing checkpoint {ckpt}. Run cer_prototype.py first.")
        sys.exit(1)
    model = RotatE(triples_factory=ds.training, embedding_dim=64).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # ─── Load descriptions + FAISS index ───
    log.info("\n--- Loading descriptions + FAISS ---")
    desc_file = CACHE_DIR / "entity_descriptions.json"
    if not desc_file.exists():
        log.error(f"Missing {desc_file}. Run cer_prototype.py first.")
        sys.exit(1)
    desc_data = json.loads(desc_file.read_text(encoding="utf-8"))
    names_by_eid = desc_data["names"]
    id_to_name = {int(k): v for k, v in names_by_eid.items()}

    sys.path = [p for p in sys.path if "radaer_knowledge" not in p]
    from sentence_transformers import SentenceTransformer
    import faiss
    sys.path = _original_path

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    index_file = CACHE_DIR / "desc_faiss.index"
    if not index_file.exists():
        log.error(f"Missing FAISS index {index_file}. Run cer_prototype.py first.")
        sys.exit(1)
    faiss_index = faiss.read_index(str(index_file))
    log.info(f"FAISS index: {faiss_index.ntotal} vectors")

    # ─── Prep ───
    entity_to_id = dict(ds.training.entity_to_id)
    relation_to_id = dict(ds.training.relation_to_id)
    num_entities = len(entity_to_id)

    rel_pred = {}
    for name, rid in relation_to_id.items():
        parts = name.strip("/").split("/")
        pred = (parts[-1] if parts else name).split(".")[-1]
        rel_pred[rid] = pred.replace("_", " ").strip()

    # Type pools
    tail_pool: dict[int, set] = defaultdict(set)
    head_pool: dict[int, set] = defaultdict(set)
    for h, r, t in ds.training.mapped_triples.tolist():
        tail_pool[r].add(t)
        head_pool[r].add(h)
    tail_pool = {r: list(s) for r, s in tail_pool.items()}
    head_pool = {r: list(s) for r, s in head_pool.items()}

    true_set = set()
    for split in [ds.training, ds.validation, ds.testing]:
        for h, r, t in split.mapped_triples.tolist():
            true_set.add((h, r, t))

    # ─── Sample queries ───
    # Use validation for tuning α, test for evaluation.
    N_VAL = 200
    N_TEST = 500
    val_triples = ds.validation.mapped_triples
    test_triples = ds.testing.mapped_triples

    val_idx = random.sample(range(len(val_triples)), min(N_VAL, len(val_triples)))
    test_idx = random.sample(range(len(test_triples)), min(N_TEST, len(test_triples)))
    val_sampled = val_triples[val_idx]
    test_sampled = test_triples[test_idx]

    POOL_SAMPLE = 80
    TOP_M_PASSAGES = 3

    def process_queries(triples, label: str):
        """For each query, compute KGE top-1 correctness, KGE margin, retrieval margin."""
        log.info(f"\nProcessing {len(triples)} {label} queries ({len(triples)*2} directions)")
        all_entities = torch.arange(num_entities, device=device)

        correct = []        # list of 0/1
        kge_margin = []     # top1 - top2
        kge_top1_score = [] # max score
        ret_margin = []     # log s(top1) - log s_pool

        for idx, (h, r, t) in enumerate(triples.tolist()):
            if idx % 50 == 0:
                log.info(f"  [{label}] {idx}/{len(triples)}")

            for direction in ("tail", "head"):
                if direction == "tail":
                    heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
                    rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
                    trps = torch.stack([heads, rels, all_entities], dim=1)
                    scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
                    target = t
                    # filter
                    for e in range(num_entities):
                        if e != t and (h, r, e) in true_set:
                            scores[e] = -np.inf
                    anchor_name = id_to_name.get(h, f"e{h}")
                    pool = tail_pool.get(r, [])
                    def make_text(cand, an=anchor_name, pr=rel_pred.get(r, f"r{r}")):
                        return f"{an} {pr} {id_to_name.get(cand, f'e{cand}')}."
                else:
                    tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
                    rels2 = torch.full((num_entities,), r, dtype=torch.long, device=device)
                    trps = torch.stack([all_entities, rels2, tails], dim=1)
                    scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
                    target = h
                    for e in range(num_entities):
                        if e != h and (e, r, t) in true_set:
                            scores[e] = -np.inf
                    anchor_name = id_to_name.get(t, f"e{t}")
                    pool = head_pool.get(r, [])
                    def make_text(cand, an=anchor_name, pr=rel_pred.get(r, f"r{r}")):
                        return f"{id_to_name.get(cand, f'e{cand}')} {pr} {an}."

                # KGE signals
                order = np.argsort(-scores)
                top1 = int(order[0])
                top2_score = float(scores[order[1]]) if len(order) >= 2 else -np.inf
                is_correct = int(top1 == target)
                correct.append(is_correct)
                kge_margin.append(float(scores[top1]) - top2_score)
                kge_top1_score.append(float(scores[top1]))

                # Retrieval signal: s(top1) vs type-pool baseline
                pool_sample = random.sample(pool, min(POOL_SAMPLE, len(pool))) if pool else []
                if len(pool_sample) < 5:
                    pool_sample = order[:30].tolist()

                texts = [make_text(top1)] + [make_text(c) for c in pool_sample]
                q_vecs = encoder.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=128)
                q_vecs = np.asarray(q_vecs, dtype="float32")
                sims, _ = faiss_index.search(q_vecs, TOP_M_PASSAGES)
                s_all = sims.max(axis=1)
                s_top1 = float(np.clip(s_all[0], 1e-6, None))
                s_bar = float(np.clip(s_all[1:], 1e-6, None).mean())
                ret_margin.append(np.log(s_top1) - np.log(max(s_bar, 1e-6)))

        return {
            "correct": np.array(correct, dtype=int),
            "kge_margin": np.array(kge_margin, dtype=float),
            "kge_top1_score": np.array(kge_top1_score, dtype=float),
            "ret_margin": np.array(ret_margin, dtype=float),
        }

    t0 = time.time()
    val_data = process_queries(val_sampled, "val")
    log.info(f"Val done: Hits@1 = {val_data['correct'].mean():.4f}")

    test_data = process_queries(test_sampled, "test")
    log.info(f"Test done: Hits@1 = {test_data['correct'].mean():.4f}")
    log.info(f"Total processing time: {time.time() - t0:.0f}s")

    # ─── Tune α on validation ───
    log.info("\n--- Tuning α on validation ---")
    v_kge = zscore(val_data["kge_margin"])
    v_ret = zscore(val_data["ret_margin"])
    v_correct = val_data["correct"]

    best_alpha, best_val_aurc = 0.0, aurc(v_kge, v_correct)
    log.info(f"  Val AURC (KGE-margin only): {best_val_aurc:.4f}")
    for a in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        fused = v_kge + a * v_ret
        ac = aurc(fused, v_correct)
        log.info(f"  α={a:.2f}: val AURC = {ac:.4f}")
        if ac < best_val_aurc:
            best_alpha, best_val_aurc = a, ac
    log.info(f"Best α on val: {best_alpha} (AURC {best_val_aurc:.4f})")

    # ─── Evaluate on test ───
    log.info("\n--- Evaluating on test ---")
    t_kge_margin = zscore(test_data["kge_margin"])
    t_ret = zscore(test_data["ret_margin"])
    t_correct = test_data["correct"]

    # KGE score absolute (not margin)
    t_kge_score = zscore(test_data["kge_top1_score"])

    signals = {
        "KGE-score": t_kge_score,
        "KGE-margin": t_kge_margin,
        "Retrieval-margin": t_ret,
        f"Fused(α={best_alpha})": t_kge_margin + best_alpha * t_ret,
    }

    opt = optimal_aurc(t_correct)
    log.info(f"Test Hits@1 = {t_correct.mean():.4f}")
    log.info(f"Optimal AURC (lower bound) = {opt:.4f}")

    results = {}
    for name, sig in signals.items():
        ac = aurc(sig, t_correct)
        c10 = coverage_at_risk(sig, t_correct, 0.10)
        c20 = coverage_at_risk(sig, t_correct, 0.20)
        c30 = coverage_at_risk(sig, t_correct, 0.30)
        results[name] = {
            "AURC": round(ac, 4),
            "cov@risk=0.10": round(c10, 4),
            "cov@risk=0.20": round(c20, 4),
            "cov@risk=0.30": round(c30, 4),
        }
        log.info(f"  {name:25s}  AURC={ac:.4f}  c@0.1={c10:.3f}  c@0.2={c20:.3f}  c@0.3={c30:.3f}")

    # Save raw data + results
    np.savez(RESULTS_DIR / "raw_signals.npz",
             v_correct=val_data["correct"],
             v_kge_margin=val_data["kge_margin"],
             v_ret_margin=val_data["ret_margin"],
             t_correct=test_data["correct"],
             t_kge_margin=test_data["kge_margin"],
             t_ret_margin=test_data["ret_margin"],
             t_kge_top1_score=test_data["kge_top1_score"])
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_alpha": best_alpha,
            "n_val_directions": len(val_data["correct"]),
            "n_test_directions": len(test_data["correct"]),
            "test_hits_at_1": float(t_correct.mean()),
            "optimal_aurc": opt,
            "results": results,
        }, f, indent=2)

    # ─── Verdict ───
    aurc_kge = results["KGE-margin"]["AURC"]
    aurc_fused = results[f"Fused(α={best_alpha})"]["AURC"]
    rel_improve = (aurc_kge - aurc_fused) / aurc_kge * 100

    print("\n" + "=" * 60)
    print("  Phase 0B Results Summary")
    print("=" * 60)
    print(f"  Baseline AURC (KGE-margin):   {aurc_kge:.4f}")
    print(f"  Retrieval-margin AURC:        {results['Retrieval-margin']['AURC']:.4f}")
    print(f"  Fused AURC (α={best_alpha}):           {aurc_fused:.4f}")
    print(f"  Optimal AURC (lower bound):   {opt:.4f}")
    print(f"  Relative AURC improvement:    {rel_improve:+.1f}%")
    print()

    if rel_improve >= 10:
        print("  [OK] CLEAR SIGNAL (>=10% AURC improvement)")
        print("  -> Proceed to Phase 1: full selective prediction pipeline")
    elif rel_improve >= 5:
        print("  [WARN] MODEST SIGNAL (5-10% AURC improvement)")
        print("  -> Worth exploring further; tune encoder/pool/template")
    elif rel_improve >= 0:
        print("  [WEAK] RETRIEVAL NOT HELPFUL (0-5% AURC improvement)")
        print("  -> Retrieval adds little over KGE confidence alone")
    else:
        print("  [FAIL] RETRIEVAL HURTS (negative AURC improvement)")
        print("  -> Retrieval signal is not predictive of correctness")

    print("=" * 60)


if __name__ == "__main__":
    main()
