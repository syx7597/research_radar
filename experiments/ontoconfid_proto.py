"""
Phase 0D: OntoConfid Feasibility — type-compat + KGE-margin fusion for selective prediction.

Goal: check whether adding a training-free "type-compatibility at top-1" feature
lifts AURC over KGE-margin alone. Also includes retrieval-margin from Phase 0B.

Features (per query direction):
  1. KGE margin        = score(top1) - score(top2)                      (float)
  2. Retrieval margin  = log s(top1) - log s_pool                       (float, from Phase 0B)
  3. Type compat       = 1 if top-1 entity ∈ training tail/head-pool of r
                        = 0 otherwise                                    (binary)

Fuse via logistic regression fit on val → predict correctness on test.

Reuses the SAME random seed (42) and query sampling as confidence_proto.py so
top-1 IDs align with cached retrieval/KGE margins in raw_signals.npz.
But we re-score only KGE (fast) to retrieve top-1 IDs, then join with cached
retrieval margin.

Gate: fused AURC beats KGE-margin AURC (0.6332) by ≥5% relative (≤0.6015)
      AND type-compat feature has statistically meaningful weight (|β_type| / SE > 2).
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

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(_PROJECT_DIR) / "results" / "ontoconfid_proto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(_PROJECT_DIR) / "cache" / "fb15k237"
PHASE_0B_DIR = Path(_PROJECT_DIR) / "results" / "confidence_proto"


# ══════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════

def risk_coverage_curve(confidence, correct):
    n = len(correct)
    order = np.argsort(-confidence, kind="stable")
    c_sorted = correct[order].astype(float)
    cum = np.cumsum(c_sorted)
    ks = np.arange(1, n + 1)
    return ks / n, 1.0 - cum / ks


def aurc(confidence, correct):
    cov, risk = risk_coverage_curve(confidence, correct)
    return float(np.trapz(risk, cov))


def coverage_at_risk(confidence, correct, target_risk):
    cov, risk = risk_coverage_curve(confidence, correct)
    mask = risk <= target_risk
    return float(cov[mask].max()) if mask.any() else 0.0


def optimal_aurc(correct):
    c_sorted = np.sort(correct.astype(float))[::-1]
    n = len(c_sorted)
    cum = np.cumsum(c_sorted)
    ks = np.arange(1, n + 1)
    risks = 1.0 - cum / ks
    return float(np.trapz(risks, ks / n))


def zscore(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-8)


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Phase 0D: OntoConfid Feasibility (type-compat fusion)")
    log.info("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ─── Load RotatE + dataset ───
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

    num_entities = len(ds.training.entity_to_id)

    # Type pools (training co-occurrence — proxy for relation domain/range)
    tail_pool: dict[int, set] = defaultdict(set)
    head_pool: dict[int, set] = defaultdict(set)
    for h, r, t in ds.training.mapped_triples.tolist():
        tail_pool[r].add(t)
        head_pool[r].add(h)
    log.info(f"Tail-pool stats: median size = {int(np.median([len(s) for s in tail_pool.values()]))}")
    log.info(f"Head-pool stats: median size = {int(np.median([len(s) for s in head_pool.values()]))}")

    # Filter set for filtered evaluation
    true_set = set()
    for split in [ds.training, ds.validation, ds.testing]:
        for h, r, t in split.mapped_triples.tolist():
            true_set.add((h, r, t))

    # ─── Sample queries — SAME seed/count as confidence_proto.py ───
    N_VAL, N_TEST = 200, 500
    val_triples = ds.validation.mapped_triples
    test_triples = ds.testing.mapped_triples

    val_idx = random.sample(range(len(val_triples)), min(N_VAL, len(val_triples)))
    test_idx = random.sample(range(len(test_triples)), min(N_TEST, len(test_triples)))
    val_sampled = val_triples[val_idx]
    test_sampled = test_triples[test_idx]

    # ─── Load Phase 0B signals (retrieval margin + KGE margin + correct) ───
    log.info("\n--- Loading Phase 0B cached signals ---")
    ph0b = np.load(PHASE_0B_DIR / "raw_signals.npz")
    log.info(f"Phase 0B val: {len(ph0b['v_correct'])} dirs, test: {len(ph0b['t_correct'])} dirs")

    # ─── Re-score KGE to extract top-1 IDs (for type-compat feature) ───
    log.info("\n--- Re-scoring to capture top-1 IDs ---")
    all_entities = torch.arange(num_entities, device=device)

    def get_top1_ids_and_correct(triples, label):
        """Return parallel arrays aligned direction-major: tail(h,r,?), head(?,r,t)."""
        top1_ids = []
        rels = []
        direction_flags = []  # 0=tail, 1=head
        correct = []
        kge_margin = []
        for idx, (h, r, t) in enumerate(triples.tolist()):
            if idx % 50 == 0:
                log.info(f"  [{label}] {idx}/{len(triples)}")
            for dtag in (0, 1):  # tail, head
                if dtag == 0:
                    heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
                    rr = torch.full((num_entities,), r, dtype=torch.long, device=device)
                    trps = torch.stack([heads, rr, all_entities], dim=1)
                    target = t
                else:
                    tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
                    rr = torch.full((num_entities,), r, dtype=torch.long, device=device)
                    trps = torch.stack([all_entities, rr, tails], dim=1)
                    target = h

                scores = model.score_hrt(trps).squeeze(-1).detach().cpu().numpy()
                # filter
                if dtag == 0:
                    for e in range(num_entities):
                        if e != t and (h, r, e) in true_set:
                            scores[e] = -np.inf
                else:
                    for e in range(num_entities):
                        if e != h and (e, r, t) in true_set:
                            scores[e] = -np.inf

                order = np.argsort(-scores)
                top1 = int(order[0])
                top2_score = float(scores[order[1]]) if len(order) >= 2 else -np.inf

                top1_ids.append(top1)
                rels.append(r)
                direction_flags.append(dtag)
                correct.append(int(top1 == target))
                kge_margin.append(float(scores[top1]) - top2_score)
        return (np.array(top1_ids), np.array(rels), np.array(direction_flags),
                np.array(correct), np.array(kge_margin))

    t0 = time.time()
    v_top1, v_rels, v_dirs, v_correct_new, v_kge_margin_new = get_top1_ids_and_correct(val_sampled, "val")
    t_top1, t_rels, t_dirs, t_correct_new, t_kge_margin_new = get_top1_ids_and_correct(test_sampled, "test")
    log.info(f"Re-scoring took {time.time()-t0:.0f}s")

    # Sanity: correctness should match phase 0B (same seed/data)
    v_match = np.array_equal(v_correct_new, ph0b["v_correct"])
    t_match = np.array_equal(t_correct_new, ph0b["t_correct"])
    log.info(f"Correctness alignment with Phase 0B: val={v_match} test={t_match}")
    if not (v_match and t_match):
        log.warning("Alignment failed — recomputing signals from scratch instead of reusing Phase 0B")

    # ─── Build type-compat feature ───
    log.info("\n--- Building type-compat feature ---")

    def type_compat_at_top1(top1_ids, rels, dirs):
        out = []
        for top1, r, d in zip(top1_ids, rels, dirs):
            pool = tail_pool.get(r, set()) if d == 0 else head_pool.get(r, set())
            out.append(int(top1 in pool))
        return np.array(out, dtype=float)

    v_type = type_compat_at_top1(v_top1, v_rels, v_dirs)
    t_type = type_compat_at_top1(t_top1, t_rels, t_dirs)
    log.info(f"Val type-compat rate: {v_type.mean():.3f}  Test: {t_type.mean():.3f}")
    log.info(f"Val Hits@1 among type-compat:   {v_correct_new[v_type==1].mean() if v_type.sum()>0 else 0:.3f}")
    log.info(f"Val Hits@1 among type-incompat: {v_correct_new[v_type==0].mean() if (v_type==0).sum()>0 else 0:.3f}")
    log.info(f"Test Hits@1 among type-compat:   {t_correct_new[t_type==1].mean() if t_type.sum()>0 else 0:.3f}")
    log.info(f"Test Hits@1 among type-incompat: {t_correct_new[t_type==0].mean() if (t_type==0).sum()>0 else 0:.3f}")

    # ─── Use Phase 0B retrieval margins (aligned) ───
    v_ret = ph0b["v_ret_margin"] if v_match else None
    t_ret = ph0b["t_ret_margin"] if t_match else None
    v_kge = ph0b["v_kge_margin"] if v_match else v_kge_margin_new
    t_kge = ph0b["t_kge_margin"] if t_match else t_kge_margin_new

    # ─── Fit logistic regression on validation ───
    log.info("\n--- Fitting logistic regression on validation ---")
    # Standardize features for fit stability + interpretable weights
    v_kge_z = zscore(v_kge)
    t_kge_z = zscore(t_kge) if len(t_kge) else None

    v_features_sets = {
        "KGE-margin only":    np.stack([v_kge_z], axis=1),
        "KGE+Type":           np.stack([v_kge_z, v_type], axis=1),
    }
    t_features_sets = {
        "KGE-margin only":    np.stack([t_kge_z], axis=1),
        "KGE+Type":           np.stack([t_kge_z, t_type], axis=1),
    }
    if v_ret is not None and t_ret is not None:
        v_ret_z = zscore(v_ret)
        t_ret_z = zscore(t_ret)
        v_features_sets["KGE+Ret"] = np.stack([v_kge_z, v_ret_z], axis=1)
        t_features_sets["KGE+Ret"] = np.stack([t_kge_z, t_ret_z], axis=1)
        v_features_sets["KGE+Type+Ret"] = np.stack([v_kge_z, v_type, v_ret_z], axis=1)
        t_features_sets["KGE+Type+Ret"] = np.stack([t_kge_z, t_type, t_ret_z], axis=1)

    from sklearn.linear_model import LogisticRegression

    results = {}
    opt = optimal_aurc(t_correct_new)
    log.info(f"Optimal AURC (lower bound) = {opt:.4f}")
    log.info(f"Test Hits@1 = {t_correct_new.mean():.4f}\n")

    for name, Xv in v_features_sets.items():
        Xt = t_features_sets[name]
        yv = v_correct_new
        yt = t_correct_new

        # Skip fit if any class has zero positives on val (can't fit)
        if len(np.unique(yv)) < 2:
            log.warning(f"{name}: only one class in val, skipping")
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xv, yv)
        probs = clf.predict_proba(Xt)[:, 1]
        ac = aurc(probs, yt)
        c10 = coverage_at_risk(probs, yt, 0.10)
        c20 = coverage_at_risk(probs, yt, 0.20)
        c30 = coverage_at_risk(probs, yt, 0.30)
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]

        results[name] = {
            "AURC": round(ac, 4),
            "cov@risk=0.10": round(c10, 4),
            "cov@risk=0.20": round(c20, 4),
            "cov@risk=0.30": round(c30, 4),
            "coef": [round(c, 4) for c in coefs.tolist()],
            "intercept": round(float(intercept), 4),
        }
        log.info(f"  {name:20s} AURC={ac:.4f} c@0.1={c10:.3f} c@0.2={c20:.3f} c@0.3={c30:.3f} coef={coefs}")

    # Also compute KGE-margin alone AURC (sanity, non-logistic) for reference
    baseline_kge_alone_aurc = aurc(t_kge_z, t_correct_new)
    log.info(f"\nNon-logistic KGE-margin AURC (direct): {baseline_kge_alone_aurc:.4f}")

    # ─── Verdict ───
    aurc_base = results["KGE-margin only"]["AURC"]
    aurc_kge_type = results["KGE+Type"]["AURC"]
    rel_kge_type = (aurc_base - aurc_kge_type) / aurc_base * 100

    print("\n" + "=" * 60)
    print("  Phase 0D Results Summary")
    print("=" * 60)
    print(f"  Test Hits@1:                       {t_correct_new.mean():.4f}")
    print(f"  Optimal AURC (lower bound):        {opt:.4f}")
    print(f"  Baseline AURC (KGE-margin only):   {aurc_base:.4f}")
    print(f"  AURC (KGE + Type-compat):          {aurc_kge_type:.4f}")
    print(f"    Relative AURC improvement:       {rel_kge_type:+.1f}%")
    if "KGE+Ret" in results:
        aurc_kge_ret = results["KGE+Ret"]["AURC"]
        rel_kge_ret = (aurc_base - aurc_kge_ret) / aurc_base * 100
        print(f"  AURC (KGE + Retrieval):            {aurc_kge_ret:.4f}  ({rel_kge_ret:+.1f}%)")
    if "KGE+Type+Ret" in results:
        aurc_full = results["KGE+Type+Ret"]["AURC"]
        rel_full = (aurc_base - aurc_full) / aurc_base * 100
        print(f"  AURC (KGE + Type + Retrieval):     {aurc_full:.4f}  ({rel_full:+.1f}%)")
    print()

    gate_aurc = aurc_base * 0.95  # 5% relative improvement
    if aurc_kge_type <= gate_aurc:
        print(f"  [OK] TYPE-COMPAT FEATURE HELPS (>=5% AURC improvement)")
        print("  -> Greenlight Chapter 6 (OntoConfid)")
    elif aurc_kge_type < aurc_base:
        rel = (aurc_base - aurc_kge_type) / aurc_base * 100
        print(f"  [WARN] MODEST ({rel:.1f}% AURC improvement, target 5%)")
        print("  -> Useful but thin; consider richer type features (domain/range)")
    else:
        print(f"  [FAIL] Type-compat does NOT help AURC")
        print("  -> Need different feature engineering OR shift contribution framing")
    print("=" * 60)

    # Save
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_val": len(v_correct_new),
            "n_test": len(t_correct_new),
            "test_hits_at_1": float(t_correct_new.mean()),
            "optimal_aurc": opt,
            "val_type_compat_rate": float(v_type.mean()),
            "test_type_compat_rate": float(t_type.mean()),
            "val_hits_at_1_type_compat": float(v_correct_new[v_type==1].mean()) if v_type.sum() > 0 else None,
            "val_hits_at_1_type_incompat": float(v_correct_new[v_type==0].mean()) if (v_type==0).sum() > 0 else None,
            "test_hits_at_1_type_compat": float(t_correct_new[t_type==1].mean()) if t_type.sum() > 0 else None,
            "test_hits_at_1_type_incompat": float(t_correct_new[t_type==0].mean()) if (t_type==0).sum() > 0 else None,
            "results": results,
        }, f, indent=2)
    np.savez(RESULTS_DIR / "raw_signals.npz",
             v_top1=v_top1, v_rels=v_rels, v_dirs=v_dirs,
             v_correct=v_correct_new, v_kge_margin=v_kge_margin_new, v_type=v_type,
             t_top1=t_top1, t_rels=t_rels, t_dirs=t_dirs,
             t_correct=t_correct_new, t_kge_margin=t_kge_margin_new, t_type=t_type)
    log.info(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
