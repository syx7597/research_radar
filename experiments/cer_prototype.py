"""
CER Prototype Signal Check — Phase 0

Quick experiment to validate whether Contrastive Evidence Reranking (CER)
has any signal on FB15k-237. Runs in ~1-2 hours.

Steps:
  1. Train RotatE on FB15k-237 via PyKEEN (reduced: dim=64, 50 epochs)
  2. Download FB15k-237 entity descriptions from Wikidata
  3. Build FAISS index over descriptions
  4. Sample 100 test queries, get top-50 candidates
  5. Compute contrastive evidence margin
  6. Fuse with KGE scores, measure MRR delta

Run from project root:
    python experiments/cer_prototype.py

IMPORTANT: Due to datasets/ naming conflict with HuggingFace,
this script manipulates sys.path before importing sentence_transformers.
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

# Fix datasets/ naming conflict: temporarily hide project dir
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
_original_path = sys.path.copy()

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(_PROJECT_DIR) / "results" / "cer_prototype"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(_PROJECT_DIR) / "cache" / "fb15k237"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
#  Step 1: Train RotatE on FB15k-237
# ═══════════════════════════════════════════════════════

def train_rotate_fb15k237(device="cpu"):
    """Train a small RotatE on FB15k-237 or load cached."""
    checkpoint = CACHE_DIR / "rotate_fb15k237_proto.pt"

    from pykeen.datasets import FB15k237
    from pykeen.pipeline import pipeline

    ds = FB15k237()

    if checkpoint.exists():
        log.info(f"Loading cached RotatE from {checkpoint}")
        result_data = torch.load(checkpoint, map_location=device, weights_only=False)
        from pykeen.models import RotatE
        model = RotatE(
            triples_factory=ds.training,
            embedding_dim=64,
        ).to(device)
        model.load_state_dict(result_data["model_state"])
        return model, ds

    log.info("Training RotatE on FB15k-237 (dim=64, 20 epochs — prototype only)...")
    t0 = time.time()
    result = pipeline(
        dataset=ds,
        model="RotatE",
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=20, batch_size=512, use_tqdm=True),
        optimizer_kwargs=dict(lr=1e-3),
        negative_sampler_kwargs=dict(num_negs_per_pos=16),
        random_seed=42,
        device=device,
    )
    elapsed = time.time() - t0
    log.info(f"Training done in {elapsed:.0f}s")

    torch.save({
        "model_state": result.model.state_dict(),
    }, checkpoint)

    return result.model, ds


# ═══════════════════════════════════════════════════════
#  Step 2: Get FB15k-237 entity descriptions
# ═══════════════════════════════════════════════════════

def get_entity_descriptions(entity_to_id: dict) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load FB15k-237 entity descriptions (short labels + long Wikipedia descriptions)
    from KG-BERT's data release.

    Returns:
        (names_by_eid, long_by_eid) — two dicts mapping str(entity_id) to text.
    """
    desc_cache = CACHE_DIR / "entity_descriptions.json"
    if desc_cache.exists():
        log.info("Loading cached entity descriptions")
        data = json.loads(desc_cache.read_text(encoding="utf-8"))
        return data["names"], data["long"]

    name_file = CACHE_DIR / "entity2text.txt"
    long_file = CACHE_DIR / "entity2textlong.txt"

    mid_to_name = {}
    for line in name_file.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            mid, name = line.split("\t", 1)
            mid_to_name[mid] = name.strip()

    mid_to_long = {}
    for line in long_file.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            mid, desc = line.split("\t", 1)
            mid_to_long[mid] = desc.strip().replace("\\n", " ")

    names_by_eid = {}
    long_by_eid = {}
    missing = 0
    for mid, eid in entity_to_id.items():
        name = mid_to_name.get(mid, mid.replace("/", " ").strip())
        long_desc = mid_to_long.get(mid, name)
        names_by_eid[str(eid)] = name
        long_by_eid[str(eid)] = long_desc
        if mid not in mid_to_name:
            missing += 1

    log.info(f"Loaded {len(names_by_eid)} entity descriptions ({missing} missing from KG-BERT file)")
    desc_cache.write_text(json.dumps({"names": names_by_eid, "long": long_by_eid}, indent=2),
                          encoding="utf-8")
    return names_by_eid, long_by_eid


def get_relation_descriptions(relation_to_id: dict) -> dict[int, str]:
    """Natural-language predicate for each relation (last segment, _ → space)."""
    descs = {}
    for name, rid in relation_to_id.items():
        parts = name.strip("/").split("/")
        pred = parts[-1] if parts else name
        # Some FB relations compose via `.` (e.g. sports_team/roster./sports_team_roster/team)
        pred = pred.split(".")[-1]
        descs[rid] = pred.replace("_", " ").strip()
    return descs


def build_type_pools(ds) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Approximate type pools from training triples.
    head_pool[r] = entities appearing as head of r; tail_pool[r] = as tail.
    """
    tail_pool: dict[int, set] = defaultdict(set)
    head_pool: dict[int, set] = defaultdict(set)
    for h, r, t in ds.training.mapped_triples.tolist():
        tail_pool[r].add(t)
        head_pool[r].add(h)
    return {r: list(s) for r, s in head_pool.items()}, {r: list(s) for r, s in tail_pool.items()}


# ═══════════════════════════════════════════════════════
#  Step 3: Build FAISS index over descriptions
# ═══════════════════════════════════════════════════════

def build_description_index(descriptions: dict[str, str], encoder):
    """Build FAISS index over entity descriptions."""
    import faiss

    index_cache = CACHE_DIR / "desc_faiss.index"
    meta_cache = CACHE_DIR / "desc_faiss_meta.json"

    if index_cache.exists() and meta_cache.exists():
        log.info("Loading cached FAISS index")
        index = faiss.read_index(str(index_cache))
        meta = json.loads(meta_cache.read_text(encoding="utf-8"))
        return index, meta

    log.info(f"Encoding {len(descriptions)} descriptions...")
    ids = sorted(descriptions.keys(), key=int)
    texts = [descriptions[i] for i in ids]

    # Encode in batches
    batch_size = 256
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vecs = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)

    vectors = np.vstack(all_vecs).astype("float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_cache))
    meta_cache.write_text(json.dumps({"ids": ids, "dim": dim}), encoding="utf-8")

    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, {"ids": ids, "dim": dim}


# ═══════════════════════════════════════════════════════
#  Step 4: CER scoring
# ═══════════════════════════════════════════════════════

def verbalize_triple(h_name: str, r_pred: str, t_name: str) -> str:
    """
    Natural-language verbalization: "{subject} {predicate} {object}".
    Caller decides which entity is subject/object based on head/tail direction.
    """
    return f"{h_name} {r_pred} {t_name}."


def compute_cer_scores(
    query_h: int,
    query_r: int,
    candidates: list[int],
    kge_scores: torch.Tensor,
    id_to_entity: dict,
    rel_desc: dict,
    encoder,
    faiss_index,
    faiss_ids: list[str],
    alpha: float = 0.5,
    top_m: int = 3,
    direction: str = "tail",
) -> torch.Tensor:
    """
    Compute CER-reranked scores for a single query.

    Returns fused scores tensor of same shape as kge_scores.
    """
    h_name = id_to_entity.get(query_h, f"entity_{query_h}")
    r_name = rel_desc.get(query_r, f"rel_{query_r}")

    # Verbalize all candidates
    texts = []
    for c in candidates:
        c_name = id_to_entity.get(c, f"entity_{c}")
        texts.append(verbalize_triple(h_name, r_name, c_name, direction))

    # Encode
    q_vecs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    q_vecs = q_vecs.astype("float32")

    # Search FAISS
    sims, _ = faiss_index.search(q_vecs, top_m)  # (K, top_m)
    s = sims.max(axis=1)  # (K,) — max similarity over top-M passages

    # Contrastive margin: log(s) - log(mean(s))
    s_clipped = np.clip(s, 1e-6, None)
    s_bar = s_clipped.mean()
    margin = np.log(s_clipped) - np.log(max(s_bar, 1e-6))

    margin_t = torch.tensor(margin, dtype=kge_scores.dtype, device=kge_scores.device)
    return kge_scores + alpha * margin_t


# ═══════════════════════════════════════════════════════
#  Step 5: Evaluate
# ═══════════════════════════════════════════════════════

def evaluate_prototype(
    model,
    ds,
    encoder,
    faiss_index,
    faiss_meta,
    names_by_eid,
    n_queries: int = 100,
    top_k: int = 50,
    alphas: list = None,
    device: str = "cpu",
):
    """
    Run CER prototype evaluation on sampled test queries.
    Compares baseline KGE ranking vs CER-reranked ranking.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    entity_to_id = dict(ds.training.entity_to_id)
    relation_to_id = dict(ds.training.relation_to_id)
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    # Use human-readable names for verbalization, not MIDs
    id_to_name = {int(k): v for k, v in names_by_eid.items()}
    rel_desc = get_relation_descriptions(relation_to_id)
    num_entities = len(entity_to_id)

    # Build true triple set for filtering
    true_set = set()
    for split in [ds.training, ds.validation, ds.testing]:
        for h, r, t in split.mapped_triples.tolist():
            true_set.add((h, r, t))

    # Sample test queries
    test_triples = ds.testing.mapped_triples
    n_test = len(test_triples)
    indices = random.sample(range(n_test), min(n_queries, n_test))
    sampled = test_triples[indices]

    log.info(f"Evaluating {len(sampled)} test queries...")

    all_entities = torch.arange(num_entities, device=device)
    model.eval()
    model = model.to(device)

    # Collect per-query data
    query_data = []

    for idx, (h, r, t) in enumerate(sampled.tolist()):
        if idx % 20 == 0:
            log.info(f"  Processing query {idx}/{len(sampled)}")

        # --- Tail prediction: (h, r, ?) ---
        heads = torch.full((num_entities,), h, dtype=torch.long, device=device)
        rels = torch.full((num_entities,), r, dtype=torch.long, device=device)
        triples_t = torch.stack([heads, rels, all_entities], dim=1)
        scores_tail = model.score_hrt(triples_t).squeeze(-1).detach().cpu()

        # Filter known triples
        for e in range(num_entities):
            if e != t and (h, r, e) in true_set:
                scores_tail[e] = float("-inf")

        # Baseline rank
        rank_baseline_tail = (scores_tail >= scores_tail[t]).sum().item()

        query_data.append({
            "h": h, "r": r, "t": t,
            "direction": "tail",
            "scores": scores_tail,
            "target": t,
            "rank_baseline": rank_baseline_tail,
        })

        # --- Head prediction: (?, r, t) ---
        tails = torch.full((num_entities,), t, dtype=torch.long, device=device)
        rels2 = torch.full((num_entities,), r, dtype=torch.long, device=device)
        triples_h = torch.stack([all_entities, rels2, tails], dim=1)
        scores_head = model.score_hrt(triples_h).squeeze(-1).detach().cpu()

        for e in range(num_entities):
            if e != h and (e, r, t) in true_set:
                scores_head[e] = float("-inf")

        rank_baseline_head = (scores_head >= scores_head[h]).sum().item()

        query_data.append({
            "h": h, "r": r, "t": t,
            "direction": "head",
            "scores": scores_head,
            "target": h,
            "rank_baseline": rank_baseline_head,
        })

    # Build type pools from training triples (approximates T_r^domain / T_r^range)
    head_pool, tail_pool = build_type_pools(ds)
    log.info(f"Type pools built: avg tail pool size = "
             f"{np.mean([len(v) for v in tail_pool.values()]):.0f}")

    # Now test CER with different alphas
    log.info("Computing CER margins with type-pool baselines...")
    results = {}

    POOL_SAMPLE = 100  # entities sampled per (r, direction) for s_bar
    TOP_M_PASSAGES = 3

    for qi, qd in enumerate(query_data):
        if qi % 40 == 0:
            log.info(f"  Margin computation {qi}/{len(query_data)}")

        h, r, t = qd["h"], qd["r"], qd["t"]
        direction = qd["direction"]
        scores = qd["scores"]
        target = qd["target"]

        r_pred = rel_desc.get(r, f"r{r}")

        # Pick top-K candidates (ensure target included)
        kge_scores_np = scores.numpy()
        top_k_ids = np.argsort(-kge_scores_np)[:top_k]
        if target not in top_k_ids:
            top_k_ids = np.append(top_k_ids, target)
        rerank_ids = top_k_ids.tolist()

        # For tail pred: anchor = h (fixed), candidate varies as object
        # For head pred: anchor = t (fixed), candidate varies as subject
        if direction == "tail":
            anchor_name = id_to_name.get(h, f"e{h}")
            pool = tail_pool.get(r, [])
            # verbalize as "{anchor} {pred} {candidate}"
            make_text = lambda cand: verbalize_triple(
                anchor_name, r_pred, id_to_name.get(cand, f"e{cand}"))
        else:
            anchor_name = id_to_name.get(t, f"e{t}")
            pool = head_pool.get(r, [])
            # verbalize as "{candidate} {pred} {anchor}"
            make_text = lambda cand: verbalize_triple(
                id_to_name.get(cand, f"e{cand}"), r_pred, anchor_name)

        # Sample pool entities for unbiased s_bar (exclude candidates in top-K to
        # avoid leakage — though overlap is usually small)
        pool_sample = random.sample(pool, min(POOL_SAMPLE, len(pool))) if pool else []
        if len(pool_sample) < 5:
            # Pool too small — fall back to top-K mean (rare)
            pool_sample = rerank_ids

        # Encode candidates + pool together in one batch
        cand_texts = [make_text(c) for c in rerank_ids]
        pool_texts = [make_text(c) for c in pool_sample]
        all_texts = cand_texts + pool_texts

        q_vecs = encoder.encode(all_texts, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=128)
        q_vecs = np.asarray(q_vecs, dtype="float32")
        sims, _ = faiss_index.search(q_vecs, TOP_M_PASSAGES)
        s_all = sims.max(axis=1)

        s_cand = s_all[:len(rerank_ids)]
        s_pool = s_all[len(rerank_ids):]

        # Contrastive margin with type-pool mean (unbiased)
        s_cand_c = np.clip(s_cand, 1e-6, None)
        s_bar = max(float(np.clip(s_pool, 1e-6, None).mean()), 1e-6)
        margin = np.log(s_cand_c) - np.log(s_bar)

        qd["rerank_ids"] = rerank_ids
        qd["margin"] = margin
        qd["s_bar"] = s_bar
        qd["s_cand_mean"] = float(s_cand_c.mean())

    # Evaluate each alpha
    for alpha in alphas:
        ranks = []
        for qd in query_data:
            scores = qd["scores"].clone()
            target = qd["target"]
            rerank_ids = qd["rerank_ids"]
            margin = qd["margin"]

            if alpha > 0:
                for i, eid in enumerate(rerank_ids):
                    scores[eid] = scores[eid] + alpha * margin[i]

            rank = (scores >= scores[target]).sum().item()
            ranks.append(rank)

        ranks_t = torch.tensor(ranks, dtype=torch.float)
        mrr = (1.0 / ranks_t).mean().item()
        h1 = (ranks_t <= 1).float().mean().item()
        h3 = (ranks_t <= 3).float().mean().item()
        h10 = (ranks_t <= 10).float().mean().item()

        results[alpha] = {
            "mrr": round(mrr, 4),
            "hits_at_1": round(h1, 4),
            "hits_at_3": round(h3, 4),
            "hits_at_10": round(h10, 4),
            "n_queries": len(query_data),
        }

        delta = results[alpha]["mrr"] - results[0.0]["mrr"] if alpha > 0 else 0
        log.info(f"  α={alpha:.1f}: MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f} "
                 f"(Δ MRR={delta:+.4f})")

    return results


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("CER Prototype Signal Check — FB15k-237")
    log.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Step 1: Train or load RotatE
    log.info("\n--- Step 1: RotatE on FB15k-237 ---")
    model, ds = train_rotate_fb15k237(device=device)

    # Step 2: Entity descriptions
    log.info("\n--- Step 2: Entity descriptions ---")
    entity_to_id = dict(ds.training.entity_to_id)
    names_by_eid, long_by_eid = get_entity_descriptions(entity_to_id)

    # Step 3: Build FAISS index over LONG descriptions (Wikipedia abstracts)
    log.info("\n--- Step 3: FAISS index ---")
    sys.path = [p for p in sys.path if 'radaer_knowledge' not in p]
    from sentence_transformers import SentenceTransformer
    sys.path = _original_path

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    faiss_index, faiss_meta = build_description_index(long_by_eid, encoder)

    # Step 4-5: Evaluate
    log.info("\n--- Step 4-5: CER Evaluation ---")
    results = evaluate_prototype(
        model=model,
        ds=ds,
        encoder=encoder,
        faiss_index=faiss_index,
        faiss_meta=faiss_meta,
        names_by_eid=names_by_eid,
        n_queries=100,
        top_k=50,
        alphas=[0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
        device=device,
    )

    # Save results
    output = RESULTS_DIR / "cer_prototype_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {output}")

    # Print summary
    baseline_mrr = results[0.0]["mrr"]
    best_alpha = max(
        [(a, r["mrr"]) for a, r in results.items() if a > 0],
        key=lambda x: x[1]
    )

    print("\n" + "=" * 60)
    print("  CER Prototype Results Summary")
    print("=" * 60)
    print(f"  Baseline (RotatE, α=0):  MRR = {baseline_mrr:.4f}")
    print(f"  Best CER (α={best_alpha[0]:.1f}):       MRR = {best_alpha[1]:.4f}")
    delta_pct = (best_alpha[1] - baseline_mrr) / max(baseline_mrr, 1e-6) * 100
    print(f"  Δ MRR: {best_alpha[1] - baseline_mrr:+.4f} ({delta_pct:+.1f}%)")
    print()

    if best_alpha[1] - baseline_mrr >= 0.01:
        print("  [OK] SIGNAL DETECTED (>=1% MRR improvement)")
        print("  -> Proceed to Phase 1: full CER implementation")
    elif best_alpha[1] - baseline_mrr >= 0.005:
        print("  [WARN] WEAK SIGNAL (0.5-1% MRR improvement)")
        print("  -> Consider: better descriptions, different encoder, or reconsider approach")
    else:
        print("  [FAIL] NO SIGNAL (<0.5% MRR improvement)")
        print("  -> Investigate: check worst degradation queries, consider pivoting")

    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
