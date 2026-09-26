"""
Generate paper figures (PDF + PNG) under paper/figures/.

  fig1_pipeline.pdf       — pipeline architecture diagram
  fig2_main_results.pdf   — per-type accuracy bar chart, Baseline / RoG / Strategy
  fig3_ablation.pdf       — per-strategy ablation drop chart

Uses matplotlib only (3.10). All English labels to avoid Chinese font issues.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────
#  Fig 1: Pipeline architecture
# ────────────────────────────────────────────────────────────

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x, y, w, h, label, color="#e8f0ff", edge="#2c5aa0"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor=edge, linewidth=1.4)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9.5)

    def arrow(x1, y1, x2, y2, label=""):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=15, color='#444', lw=1.2)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, fontsize=8,
                    ha='center', color='#333')

    # Top row: Question
    box(0.3, 5.7, 1.7, 0.9, "Question\n(Chinese / English)", color="#fff5e6", edge="#cc7a00")
    arrow(2.0, 6.15, 2.7, 6.15)

    # Router (LLM)
    box(2.7, 5.7, 1.8, 0.9, "Question Router\n(LLM zero-shot)", color="#e8f0ff", edge="#2c5aa0")
    arrow(4.5, 6.15, 5.2, 6.15, "(qtype, strategy)")

    # Parser (LLM)
    box(5.2, 5.7, 1.8, 0.9, "Strategy Parser\n(LLM extracts args)", color="#e8f0ff", edge="#2c5aa0")
    arrow(6.1, 5.7, 6.1, 4.9)

    # Strategy executor (KG-only) — outer container without centered label
    container = FancyBboxPatch((2.0, 3.2), 8.2, 1.7, boxstyle="round,pad=0.05",
                                facecolor="#eef9ee", edgecolor="#2d8c2d", linewidth=1.4)
    ax.add_patch(container)
    ax.text(6.1, 4.78, "Strategy Executor  (KG queries, no LLM)",
            ha='center', va='center', fontsize=10, fontweight='bold', color="#1d6b1d")

    # Six strategy boxes inside the executor — slightly lower to leave header space
    sx = [2.2, 3.55, 4.9, 6.25, 7.6, 8.95]
    sw = 1.15
    sy = 3.35
    sh = 1.05
    strategies = [
        ("lookup", "BM25+Vec\n+Graph"),
        ("exhaust.", "type-pool\nfull set"),
        ("compl.", "absence\ncheck"),
        ("path", "relation\nchain walk"),
        ("c.join", "intersect\nN constr."),
        ("dual", "entity-pair\ncompare"),
    ]
    for x, (h, sub) in zip(sx, strategies):
        box(x, sy, sw, sh, f"{h}\n{sub}", color="#ffffff", edge="#2d8c2d")

    arrow(6.1, 3.2, 6.1, 2.7)

    # Renderer
    box(4.7, 1.8, 2.8, 0.9, "Context Renderer\n(strategy-aware)", color="#f5e8ff", edge="#7733aa")
    arrow(6.1, 1.8, 6.1, 1.0)

    # Answerer (LLM)
    box(4.7, 0.1, 2.8, 0.9, "Answerer (LLM)", color="#e8f0ff", edge="#2c5aa0")
    arrow(7.5, 0.55, 9.0, 0.55)

    # Final
    box(9.0, 0.1, 2.5, 0.9, "Final answer\n(natural language)", color="#fff5e6", edge="#cc7a00")

    # KG indicator
    box(0.4, 3.5, 1.4, 1.0, "RadarKG\n16.5k triples\n98 relations & attrs", color="#fafafa", edge="#777")

    # Legend / annotation
    ax.text(0.3, 7.05, "Strategy-Routed GraphRAG Pipeline", fontsize=13, fontweight="bold")
    ax.text(11.0, 7.05, "(3 LLM calls / question)", fontsize=9, ha="right", color="#666")

    # KG arrow into executor
    arrow(1.8, 3.95, 2.2, 3.95)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig1_pipeline.png", bbox_inches="tight", dpi=180)
    plt.close()
    print("Saved fig1_pipeline.{pdf,png}")


# ────────────────────────────────────────────────────────────
#  Fig 2: Main results — per-type 3-way bar chart
# ────────────────────────────────────────────────────────────

def fig2_main_results():
    """Per-type 3-way bar chart on the FULL 499-question evaluation."""
    import json
    from collections import defaultdict

    full_path = ROOT / "results" / "qa500_3way_full.json"
    with open(full_path, encoding="utf-8") as f:
        recs = json.load(f)["per_question"]

    by_t = defaultdict(lambda: {"b": [], "r": [], "s": []})
    for rec in recs:
        by_t[rec["type"]]["b"].append(rec["baseline"]["score"].get("correct", False))
        by_t[rec["type"]]["r"].append(rec["rog_style"]["score"].get("correct", False))
        by_t[rec["type"]]["s"].append(rec["strategy"]["score"].get("correct", False))

    # ordering: by n desc, breaking ties alphabetically so the chart layout is stable
    order = [
        "single_hop", "two_hop_bridge", "agg_count", "agg_enum", "attr_filter",
        "relation_inverse", "negation", "set_compare", "three_hop_chain",
        "unanswerable", "distractor",
    ]
    data = []
    for t in order:
        d = by_t[t]
        n = len(d["b"])
        data.append((t, sum(d["b"])/n, sum(d["r"])/n, sum(d["s"])/n, n))

    types = [d[0] for d in data]
    base  = [d[1] for d in data]
    rog   = [d[2] for d in data]
    strat = [d[3] for d in data]
    ns    = [d[4] for d in data]

    overall_b = sum(sum(by_t[t]["b"]) for t in order) / sum(ns)
    overall_r = sum(sum(by_t[t]["r"]) for t in order) / sum(ns)
    overall_s = sum(sum(by_t[t]["s"]) for t in order) / sum(ns)

    import numpy as np
    x = np.arange(len(types))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(x - width, base,  width, label="Baseline (top-K)",         color="#bbb",     edgecolor="#666")
    ax.bar(x,         rog,   width, label="RoG-style (uniform plan)", color="#f6a85a", edgecolor="#7a4513")
    ax.bar(x + width, strat, width, label="Strategy-Routed (ours)",   color="#2c8c2d", edgecolor="#13441b")

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nn={n}" for t, n in zip(types, ns)], rotation=30, ha='right', fontsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)

    # overall reference lines
    ax.axhline(y=overall_s, color="#2c8c2d", linestyle=":", alpha=0.55, linewidth=1.0)
    ax.axhline(y=overall_r, color="#f6a85a", linestyle=":", alpha=0.55, linewidth=1.0)
    ax.axhline(y=overall_b, color="#888",    linestyle=":", alpha=0.55, linewidth=1.0)
    ax.text(len(types) - 0.4, overall_s + 0.01, f"Strategy {overall_s*100:.1f}%", color="#13441b", fontsize=8.5, ha='right')
    ax.text(len(types) - 0.4, overall_r + 0.01, f"RoG {overall_r*100:.1f}%",      color="#7a4513", fontsize=8.5, ha='right')
    ax.text(len(types) - 0.4, overall_b + 0.01, f"Baseline {overall_b*100:.1f}%", color="#444",    fontsize=8.5, ha='right')

    plt.title(f"End-to-End Accuracy on RadarKG-QA-499 (full 499-question benchmark)\n"
              f"Strategy +{(overall_s-overall_b)*100:.1f}pp vs Baseline, +{(overall_s-overall_r)*100:.1f}pp vs RoG-style",
              fontsize=11.5, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_main_results.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig2_main_results.png", bbox_inches="tight", dpi=180)
    plt.close()
    print(f"Saved fig2_main_results.{{pdf,png}}  "
          f"(B={overall_b*100:.1f}%, R={overall_r*100:.1f}%, S={overall_s*100:.1f}%)")


# ────────────────────────────────────────────────────────────
#  Fig 3: Per-strategy ablation drop
# ────────────────────────────────────────────────────────────

def fig3_ablation():
    # Each row: (strategy ablated, total drop pp, top-affected category, drop on that category)
    data = [
        ("exhaustive",       18.0, "agg_count -60 / agg_enum -60 / rel_inverse -60"),
        ("constrained_join",  8.0, "attr_filter -80"),
        ("path_plan",         6.0, "three_hop -37.5 / two_hop -25"),
        ("complement",        0.0, "negation 0 (lookup gracefully degrades)"),
        ("dual_subgraph",     0.0, "set_compare 0 (lookup gracefully degrades)"),
    ]
    labels  = [d[0] for d in data]
    drops   = [d[1] for d in data]
    annot   = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#c0392b" if d >= 5 else "#888" for d in drops]
    bars = ax.barh(labels, drops, color=colors, edgecolor="#2c2c2c")
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy drop when this strategy is ablated to lookup (pp)", fontsize=10)
    ax.set_xlim(0, 22)

    for bar, d, a in zip(bars, drops, annot):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        ax.text(x + 0.3, y, f"{d:>5.1f} pp   ({a})", va='center', fontsize=9, color="#222")

    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.title("Per-Strategy Ablation: Contribution to Overall Accuracy\n(100-question stratified subset; ablations not re-run on full 499-Q set)",
              fontsize=11.5, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_ablation.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig3_ablation.png", bbox_inches="tight", dpi=180)
    plt.close()
    print("Saved fig3_ablation.{pdf,png}")


def main():
    fig1_pipeline()
    fig2_main_results()
    fig3_ablation()
    print(f"\nAll figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
