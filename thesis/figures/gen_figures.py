# -*- coding: utf-8 -*-
"""Generate thesis figures from REAL project data (no fabricated numbers).
Labels are kept in English to avoid CJK glyph issues in matplotlib; Chinese
captions live in the thesis text. Run with the project's python.
"""
import json, sys
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _triples():
    return json.load(open(ROOT / "graphrag_index" / "merged_triples.json", encoding="utf-8"))


def fig_kg_relations():
    t = _triples()
    c = Counter(x["relation"] for x in t)
    top = c.most_common(15)
    names = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(names, vals, color="#3b6fb6")
    ax.set_xlabel("Number of triples")
    ax.set_title(f"Top-15 relations in the radar KG (total {len(t)} triples)")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_relations.png"); plt.close(fig)
    print("wrote fig3_relations.png")


def fig_kg_sources():
    t = _triples()
    c = Counter(x.get("source", "?") for x in t)
    top = c.most_common(10)
    names = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(names, vals, color="#5a9e6f")
    ax.set_xlabel("Number of triples")
    ax.set_title("Provenance: triples by extraction source")
    fig.tight_layout(); fig.savefig(FIG / "fig3_sources.png"); plt.close(fig)
    print("wrote fig3_sources.png")


def fig_threat_country():
    """Ch6: threat-radar library composition by country (real data)."""
    p = ROOT / "data" / "ew" / "threat_radars.json"
    if not p.exists():
        return
    radars = json.load(open(p, encoding="utf-8"))["radars"]
    c = Counter((r.get("country", "?").split("/")[0]) for r in radars)
    items = c.most_common()
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(names, vals, color="#b6603b")
    ax.set_ylabel("Number of threat radars")
    ax.set_title(f"Threat-radar library by country (total {len(radars)})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig6_threat_country.png"); plt.close(fig)
    print("wrote fig6_threat_country.png")


def fig_qa_accuracy():
    """Ch4: overall accuracy of Baseline vs RoG vs Strategy-Routed (paper numbers)."""
    methods = ["Top-K\nBaseline", "RoG\n(path planning)", "Strategy-Routed\n(ours)"]
    acc = [52.7, 60.3, 88.6]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bars = ax.bar(methods, acc, color=["#9aa0a6", "#e0a000", "#3b6fb6"])
    ax.set_ylabel("Accuracy on RadarKG-QA-499 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall KGQA accuracy (499 questions)")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)
    ax.annotate("+35.9 pp", xy=(2, 88.6), xytext=(1.2, 95),
                arrowprops=dict(arrowstyle="->", color="#3b6fb6"), color="#3b6fb6", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_qa_accuracy.png"); plt.close(fig)
    print("wrote fig4_qa_accuracy.png")


def fig_op_ablation():
    """Ch4: per-operator ablation drop (paper numbers)."""
    ops = ["–exhaustive", "–path-plan", "–constrained-join", "–complement", "–dual-subgraph"]
    drop = [-19, -10, -8, 0, 0]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(ops[::-1], drop[::-1], color="#b6603b")
    ax.set_xlabel("Accuracy change when operator removed (pp)")
    ax.set_title("Per-operator ablation: gain is operator-driven")
    fig.tight_layout(); fig.savefig(FIG / "fig4_ablation.png"); plt.close(fig)
    print("wrote fig4_ablation.png")


def fig_roadmap():
    """Ch1: overall four-stage roadmap (architecture overview)."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.axis("off")
    stages = [
        ("Ch.3\nKnowledge Graph", "Provenance-aware\nradar KG", "#3b6fb6"),
        ("Ch.4\nExplainable QA", "Strategy-routed\nGraphRAG (AGM)", "#5a9e6f"),
        ("Ch.5\nAuditable Analysis", "Two-layer\nagent", "#e0a000"),
        ("Ch.6\nDecision Support", "Countermeasure\nreasoning", "#b6603b"),
    ]
    n = len(stages); w = 1.0 / n
    for i, (title, sub, c) in enumerate(stages):
        x = i * w
        box = mpatches.FancyBboxPatch((x + 0.012, 0.30), w - 0.03, 0.42,
                                      boxstyle="round,pad=0.01", linewidth=1.2,
                                      edgecolor=c, facecolor=c + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, 0.60, title, ha="center", va="center", fontsize=9, fontweight="bold", color=c)
        ax.text(x + w / 2, 0.42, sub, ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.004, 0.51), xytext=(x + w - 0.018, 0.51),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
    ax.text(0.5, 0.06, "Unifying threads:  typed operators   +   explainable / traceable / uncertainty-aware",
            ha="center", va="center", fontsize=8.5, style="italic", color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG / "fig1_roadmap.png"); plt.close(fig)
    print("wrote fig1_roadmap.png")


def fig_qa_per_category():
    """Ch4: per-category accuracy, baseline vs strategy (paper numbers)."""
    import numpy as np
    cats = ["single_hop", "relation_inverse", "three_hop_chain", "OVERALL"]
    base = [83.8, 40.0, 23.3, 52.7]
    strat = [86.2, 86.0, 93.3, 88.6]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x - w / 2, base, w, label="Top-K Baseline", color="#9aa0a6")
    ax.bar(x + w / 2, strat, w, label="Strategy-Routed", color="#3b6fb6")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Per-category accuracy: baseline vs. strategy-routed")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_per_category.png"); plt.close(fig)
    print("wrote fig4_per_category.png")


def fig_agent_eval():
    """Ch5: agent evaluation metrics."""
    metrics = ["Composition\naccuracy", "Report citation\ncoverage", "Summary\nfaithfulness"]
    vals = [99, 100, 100]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(metrics, vals, color=["#3b6fb6", "#5a9e6f", "#5a9e6f"])
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Two-layer agent: evaluation")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig5_agent_eval.png"); plt.close(fig)
    print("wrote fig5_agent_eval.png")


def fig_ew_diversity():
    """Ch6: conclusion-diversity before/after (overcoming rule collapse)."""
    import numpy as np
    labels = ["Distinct\nconclusion types", "Distinct rationale\nsentences", "Largest single\ngroup share (%)"]
    before = [31, 3, 37]
    after = [143, 12, 19]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x - w / 2, before, w, label="category fields only", color="#9aa0a6")
    ax.bar(x + w / 2, after, w, label="+ parametric & enriched doctrine", color="#b6603b")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Overcoming rule collapse (lower share = more differentiated)")
    ax.legend(fontsize=8)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w / 2, b + 2, str(b), ha="center", fontsize=8)
        ax.text(i + w / 2, a + 2, str(a), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig6_diversity.png"); plt.close(fig)
    print("wrote fig6_diversity.png")


if __name__ == "__main__":
    fig_roadmap()
    fig_kg_relations()
    fig_kg_sources()
    fig_qa_accuracy()
    fig_qa_per_category()
    fig_op_ablation()
    fig_agent_eval()
    fig_threat_country()
    fig_ew_diversity()
