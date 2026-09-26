# -*- coding: utf-8 -*-
"""Publication-quality figures for the journal submissions.

Regenerates the two data charts in a clean, consistent style (claims live in
the LaTeX caption, not the chart title) and adds a cross-benchmark cardinality
figure that visually motivates RadarKG-QA-499. Numbers match the manuscript
tables exactly. Outputs vector PDF (+ PNG preview) to the current directory.

Run from paper/figures/:  python make_pub_figs.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- consistent, restrained style ----------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 150,
})
C_BASE = "#AEB6BD"   # muted slate  -- baseline
C_ROG  = "#E0A458"   # muted amber  -- RoG-style
C_STR  = "#2F8F76"   # teal-green   -- Strategy (ours, emphasized)
C_DROP = "#B5544C"   # muted brick  -- ablation drop
C_GREY = "#C9CDD2"


def save(fig, stem):
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("wrote", stem + ".pdf")


# === Fig: main per-type results ===========================================
types = ["agg_count", "agg_enum", "attr_filter", "relation_inverse",
         "three_hop_chain", "two_hop_bridge", "single_hop", "set_compare",
         "negation", "unanswerable", "distractor"]
n     = [50, 50, 50, 50, 30, 80, 80, 40, 40, 20, 9]
base  = [4.0, 46.0, 12.0, 40.0, 23.3, 40.0, 83.8, 97.5, 100, 90.0, 100]
rog   = [38.0, 92.0, 0.0, 74.0, 50.0, 95.0, 11.2, 85.0, 97.5, 100, 66.7]
strat = [62.0, 96.0, 92.0, 86.0, 93.3, 88.8, 86.2, 97.5, 100, 90.0, 100]

fig, ax = plt.subplots(figsize=(12, 4.6))
x = range(len(types)); w = 0.27
ax.bar([i - w for i in x], base,  w, label="Baseline (top-$K$)", color=C_BASE, edgecolor="white", linewidth=0.5)
ax.bar(list(x),           rog,   w, label="RoG-style (uniform plan)", color=C_ROG, edgecolor="white", linewidth=0.5)
ax.bar([i + w for i in x], strat, w, label="Strategy-Routed (ours)", color=C_STR, edgecolor="white", linewidth=0.5)
ax.set_xlim(-0.7, len(types) + 0.2)
for y, c, lab in [(52.7, C_BASE, "overall 52.7"), (88.6, C_STR, "overall 88.6")]:
    ax.axhline(y, ls=(0, (4, 3)), lw=0.9, color=c, alpha=0.9, zorder=0)
    ax.text(len(types) - 0.1, y, lab, color=c, fontsize=8.5, va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_ylim(0, 105)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{t}\n$n{{=}}{m}$" for t, m in zip(types, n)], fontsize=8.5, rotation=30, ha="right")
ax.grid(axis="y", ls=":", lw=0.6, color=C_GREY)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=10)
save(fig, "fig2_main_results")


# === Fig: per-operator ablation (numbers match RQ2 table) ==================
ops   = ["exhaustive", "path-plan", "constrained-join", "complement", "dual-subgraph"]
drop  = [19, 10, 8, 0, 0]                      # pp, matches Table (RQ2)
agm   = ["unbounded enumeration", "composition ($k\\geq2$)", "intersection (AND)",
         "non-membership", "binary comparison"]
note  = ["agg_count / agg_enum / rel_inverse $-60$ each", "three_hop $-62.5$ / two_hop $-41.7$",
         "attr_filter $-90$", "lookup graceful-degrades", "lookup graceful-degrades"]

fig, ax = plt.subplots(figsize=(9.2, 3.6))
yy = list(range(len(ops)))[::-1]
cols = [C_DROP if d > 0 else C_GREY for d in drop]
ax.barh(yy, drop, color=cols, edgecolor="white", height=0.62)
for y, d, a, nt in zip(yy, drop, agm, note):
    ax.text(d + 0.4, y, f"$-{d}$ pp  ({a})" if d else f"$0$ pp  ({a})",
            va="center", fontsize=9.5)
ax.set_yticks(yy); ax.set_yticklabels(ops, fontsize=11)
ax.set_xlim(0, 30)
ax.set_xlabel("Accuracy drop when ablated to lookup (pp)")
ax.grid(axis="x", ls=":", lw=0.6, color=C_GREY); ax.set_axisbelow(True)
save(fig, "fig3_ablation")


# === Fig: cross-benchmark Count cardinality (motivates the benchmark) ======
bench = ["RadarKG-\nQA-499", "KQA Pro\nval", "KQA Pro\npure-enum", "Mintaka\ndev"]
med   = [14, 2, 1, 4]
hi    = [18, 6, 2, 3]    # % of Count questions with cardinality >= 30
cols  = [C_STR, C_BASE, C_BASE, C_BASE]

fig, ax = plt.subplots(figsize=(7.0, 3.8))
xx = range(len(bench))
bars = ax.bar(xx, med, color=cols, edgecolor="white", width=0.6)
for i, (m, h) in enumerate(zip(med, hi)):
    ax.text(i, m + 0.3, f"med {m}\n({h}% $\\geq$30)", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 17)
ax.set_ylabel("Median Count answer cardinality")
ax.set_xticks(list(xx)); ax.set_xticklabels(bench, fontsize=9.5)
ax.grid(axis="y", ls=":", lw=0.6, color=C_GREY); ax.set_axisbelow(True)
ax.annotate("$3.4\\times$ the next-closest\npublic benchmark",
            xy=(0, 14), xytext=(1.4, 13.0), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#444"))
save(fig, "fig4_cardinality")

print("done")
