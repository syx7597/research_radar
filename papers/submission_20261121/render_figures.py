"""Native vector diagram and 1200 dpi line-art export for the submission draft."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.6, 3.6))
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975)
ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis("off")


def box(x, y, w, h, text, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=0.8))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=11, linewidth=0.8, color="black"))


box(.2,5.15,9.6,.65,"Training questions, reference programs, and fixed initial model I")
box(.2,3.95,9.6,.75,"Failed episodes: common successful prefix\nand one actual reference divergence")
arrow((5,5.1),(5,4.75))
box(.2,1.8,4.65,1.55,"Ordinary continuation A\nCommon prefix\nExecute reference suffix")
box(5.15,1.8,4.65,1.55,"Recovery continuation C\nCommon prefix + divergence\nRemap and execute suffix")
arrow((2.5,3.9),(2.5,3.4)); arrow((7.5,3.9),(7.5,3.4))
box(.2,.2,9.6,1.0,"Verify observations and answers; supervise suffix actions\nShare successful examples, order, and target-token budget")
arrow((2.5,1.75),(2.5,1.25));arrow((7.5,1.75),(7.5,1.25))
for ext in ("pdf", "eps", "png"):
    fig.savefig(HERE/f"Fig1.{ext}", dpi=1200, metadata={"Creator":"Research manuscript figure script"} if ext in ("pdf","eps") else None)
plt.close(fig)
