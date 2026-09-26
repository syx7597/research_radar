# -*- coding: utf-8 -*-
"""Generate restrained, publication-style figures for the proposal deck."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "academic_figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1B3655"
BLUE = "#2E5F8A"
TEAL = "#5F8784"
GOLD = "#B58B3C"
GRAY = "#66727E"
LINE = "#C8D2DC"
PALE = "#EDF2F6"
PALE_GOLD = "#F6F1E7"
WHITE = "#FFFFFF"

FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")
CN = font_manager.FontProperties(fname=str(FONT_PATH))
CN_BOLD = font_manager.FontProperties(fname=str(FONT_BOLD_PATH))


def setup(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, subtitle="", fill=WHITE, edge=BLUE,
        title_color=NAVY, lw=1.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h * (0.58 if subtitle else 0.50), title,
            ha="center", va="center", color=title_color,
            fontsize=12, fontproperties=CN_BOLD)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha="center", va="center", color=GRAY,
                fontsize=8.5, fontproperties=CN)


def arrow(ax, start, end, color=BLUE, lw=1.4, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color,
                                 connectionstyle=connectionstyle))


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.04,
                facecolor=WHITE)
    plt.close(fig)


def unified_framework():
    fig, ax = setup((14.5, 6.5))
    ax.text(0.02, 0.94, "推理执行闭环", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    stages = [
        (0.02, 0.61, 0.13, 0.19, "复杂问题", "多跳 · 集合 · 计数"),
        (0.19, 0.61, 0.15, 0.19, "检索策略  πθ", "状态驱动的语义决策"),
        (0.38, 0.61, 0.15, 0.19, "类型化计划", "算子 · 参数 · 变量依赖"),
        (0.57, 0.61, 0.15, 0.19, "确定性执行器", "图 / 文本工具与精确计算"),
        (0.76, 0.61, 0.13, 0.19, "结构化观察", "结果 · 状态 · 错误"),
        (0.92, 0.61, 0.07, 0.19, "答案", "终止"),
    ]
    for i, (x, y, w, h, title, sub) in enumerate(stages):
        fill = PALE_GOLD if i == 1 else (PALE if i in (2, 3, 4) else WHITE)
        edge = GOLD if i == 1 else BLUE
        box(ax, x, y, w, h, title, sub, fill, edge)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.006, y + h / 2),
                  (stages[i + 1][0] - 0.006, y + h / 2), BLUE)

    arrow(ax, (0.825, 0.59), (0.265, 0.59), TEAL, 1.2,
          "arc3,rad=-0.18")
    ax.text(0.55, 0.48, "继续规划 / 参数修正 / 错误恢复",
            ha="center", va="center", color=TEAL, fontsize=9.5,
            fontproperties=CN_BOLD)

    ax.plot([0.02, 0.99], [0.40, 0.40], color=LINE, linewidth=0.9)
    ax.text(0.02, 0.34, "策略学习闭环", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    box(ax, 0.10, 0.08, 0.15, 0.17, "成功轨迹", "可执行计划与答案", WHITE, BLUE)
    box(ax, 0.31, 0.08, 0.13, 0.17, "SFT 冷启动", "学习动作协议", PALE, BLUE)
    box(ax, 0.50, 0.08, 0.16, 0.17, "策略优化", "利用环境反馈更新 πθ", PALE_GOLD, GOLD)
    box(ax, 0.72, 0.08, 0.19, 0.17, "多粒度执行反馈",
        "Answer · Plan · Progress · Recovery · Cost", WHITE, TEAL)
    arrow(ax, (0.25, 0.165), (0.31, 0.165))
    arrow(ax, (0.44, 0.165), (0.50, 0.165))
    arrow(ax, (0.72, 0.165), (0.66, 0.165), GOLD)
    arrow(ax, (0.825, 0.27), (0.825, 0.59), TEAL, 1.2)

    ax.text(0.265, 0.86, "A  策略空间构造", color=BLUE, fontsize=10,
            fontproperties=CN_BOLD, ha="center")
    ax.text(0.58, 0.30, "B  执行反馈优化", color=GOLD, fontsize=10,
            fontproperties=CN_BOLD, ha="center")
    save(fig, "unified_framework.png")


def policy_space():
    fig, ax = setup((14.5, 6.3))
    ax.text(0.03, 0.93, "策略状态与动作参数化", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    box(ax, 0.04, 0.70, 0.16, 0.15, "状态  s_t", "问题 + 历史 + 中间结果 + 错误", WHITE, BLUE)
    box(ax, 0.27, 0.67, 0.43, 0.21, "动作  a_t = (macro, operator, args, bindings, control)",
        "宏动作选择 · 类型化参数 · 变量绑定 · 继续 / 修正 / 终止", PALE_GOLD, GOLD)
    box(ax, 0.77, 0.70, 0.18, 0.15, "状态转移", "执行器返回结构化观察", WHITE, BLUE)
    arrow(ax, (0.20, 0.775), (0.27, 0.775))
    arrow(ax, (0.70, 0.775), (0.77, 0.775))

    ax.plot([0.03, 0.97], [0.58, 0.58], color=LINE, linewidth=0.9)
    ax.text(0.03, 0.52, "类型系统约束", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    types = [("Entity", "实体"), ("EntitySet", "实体集合"),
             ("Relation", "关系"), ("Scalar", "标量"), ("Evidence", "证据")]
    for i, (name, cn) in enumerate(types):
        x = 0.06 + i * 0.18
        box(ax, x, 0.39, 0.14, 0.10, name, cn, WHITE, LINE, BLUE, 0.9)

    ax.text(0.03, 0.30, "可执行计划示例", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    box(ax, 0.41, 0.18, 0.18, 0.09, "count($s3)", "EntitySet → Scalar", NAVY, NAVY, WHITE)
    box(ax, 0.36, 0.07, 0.28, 0.08, "intersect($s1, $s2)", "EntitySet × EntitySet → EntitySet",
        BLUE, BLUE, WHITE)
    box(ax, 0.04, 0.05, 0.25, 0.11, "constraint", "countryOfOrigin = 美国 → $s1",
        PALE, BLUE)
    box(ax, 0.71, 0.05, 0.25, 0.11, "constraint", "hasFrequencyBand = J → $s2",
        PALE, BLUE)
    arrow(ax, (0.50, 0.18), (0.50, 0.15), BLUE, 1.1)
    arrow(ax, (0.29, 0.105), (0.36, 0.105), BLUE, 1.1)
    arrow(ax, (0.71, 0.105), (0.64, 0.105), BLUE, 1.1)
    save(fig, "policy_space.png")


def policy_learning():
    fig, ax = setup((14.5, 6.3))
    ax.text(0.03, 0.93, "同一检索策略的训练与优化", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    stages = [
        (0.04, "成功轨迹", "4,277 条"),
        (0.25, "SFT 冷启动", "动作协议与基础调用"),
        (0.48, "环境 Rollout", "执行计划并生成轨迹"),
        (0.72, "策略优化", "多粒度反馈更新 πθ"),
    ]
    for i, (x, title, sub) in enumerate(stages):
        fill = PALE_GOLD if i == 3 else (PALE if i in (1, 2) else WHITE)
        edge = GOLD if i == 3 else BLUE
        box(ax, x, 0.68, 0.17, 0.16, title, sub, fill, edge)
        if i < len(stages) - 1:
            arrow(ax, (x + 0.17, 0.76), (stages[i + 1][0], 0.76))

    ax.text(0.03, 0.54, "奖励由可执行环境自动计算", color=TEAL, fontsize=11,
            fontproperties=CN_BOLD)
    feedback = [
        ("Answer", "答案正确性"), ("Plan", "计划合法性"),
        ("Progress", "状态进展"), ("Recovery", "错误恢复"),
        ("Cost", "步数与重复"),
    ]
    for i, (name, desc) in enumerate(feedback):
        x = 0.04 + i * 0.19
        box(ax, x, 0.38, 0.16, 0.10, name, desc,
            PALE_GOLD if name in ("Progress", "Recovery") else WHITE,
            GOLD if name in ("Progress", "Recovery") else LINE,
            NAVY, 0.9)

    ax.text(0.50, 0.27,
            "R = R_answer + λp R_plan + λg R_progress + λf R_format − λs C_step − λr C_repeat",
            ha="center", va="center", color=NAVY, fontsize=13,
            fontproperties=CN_BOLD)
    ax.plot([0.03, 0.97], [0.18, 0.18], color=LINE, linewidth=0.9)
    curriculum = ["基础调用", "组合任务", "错误恢复", "组合泛化验证"]
    for i, value in enumerate(curriculum):
        x = 0.06 + i * 0.23
        box(ax, x, 0.04, 0.18, 0.08, f"{i + 1}  {value}", "", WHITE,
            GOLD if i == 3 else BLUE, NAVY, 1.0)
        if i < len(curriculum) - 1:
            arrow(ax, (x + 0.18, 0.08), (x + 0.23, 0.08), BLUE, 1.0)
    save(fig, "policy_learning.png")


def representative_results():
    labels = ["Baseline", "RoG-style", "类型化策略原型"]
    values = [52.7, 60.3, 88.6]
    lower = [48.3, 55.5, 86.2]
    upper = [57.1, 64.9, 91.2]
    colors = ["#7A91A5", TEAL, BLUE]

    fig, ax = plt.subplots(figsize=(8.0, 3.7), dpi=200)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    y = [2, 1, 0]
    for yi, label, value, lo, hi, color in zip(y, labels, values, lower, upper, colors):
        ax.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt="o",
                    markersize=6.5, color=color, ecolor=color,
                    elinewidth=2.0, capsize=4, capthick=1.4)
        ax.text(hi + 1.5, yi, f"{value:.1f}", va="center", ha="left",
                color=color, fontsize=10, fontproperties=CN_BOLD)
    ax.set_yticks(y, labels, fontproperties=CN, fontsize=10)
    ax.set_xlim(40, 100)
    ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("答案准确率（%）", fontproperties=CN, fontsize=9, color=GRAY)
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=GRAY, labelsize=8)
    ax.set_title("RadarKG-QA-499：端到端结果与 95% Bootstrap CI",
                 loc="left", pad=14, color=NAVY, fontsize=12,
                 fontproperties=CN_BOLD)
    fig.tight_layout()
    save(fig, "representative_results.png")


if __name__ == "__main__":
    unified_framework()
    policy_space()
    policy_learning()
    representative_results()
    print(f"saved figures to: {OUT}")
