# -*- coding: utf-8 -*-
"""Finalize the 11-slide thesis proposal deck and add speaker notes."""

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports" / "论文开题_图检索策略学习.pptx"
OUTPUT = ROOT / "reports" / "论文开题_图检索策略学习_最终答辩版_11页.pptx"

NAVY = RGBColor(0x18, 0x36, 0x57)
BLUE = RGBColor(0x2F, 0x64, 0x91)
TEAL = RGBColor(0x4F, 0x7D, 0x7B)
GOLD = RGBColor(0xB4, 0x88, 0x38)
INK = RGBColor(0x24, 0x2D, 0x36)
GRAY = RGBColor(0x68, 0x73, 0x7D)
MID = RGBColor(0xB5, 0xC1, 0xCC)
LINE = RGBColor(0xD8, 0xE0, 0xE7)
PALE_BLUE = RGBColor(0xEC, 0xF1, 0xF5)
PALE_TEAL = RGBColor(0xEB, 0xF2, 0xF1)
PALE_GOLD = RGBColor(0xF6, 0xF1, 0xE6)
BG = RGBColor(0xFA, 0xFB, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"
MONO = "Consolas"


prs = Presentation(str(SOURCE))


def set_typeface(run, name=FONT):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = rpr.find(qn(tag))
        if node is None:
            node = rpr.makeelement(qn(tag), {})
            rpr.append(node)
        node.set("typeface", name)


def style_run(run, size=12, bold=False, color=INK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    set_typeface(run, name)


def add_text(slide, value, x, y, w, h, size=12, bold=False, color=INK,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.03,
             name=FONT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    style_run(run, size, bold, color, name)
    return shape


def add_rect(slide, x, y, w, h, fill=WHITE, border=LINE,
             kind=MSO_SHAPE.RECTANGLE, width=0.8):
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if border is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = border
        shape.line.width = Pt(width)
    shape.shadow.inherit = False
    return shape


def add_line(slide, x1, y1, x2, y2, color=MID, width=1.0):
    shape = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    return shape


def style_cell(cell, fill, color, size, bold=False, align=PP_ALIGN.LEFT,
               name=FONT):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Inches(0.07)
    cell.margin_top = cell.margin_bottom = Inches(0.035)
    frame = cell.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in frame.paragraphs:
        paragraph.alignment = align
        paragraph.space_before = paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.0
        for run in paragraph.runs:
            style_run(run, size, bold, color, name)


def make_related_work_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    sld_id = prs.slides._sldIdLst[-1]
    prs.slides._sldIdLst.remove(sld_id)
    prs.slides._sldIdLst.insert(2, sld_id)

    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_rect(slide, 0, 0, 0.07, 7.5, NAVY, None)
    add_text(slide, "RELATED WORK  |  代表性研究与技术演进", 0.47, 0.18,
             4.8, 0.22, 8.8, True, TEAL)
    add_text(slide, "图检索增强问答正从提示驱动走向可学习的多步策略",
             0.47, 0.48, 8.70, 0.50, 19.5, True, NAVY)
    add_line(slide, 0.47, 1.08, 9.55, 1.08, MID, 0.8)

    rows = [
        ["代表工作", "核心机制", "技术推进", "与本文的关系"],
        ["ToG / RoG\nICLR 2024",
         "LLM 迭代探索知识图；先规划关系路径，再检索证据链",
         "图上多步推理\n显式路径规划",
         "从文本召回推进到\n规划并执行图路径"],
        ["ToG-2\nICLR 2025",
         "知识图谱引导文本检索，文本上下文反向筛选图中候选实体",
         "图文紧耦合\n迭代协同检索",
         "支撑图谱与叙述库\n协同的知识环境"],
        ["Graph-R1\nICML 2026",
         "知识超图上的多轮智能体交互，以端到端强化学习优化检索轨迹",
         "检索过程由固定流程\n转向可学习策略",
         "证明多轮图检索策略\n可通过交互轨迹学习"],
        ["GraphRAG-R1\nWWW 2026",
         "图文混合检索；SFT 冷启动；渐进检索衰减与成本感知奖励",
         "过程约束与\n质量—成本平衡",
         "说明执行过程反馈\n可超越终局答案信号"],
        ["HyperGraphPro\n(v1: ProGraph-R1)\narXiv 2026",
         "结构感知超图检索；按中间推理进展进行逐步策略优化",
         "中间进展形成\n更密集学习信号",
         "本文进一步关注类型、\n变量依赖与确定性执行"],
    ]
    shape = slide.shapes.add_table(6, 4, Inches(0.55), Inches(1.37),
                                   Inches(8.90), Inches(3.86))
    table = shape.table
    widths = [1.45, 3.18, 1.70, 2.57]
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    heights = [0.42, 0.65, 0.65, 0.65, 0.68, 0.81]
    for row, height in zip(table.rows, heights):
        row.height = Inches(height)
    for row_idx, values in enumerate(rows):
        for col_idx, value in enumerate(values):
            cell = table.cell(row_idx, col_idx)
            cell.text = value
            if row_idx == 0:
                style_cell(cell, NAVY, WHITE, 8.4, True, PP_ALIGN.CENTER)
            else:
                fill = WHITE if row_idx % 2 else PALE_BLUE
                if row_idx == 4:
                    fill = PALE_TEAL
                elif row_idx == 5:
                    fill = PALE_GOLD
                style_cell(
                    cell, fill, NAVY if col_idx == 0 else INK,
                    7.65 if row_idx < 5 else 7.35,
                    col_idx == 0, PP_ALIGN.CENTER if col_idx != 1 else PP_ALIGN.LEFT,
                    MONO if col_idx == 0 else FONT,
                )

    add_text(slide, "技术演进", 0.62, 5.49, 0.72, 0.20, 8.0, True, TEAL,
             PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    add_rect(slide, 1.47, 5.43, 7.86, 0.40, WHITE, LINE)
    add_text(slide,
             "固定图路径 / 提示驱动  →  显式规划与图文协同  →  可学习的多轮检索  →  过程约束与进展反馈",
             1.62, 5.51, 7.53, 0.20, 8.6, True, NAVY,
             PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

    add_rect(slide, 0.64, 6.02, 0.045, 0.46, TEAL, None)
    add_text(slide,
             "进一步问题：如何让集合、计数、比较及中间变量依赖形成类型安全、可确定执行的策略空间，"
             "并由执行状态产生可验证的多粒度反馈？",
             0.83, 6.05, 8.35, 0.39, 9.5, True, NAVY,
             PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE)
    add_text(slide,
             "文献状态核验：ICLR 2024/2025，ICML 2026，WWW 2026，arXiv:2601.17755；完整题名与链接见备注。",
             0.82, 6.65, 8.20, 0.18, 6.8, False, GRAY)
    add_line(slide, 0.45, 7.13, 9.56, 7.13, LINE, 0.6)
    add_text(slide, "硕士学位论文开题答辩  |  图检索策略学习",
             0.47, 7.18, 5.8, 0.17, 7.8, False, GRAY)
    add_text(slide, "03", 9.02, 7.17, 0.45, 0.18, 8.0, True, GRAY,
             PP_ALIGN.RIGHT, name=MONO)
    return slide


def replace_shape_text(slide, old, new):
    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        if shape.text.strip() == old:
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = ""
            paragraph = shape.text_frame.paragraphs[0]
            run = paragraph.add_run()
            run.text = new
            # Preserve the first existing run's common title style when possible.
            style_run(run, 20 if shape.top < Inches(1) else 10.0,
                      True, NAVY)
            return shape
    return None


def polish_science_problem_slide():
    slide = prs.slides[3]
    title = next(
        shape for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
        and shape.text.strip() == "现有方法的核心缺口是计划表达与长程学习"
    )
    title.text_frame.clear()
    run = title.text_frame.paragraphs[0].add_run()
    run.text = "复杂组合问答进一步提出两个关键研究问题"
    style_run(run, 20, True, NAVY)

    table = next(shape.table for shape in slide.shapes if getattr(shape, "has_table", False))
    values = [
        ["方法范式", "主要能力", "复杂组合问答中的进一步问题"],
        ["文本 RAG", "文本证据召回", "组合关系依赖跨片段共现，难以直接完成精确计算"],
        ["图路径 / GraphRAG", "多跳关系检索", "集合、聚合和中间变量仍需结构化操作表达"],
        ["工具 Agent", "多轮工具调用", "扁平工具与参数组合增大长程决策难度"],
    ]
    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            table.cell(row_idx, col_idx).text = value
            if row_idx == 0:
                style_cell(table.cell(row_idx, col_idx), NAVY, WHITE, 8.5, True,
                           PP_ALIGN.CENTER)
            else:
                style_cell(table.cell(row_idx, col_idx),
                           WHITE if row_idx % 2 else PALE_BLUE,
                           NAVY if col_idx == 0 else INK, 8.3,
                           col_idx == 0, PP_ALIGN.CENTER if col_idx < 2 else PP_ALIGN.LEFT)

    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text.strip()
        if text.startswith("Q1  策略空间如何设计？"):
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = ("Q1  策略空间如何设计？\n"
                      "将任务分解、算子、参数和变量依赖组织为类型安全、可执行的计划")
            style_run(r, 10.0, True, NAVY)
        elif text.startswith("Q2  策略如何有效学习？"):
            shape.text_frame.clear()
            p = shape.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = ("Q2  策略如何有效学习？\n"
                      "利用计划、状态进展、错误恢复和成本反馈缓解终局奖励稀疏")
            style_run(r, 10.0, True, NAVY)


def restore_research_foundation_counts():
    """Use the thesis build-report scope requested for the research-foundation slide."""
    slide = prs.slides[8]
    replacements = {
        "10,919": "10,744",
        "21,928": "22,241",
        "现行可用关系边": "建图报告关系边",
    }
    for shape in slide.shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        old = shape.text.strip()
        if old not in replacements:
            continue
        shape.text_frame.clear()
        paragraph = shape.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = replacements[old]
        if old in {"10,919", "21,928"}:
            style_run(run, 18.5, True, NAVY, MONO)
        else:
            style_run(run, 8.6, True, GRAY)


def renumber_slides():
    for page, slide in enumerate(prs.slides, 1):
        found = False
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            if shape.top < Inches(6.9):
                continue
            if re.fullmatch(r"\d{2}|KG", shape.text.strip()):
                shape.text_frame.clear()
                paragraph = shape.text_frame.paragraphs[0]
                paragraph.alignment = PP_ALIGN.RIGHT
                run = paragraph.add_run()
                run.text = f"{page:02d}"
                style_run(run, 8.0, True, GRAY, MONO)
                found = True
                break
        if not found and page > 1:
            add_text(slide, f"{page:02d}", 9.02, 7.17, 0.45, 0.18,
                     8.0, True, GRAY, PP_ALIGN.RIGHT, name=MONO)


SPEAKER_NOTES = [
    """各位老师好，我的论文题目是《面向复杂雷达情报问答的图检索策略学习方法研究》。本研究关注的不是单次检索命中率，而是复杂问题需要经过多步图检索、集合运算和精确计算时，模型如何形成稳定、可执行的决策过程。汇报将按照问题场景、相关研究、方法设计、已有基础和实验方案展开。整条主线可以概括为：先构造类型安全的检索策略空间，再利用执行环境产生的反馈优化这套策略，最后通过组合隔离和公开数据集验证模型学到的是可组合能力。""",

    """先看一个典型问题：美国研制、工作在 J 波段的雷达有多少款。这个问题不能靠检索一段相似文本直接回答，它至少包含国家约束、频段约束、两个候选集合求交以及最终计数。系统需要把自然语言条件转成一连串可执行步骤，并根据中间结果决定下一步。这里的 RadarKG 提供结构化事实，叙述文本补充原理和背景，工具接口负责精确的图查询和集合计算。因此，复杂雷达问答的本质不是简单路由，而是组合检索计划的生成、执行和修正。""",

    """这一页说明相关技术如何发展。2024 年的 ToG 和 RoG 已经把大模型与知识图谱结合起来，通过迭代图探索或先规划关系路径再检索证据，实现图上的多步推理。2025 年 ToG-2 进一步把知识图谱和文本检索紧密耦合。到 2026 年，Graph-R1 开始用强化学习训练多轮图检索策略；GraphRAG-R1加入了检索深度和成本约束；HyperGraphPro，也就是 ProGraph-R1 的后续版本，又把中间推理进展用于逐步优化。因此本文不会声称首次把强化学习用于 GraphRAG。我的切入点更具体：面对集合、计数、比较和变量依赖，怎样构造类型安全、可确定执行的策略空间，并利用执行状态形成可验证反馈。\n\n参考文献（备注区，不必逐条口述）：\n[1] Sun et al. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. ICLR 2024. https://openreview.net/forum?id=nnVO1PvbTv\n[2] Luo et al. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. ICLR 2024. https://openreview.net/forum?id=ZGNWW7xZ6Q\n[3] Ma et al. Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/830b1abc6d2da85f23d41169fa44d185-Paper-Conference.pdf\n[4] Luo et al. Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning. ICML 2026. https://arxiv.org/abs/2507.21892\n[5] Yu et al. GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning. WWW 2026. https://doi.org/10.1145/3774904.3792589\n[6] Park et al. HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided Hypergraph RAG. arXiv 2026, v1 title: ProGraph-R1. https://arxiv.org/abs/2601.17755""",

    """基于前面的研究演进，论文聚焦两个递进问题。第一个问题是策略空间怎么设计，也就是模型究竟要学习什么。我要把任务分解、算子选择、参数生成和中间变量依赖组织成类型安全、能够实际执行的计划。第二个问题是策略怎么学习。复杂任务往往到最后才知道答案是否正确，反馈过于稀疏，所以需要利用计划是否合法、中间状态是否推进、错误能否修复以及执行成本等信息。两者不是两套无关方法：前者定义可学习的决策结构，后者在同一个结构上解决怎么优化。""",

    """总体方法包含两个闭环，但优化的是同一个检索策略。上半部分是推理执行闭环：输入复杂问题后，策略生成类型化查询计划，确定性执行环境完成图查询、集合运算、计数比较和文本检索，再把结构化观察返回给策略，策略决定继续、修正、恢复还是结束。下半部分是学习闭环：先用成功轨迹进行监督微调，让模型掌握合法动作和基本流程；再与执行环境交互产生完整轨迹，根据答案、计划、进展、恢复和成本计算反馈，持续更新同一个策略。最后通过组合隔离和公开集检验泛化。""",

    """这一页回答模型到底学习什么，而不仅是给几个工具做一次分类。状态中包含问题、历史动作、中间结果和错误信息。每个动作由任务类型、具体算子、参数、中间变量绑定和控制信号组成。类型系统限制每个算子的输入输出，例如约束查询产生实体集合，求交仍然得到实体集合，计数才把实体集合转成标量。下方计划树展示了一个完整过程：先分别执行国家和频段约束，把结果绑定到两个变量，再求交，最后计数。低层计算由执行器确定完成，模型重点学习高层分解、参数选择、变量依赖和何时修正或终止。""",

    """在策略学习阶段，我先从已经能够成功执行的轨迹做监督微调，解决模型一开始不会生成合法动作的问题。之后策略与环境进行多轮交互，环境记录完整轨迹，并从多个层面评价：最终答案是否正确，计划和类型是否合法，中间结果是否向目标推进，遇到空结果或错误参数后能否恢复，以及是否存在重复调用和不必要成本。奖励公式只是这些信号的统一表达，核心并不依赖某一种具体优化器。训练按照基础任务、组合任务和错误恢复逐步增加难度，重点观察强化学习是否真正超过成功轨迹模仿。""",

    """这页是知识环境的构建基础。数据来自百科、专业网站、公开装备指南、Wikidata，以及机载和海用扫描手册。扫描材料使用 Qwen3-VL 做视觉转录，再区分结构字段和叙述段。抽取阶段同时采用确定性规则和多智能体流程：先盘点文本中的实体，再抽取关系，最后做忠实度批判。融合阶段进行类型签名检查、别名和单位归一、三元组去重、多源合并与冲突记录。最终形成知识图谱和叙述文本两层环境。这里 21,928 是当前可执行过滤视图；下一页的 22,241 是建图报告记录的全量边数，两个口径用途不同。""",

    """目前已经具备三类研究基础。第一是知识环境：建图报告记录 22,241 条边、10,744 个实体，并建立 1,432 段叙述文本索引。第二是可执行工具和类型化算子，能够完成关系查询、约束、集合运算、路径、聚合等十类操作。第三是复杂问答数据和成功轨迹，训练、验证、测试分别为 4,277、582 和 542，并完成了环境回放。这里我把已完成和拟开展严格区分：知识库、数据、执行器和训练原型已经具备；多粒度反馈的完整训练、消融和公开集实验是论文后续工作的重点。""",

    """这些结果定位为诊断实验，不是最终方法成绩。第一组结果表明，加入图结构和规划后，领域问答从 52.7 提升到 60.3，再到 88.6，说明结构化计划是有效方向。但 LLM 路由 88.6 与 Oracle 88.2 基本相当，说明单步路由并不是主要瓶颈。第二组跨域实验在 KQA Pro 上变化约负 0.2 个百分点，而且置信区间跨零，说明固定策略存在明显任务依赖。Qwen Planner 从 31.5 提升到 61.7，则再次支持显式计划表示。综合来看，后续重点应转向可学习的多步策略以及它与执行环境的适配。""",

    """最后是核心实验设计。二维矩阵把两个贡献分别验证：横向从扁平 SFT 到类型化 SFT，考察策略空间构造本身的价值；纵向从类型化 SFT 到完整方法，考察多粒度执行反馈是否带来额外学习收益。RadarKG-QA 设置常规划分、实体隔离、组合隔离和错误恢复。其中组合隔离是指训练和测试使用不同的算子组合或程序结构，用来判断模型学到的是可组合策略，而不是记住问题模板。KQA Pro 和 MetaQA 用于公开验证。报告时分为环境上限、策略执行、最终答案以及成本和恢复四层，避免把知识库覆盖不足误判成策略问题。""",
]


def set_speaker_notes():
    if len(prs.slides) != len(SPEAKER_NOTES):
        raise RuntimeError(
            f"Speaker-note count mismatch: slides={len(prs.slides)}, notes={len(SPEAKER_NOTES)}"
        )
    for slide, note in zip(prs.slides, SPEAKER_NOTES):
        frame = slide.notes_slide.notes_text_frame
        frame.text = note


make_related_work_slide()
polish_science_problem_slide()
restore_research_foundation_counts()
renumber_slides()
set_speaker_notes()

prs.core_properties.title = "面向复杂雷达情报问答的图检索策略学习方法研究"
prs.core_properties.subject = "硕士学位论文开题答辩最终版（11页，含逐页讲稿）"
prs.core_properties.comments = "新增代表性研究与技术演进页；全部页面含 Speaker Notes。"
prs.save(str(OUTPUT))
print(f"saved: {OUTPUT}")
print(f"slides: {len(prs.slides)}; notes: {len(SPEAKER_NOTES)}")
