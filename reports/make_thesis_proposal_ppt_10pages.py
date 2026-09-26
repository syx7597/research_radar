# -*- coding: utf-8 -*-
"""Generate a concise, data-enhanced 11-slide academic proposal deck."""

from io import BytesIO
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "PPT模板1.pptx"
OUTPUT = ROOT / "reports" / "硕士学位论文开题答辩_图检索策略学习_11页数据增强版.pptx"
FRAMEWORK_IMAGE = ROOT / "reports" / "ChatGPT Image 2026年8月18日 00_06_00.png"
TRAINING_IMAGE = ROOT / "reports" / "ChatGPT Image 2026年8月18日 00_16_20.png"

NAVY = RGBColor(0x1B, 0x36, 0x55)
BLUE = RGBColor(0x2E, 0x5F, 0x8A)
BLUE_GRAY = RGBColor(0x6F, 0x88, 0x9E)
TEAL = RGBColor(0x5F, 0x87, 0x84)
GOLD = RGBColor(0xB5, 0x8B, 0x3C)
INK = RGBColor(0x23, 0x2C, 0x35)
GRAY = RGBColor(0x67, 0x72, 0x7E)
MID = RGBColor(0xB9, 0xC4, 0xCE)
LINE = RGBColor(0xD7, 0xDF, 0xE6)
PALE_BLUE = RGBColor(0xEB, 0xF0, 0xF4)
PALE_TEAL = RGBColor(0xEC, 0xF2, 0xF1)
PALE_GOLD = RGBColor(0xF6, 0xF1, 0xE7)
BG = RGBColor(0xF8, 0xF9, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"
MONO = "Consolas"


prs = Presentation(str(TEMPLATE))
logo_shape = next(sh for sh in prs.slides[0].shapes if sh.shape_type == 13)
LOGO = BytesIO(logo_shape.image.blob)
while len(prs.slides):
    sid = prs.slides._sldIdLst[0]
    prs.part.drop_rel(sid.rId)
    del prs.slides._sldIdLst[0]
BLANK = prs.slide_layouts[6]
prs.core_properties.title = "面向复杂雷达情报问答的图检索策略学习方法研究"
prs.core_properties.subject = "硕士学位论文开题答辩（11页数据增强版）"


def set_typeface(run, name=FONT):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = rpr.find(qn(tag))
        if node is None:
            node = rpr.makeelement(qn(tag), {})
            rpr.append(node)
        node.set("typeface", name)


def run_style(run, size=13, bold=False, color=INK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    set_typeface(run, name)


def text(slide, value, x, y, w, h, size=13, bold=False, color=INK,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, name=FONT, margin=0.03):
    obj = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = obj.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = value
    run_style(r, size, bold, color, name)
    return obj


def shape(slide, x, y, w, h, fill=WHITE, border=LINE,
          kind=MSO_SHAPE.RECTANGLE, width=0.9):
    obj = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    obj.fill.solid()
    obj.fill.fore_color.rgb = fill
    if border is None:
        obj.line.fill.background()
    else:
        obj.line.color.rgb = border
        obj.line.width = Pt(width)
    obj.shadow.inherit = False
    return obj


def box(slide, value, x, y, w, h, fill=WHITE, border=LINE, size=12,
        bold=False, color=INK, align=PP_ALIGN.CENTER,
        kind=MSO_SHAPE.RECTANGLE, line_width=0.9, margin=0.05):
    obj = shape(slide, x, y, w, h, fill, border, kind, line_width)
    tf = obj.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = value
    run_style(r, size, bold, color)
    return obj


def line(slide, x1, y1, x2, y2, color=MID, width=1.2):
    obj = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    obj.line.color.rgb = color
    obj.line.width = Pt(width)
    return obj


def arrow(slide, x, y, w, h, color=BLUE_GRAY, direction="right"):
    kind = {
        "right": MSO_SHAPE.RIGHT_ARROW,
        "left": MSO_SHAPE.LEFT_ARROW,
        "down": MSO_SHAPE.DOWN_ARROW,
        "up": MSO_SHAPE.UP_ARROW,
    }[direction]
    return shape(slide, x, y, w, h, color, None, kind)


def footer(slide, page):
    line(slide, 0.46, 7.15, 9.57, 7.15, LINE, 0.7)
    text(slide, "硕士学位论文开题答辩  ·  图检索策略学习",
         0.48, 7.19, 6.3, 0.19, 8.3, False, GRAY)
    text(slide, f"{page:02d}", 9.12, 7.18, 0.36, 0.20,
         8.5, True, GRAY, PP_ALIGN.RIGHT)


def title(slide, main, section, page):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    shape(slide, 0, 0, 0.08, 7.5, NAVY, None)
    text(slide, section, 0.46, 0.18, 3.0, 0.25, 9.2, True, TEAL)
    text(slide, main, 0.46, 0.48, 7.25, 0.58, 21, True, NAVY)
    line(slide, 0.46, 1.10, 9.57, 1.10, MID, 0.9)
    LOGO.seek(0)
    slide.shapes.add_picture(LOGO, Inches(7.70), Inches(0.12), width=Inches(2.12))
    footer(slide, page)
    return slide


def new_slide(main, section, page):
    return title(prs.slides.add_slide(BLANK), main, section, page)


def section_label(slide, value, x, y, w, color=BLUE):
    shape(slide, x, y + 0.03, 0.045, 0.28, color, None)
    text(slide, value, x + 0.13, y, w - 0.13, 0.34, 10.8, True, color)


def metric(slide, value, label, x, y, w=2.0, accent=BLUE):
    shape(slide, x, y, w, 0.90, WHITE, LINE)
    text(slide, value, x + 0.05, y + 0.09, w - 0.10, 0.38,
         19, True, accent, PP_ALIGN.CENTER)
    text(slide, label, x + 0.05, y + 0.54, w - 0.10, 0.23,
         9.2, False, GRAY, PP_ALIGN.CENTER)


def table(slide, headers, rows, x, y, w, h, widths, size=9.0):
    t = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h)
    ).table
    for i, value in enumerate(widths):
        t.columns[i].width = Inches(value)
    for j, value in enumerate(headers):
        c = t.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.margin_left = c.margin_right = Inches(0.03)
        c.margin_top = c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = value
        run_style(r, size, True, WHITE)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 else PALE_BLUE
            c.margin_left = c.margin_right = Inches(0.03)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(value)
            run_style(r, size, False, INK)
    return t


def mini_bar(slide, label, value, x, y, w, max_value=100, color=BLUE):
    text(slide, label, x, y, 1.28, 0.25, 9.2, False, INK)
    shape(slide, x + 1.30, y + 0.02, w - 2.00, 0.20, PALE_BLUE, None)
    shape(slide, x + 1.30, y + 0.02,
          (w - 2.00) * value / max_value, 0.20, color, None)
    text(slide, f"{value:.1f}%", x + w - 0.62, y - 0.02, 0.60, 0.26,
         9.2, True, color, PP_ALIGN.RIGHT)


def academic_bar_chart(slide, chart_title, subtitle, data, x, y, w, h,
                       max_value=100, ticks=(0, 25, 50, 75, 100)):
    shape(slide, x, y, w, h, WHITE, LINE)
    text(slide, chart_title, x + 0.18, y + 0.16, w - 0.36, 0.30,
         12.2, True, NAVY)
    text(slide, subtitle, x + 0.18, y + 0.48, w - 0.36, 0.22,
         8.5, False, GRAY)
    plot_x = x + 1.38
    plot_w = w - 1.72
    plot_top = y + 0.92
    plot_bottom = y + h - 0.42
    for tick in ticks:
        gx = plot_x + plot_w * tick / max_value
        line(slide, gx, plot_top - 0.08, gx, plot_bottom, LINE, 0.65)
        text(slide, str(tick), gx - 0.20, plot_bottom + 0.04, 0.40, 0.20,
             7.7, False, GRAY, PP_ALIGN.CENTER)
    line(slide, plot_x, plot_top - 0.08, plot_x, plot_bottom, MID, 0.85)
    count = len(data)
    step = (plot_bottom - plot_top) / count
    for i, (name, value, color) in enumerate(data):
        cy = plot_top + i * step + step * 0.20
        bh = min(0.32, step * 0.48)
        text(slide, name, x + 0.14, cy - 0.01, 1.15, bh + 0.08,
             8.8, False, INK, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
        shape(slide, plot_x, cy, plot_w * value / max_value, bh, color, None)
        text(slide, f"{value:.1f}", plot_x + plot_w * value / max_value + 0.06,
             cy - 0.02, 0.52, bh + 0.10, 8.8, True, color,
             PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE)


def interval_chart(slide, chart_title, subtitle, data, x, y, w, h,
                   max_value=100, ticks=(0, 25, 50, 75, 100)):
    """Draw a compact point-and-interval chart for accuracy with bootstrap CI."""
    shape(slide, x, y, w, h, WHITE, LINE)
    text(slide, chart_title, x + 0.18, y + 0.16, w - 0.36, 0.30,
         12.2, True, NAVY)
    text(slide, subtitle, x + 0.18, y + 0.48, w - 0.36, 0.22,
         8.5, False, GRAY)
    plot_x = x + 1.30
    plot_w = w - 1.70
    plot_top = y + 1.00
    plot_bottom = y + h - 0.48
    for tick in ticks:
        gx = plot_x + plot_w * tick / max_value
        line(slide, gx, plot_top - 0.08, gx, plot_bottom, LINE, 0.65)
        text(slide, str(tick), gx - 0.20, plot_bottom + 0.05, 0.40, 0.20,
             7.7, False, GRAY, PP_ALIGN.CENTER)
    step = (plot_bottom - plot_top) / len(data)
    for i, (name, value, low, high, color) in enumerate(data):
        cy = plot_top + i * step + step * 0.42
        lx = plot_x + plot_w * low / max_value
        hx = plot_x + plot_w * high / max_value
        vx = plot_x + plot_w * value / max_value
        text(slide, name, x + 0.12, cy - 0.13, 1.06, 0.28,
             8.8, False, INK, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
        line(slide, lx, cy, hx, cy, color, 2.0)
        line(slide, lx, cy - 0.08, lx, cy + 0.08, color, 1.2)
        line(slide, hx, cy - 0.08, hx, cy + 0.08, color, 1.2)
        shape(slide, vx - 0.055, cy - 0.055, 0.11, 0.11, color, None,
              MSO_SHAPE.OVAL)
        text(slide, f"{value:.1f}", hx + 0.06, cy - 0.12, 0.52, 0.25,
             8.7, True, color)


# 1 Cover --------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = WHITE
shape(s, 0, 0, 0.12, 7.5, NAVY, None)
LOGO.seek(0)
s.shapes.add_picture(LOGO, Inches(6.95), Inches(0.12), width=Inches(2.85))
text(s, "硕士学位论文开题答辩", 0.78, 1.45, 2.50, 0.34, 11, True, TEAL)
text(s, "面向复杂雷达情报问答的\n图检索策略学习方法研究",
     0.76, 2.05, 8.45, 1.38, 29, True, NAVY, valign=MSO_ANCHOR.MIDDLE)
text(s, "Graph Retrieval Policy Learning for Complex Radar Intelligence Question Answering",
     0.80, 3.64, 8.30, 0.40, 12, False, GRAY)
shape(s, 0.80, 4.32, 3.05, 0.035, BLUE, None)
text(s, "汇报人：XXX    指导教师：XXX\n专业：XXX      日期：2026 年  月  日",
     0.80, 4.68, 5.70, 0.85, 14.5, False, INK)
text(s, "类型化可执行决策  ·  多粒度执行反馈  ·  组合泛化验证",
     0.80, 6.62, 8.30, 0.32, 12.5, True, BLUE)


# 2 Background ---------------------------------------------------------------
s = new_slide("复杂雷达问答的核心需求是组合检索", "研究背景与问题场景", 2)
section_label(s, "异构知识环境", 0.55, 1.35, 1.75)
for i, (a, b) in enumerate([
    ("装备手册", "PDF / 扫描语料"),
    ("知识图谱", "实体 · 关系 · 属性"),
    ("叙述文本", "原理 · 结构 · 背景"),
]):
    y = 1.78 + i * 0.78
    box(s, a, 0.62, y, 1.18, 0.48, PALE_BLUE, BLUE, 10.5, True, NAVY)
    text(s, b, 1.94, y + 0.04, 1.35, 0.38, 9.4, False, GRAY,
         valign=MSO_ANCHOR.MIDDLE)
arrow(s, 3.33, 2.42, 0.38, 0.24, BLUE_GRAY)
box(s, "复杂问题", 3.78, 2.16, 1.30, 0.76, NAVY, None, 13, True, WHITE)
text(s, "美国研制且工作在 J 波段的雷达有多少款？",
     3.40, 3.12, 2.08, 0.72, 11.2, True, INK, PP_ALIGN.CENTER)
arrow(s, 5.18, 2.42, 0.38, 0.24, BLUE_GRAY)
section_label(s, "可执行组合结构", 5.72, 1.35, 2.15)
ops = [
    ("路径", "Radar → Country"),
    ("约束", "country = 美国；band = J"),
    ("集合", "intersect($s1, $s2)"),
    ("聚合", "count($result)"),
]
for i, (a, b) in enumerate(ops):
    y = 1.76 + i * 0.70
    text(s, f"0{i + 1}", 5.80, y + 0.04, 0.34, 0.26,
         9.0, True, TEAL, PP_ALIGN.CENTER)
    text(s, a, 6.23, y, 0.72, 0.34, 10.3, True, NAVY)
    text(s, b, 7.03, y, 2.15, 0.38, 9.1, False, GRAY)
    line(s, 6.18, y + 0.41, 9.22, y + 0.41, LINE, 0.65)
shape(s, 0.72, 5.38, 8.50, 0.98, WHITE, LINE)
text(s, "研究对象", 0.96, 5.65, 0.92, 0.30, 10.5, True, TEAL)
text(s, "将多跳、集合、计数、比较与图文协同问题转化为可执行的图检索策略",
     1.92, 5.56, 7.00, 0.48, 15, True, NAVY,
     PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


# 3 Challenges and questions -------------------------------------------------
s = new_slide("研究聚焦动作空间设计与长程信用分配", "核心科学问题", 3)
for x, num, en, claim, question, method in [
    (0.62, "01", "ACTION-SPACE DESIGN",
     "扁平工具调用耦合任务分解、工具选择、参数生成与精确计算",
     "复杂图检索决策如何被可执行地表示？",
     "类型化宏动作 + 结构化参数 + 确定性执行"),
    (5.14, "02", "CREDIT ASSIGNMENT",
     "最终答案奖励难以识别中间步骤贡献，长程策略学习信号稀疏",
     "长程检索策略如何获得有效学习信号？",
     "计划、进展、恢复与成本构成多粒度反馈"),
]:
    shape(s, x, 1.42, 4.20, 4.38, WHITE, LINE)
    box(s, num, x + 0.20, 1.66, 0.52, 0.52, BLUE, None,
        12, True, WHITE, kind=MSO_SHAPE.OVAL)
    text(s, en, x + 0.88, 1.72, 2.65, 0.28, 9.1, True, BLUE_GRAY)
    text(s, claim, x + 0.25, 2.35, 3.70, 0.70, 13.5, True, NAVY,
         valign=MSO_ANCHOR.MIDDLE)
    text(s, "科学问题", x + 0.25, 3.33, 0.88, 0.26, 9.3, True, TEAL)
    text(s, question, x + 1.15, 3.24, 2.76, 0.52, 11.0, True, INK,
         valign=MSO_ANCHOR.MIDDLE)
    line(s, x + 0.25, 3.93, x + 3.94, 3.93, LINE, 0.8)
    text(s, "方法响应", x + 0.25, 4.22, 0.88, 0.26, 9.3, True, TEAL)
    text(s, method, x + 1.15, 4.13, 2.76, 0.55, 10.7, True, BLUE,
         valign=MSO_ANCHOR.MIDDLE)
box(s, "H1 结构化动作空间提升复杂计划的有效性",
    1.02, 6.10, 3.75, 0.48, PALE_BLUE, BLUE, 10.3, True, NAVY)
box(s, "H2 执行反馈提升组合策略与错误恢复能力",
    5.24, 6.10, 3.75, 0.48, PALE_GOLD, GOLD, 10.3, True, NAVY)


# 4 Overall framework image --------------------------------------------------
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = WHITE
shape(s, 0, 0, 0.08, 7.5, NAVY, None)
s.shapes.add_picture(str(FRAMEWORK_IMAGE), Inches(0.16), Inches(0.20), width=Inches(9.68))
shape(s, 0.72, 6.12, 8.56, 0.62, PALE_GOLD, GOLD)
text(s, "类型化决策解决“如何表示”，执行反馈解决“如何学习”，组合隔离检验“是否泛化”",
     0.93, 6.26, 8.14, 0.32, 12.2, True, NAVY, PP_ALIGN.CENTER)
footer(s, 4)


# 5 Method one ---------------------------------------------------------------
s = new_slide("类型化决策降低动作空间并生成可执行计划", "方法一  ·  Action-space design", 5)
section_label(s, "扁平 Agent", 0.56, 1.32, 1.55, BLUE_GRAY)
shape(s, 0.62, 1.77, 4.00, 1.80, WHITE, LINE)
tools = ["graph_lookup", "subgraph", "set_op", "count", "attr_filter", "text_search"]
for i, item in enumerate(tools):
    row, col = divmod(i, 3)
    box(s, item, 0.88 + col * 1.17, 2.02 + row * 0.54,
        1.02, 0.36, PALE_BLUE, LINE, 7.8, False, NAVY)
text(s, "工具 × 参数 × 历史状态", 1.18, 3.11, 2.85, 0.28,
     11.0, True, BLUE_GRAY, PP_ALIGN.CENTER)

section_label(s, "类型化图检索决策", 5.02, 1.32, 2.35)
shape(s, 5.08, 1.77, 4.28, 1.80, WHITE, BLUE)
for i, (n, a, b) in enumerate([
    ("1", "语义宏动作", "路径 / 约束 / 集合 / 聚合"),
    ("2", "结构化参数", "Entity · EntitySet · Scalar"),
    ("3", "确定性执行", "交并差 · 路径 · 计数 · 比较"),
]):
    cy = 2.02 + i * 0.49
    box(s, n, 5.36, cy, 0.34, 0.34, BLUE, None, 8.8, True, WHITE,
        kind=MSO_SHAPE.OVAL)
    text(s, a, 5.86, cy, 1.06, 0.30, 9.4, True, NAVY)
    text(s, b, 7.01, cy, 2.05, 0.30, 8.6, False, GRAY)

section_label(s, "可执行计划示例", 0.56, 3.93, 1.88, TEAL)
box(s, "count\nEntitySet → Scalar", 4.02, 4.20, 1.94, 0.58,
    NAVY, None, 9.2, True, WHITE)
line(s, 4.99, 4.78, 4.99, 5.06, BLUE, 1.3)
box(s, "intersect($s1, $s2)\nEntitySet × EntitySet → EntitySet",
    3.42, 5.06, 3.14, 0.62, BLUE, None, 8.9, True, WHITE)
line(s, 4.06, 5.68, 2.55, 5.98, BLUE, 1.1)
line(s, 5.92, 5.68, 7.44, 5.98, BLUE, 1.1)
box(s, "constraint\ncountryOfOrigin = 美国", 0.90, 5.98, 3.20, 0.56,
    PALE_BLUE, BLUE, 8.9, True, NAVY)
box(s, "constraint\nhasFrequencyBand = J", 5.90, 5.98, 3.20, 0.56,
    PALE_BLUE, BLUE, 8.9, True, NAVY)
text(s, "中间变量、输入输出类型与数据依赖共同构成可校验的执行协议",
     2.12, 6.70, 5.78, 0.24, 9.5, True, TEAL, PP_ALIGN.CENTER)


# 6 Method two image ---------------------------------------------------------
s = new_slide("多粒度可执行反馈驱动检索策略学习", "方法二  ·  Credit assignment", 6)
s.shapes.add_picture(str(TRAINING_IMAGE), Inches(0.34), Inches(1.23), width=Inches(9.34))


# 7 Existing foundation ------------------------------------------------------
s = new_slide("知识环境、训练数据与执行器构成完整研究基础", "已有研究基础", 7)
for value, label, x, color in [
    ("22,241", "RadarKG 边", 0.58, BLUE),
    ("10,744", "实体", 2.82, BLUE),
    ("1,432", "叙述文本段", 5.06, TEAL),
    ("4,277", "成功轨迹", 7.30, TEAL),
]:
    metric(s, value, label, x, 1.36, 2.02, color)

section_label(s, "问题与训练数据", 0.56, 2.58, 1.86)
table(s, ["数据项", "规模", "当前用途"], [
    ["Train / Dev / Test", "4,277 / 582 / 542", "训练与独立评测"],
    ["Oracle 回放", "5,401", "工具与金标轨迹核验"],
    ["类型化算子", "10 类", "计划生成与执行"],
    ["检索环境", "BM25 + FAISS + 图扩展", "图文协同召回"],
], 0.62, 2.98, 4.42, 2.26, [1.26, 1.38, 1.78], 8.5)

section_label(s, "72 道组合题原型验证", 5.32, 2.58, 2.38, GOLD)
shape(s, 5.38, 2.98, 3.94, 2.26, WHITE, LINE)
mini_bar(s, "计划有效率", 100.0, 5.62, 3.38, 3.42, 100, BLUE)
mini_bar(s, "执行成功率", 100.0, 5.62, 4.02, 3.42, 100, TEAL)
mini_bar(s, "答案正确率", 98.6, 5.62, 4.66, 3.42, 100, GOLD)
text(s, "71 / 72", 8.17, 4.93, 0.72, 0.22, 8.2, False, GRAY, PP_ALIGN.RIGHT)

shape(s, 0.76, 5.65, 4.12, 0.76, PALE_BLUE, BLUE)
text(s, "人工分层抽样 n=173", 0.98, 5.80, 1.52, 0.26, 9.0, True, TEAL)
text(s, "加权精度 85.1%  [75.1, 95.0]", 2.42, 5.75, 2.20, 0.36,
     10.5, True, NAVY, PP_ALIGN.CENTER)
shape(s, 5.10, 5.65, 4.14, 0.76, PALE_GOLD, GOLD)
text(s, "研究基础", 5.34, 5.82, 0.82, 0.26, 9.0, True, GOLD)
text(s, "KG、文本库、成功轨迹与可执行环境已贯通",
     6.18, 5.74, 2.80, 0.40, 10.3, True, NAVY, PP_ALIGN.CENTER)
text(s, "注：85.1% 为可核验抽取层的分层加权估计；推断富化层在正式实验中单独过滤和报告。",
     0.80, 6.62, 8.40, 0.22, 8.0, False, GRAY, PP_ALIGN.CENTER)


# 8 Retrieval and strategy evidence -----------------------------------------
s = new_slide("图扩展与类型化策略在既有实验中显示互补价值", "已有实验  ·  检索与问答", 8)
academic_bar_chart(s, "混合检索消融", "21 题原型 · Overall accuracy（%）", [
    ("Vector", 52.4, BLUE_GRAY),
    ("BM25", 61.9, TEAL),
    ("RRF", 66.7, BLUE),
    ("Full + Graph", 95.2, GOLD),
], 0.58, 1.42, 4.28, 4.18, 100, (0, 25, 50, 75, 100))
interval_chart(s, "RadarKG-QA-499", "答案准确率与 95% bootstrap CI（%）", [
    ("Baseline", 52.7, 48.3, 57.1, BLUE_GRAY),
    ("RoG-style", 60.3, 55.5, 64.9, TEAL),
    ("类型化策略", 88.6, 86.2, 91.2, BLUE),
], 5.05, 1.42, 4.38, 4.18, 100, (0, 25, 50, 75, 100))
shape(s, 0.80, 5.92, 8.40, 0.64, PALE_GOLD, GOLD)
text(s, "图扩展提升知识覆盖；类型化算子进一步解决计数、集合约束与多跳计划的结构表达",
     1.02, 6.08, 7.96, 0.34, 11.3, True, NAVY, PP_ALIGN.CENTER)
text(s, "来源：evaluation/final_results_summary.json；results/qa500_3way_full_ci_summary.md",
     0.82, 6.68, 8.36, 0.20, 7.7, False, GRAY, PP_ALIGN.CENTER)


# 9 Planner, router, and operator evidence ----------------------------------
s = new_slide("已有实验将研究重点定位到计划表达能力", "已有实验  ·  决策机制", 9)
academic_bar_chart(s, "Planner 与类型化策略", "RadarKG-QA-499 · Overall accuracy（%）", [
    ("Qwen-ZS", 31.5, BLUE_GRAY),
    ("Qwen-FT", 61.7, TEAL),
    ("LLaMA-ZS", 26.5, BLUE_GRAY),
    ("LLaMA-FT", 55.9, TEAL),
    ("类型化策略", 88.6, GOLD),
], 0.58, 1.42, 4.46, 4.56, 100, (0, 25, 50, 75, 100))

shape(s, 5.24, 1.42, 4.18, 1.56, WHITE, LINE)
text(s, "LLM Dispatcher vs Oracle", 5.46, 1.62, 3.74, 0.30, 11.6, True, NAVY)
text(s, "88.6%", 5.52, 2.10, 1.10, 0.42, 18, True, BLUE, PP_ALIGN.CENTER)
text(s, "LLM 路由", 5.52, 2.51, 1.10, 0.22, 8.5, False, GRAY, PP_ALIGN.CENTER)
text(s, "88.2%", 7.10, 2.10, 1.10, 0.42, 18, True, TEAL, PP_ALIGN.CENTER)
text(s, "Oracle 路由", 7.10, 2.51, 1.10, 0.22, 8.5, False, GRAY, PP_ALIGN.CENTER)
box(s, "Δ = −0.4 pp", 8.34, 2.12, 0.80, 0.42, PALE_BLUE, BLUE, 8.5, True, NAVY)

shape(s, 5.24, 3.18, 4.18, 2.80, WHITE, LINE)
text(s, "关键算子消融：移除后的准确率下降", 5.46, 3.38, 3.72, 0.30,
     11.2, True, NAVY)
for i, (label, value, color) in enumerate([
    ("exhaustive", 18.0, GOLD),
    ("constrained_join", 8.0, BLUE),
    ("path_plan", 6.0, TEAL),
    ("complement", 0.0, BLUE_GRAY),
    ("dual_subgraph", 0.0, BLUE_GRAY),
]):
    y = 3.88 + i * 0.37
    text(s, label, 5.48, y, 1.35, 0.24, 8.4, False, INK)
    shape(s, 6.90, y + 0.03, 1.60, 0.16, PALE_BLUE, None)
    if value > 0:
        shape(s, 6.90, y + 0.03, 1.60 * value / 20, 0.16, color, None)
    text(s, f"−{value:.0f} pp" if value else "0 pp", 8.58, y - 0.01, 0.56, 0.24,
         8.4, True, color, PP_ALIGN.RIGHT)
text(s, "n=100 分层子集", 7.74, 5.70, 1.38, 0.18, 7.7, False, GRAY, PP_ALIGN.RIGHT)

shape(s, 0.78, 6.20, 8.44, 0.56, PALE_GOLD, GOLD)
text(s, "KQA Pro-502 初测：Δ = −0.2 pp，95% CI [−2.4, +2.0]  →  正式公开验证需重建可执行环境并分层报告",
     0.96, 6.34, 8.08, 0.28, 9.8, True, NAVY, PP_ALIGN.CENTER)


# 10 Experiment and public validation ---------------------------------------
s = new_slide("2×2 对照与公开数据集共同验证方法有效性", "实验设计与泛化验证", 10)
text(s, "动作空间", 0.62, 1.38, 0.90, 0.25, 9.3, True, GRAY, PP_ALIGN.CENTER)
text(s, "扁平工具调用", 3.15, 1.38, 1.72, 0.28, 10.5, True, BLUE_GRAY, PP_ALIGN.CENTER)
text(s, "类型化 / 分层决策", 5.66, 1.38, 1.96, 0.28, 10.5, True, BLUE, PP_ALIGN.CENTER)
text(s, "训练方式", 1.10, 2.20, 0.92, 0.26, 9.3, True, GRAY, PP_ALIGN.CENTER)
cells = [
    ("扁平 SFT", "成功轨迹模仿基线", 2.82, 1.82, PALE_BLUE, BLUE_GRAY),
    ("类型化 SFT", "结构化动作 + 成功轨迹", 5.45, 1.82, WHITE, BLUE),
    ("扁平 RL", "扁平空间策略优化", 2.82, 3.05, PALE_BLUE, BLUE_GRAY),
    ("完整方法 RL", "类型化动作 + 多粒度反馈", 5.45, 3.05, PALE_GOLD, GOLD),
]
for a, b, x, y, fill, border in cells:
    shape(s, x, y, 2.22, 0.92, fill, border, width=1.1)
    text(s, a, x + 0.08, y + 0.14, 2.06, 0.28, 11.0, True, NAVY, PP_ALIGN.CENTER)
    text(s, b, x + 0.08, y + 0.52, 2.06, 0.22, 8.5, False, GRAY, PP_ALIGN.CENTER)
text(s, "SFT", 1.58, 2.15, 0.72, 0.28, 10.2, True, GRAY, PP_ALIGN.CENTER)
text(s, "SFT + RL", 1.42, 3.37, 1.04, 0.28, 10.2, True, GRAY, PP_ALIGN.CENTER)
arrow(s, 5.08, 2.15, 0.28, 0.18, BLUE)
text(s, "H1 动作空间设计", 3.92, 2.76, 1.50, 0.24, 9.0, True, BLUE, PP_ALIGN.CENTER)
arrow(s, 6.36, 2.77, 0.22, 0.28, GOLD, "down")
text(s, "H2 执行反馈学习", 6.66, 2.78, 1.50, 0.24, 9.0, True, GOLD)

section_label(s, "多层验证数据", 0.56, 4.34, 1.50)
table(s, ["数据集", "任务结构", "主要验证", "指标"], [
    ["RadarKG-QA", "多跳 / 集合 / 计数 / 恢复", "领域主实验与组合隔离", "EM / F1 / 执行成功率"],
    ["KQA Pro", "约 12 万问题 + KoPL / SPARQL", "程序计划与集合运算", "Plan EM / Answer EM"],
    ["GrailQA（增强）", "64,331 问题；IID / 组合 / 零样本", "资源允许时扩展泛化验证", "EM / F1"],
    ["MetaQA", "1 / 2 / 3-hop 路径", "跨域多跳路径决策", "Hits / F1"],
], 0.62, 4.72, 8.76, 1.78, [1.35, 2.45, 2.52, 2.44], 7.8)
text(s, "统一分层报告：环境 oracle  →  策略执行  →  最终答案；核心实验至少运行 3 个随机种子。",
     0.78, 6.68, 8.44, 0.20, 8.0, False, GRAY, PP_ALIGN.CENTER)


# 11 Innovation and plan -----------------------------------------------------
s = new_slide("两项方法创新沿“建模—学习—验证”推进", "创新点、计划与预期成果", 11)
table(s, ["核心问题", "方法创新", "关键机制", "主要验证"], [
    ["动作空间复杂", "类型化图检索决策建模", "宏动作、类型参数、变量依赖、确定性执行", "2×2 横向对照"],
    ["长程信用分配", "多粒度执行反馈策略学习", "答案、计划、进展、恢复、成本", "2×2 纵向对照"],
], 0.60, 1.42, 8.82, 1.82, [1.40, 2.10, 3.05, 2.27], 8.8)

section_label(s, "研究计划", 0.56, 3.66, 1.30, TEAL)
phases = [
    ("1–2 月", "问题与数据冻结", BLUE),
    ("3–4 月", "类型化环境与基线", TEAL),
    ("5–7 月", "SFT / 策略优化 / 消融", GOLD),
    ("8–10 月", "公开验证与论文写作", BLUE),
]
for i, (period, name, color) in enumerate(phases):
    x = 0.72 + i * 2.17
    box(s, period, x, 4.10, 0.72, 0.42, color, None, 9.0, True, WHITE)
    box(s, name, x + 0.78, 4.10, 1.27, 0.42, WHITE, color, 8.5, True, NAVY)
    if i < len(phases) - 1:
        arrow(s, x + 2.06, 4.22, 0.16, 0.15, BLUE_GRAY)

section_label(s, "预期成果", 0.56, 4.92, 1.30, BLUE)
for i, (a, b) in enumerate([
    ("方法", "类型化决策 + 执行反馈学习"),
    ("数据", "组合隔离雷达问答集"),
    ("实验", "RadarKG / KQA Pro / MetaQA"),
    ("系统", "可执行轨迹展示原型"),
]):
    x = 0.72 + i * 2.17
    text(s, a, x, 5.36, 0.48, 0.28, 9.0, True, TEAL, PP_ALIGN.CENTER)
    text(s, b, x + 0.52, 5.28, 1.50, 0.46, 8.7, True, NAVY,
         PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)

shape(s, 0.78, 6.02, 8.44, 0.68, PALE_GOLD, GOLD)
text(s, "复杂雷达问答 → 类型化可执行决策 → 多粒度反馈学习 → 组合泛化验证",
     1.02, 6.20, 7.96, 0.32, 13.0, True, NAVY, PP_ALIGN.CENTER)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUTPUT))
print(f"saved: {OUTPUT}")
print(f"slides: {len(prs.slides)}; size={prs.slide_width / 914400:.2f}x{prs.slide_height / 914400:.2f}")
