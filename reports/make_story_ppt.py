# -*- coding: utf-8 -*-
"""Narrative deck v3: academic style — tables for data, ONE technical-route figure
for the core method, and every experiment slide = result table + explicit 结论 box.
Run with an env that has python-pptx (minimind).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ACCENT = RGBColor(0x2F, 0x5C, 0x8A)
ACCENT2 = RGBColor(0x3E, 0x8E, 0x7E)
RED = RGBColor(0xC0, 0x4A, 0x3B)
DARK = RGBColor(0x23, 0x27, 0x2B)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xED, 0xF1, 0xF6)
MINT = RGBColor(0xE9, 0xF3, 0xF0)
ROWALT = RGBColor(0xF4, 0xF7, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "微软雅黑"
prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def set_font(run, size=18, bold=False, color=DARK, name=FONT):
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color; run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); rPr.append(el)
        el.set("typeface", name)


def box(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h); tb.text_frame.word_wrap = True
    return tb


def rect(slide, l, t, w, h, color, shape=MSO_SHAPE.RECTANGLE, line=None):
    sp = slide.shapes.add_shape(shape, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    else:
        sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def title_bar(slide, title, kicker=None):
    rect(slide, 0, 0, SW, Inches(0.12), ACCENT)
    tb = box(slide, Inches(0.6), Inches(0.30), SW - Inches(1.2), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    if kicker:
        r = p.add_run(); r.text = kicker + "\n"; set_font(r, 12, True, ACCENT2)
    r = p.add_run(); r.text = title; set_font(r, 25, True, ACCENT)


def bullets(slide, items, left=Inches(0.7), top=Inches(1.5), width=None, height=None,
            size=18, gap=10):
    width = width or (SW - Inches(1.4)); height = height or (SH - top - Inches(0.4))
    tf = box(slide, left, top, width, height).text_frame
    for i, it in enumerate(items):
        text, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.level = lvl
        bullet = "▪ " if lvl == 0 else "– "
        r = p.add_run(); r.text = ("" if text.startswith("  ") else bullet) + text
        set_font(r, size - (2 if lvl else 0), False, DARK if lvl == 0 else GRAY)


def table(slide, headers, rows, left, top, width, height, fs=13, hfs=13,
          hcol=None, hrows=None, col_align=None):
    hrows = hrows or []
    t = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height).table
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = ACCENT
        c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.margin_top = Pt(2); c.margin_bottom = Pt(2)
        pp = c.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
        r = pp.add_run(); r.text = h; set_font(r, hfs, True, WHITE)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.fill.solid()
            c.fill.fore_color.rgb = (LIGHT if hcol == j else
                                     (RGBColor(0xE6, 0xEE, 0xF6) if i in hrows else
                                      (ROWALT if i % 2 == 0 else WHITE)))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.margin_top = Pt(1); c.margin_bottom = Pt(1)
            pp = c.text_frame.paragraphs[0]
            if col_align and j < len(col_align):
                pp.alignment = col_align[j]
            else:
                pp.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            r = pp.add_run(); r.text = str(val)
            set_font(r, fs, (hcol == j or i in hrows),
                     ACCENT if (hcol == j or i in hrows) else DARK)
    return t


def concl(slide, text, top, height=Inches(1.05)):
    left = Inches(0.7); width = SW - Inches(1.4)
    rect(slide, left, top, width, height, MINT)
    rect(slide, left, top, Inches(0.13), height, ACCENT2)
    tag = box(slide, left + Inches(0.32), top + Inches(0.12), Inches(1.4), Inches(0.4))
    r = tag.text_frame.paragraphs[0].add_run(); r.text = "结论"; set_font(r, 15, True, ACCENT2)
    tf = box(slide, left + Inches(1.5), top + Inches(0.10), width - Inches(1.8), height - Inches(0.2)).text_frame
    if isinstance(text, str):
        text = [text]
    for i, line in enumerate(text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(3)
        r = p.add_run(); r.text = line; set_font(r, 14, False, DARK)


def note(slide, text, top=None):
    top = top or (SH - Inches(0.55))
    tf = box(slide, Inches(0.7), top, SW - Inches(1.4), Inches(0.45)).text_frame
    r = tf.paragraphs[0].add_run(); r.text = text; set_font(r, 10.5, False, GRAY)


# ===================================================== 1 cover
s = prs.slides.add_slide(BLANK)
rect(s, 0, Inches(2.3), SW, Inches(0.10), ACCENT)
rect(s, 0, Inches(4.55), SW, Inches(0.04), ACCENT2)
tf = box(s, Inches(1.0), Inches(2.55), SW - Inches(2.0), Inches(2.0)).text_frame
r = tf.paragraphs[0].add_run(); r.text = "Beyond Top-K"; set_font(r, 44, True, ACCENT)
p = tf.add_paragraph(); r = p.add_run()
r.text = "面向「答案几何不匹配」的雷达知识图谱问答系统"; set_font(r, 23, True, DARK)
p = tf.add_paragraph(); p.space_before = Pt(8); r = p.add_run()
r.text = "诊断 top-K 的结构性失败，设计类型化检索算子套件"; set_font(r, 15, False, GRAY)
tf2 = box(s, Inches(1.0), Inches(4.85), SW - Inches(2.0), Inches(0.8)).text_frame
r = tf2.paragraphs[0].add_run()
r.text = "项目工作汇报   ·   需求 · 缺陷 · 方法 · 实验 · 成果"; set_font(r, 14, False, GRAY)

# ===================================================== 2 domain need
s = prs.slides.add_slide(BLANK); title_bar(s, "领域需求：工业知识图谱问答", "① 需求")
bullets(s, [
    "场景：装备 / 工业领域知识库（本项目以雷达装备为载体）需要可审计、可溯源的精确问答",
    "真实用户问题远不止「查一个事实」，而是大量分析型查询：",
    ("计数：「美国一共运营多少款雷达？」（答案集 >400）", 1),
    ("多约束筛选：「美国研制且工作在 S 波段的雷达有哪些？」", 1),
    ("否定 / 缺失：「AN/TPY-2 是否出口到日本？」", 1),
    "工业部署还要求：成本可控、证据可审计、跨语言（schema 英文 + value 中英混杂）",
], size=17, gap=12)
concl(s, "这类「分析型 KGQA」正是现有系统最薄弱的一环——也是本项目要攻克的需求缺口。",
      top=Inches(5.7))

# ===================================================== 3 flaw (TABLE + concl)
s = prs.slides.add_slide(BLANK); title_bar(s, "现有方法的缺陷：top-K 在分析型问题上崩溃", "② 问题")
tf = box(s, Inches(0.7), Inches(1.35), Inches(12), Inches(0.5)).text_frame
r = tf.paragraphs[0].add_run()
r.text = "GraphRAG / RoG / ToG 等统一使用 top-K 检索。top-K baseline 按题型的准确率："
set_font(r, 15, False, DARK)
table(s, ["题型", "代表问题", "top-K 准确率"],
    [["agg_count 计数", "美国有多少款雷达", "4.0%"],
     ["attr_filter 多约束", "美国研制 + S 波段", "12.0%"],
     ["three_hop 三跳", "X 的研制方的所属国的…", "23.3%"],
     ["relation_inverse 反查", "谁研制了哪些雷达", "40.0%"],
     ["single_hop 单跳事实", "AN/TPY-2 谁研制", "83.8%"],
     ["negation 否定", "是否出口日本", "100.0%"]],
    Inches(0.9), Inches(1.85), Inches(11.5), Inches(3.3), fs=14, hfs=14, hcol=2)
concl(s, ["分析型题（计数/多约束/多跳）仅 4–40%，事实型题 80–100%。",
          "缺陷集中且系统性——不是 LLM 不行，是 top-K 检索范式与问题不匹配。"],
      top=Inches(5.45))

# ===================================================== 4 diagnosis AGM (detailed)
s = prs.slides.add_slide(BLANK); title_bar(s, "诊断：Answer-Geometry Mismatch (AGM)", "③ 为什么失败")
bullets(s, [
    "top-K 隐含一个关于「答案几何形状」的假设：答案是可被相关性排序 capture 的小集合。",
    "三类工业 KG 常见模式从信息论上违反此假设（与 LLM 能力无关）：",
    ("无界枚举：答案集无上界，但 K=8 永远装不下 >400 的答案", 1),
    ("多约束交集：ranking 把多约束压成一个标量，无法表达 AND", 1),
    ("补集 / 否定：KG 只存正面三元组，「缺失」不在任何局部窗口里", 1),
    "类比数据库 OLTP / OLAP：top-K 是 GraphRAG 的「B-tree」——适合点查，",
    ("对聚合/分析型查询结构性不匹配。数据库的解法不是改 B-tree，而是加「执行计划」。", 1),
], size=16, gap=10)
concl(s, "AGM 暴露度是基准「问题分布」的属性，不是 top-K 本身——这解释了为什么换个数据集增益会变。",
      top=Inches(5.85), height=Inches(0.9))

# ===================================================== 5 related work
s = prs.slides.add_slide(BLANK); title_bar(s, "现有方法全景与我们的定位", "④ 相关工作")
table(s, ["方法", "改变什么", "检索原语", "能否解决 AGM"],
    [["GraphRAG / HippoRAG", "更聪明的 top-K", "ranking 不变", "✗ 仍受 K 限"],
     ["RoG / ToG", "路径规划", "ranking 不变", "✗ 单一策略"],
     ["Adaptive-RAG", "检索深度 (0/1/N 步)", "ranking 不变", "✗ 每步仍 top-K"],
     ["ByoKG-RAG", "检索来源 (多工具融合)", "ranking 不变", "✗ 仍受融合预算"],
     ["本工作", "检索语义 (每算子不同 KG 操作)", "交集/枚举/补集/路径", "✓ 按需求选算子"]],
    Inches(0.7), Inches(1.7), Inches(11.9), Inches(3.4), fs=14, hfs=14, hrows=[5])
concl(s, ["他人都在「同一个 ranking 原语上变深度 / 变来源」；我们改变的是检索的语义本身。",
          "→ 他们的增益无法靠重配置达到，因为底层原语仍受 top-K 的 AGM 限制。"],
      top=Inches(5.4))

# ===================================================== 6 CORE INNOVATION 1: 技术路线图 (the ONE figure)
s = prs.slides.add_slide(BLANK); title_bar(s, "核心创新（一）：题型路由 + 类型化算子（技术路线）", "⑤ 创新点")
# Question
qb = rect(s, Inches(0.7), Inches(1.55), Inches(2.2), Inches(0.75), DARK, MSO_SHAPE.ROUNDED_RECTANGLE)
p = qb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "问题 Question"; set_font(r, 13, True, WHITE)
# Dispatcher
db = rect(s, Inches(3.4), Inches(1.55), Inches(2.6), Inches(0.75), ACCENT2, MSO_SHAPE.ROUNDED_RECTANGLE)
p = db.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Dispatcher（LLM）\n题型分类 · O(1) 查表"; set_font(r, 12, True, WHITE)
rect(s, Inches(3.0), Inches(1.78), Inches(0.35), Inches(0.3), GRAY, MSO_SHAPE.RIGHT_ARROW)
# 6 operator boxes row
ops = [("lookup", "相关性检索"), ("exhaustive", "全集枚举 (无K)"),
       ("constrained-join", "集合交 ∩"), ("complement", "补集/非成员"),
       ("path-plan", "路径 r·r⁻¹"), ("dual-subgraph", "双子图对比")]
ox = Inches(0.7); oy = Inches(3.0); ow = Inches(1.97); oh = Inches(1.15); ogx = Inches(0.13)
for i, (nm, op) in enumerate(ops):
    col = ACCENT if i in (1, 2, 3) else ACCENT2
    bx = rect(s, ox, oy, ow, oh, col, MSO_SHAPE.ROUNDED_RECTANGLE)
    tfr = bx.text_frame; p = tfr.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = nm; set_font(r, 12, True, WHITE)
    p2 = tfr.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = op; set_font(r, 10.5, False, WHITE)
    ox = Emu(int(ox) + int(ow) + int(ogx))
# fan arrow (dispatcher down to operators band)
rect(s, Inches(4.5), Inches(2.35), Inches(0.3), Inches(0.5), GRAY, MSO_SHAPE.DOWN_ARROW)
# Executor + Answerer
eb = rect(s, Inches(2.0), Inches(4.6), Inches(4.2), Inches(0.7), DARK, MSO_SHAPE.ROUNDED_RECTANGLE)
p = eb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Executor（仅查 KG）→ 结构化证据"; set_font(r, 12, True, WHITE)
ab = rect(s, Inches(7.0), Inches(4.6), Inches(4.2), Inches(0.7), ACCENT2, MSO_SHAPE.ROUNDED_RECTANGLE)
p = ab.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Answerer（LLM）→ 答案"; set_font(r, 12, True, WHITE)
rect(s, Inches(6.25), Inches(4.78), Inches(0.7), Inches(0.32), GRAY, MSO_SHAPE.RIGHT_ARROW)
concl(s, "核心思想：不做更好的 ranking，而是按问题的「信息需求」把它路由到执行结构性不同 KG 操作的算子。变的是检索语义。",
      top=Inches(5.7), height=Inches(0.95))

# ===================================================== 7 CORE INNOVATION 2: operator derivation + mechanism
s = prs.slides.add_slide(BLANK); title_bar(s, "核心创新（二）：六算子从信息需求推导（非经验拼凑）", "⑤ 创新点")
table(s, ["算子", "KG 操作机制", "服务的题型（11→6 映射）"],
    [["exhaustive", "去掉 K 截断，返回全集", "agg_count / agg_enum / relation_inverse"],
     ["constrained-join", "集合交 ∩ + 等价类回退", "attr_filter"],
     ["complement", "枚举已知 tail 后判定非成员", "negation / unanswerable"],
     ["path-plan", "关系链遍历 + 反向 r⁻¹", "two_hop_bridge / three_hop_chain"],
     ["dual-subgraph", "并行 tails + 集合对比", "set_compare"],
     ["lookup", "BM25⊕向量⊕RRF（安全兜底）", "single_hop / distractor"]],
    Inches(0.7), Inches(1.6), Inches(11.9), Inches(3.5), fs=13.5, hfs=13.5, hcol=2)
concl(s, ["每个算子=满足某类「信息需求」的最小访问模式 → 推导同时决定了套件的数量(6)和内容。",
          "这不是「试了试发现好用」，而是从 top-K 无法满足的信息需求类反推；但不主张形式完备（诚实边界）。"],
      top=Inches(5.4))

# ===================================================== 8 CORE INNOVATION 3: why真创新
s = prs.slides.add_slide(BLANK); title_bar(s, "核心创新（三）：为什么是方法创新，而非分类器/工程 trick", "⑤ 创新点")
bullets(s, [
    "差异化定位：相关工作变「检索深度 / 来源」，我们变「检索语义」——每个算子是结构性不同的 KG 操作。",
    "派发器只是 O(1) 查表，不是创新主体；真正承重的是算子套件本身。",
    "关键证据（oracle 实验预告）：用 gold 题型替换 LLM 派发，端到端 88.2% vs LLM 88.6%，Δ 仅 −0.4pp。",
    ("→ 全部 +35.9pp 增益来自算子设计，与分类质量无关。「这是一篇关于算子的论文，不是关于分类器」。", 1),
    "副创新（方法学发现）：双语别名层在「自动生成题」上消融 0pp（看似无用），但在「注入英文缩写的 OOD 题」上 +11.5pp（实则关键）。",
    ("→ 别名/归一化类组件若只在分布内评测会得假阴性，必须用 OOD 替换集才测得出价值——可推广的评测教训。", 1),
], size=15, gap=11)
concl(s, "创新不在「做了分类」，而在 ①AGM 诊断（为什么 top-K 必然失败）②按信息需求推导的算子套件（怎么修）。",
      top=Inches(5.85), height=Inches(0.9))

# ===================================================== 9 EXP1 main results
s = prs.slides.add_slide(BLANK); title_bar(s, "实验①  主结果：三方对比（全 499 题）", "⑥ 实验")
table(s, ["题型", "对应算子", "Baseline", "RoG-zs", "Strategy", "Δ vs B (95% CI)"],
    [["agg_count", "exhaustive", "4.0", "38.0", "62.0", "+58 [+44,+72]"],
     ["attr_filter", "constrained-join", "12.0", "0.0", "92.0", "+80 [+66,+92]"],
     ["three_hop_chain", "path-plan", "23.3", "50.0", "93.3", "+70 [+53,+87]"],
     ["relation_inverse", "exhaustive", "40.0", "74.0", "86.0", "+46 [+32,+60]"],
     ["single_hop", "lookup", "83.8", "11.2", "86.2", "+2.5 [+0,+6]"],
     ["OVERALL (n=499)", "— (全 6 算子)", "52.7", "60.3", "88.6", "+35.9 [+31.7,+40.1]"]],
    Inches(0.6), Inches(1.65), Inches(12.1), Inches(3.5), fs=12.5, hfs=12.5,
    hcol=4, hrows=[6])
concl(s, ["Strategy 88.6%，相对 baseline +35.9pp、相对 RoG +28.3pp，9/11 题型配对 bootstrap 显著（CI>0）。",
          "K=8→20 检索预算下 Strategy 仍 +33.5pp → 增益不是「K 太小」的 artifact。成本 $0.00041/题。"],
      top=Inches(5.4))

# ===================================================== 10 EXP2 attribution
s = prs.slides.add_slide(BLANK); title_bar(s, "实验②  增益归因：来自算子，不是分类器", "⑥ 实验")
table(s, ["对照", "端到端准确率", "说明"],
    [["LLM 派发（默认）", "88.6%", "zero-shot 题型分类 93%"],
     ["Oracle 派发（gold 路由）", "88.2%", "完美分类的上界"],
     ["Δ（Oracle − LLM）", "−0.4pp", "派发器噪声 ≈ 0"]],
    Inches(0.9), Inches(1.7), Inches(11.0), Inches(2.1), fs=14, hfs=14, hrows=[3])
table(s, ["检索预算消融", "Baseline", "Strategy", "Δ"],
    [["K=8 → K=20", "52.7 → 55.1", "88.6", "仍 +33.5 [+29.5,+37.9]"]],
    Inches(0.9), Inches(4.0), Inches(11.0), Inches(1.0), fs=14, hfs=14, hcol=3)
concl(s, ["增益 100% 来自算子设计，与分类质量、检索预算无关——两个消融分别排除了这两个 confound。"],
      top=Inches(5.5), height=Inches(0.9))

# ===================================================== 11 EXP3 per-operator
s = prs.slides.add_slide(BLANK); title_bar(s, "实验③  单算子消融：每个算子的边际贡献", "⑥ 实验")
table(s, ["关掉的算子", "整体掉幅", "最受影响题型", "性质"],
    [["exhaustive", "−19 pp", "agg_count/enum −60", "load-bearing"],
     ["path-plan", "−10 pp", "three_hop −62", "load-bearing"],
     ["constrained-join", "−8 pp", "attr_filter −90", "load-bearing"],
     ["complement", "0 pp", "—", "本基准下可降级"],
     ["dual-subgraph", "0 pp", "—", "本基准下可降级"]],
    Inches(0.7), Inches(1.65), Inches(11.9), Inches(3.2), fs=14, hfs=13.5, hcol=1)
concl(s, ["三件 load-bearing 算子的掉幅线性可加（−19−10=−29）→ 各自针对结构性不同、互不相交的失败模式。",
          "两件 0pp 算子保留作 derivation 完备性 + 证据可审计（对抗集下预期会承重）。"],
      top=Inches(5.3))

# ===================================================== 12 EXP4 AGM predicts / cross-domain
s = prs.slides.add_slide(BLANK); title_bar(s, "实验④  AGM 暴露度可预测增益（跨域可证伪）", "⑥ 实验")
table(s, ["题型分组", "代表题型", "中位 Δ(S−B)"],
    [["AGM-exposed（高基数/多约束/多跳）", "attr_filter, agg_count, three_hop…", "+58 pp"],
     ["non-exposed（事实/可降级）", "single_hop, negation, distractor…", "0 pp"]],
    Inches(0.7), Inches(1.7), Inches(11.9), Inches(1.5), fs=14, hfs=13.5, hcol=2)
table(s, ["跨域 KQA Pro（Wikidata）", "整体 Δ", "SelectBetween（唯一干净 AGM 题型）"],
    [["AGM 暴露度低（Count 中位 2）", "±0.2pp 平局", "+11.1pp [+2.2,+22.2] 显著"]],
    Inches(0.7), Inches(3.5), Inches(11.9), Inches(1.0), fs=13.5, hfs=13)
concl(s, ["增益由基准的 AGM 暴露度决定——KQA Pro 平局是预测的成功而非失败（它本就缺 AGM 题型）。",
          "Strategy 是「结构性保险」：有 AGM 的分布上大赢，没有时持平。这是可证伪的强主张。"],
      top=Inches(5.0))

# ===================================================== 13 EXP5 faithful RoG
s = prs.slides.add_slide(BLANK); title_bar(s, "实验⑤  忠实微调 RoG：驳倒「弱 baseline」质疑", "⑥ 实验 · 最新")
table(s, ["系统", "OVERALL", "说明"],
    [["Baseline", "52.7", "BM25+向量+RRF"],
     ["RoG-zs (DeepSeek)", "60.3", "主表 RoG（zero-shot）"],
     ["RoG-zs (Qwen-7B)", "31.5", "同 base，未微调"],
     ["RoG-FT (Qwen-7B)", "61.7", "LoRA 微调 planner"],
     ["Strategy", "88.6", "本工作"]],
    Inches(0.9), Inches(1.65), Inches(8.0), Inches(3.4), fs=14, hfs=14, hrows=[4, 5])
bullets(s, [
    "微调效应：",
    ("+30.3pp", 1),
    ("[+25.9,+34.7]", 1),
    "→ 排除弱 baseline",
    "",
    "Strategy−RoG-FT：",
    ("+26.9pp", 1),
    ("[+22.6,+31.3]", 1),
], left=Inches(9.3), top=Inches(1.7), width=Inches(3.6), size=14, gap=7)
concl(s, ["微调让 planner 逐题型「重新发明」exhaustive（agg_count 2→64 追平 Strategy），",
          "但单路径范式无法表达交集（attr_filter 仅 28 vs 92）→ Strategy 仍领先 +26.9pp。"],
      top=Inches(5.35))

# ===================================================== 14 status
s = prs.slides.add_slide(BLANK); title_bar(s, "当前成果", "⑦ 成果")
bullets(s, [
    "论文 v12.2（19 页，中英双版本），目标 NAACL / EMNLP Industry Track 2026",
    "三层贡献：① AGM 诊断（analytical）② 六算子套件（system）③ RadarKG-QA-499（benchmark）",
    ("＋ 副产物：双语层悖论（multilingual KGQA 评测方法学发现，可拆 short paper）", 1),
    "完整实验矩阵：主结果 + oracle + K=20 + 单算子 + suite-size + 双语 + KQA Pro 4-config + 忠实微调 RoG",
    "工程：9 源 KG 融合 + 混合检索 + 4400 次 LLM 调用工程 + 13 套消融脚本（≈8K 行）",
    "工件：代码 Apache-2.0 + KG/基准 CC-BY-4.0，全流程复现 ≈ $1.5",
], size=16, gap=11)

# ===================================================== 15 future
s = prs.slides.add_slide(BLANK); title_bar(s, "后续计划", "⑧ 计划")
bullets(s, [
    "投稿前（高 ROI）：arXiv preprint；论文转 ACL 模板；核对 ByoKG-RAG 引用",
    "拆 short paper：「Bilingual Paradox in Multilingual KGQA Eval」（方法学发现单独投 workshop）",
    "LLaMA-2 planner 复现：补完全忠实的 RoG（当前用 Qwen-7B，因服务器无外网）",
    "Schema-derived parser：从任意 KG schema 自动衍生 parser → 跨域迁移从 1 天降到零成本",
    "跨域迁移：DrugBank / Materials Project（天然高基数+多约束，验证 AGM 普适性）",
    "对抗集：KG 模糊否定 + 3-way 比较，验证 complement / dual-subgraph 变 load-bearing",
], size=16, gap=11)

# ===================================================== 16 takeaway
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, ACCENT)
tf = box(s, Inches(1.1), Inches(2.4), SW - Inches(2.2), Inches(2.8)).text_frame
r = tf.paragraphs[0].add_run(); r.text = "一句话总结"; set_font(r, 18, True, RGBColor(0xBF, 0xD4, 0xE8))
p = tf.add_paragraph(); p.space_before = Pt(14); r = p.add_run()
r.text = "GraphRAG 不缺更聪明的 top-K，缺的是「为查询的信息需求选对检索算子」。"
set_font(r, 26, True, WHITE)
p = tf.add_paragraph(); p.space_before = Pt(16); r = p.add_run()
r.text = "RadarKG-QA-499 上 +35.9pp，增益 100% 来自算子；微调一个 planner 也只能逐题型重新发明它们，无法在单路径范式内复制交集与补集。"
set_font(r, 16, False, RGBColor(0xDD, 0xE8, 0xF2))

import datetime
OUT = "雷达GraphRAG_项目汇报_v3.pptx"
try:
    prs.save(OUT)
except PermissionError:
    OUT = "雷达GraphRAG_项目汇报_v3_" + datetime.datetime.now().strftime("%H%M") + ".pptx"
    prs.save(OUT)
print("saved:", OUT, "slides:", len(prs.slides._sldIdLst))
