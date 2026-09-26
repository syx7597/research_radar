# -*- coding: utf-8 -*-
"""Generate a clean summary PPTX of the Radar GraphRAG project.
Run with an env that has python-pptx (e.g. minimind).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ---- palette ----
ACCENT = RGBColor(0x2F, 0x5C, 0x8A)   # steel blue
ACCENT2 = RGBColor(0x3E, 0x8E, 0x7E)  # teal (secondary)
DARK = RGBColor(0x23, 0x27, 0x2B)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xED, 0xF1, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEADFILL = RGBColor(0x2F, 0x5C, 0x8A)
ROWALT = RGBColor(0xF4, 0xF7, 0xFA)
FONT = "微软雅黑"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def set_font(run, size=18, bold=False, color=DARK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", name)


def box(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tb.text_frame.word_wrap = True
    return tb


def rect(slide, l, t, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def add_title_bar(slide, title, kicker=None):
    rect(slide, 0, 0, SW, Inches(0.12), ACCENT)
    tb = box(slide, Inches(0.6), Inches(0.35), SW - Inches(1.2), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    if kicker:
        r = p.add_run(); r.text = kicker + "\n"; set_font(r, 12, True, ACCENT2)
    r = p.add_run(); r.text = title; set_font(r, 27, True, ACCENT)
    return tb


def bullets(slide, items, left=Inches(0.7), top=Inches(1.5),
            width=None, height=None, size=18, gap=10):
    width = width or (SW - Inches(1.4))
    height = height or (SH - top - Inches(0.5))
    tb = box(slide, left, top, width, height)
    tf = tb.text_frame
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            text, lvl = it
        else:
            text, lvl = it, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.level = lvl
        bullet = "▪ " if lvl == 0 else "– "
        r = p.add_run()
        r.text = ("" if text.startswith("  ") else bullet) + text
        set_font(r, size - (2 if lvl else 0), bold=(lvl == 0 and text.endswith("：")),
                 color=DARK if lvl == 0 else GRAY)
        if lvl:
            p.left_indent = Inches(0.4)
    return tb


def add_table(slide, headers, rows, left, top, width, height,
              fs=13, hfs=13, highlight_col=None, highlight_rows=None):
    nr, nc = len(rows) + 1, len(headers)
    gtbl = slide.shapes.add_table(nr, nc, left, top, width, height).table
    highlight_rows = highlight_rows or []
    for j, h in enumerate(headers):
        c = gtbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = HEADFILL
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_top = Pt(2); c.margin_bottom = Pt(2)
        para = c.text_frame.paragraphs[0]; para.alignment = PP_ALIGN.CENTER
        r = para.add_run(); r.text = h; set_font(r, hfs, True, WHITE)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = gtbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = (LIGHT if (highlight_col == j) else
                                     (ROWALT if i % 2 == 0 else WHITE))
            if i in highlight_rows:
                c.fill.fore_color.rgb = RGBColor(0xE6, 0xEE, 0xF6)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = Pt(1); c.margin_bottom = Pt(1)
            para = c.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = para.add_run(); r.text = str(val)
            bold = (highlight_col == j) or (i in highlight_rows) or (j == 0 and i in highlight_rows)
            set_font(r, fs, bold=bold,
                     color=ACCENT if (highlight_col == j or i in highlight_rows) else DARK)
    return gtbl


def note(slide, text, top=None):
    top = top or (SH - Inches(0.62))
    tb = box(slide, Inches(0.7), top, SW - Inches(1.4), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text; set_font(r, 11, False, GRAY)


# ============================================================ SLIDE 1: cover
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, WHITE)
rect(s, 0, Inches(2.4), SW, Inches(0.10), ACCENT)
rect(s, 0, Inches(4.7), SW, Inches(0.04), ACCENT2)
tb = box(s, Inches(1.0), Inches(2.7), SW - Inches(2.0), Inches(1.7))
p = tb.text_frame.paragraphs[0]
r = p.add_run(); r.text = "Beyond Top-K"; set_font(r, 44, True, ACCENT)
p2 = tb.text_frame.add_paragraph()
r = p2.add_run(); r.text = "面向「答案几何不匹配」的雷达知识图谱问答"; set_font(r, 24, True, DARK)
p3 = tb.text_frame.add_paragraph(); p3.space_before = Pt(8)
r = p3.add_run(); r.text = "A Typed Operator Suite for Answer-Geometry Mismatches in KGQA"
set_font(r, 15, False, GRAY)
tb2 = box(s, Inches(1.0), Inches(5.0), SW - Inches(2.0), Inches(1.0))
for txt in ["项目工作总结  ·  方法 / 成果 / 后续计划",
            "目标会议：NAACL / EMNLP Industry Track 2026   ·   论文 v12.2（19 页）"]:
    p = tb2.text_frame.add_paragraph()
    r = p.add_run(); r.text = txt; set_font(r, 14, False, GRAY)

# ============================================================ SLIDE 2: 一页概览
s = prs.slides.add_slide(BLANK); add_title_bar(s, "一页概览", "EXECUTIVE SUMMARY")
bullets(s, [
    "问题：GraphRAG 普遍用统一的 top-K 检索，但 ~30% 的自然 KGQA 问题（计数 / 多约束 / 否定）",
    ("top-K 在信息论意义上无法回答——这是结构性失败，与 LLM 能力无关", 1),
    "诊断：提出 Answer-Geometry Mismatch (AGM)——top-K 隐含「答案是可排序的小集合」假设；类比数据库 OLTP/OLAP",
    "方法：六个「类型化检索算子」套件（按信息需求推导），轻量 LLM 派发器 O(1) 查表",
    ("oracle 实验证明：全部增益来自算子，不来自分类器（Δ −0.4pp）", 1),
    "核心成果：自建 RadarKG-QA-499 上 88.6%，相对 baseline +35.9pp [+31.7,+40.1]（配对 bootstrap 显著）",
    "产出：1.6万三元组中文 KG + 499 题分层基准 + 论文 v12.2 + 13 套消融 + 忠实微调 RoG 对照",
], size=17, gap=11)

# ============================================================ SLIDE 3: 背景动机
s = prs.slides.add_slide(BLANK); add_title_bar(s, "背景：top-K 检索的结构性失败", "MOTIVATION")
box(s, Inches(0.7), Inches(1.35), SW - Inches(1.4), Inches(0.5)).text_frame.paragraphs[0].add_run().text = ""
add_table(s,
    ["问题形态", "top-K 为什么失败"],
    [["「美国一共运营多少款雷达？」", "答案集 >400，任何固定 K 都枚举不完"],
     ["「列出美国研制且 S 波段的雷达」", "ranking 无法表达跨约束的 AND 交集"],
     ["「AN/TPY-2 是否出口到日本？」", "KG 只存正面事实，缺失非局部、不在任何窗口里"]],
    Inches(0.9), Inches(1.7), Inches(11.5), Inches(2.6), fs=16, hfs=15)
bullets(s, [
    "这些不是 LLM 失败：K=8 时无法返回 400 个答案、无法表达 AND、无法宣告「无」",
    "baseline top-K 在这三类题上仅 4–12%，与 LLM 选择、K∈{8,20} 无关",
], top=Inches(4.7), size=16, gap=10)

# ============================================================ SLIDE 4: AGM 诊断
s = prs.slides.add_slide(BLANK); add_title_bar(s, "核心诊断：Answer-Geometry Mismatch (AGM)", "DIAGNOSIS")
bullets(s, [
    "top-K 隐含假设：答案是一个可被 surface relevance 排序 capture 的小集合",
    "三类工业 KG 常见模式系统性违反此假设：",
    ("无界枚举（计数/列举）— 需完整集合，top-K 受 K 上界约束", 1),
    ("多约束交集（AND）— 需集合交，ranking 混淆约束", 1),
    ("补集/非成员（否定）— 需全局否定，KG 只存正面三元组", 1),
    "类比数据库：top-K 是 GraphRAG 的「B-tree」——适合点查，对分析型查询结构性不匹配",
    "AGM 暴露度是基准分布的属性，不是 top-K 本身——公开基准系统性欠采样此模式",
], size=17, gap=11)

# ============================================================ SLIDE 5: 方法-算子
s = prs.slides.add_slide(BLANK); add_title_bar(s, "方法：从信息需求推导六算子套件", "METHOD")
add_table(s,
    ["计算模式", "信息需求", "推导出的算子"],
    [["身份 / factoid", "一条最匹配三元组", "lookup (BM25⊕vec⊕RRF)"],
     ["无界枚举 / 计数", "完整 head/tail 集", "exhaustive（无 K 截断）"],
     ["多约束交集", "N 关系 head 集的交", "constrained-join（+等价类回退）"],
     ["补集 / 非成员", "完整已知集 + 显式 gap", "complement（枚举后判定）"],
     ["关系组合 k≥2", "保留 hop 结构的路径", "path-plan（含 r⁻¹ 反向）"],
     ["二元比较", "双侧子图显式对齐", "dual-subgraph（并行+集合操作）"]],
    Inches(0.7), Inches(1.6), Inches(11.9), Inches(3.6), fs=14, hfs=14, highlight_col=2)
note(s, "每个算子是满足该行信息需求的「最小访问模式」；推导决定了套件的数量(6)和内容（非经验拼凑，但不主张形式完备）")

# ============================================================ SLIDE 6: 系统架构
s = prs.slides.add_slide(BLANK); add_title_bar(s, "系统：Strategy-Routed GraphRAG", "ARCHITECTURE")
# pipeline boxes
stages = [("Dispatcher", "题型→算子\n(LLM, O(1)查表)", ACCENT2),
          ("Parser", "抽结构化参数\n(LLM)", ACCENT2),
          ("Executor", "6 算子之一\n(仅查 KG)", ACCENT),
          ("Answerer", "证据→答案\n(LLM)", ACCENT2)]
x = Inches(0.7); y = Inches(1.9); w = Inches(2.7); h = Inches(1.4); gapx = Inches(0.35)
for i, (name, desc, col) in enumerate(stages):
    sp = rect(s, x, y, w, h, col)
    tf = sp.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = name; set_font(r, 16, True, WHITE)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = desc; set_font(r, 11, False, WHITE)
    x = Emu(int(x) + int(w) + int(gapx))
bullets(s, [
    "三个 LLM 阶段 + 一个仅查 KG 的执行器；KG 只在 executor 被查询，LLM 不直接访问",
    "派发器是 O(1) 查表——oracle 消融（gold 路由）88.2% vs LLM 88.6%，Δ −0.4pp",
    ("→ 全部 +35.9pp 增益来自算子设计，不来自分类器。「这是一篇关于算子的论文」", 1),
    "跨切面双语别名层（5 级级联）处理工业中文 KG 的 schema 英文 + value 中英混杂",
], top=Inches(3.7), size=16, gap=10)

# ============================================================ SLIDE 7: 基准
s = prs.slides.add_slide(BLANK); add_title_bar(s, "基准：RadarKG-QA-499", "BENCHMARK & DATA")
bullets(s, [
    "RadarKG-v2：4,827 实体 / 12,219 实体间边 / 16,513 平铺三元组 / 98 关系+属性",
    ("源：472 页机载雷达手册全 OCR + Wikidata + Wikipedia；country/developer 覆盖 93.4%/87.0%", 1),
    "RadarKG-QA-499：499 题分层 11 题型，机器可验证 gold + 26 题 OOD 双语压力集",
], top=Inches(1.4), size=16, gap=9, height=Inches(2.0))
add_table(s,
    ["基准", "Count 中位", "中位≥30", "备注"],
    [["RadarKG-QA-499（我们）", "14", "18%", "首个为暴露 AGM 设计"],
     ["KQA Pro val", "2", "6%", ""],
     ["Mintaka dev", "4", "3%", ""],
     ["WebQSP", "—", "≈0%", "88% single-hop"]],
    Inches(0.9), Inches(3.7), Inches(11.0), Inches(2.6), fs=14, hfs=14, highlight_rows=[1])
note(s, "首个明确按高基数答案集分层的 KGQA 基准（Count 中位是次近公开基准的 3.4×）—— 把「自建 bench」从弱点变成贡献")

# ============================================================ SLIDE 8: 主结果
s = prs.slides.add_slide(BLANK); add_title_bar(s, "主结果：3-way 对比（全 499 题）", "MAIN RESULTS")
add_table(s,
    ["题型", "Baseline", "RoG-zs", "Strategy", "Δ vs B"],
    [["agg_count", "4.0", "38.0", "62.0", "+58"],
     ["agg_enum", "46.0", "92.0", "96.0", "+50"],
     ["attr_filter", "12.0", "0.0", "92.0", "+80"],
     ["three_hop_chain", "23.3", "50.0", "93.3", "+70"],
     ["relation_inverse", "40.0", "74.0", "86.0", "+46"],
     ["single_hop", "83.8", "11.2", "86.2", "+2.5"],
     ["OVERALL (n=499)", "52.7", "60.3", "88.6", "+35.9"]],
    Inches(0.9), Inches(1.6), Inches(8.2), Inches(3.9), fs=13, hfs=13,
    highlight_col=3, highlight_rows=[7])
bullets(s, [
    "Strategy 88.6%",
    ("vs Baseline +35.9pp", 1),
    ("  [+31.7, +40.1]", 1),
    ("vs RoG +28.3pp", 1),
    ("  [+23.8, +32.7]", 1),
    "9/11 题型显著胜",
    "2 处「失利」CI 跨 0=打平",
    "成本 $0.00041/题",
], left=Inches(9.5), top=Inches(1.6), width=Inches(3.4), size=14, gap=7)
note(s, "配对 bootstrap 1000 resamples；CI 严格 >0 → 统计显著。K=8→20 检索预算消融下 Strategy 仍 +33.5pp 显著（非 K-artifact）")

# ============================================================ SLIDE 9: 消融
s = prs.slides.add_slide(BLANK); add_title_bar(s, "关键消融：增益从哪来", "ABLATIONS")
add_table(s,
    ["实验", "发现"],
    [["Oracle dispatcher", "gold 路由 88.2 vs LLM 88.6（Δ −0.4）→ 增益 100% 来自算子"],
     ["K=20 检索预算", "baseline 仅 +2.4pp；Strategy 仍 +33.5pp 显著 → 非 K-artifact"],
     ["单算子消融", "exhaustive −19 / path-plan −10 / constrained-join −8（可加，互不相交）"],
     ["Suite-size", "4-op = 6-op = 94%；load-bearing 三件套贡献精确可加"],
     ["双语层悖论", "in-dist 0pp，OOD +11.5pp [+0.0,+26.9] → 分布内评测假阴性（方法学发现）"]],
    Inches(0.7), Inches(1.7), Inches(11.9), Inches(3.5), fs=14, hfs=14, highlight_col=1)
note(s, "每个 load-bearing 算子修复其 AGM-class 推导预测的失败模式（1:1 对应）；0pp 算子保留作 derivation 完备性 + 可审计性")

# ============================================================ SLIDE 10: AGM 可预测
s = prs.slides.add_slide(BLANK); add_title_bar(s, "AGM 暴露度可预测 Δ（跨域可证伪）", "RQ5 · CROSS-DOMAIN")
bullets(s, [
    "可证伪预测：per-type Δ 应随该题型的「AGM 暴露度」（基数 + 组合）增长",
    "RadarKG：6 个 AGM-exposed 题型 中位 Δ = +58pp；5 个 non-exposed = 0pp",
    "KQA Pro（Wikidata 跨域）：整体 ±0.2pp 平局——但这是预测的成功，非失败：",
    ("KQA Pro 的 AGM 暴露度低（Count 中位 2 vs RadarKG 14）", 1),
    ("唯一干净 AGM 题型 SelectBetween：+11.1pp [+2.2,+22.2] 显著（dual_subgraph 跨域真赢）", 1),
    "结论：Strategy 是「结构性保险」——增益由基准 AGM 分布决定，不是通用乘子",
], size=16, gap=11)

# ============================================================ SLIDE 11: 忠实微调 RoG
s = prs.slides.add_slide(BLANK); add_title_bar(s, "忠实微调 RoG：驳倒「弱 baseline」质疑", "RQ6 · NEW")
add_table(s,
    ["系统", "OVERALL", "说明"],
    [["Baseline", "52.7", "BM25+向量+RRF"],
     ["RoG-zs (DeepSeek)", "60.3", "主表 RoG，zero-shot"],
     ["RoG-zs (Qwen-7B)", "31.5", "同 base，未微调"],
     ["RoG-FT (Qwen-7B)", "61.7", "LoRA 微调 planner"],
     ["Strategy", "88.6", "本工作"]],
    Inches(0.8), Inches(1.7), Inches(7.6), Inches(3.2), fs=14, hfs=14, highlight_rows=[4, 5])
bullets(s, [
    "微调效应",
    ("+30.3pp", 1),
    ("[+25.9,+34.7]", 1),
    "→ 排除弱 baseline",
    "Strategy − RoG-FT",
    ("+26.9pp", 1),
    ("[+22.6,+31.3]", 1),
], left=Inches(8.8), top=Inches(1.7), width=Inches(4.0), size=15, gap=7)
note(s, "微调让 planner 逐题型「重新发明」exhaustive（agg_count 2→64 追平 Strategy），但单路径范式无法表达交集（attr_filter 28 vs 92）")

# ============================================================ SLIDE 12: 工程
s = prs.slides.add_slide(BLANK); add_title_bar(s, "工程实现", "ENGINEERING")
bullets(s, [
    "多源 KG 构建 pipeline：9 源融合（规则 / LLM / PDF / Wikidata / Wikipedia…），每条三元组带 source+confidence+evidence",
    ("relation-level 可信度阈值 + 实体去重安全边界（country 冲突自动 skip，0 误删，888→849）", 1),
    "混合检索后端：BM25 + FAISS 向量 + RRF 融合 + 2-hop 图扩展 + KG-direct 旁路",
    "LLM 调用工程：ThreadPool 并发，4400 次调用 12 分钟 / 总成本 < $5，三层持久化缓存",
    "评测框架：499 题分层 + 26 题 OOD + 配对 bootstrap CI + 13 套消融脚本（约 8K 行代码）",
    "本工作新增：服务器 LoRA 微调栈（Qwen-7B + peft）+ 5-way RoG 对照（experiments/rog_finetune/）",
], size=16, gap=11)

# ============================================================ SLIDE 13: 当前成果
s = prs.slides.add_slide(BLANK); add_title_bar(s, "当前成果", "STATUS")
bullets(s, [
    "论文 v12.2（19 页，中英双版本），目标 NAACL / EMNLP Industry Track 2026",
    ("framing：从「分类器+派发」重构为「Answer-Geometry Mismatch + 信息需求匹配的算子套件」", 1),
    ("自评：clear accept / borderline strong（reviewer-style 多轮自审）", 1),
    "三层贡献：① AGM 诊断（analytical）② 六算子套件（system）③ RadarKG-QA-499（benchmark）+ 双语悖论（方法学）",
    "完整实验矩阵：主结果 + K=20 + oracle + 单算子 + suite-size + 双语 + KQA Pro 4-config + 忠实微调 RoG",
    "工件：代码（Apache-2.0）+ KG/基准（CC-BY-4.0），全流程复现 ≈ $1.5",
], size=16, gap=11)

# ============================================================ SLIDE 14: 后续计划
s = prs.slides.add_slide(BLANK); add_title_bar(s, "后续计划", "FUTURE WORK")
bullets(s, [
    "投稿前（高 ROI）：arXiv preprint；ByoKG-RAG 引用核对；论文转 ACL 模板",
    "拆分 short paper：「Bilingual Paradox in Multilingual KGQA Eval」（方法学发现，可单独投 workshop）",
    "LLaMA-2 planner 复现：补完全忠实的 RoG（当前用 Qwen-7B，因服务器无外网）",
    "Schema-derived parser：从任意 KG schema 自动衍生 parser → 把跨域迁移从 1 天降到零成本",
    "跨域迁移：DrugBank / Materials Project（天然高基数 + 多约束，验证 AGM 普适）",
    "对抗集：构造 KG 模糊否定 + 3-way 比较，验证 complement / dual-subgraph 变 load-bearing",
    "社区高基数 KGQA 基准：把「公开基准欠采样 AGM」从弱点变成 research gap",
], size=15.5, gap=9)

# ============================================================ SLIDE 15: takeaway
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, ACCENT)
tb = box(s, Inches(1.1), Inches(2.5), SW - Inches(2.2), Inches(2.6))
p = tb.text_frame.paragraphs[0]
r = p.add_run(); r.text = "一句话总结"; set_font(r, 18, True, RGBColor(0xBF, 0xD4, 0xE8))
p2 = tb.text_frame.add_paragraph(); p2.space_before = Pt(14)
r = p2.add_run()
r.text = "GraphRAG 不缺更聪明的 top-K，缺的是「为查询的信息需求选对检索算子」。"
set_font(r, 26, True, WHITE)
p3 = tb.text_frame.add_paragraph(); p3.space_before = Pt(16)
r = p3.add_run()
r.text = "RadarKG-QA-499 上 +35.9pp，增益 100% 来自算子；微调一个 planner 也只能逐题型重新发明它们，无法在单路径范式内复制交集与补集。"
set_font(r, 16, False, RGBColor(0xDD, 0xE8, 0xF2))

OUT = "雷达GraphRAG_项目总结.pptx"
prs.save(OUT)
print("saved:", OUT, "slides:", len(prs.slides._sldIdLst))
