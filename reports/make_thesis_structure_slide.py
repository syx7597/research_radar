# -*- coding: utf-8 -*-
"""Create one editable thesis-structure PowerPoint slide without third-party deps."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "论文结构目录_单页.pptx"

EMU = 914400
SLIDE_W = 10.0
SLIDE_H = 5.625

NAVY = "183657"
BLUE = "2F6491"
TEAL = "4F7D7B"
GOLD = "B48838"
INK = "242D36"
GRAY = "68737D"
MID = "B5C1CC"
LINE = "D8E0E7"
PALE_BLUE = "ECF1F5"
PALE_TEAL = "EBF2F1"
PALE_GOLD = "F6F1E6"
BG = "FAFBFC"
WHITE = "FFFFFF"
FONT = "Microsoft YaHei"
MONO = "Consolas"


def emu(value: float) -> int:
    return round(value * EMU)


def font_size(value: float) -> int:
    return round(value * 100)


def solid_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color: str | None = LINE, width_pt: float = 0.75) -> str:
    if color is None:
        return "<a:ln><a:noFill/></a:ln>"
    return (
        f'<a:ln w="{round(width_pt * 12700)}">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        "</a:ln>"
    )


class SlideBuilder:
    def __init__(self) -> None:
        self.shape_id = 1
        self.parts: list[str] = []

    def next_id(self) -> int:
        self.shape_id += 1
        return self.shape_id

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = WHITE,
        border: str | None = LINE,
        border_width: float = 0.75,
        radius: str = "rect",
        name: str = "Rectangle",
    ) -> None:
        sid = self.next_id()
        self.parts.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
    {solid_fill(fill)}
    {line_xml(border, border_width)}
  </p:spPr>
  <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
</p:sp>"""
        )

    def textbox(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        size: float = 10,
        color: str = INK,
        bold: bool = False,
        align: str = "l",
        font: str = FONT,
        valign: str = "t",
        margin: float = 0.03,
    ) -> None:
        sid = self.next_id()
        b = ' b="1"' if bold else ""
        paras = []
        for line in text.split("\n"):
            safe = escape(line)
            paras.append(
                f"""
    <a:p>
      <a:pPr algn="{align}"/>
      <a:r><a:rPr lang="zh-CN" sz="{font_size(size)}"{b}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{font}"/><a:ea typeface="{font}"/><a:cs typeface="{font}"/></a:rPr><a:t>{safe}</a:t></a:r>
    </a:p>"""
            )
        self.parts.append(
            f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    <a:noFill/>
    <a:ln><a:noFill/></a:ln>
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" anchor="{valign}" lIns="{emu(margin)}" rIns="{emu(margin)}" tIns="{emu(margin)}" bIns="{emu(margin)}"/>
    <a:lstStyle/>
    {''.join(paras)}
  </p:txBody>
</p:sp>"""
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = MID, width: float = 0.8) -> None:
        sid = self.next_id()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1) or 0.001
        h = abs(y2 - y1) or 0.001
        flip_h = ' flipH="1"' if x2 < x1 else ""
        flip_v = ' flipV="1"' if y2 < y1 else ""
        self.parts.append(
            f"""
<p:cxnSp>
  <p:nvCxnSpPr><p:cNvPr id="{sid}" name="Connector {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm{flip_h}{flip_v}><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="{round(width * 12700)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
  </p:spPr>
</p:cxnSp>"""
        )

    def chapter_card(self, num: int, title: str, role: str, x: float, y: float, w: float, h: float, fill: str, accent: str) -> None:
        self.rect(x, y, w, h, fill, LINE, 0.75)
        self.textbox(f"{num:02d}", x + 0.12, y + 0.12, 0.35, 0.20, 8.0, accent, True, font=MONO)
        self.textbox(title, x + 0.49, y + 0.10, w - 0.60, 0.25, 9.8, NAVY, True)
        self.textbox(role, x + 0.49, y + 0.43, w - 0.58, h - 0.50, 8.1, INK)

    def xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr>{solid_fill(BG)}<a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(self.parts)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def build_slide_xml() -> str:
    s = SlideBuilder()
    s.rect(0, 0, 0.07, SLIDE_H, NAVY, None)
    s.textbox("THESIS STRUCTURE", 0.47, 0.18, 2.9, 0.22, 8.8, TEAL, True, font=MONO)
    s.textbox("论文结构目录：从知识环境到策略学习与实验验证", 0.47, 0.47, 8.2, 0.38, 19.2, NAVY, True)
    s.textbox("面向复杂雷达情报问答的图检索策略学习方法研究", 0.47, 0.90, 6.4, 0.24, 8.8, GRAY)

    stages = [
        ("问题提出", 0.74, 0.94),
        ("任务与环境", 2.12, 1.22),
        ("核心方法", 3.84, 2.72),
        ("验证总结", 7.10, 1.90),
    ]
    for label, x, w in stages:
        s.rect(x, 1.35, w, 0.28, PALE_BLUE, LINE)
        s.textbox(label, x, 1.39, w, 0.13, 7.8, BLUE, True, "ctr")
    for x in (1.78, 3.52, 6.78):
        s.line(x, 1.49, x + 0.25, 1.49, BLUE, 1.1)

    cards = [
        (1, "绪论", "研究背景、问题来源、研究目标与两项核心创新。", 0.70, 1.88, 1.16, 2.30, WHITE, BLUE),
        (2, "相关工作", "梳理 RAG、KGQA、GraphRAG、工具 Agent 与策略学习研究。", 2.05, 1.88, 1.44, 2.30, WHITE, BLUE),
        (3, "图文知识环境与任务构建", "构建 RadarKG、叙述文本库与复杂问答数据集，定义问题类型。", 3.74, 1.88, 1.48, 2.30, PALE_BLUE, BLUE),
        (4, "类型化可执行图检索决策建模", "定义状态、结构化动作、类型约束、中间变量与确定性执行器。", 5.38, 1.88, 1.48, 2.30, PALE_TEAL, TEAL),
        (5, "多粒度执行反馈的策略学习", "以成功轨迹 SFT 冷启动，并利用计划、进展、恢复和成本反馈优化策略。", 7.02, 1.88, 1.48, 2.30, PALE_GOLD, GOLD),
        (6, "实验设计与结果分析", "通过 2x2 对照、组合隔离和公开数据集验证两个方法贡献。", 5.18, 4.62, 1.55, 1.32, WHITE, BLUE),
        (7, "总结与展望", "总结研究结论，讨论跨领域迁移、图文联合推理与系统应用扩展。", 6.95, 4.62, 1.55, 1.32, WHITE, BLUE),
    ]
    for card in cards:
        s.chapter_card(*card)

    s.line(1.86, 3.03, 2.05, 3.03, MID, 0.8)
    s.line(3.49, 3.03, 3.74, 3.03, MID, 0.8)
    s.line(5.22, 3.03, 5.38, 3.03, MID, 0.8)
    s.line(6.86, 3.03, 7.02, 3.03, MID, 0.8)
    s.line(7.76, 4.18, 7.76, 4.62, MID, 0.8)
    s.line(6.73, 5.28, 6.95, 5.28, MID, 0.8)

    s.rect(0.70, 4.62, 3.96, 1.32, WHITE, LINE)
    s.textbox("章节关系", 0.90, 4.77, 0.90, 0.22, 9.2, NAVY, True)
    s.textbox(
        "第 3 章定义知识环境与任务；第 4 章回答“策略空间如何表示”；第 5 章回答“策略如何学习”；第 6 章用控制变量实验分别验证二者贡献。",
        0.90,
        5.12,
        3.42,
        0.50,
        8.6,
        INK,
    )
    s.textbox("主线：复杂雷达问答 → 可执行图检索决策 → 多粒度反馈学习 → 组合泛化验证", 0.92, 6.32, 7.8, 0.24, 9.2, NAVY, True)
    s.line(0.47, 6.75, 9.52, 6.75, LINE, 0.8)
    s.textbox("论文目录页", 8.95, 6.88, 0.55, 0.14, 6.8, MID, False, "r")
    return s.xml()


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

PRESENTATION = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr></p:defaultTextStyle>
</p:presentation>"""

PRES_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>"""

SLIDE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

SLIDE_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Academic Navy">
  <a:themeElements>
    <a:clrScheme name="Academic Navy"><a:dk1><a:srgbClr val="242D36"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="183657"/></a:dk2><a:lt2><a:srgbClr val="ECF1F5"/></a:lt2><a:accent1><a:srgbClr val="2F6491"/></a:accent1><a:accent2><a:srgbClr val="4F7D7B"/></a:accent2><a:accent3><a:srgbClr val="B48838"/></a:accent3><a:accent4><a:srgbClr val="B5C1CC"/></a:accent4><a:accent5><a:srgbClr val="68737D"/></a:accent5><a:accent6><a:srgbClr val="D8E0E7"/></a:accent6><a:hlink><a:srgbClr val="2F6491"/></a:hlink><a:folHlink><a:srgbClr val="4F7D7B"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>1</Slides>
</Properties>"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>论文结构目录</dc:title><dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def write_pptx() -> None:
    files = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": ROOT_RELS,
        "docProps/app.xml": app_xml(),
        "docProps/core.xml": core_xml(),
        "ppt/presentation.xml": PRESENTATION,
        "ppt/_rels/presentation.xml.rels": PRES_RELS,
        "ppt/slides/slide1.xml": build_slide_xml(),
        "ppt/slides/_rels/slide1.xml.rels": SLIDE_RELS,
        "ppt/slideLayouts/slideLayout1.xml": SLIDE_LAYOUT,
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": SLIDE_LAYOUT_RELS,
        "ppt/slideMasters/slideMaster1.xml": SLIDE_MASTER,
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": SLIDE_MASTER_RELS,
        "ppt/theme/theme1.xml": THEME,
    }
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content.encode("utf-8"))


if __name__ == "__main__":
    write_pptx()
    print(OUTPUT)
