# -*- coding: utf-8 -*-
"""Minimal Markdown -> LaTeX converter tailored to this thesis's constructs:
headings, pipe tables, images(+caption), fenced code (ASCII diagrams), inline
math ($...$) and display math ($$...$$), bold/italic, inline code, lists,
blockquotes. Produces a ctexart document compilable by XeLaTeX.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "thesis_full.md"
OUT = HERE / "thesis.tex"

PREAMBLE = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage[margin=2.6cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{caption}
\usepackage{enumitem}
\pagestyle{plain}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setcounter{tocdepth}{3}
\graphicspath{{figures/}{./}}
\setlength{\parindent}{2em}
\linespread{1.25}
% 等宽字体含制表符/箭头(Consolas)，中文等宽回退(宋体)，供 ASCII 框图使用
\IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}[Scale=0.9]}{\IfFontExistsTF{Consolas}{\setmonofont{Consolas}}{}}
\IfFontExistsTF{SimSun}{\setCJKmonofont{SimSun}}{}
\fvset{fontsize=\footnotesize,frame=single,framesep=4pt}
\title{\bfseries 面向雷达情报的可解释知识图谱问答与对抗决策支持方法研究}
\author{}
\date{}
\begin{document}
\maketitle
\tableofcontents
\newpage
"""


def esc(s):
    """Escape LaTeX specials in plain (non-math, non-code) text."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def inline(s):
    """Convert one line of inline markdown to LaTeX, protecting $...$ math."""
    out = []
    parts = re.split(r"(\$[^$]+\$)", s)          # protect inline math
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(part)                      # keep math verbatim
            continue
        # inline code `...`
        seg = re.split(r"(`[^`]+`)", part)
        for s2 in seg:
            if s2.startswith("`") and s2.endswith("`") and len(s2) > 1:
                out.append(r"\texttt{" + esc(s2[1:-1]) + "}")
                continue
            t = esc(s2)
            # bold **x** and italic *x* (after escaping; markers survive esc)
            t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", t)
            out.append(t)
    return "".join(out)


def conv_table(rows):
    """rows: list of raw '| a | b |' lines (incl. separator). -> LaTeX tabular."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header = cells[0]
    body = cells[2:]                              # skip separator row (cells[1])
    ncol = len(header)
    colspec = "|" + "|".join(["p{%.3f\\textwidth}" % (0.92 / ncol)] * ncol) + "|"
    lines = [r"\begin{center}\small", r"\begin{tabular}{" + colspec + "}", r"\hline"]
    lines.append(" & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\ \hline")
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        lines.append(" & ".join(inline(c) for c in row) + r" \\ \hline")
    lines += [r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def convert(md):
    out = []
    lines = md.split("\n")
    i = 0
    first_h1 = True
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]

        # fenced code -> asciibox (Verbatim)
        if ln.strip().startswith("```"):
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(r"\begin{Verbatim}")
            out.extend(buf)
            out.append(r"\end{Verbatim}")
            continue

        # display math $$ ... $$
        if ln.strip().startswith("$$"):
            close_list()
            inner = ln.strip()[2:]
            if inner.strip().endswith("$$"):       # single-line
                expr = inner.strip()[:-2]
                out.append(r"\[" + expr + r"\]")
                i += 1; continue
            buf = [inner] if inner.strip() else []
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                buf.append(lines[i]); i += 1
            tail = lines[i].strip()[:-2] if i < len(lines) else ""
            if tail.strip():
                buf.append(tail)
            i += 1
            out.append(r"\[" + "\n".join(buf) + r"\]")
            continue

        # tables
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_list()
            tb = [ln]; i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tb.append(lines[i]); i += 1
            out.append(conv_table(tb)); continue

        # images ![alt](path)  + optional following bold caption
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln.strip())
        if m:
            close_list()
            path = m.group(1)
            path = re.sub(r"^figures/", "", path)
            path = re.sub(r"\.png$", "", path)
            cap = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("**"):
                cap = re.sub(r"\*\*", "", lines[j].strip())
                i = j
            out.append(r"\begin{figure}[htbp]\centering")
            out.append(r"\includegraphics[width=0.78\textwidth]{" + path + "}")
            if cap:
                out.append(r"\caption*{" + inline(cap) + "}")   # 无 LaTeX 自动号，用正文手写"图X-Y"
            out.append(r"\end{figure}")
            i += 1; continue

        # headings (all unnumbered: manual numbers already in heading text)
        if ln.startswith("### "):
            close_list(); t = inline(ln[4:].strip())
            out.append(r"\subsubsection*{" + t + "}")
            out.append(r"\addcontentsline{toc}{subsubsection}{" + t + "}"); i += 1; continue
        if ln.startswith("## "):
            close_list(); t = inline(ln[3:].strip())
            out.append(r"\subsection*{" + t + "}")
            out.append(r"\addcontentsline{toc}{subsection}{" + t + "}"); i += 1; continue
        if ln.startswith("# "):
            close_list()
            title = ln[2:].strip()
            if first_h1:
                first_h1 = False; i += 1; continue       # document title (in \maketitle)
            t = inline(title)
            out.append(r"\section*{" + t + "}")
            out.append(r"\addcontentsline{toc}{section}{" + t + "}")
            i += 1; continue

        # horizontal rule / separator
        if ln.strip() == "---":
            close_list(); i += 1; continue

        # blockquote
        if ln.strip().startswith(">"):
            close_list()
            out.append(r"\begin{quote}\itshape " + inline(ln.strip().lstrip(">").strip()) + r"\end{quote}")
            i += 1; continue

        # list items
        m = re.match(r"^(\s*)[-*]\s+(.*)", ln)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}[leftmargin=2em]"); in_list = True
            out.append(r"\item " + inline(m.group(2))); i += 1; continue
        m = re.match(r"^\s*\d+\.\s+(.*)", ln)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}[leftmargin=2em]"); in_list = True
            out.append(r"\item " + inline(m.group(1))); i += 1; continue

        # blank line
        if not ln.strip():
            close_list(); out.append(""); i += 1; continue

        # normal paragraph line
        close_list()
        out.append(inline(ln))
        i += 1

    close_list()
    return "\n".join(out)


if __name__ == "__main__":
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    OUT.write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(body)} chars)")
