# -*- coding: utf-8 -*-
"""Produce a condensed double-column version of the paper (mechanical pass):
twocolumn 10pt + tight margins + wide tables span both columns + drop appendices.
Content trimming (RQ5/RQ6 etc.) is done separately."""
import re
from pathlib import Path

src = Path("strategy_routed_graphrag.tex").read_text(encoding="utf-8")
t = src

# 1) two-column 10pt
t = t.replace(r"\documentclass[11pt,a4paper]{article}",
              r"\documentclass[10pt,a4paper,twocolumn]{article}")

# 2) tight margins (replace existing geometry, else inject)
if re.search(r"\\usepackage(\[[^\]]*\])?\{geometry\}", t):
    t = re.sub(r"\\usepackage(\[[^\]]*\])?\{geometry\}",
               r"\\usepackage[margin=1.9cm]{geometry}", t)
else:
    t = t.replace(r"\begin{document}",
                  "\\usepackage[margin=1.9cm]{geometry}\n\\begin{document}", 1)

# 3) wide tables -> table* (span both columns to avoid column overflow)
t = t.replace(r"\begin{table}", r"\begin{table*}").replace(r"\end{table}", r"\end{table*}")

# 4) drop appendices (from \appendix to \end{document})
i, j = t.find(r"\appendix"), t.find(r"\end{document}")
if 0 < i < j:
    t = t[:i] + "% --- appendices A-I moved to extended (arXiv) version ---\n\\end{document}\n"

# 5) content compression: replace verbose subsections (drop their big tables),
#    keeping any \label that is cross-referenced elsewhere.
def replace_span(text, start, end, new):
    a = text.find(start)
    if a < 0:
        print("WARN start not found:", start[:40]); return text
    b = text.find(end, a + len(start))
    if b < 0:
        print("WARN end not found:", end[:40]); return text
    return text[:a] + new + text[b:]

RQ4 = (r"""\subsection{RQ4: Cross-domain transfer}
\label{sec:rq4}
On \textbf{KQA Pro} (Wikidata) the typology and dispatcher transfer without retraining (67.4\% zero-shot routing match); \op{dual-subgraph} yields $+11$ pp on SelectBetween and \op{exhaustive}/\op{constrained-join} recover the same structural gains. The absolute gain tracks how much of a benchmark exhibits answer-geometry mismatch (per-configuration results in the extended version).

""")
RQ5 = (r"""\subsection{RQ5: Answer-geometry exposure predicts \texorpdfstring{$\Delta$}{Delta}}
\label{sec:rq5}
Per-type gains scale with \textbf{answer-geometry exposure}: types with high answer cardinality or multi-hop / intersection / complement composition show the largest $\Delta$ (\qtype{agg\_count}, \qtype{relation\_inverse}, \qtype{three\_hop\_chain}, \qtype{attr\_filter} all $+$40--70 pp), whereas atomic single-step types (\qtype{single\_hop}) show $\sim$0 --- consistent with the diagnosis that the gain comes from satisfying information requirements top-$K$ cannot meet, not from better ranking.

""")
RQ6 = (r"""\subsection{RQ6: A fine-tuned planner does not close the gap}
\label{sec:rq6}
To rule out a weak-baseline artifact, we LoRA-fine-tune faithful RoG-style planners on two 7B bases --- LLaMA-2-7B-Chat (the original RoG base) and a Qwen-7B control --- on 240 KG-mined (question, relation-path) pairs disjoint from the eval set, replacing only the planner. Fine-tuning lifts each base ${\sim}{+}30$ pp over its zero-shot self (LLaMA-2 26.5$\to$55.9, Qwen 31.5$\to$61.7), letting it re-derive our single-relation operators type-by-type. \textbf{But a single-path planner cannot express set-algebraic operators} (\qtype{attr\_filter} 28/30 vs Strategy 92; intersection needs two joined paths) and overfits its training path-length distribution; the best fine-tuned variant still trails Strategy by $+27$ to $+33$ pp --- robust to base-model choice.

""")
FUTURE = (r"""\section{Future Work}
\label{sec:future}
Key directions: (i) a dispatcher that emits operator \emph{expressions} rather than singletons (the suite is composable); (ii) adversarial-negation and 3-way-comparison stress sets that should activate \op{complement} and \op{dual-subgraph}; (iii) community high-cardinality KGQA benchmarks with explicit per-pattern cardinality strata, of which RadarKG-QA-499 is a first contribution.

""")
ROUTING = (r"""\paragraph{Routing-based retrieval (closest related work).} Adaptive-RAG \citep{jeong2024adaptiverag} routes by query \emph{complexity} (varying retrieval depth) and ByoKG-RAG \citep{byokgrag2025} fuses multiple KG-retrieval tools, but both keep the retrieval primitive constant (ranking-by-relevance), varying only how often or which source it is applied. We instead vary the \emph{semantics} of the primitive itself (intersection vs enumeration vs complement vs path composition); the $+35.9$ pp gain is not reachable by adding depth or sources to a top-$K$ core.

""")

t = replace_span(t, r"\subsection{RQ4:", r"\subsection{RQ5:", RQ4)
t = replace_span(t, r"\subsection{RQ5:", r"\subsection{RQ6:", RQ5)
t = replace_span(t, r"\subsection{RQ6:", r"\subsection{Cost", RQ6)
t = replace_span(t, r"\section{Future Work}", r"\section{Conclusion}", FUTURE)
t = replace_span(t, r"\paragraph{Routing-based retrieval", r"\section{Future Work}", ROUTING)

# 6) neutralize any remaining references to the (removed) appendices
t = re.sub(r"(Appendix~)?\\ref\{app:[^}]*\}", "the extended version", t)

Path("strategy_routed_graphrag_short.tex").write_text(t, encoding="utf-8")
print("wrote strategy_routed_graphrag_short.tex; appendix cut:", 0 < i < j)
