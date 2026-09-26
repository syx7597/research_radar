# -*- coding: utf-8 -*-
"""Knowledge-Based Systems (Elsevier) submission version.

Light-touch adaptation of the method-first manuscript: elsarticle 'review'
layout, frontmatter + keywords + highlights, a <=250-word abstract, and the
mandatory Elsevier back-matter (CRediT, competing interest, data availability,
funding, generative-AI disclosure). Full manuscript incl. appendices retained.

Run:  python make_kbs.py
Then: xelatex kbs; bibtex kbs; xelatex kbs; xelatex kbs
"""
import re
from pathlib import Path

SRC = Path("strategy_routed_graphrag.tex").read_text(encoding="utf-8")

TITLE = (r"Beyond Top-K: A Typed Operator Suite for Answer-Geometry "
         r"Mismatches in Knowledge Graph Question Answering")
JOURNAL = "Knowledge-Based Systems"
KEYWORDS = [
    "Knowledge graph question answering",
    "Graph retrieval-augmented generation",
    "Typed retrieval operators",
    "Large language models",
    "Knowledge-based decision support",
]
HIGHLIGHTS = [
    r"Top-$K$ GraphRAG fails on enumeration, intersection and complement geometries",
    r"We derive six typed retrieval operators, one per answer-geometry class",
    r"An oracle ablation localizes the full $+35.9$ pp gain to the operators",
    r"RadarKG-QA-499, a bilingual answer-geometry-mismatch benchmark, is released",
    r"Deployable at \$0.00041 per query for industrial knowledge-based QA",
]

# <=250-word, method-first abstract (KBS values the methodological advance).
ABSTRACT = (
    "GraphRAG systems converge on ever more sophisticated top-$K$ retrieval "
    "--- community summarization, Personalized PageRank, LLM path templates, "
    "beam search --- yet top-$K$ embeds an implicit assumption about "
    "\\emph{answer geometry}: that the answer is a small set rankable by "
    "surface relevance. We show this assumption fails systematically, for "
    "\\emph{information-theoretic} rather than model-capacity reasons, on three "
    "computable query patterns common in industrial knowledge graphs: "
    "unbounded enumeration, multi-constraint intersection, and "
    "complement / non-membership. We name this the \\textbf{answer-geometry "
    "mismatch} and argue it is analogous to the OLTP/OLAP split in databases: "
    "a B-tree index is excellent for point queries and structurally wrong for "
    "aggregates, however it is tuned. We propose \\textbf{Strategy-Routed "
    "GraphRAG}, a typed suite of six retrieval operators (\\op{lookup}, "
    "\\op{exhaustive}, \\op{complement}, \\op{path-plan}, \\op{constrained-join}, "
    "\\op{dual-subgraph}), each \\emph{derived} as the primitive that satisfies "
    "its target class's information requirement (a four-operator subset is "
    "load-bearing on this benchmark; the other two are retained by derivation); "
    "a lightweight LLM dispatcher selects among them. An oracle ablation "
    "localizes the entire gain to the "
    "operators (88.2\\% oracle vs 88.6\\% LLM dispatch), so the contribution is "
    "the operators, not the classifier. On \\textbf{RadarKG-QA-499}, a new "
    "bilingual benchmark designed to expose answer-geometry mismatch, the suite "
    "reaches \\textbf{88.6\\%} versus 52.7\\% for a strong baseline and 60.3\\% "
    "for a zero-shot path planner ($+35.9$ / $+28.3$ pp; paired bootstrap "
    "confidence intervals strictly positive; robust to $K{=}8$--$20$). The "
    "typology and dispatcher transfer to KQA Pro without retraining. At "
    "\\$0.00041 per query the method is deployable for industrial knowledge "
    "graph question answering."
)

DECLARATIONS = r"""
\section*{CRediT authorship contribution statement}
\textbf{Author Name:} Conceptualization, Methodology, Software, Data curation, Formal analysis, Validation, Visualization, Writing -- original draft, Writing -- review \& editing.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Data availability}
The RadarKG-QA-499 benchmark, the RadarKG-v2 knowledge graph, and the evaluation code are openly available at \url{https://github.com/syx7597/radarkg-qa} and will be archived with a permanent DOI upon acceptance.

\section*{Funding}
This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\section*{Generative AI in the scientific writing process}
A large language model is a functional component of the proposed system (the dispatcher, parser, and answer-synthesis stages), as described in the manuscript. Separately, during manuscript preparation the authors used an AI assistant for language editing only; all scientific content was reviewed and verified by the authors, who take full responsibility for the publication.

"""


def build():
    t = SRC
    # 1) elsarticle review class + journal
    t = t.replace(
        r"\documentclass[11pt,a4paper]{article}",
        "\\PassOptionsToPackage{hidelinks}{hyperref}\n"
        "\\documentclass[review]{elsarticle}\n"
        "\\journal{%s}" % JOURNAL)
    # 2) drop packages elsarticle manages / conflicts with
    t = t.replace("\\usepackage[margin=1in]{geometry}\n", "")
    t = t.replace("\\usepackage{natbib}\n", "")
    t = t.replace("\\usepackage[hidelinks]{hyperref}", "\\usepackage{hyperref}")
    # 3) remove old article title block
    t = re.sub(r"\\title\{.*?\\date\{\}", "", t, flags=re.S)
    # 4) trim the abstract to <=250 words (lambda repl: no backslash escaping)
    t = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
               lambda m: "\\begin{abstract}\n" + ABSTRACT + "\n\\end{abstract}",
               t, flags=re.S)
    # 5) wrap frontmatter
    fm_open = ("\\begin{frontmatter}\n\n"
               "\\title{%s}\n\n"
               "\\author{Author Name}\n"
               "\\affiliation{organization={Your Institution}, country={China}}\n"
               % TITLE)
    t = t.replace("\\maketitle", fm_open, 1)
    kw = " \\sep ".join(KEYWORDS)
    t = t.replace("\\end{abstract}",
                  "\\end{abstract}\n\n\\begin{keyword}\n%s\n\\end{keyword}\n\n"
                  "\\end{frontmatter}\n" % kw, 1)
    # 6) declarations before the bibliography
    t = t.replace("\\bibliographystyle{plainnat}",
                  DECLARATIONS + "\\bibliographystyle{elsarticle-num-names}")
    Path("kbs.tex").write_text(t, encoding="utf-8")

    items = "\n".join("\\item %s" % h for h in HIGHLIGHTS)
    Path("kbs_highlights.tex").write_text(
        "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{enumitem}\n\\usepackage[margin=2.5cm]{geometry}\n"
        "\\begin{document}\n\\section*{Highlights}\n"
        "\\textbf{%s} --- submission to \\emph{%s}\\\\[4pt]\n"
        "\\begin{itemize}[leftmargin=1.2em,nosep]\n%s\n\\end{itemize}\n"
        "\\end{document}\n" % (TITLE, JOURNAL, items), encoding="utf-8")
    print("wrote kbs.tex + kbs_highlights.tex")


build()
