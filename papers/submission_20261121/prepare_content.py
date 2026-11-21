"""Create venue-neutral revision from the audited English paper; no model runs."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "papers/recovery_short_20261114"

abstract = r"""Small language models performing multi-step knowledge queries must continue from histories that depart from successful demonstrations. Extra executable actions can change intermediate-result identifiers and invalidate direct reuse of a reference suffix. We present dependency-consistent recovery supervision: a fixed initial model supplies unsuccessful training histories, reference dependencies are mapped to current result handles, and deterministic replay validates the continuation. Training retains the actual divergence as context and supervises subsequent assistant actions. Using Qwen2.5-3B-Instruct with low-rank adaptation, we compare recovery and ordinary continuation training under matched initialization, successful material, question order, and supervised-token exposure. On 2,000 project-held-out KQA Pro questions, mean accuracy across two continuation seeds increases from 78.75\% to 83.50\%, a gain of 4.75 percentage points with a question-paired 95\% bootstrap interval of $[3.53,5.98]$. Cumulative input-plus-generated tokens decrease by 8.84\% and 4.17\% relative to the matched stepwise controls. An offline diagnostic on 414 comparable historical training pairs finds that all suffix calls execute in 65 direct-splice cases and all 414 remapped cases. These results support reference-program-based recovery supervision for small models in deterministic, read-only knowledge-query environments."""


def main():
    HERE.mkdir(exist_ok=True)
    body = (OLD / "manuscript_en.tex").read_text()
    body = body[body.index(r"\section{Introduction}"):]
    body = re.sub(r"\\paragraph\{([^}]+)\}", r"\\subsubsection{\1}", body)
    method_start=body.index(r"\section{Dependency-Consistent Recovery Supervision}")
    body=body[:method_start].replace(r"\subsubsection{",r"\subsection{")+body[method_start:]
    body = body.replace(r"\centering\small", r"\centering").replace(r"\centering\footnotesize", r"\centering")
    body = body.replace("Reference divergence identifies a construction point rather than a semantic error label. Different valid programs may solve the same question.", "Reference divergence identifies the point at which action sequences first differ. Different valid programs may solve the same question.")
    body = body.replace("Monte Carlo tree search and performs", "Monte Carlo tree search (MCTS) and performs")
    body = body.replace("ReTerm builds recovery supervision", "ReTerm builds recovery supervision for supervised fine-tuning (SFT)")
    body = body.replace("using reinforcement learning~", "using reinforcement learning (RL)~")
    body = body.replace("We compare these designs conceptually; our empirical controls operate on the same knowledge-query task and training protocol. Performance values from different environments are not used as a common leaderboard.", r"Table~\ref{tab:related} compares their supervision sources and construction requirements. The empirical controls in this study use the same knowledge-query task and training protocol.")
    anchor = "For example, suppose the shared prefix produces"
    invariant = r"""Writing the reference result store as $V^*$ and the actual store as $V$, the mapping maintains
\begin{equation}
 V[m(j)]\equiv V^*[j],\qquad j\in\mathrm{dom}(m),
 \label{eq:dependency-invariant}
\end{equation}
where equivalence compares result type and content after handle normalization. The common prefix establishes the correspondence. A read-only divergent action preserves existing values. Each subsequent reference operation then consumes corresponding inputs, and determinism on the fixed knowledge base gives corresponding outputs. Extending $m$ with the new result preserves the invariant, including the result selected at submission.

"""
    assert anchor in body
    body = body.replace(anchor, invariant + anchor, 1)
    fig = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{Fig1.pdf}
\caption{Recovery-supervision construction with matched continuation targets}
\label{fig:method-en}
\end{figure}"""
    body = body.replace(r"\input{papers/recovery_short_20261114/method_figure_en.tex}", r"Figure~\ref{fig:method-en} summarizes the paired construction and its shared training conditions."+"\n\n"+fig)
    body = body.replace("LoRA~\\cite{hu2022lora} uses", "Low-rank adaptation (LoRA)~\\cite{hu2022lora} uses")
    body = body.replace("BF16, gradient checkpointing, and SDPA attention", "bfloat16 precision, gradient checkpointing, and scaled dot-product attention")
    body = body.replace("including EOS", "including the end-of-sequence token")
    body = body.replace("A separate CPU diagnostic", "A separate central-processing-unit (CPU) diagnostic")
    body = body.replace("Monte Carlo tree search and performs", "Monte Carlo tree search and performs")
    start = body.index(r"\subsection{Feedback-text diagnostic}")
    end = body.index(r"\section{Discussion and Scope}", start)
    auxiliary = body[start:end].replace(r"\subsection{Feedback-text diagnostic}", r"\subsection{Auxiliary feedback analysis}")
    body = body[:start] + body[end:]
    body = body.replace(r"\section{Discussion and Scope}", r"\section{Discussion}")
    body = body.replace("Together, they support the usefulness of a verified continuation construction without equating construction success with a model recovery rate.", "The two analyses establish the training scheme's task-level benefit and the construction mechanism's execution consistency.")
    body = body.replace("The bootstrap interval conditions on the fixed checkpoints rather than estimating all training variance.", "The bootstrap interval quantifies question-level uncertainty conditional on the fixed checkpoints.")
    future = "A registered evaluation from identical normal and divergent histories is prepared but has not yet been run. It will complement the current free-running and construction evidence by controlling the states at which the models resume. The present accuracy gain concerns the complete recovery-supervision recipe; it does not separately identify every contribution of history exposure, handle renumbering, and target filtering."
    replacement = "The accuracy gain evaluates the complete recovery-supervision recipe. A useful next step is to compare ordinary and recovery models from identical normal and divergent histories under matched remaining budgets. This would connect free-running improvements with state-conditional continuation performance and guide component-level training comparisons."
    assert future in body
    body = body.replace(future, replacement + "\n\n" + auxiliary.strip())
    (HERE / "manuscript_body_en.tex").write_text(body)
    (HERE / "abstract_en.tex").write_text(abstract + "\n")
    tables = (OLD / "generated_tables.tex").read_text()
    table_blocks = re.findall(r"\\newcommand\{\\(Recovery\w+EN)\}\{%\n(.*?)\n\}\n", tables, re.S)
    (HERE / "tables_en.tex").write_text("% English tables retain the audited numerical values.\n" + "\n\n".join("\\newcommand{\\" + name + "}{%\n" + block + "\n}\n" for name, block in table_blocks))
    bibliography=(OLD / "references.bib").read_text()
    blocks=re.split(r'(?m)(?=^@)',bibliography)
    linked=[]
    for block in blocks:
        doi=re.search(r'(?m)^\s*doi\s*=\s*\{([^}]+)\}',block)
        if doi and 'https://doi.org/'+doi[1] not in block:
            block=re.sub(r'(?m)^(\s*note\s*=\s*\{)(.*)(\}\s*)$',lambda m:m[1]+m[2]+r'. \url{https://doi.org/'+doi[1]+'}'+m[3],block)
        linked.append(block)
    (HERE / "references.bib").write_text(''.join(linked))
    plain = re.sub(r"\\[A-Za-z]+|[{}$]", "", abstract)
    words = len(plain.split())
    assert words < 250, words
    meta = {
        "date": "2026-11-21", "title": "Dependency-consistent recovery supervision for executable knowledge question answering with small language models",
        "short_title": "Dependency-consistent recovery supervision", "keywords": ["Knowledge graph question answering", "Small language models", "Recovery supervision", "Dependency mapping", "Execution feedback"],
        "abstract_words_whitespace": words, "base_paper": str(OLD.relative_to(ROOT)),
        "source_manuscript_sha256": hashlib.sha256((OLD / "manuscript_en.tex").read_bytes()).hexdigest(),
        "source_numeric_table_sha256": hashlib.sha256((OLD / "tables_data.json").read_bytes()).hexdigest(),
        "new_model_runs": 0, "radar_application_included": False,
        "changes": ["State-invariant explanation added", "Acronyms expanded", "Feedback-text diagnostic moved into discussion", "Scope and future work concentrated in discussion", "Vector figure redrawn", "Venue-specific declarations and reproducibility inventory prepared"],
        "author_information": "pending", "funding_and_interests": "pending author confirmation", "submitted": False,
        "venue_choice": "provisional; institutional acceptance criteria and fee budget not yet supplied"
    }
    (HERE / "paper_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"abstract_words": words, "sections": re.findall(r"\\section\{([^}]+)\}", body)}))


if __name__ == "__main__":
    main()
