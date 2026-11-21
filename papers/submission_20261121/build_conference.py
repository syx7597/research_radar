#!/usr/bin/env python3
"""Build an anonymous IEA/AIE 2027 backup manuscript from shared paper content.

No submission or model run is performed. The official LNCS package is read from
the local cache and its original files are kept intact in the source archive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "conference"
TEMPLATE = ROOT / "cache/submission_20261121/lncs/template"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shellrun(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def expanded_body():
    body = (HERE / "manuscript_body_en.tex").read_text()
    tables = (HERE / "tables_en.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\([^}]+)\}\{%\n(.*?)\n\}\n", tables, re.S))
    if len(macros) != 4:
        raise ValueError("Expected four audited table definitions")
    for name, value in macros.items():
        body = body.replace("\\" + name, value.replace("\\end{tabular}%", "\\end{tabular}"))
    if "\\Recovery" in body:
        raise ValueError("Unexpanded table macro")
    # The conference has two numbered heading levels; lower levels remain run-in.
    body = re.sub(r"\\subsubsection\{([^}]+)\.\}", r"\\subsubsection{\1}", body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "cache/thesis_build_tools/tectonic_0.17.0/tectonic")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build = OUT / "build"
    build.mkdir(exist_ok=True)
    input_paths = [HERE / n for n in ("manuscript_body_en.tex", "abstract_en.tex", "tables_en.tex", "references.bib", "Fig1.pdf", "paper_metadata.json")] + [Path(__file__)]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    meta = json.loads((HERE / "paper_metadata.json").read_text())
    abstract = (HERE / "abstract_en.tex").read_text().strip()
    body = expanded_body()
    words = len(abstract.split())
    if not 150 <= words <= 250:
        raise ValueError(f"LNCS sample abstract range is 150–250 words; found {words}")
    for name in ("llncs.cls", "splncs04.bst", "readme.txt", "history.txt", "DescriptionTexts.txt"):
        shutil.copy2(TEMPLATE / name, OUT / name)
    shutil.copy2(HERE / "references.bib", OUT / "references.bib")
    shutil.copy2(HERE / "Fig1.pdf", OUT / "Fig1.pdf")
    title = "Dependency-Consistent Recovery Supervision for Executable Knowledge Question Answering with Small Language Models"
    header = r"""% Anonymous IEA/AIE 2027 review draft. Not submitted.
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage[varvw]{newtxmath}
\usepackage{amsmath,booktabs,array,graphicx}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdftitle={},pdfsubject={},pdfkeywords={},pdfcreator={},pdfproducer={}}
\setlength{\emergencystretch}{1em}
\urlstyle{rm}
\begin{document}
"""
    source = header + "\\title{" + title + "}\n" + r"""\titlerunning{Recovery Supervision for Executable Knowledge Questions}
\author{Anonymous Authors}
\authorrunning{Anonymous Authors}
\institute{}
\maketitle
\begin{abstract}
""" + abstract + "\n\\keywords{" + " \\and ".join(meta["keywords"]) + "}\n\\end{abstract}\n\n" + body + r"""

\subsubsection*{AI-Assisted Preparation}
OpenAI Codex assisted with code development, literature retrieval, manuscript drafting, and consistency checks. Reported metrics derive from saved model outputs and execution records.

\bibliographystyle{splncs04}
\bibliography{references}
\end{document}
"""
    identifiers = [r"syx7597", r"research_radar", r"clear-water", r"/Users/", r"East China Normal", r"ECNU", r"222\.204\.232\.33"]
    hits = [pattern for pattern in identifiers if re.search(pattern, source, re.I)]
    if hits:
        raise ValueError("Anonymous source contains identifying strings: " + str(hits))
    tex = OUT / "anonymous.tex"
    tex.write_text(source)
    env = dict(os.environ)
    env.setdefault("TECTONIC_CACHE_DIR", str(ROOT / "cache/tectonic"))
    with (build / "console.log").open("w") as stream:
        shellrun([str(args.engine.resolve()), "--keep-logs", "--keep-intermediates", "--outdir", str(build), "anonymous.tex"], cwd=OUT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = (build / "anonymous.log").read_text(errors="replace")
    blockers = [line for line in log.splitlines() if "undefined references" in line.lower() or re.search(r"(?:Citation|Reference) .+ undefined", line) or line.startswith("Missing character:")]
    if blockers:
        raise ValueError("Unresolved reference or missing glyph: " + " | ".join(blockers[:6]))
    pdf = OUT / "anonymous.pdf"
    shutil.copy2(build / "anonymous.pdf", pdf)
    inspector = build / "inspect.swift"
    inspector.write_text('''import Foundation\nimport PDFKit\nlet p = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1]))!\nvar text = ""\nfor i in 0..<p.pageCount { text += "\\n--- PAGE \\(i+1) ---\\n" + (p.page(at: i)!.string ?? "") }\ntry text.write(toFile: CommandLine.arguments[2], atomically: true, encoding: .utf8)\nvar attrs: [String: String] = [:]\nfor (k,v) in (p.documentAttributes ?? [:]) { attrs[String(describing: k)] = String(describing: v) }\nlet d: [String: Any] = ["pages":p.pageCount,"attributes":attrs]\nlet b = try JSONSerialization.data(withJSONObject:d,options:[.sortedKeys])\nprint(String(data:b,encoding:.utf8)!)\n''')
    result = subprocess.check_output(["swift", str(inspector), str(pdf), str(build / "text.txt")], text=True)
    inspected = json.loads(result)
    if inspected["pages"] > 12:
        raise ValueError(f"Conference draft has {inspected['pages']} pages; limit is 12")
    pdf_text = (build / "text.txt").read_text()
    if any(re.search(pattern, pdf_text, re.I) for pattern in identifiers):
        raise ValueError("Anonymous PDF contains an identifying string")
    for key in ("Author", "Title", "Subject", "Keywords"):
        if inspected["attributes"].get(key, ""):
            raise ValueError("Nonempty PDF metadata: " + key)
    for value in ("78.75", "83.50", "4.75", "3.53", "5.98", "414", "418", "65.96", "68.37"):
        if value not in pdf_text:
            raise ValueError("Missing required result: " + value)
    citation_keys = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source) for k in group.split(",")}
    bbl = (build / "anonymous.bbl").read_text()
    if len(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{", bbl)) != len(citation_keys):
        raise ValueError("Bibliography entry/citation mismatch")
    shutil.copy2(build / "anonymous.bbl", OUT / "anonymous.bbl")
    origin = {"date":"2026-11-21", "official_author_page":"https://link.springer.com/series/558/information-for-authors-and-editors", "template_record":"cache/submission_20261121/lncs/download_record.json", "class_version":"2.25 (03-Sep-2026)", "handling":"Official class, BibTeX style, readme, history and description files copied without alteration; original notices preserved. No standalone LICENSE file was supplied in the official archive."}
    (OUT / "template_origin.json").write_text(json.dumps(origin, indent=2) + "\n")
    alt = "Figure 1 shows an offline recovery-supervision workflow. A fixed initial policy produces failed training episodes. Ordinary continuation uses a common successful prefix; recovery continuation retains the same prefix plus one actual divergence. Reference suffix dependencies are remapped, replay checks observations and answers, and only suffix assistant actions are supervised. Both groups share successful material, question order and target-token exposure."
    (OUT / "figure_alt_text.txt").write_text(alt + "\n")
    archive_files = ["anonymous.tex", "anonymous.bbl", "references.bib", "Fig1.pdf", "llncs.cls", "splncs04.bst", "readme.txt", "history.txt", "DescriptionTexts.txt", "template_origin.json", "figure_alt_text.txt"]
    archive = OUT / "anonymous_source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in archive_files:
            z.write(OUT / name, arcname=name)
    if any(sha(ROOT / p) != h for p, h in input_hashes.items()):
        raise ValueError("Shared sources changed during the build; rerun on the current inputs")
    report = {"schema":"ieaaie_anonymous_build_v1", "date":"2026-11-21", "venue":"IEA/AIE 2027", "status":"compiled_anonymous_backup_not_submitted", "pages":inspected["pages"], "abstract_words_whitespace":words, "cited_references":len(citation_keys), "overfull_boxes":sum("Overfull \\" in line for line in log.splitlines()), "unresolved_references_or_missing_glyphs":0, "metadata":inspected["attributes"], "identifying_string_scan":"passed", "compression":[], "source_sha256":input_hashes, "output_sha256":{name:sha(OUT/name) for name in archive_files+["anonymous.pdf","anonymous_source.zip"]}, "new_model_runs":0, "submitted":False, "visual_review":"Parent agent owns page inspection"}
    (OUT / "build_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ("status","pages","abstract_words_whitespace","cited_references","overfull_boxes","metadata")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
