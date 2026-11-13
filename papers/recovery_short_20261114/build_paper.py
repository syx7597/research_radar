#!/usr/bin/env python3
"""Build matching Chinese/English PDFs and editable Word/Markdown drafts.

Uses local Tectonic, Pandoc and macOS PDFKit; no experiments, model weights or raw
predictions are opened. Table evidence is checked before rendering.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    subprocess.run([sys.executable, str(HERE/"export_tables.py"), "--check"], cwd=ROOT, check=True)
    bibliography = (HERE/"references.bib").read_text()
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bibliography)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate bibliography key")
    reports = {}
    for lang in ("zh", "en"):
        content = (HERE/f"manuscript_{lang}.tex").read_text()
        if re.search(r"雷达|\bradar\b", content, re.I):
            raise ValueError("The short-paper body must be independent of the radar application")
        cites = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", content) for k in group.split(",")}
        if cites - set(keys):
            raise ValueError("Missing bibliography keys: "+str(cites-set(keys)))
        if "4.75" not in content or "414" not in content or "788534" not in content.replace(",", ""):
            raise ValueError("Core evidence missing from a language version")
        reports[lang] = {"cited_keys": sorted(cites), "sections": re.findall(r"\\section\{([^}]+)\}", content),
                         "body_sha256": sha(HERE/f"manuscript_{lang}.tex")}
    if reports["zh"]["cited_keys"] != reports["en"]["cited_keys"]:
        raise ValueError("Bilingual reference sets differ")
    return reports


def compile_tex(engine, entry, out):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.setdefault("TECTONIC_CACHE_DIR", str(ROOT/"cache/tectonic"))
    with (out/"console.log").open("w") as stream:
        subprocess.run([str(engine), "--keep-logs", "--keep-intermediates", "-Z", f"search-path={ROOT}",
                        "--outdir", str(out), str(entry)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log = (out/(entry.stem+".log")).read_text(errors="replace")
    bad = [line for line in log.splitlines() if "undefined references" in line.lower()
           or re.search(r"(?:Citation|Reference) .+ undefined", line) or line.startswith("Missing character:")]
    if bad:
        raise ValueError("Unresolved source or glyph: "+" | ".join(bad[:5]))
    return out/(entry.stem+".pdf"), {"overfull_boxes": sum("Overfull \\" in line for line in log.splitlines()),
                                      "unresolved_references_or_missing_glyphs": 0,
                                      "log_sha256": sha(out/(entry.stem+".log"))}


def render_diagram(pdf, output):
    script = HERE/"render_diagram.swift"
    subprocess.run(["swift", str(script), str(pdf), str(output)], cwd=ROOT, check=True)


def expanded_body(lang):
    body = (HERE/f"manuscript_{lang}.tex").read_text()
    tables = (HERE/"generated_tables.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\([^}]+)\}\{%\n(.*?)\n\}\n", tables, re.S))
    for name, value in macros.items():
        body = body.replace("\\"+name, value.replace("\\end{tabular}%", "\\end{tabular}"))
    meta = json.loads((HERE/"paper_metadata.json").read_text())
    fig = (r"\begin{figure}\centering\includegraphics[width=0.95\linewidth]{assets/method_"+lang+r".png}\caption{"
           +meta["figure_captions"][lang]+r"}\label{fig:method-"+lang+r"}\end{figure}")
    body = body.replace(r"\input{papers/recovery_short_20261114/method_figure_"+lang+r".tex}", fig)
    aux = (HERE/"build"/lang/f"paper_{lang}.aux").read_text()
    refs = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: refs[m[1]], body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    counts = {"table": 0, "figure": 0}
    def numbered_caption(match):
        kind, inside = match[1], match[2]
        counts[kind] += 1
        label = (("表" if kind=="table" else "图") if lang=="zh" else ("Table " if kind=="table" else "Figure "))
        prefix = label+str(counts[kind])+("：" if lang=="zh" else ". ")
        return "\\begin{"+kind+"}"+inside.replace("\\caption{", "\\caption{"+prefix, 1)+"\\end{"+kind+"}"
    body = re.sub(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", numbered_caption, body, flags=re.S)
    return body


def editable(pandoc, lang):
    meta = json.loads((HERE/"paper_metadata.json").read_text())
    source = HERE/"build"/f"editable_{lang}.tex"
    body = expanded_body(lang)
    body = body.replace(r"\begin{abstract}", r"\section*{"+("摘要" if lang=="zh" else "Abstract")+"}")
    body = body.replace(r"\end{abstract}", "")
    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", lambda m: r"\["+m[1]+r"\]", body, flags=re.S)
    body = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", lambda m: r"\[\begin{aligned}"+m[1]+r"\end{aligned}\]", body, flags=re.S)
    body += "\n"+r"\section*{"+("参考文献" if lang=="zh" else "References")+"}\n"
    source.write_text(r"\title{"+meta["titles"][lang]+"}\n"+r"\author{}\date{}\begin{document}\maketitle"+"\n"
                      +body+"\n"+r"\end{document}"+"\n")
    common = [str(pandoc), str(source), "--from=latex", "--citeproc",
              "--bibliography="+str(HERE/"references.bib"), "--csl="+str(ROOT/"cache/paper_build_tools/ieee.csl"),
              "--resource-path="+str(HERE), "--metadata=lang:"+("zh-CN" if lang=="zh" else "en-US")]
    reference = HERE/"assets"/f"word_style_{lang}.docx"
    docx = HERE/f"recovery_supervision_{lang}.docx"
    md = HERE/f"manuscript_{lang}.md"
    for destination, extra in ((docx, ["--standalone", "--reference-doc="+str(reference)]),
                               (md, ["--to=markdown+tex_math_dollars+pipe_tables-simple_tables-multiline_tables-grid_tables", "--wrap=none"])):
        result = subprocess.run(common+extra+["--output="+str(destination)], cwd=HERE, capture_output=True, text=True)
        (HERE/"build"/f"pandoc_{lang}_{destination.suffix[1:]}.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise ValueError(result.stderr)
    markdown = md.read_text()
    markdown = re.sub(r"(?m)^(#{1,6} .+?)\s+\{[^}\n]*\}$", r"\1", markdown)
    markdown = "\n".join(line for line in markdown.splitlines() if not line.strip().startswith(":::"))
    md.write_text("# "+meta["titles"][lang]+"\n\n"+markdown+"\n")
    return docx, md


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--tectonic", type=Path, default=ROOT/"cache/thesis_build_tools/tectonic_0.17.0/tectonic")
    p.add_argument("--pandoc", type=Path, default=ROOT/"cache/paper_build_tools/pandoc")
    args = p.parse_args(); checked = check()
    if args.check_only:
        print(json.dumps({"status": "checked", "languages": checked}, ensure_ascii=False)); return
    outputs, engines = {}, {}
    for name, binary in (("tectonic", args.tectonic), ("pandoc", args.pandoc)):
        engines[name] = subprocess.check_output([str(binary), "--version"], text=True).splitlines()[0]
    for lang in ("zh", "en"):
        diagram, _ = compile_tex(args.tectonic, HERE/f"method_diagram_{lang}.tex", HERE/"build/diagrams")
        render_diagram(diagram, HERE/"assets"/f"method_{lang}.png")
        pdf, checks = compile_tex(args.tectonic, HERE/f"paper_{lang}.tex", HERE/"build"/lang)
        destination = HERE/f"recovery_supervision_{lang}.pdf"; shutil.copy2(pdf, destination)
        docx, md = editable(args.pandoc, lang)
        outputs[lang] = {"files": {str(x.relative_to(ROOT)): sha(x) for x in (destination,docx,md)}, "compile": checks}
        print(json.dumps({"language": lang, "status": "built", "pdf": str(destination.relative_to(ROOT)), **checks}), flush=True)
    source_files = sorted([*HERE.glob("*.tex"), HERE/"references.bib", HERE/"tables_data.json", HERE/"literature_audit.json",
                           HERE/"evidence_audit.json", HERE/"table_export_manifest.json", HERE/"paper_metadata.json",
                           HERE/"export_tables.py", *HERE.glob("assets/word_style_*.docx"),
                           ROOT/"cache/paper_build_tools/ieee.csl", Path(__file__), HERE/"render_diagram.swift"])
    manifest = {"schema": "bilingual_recovery_paper_build_v1", "date": "2026-11-14", "status": "compiled_and_editable_exports_created",
                "scope": "Standalone paper without radar content; two languages of the same study",
                "language_checks": checked, "engines": engines, "outputs": outputs,
                "source_sha256": {str(x.relative_to(ROOT)): sha(x) for x in source_files},
                "new_model_runs": 0, "visual_and_language_review": "recorded separately"}
    (HERE/"build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
