#!/usr/bin/env python3
"""Check public evidence and build the seven-chapter review manuscript.

Uses an already available Tectonic or XeLaTeX installation; never installs tools,
starts experiments, or reads private questions/source documents. Run from any cwd.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENTRY = HERE / "main.tex"
EXPORTS = ("export_evidence.py", "data_export.py", "domain_export.py", "coverage_export.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_inputs(path, seen=None):
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Duplicate/circular manuscript input: {path.relative_to(ROOT)}")
    if not path.is_relative_to(HERE):
        raise ValueError("Manuscript input must stay in thesis/current")
    seen.add(path)
    text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
    parts = [text]
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        child = ROOT / name
        if not child.suffix:
            child = child.with_suffix(".tex")
        parts.append(scan_inputs(child, seen)[1])
    return seen, "\n".join(parts)


def check_sources():
    import sys
    for exporter in EXPORTS:
        subprocess.run([sys.executable, str(HERE / exporter), "--check"], cwd=ROOT, check=True)
    files, text = scan_inputs(ENTRY)
    chapters = re.findall(r"\\chapter\{([^}]+)\}", text)
    if len(chapters) != 7:
        raise ValueError(f"Expected exactly seven numbered chapters, got {len(chapters)}")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    if len(labels) != len(set(labels)) or set(refs) - set(labels):
        raise ValueError("Duplicate label or missing cross-reference")
    bib = HERE / "references.bib"
    bibkeys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib.read_text())
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}
    if len(bibkeys) != len(set(bibkeys)) or citations - set(bibkeys):
        raise ValueError(f"Duplicate bibliography key or missing citation: {sorted(citations-set(bibkeys))}")
    files.add(bib)
    files.add(HERE / "literature_audit.json")
    for name in ("evidence_manifest.json", "data_evidence_manifest.json", "domain_evidence_manifest.json", "coverage_evidence_manifest.json"):
        files.add(HERE / name)
    return {"chapters": chapters, "numbered_chapters": len(chapters), "labels": len(labels),
            "cited_works": len(citations), "bibliography_entries": len(bibkeys),
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--engine", choices=("tectonic", "xelatex"), default="tectonic")
    parser.add_argument("--engine-path", help="Existing compiler path; not downloaded automatically")
    parser.add_argument("--outdir", type=Path, default=HERE / "build")
    args = parser.parse_args()
    checked = check_sources()
    if args.check_only:
        print(json.dumps({"status": "sources_checked", **checked}, ensure_ascii=False))
        return
    binary = args.engine_path or shutil.which(args.engine)
    if not binary:
        raise SystemExit("Compiler unavailable. Use --engine-path with an existing Tectonic, or --engine xelatex.")
    binary = str(Path(binary).resolve())
    out = args.outdir.resolve()
    if not out.is_relative_to(HERE) or out == HERE:
        raise ValueError("Use a separate output directory inside thesis/current")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # A legitimate compiler cache setting, not a replacement for HOME.
    env.setdefault("TECTONIC_CACHE_DIR", str(ROOT / "cache" / "tectonic"))
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if args.engine == "tectonic":
        commands = [[binary, "--keep-logs", "--keep-intermediates",
                     "-Z", f"search-path={ROOT}", "--outdir", str(out), str(ENTRY)]]
    else:
        bibtex = shutil.which("bibtex")
        if not bibtex:
            raise SystemExit("XeLaTeX route requires bibtex in PATH")
        tex = [binary, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", str(ENTRY)]
        commands = [tex, [bibtex, str(out / "main")], tex, tex]
    with (out / "build-console.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    pdf, logfile = out / "main.pdf", out / "main.log"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise ValueError("Compiler did not produce a PDF")
    log = logfile.read_text(errors="replace")
    blockers = [line for line in log.splitlines() if "undefined references" in line.lower()
                or re.search(r"(?:Citation|Reference) .+ undefined", line) or line.startswith("Missing character:")]
    if blockers:
        raise ValueError("PDF has unresolved references or missing glyphs: " + " | ".join(blockers[:8]))
    report = {"schema": "radar_seven_chapter_build_v1", "status": "compiled",
              **checked, "engine": version, "builder_sha256": sha(Path(__file__)),
              "pdf": {"path": str(pdf.relative_to(ROOT)), "sha256": sha(pdf), "bytes": pdf.stat().st_size},
              "final_log_sha256": sha(logfile),
              "overfull_box_warnings": sum("Overfull \\" in line for line in log.splitlines()),
              "unresolved_references_or_missing_glyphs": 0,
              "limits": ["Generic review layout, not a university-approved thesis template.",
                         "Compilation and source consistency do not establish factual validity or human gold.",
                         "Visual review is recorded separately; this compiler report alone is not visual approval."]}
    (HERE / "manuscript_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "chapters": 7, "cited_works": checked["cited_works"], "pdf": report["pdf"],
                      "overfull_box_warnings": report["overfull_box_warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
