"""Build a Discover Computing review draft from the shared, audited study.

Uses a 12-point author manuscript consistent with the journal's initial-submission
instructions. This is not a publisher production template and makes no submission.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'journal'
ENGINE=ROOT/'cache/thesis_build_tools/tectonic_0.17.0/tectonic'
PANDOC=ROOT/'cache/paper_build_tools/pandoc'
PYDOCX=ROOT/'cache/proposal_env/bin/python'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expand():
    body=(HERE/'manuscript_body_en.tex').read_text()
    macros=dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{%\n(.*?)\n\}\n',(HERE/'tables_en.tex').read_text(),re.S))
    for name,val in macros.items():body=body.replace('\\'+name,val.replace('\\end{tabular}%',r'\end{tabular}'))
    # Cite the related-work table in numerical order as required by the guide.
    return body


def main():
    OUT.mkdir(exist_ok=True);build=OUT/'build';build.mkdir(exist_ok=True)
    meta=json.loads((HERE/'paper_metadata.json').read_text())
    abstract=(HERE/'abstract_en.tex').read_text().strip();body=expand()
    declarations=(HERE/'declarations_journal.tex').read_text()
    preamble=r'''\documentclass[a4paper,12pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{fontspec}\setmainfont{Arial}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,url}
\usepackage[hidelinks,unicode]{hyperref}
\usepackage{caption}
\captionsetup[figure]{name=Fig.,labelsep=space}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.25em}
\renewcommand{\arraystretch}{1.1}
\renewenvironment{abstract}{\begin{quote}\noindent\textbf{Abstract}\par\normalsize}{\end{quote}}
\hypersetup{pdfauthor={},pdftitle={TITLE}}
\title{TITLE}
\author{Author information to be completed}
\date{}
\begin{document}\maketitle
'''.replace('TITLE',meta['title'])
    full=preamble+'\n\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n\n'+r'\noindent\textbf{Keywords:} '+'; '.join(meta['keywords'])+'\n\n'+body+'\n'+declarations+'\n'+r'\bibliographystyle{unsrt}\bibliography{references}'+'\n'+r'\end{document}'+'\n'
    tex=OUT/'manuscript.tex';tex.write_text(full)
    for name in ['references.bib','Fig1.pdf','Fig1.eps','Fig1.png']:shutil.copy2(HERE/name,OUT/name)
    env=dict(os.environ);env.setdefault('TECTONIC_CACHE_DIR',str(ROOT/'cache/tectonic'))
    with (build/'console.log').open('w') as log:
        subprocess.run([str(ENGINE),'--keep-logs','--keep-intermediates','--outdir',str(build),str(tex)],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    logfile=(build/'manuscript.log').read_text(errors='replace')
    bad=[l for l in logfile.splitlines() if 'undefined references' in l.lower() or re.search(r'(Citation|Reference) .+ undefined',l) or l.startswith('Missing character:')]
    if bad:raise ValueError(bad)
    pdf=OUT/'discover_computing_review.pdf';shutil.copy2(build/'manuscript.pdf',pdf)
    # Resolve numeric cross-references for Word and retain native equation objects.
    refs=dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',(build/'manuscript.aux').read_text()))
    edit=body+'\n'+declarations
    edit=re.sub(r'\\ref\{([^}]+)\}',lambda m:refs[m[1]],edit)
    edit=re.sub(r'\\label\{[^}]+\}','',edit).replace('{Fig1.pdf}','{Fig1.png}')
    counts={'table':0,'figure':0}
    def caption(m):
        k=m[1];counts[k]+=1
        prefix=('Table ' if k=='table' else 'Fig. ')+str(counts[k])+' '
        return '\\begin{'+k+'}'+m[2].replace('\\caption{','\\caption{'+prefix,1)+'\\end{'+k+'}'
    edit=re.sub(r'\\begin\{(table|figure)\}(.*?)\\end\{\1\}',caption,edit,flags=re.S)
    edit=re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}',lambda m:r'\['+m[1]+r'\]',edit,flags=re.S)
    edit=re.sub(r'\\begin\{align\}(.*?)\\end\{align\}',lambda m:r'\[\begin{aligned}'+m[1]+r'\end{aligned}\]',edit,flags=re.S)
    edit=r'\title{'+meta['title']+r'}\author{Author information to be completed}\date{}\begin{document}\maketitle'+'\n'+r'\section*{Abstract}'+'\n'+abstract+'\n'+r'\noindent\textbf{Keywords:} '+'; '.join(meta['keywords'])+'\n'+edit+'\n'+r'\section*{References}\end{document}'
    editable=build/'editable.tex';editable.write_text(edit)
    style=build/'word_style.docx'
    create_style="""from docx import Document
from docx.shared import Pt,Cm
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import sys
d=Document(sys.argv[1])
for s in d.styles:
 if s.type==WD_STYLE_TYPE.PARAGRAPH:
  s.font.name='Arial';s.font.size=Pt(12)
  s.paragraph_format.line_spacing=1.0
for name in ['Table Caption','Image Caption','Captioned Figure','Bibliography']:
 if name not in d.styles:d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
 d.styles[name].font.name='Arial';d.styles[name].font.size=Pt(12)
for section in d.sections:
 section.top_margin=section.bottom_margin=section.left_margin=section.right_margin=Cm(2.5)
 p=section.footer.paragraphs[0];p.alignment=1
 field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field)
d.save(sys.argv[2])
"""
    subprocess.run([str(PYDOCX),'-c',create_style,str(ROOT/'papers/recovery_short_20261114/assets/word_style_en.docx'),str(style)],check=True)
    docx=OUT/'discover_computing_review.docx'
    result=subprocess.run([str(PANDOC),str(editable),'--from=latex','--standalone','--citeproc','--bibliography='+str(OUT/'references.bib'),'--csl='+str(ROOT/'cache/paper_build_tools/ieee.csl'),'--resource-path='+str(OUT),'--reference-doc='+str(style),'--metadata=lang:en-US','--output='+str(docx)],cwd=OUT,text=True,capture_output=True)
    (build/'pandoc.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    archive=OUT/'discover_computing_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['manuscript.tex','references.bib','Fig1.pdf','Fig1.eps']:z.write(OUT/name,name)
    record={'date':'2026-11-21','target':'Discover Computing','article_type':'Research','status':'review_draft_author_fields_pending','format':'12-point Arial author manuscript, not publisher production template','abstract_words':meta['abstract_words_whitespace'],'keywords':len(meta['keywords']),'heading_levels_max':3,'overfull_boxes':logfile.count('Overfull \\'),'unresolved_citations_or_glyphs':0,'new_model_runs':0,'submitted':False,'sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'manuscript_body_en.tex',HERE/'abstract_en.tex',HERE/'tables_en.tex',HERE/'declarations_journal.tex',Path(__file__)]},'outputs_sha256':{p.name:sha(p) for p in [tex,pdf,docx,archive,OUT/'references.bib',OUT/'Fig1.pdf',OUT/'Fig1.eps']}}
    (OUT/'build_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf.relative_to(ROOT)),'overfull_boxes':record['overfull_boxes'],'abstract_words':record['abstract_words']}))


if __name__=='__main__':main()
