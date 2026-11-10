"""Fill the supplied ECNU proposal form, retaining personal/signature fields."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMPLATE = ROOT / "华东师范大学研究生学位论文开题报告登记表.docx"
OUTPUT = HERE / "华东师范大学研究生学位论文开题报告登记表_已填写.docx"


def font(run, size=12, bold=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    rfonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "Songti SC")
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")


def paragraph_format(p, *, heading=False, reference=False):
    f = p.paragraph_format
    f.alignment = WD_ALIGN_PARAGRAPH.LEFT if heading or reference else WD_ALIGN_PARAGRAPH.JUSTIFY
    f.first_line_indent = Pt(0 if heading or reference else 24)
    f.space_before = Pt(5 if heading else 0)
    f.space_after = Pt(4 if heading else 3)
    f.line_spacing = Pt(15 if reference else 20)
    f.keep_with_next = heading
    f.keep_together = heading
    f.widow_control = True
    props = p._p.get_or_add_pPr()
    for old in list(props.findall(qn("w:snapToGrid"))):
        props.remove(old)
    node = OxmlElement("w:snapToGrid")
    node.set(qn("w:val"), "0")
    props.append(node)


def border(cell, side):
    props = cell._tc.get_or_add_tcPr()
    borders = props.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        props.append(borders)
    node = OxmlElement("w:" + side)
    node.set(qn("w:val"), "single")
    node.set(qn("w:sz"), "4")
    node.set(qn("w:color"), "000000")
    borders.append(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=HERE / "content.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--record", type=Path, default=HERE / "build_record.json")
    parser.add_argument("--join-section-openings", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    content = json.loads(args.content.read_text())
    doc = Document(TEMPLATE)
    assert len(doc.tables) == 2 and len(doc.tables[1].rows) == 8
    original_personal = [
        deepcopy(row.cells[1]._tc) for row in doc.tables[0].rows[1:]
    ]
    original_signature = deepcopy(doc.tables[1].rows[7]._tr)
    title = doc.tables[0].cell(0, 1)
    title.text = content["title"]
    title.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for p in title.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = Pt(21)
        for run in p.runs:
            font(run, 14)

    table = doc.tables[1]
    original_rows = [deepcopy(row._tr) for row in table.rows]
    for row in list(table._tbl.findall(qn("w:tr"))):
        table._tbl.remove(row)
    # Paragraph-sized continuation rows preserve the seven bordered sections
    # while avoiding oversized single cells in Word-compatible readers.
    table._tbl.tblPr.find(qn("w:tblBorders")).find(qn("w:insideH")).set(qn("w:val"), "nil")
    section_ranges = []
    for i, section in enumerate(content["sections"]):
        start = len(table.rows)
        table._tbl.append(deepcopy(original_rows[i]))
        row = table.rows[-1]
        trpr = row._tr.get_or_add_trPr()
        # The blank template's very tall minimum rows are writing space. Filled
        # content determines row height and may continue across pages.
        for tag in ("w:trHeight", "w:cantSplit"):
            for node in list(trpr.findall(qn(tag))):
                trpr.remove(node)
        cell = row.cells[0]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        for p in list(cell.paragraphs):
            if not p.text.strip():
                cell._tc.remove(p._p)
            else:
                paragraph_format(p, heading=True)
                is_english = not re.search(r"[\u4e00-\u9fff]", p.text)
                for run in p.runs:
                    font(run, 10 if is_english else 12, not is_english)
        border(cell, "top")
        groups, pending = [], []
        for line in section.splitlines():
            if line.strip():
                pending.append(line)
                if not line.startswith("#"):
                    groups.append(pending)
                    pending = []
        if pending:
            groups.append(pending)
        for group_index, group in enumerate(groups):
            joined_opening = args.join_section_openings and group_index == 0
            cell = table.rows[start].cells[0] if joined_opening else table.add_row().cells[0]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            for j, line in enumerate(group):
                heading = line.startswith("#")
                reference = bool(re.match(r"^\[\d+\]", line))
                text = re.sub(r"^#{2,3}\s+", "", line)
                p = cell.paragraphs[0] if j == 0 and not joined_opening else cell.add_paragraph()
                paragraph_format(p, heading=heading, reference=reference)
                font(p.add_run(text), 10 if reference else 12, heading)
        border(table.rows[-1].cells[0], "bottom")
        section_ranges.append((start, len(table.rows)))
    table._tbl.append(deepcopy(original_rows[7]))
    if args.join_section_openings:
        # Word keeps an empty paragraph after a table. Keep that required
        # paragraph compact so it does not create an otherwise blank last page.
        tail = table._tbl.getnext()
        while tail is not None and tail.tag == qn("w:p"):
            if not "".join(tail.itertext()).strip():
                from docx.text.paragraph import Paragraph
                p = Paragraph(tail, doc)
                paragraph_format(p)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = Pt(1)
                font(p.add_run(""), 1)
            tail = tail.getnext()

    doc.core_properties.title = content["title"]
    doc.core_properties.subject = "华东师范大学研究生学位论文开题报告"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.save(output)

    # Inspect saved OOXML to ensure content and unfilled personal fields survived.
    saved = Document(output)
    canonical = lambda node: etree.tostring(node, method="c14n", exclusive=True)
    for before, row in zip(original_personal, saved.tables[0].rows[1:]):
        assert canonical(before) == canonical(row.cells[1]._tc), "Personal field changed"
    assert canonical(original_signature) == canonical(saved.tables[1].rows[-1]._tr), "Signature area changed"
    for i, section in enumerate(content["sections"]):
        first, last = section_ranges[i]
        actual = "\n".join(row.cells[0].text for row in saved.tables[1].rows[first:last])
        for line in section.splitlines():
            if line.strip():
                assert re.sub(r"^#{2,3}\s+", "", line) in actual
    body = "\n".join(content["sections"])
    assert not re.search(r"不能|不代表|不构成|不声称|不冒充|不等于|审辩", body)
    refs = re.findall(r"^\[(\d+)\]", content["sections"][0], re.M)
    assert refs and refs == [str(i) for i in range(1, len(refs) + 1)]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "template": str(TEMPLATE.relative_to(ROOT)),
        "template_sha256": digest(TEMPLATE),
        "content": str(args.content.resolve().relative_to(ROOT)),
        "content_sha256": digest(args.content),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": digest(output),
        "sections_filled": 7,
        "references": len(refs),
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", body)),
        "personal_fields_unchanged": True,
        "signature_area_unchanged": True,
        "section_row_ranges": section_ranges,
        "layout": "Original seven form sections with paragraph-sized continuation rows",
        "section_headings_grouped_with_opening_paragraph": args.join_section_openings,
        "source_results": [
            "results/agent_feedback/holdout_v1_analysis.json",
            "results/agent_feedback/state_transitions_v1/summary.json",
            "results/radar_domain/coverage_v2/semantic_summary.json",
            "results/radar_domain/coverage_v2_citation_sensitivity_v1/semantic_summary.json",
        ],
    }
    args.record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
