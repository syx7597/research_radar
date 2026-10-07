"""Display every frozen semantic record, without answering or gold filtering."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import time

from .coverage_representation import decode_flat, validate_equal_information
from .source_readings import ROOT, sha, write_new

PREFLIGHT = ROOT / "artifacts/thesis_direction_review/radar_representation_v1/semantic_preflight.json"
INPUTS = ROOT / "data/radar_sources_v2/representation_v1/semantic_payload_v1/inputs.jsonl"
OUTPUT_ROOT = ROOT / "data/radar_sources_v2/evidence_display_v1"


def render(question, records):
    """One escaped JSON block per full record, preserving every field and order."""
    blocks = [json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) for record in records]
    recovered = [json.loads(html.unescape(html.escape(block))) for block in blocks]
    validate_equal_information(records, flat=json.dumps(recovered, ensure_ascii=False))
    body = "\n".join(f'<pre>{html.escape(block)}</pre>' for block in blocks)
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<title>来源资料</title><style>body{max-width:1000px;margin:2em auto;padding:1em;'
            'font-family:system-ui}pre{white-space:pre-wrap;border:1px solid #ddd;padding:1em}</style>'
            f'<h1>{html.escape(question)}</h1><p>以下为检索得到的全部来源记录，未自动筛选答案。</p>'
            f'<p>共 {len(records)} 条记录。</p>{body}</html>')


def build(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT) or output == OUTPUT_ROOT:
        raise ValueError("Use a new local evidence-display subdirectory")
    preflight = json.loads(PREFLIGHT.read_text())
    if sha(INPUTS) != preflight["local_inputs_sha256"]:
        raise ValueError("Frozen model input file changed")
    started = time.perf_counter()
    all_rows = [json.loads(line) for line in INPUTS.read_text().splitlines()]
    rows = [row for row in all_rows if row["arm"] == "flat"]
    if len(rows) != 96 or len({r["question_id"] for r in rows}) != 96:
        raise ValueError("Expected all 96 fixed questions")
    output.mkdir(parents=True)
    counts, files = [], {}
    for ordinal, row in enumerate(rows):
        supplied = json.loads(row["messages"][1]["content"])
        records = decode_flat(json.dumps(supplied["evidence"], ensure_ascii=False))
        page = render(supplied["question"], records)
        path = output / f"{ordinal + 1:03d}.html"
        path.write_text(page, encoding="utf-8")
        files[path.name] = {"question_id": row["question_id"], "sha256": sha(path),
                            "records": len(records)}
        counts.append(len(records))
    result = {"schema": "radar_full_semantic_evidence_display_v1", "questions": len(rows),
              "record_occurrences": sum(counts), "minimum_records": min(counts),
              "maximum_records": max(counts), "roundtrip_verified_pages": len(rows),
              "input_filtering": False, "model_runs": 0, "answer_accuracy": None,
              "seconds_including_input_read_render_and_verification": time.perf_counter() - started,
              "inputs_sha256": {str(PREFLIGHT.relative_to(ROOT)): sha(PREFLIGHT),
                                str(INPUTS.relative_to(ROOT)): sha(INPUTS),
                                str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
              "files": files,
              "scope": "Displays all E_sem only; no question-conditioned selection, generated answer, additional interpretation or access to citation sidecar. Faithful display is not QA accuracy."}
    if result["record_occurrences"] != preflight["represented_record_occurrences"]:
        raise ValueError("Display lost frozen evidence occurrences")
    write_new(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = build(parser.parse_args().output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "inputs_sha256")}))


if __name__ == "__main__":
    main()
