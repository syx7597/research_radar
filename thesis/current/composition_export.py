#!/usr/bin/env python3
"""Export the separate composition-evaluation preparation from public metadata."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = "results/radar_domain/compositional_supplement_20261108/"
FILES = {"summary": BASE + "summary.json", "design": BASE + "design.json",
         "review": BASE + "source_review.safe.json", "matrix": BASE + "experiment_matrix.json",
         "previous_coverage": "results/radar_domain/frozen_20261107/task_coverage_review.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    docs = {k: json.loads((ROOT / p).read_text()) for k, p in FILES.items()}
    hashes = {p: sha(ROOT / p) for p in FILES.values()}
    summary, review = docs["summary"], docs["review"]
    assert summary["status"] == "frozen_before_model_runs" and summary["question_count"] == 12
    assert summary["design"]["sha256"] == hashes[FILES["design"]]
    assert summary["independent_review"]["sha256"] == hashes[FILES["review"]]
    assert summary["original_24_unchanged"] and summary["model_runs"] == 0
    assert not review["blocking_findings"] and len(review["accepted_ids"]) == 12
    tex = ["% Generated from public source-review and CPU metadata, not model predictions."]
    rows = []

    def value(table, label, source, pointer):
        v = docs[source]
        for key in pointer.strip("/").split("/"):
            v = v[int(key)] if isinstance(v, list) else v[key]
        rows.append({"table": table, "label": label, "value": v, "source": FILES[source], "json_pointer": pointer})
        return v

    def table(command, header, body):
        tex.extend(["\\newcommand{\\" + command + "}{%", "\\begin{tabular}{l" + "r"*(len(header)-1) + "}",
                    "\\toprule", " & ".join(header) + r" \\", "\\midrule"])
        tex.extend(" & ".join(map(str, r)) + r" \\" for r in body)
        tex.extend(["\\bottomrule", "\\end{tabular}%", "}"])

    names = {"furuno_drs4w": "FURUNO DRS4W", "ott_rls": "OTT RLS",
             "smartmicro_umrr11": "smartmicro UMRR-11", "campbell_cs475a": "Campbell CS475A"}
    body = []
    for family, label in names.items():
        count = value("sources", label, "summary", "/family_counts/" + family)
        calls = []
        for i, profile in enumerate(summary["question_profiles"]):
            if profile["family_id"] == family:
                calls.append(value("sources", profile["id"], "summary", f"/question_profiles/{i}/calls"))
        body.append([label, count, min(calls), max(calls)])
    table("CompositionSourceTable", ["来源家族", "题数", "最少调用", "最多调用"], body)
    body = []
    for op in ("Expand", "Join", "Filter", "Derive", "Assess"):
        before = value("operations", op, "previous_coverage", "/profiles/holdout/question_counts_using_operation/" + op)
        after = sum(op in p["live_operations"] for p in summary["question_profiles"])
        rows.append({"table": "operations", "label": op, "value": after, "source": FILES["summary"],
                     "json_pointer": "/question_profiles/*/live_operations (question count)"})
        body.append(["\\texttt{" + op + "}", before, after])
    table("CompositionOperationTable", ["操作", "原24题", "组合12题"], body)
    for command, pointer in (("CompositionQuestionCount", "/question_count"),
                             ("CompositionCalls", "/reference_calls"),
                             ("CompositionMaxContext", "/token_preflight/max_context_tokens"),
                             ("CompositionMaxAction", "/token_preflight/max_action_tokens"),
                             ("CompositionMaxTargets", "/token_preflight/max_total_action_tokens")):
        v = value("macros", command, "summary", pointer)
        tex.append("\\newcommand{\\" + command + "}{" + str(v) + "}")
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    outputs = {"composition_tables.tex": "\n".join(tex) + "\n", "composition_results.csv": buf.getvalue()}
    chapters = ["01_introduction.tex", "01b_experiment_matrix.tex", "03c_composition_evaluation.tex", "06d_composition_validation.tex"]
    manifest = {"schema": "radar_composition_thesis_export_v1", "date": "2026-11-08", "rows": len(rows),
                "sources_sha256": hashes, "exporter_sha256": sha(__file__),
                "chapter_sha256": {str((HERE/n).relative_to(ROOT)): sha(HERE/n) for n in chapters},
                "outputs_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in outputs.items()},
                "source_access": "Only allowlisted public summaries; full questions and source answers remain private",
                "evidence_type": "Independent AI source review and deterministic reference/token checks", "model_runs": 0}
    outputs["composition_evidence_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--check", action="store_true")
    args = p.parse_args(); outputs = build()
    for name, content in outputs.items():
        target = HERE / name
        if args.check:
            if not target.exists() or target.read_text() != content:
                raise ValueError("Stale export: " + name)
        else:
            target.write_text(content)
    print(json.dumps({"checked" if args.check else "written": list(outputs)}))


if __name__ == "__main__":
    main()
