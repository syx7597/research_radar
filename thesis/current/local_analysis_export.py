#!/usr/bin/env python3
"""Export local construction/source-view/review evidence from public aggregates."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = {
    "construction": "results/agent_feedback/construction_diagnostic_20261113/summary.json",
    "construction_design": "results/agent_feedback/construction_diagnostic_20261113/design.json",
    "platform": "results/agent_feedback/construction_diagnostic_20261113/replay_detail_summary.json",
    "views": "results/radar_domain/dual_view_audit_20261113/summary.json",
    "source": "artifacts/thesis_direction_review/radar_source_review_20261113/summary.safe.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    docs = {k: json.loads((ROOT/p).read_text()) for k, p in FILES.items()}
    hashes = {p: sha(ROOT/p) for p in FILES.values()}
    construction = docs["construction"]
    if (construction["status"] != "completed" or construction["fixed_training_pair_denominator"] != 418
            or construction["compared"] + construction["unavailable"] != 418
            or construction["design"]["sha256"] != hashes[FILES["construction_design"]]
            or construction["new_model_runs"] != 0):
        raise ValueError("Complete fixed construction diagnostic required")
    if docs["views"]["status"] != "passed" or docs["views"]["readings"] != 502:
        raise ValueError("Complete source dual-view audit required")
    if (docs["source"]["reviewer_type"] != "AI" or not docs["source"]["sealed_before_reference_comparison"]
            or docs["source"]["counts"]["semantic_agreement"] != 24):
        raise ValueError("Completed independent AI review required")
    rows, tex = [], ["% Generated from public CPU diagnostics and source-review metadata; no model scores added."]

    def value(table, label, source, pointer):
        v = docs[source]
        for key in pointer.strip("/").split("/"):
            v = v[key]
        rows.append({"table": table, "label": label, "value": v, "source": FILES[source], "json_pointer": pointer})
        return v

    body = []
    for key, label in (("suffix_all_calls_execute", "后缀全部调用可执行"),
                       ("submitted_answer", "成功提交答案"),
                       ("final_answer_correct", "答案与参考一致"),
                       ("reference_observations_match", "后缀执行观察与参考一致"),
                       ("full_gate_pass", "完整构造验收通过")):
        values = [value("construction", label+":"+arm, "construction", "/arms/"+arm+"/"+key)
                  for arm in ("direct", "mapped_unfiltered")]
        body.append([label, *values])
    tex += [r"\newcommand{\LocalConstructionTable}{%", r"\begin{tabular}{lrr}", r"\toprule",
            r"检查项 & 原编号拼接 & 依赖映射执行 \\", r"\midrule"]
    tex += [" & ".join(map(str, row))+r" \\" for row in body]
    tex += [r"\bottomrule", r"\end{tabular}%", "}"]
    commands = {
        "LocalConstructionTotal": ("construction", "/fixed_training_pair_denominator"),
        "LocalConstructionCompared": ("construction", "/compared"),
        "LocalConstructionUnavailable": ("construction", "/unavailable"),
        "LocalViewQueries": ("views", "/source_field_queries_checked"),
        "LocalSourceReviewed": ("source", "/counts/semantic_agreement"),
        "LocalSourceSpans": ("source", "/counts/checked_citation_spans"),
    }
    for command, (source, pointer) in commands.items():
        v = value("macros", command, source, pointer)
        tex.append("\\newcommand{\\"+command+"}{"+str(v)+"}")
    for source, pointer in (("views", "/subject_reading_edges"), ("views", "/shared_subject_readings"),
                            ("views", "/citation_spans_checked"),
                            ("construction", "/arms/direct/any_step_dependency_mismatch"),
                            ("construction", "/arms/direct/finish_dependency_mismatch"),
                            ("construction", "/paired_outcomes/direct_answer_equal_despite_reference_dependency_mismatch")):
        value("prose", pointer, source, pointer)
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    outputs = {"local_analysis_tables.tex": "\n".join(tex)+"\n", "local_analysis_results.csv": buf.getvalue()}
    chapters = ("03d_source_view_validation.tex", "03e_fixed_source_review.tex",
                "05b_construction_diagnostic.tex", "05c_same_state_design.tex")
    manifest = {"schema": "local_research_thesis_export_v1", "date": "2026-11-13", "sources_sha256": hashes,
        "exporter_sha256": sha(__file__), "rows": len(rows),
        "chapter_sha256": {str((HERE/p).relative_to(ROOT)): sha(HERE/p) for p in chapters},
        "outputs_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in outputs.items()},
        "access": "Only allowlisted public aggregates; full sources, questions and per-pair histories stay private",
        "new_model_evaluations": 0, "review_type": "AI"}
    outputs["local_analysis_evidence_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)+"\n"
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--check", action="store_true")
    args = p.parse_args()
    outputs = build()
    for name, content in outputs.items():
        target = HERE/name
        if args.check:
            if not target.exists() or target.read_text() != content:
                raise ValueError("Stale local evidence export: "+name)
        else:
            target.write_text(content)
    print(json.dumps({"checked" if args.check else "written": list(outputs)}))


if __name__ == "__main__":
    main()
