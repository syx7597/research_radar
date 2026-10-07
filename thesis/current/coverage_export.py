#!/usr/bin/env python3
"""Export the fixed 96-question evidence-view study using public aggregates only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = "results/radar_domain/coverage_v2/"
FILES = {"semantic": BASE + "semantic_summary.json", "execution": BASE + "execution_audit.json",
         "protocol": BASE + "protocol.json", "policy": BASE + "evaluation_policy.json",
         "diagnosis": BASE + "posthoc_diagnosis.json"}
LABELS = {"raw": "原文页", "flat": "平铺记录", "bound": "条件绑定"}
GROUPS = {"vaisala_wrs300": "Vaisala WRS300", "eec_ranger_x_band": "EEC Ranger-X",
          "gamic_gmwr": "GAMIC GMWR", "metek_mrr": "METEK MRR",
          "furuno_drs_nxt": "FURUNO DRS-NXT", "jrc_jma5200mk2": "JRC JMA-5200Mk2",
          "garmin_gmr_fantom": "Garmin Fantom", "raymarine_cyclone": "Raymarine Cyclone"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    docs = {k: json.loads((ROOT / p).read_text()) for k, p in FILES.items()}
    hashes = {p: digest(ROOT / p) for p in FILES.values()}
    sem, exe, protocol, policy = (docs[k] for k in ("semantic", "execution", "protocol", "policy"))
    if not exe["passed"] or sem["evaluated_outputs"] != 288:
        raise ValueError("The complete mechanically verified round is required")
    if (sem["protocol_sha256"] != hashes[FILES["protocol"]]
            or exe["protocol_sha256"] != hashes[FILES["protocol"]]
            or sem["policy_sha256"] != hashes[FILES["policy"]]
            or sem["execution_audit_sha256"] != hashes[FILES["execution"]]
            or docs["diagnosis"]["semantic_summary_sha256"] != hashes[FILES["semantic"]]):
        raise ValueError("Public evidence bindings changed")
    rows, tex = [], ["% Generated from public aggregates by coverage_export.py; no private QA is read."]

    def value(table, label, metric, source, pointer):
        obj = docs[source]
        for key in pointer.strip("/").split("/"):
            obj = obj[key]
        rows.append({"table": table, "label": label, "metric": metric, "value": obj,
                     "source": FILES[source], "json_pointer": pointer})
        return obj

    def table(command, headers, body):
        tex.extend(["\\newcommand{\\" + command + "}{%", "\\begin{tabular}{l" + "r" * (len(headers)-1) + "}",
                    "\\toprule", " & ".join(headers) + r" \\", "\\midrule"])
        tex.extend(" & ".join(map(str, r)) + r" \\" for r in body)
        tex.extend(["\\bottomrule", "\\end{tabular}%", "}"])

    body = []
    for arm, label in LABELS.items():
        prefix = f"/adjudicated/overall/{arm}"
        denominator = value("answers", label, "denominator", "semantic", prefix + "/denominator")
        if denominator != 96:
            raise ValueError("Question denominator changed")
        body.append([label] + [value("answers", label, key, "semantic", prefix + "/" + key)
                              for key in ("answer_correct", "joint_correct", "binding_error_any", "refusal")])
    table("CoverageAnswerTable", ["证据输入", "正文正确", "联合正确", "绑定错误", "拒答"], body)
    body = []
    for group, label in GROUPS.items():
        prefix = f"/adjudicated/source_groups/{group}"
        counts = [value("groups", label, arm, "semantic", prefix + f"/{arm}/joint_correct") for arm in LABELS]
        net = value("groups", label, "paired_net", "semantic", prefix + "/paired_bound_minus_flat/net")
        body.append([label, *counts, f"{net:+d}"])
    table("CoverageGroupTable", ["来源组", "原文页", "平铺", "绑定", "绑定减平铺"], body)
    body = []
    for arm, label in LABELS.items():
        metrics = [value("cost", label, key, "execution", f"/arms/{arm}/{key}")
                   for key in ("input_tokens", "generated_tokens", "gpu_hours")]
        body.append([label, f"{metrics[0]:,}", f"{metrics[1]:,}", f"{metrics[2]:.4f}"])
    table("CoverageCostTable", ["证据输入", "输入token", "生成token", "GPU小时"], body)
    for command, pointer in {
            "CoverageAnswerWins": "/answer_correct_bound_minus_flat/wins",
            "CoverageAnswerLosses": "/answer_correct_bound_minus_flat/losses",
            "CoverageAnswerNet": "/answer_correct_bound_minus_flat/net",
            "CoverageJointAlreadyCorrectWins": "/joint_win_strata/both_answers_correct",
            "CoverageJointAnswerWins": "/joint_win_strata/flat_answer_incorrect"}.items():
        number = value("posthoc_diagnosis", "bound_minus_flat", command, "diagnosis", pointer)
        tex.append("\\newcommand{\\" + command + "}{" + str(number) + "}")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    outputs = {"coverage_tables.tex": "\n".join(tex) + "\n", "coverage_results.csv": buf.getvalue()}
    manifest = {"schema": "radar_coverage_thesis_export_v1", "sources_sha256": hashes,
                "code_sha256": digest(Path(__file__)), "rows": len(rows),
                "chapter_sha256": {"thesis/current/06b_evidence_representation.tex": digest(OUT / "06b_evidence_representation.tex")},
                "outputs_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()},
                "source_access": "Only five allowlisted public aggregate/protocol/diagnosis files plus this chapter/code; nested private references are not opened.",
                "limits": "AI references and reviews, not human gold; correlated sources, one fixed round; descriptive counts only."}
    outputs["coverage_evidence_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, count = build()
    for name, text in outputs.items():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError(f"Stale export: {name}")
        else:
            path.write_text(text, encoding="utf-8")
    print(json.dumps({"checked" if args.check else "written": list(outputs), "rows": count}))


if __name__ == "__main__":
    main()
