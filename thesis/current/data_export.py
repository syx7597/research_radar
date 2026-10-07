#!/usr/bin/env python3
"""Export the data-construction chapter from public metadata and code hashes.

No raw corpus, private QA, model outputs or weights are opened. The fixed
allowlist is not expanded through nested manifests. --check never writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ART = "artifacts/thesis_direction_review/"
FILES = {
    "inventory": "artifacts/data_manifest.json",
    "profile": "artifacts/public_benchmark_profile.json",
    "candidate": ART + "radar_review_batch/source_manifest.json",
    "workflow": ART + "radar_review_workflow/audit_report.json",
    "cross_review": ART + "radar_ai_cross_review_v1/summary.json",
    "development": ART + "radar_development_v2/manifest.json",
    "calibration": ART + "radar_lookup_calibration_v1/manifest.json",
    "synthetic": ART + "radar_interface_synthetic_v1/manifest.json",
    "synthetic_quality": ART + "radar_interface_synthetic_v1/quality.json",
    "synthetic_audit": ART + "radar_interface_synthetic_v1/independent_ai_audit.json",
    "sources": ART + "radar_sources_v1/manifest.json",
    "source_index": ART + "radar_sources_v1/source_index.json",
    "source_audit": ART + "radar_sources_v1/independent_ai_audit.json",
}
IMPLEMENTATION = (
    "scripts/audit_workspace.py", "scripts/profile_public_benchmark.py",
    "scripts/prepare_radar_review_packet.py", "scripts/audit_radar_review_workflow.py",
    "experiments/radar_domain/development_environment.py",
    "experiments/radar_domain/development_data.py",
    "experiments/radar_domain/interface_data.py", "experiments/radar_domain/source_packet.py",
    "docs/research/RADAR_DATA_ANNOTATION.md", "templates/radar_review/facts.csv",
    "templates/radar_review/questions.csv",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build():
    docs = {key: json.loads((ROOT / path).read_text()) for key, path in FILES.items()}
    hashes = {path: sha(ROOT / path) for path in (*FILES.values(), *IMPLEMENTATION)}
    checked_bindings = 0
    for doc in docs.values():
        for field in ("inputs_sha256", "files_sha256", "reviewed_file_sha256"):
            for path, expected in doc.get(field, {}).items():
                if path in hashes:
                    require(expected == hashes[path], f"Bound source changed: {path}")
                    checked_bindings += 1
    old, profile = docs["inventory"]["summary"], docs["profile"]["radar"]
    require(old["entities"] == profile["entities"], "Historical entity snapshot differs")
    inventory_hashes = {f["path"]: f["sha256"] for f in docs["inventory"]["files"]}
    entity_input = docs["profile"]["inputs"]["radar_entities"]
    require(entity_input["sha256"] == inventory_hashes[entity_input["path"]], "Historical profile input binding differs")
    require(sum(profile["attribute_counts"].values()) == profile["attribute_records"], "Attribute counts differ")
    for split in old["splits"].values():
        require(sum(split["types"].values()) == split["count"], "Legacy split count differs")
    candidate, workflow = docs["candidate"], docs["workflow"]
    require(candidate["facts"] == workflow["facts"] == docs["development"]["record_count"] == 12, "Old development scope changed")
    require(workflow["machine_checks_passed"] and workflow["exact_source_bindings"] == candidate["facts"], "Old source-binding audit failed")
    require(candidate["accepted_gold_records"] == workflow["review_decisions"]["independent_fact_eligible"] == 0, "Unexpected human gold admission")
    require(not docs["development"]["human_verified"] and not docs["development"]["independent_gold"], "Development was relabeled as gold")
    require(docs["calibration"]["permanently_excluded_from_final_evaluation"], "Exposed calibration exclusion changed")
    require(docs["calibration"]["question_count_per_language"] == docs["development"]["question_count"] == 12, "Calibration scope differs")
    require(sum(docs["cross_review"]["development_dispositions"].values()) == docs["cross_review"]["reviewed_records"] == 12, "AI disposition counts differ")
    syn, quality = docs["synthetic"], docs["synthetic_quality"]
    require(syn["synthetic"] and quality["status"] == "cpu_validated", "Synthetic material not validated")
    require(sum(v["groups"] for v in syn["split_counts"].values()) == quality["group_count"] == 100, "Synthetic group count differs")
    require(sum(v["questions"] for v in syn["split_counts"].values()) == quality["bilingual_questions"] == 200, "Synthetic view count differs")
    require(sum(quality["family_value_kind_counts"].values()) == quality["group_count"], "Synthetic family counts differ")
    require(docs["synthetic_audit"]["status"] == "passed" and docs["synthetic_audit"]["data_manifest_sha256"] == hashes[FILES["synthetic"]], "Synthetic audit binding differs")
    source, index, audit = (docs[k] for k in ("sources", "source_index", "source_audit"))
    require(source["status"] == "cpu_validated" and audit["status"] == "passed", "Source packet admission incomplete")
    require(not source["human_gold"] and not source["independent_final_evaluation"], "Exploratory packet relabeled as final gold")
    require(len(index["sources"]) == source["source_count"] == 4, "Source count differs")
    require(sum(x["page_count"] for x in index["sources"]) == source["chunk_count"] == 15, "Page count differs")
    require(sum(source["question_counts_by_source"].values()) == source["question_count"] == 24, "Question count differs")
    require(sum(source["record_counts_by_value_kind"].values()) == source["record_count"] == 62, "Reading count differs")
    require(audit["facts_review"]["record_readings_passed"] == source["record_count"], "Source reading review count differs")
    require(audit["facts_review"]["semantic_reference_questions_reviewed"] == source["question_count"], "Question review count differs")

    rows, tex = [], ["% Generated by data_export.py; metadata counts are not verified equipment facts."]

    def row(table, label, metric, value, source, pointer, derivation="copied", unit="count"):
        rows.append(dict(table=table, label=label, metric=metric, value=value, unit=unit,
                         source=FILES[source], json_pointer=pointer, derivation=derivation))

    def table(command, headers, records, columns):
        tex.extend(["\\newcommand{\\" + command + "}{%", "\\begin{tabular}{" + columns + "}",
                    "\\toprule", " & ".join(headers) + r" \\", "\\midrule"])
        tex.extend(" & ".join(map(str, record)) + r" \\" for record in records)
        tex.extend(["\\bottomrule", "\\end{tabular}%", "}"])

    inventory_rows = []
    for key, label in (("entities", "实体记录"), ("raw_edges", "边记录"), ("relations", "关系名称"),
                       ("rl_filtered_edge_records", "旧工具过滤后边记录"), ("sft_trajectories", "旧 SFT 轨迹")):
        row("legacy_inventory", "snapshot", key, old[key], "inventory", "/summary/" + key)
        inventory_rows.append([label, f"{old[key]:,}"])
    for key, label in (("attribute_records", "属性记录"), ("with_numeric_value_and_unit", "含数值及单位的属性"),
                       ("with_condition_or_qualifier_field", "具所查显式条件字段的属性")):
        row("legacy_inventory", "snapshot", key, profile[key], "profile", "/radar/" + key)
        inventory_rows.append([label, f"{profile[key]:,}"])
    table("DataLegacyInventoryTable", ["历史资产口径", "数量"], inventory_rows, "lr")
    split_rows = []
    for split, value in old["splits"].items():
        row("legacy_split", split, "questions", value["count"], "inventory", f"/summary/splits/{split}/count")
        split_rows.append([split, f"{value['count']:,}"])
    total = sum(v["count"] for v in old["splits"].values())
    row("legacy_split", "all", "questions", total, "inventory", "/summary/splits", "sum(split.count)")
    split_rows.append(["合计", f"{total:,}"])
    table("DataLegacySplitTable", ["原有标签", "自动问题数"], split_rows, "lr")
    for key in ("train_test_anchor_overlap", "test_explicit_query_entities", "train_test_explicit_query_entity_overlap"):
        row("legacy_isolation", "train_test", key, old[key], "inventory", "/summary/" + key)
    for key in ("facts", "selected_existing_attributes", "accepted_gold_records"):
        row("old_development", "candidate", key, candidate[key], "candidate", "/" + key)
    for key in ("exact_source_bindings", "secondary_source_snapshots"):
        row("old_development", "workflow", key, workflow[key], "workflow", "/" + key)
    for key, value in workflow["review_decisions"].items():
        row("old_development", "review_decisions", key, value, "workflow", "/review_decisions/" + key)
    for key, value in docs["cross_review"]["development_dispositions"].items():
        row("old_development", "AI_disposition", key, value, "cross_review", "/development_dispositions/" + key)
    for key in ("record_count", "question_count"):
        row("old_development", "v2", key, docs["development"][key], "development", "/" + key)
    row("old_development", "bilingual_calibration", "questions_per_language", docs["calibration"]["question_count_per_language"], "calibration", "/question_count_per_language")

    split_rows = []
    for split, value in syn["split_counts"].items():
        require(value["questions"] == 2 * value["groups"], "Bilingual group split differs")
        split_rows.append([split, value["groups"], value["questions"]])
        for key, count in value.items():
            row("synthetic_split", split, key, count, "synthetic", f"/split_counts/{split}/{key}")
    split_rows.append(["合计", quality["group_count"], quality["bilingual_questions"]])
    table("DataSyntheticSplitTable", ["用途", "虚构实例组", "中英文视图"], split_rows, "lrr")
    for key in ("group_count", "bilingual_questions", "successful_program_validations", "successful_agent_trajectories", "invalid_or_recovery_training_calls"):
        row("synthetic_validation", "all", key, quality[key], "synthetic_quality", "/" + key)
    for key in ("source_instances", "source_records_checked"):
        row("synthetic_validation", "AI_audit", key, docs["synthetic_audit"][key], "synthetic_audit", "/" + key)
    for family, count in quality["family_value_kind_counts"].items():
        row("synthetic_family", family, "groups", count, "synthetic_quality", "/family_value_kind_counts/" + family)

    names = {"vaisala_wrm200": "Vaisala / WRM200", "leonardo_meteor735c": "Leonardo / METEOR 735C",
             "furuno_wr2120": "FURUNO / WR2120", "jrc_jma1030": "JRC / JMA-1032、1034"}
    source_rows = []
    for i, item in enumerate(index["sources"]):
        sid = item["source_id"]
        source_rows.append([names[sid], item["page_count"], source["question_counts_by_source"][sid]])
        for key in ("page_count", "title", "url", "pdf_sha256", "historical_entity_or_family_seen"):
            row("source_metadata", sid, key, item[key], "source_index", f"/sources/{i}/{key}", unit="count" if key == "page_count" else "metadata")
        row("source_metadata", sid, "questions", source["question_counts_by_source"][sid], "sources", "/question_counts_by_source/" + sid)
    table("DataSourceTable", ["制造商 / 型号", "完整页数", "中文题数"], source_rows, "lrr")
    for key in ("source_count", "family_count", "entity_count", "chunk_count", "record_count", "question_count", "qualified_question_count"):
        row("source_packet", "all", key, source[key], "sources", "/" + key)
    types = {"scalar": "标量", "range": "连续范围", "options": "离散选项", "text": "文本"}
    type_rows = []
    for key in ("scalar", "range", "options", "text"):
        count = source["record_counts_by_value_kind"][key]
        row("source_reading_type", key, "records", count, "sources", "/record_counts_by_value_kind/" + key)
        type_rows.append([types[key], count])
    table("DataValueTypeTable", ["新来源读法类型", "记录数"], type_rows, "lr")
    for key in ("source_span_checks", "canonical_programs", "exact_targets", "authored_semantic_claims_match_readings", "empty_lookup_checks"):
        row("source_validation", "CPU", key, source["validation"][key], "sources", "/validation/" + key)
    for key in ("record_readings_passed", "semantic_reference_questions_reviewed", "unknown_records"):
        row("source_validation", "AI_audit", key, audit["facts_review"][key], "source_audit", "/facts_review/" + key)
    quality_rows = [["旧候选精确来源绑定", workflow["exact_source_bindings"], "位置一致，不证明设备真值"],
                    ["旧候选人工独立事实准入", workflow["review_decisions"]["independent_fact_eligible"], "尚未形成独立人工金标"],
                    ["合成参考程序检查", quality["successful_program_validations"], "虚构组的接口一致性"],
                    ["新来源精确片段检查", source["validation"]["source_span_checks"], "固定页文本中的精确位置"],
                    ["新来源语义参考 AI 审阅", audit["facts_review"]["semantic_reference_questions_reviewed"], "来源读法，不是人工评价"],
                    ["新来源参考程序检查", source["validation"]["canonical_programs"], "语义声明之后的程序核对"]]
    table("DataQualityTable", ["检查项", "数量", "能够支持的结论"], quality_rows, "lrl")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    outputs = {"data_results.csv": stream.getvalue(), "data_tables.tex": "\n".join(tex) + "\n"}
    authored = ("03_radar_data_construction.tex", "data_review.tex", "data_export.py")
    manifest = {
        "version": "thesis_data_evidence_v1", "evidence_cutoff": "2026-10-05",
        "scope": "Public inventory, construction and audit metadata plus implementation hashes; no underlying raw sources or private QA opened; no recursive manifest reads.",
        "sources": {path: {"sha256": hashes[path], "role": key} for key, path in FILES.items()},
        "implementation_and_schema_sha256": {path: hashes[path] for path in IMPLEMENTATION},
        "outputs_sha256": {**{name: hashlib.sha256(body.encode()).hexdigest() for name, body in outputs.items()},
                           **{name: sha(OUT / name) for name in authored}},
        "rows": len(rows),
        "validation": {"aggregate_arithmetic": True, "checked_public_binding_edges": checked_bindings,
                       "historical_profile_and_inventory_share_entity_hash": True,
                       "private_data_revalidated": False, "new_training_or_inference": False,
                       "pdf_compilation": "not_performed; xelatex unavailable in authoring environment"},
        "limits": ["Historical asset counts are not verified radar facts or radar model counts",
                   "Anchor isolation is not full entity or source-family isolation",
                   "Old twelve readings and bilingual rewrites remain exposed development material",
                   "Synthetic groups share ontology and short program structures and have no physical realism",
                   "AI multi-agent review is not human gold or verified different-model independence",
                   "Four related source groups and 24 questions are an exploratory sample, not final domain coverage",
                   "Source snapshots and query records do not implement universal unit conversion, component ontology, version reasoning or graph traversal",
                   "Required runtime fields are narrower than the annotation schema; raw units and variant fields are not universally queryable",
                   "Record counts can include alternative encodings of the same source configuration",
                   "CPU source and program checks do not establish equipment truth or natural-language answer accuracy"]}
    outputs["data_evidence_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    for name, body in outputs.items():
        target = OUT / name
        if args.check:
            require(target.is_file() and target.read_text() == body, f"Stale data export: {target.relative_to(ROOT)}")
        else:
            target.write_text(body)
    print(json.dumps({"status": "checked" if args.check else "exported", "outputs": list(outputs)}))


if __name__ == "__main__":
    main()
