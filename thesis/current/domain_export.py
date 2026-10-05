#!/usr/bin/env python3
"""Rebuild domain-chapter tables from an explicit public-metadata allowlist.

No private data, QA, predictions, model weights, or training code is opened.
Nested input manifests are NOT recursively followed. --check is read-only.
Replay and weight integrity are reported from bound execution attestations;
this exporter does not repeat remote tensor checks or semantic adjudication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = "results/radar_domain/source_eval_v1/"
FILES = {
    "semantic": SOURCE + "ai_semantic_summary.json",
    "mechanical": SOURCE + "mechanical_summary.json",
    "execution": SOURCE + "execution_summary.json",
    "completed": SOURCE + "pipeline_completed.json",
    "protocol": SOURCE + "protocol.json",
    "decision": SOURCE + "stage_decision.json",
    "failures": SOURCE + "failure_diagnosis.json",
    "source_manifest": "artifacts/thesis_direction_review/radar_sources_v1/manifest.json",
    "source_index": "artifacts/thesis_direction_review/radar_sources_v1/source_index.json",
    "synthetic_manifest": "artifacts/thesis_direction_review/radar_interface_synthetic_v1/manifest.json",
    "interface": "results/radar_domain/interface_adapt_v1/review.json",
    "interface_protocol": "results/radar_domain/interface_adapt_v1/protocol.json",
    "lookup": "results/radar_domain/lookup_probe_v1/stage_decision.json",
}
ARMS = ("RAG", "P", "A1", "C1", "A2", "C2")
LABELS = {"RAG": "RAG", "P": "P", "A1": "$A_1$", "C1": "$C_1$", "A2": "$A_2$", "C2": "$C_2$"}
SOURCE_LABELS = {
    "vaisala_wrm200": "Vaisala / WRM200",
    "leonardo_meteor735c": "Leonardo / METEOR 735C",
    "furuno_wr2120": "FURUNO / WR2120",
    "jrc_jma1030": "JRC / JMA-1032、1034",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), f"Aggregate mismatch: {a} != {b}")


def build():
    docs = {key: json.loads((ROOT / path).read_text()) for key, path in FILES.items()}
    hashes = {path: digest(ROOT / path) for path in FILES.values()}
    # Verify only edges inside this allowlist. Never open a nested private path.
    binding_edges = 0
    for doc in docs.values():
        for field in ("inputs_sha256", "input_files_sha256", "files_sha256", "evidence_sha256"):
            for path, expected in doc.get(field, {}).items():
                if path in hashes:
                    require(hashes[path] == expected, f"Public source binding changed: {path}")
                    binding_edges += 1
    sem, mech, exe, done, protocol = (docs[x] for x in ("semantic", "mechanical", "execution", "completed", "protocol"))
    data, sources, adapt = (docs[x] for x in ("source_manifest", "source_index", "interface"))
    require(sem["status"] == exe["status"] == done["status"] == "completed", "Incomplete source evaluation")
    require(exe["jobs_completed"] == done["jobs_returned"] == 11 and exe["jobs_failed"] == 0, "Expected all eleven jobs")
    require(done["eligible_for_analysis"] and done["final_error"] is None, "Final integrity gate failed")
    require(all(exe["checks"].values()), "Execution attestation failed")
    require(exe["job_counts_by_stage"] == {"select": 5, "answer": 6}, "Missing stage jobs")
    for obj in (exe, done, docs["decision"]):
        require(obj["protocol_sha256"] == hashes[FILES["protocol"]], "Protocol hash changed")
    for entry in protocol["base_weights"]["files"]:
        final = done["base_files_final"][protocol["model"] + "/" + entry["file"]]
        require(final["sha256"] == entry["sha256"] and final["bytes"] == entry["bytes"], "Final base attestation differs")
    require(mech["selection_replay"] == {"rows": 120, "mismatches": 0}, "Selector replay failed")
    require(mech["answer_input_and_parser_replay"] == {"rows": 144, "mismatches": 0}, "Answer input replay failed")
    require(mech["all_outputs_verified_before_reference_access"], "Reference access ordering not verified")
    require(not sem["human_gold"] and not data["human_gold"], "Unexpected human-label claim")
    require(sem["reviewer_arm_labels_seen"] == [False, False], "Review masking changed")
    require(sem["arm_question_outputs"] == 144 and set(sem["arms"]) == set(ARMS), "Incomplete six-arm semantic report")
    require(protocol["arms"] == list(ARMS) and protocol["question_count"] == data["question_count"] == 24, "Question scope changed")
    require(len(sources["sources"]) == data["source_count"] == 4, "Source count mismatch")
    require(sum(s["page_count"] for s in sources["sources"]) == data["chunk_count"], "Page count mismatch")
    require(sum(data["record_counts_by_value_kind"].values()) == data["record_count"], "Record count mismatch")
    require(sum(data["question_counts_by_source"].values()) == 24, "Question source partition mismatch")
    require(adapt["status"] == "passed" and not adapt["blocking_findings"], "Interface review did not pass")
    require(adapt["verification"]["new_cpu_replay_mismatches"] == 0, "Interface replay mismatch")
    require(adapt["verification"]["A_C_actual_sample_order_and_budget_match"], "Paired continuation mismatch")
    require(docs["lookup"]["cpu_replay"]["mismatches"] == 0, "Lookup replay mismatch")

    rows, tex = [], ["% Generated by domain_export.py from public aggregate metadata; no private QA."]

    def row(table, label, metric, value, unit, source, pointer, derivation="copied"):
        rows.append(dict(table=table, label=label, metric=metric, value=value, unit=unit,
                         source=FILES[source], json_pointer=pointer, derivation=derivation))

    def table(command, headers, records, columns):
        tex.extend(["\\newcommand{\\" + command + "}{%", "\\begin{tabular}{" + columns + "}",
                    "\\toprule", " & ".join(headers) + r" \\", "\\midrule"])
        tex.extend(" & ".join(map(str, record)) + r" \\" for record in records)
        tex.extend(["\\bottomrule", "\\end{tabular}%", "}"])

    source_rows = []
    for i, source in enumerate(sources["sources"]):
        sid = source["source_id"]
        for metric in ("title", "url", "pdf_sha256", "page_count", "historical_entity_or_family_seen"):
            row("source_metadata", sid, metric, source[metric], "metadata", "source_index", f"/sources/{i}/{metric}")
        count = data["question_counts_by_source"][sid]
        row("source_metadata", sid, "questions", count, "count", "source_manifest", f"/question_counts_by_source/{sid}")
        source_rows.append([SOURCE_LABELS[sid], source["page_count"], count,
                            "曾出现" if source["historical_entity_or_family_seen"] else "未检出"])
    table("DomainSourceTable", ["制造商 / 型号", "原文页", "题数", "历史型号或家族"], source_rows, "lrrl")
    for metric in ("source_count", "family_count", "entity_count", "chunk_count", "record_count", "question_count", "qualified_question_count"):
        row("source_dataset", "all", metric, data[metric], "count", "source_manifest", "/" + metric)
    for kind, value in data["record_counts_by_value_kind"].items():
        row("source_dataset", kind, "records", value, "count", "source_manifest", "/record_counts_by_value_kind/" + kind)
    for split, counts in docs["synthetic_manifest"]["split_counts"].items():
        for metric, value in counts.items():
            row("synthetic_split", split, metric, value, "count", "synthetic_manifest", f"/split_counts/{split}/{metric}")

    transfer = []
    for label in ARMS[1:]:
        previous = []
        for language in ("zh", "en"):
            lookup = docs["lookup"]["counts"][f"{label}_{language}"]
            require(lookup["questions"] == 12, "Lookup denominator changed")
            for metric, value in lookup.items():
                row("direct_transfer", f"{label}_{language}", metric, value, "count", "lookup", f"/counts/{label}_{language}/{metric}")
            previous.append(lookup["exact"])
        r = adapt["results"][label]
        for metric, value in r.items():
            row("common_adaptation", label, metric, value, "count", "interface", f"/results/{label}/{metric}")
        transfer.append([LABELS[label], f"{previous[0]}/12", f"{previous[1]}/12",
                         f"{r['synthetic_before_exact_out_of_40']}/40", f"{r['synthetic_after_exact_out_of_40']}/40",
                         f"{r['seen_radar_after_zh_exact_out_of_12']}/12", f"{r['seen_radar_after_en_exact_out_of_12']}/12"])
        training = adapt["training"][label]
        cache = docs["interface_protocol"]["caches"]["program" if label == "P" else "agent"]
        epochs = docs["interface_protocol"]["config"]["epochs"]
        close(training["actual_supervised_tokens"], cache["supervised_tokens"] * epochs)
        close(training["actual_input_tokens"], cache["input_tokens"] * epochs)
        require(training["starting_adapter_sha256"] != training["final_adapter_sha256"], "Adapter did not change")
        for metric in ("actual_supervised_tokens", "actual_input_tokens", "sampled_examples", "microbatches", "global_step", "seed", "train_loss"):
            row("adaptation_training", label, metric, training[metric], "count" if metric != "train_loss" else "loss", "interface", f"/training/{label}/{metric}")
    for a, c in (("A1", "C1"), ("A2", "C2")):
        for key in ("sampled_order_sha256", "actual_supervised_tokens", "actual_input_tokens", "sampled_examples", "global_step", "seed"):
            require(adapt["training"][a][key] == adapt["training"][c][key], f"Adaptation pair differs: {key}")
    table("DomainTransferTable", ["模型", "旧题前中", "旧题前英", "合成前", "合成后", "旧题后中", "旧题后英"], transfer, "lrrrrrr")
    for metric, value in docs["interface_protocol"]["config"].items():
        row("adaptation_setup", "all", metric, value, "configuration", "interface_protocol", "/config/" + metric)

    result_rows, support_rows, cost_rows, error_rows, group_rows = [], [], [], [], []
    job_costs = Counter()
    for i, job in enumerate(exe["jobs"]):
        require(job["status"] == "completed" and job["returncode"] == 0, "Unsuccessful job")
        close(job["gpu_hours"], job["gpu_seconds"] / 3600)
        job_costs[job["label"]] += job["gpu_hours"]
        row("source_job_cost", job["label"] + "_" + job["stage"], "gpu_hours", job["gpu_hours"], "GPU-hour", "execution", f"/jobs/{i}/gpu_hours")
    for arm in ARMS:
        report, run = sem["arms"][arm], mech["runs"][arm]
        judged, counts, tokens = report["adjudicated"], report["mechanical"], report["tokens"]
        require(judged["questions"] == run["questions"] == 24, "Arm denominator mismatch")
        require(judged["qualified_questions"] == data["qualified_question_count"] == 8, "Condition subset denominator mismatch")
        require(counts == run["counts"], "Semantic and mechanical copies differ")
        require(tokens["total"] == tokens["selection"] + tokens["answer"], "Token sum mismatch")
        for stage, runtime_key in (("answer", "answer_runtime"), ("selection", "selection_runtime")):
            rt = run[runtime_key]
            if rt is None:
                require(arm == "RAG" and stage == "selection" and tokens[stage] == 0, "Missing model runtime")
                continue
            require(rt["status"] == "completed" and rt["protocol_sha256"] == hashes[FILES["protocol"]], "Runtime binding differs")
            totals = rt["totals"]
            close(totals["total_tokens"], totals["input_tokens"] + totals["generated_tokens"])
            close(tokens[stage], totals["total_tokens"])
            if stage == "answer":
                require(rt["model"] == protocol["model"] and rt["adapter"] is None, "Common generator differs")
                require(rt["config"] == protocol["answer_inference"], "Common answer inference differs")
            else:
                require(rt["model"] == protocol["models"][arm], "Selector adapter binding differs")
                require(rt["config"] == protocol["inference"], "Selector inference differs")
            for metric in ("input_tokens", "generated_tokens", "total_tokens"):
                row("source_runtime_tokens", arm + "_" + stage, metric, totals[metric], "token", "mechanical", f"/runs/{arm}/{runtime_key}/totals/{metric}")
        for key, value in judged.items():
            row("source_semantic", arm, key, value, "count", "semantic", f"/arms/{arm}/adjudicated/{key}")
        for key, value in counts.items():
            row("source_mechanical", arm, key, value, "count", "mechanical", f"/runs/{arm}/counts/{key}")
        for key, value in tokens.items():
            row("source_tokens", arm, key, value, "token", "semantic", f"/arms/{arm}/tokens/{key}")
        row("source_semantic", arm, "answer_accuracy_percent", 100 * judged["answer_correct"] / judged["questions"],
            "percent", "semantic", f"/arms/{arm}/adjudicated", "100 * answer_correct / questions")
        row("source_cost", arm, "gpu_hours", job_costs[arm], "GPU-hour", "execution", "/jobs",
            "sum(gpu_hours) where job.label equals this arm; includes selection and answer jobs")
        errors = report["incorrect_answer_partition"]
        require(sum(errors.values()) + judged["answer_correct"] == 24, "Incorrect-answer partition differs")
        for key, value in errors.items():
            row("source_failure_stage", arm, key, value, "count", "semantic", f"/arms/{arm}/incorrect_answer_partition/{key}")
        counts_record = counts.get("exact_record_selection", "---")
        result_rows.append([LABELS[arm], counts_record, counts["all_required_chunks_available"], judged["answer_correct"],
                            f"{100*judged['answer_correct']/24:.2f}", f"{judged['qualified_conditions_preserved']}/8"])
        support_rows.append([LABELS[arm], counts["json_contract_valid"], judged["answer_supported"], judged["citations_support_answer"], judged["correct_supported_and_cited"]])
        cost_rows.append([LABELS[arm], f"{tokens['selection']:,}", f"{tokens['answer']:,}", f"{tokens['total']:,}", f"{job_costs[arm]:.5f}"])
        error_rows.append([LABELS[arm], errors["missing_required_evidence_page"], errors["required_pages_present_but_answer_incorrect"]])
        source_values = []
        for sid in SOURCE_LABELS:
            group = report["by_source"][sid]
            require(group["questions"] == data["question_counts_by_source"][sid] == 6, "Source group denominator mismatch")
            row("source_groups", arm + "_" + sid, "answer_correct", group["answer_correct"], "count", "semantic", f"/arms/{arm}/by_source/{sid}/answer_correct")
            source_values.append(group["answer_correct"])
        require(sum(source_values) == judged["answer_correct"], "Source group count mismatch")
        group_rows.append([LABELS[arm], *source_values])
    table("DomainAnswerTable", ["路线", "记录精确", "支持页齐全", "正文正确", r"正确率（\%）", "条件保留"], result_rows, "lrrrrr")
    table("DomainSupportTable", ["路线", "JSON 有效", "正文有据", "引用支撑", "正确且有据有引"], support_rows, "lrrrr")
    table("DomainCostTable", ["路线", "选择 token", "回答 token", "合计 token", "GPU 小时"], cost_rows, "lrrrr")
    table("DomainErrorStageTable", ["路线", "缺支持页且正文错误", "有支持页仍正文错误"], error_rows, "lrr")
    table("DomainSourceGroupTable", ["路线", "Vaisala", "Leonardo", "FURUNO", "JRC"], group_rows, "lrrrr")
    for name, pair in sem["paired_natural_answer_differences"].items():
        c, a = name.split("_minus_")
        require(pair["net_questions"] == sem["arms"][c]["adjudicated"]["answer_correct"] - sem["arms"][a]["adjudicated"]["answer_correct"], "Paired difference mismatch")
        row("source_pair", name, "net_correct_questions", pair["net_questions"], "count", "semantic", f"/paired_natural_answer_differences/{name}/net_questions")
    for metric in ("unique_answers", "arm_question_outputs", "disagreement_unique_answers"):
        row("ai_review", "all", metric, sem[metric], "count", "semantic", "/" + metric)
    for metric, count in sem["disagreements_by_field"].items():
        row("ai_review", "disagreements", metric, count, "count", "semantic", "/disagreements_by_field/" + metric)
    # The public diagnosis contains anonymous IDs and tags, but no QA text.
    # Export aggregate tags only; never export the IDs or individual judgments.
    failures = docs["failures"]
    row("source_failure_tags", "four_agents", "common_incorrect_count", len(failures["all_four_agent_arms_common_incorrect_question_ids"]), "count", "failures", "/all_four_agent_arms_common_incorrect_question_ids", "len; IDs not exported")
    tags = Counter(r["observed_error_type"] for r in failures["records"])
    require(sum(tags.values()) == sum(24 - sem["arms"][a]["adjudicated"]["answer_correct"] for a in ARMS), "Failure tag count mismatch")
    for tag, count in sorted(tags.items()):
        row("source_failure_tags", "all_outputs", tag, count, "count", "failures", "/records", "post-run count by observed_error_type; correlated arm-question outputs")
    total_tokens = sum(sem["arms"][a]["tokens"]["total"] for a in ARMS)
    row("source_cost", "all", "total_tokens", total_tokens, "token", "semantic", "/arms", "sum(tokens.total) over six arms")
    close(sum(job_costs.values()), exe["gpu_hours_this_run"])
    close(sum(exe["gpu_hours_by_stage"].values()), exe["gpu_hours_this_run"])
    close(exe["gpu_hours_after"] - exe["gpu_hours_before"], exe["gpu_hours_this_run"])
    for metric in ("jobs_completed", "pipeline_wall_seconds", "gpu_hours_this_run", "gpu_hours_after", "maximum_additional_gpu_hours", "cumulative_project_budget_gpu_hours"):
        row("source_cost", "all", metric, exe[metric], "seconds" if metric.endswith("seconds") else "count" if metric == "jobs_completed" else "GPU-hour", "execution", "/" + metric)
    for metric in ("charged_gpu_hours", "supervised_training_tokens_per_agent", "supervised_training_tokens_P", "same_question_exposures_per_model"):
        row("adaptation_cost", "all", metric, adapt["budget"][metric], "GPU-hour" if metric.endswith("hours") else "count", "interface", "/budget/" + metric)
    row("lookup_cost", "all", "probe_gpu_hours", docs["lookup"]["probe_gpu_hours"], "GPU-hour", "lookup", "/probe_gpu_hours")
    tex.extend(["\\newcommand{\\DomainTotalTokens}{" + f"{total_tokens:,}" + "}",
                "\\newcommand{\\DomainGPUHours}{" + f"{exe['gpu_hours_this_run']:.5f}" + "}"])

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    outputs = {"domain_results.csv": stream.getvalue(), "domain_tables.tex": "\n".join(tex) + "\n"}
    authored = ("06_radar_application.tex", "domain_review.tex", "domain_export.py")
    manifest = {
        "version": "thesis_domain_evidence_v1", "evidence_cutoff": "2026-10-05",
        "scope": "Public aggregate metadata only. No QA, source fulltext, raw predictions, weights or training opened; nested manifests are not recursively followed.",
        "sources": {path: {"sha256": hashes[path], "role": key} for key, path in FILES.items()},
        "outputs_sha256": {**{name: hashlib.sha256(body.encode()).hexdigest() for name, body in outputs.items()}, **{name: digest(OUT / name) for name in authored}},
        "table_rows": len(rows),
        "validation": {"verified_public_binding_edges": binding_edges, "aggregate_arithmetic": True,
                       "completed_jobs": 11, "selector_replay_rows_reported": 120, "answer_input_replay_rows_reported": 144,
                       "replay_mismatches_reported": 0, "shared_answer_model_and_configuration": True,
                       "paired_adaptation_order_and_tokens": True, "final_base_hash_attestation_matches_protocol": True,
                       "local_raw_or_tensor_reverification": False,
                       "semantic_judgments": "Copied from two masked AI reviews and an arm-aware source-grounded adjudication; not rejudged by exporter",
                       "pdf_compilation": "not_performed; xelatex unavailable in authoring environment"},
        "claim_limits": ["AI authorship/review is not human gold or different-model-family consensus",
                         "24 questions from four correlated sources; no significance claim",
                         "new source snapshots are not certified historical equipment-family isolation",
                         "common interface adaptation is effective on bounded lookup checks; no C domain advantage demonstrated",
                         "original public benchmark improvements belong to original adapters; adapted public retention unmeasured",
                         "curated KB and raw-page RAG have unequal preparation costs and training histories",
                         "strict claim field mismatch is not natural-answer error",
                         "page coverage can recover a wrong record selection; selection and answer metrics are separate",
                         "token counts include complete repeated prefills; not FLOPs or total preparation cost",
                         "pre-freeze retrieval coverage was observed and disclosed; no score-driven retuning",
                         "post-run error labels are descriptive; no causal mechanism established",
                         "no additional domain training planned; application evidence and the thesis remain preliminary"],
    }
    outputs["domain_evidence_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, body in build().items():
        path = OUT / name
        if args.check:
            require(path.is_file() and path.read_text() == body, f"Stale domain export: {path.relative_to(ROOT)}")
        else:
            path.write_text(body)
    print(json.dumps({"status": "checked" if args.check else "exported", "scope": "domain public aggregates; no private QA"}))


if __name__ == "__main__":
    main()
