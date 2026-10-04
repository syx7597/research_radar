#!/usr/bin/env python3
"""Render source-bound AI review findings without changing gold or the old packet."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_radar_review_workflow as workflow

DEST = ROOT / "artifacts/thesis_direction_review/radar_ai_cross_review_v1"
LABELS = {
    "revise_or_quarantine": "修订或隔离型号断言",
    "retain_scoped_reading": "保留限定来源的开发读法",
    "retain_reading_with_scope_question": "保留读法，另查精度／选项范围",
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build():
    original_audit = workflow.audit(workflow.DEFAULT_WORKFLOW)
    facts, _ = workflow.load_packet()
    by_id = {fact["fact_id"]: fact for fact in facts}
    report_files = [DEST / "reviewer_report.json", DEST / "root_semantic_review.json"]
    reports = [json.loads(p.read_bytes()) for p in report_files]
    for report in reports:
        if report["reviewer_kind"] != "AI" or report["human_verified"] or report["independent_gold"]:
            raise ValueError("AI reports must not claim human or independent-gold verification")
        records = report.get("records", report.get("facts"))
        if len(records) != len(by_id) or {row["fact_id"] for row in records} != set(by_id):
            raise ValueError("Each review must cover the 12 bound candidate records exactly once")
        for row in records:
            fact = by_id[row["fact_id"]]
            row_hash = row.get("fact_record_sha256", row.get("original_row", {}).get("fact_record_sha256"))
            source_hash = row.get("source_sha256", row.get("source_binding", {}).get("source_sha256_observed"))
            if row_hash != workflow.fact_digest(fact) or source_hash != fact["source_sha256"]:
                raise ValueError("Review is not bound to the current candidate and source")
    root = reports[1]
    child_by_id = {row["fact_id"]: row for row in reports[0]["facts"]}
    counts = dict.fromkeys(LABELS, 0)
    rows = []
    readings = []
    cards = []
    escape = html.escape
    for row in root["records"]:
        fact = by_id[row["fact_id"]]
        action = row["development_disposition"]
        counts[action] += 1
        child = child_by_id[row["fact_id"]]
        reconciliation = "保留子agent的证据边界；主agent裁决仅描述开发用途，不是事实准入。"
        if row["fact_id"].endswith(("09", "10")):
            reconciliation = "子agent的隔离针对型号真值；仍可保留快照读法作开发，禁止提升为无条件型号事实。"
        elif row["fact_id"].endswith("12"):
            reconciliation = "采纳子agent补结构化unknown_scope的建议；保留读法并将未知明确限定到绑定来源字段。"
        record = {"fact_id": row["fact_id"], "entity_name": fact["entity_name"],
                  "subagent_action": child["recommended_action"],
                  "development_disposition": action, "reviewer_kind": "AI",
                  "adjudication_note_zh": reconciliation,
                  "reason_zh": row["reason_zh"], "recommended_revision_zh": row["recommended_revision_zh"],
                  "source_uri": fact["source_uri"], "source_sha256": fact["source_sha256"],
                  "human_verified": "false", "independent_gold": "false"}
        rows.append(record)
        candidate_fields = {key: fact[key] for key in ("entity_id", "entity_name", "variant", "attribute", "event_type", "value_kind", "value_status", "value", "min_value", "max_value", "unit_raw", "unit_std", "condition_status", "condition_raw", "conditions_json")}
        reading = {"fact_id": fact["fact_id"], "review_status": "AI_cross_reviewed_development_only",
                   "fact_record_sha256": row["fact_record_sha256"], "candidate_fields": candidate_fields,
                   "source_binding": {key: fact[key] for key in ("source_uri", "source_file", "source_sha256", "locator", "evidence_text")},
                   "legacy_value_raw": fact["value_raw"],
                   "usage_scope": "bound_source_snapshot_interpretation_only",
                   "equipment_truth_verified": False, "human_verified": False, "independent_gold": False,
                   "development_disposition": action, "restrictions_zh": row["reason_zh"],
                   "proposed_revision_zh": row["recommended_revision_zh"], "adjudication_note_zh": reconciliation}
        if fact["fact_id"].endswith(("01", "02", "03")):
            reading["source_subject_proposed"] = "MIM-104 Patriot system, based on bound source URI and content"
            reading["original_entity_attribution_usable"] = False
        if fact["fact_id"].endswith("11"):
            reading["proposed_field_changes"] = {"attribute": "lifecycle_event", "event_type": "cancellation"}
        if fact["fact_id"].endswith("12"):
            reading["unknown_scope"] = "bound_source_field"
            reading["unknown_reason"] = "no_numeric_value_in_bound_field"
            reading["source_claimed_status"] = fact["evidence_text"]
        readings.append(reading)
        cards.append(f'''<article><h2>{escape(row['fact_id'])} · {escape(fact['entity_name'])}</h2>
<p class="state">主agent开发用途裁决：{escape(LABELS[action])}</p><p>子agent建议：{escape(child['recommended_action'])}。{escape(reconciliation)}</p><p>{escape(row['reason_zh'])}</p>
<p><b>处理：</b>{escape(row['recommended_revision_zh'])}</p>
<p><b>已绑定快照摘录：</b><q>{escape(fact['evidence_text'])}</q></p>
<p><a href="{escape(fact['source_uri'], quote=True)}">原网页</a> · <a href="../../../{escape(fact['source_file'], quote=True)}">本地历史快照</a></p></article>''')
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    sources = "".join(f'<li><a href="{escape(s["url"], quote=True)}">{escape(s["source_id"])}</a>：{escape(s["supports"])}</li>' for s in root["sources"])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>雷达首批 AI 交叉审阅结果</title><style>body{font-family:system-ui,sans-serif;max-width:950px;margin:32px auto;padding:0 18px;line-height:1.7;color:#172b3a;background:#f6f8fa}article,section{background:white;padding:18px;border:1px solid #d5dde4;border-radius:8px;margin:16px 0}h1{font-size:25px}h2{font-size:18px}.state{font-weight:bold;color:#18557b}a{color:#125a95}q{background:#eef3f7}</style>
<h1>首批 12 条：AI 交叉审阅结果</h1><section>
<p>按用户要求，另开未继承会话的新子 agent 对照本地来源审阅，主 agent 再核查证据并裁决开发用途。子 agent 可见候选中的旧备注；主 agent 有项目历史，并接收了子 agent 提供的新来源线索。这不是双盲或不同模型的统计独立验证。</p>
<p><b>不需要用户先判断专业参数才能继续开发。</b>本轮结果用于候选修订与开发材料整理，统一标为 AI 审阅，不能改称人工金标或已核实的装备参数。原始事实、人工决定与最终 QA 答案均未修改。</p>
<p>子agent建议：开发保留4条、隔离5条、修订3条。主agent进一步按开发用途裁决：5条需修订／隔离型号断言；5条可保留限定来源的读法（包括为12补未知范围）；2条保留读法但补查精度或选项范围。两套分布评价对象不同，未把分歧隐藏成一致投票。12条都不是独立人工金标。</p>
<p><a href="reviewer_report.json">新子 agent 完整审阅</a> · <a href="root_semantic_review.json">主 agent 复核</a> · <a href="review_summary.csv">逐条处理表</a></p></section>
''' + "\n".join(cards) + '<section><h2>复核所用补充资料</h2><ul>' + sources + '''</ul>
<p>厂商文章支持频段类别，不支持本候选的完整数值范围；手册页是第三方转录，未核对原扫描，精度差异可能是舍入，单值也可能只是多选项之一。保留这些不确定性，不直接覆盖原事实。</p></section></html>'''
    if counts != {"revise_or_quarantine": 5, "retain_scoped_reading": 5, "retain_reading_with_scope_question": 2}:
        raise ValueError("Update the summary prose after an explicit adjudication change")
    outputs = {"review_summary.csv": table.getvalue().encode(), "index.html": page.encode(),
               "development_readings.json": encoded({"version": "radar_source_readings_ai_review_v1", "split": "development_only", "reviewer_kind": "AI", "independent_gold": False, "human_verified": False, "not_an_accepted_KB": True, "final_question_answers_included": False, "records": readings})}
    summary = {"version": "radar_ai_cross_review_v1", "reviewer_kind": "AI", "reviewed_records": 12,
               "independent_context_reviewer": reports[0]["reviewer_agent"],
               "statistically_independent_or_different_model_claim": False,
               "human_verified": False, "independent_gold": False,
               "development_dispositions": counts, "user_expert_review_required_to_continue_development": False,
               "subagent_summary": reports[0]["summary"],
               "reconciliation": "Child recommendations and root development dispositions are distinct scopes; both are preserved per record, including 09/10 quarantine of truth promotion and 12 structured unknown scope.",
               "next_step": "Version source-scoped development readings and proposed fixes, preserving uncertainty; then test Chinese query adaptation on development material only.",
               "original_packet_modified": False, "original_review_decisions_modified": False,
               "original_question_answers_modified": False, "new_training_started": False,
               "original_workflow_audit": original_audit,
               "inputs_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in report_files + [Path(__file__)]},
               "outputs_sha256": {name: digest(data) for name, data in outputs.items()}}
    outputs["summary.json"] = encoded(summary)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    artifacts = build()
    for name, data in artifacts.items():
        target = DEST / name
        if args.check_only:
            if target.read_bytes() != data:
                raise ValueError(f"Review summary differs: {name}")
        else:
            with target.open("xb") as handle:
                handle.write(data)
    print(json.dumps({"mode": "verified" if args.check_only else "created", "records": 12,
                      "human_verified": False, "independent_gold": False}))
