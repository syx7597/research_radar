"""Version the twelve AI-reviewed source readings for a development-only probe.

This is a source-reading dataset, not equipment truth or an independent test set.
Inference questions and AI reference specifications are separate artifacts.  The
original packet, blank-answer questions, and review decisions are never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from scripts import audit_radar_review_workflow as workflow
from scripts import summarize_radar_ai_review as ai_review

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/thesis_direction_review"
DEST = BASE / "radar_development_v2"
VERSION = "radar_ai_source_development_v2"

# These are review-derived development rubrics, not human answers or held-out
# labels.  They are deliberately absent from questions.jsonl and runtime input.
REFERENCE_SPECS = {
    "01": (
        "该快照的 Type 栏描述 MIM-104 Patriot 整套系统；历史入库标签为 AN/MPQ-65，不能据此把系统类型作为该雷达型号的类型。",
        ["识别来源主体为 Patriot 系统", "区分来源主体与历史雷达检索锚点", "说明不能直接支持 AN/MPQ-65 的型号类型"],
        ["把系统类型断言为雷达类型"]),
    "02": (
        "绑定系统页面的 Since 1981 是服役事件表述。它属于来源中的 Patriot 系统陈述，不能仅凭该字段确定 AN/MPQ-65 雷达的服役年份。",
        ["保留来源年份 1981 和 in_service 事件", "限定主体为系统", "拒绝据此填写雷达服役年份"],
        ["把 1981 确认为 AN/MPQ-65 服役年份", "补造替代年份"]),
    "03": (
        "绑定系统页面写有 initial operational capacity 1984，即初始作战能力事件；应与来源中的服役事件分别记录。该快照没有把 1984 直接绑定为 AN/MPQ-65 的型号事件。",
        ["保留 1984 与 initial_operational_capacity", "与 in_service 事件分开", "保留系统主体范围"],
        ["将初始能力与服役、交付或部署事件互换", "把年份直接归给雷达型号"]),
    "04": (
        "Frequency 快照写有 UHF-Band 0.3–3.0 GHz；开发记录保留两个端点。它是否为该型号实际工作范围、还是频段解释，现有绑定资料不能确定，不能提升为已核实型号参数。",
        ["保留 0.3 和 3.0 两端及 GHz", "保留来源范围", "明确型号实际参数与频段解释未判定"],
        ["确认整个范围为该型号实际工作频率", "只保留单个端点"]),
    "05": (
        "绑定快照中 Illuminator 记录为 10.25–10.5 GHz；应保留两个端点和原文部件标签 Illuminator，独立于 Tracking Radar 记录。",
        ["10.25 和 10.5", "GHz", "Illuminator 标签与独立记录"],
        ["把两个部件频率合并为一个连续范围", "补造型号版本或配置"]),
    "06": (
        "绑定快照中 Tracking Radar 记录为 5.45–5.825 GHz；应保留两个端点和原文部件标签 Tracking Radar，独立于 Illuminator 记录。",
        ["5.45 和 5.825", "GHz", "Tracking Radar 标签与独立记录"],
        ["把两个部件频率合并为一个连续范围", "补造型号版本或配置"]),
    "07": (
        "绑定快照把 surface 标签下的 PRF 写为标量 4100 pps。surface 仅是原文标签，其进一步对应何种模式或对象尚未核实。",
        ["标量 4100", "pps", "保留 surface 字面标签且不扩展其含义"],
        ["把 surface 的具体模式或目标含义当作已核实", "与 air 数值混合"]),
    "08": (
        "绑定快照中 air 标签下的 PRF 为 9600–16700 pps，应保留这两个端点、单位及 air 原文标签，与 surface 标量分开。",
        ["9600 和 16700", "pps", "保留 air 标签及独立范围"],
        ["只保留一个端点", "把 surface 的 4100 当作 air 数值"]),
    "09": (
        "绑定 Frequency 快照是频率量，记录为 9.0–9.2 GHz；不能按质量量纲或旧图谱的 kg 保存。这只是该快照读法，尚不能消除其他资料的精度或版本差异。",
        ["频率量纲", "9.0 和 9.2", "GHz", "限定快照读法"],
        ["保留旧图谱 kg 量纲", "直接用未绑定的手册转录值替换快照", "宣称已证明两个来源矛盾"]),
    "10": (
        "绑定 Pulsewidth 快照记录 0.2 microseconds，是时间量；标准单位记为 us（微秒）。该快照单值不证明设备只有这一脉宽或无条件适用。",
        ["时间量纲", "microseconds 对应 us/微秒", "快照值为 0.2 且不宣称唯一"],
        ["保存为长度或 km", "将外部离散选项改写成连续区间", "断言 0.2 是唯一无条件值"]),
    "11": (
        "Canceled 1963 表述取消事件，应记录为 lifecycle_event / cancellation，年份 1963；它不能直接回答服役年份。",
        ["取消事件 cancellation", "1963", "与服役事件分开"],
        ["把 1963 作为服役年份", "沿用 service_entry 表示取消事件"]),
    "12": (
        "仅依据绑定快照 Power 字段，没有可填写的功率数值，数值和单位均留空。该字段写有 Classified；这只是来源声称和该字段的缺值状态，不能扩展为所有来源都未知或已证实的保密状态。",
        ["没有数值且单位留空", "unknown_scope 限定 bound_source_field", "Classified 仅为来源声称"],
        ["用 0 代替未知", "推测功率或用途能力", "声称所有资料均不存在数值", "将 Classified 提升为独立核实状态"]),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def source_reading(item: dict, fact: dict) -> dict:
    """Apply the adjudicated changes while retaining the original fields."""
    fields = item["candidate_fields"]
    suffix = item["fact_id"].rsplit("-", 1)[1]
    record = {key: (None if value == "" else value) for key, value in fields.items()}
    record.update(
        fact_id=item["fact_id"],
        source_context_id=fact["doc_id"],
        source_subject="MIM-104 Patriot system" if suffix in {"01", "02", "03"} else fields["entity_name"],
        original_entity_attribution_usable=suffix not in {"01", "02", "03"},
        review_status="AI_only_development",
        reviewer_kind="AI",
        equipment_truth_verified=False,
        human_verified=False,
        independent_gold=False,
        usage_scope="bound_source_snapshot_interpretation_only",
        scope_notes_zh=item["restrictions_zh"] + " " + item["proposed_revision_zh"],
        unknown_scope=None,
        unknown_reason=None,
        source_claimed_status=None,
        original_candidate_fields=dict(fields),
        original_fact_record_sha256=item["fact_record_sha256"],
        legacy_value_raw=item["legacy_value_raw"],
        conditions=json.loads(fields["conditions_json"]),
        citation={key: item["source_binding"][key]
                  for key in ("source_uri", "source_sha256", "source_file", "evidence_text")},
    )
    record["citation"]["locator"] = json.loads(item["source_binding"]["locator"])
    if suffix == "11":
        record.update(attribute="lifecycle_event", event_type="cancellation")
    if suffix == "12":
        record.update(unknown_scope="bound_source_field",
                      unknown_reason="no_numeric_value_in_bound_field",
                      source_claimed_status="Classified")
    if suffix == "09":
        record["scope_notes_zh"] += " 第三方手册转录出现 9.0–9.16 的端点，尚未核对原扫描；舍入或版本差异未排除，不覆盖本快照，也不判定为已证实冲突。"
    if suffix == "10":
        record["scope_notes_zh"] += " 第三方手册转录出现 0.2 或 0.8 微秒的离散选项，尚未核对原扫描；不覆盖本快照的 0.2，也不能将选项改写为连续区间。"
    return record


def build() -> dict[str, bytes]:
    # Existing audit verifies the original source bytes, pointer and span, not
    # merely the current date's remotely served web page.
    workflow.audit(workflow.DEFAULT_WORKFLOW)
    facts, _ = workflow.load_packet()
    by_id = {fact["fact_id"]: fact for fact in facts}
    review_outputs = ai_review.build()
    for name, data in review_outputs.items():
        if (ai_review.DEST / name).read_bytes() != data:
            raise ValueError(f"AI cross-review artifact drift: {name}")
    readings = json.loads(review_outputs["development_readings.json"])
    if readings["human_verified"] or readings["independent_gold"]:
        raise ValueError("Development readings must not become human or independent gold")
    records = [source_reading(item, by_id[item["fact_id"]]) for item in readings["records"]]
    if len(records) != 12 or len({record["fact_id"] for record in records}) != 12:
        raise ValueError("This version is bound to exactly twelve reviewed readings")

    question_file = workflow.DEFAULT_WORKFLOW / "development_questions.csv"
    old_questions = list(csv.DictReader(io.StringIO(question_file.read_text(encoding="utf-8"))))
    questions, references = [], []
    for row in old_questions:
        if row["answer_json"] or row["answer_status"] or row["split"] != "development_only":
            raise ValueError("Original development questions must remain blank-answer development material")
        support = json.loads(row["support_fact_ids"])
        source_facts = [by_id[fact_id] for fact_id in support]
        contexts = {(fact["doc_id"], fact["source_uri"], fact["entity_name"]) for fact in source_facts}
        if len(contexts) != 1:
            raise ValueError("This initial development version expects one source context per question")
        context_id, source_uri, entity_anchor = next(iter(contexts))
        context = (f"来源上下文：source_context_id={context_id}；source_uri={source_uri}；"
                   f"检索实体锚点={entity_anchor}。该锚点是历史入库标签，来源主体请依据检索记录判断。")
        questions.append({"id": row["qid"], "question": row["question"] + "\n" + context})
        suffix = row["qid"].rsplit("-", 1)[1]
        expected, required, forbidden = REFERENCE_SPECS[suffix]
        references.append({
            "id": row["qid"],
            "original_question": row["question"],
            "expected_fact_ids": support,
            "acceptable_context_fact_ids": (
                ["radar-review-001-03"] if suffix == "02" else
                ["radar-review-001-02"] if suffix == "03" else []),
            "context_justification_zh": (
                "同一来源字段的服役与初始作战能力是两个事件；允许另一事件仅作为对照背景，主目标的严格集合匹配仍单独计分。"
                if suffix in {"02", "03"} else "无预先指定的额外背景记录。"),
            "expected_response_zh": expected,
            "rubric": {"required_semantics_zh": required, "forbidden_overclaims_zh": forbidden},
            "reference_kind": "AI_draft_source_scoped_development_only",
            "human_verified": False,
            "independent_gold": False,
            "reasoning_type": row["reasoning_type"],
            "source_group": row["source_group"],
            "model_family": row["model_family"],
        })
    outputs = {
        "readings.json": encoded({
            "version": VERSION, "split": "development_only", "reviewer_kind": "AI",
            "human_verified": False, "independent_gold": False,
            "equipment_truth_verified": False, "not_an_accepted_KB": True,
            "entity_name_semantics": "Historical source-lookup anchor, not an assertion that all source fields describe that equipment.",
            "attribution_flag_semantics": "Usable only for source-subject attribution; never establishes equipment parameter truth.",
            "missing_value_semantics": "null means not recorded; it never means zero or unconditional applicability.",
            "numeric_representation": "Exact decimal text from reviewed candidates; no floating-point coercion.",
            "records": records}),
        "questions.jsonl": ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions)).encode("utf-8"),
        "references.json": encoded({
            "version": VERSION, "split": "development_only", "reference_kind": "AI_draft",
            "human_verified": False, "independent_gold": False,
            "must_not_be_loaded_by_inference": True,
            "scoring_scope": "Expected-record selection and source-scoped development diagnosis; no independent domain accuracy claim.",
            "records": references}),
    }
    inputs = [BASE / "radar_review_batch/facts.csv", BASE / "radar_review_batch/source_manifest.json",
              question_file, workflow.DEFAULT_WORKFLOW / "review_decisions.csv",
              workflow.DEFAULT_WORKFLOW / "workflow_manifest.json",
              ai_review.DEST / "reviewer_report.json", ai_review.DEST / "root_semantic_review.json",
              ai_review.DEST / "development_readings.json", ai_review.DEST / "summary.json", Path(__file__)]
    outputs["manifest.json"] = encoded({
        "version": VERSION, "record_count": len(records), "question_count": len(questions),
        "split": "permanently_exposed_development_only", "new_training_started": False,
        "human_verified": False, "independent_gold": False, "equipment_truth_verified": False,
        "source_file_sha256": {record["citation"]["source_file"]: record["citation"]["source_sha256"] for record in records},
        "inputs_sha256": {str(item.relative_to(ROOT)): digest(item.read_bytes()) for item in inputs},
        "outputs_sha256": {name: digest(data) for name, data in outputs.items()},
        "inference_input_files": ["readings.json", "questions.jsonl"],
        "excluded_from_inference": ["references.json", "manifest.json"],
        "original_artifacts_modified": False,
        "changes": ["Separate source subject from the historical entity anchor for 01–03.",
                    "Preserve uncertain source ranges and the two source-scoped SPN-35 readings.",
                    "Change SPG-59 cancellation attribute to lifecycle_event.",
                    "Add bound-source-field unknown scope; never replace missing power by zero.",
                    "Add neutral source context uniformly to the twelve original Chinese questions.",
                    "Create separate AI draft references; preserve original blank answers."],
    })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing version bytes without rewriting")
    args = parser.parse_args()
    artifacts = build()
    if args.check:
        for name, data in artifacts.items():
            if (DEST / name).read_bytes() != data:
                raise ValueError(f"Development artifact drift: {name}")
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        existing = [name for name in artifacts if (DEST / name).exists()]
        if existing:
            raise FileExistsError(f"Immutable version already exists; use --check or a new version: {existing}")
        for name, data in artifacts.items():
            with (DEST / name).open("xb") as handle:
                handle.write(data)
    print(json.dumps({"mode": "verified" if args.check else "created", "version": VERSION,
                      "records": 12, "questions": 12, "human_verified": False,
                      "independent_gold": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
