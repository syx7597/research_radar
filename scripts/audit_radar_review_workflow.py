#!/usr/bin/env python3
"""Prepare and audit the bounded radar review workflow, without accepted gold.

Reads only the existing 12-fact packet and its explicitly declared local sources.
Never reads benchmark splits, calls a model, or fetches a network resource.
--initialize creates pending decision rows and answer-free development questions.
The default audit is read-only; --report explicitly writes its JSON summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_radar_review_packet as packet_builder  # noqa: E402

PACKET = ROOT / "artifacts/thesis_direction_review/radar_review_batch"
DEFAULT_WORKFLOW = ROOT / "artifacts/thesis_direction_review/radar_review_workflow"
CHECK_FIELDS = ("subject_check", "variant_check", "event_check", "value_check",
                "unit_check", "condition_check")
DECISION_HEADERS = ["fact_id", "fact_record_sha256", "source_sha256", "decision",
                    "decision_scope", "reviewer_kind", "reviewer", "reviewed_at", "reason",
                    *CHECK_FIELDS, "independent_source_class", "independent_source_uri",
                    "independent_source_file", "independent_source_sha256",
                    "independent_source_locator", "independent_evidence_text"]
QUESTION_HEADERS = (ROOT / "templates/radar_review/questions.csv").read_text().strip().split(",")

# These questions exercise the annotation workflow. They are already exposed to
# development and are permanently ineligible for an independent evaluation set.
QUESTIONS = [
    ("01", "所给来源快照的Type栏描述雷达型号还是整套系统？该栏能否直接支持AN/MPQ-65的类型归属？", "subject_attribution"),
    ("02", "该来源快照的Since表述对应什么主体和事件？仅凭它能否确定AN/MPQ-65的服役年份？", "event_scope"),
    ("03", "该来源快照的initial operational capacity表述对应哪一事件？与服役事件应否分开记录？", "event_scope"),
    ("04", "AN/APY-9来源快照的Frequency栏给出什么区间？该区间是型号参数还是频段解释能否确定？", "range_scope"),
    ("05", "AN/SPG-51来源快照中标为Illuminator的频率记录应保留什么范围、单位与部件标签？", "component_range"),
    ("06", "AN/SPG-51来源快照中标为Tracking Radar的频率记录应保留什么范围、单位与部件标签？", "component_range"),
    ("07", "AN/SPG-51来源快照中surface标签下的PRF以什么数值形式记录？该标签的进一步含义是否已明确？", "condition_scope"),
    ("08", "AN/SPG-51来源快照中air标签下的PRF应保留哪两个端点和什么单位？", "condition_range"),
    ("09", "AN/SPN-35来源快照的Frequency栏应按什么量纲和区间记录？", "unit_range"),
    ("10", "AN/SPN-35来源快照的Pulsewidth栏记录的是什么量纲和单位？", "unit_dimension"),
    ("11", "AN/SPG-59来源快照中Canceled表述的是何种事件？它能否直接回答服役年份？", "event_type"),
    ("12", "仅依据EL/M-2080来源快照的Power栏，是否存在可填写的功率数值？回答范围应如何限定？", "source_scoped_unknown"),
]
FAMILIES = {"AN/MPQ-65": "Patriot_system_and_components", "AN/APY-9": "AN_APY_9_family",
            "AN/SPG-51": "AN_SPG_51_family", "AN/SPN-35": "AN_SPN_35_family",
            "AN/SPG-59": "AN_SPG_59_family", "EL/M-2080": "EL_M_2080_family"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def fact_digest(row: dict) -> str:
    return digest(encoded(row))


def csv_bytes(headers: list[str], rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def read_csv(path: Path, headers: list[str]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != headers:
            raise ValueError(f"Unexpected CSV columns: {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Ragged CSV rows: {path.name}")
    return rows


def local_path(relative: str) -> Path:
    path = Path(relative)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(ROOT):
        raise ValueError("Source path must stay inside the workspace and be relative")
    if not resolved.is_file():
        raise ValueError(f"Missing explicitly bound local source: {relative}")
    return resolved


def load_packet() -> tuple[list[dict], dict]:
    packet = {name: (PACKET / name).read_bytes()
              for name in ("facts.csv", "README.md", "source_manifest.json")}
    # Uses the existing source hash, pointer, excerpt, range and condition checks.
    # Does not rebuild or overwrite the frozen packet.
    packet_builder.validate_packet(packet)
    rows = read_csv(PACKET / "facts.csv", packet_builder.HEADERS)
    return rows, json.loads(packet["source_manifest.json"])


def initial_artifacts(facts: list[dict]) -> dict[str, bytes]:
    decisions = []
    fact_map = {row["fact_id"]: row for row in facts}
    packet_hash = digest((PACKET / "source_manifest.json").read_bytes())
    for row in facts:
        decision = dict.fromkeys(DECISION_HEADERS, "")
        decision.update(fact_id=row["fact_id"], fact_record_sha256=fact_digest(row),
                        source_sha256=row["source_sha256"], decision="pending")
        decisions.append(decision)
    questions = []
    for suffix, question, reasoning in QUESTIONS:
        fact_id = "radar-review-001-" + suffix
        fact = fact_map[fact_id]
        questions.append(dict(qid="radar-workflow-dev-" + suffix, question=question,
                              answer_json="", answer_status="", support_fact_ids=json.dumps([fact_id]),
                              knowledge_version="unaccepted-packet-sha256:" + packet_hash,
                              reasoning_type=reasoning, source_group=fact["doc_id"],
                              model_family=FAMILIES[fact["entity_name"]], split="development_only",
                              annotator="AI_workflow_draft", reviewer="", review_status="needs_review",
                              notes="流程开发草稿，答案留空；已见开发题，永久不得改名为独立评价。支持候选未人工验收。"))
    sources = []
    for info in json.loads((PACKET / "source_manifest.json").read_bytes())["inputs"]:
        if info["path"].startswith("radar_corpus/raw/"):
            sources.append(dict(info, source_class="secondary_web_snapshot",
                                primary_source_verified=False))
    manifest = dict(protocol_version="radar_review_workflow_v1", packet_manifest_sha256=packet_hash,
                    source_classes=sources, fact_count=len(facts), question_count=len(questions),
                    development_only=True, answers_prepared=False, human_verified=False,
                    source_truth_verified=False,
                    grouping_warning="Source/model groups are workflow labels, not an independent split audit.")
    return {"review_decisions.csv": csv_bytes(DECISION_HEADERS, decisions),
            "development_questions.csv": csv_bytes(QUESTION_HEADERS, questions),
            "workflow_manifest.json": encoded(manifest)}


def validate_decisions(facts: list[dict], decisions: list[dict]) -> dict:
    fact_map = {row["fact_id"]: row for row in facts}
    ids = [row["fact_id"] for row in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(fact_map):
        raise ValueError("Decision rows must cover each packet fact exactly once")
    result = {"pending": 0, "needs_changes": 0, "accepted": 0, "rejected": 0,
              "accepted_snapshot_readings": 0, "independent_fact_eligible": 0}
    for decision in decisions:
        fact = fact_map[decision["fact_id"]]
        if decision["fact_record_sha256"] != fact_digest(fact) or decision["source_sha256"] != fact["source_sha256"]:
            raise ValueError("Decision is not bound to the exact fact and source version")
        state = decision["decision"]
        if state not in {"pending", "needs_changes", "accepted", "rejected"}:
            raise ValueError("Unknown review decision")
        result[state] += 1
        if state == "pending":
            if any(decision[key] for key in DECISION_HEADERS[4:]):
                raise ValueError("Pending rows must not imply a completed review")
            continue
        if decision["reviewer_kind"] not in {"human", "AI"} or not decision["reviewer"].strip() or not decision["reason"].strip():
            raise ValueError("A review requires declared reviewer kind, reviewer and reason")
        try:
            reviewed_at = datetime.fromisoformat(decision["reviewed_at"])
            if reviewed_at.tzinfo is None:
                raise ValueError("Missing time zone")
        except ValueError as exc:
            raise ValueError("A review requires an ISO timestamp including timezone") from exc
        if state != "accepted":
            continue
        if decision["reviewer_kind"] != "human" or re.search(
                r"(?i)(?:^|[^a-z0-9])(?:ai|gpt\d*|chatgpt|codex|assistant|model)(?:$|[^a-z0-9])",
                decision["reviewer"]):
            raise ValueError("Accepted facts require an explicitly declared human reviewer")
        if any(decision[key] not in {"confirmed", "not_applicable", "not_specified_in_source"}
               for key in CHECK_FIELDS):
            raise ValueError("Accepted decisions require each subject/variant/event/value/unit/condition check")
        scope = decision["decision_scope"]
        if scope == "source_snapshot_reading":
            result["accepted_snapshot_readings"] += 1
            continue
        if scope != "independent_fact":
            raise ValueError("Acceptance scope must distinguish snapshot reading from independent fact")
        if fact["value_status"] in {"ambiguous", "conflicting"}:
            raise ValueError("Resolve ambiguous/conflicting candidates in a new version before independent acceptance")
        if decision["subject_check"] != "confirmed":
            raise ValueError("Independent fact acceptance must resolve the subject")
        if (decision["independent_source_file"] == fact.get("source_file")
                or decision["independent_source_sha256"] == fact["source_sha256"]
                or decision["independent_source_uri"] == fact.get("source_uri")):
            raise ValueError("The known secondary snapshot cannot be promoted to an independent primary source")
        validate_independent_source(decision)
        result["independent_fact_eligible"] += 1
    return result


def validate_independent_source(decision: dict) -> None:
    if decision["independent_source_class"] != "primary":
        raise ValueError("Independent acceptance requires a declared primary source")
    uri = urlparse(decision["independent_source_uri"])
    if uri.scheme not in {"http", "https", "urn"} or (uri.scheme != "urn" and not uri.netloc):
        raise ValueError("Independent source needs a source URI or bibliographic URN")
    if (uri.hostname or "").endswith("wikipedia.org"):
        raise ValueError("Wikipedia is a secondary source in this review protocol")
    data = local_path(decision["independent_source_file"]).read_bytes()
    if digest(data) != decision["independent_source_sha256"]:
        raise ValueError("Independent source hash mismatch")
    locator = json.loads(decision["independent_source_locator"])
    if not isinstance(locator, dict) or locator.get("offset_unit") != "unicode_codepoint" or locator.get("end_exclusive") is not True:
        raise ValueError("Independent text locator must declare Unicode offsets and exclusive end")
    document_keys = {"original_file", "original_sha256", "page"}
    if document_keys.intersection(locator):
        if not document_keys.issubset(locator):
            raise ValueError("A document rendition requires the original document hash and page")
        if digest(local_path(locator["original_file"]).read_bytes()) != locator["original_sha256"]:
            raise ValueError("Original document hash mismatch")
        if type(locator["page"]) is not int or locator["page"] < 1:
            raise ValueError("Original document page must be a positive one-based page number")
    text = data.decode("utf-8")
    if locator.get("json_pointer"):
        text = packet_builder.resolve_pointer(json.loads(text), locator["json_pointer"])
    start, end = locator.get("char_start"), locator.get("char_end")
    if not isinstance(text, str) or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Independent source character range is invalid")
    excerpt = decision["independent_evidence_text"]
    if not excerpt or len(excerpt) > 180 or text[start:end] != excerpt:
        raise ValueError("Independent evidence must be an exact short located excerpt")


def validate_questions(facts: list[dict], questions: list[dict]) -> dict:
    expected = list(csv.DictReader(io.StringIO(initial_artifacts(facts)["development_questions.csv"].decode())))
    if questions != expected:
        raise ValueError("Development questions are frozen answer-free workflow drafts; create a new version for annotations")
    return {"development_questions": len(questions), "answers_filled": 0,
            "independent_evaluation_questions": 0,
            "source_groups": len({row["source_group"] for row in questions}),
            "model_family_groups": len({row["model_family"] for row in questions})}


def audit(workflow: Path) -> dict:
    facts, _ = load_packet()
    manifest = json.loads((workflow / "workflow_manifest.json").read_bytes())
    expected_manifest = json.loads(initial_artifacts(facts)["workflow_manifest.json"])
    if manifest != expected_manifest:
        raise ValueError("Workflow manifest no longer matches the fixed packet and source classes")
    decisions = read_csv(workflow / "review_decisions.csv", DECISION_HEADERS)
    questions = read_csv(workflow / "development_questions.csv", QUESTION_HEADERS)
    return dict(protocol_version=manifest["protocol_version"], machine_checks_passed=True,
                facts=len(facts), exact_source_bindings=len(facts),
                secondary_source_snapshots=len(manifest["source_classes"]),
                review_decisions=validate_decisions(facts, decisions),
                questions=validate_questions(facts, questions),
                human_identity_verified_by_software=False, factual_truth_verified_by_software=False,
                final_test_answers_generated=False, public_benchmark_holdout_read=False,
                packet_manifest_sha256=manifest["packet_manifest_sha256"],
                checked_files={name: digest((workflow / name).read_bytes()) for name in
                               ("review_decisions.csv", "development_questions.csv", "workflow_manifest.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.initialize:
        facts, _ = load_packet()
        outputs = initial_artifacts(facts)
        if any((args.workflow_dir / name).exists() for name in outputs):
            raise FileExistsError("Refusing to overwrite review work; use the default read-only audit")
        args.workflow_dir.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            (args.workflow_dir / name).write_bytes(data)
    report = audit(args.workflow_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(encoded(report))
    print(encoded(report).decode(), end="")


if __name__ == "__main__":
    main()
