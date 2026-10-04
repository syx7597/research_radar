"""Replay ten already exposed workflow questions as manually specified queries.

Outputs are development demonstrations, never model accuracy or independent gold.
The original question and review files are read-only and remain answer-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .adapter import LIMITS, MODES, RadarRecordAdapter
from scripts import audit_radar_review_workflow as workflow

DEFAULT_OUTPUT = workflow.ROOT / "artifacts/thesis_direction_review/radar_domain_pilot"
DEMO_QUERIES = [
    ("02", dict(entity="AN/MPQ-65", attribute="service_entry", event_type="in_service")),
    ("03", dict(entity="AN/MPQ-65", attribute="service_entry", event_type="initial_operational_capacity")),
    ("04", dict(entity="AN/APY-9", attribute="frequency", variant=None)),
    ("05", dict(entity="AN/SPG-51", attribute="frequency", condition_raw="Illuminator")),
    ("06", dict(entity="AN/SPG-51", attribute="frequency", condition_raw="Tracking Radar")),
    ("07", dict(entity="AN/SPG-51", attribute="prf", condition_raw="surface")),
    ("08", dict(entity="AN/SPG-51", attribute="prf", condition_raw="air")),
    ("09", dict(entity="AN/SPN-35", attribute="frequency", condition_raw=None)),
    ("11", dict(entity="AN/SPG-59", attribute="service_entry", event_type="cancellation")),
    ("12", dict(entity="EL/M-2080", attribute="power", variant=None)),
]


def build_artifacts() -> dict[str, bytes]:
    adapter = RadarRecordAdapter()
    questions = workflow.read_csv(workflow.DEFAULT_WORKFLOW / "development_questions.csv", workflow.QUESTION_HEADERS)
    questions_by_id = {row["qid"]: row for row in questions}
    replays = []
    for suffix, query in DEMO_QUERIES:
        qid = "radar-workflow-dev-" + suffix
        question = questions_by_id[qid]
        if question["answer_json"] or question["answer_status"] or question["split"] != "development_only":
            raise ValueError("The original workflow question must remain answer-free development material")
        preview = adapter.query(query, "source_preview")
        default = adapter.query(query)
        replays.append(dict(
            qid=qid, original_development_question=question["question"],
            query_authorship="AI_manually_specified_for_interface_development_not_model_prediction",
            query=query, source_preview=preview, default_accepted_independent=default,
            is_final_qa_answer=False, is_independent_evaluation=False,
        ))
    if len(replays) != 10:
        raise ValueError("The bounded pilot contains exactly ten existing development questions")
    report = dict(
        pilot_version="radar_domain_pilot_v1", purpose="manual_structured_query_development_demo",
        whole_packet_audit=adapter.audit_summary(), development_questions_replayed=len(replays),
        query_calls=len(replays) * 2, model_calls=0, gpu_calls=0, training_runs=0,
        independent_qa_examples=0, final_answers_written=0,
        default_admitted_results=sum(item["default_accepted_independent"]["returned_records"] for item in replays),
        source_preview_records=sum(item["source_preview"]["returned_records"] for item in replays),
        performance_metric=None,
        performance_metric_reason="Manual query replay demonstrates interface behavior; no model accuracy or effectiveness claim.",
        original_question_file_unmodified=True, original_review_decisions_unmodified=True,
        replays=replays,
    )
    contract = dict(
        interface_version="radar_record_query_v1", default_mode="accepted-independent", modes=list(MODES),
        query_schema=dict(required={"entity": "exact entity_id or entity_name", "attribute": "exact attribute"},
                          optional={"event_type": "exact event string or null for absent field",
                                    "variant": "exact version string or null for unrecorded version",
                                    "condition_raw": "exact original condition label or null for unrecorded condition"},
                          matching="AND of all supplied selectors; unknown fields are errors; no inference or aliases"),
        response_namespaces=dict(snapshot_readings="source_preview only; exact source citation and candidate interpretation",
                                 accepted_facts="accepted-independent only after the complete independent review gate"),
        source_binding="whole packet and workflow audit, fact record hash, source file hash, JSON pointer and Unicode offsets",
        value_encoding="decimal strings preserve original precision; ranges retain two endpoints; unknown values/units are null",
        legacy_value_raw="Unmodified old extracted attribute string, potentially erroneous; kept separate from normalized candidate value and exact snapshot citation.evidence_text.",
        no_match_meaning="no matching record in this bounded and admitted view; never zero, false, or global absence",
        reuse=dict(validators=["scripts/prepare_radar_review_packet.py::validate_packet",
                               "scripts/audit_radar_review_workflow.py::audit",
                               "scripts/audit_radar_review_workflow.py::validate_decisions"],
                   legacy_executors_not_imported=[
                       dict(file="agent/composition.py", symbol="CompositionExecutor._numeric",
                            limitation="Extracts only the first numeric substring; ranges and attached scopes are unavailable."),
                       dict(file="ca_agraphrag/kg_tools.py", symbol="KGTools.__init__/lookup",
                            limitation="Indexes head/relation/tail sets; event, variant, condition, source and review records are not preserved."),
                   ]),
        scope_limits=LIMITS,
        example_commands=[
            'python3 -m experiments.radar_domain --query \'{"entity":"AN/SPG-51","attribute":"prf","condition_raw":"air"}\'',
            'python3 -m experiments.radar_domain --mode source_preview --query \'{"entity":"AN/SPG-51","attribute":"prf","condition_raw":"air"}\'',
            "python3 -m experiments.radar_domain.pilot --check-only",
        ],
    )
    files = {"query_replay.json": workflow.encoded(report), "interface_contract.json": workflow.encoded(contract)}
    source_files = ["experiments/radar_domain/adapter.py", "experiments/radar_domain/__main__.py",
                    "experiments/radar_domain/pilot.py", "tests/test_radar_domain.py"]
    manifest = dict(
        pilot_version="radar_domain_pilot_v1", source_mode="existing_local_snapshots_only",
        independently_accepted_facts=adapter.audit_summary()["review_decisions"]["independent_fact_eligible"],
        original_workflow_files_sha256=adapter.audit_summary()["checked_files"],
        packet_manifest_sha256=adapter.audit_summary()["packet_manifest_sha256"],
        code_sha256={name: hashlib.sha256((workflow.ROOT / name).read_bytes()).hexdigest() for name in source_files},
        output_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        changes_to_original_data=False, benchmark_questions_read=False, model_or_gpu_used=False,
    )
    files["manifest.json"] = workflow.encoded(manifest)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(DEFAULT_OUTPUT.resolve()):
        parser.error("Output must stay within radar_domain_pilot")
    files = build_artifacts()
    if args.check_only:
        for name, expected in files.items():
            if (args.output_dir / name).read_bytes() != expected:
                raise ValueError(f"Pilot differs from deterministic read-only replay: {name}")
    else:
        if any((args.output_dir / name).exists() for name in files):
            raise FileExistsError("Pilot output exists; use --check-only or a new version subdirectory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            with (args.output_dir / name).open("xb") as handle:
                handle.write(data)
    print(json.dumps(dict(mode="verified_without_writing" if args.check_only else "created",
                          development_questions=10, model_accuracy=None,
                          independent_facts=json.loads(files["manifest.json"])["independently_accepted_facts"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
