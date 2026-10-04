"""Query the fixed radar review packet without promoting candidates to facts.

Only Python's standard library and the existing packet/review auditors are used.
This intentionally does not import the legacy KG executors: their triple-only
values and first-number extraction cannot retain the packet's evidence scope.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from scripts import audit_radar_review_workflow as workflow

MODES = ("accepted-independent", "source_preview")
QUERY_FIELDS = {"entity", "attribute", "event_type", "variant", "condition_raw"}
OPTIONAL_FIELDS = {"event_type", "variant", "condition_raw"}
LIMITS = [
    "Exact structured record matching only; no natural-language interpretation or alias expansion.",
    "Omitted event/variant/condition selectors retain separate matching records; no scalar collapse.",
    "A null optional selector matches an unrecorded field, not an unconditional or universal fact.",
    "No match means no matching admitted record in this bounded packet, not real-world absence or zero.",
    "Snapshot readings retain AI-prepared interpretations and are not independent accepted facts.",
    "Hash/locator/review-form checks do not establish factual truth or authenticate a human identity.",
    "This fixed 12-record pilot is not a representative corpus or independent QA evaluation.",
]


def validate_query(query: dict) -> dict:
    if not isinstance(query, dict):
        raise ValueError("Query must be a JSON object")
    if set(query) - QUERY_FIELDS:
        raise ValueError("Unsupported query fields; no selector may be silently ignored")
    for key in ("entity", "attribute"):
        if not isinstance(query.get(key), str) or not query[key].strip():
            raise ValueError(f"Query requires a nonempty exact {key}")
    for key in OPTIONAL_FIELDS & set(query):
        value = query[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty exact string or null for unrecorded")
    return dict(query)


def matches(fact: dict, query: dict) -> bool:
    if query["entity"] not in (fact["entity_id"], fact["entity_name"]):
        return False
    if query["attribute"] != fact["attribute"]:
        return False
    return all((fact[key] or None) == query[key]
               for key in OPTIONAL_FIELDS & set(query))


class RadarRecordAdapter:
    """Validated immutable in-memory view of the fixed review packet.

    Each construction validates the entire packet and workflow before reading any
    query. The caller cannot provide an unvalidated list of accepted records.
    Create a fresh adapter to observe a later review version.
    """

    def __init__(self, workflow_dir: Path = workflow.DEFAULT_WORKFLOW):
        workflow_dir = Path(workflow_dir)
        self._audit = workflow.audit(workflow_dir)
        facts, _ = workflow.load_packet()
        # Bind the rows consumed here to the audit result, including all decisions,
        # and reject a concurrent workflow replacement between reads. Parse those
        # exact bytes: legitimate CSV quoting/newline styles are not canonicalized.
        bound_files = {name: (workflow_dir / name).read_bytes()
                       for name in self._audit["checked_files"]}
        for name, expected in self._audit["checked_files"].items():
            if workflow.digest(bound_files[name]) != expected:
                raise ValueError("Workflow changed while constructing the read-only view")
        reader = csv.DictReader(io.StringIO(bound_files["review_decisions.csv"].decode("utf-8"), newline=""))
        if reader.fieldnames != workflow.DECISION_HEADERS:
            raise ValueError("Unexpected decision CSV columns")
        decisions = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in decisions):
            raise ValueError("Ragged decision CSV rows")
        if workflow.digest((workflow.PACKET / "source_manifest.json").read_bytes()) != self._audit["packet_manifest_sha256"]:
            raise ValueError("Packet changed while constructing the read-only view")
        decision_audit = workflow.validate_decisions(facts, decisions)
        if decision_audit != self._audit["review_decisions"]:
            raise ValueError("Decision admission changed during validation")
        self._facts = tuple(facts)
        self._decisions = {item["fact_id"]: item for item in decisions}
        self._eligible = frozenset(
            item["fact_id"] for item in decisions
            if item["decision"] == "accepted" and item["decision_scope"] == "independent_fact"
        )
        if len(self._eligible) != decision_audit["independent_fact_eligible"]:
            raise ValueError("Admission count differs from the full review audit")

    def audit_summary(self) -> dict:
        return json.loads(json.dumps(self._audit))

    def _record(self, fact: dict, *, preview: bool) -> dict:
        decision = self._decisions[fact["fact_id"]]
        admitted = fact["fact_id"] in self._eligible
        citation = dict(source_uri=fact["source_uri"], source_file=fact["source_file"],
                        source_sha256=fact["source_sha256"],
                        source_class="secondary_web_snapshot",
                        locator=json.loads(fact["locator"]), evidence_text=fact["evidence_text"])
        result = dict(
            fact_id=fact["fact_id"], fact_record_sha256=workflow.fact_digest(fact),
            entity_id=fact["entity_id"], entity_name=fact["entity_name"],
            variant=fact["variant"] or None,
            variant_status="recorded" if fact["variant"] else "not_recorded",
            attribute=fact["attribute"], event_type=fact["event_type"] or None,
            legacy_value_raw=fact["value_raw"],
            legacy_value_scope="Old extracted attribute string; may contain extraction/unit errors. Not the normalized value or an accepted fact; exact snapshot text is citation.evidence_text.",
            value={key: fact[key] or None for key in
                   ("value_kind", "value_status", "value", "min_value", "max_value", "unit_raw", "unit_std")},
            condition=dict(status=fact["condition_status"], raw=fact["condition_raw"] or None,
                           items=json.loads(fact["conditions_json"])),
            citation=citation,
            review=dict(packet_status=fact["review_status"], packet_preparer=fact["reviewer"],
                        decision=decision["decision"], decision_scope=decision["decision_scope"] or None,
                        reviewer_kind=decision["reviewer_kind"] or None,
                        reviewer=decision["reviewer"] or None, reviewed_at=decision["reviewed_at"] or None,
                        human_review_declared=(decision["reviewer_kind"] == "human"
                                               and decision["decision"] != "pending"),
                        independent_fact_admitted=admitted, human_identity_verified_by_software=False,
                        value_status_is_not_a_truth_label=True,
                        notes=fact["review_notes"], decision_reason=decision["reason"] or None),
            reading_scope="该快照记载（候选结构化解释，非事实答案）" if preview else "independent_fact_review_scope",
            factual_truth_verified_by_software=False,
        )
        if admitted:
            result["independent_citation"] = dict(
                source_uri=decision["independent_source_uri"],
                source_file=decision["independent_source_file"],
                source_sha256=decision["independent_source_sha256"], source_class="primary",
                locator=json.loads(decision["independent_source_locator"]),
                evidence_text=decision["independent_evidence_text"],
            )
        return result

    def query(self, query: dict, mode: str = "accepted-independent") -> dict:
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        query = validate_query(query)
        matched = [fact for fact in self._facts if matches(fact, query)]
        preview = mode == "source_preview"
        admitted = matched if preview else [fact for fact in matched if fact["fact_id"] in self._eligible]
        records = [self._record(fact, preview=preview) for fact in admitted]
        if preview:
            status = "snapshot_readings_available" if records else "no_matching_snapshot_record"
        elif not self._eligible:
            status = "no_accepted_independent_data"
        else:
            status = "accepted_fact_records_available" if records else "no_matching_accepted_fact"
        return dict(
            interface_version="radar_record_query_v1", mode=mode, query=query, status=status,
            answer_status="not_a_fact_answer" if preview else status,
            snapshot_readings=records if preview else [],
            accepted_facts=[] if preview else records,
            matching_candidate_records=len(matched), returned_records=len(records),
            dataset_independent_fact_eligible=len(self._eligible),
            knowledge_version=dict(packet_manifest_sha256=self._audit["packet_manifest_sha256"],
                                   workflow_files_sha256=dict(self._audit["checked_files"])),
            whole_packet_and_review_checks_passed=True,
            scope_limits=list(LIMITS),
        )
