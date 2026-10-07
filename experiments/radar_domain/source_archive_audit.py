"""Verify local source bytes and rebuild a public preparation summary.

No network, questions, models or factual gold certification. Original bytes and
derived text/images stay in ignored data/. This does not replace semantic review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "artifacts/thesis_direction_review/radar_sources_v2"
LOCAL = ROOT / "data/radar_sources_v2"
NAMES = ("weather_admission.json", "marine_admission.json", "replacement_admission.json")
OUT = PUBLIC / "archive_summary.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    rows, groups, bindings, verified = [], [], {}, set()

    def check_file(relative, expected, size=None):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(LOCAL.resolve()):
            raise ValueError("Source artifact outside local archive")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or (size is not None and len(raw) != size):
            raise ValueError(f"Changed source bytes: {relative}")
        verified.add(relative)
        return raw

    for name in NAMES:
        p = PUBLIC / name
        doc = json.loads(p.read_text())
        if doc["model_run_ready"] or doc["human_review"]:
            raise ValueError("Unexpected readiness or human-review claim")
        bindings[str(p.relative_to(ROOT))] = sha(p)
        for path, expected in doc["inputs_sha256"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Changed input: {path}")
            bindings[path] = expected
        groups.extend(doc["candidate_groups"])
        for row in doc["sources"]:
            relative = row.get("file") or row["local_relative_path"]
            raw = check_file(relative, row["sha256"], row.get("bytes", row.get("size_bytes")))
            pdf = row.get("pdf")
            if pdf and (not raw.startswith(b"%PDF-") or pdf["page_count"] <= 0):
                raise ValueError("PDF identity mismatch")
            for item in row.get("derived_local_artifacts", []):
                check_file(item["path"], item["sha256"], item["size_bytes"])
            if "capture_metadata" in row:
                item = row["capture_metadata"]
                meta = json.loads(check_file(item["path"], item["sha256"], item["size_bytes"]))
                if any(row[k] != meta[k] for k in meta):
                    raise ValueError("Capture report differs from saved HTTP metadata")
            rows.append(row)
    ids = {r["source_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate source identifiers")
    for group in groups:
        if group["evaluation_admitted"]:
            raise ValueError("No evaluation questions have been admitted")
        primary = group.get("primary_source_ids") or ([group["primary_source_id"]] if "primary_source_id" in group else [])
        if group["eligible_for_pending_annotation"] and not primary:
            raise ValueError("Missing primary source")
        if not set(primary) <= ids:
            raise ValueError("Unknown primary source")
        if any(r["archive_role"] == "access_failure_evidence" for r in rows if r["source_id"] in primary):
            raise ValueError("Access failure used as primary material")
    registry_path = ROOT / "artifacts/thesis_direction_review/radar_coverage_v2/exposure_registry.json"
    registry = json.loads(registry_path.read_text())
    bindings[str(registry_path.relative_to(ROOT))] = sha(registry_path)
    known = set(registry["source_sha256"])
    uris = set(registry["source_uris"])
    overlaps = [r["source_id"] for r in rows if r["sha256"] in known or
                (r.get("request_uri") or r.get("requested_url")) in uris or
                (r.get("final_uri") or r.get("resolved_url")) in uris]
    history = json.loads((PUBLIC / "history_screening.json").read_text())
    for name in ("history_screening.json", "history_inputs.json", "candidate_extension.json", "annotation_scope.json", "source_preparation_review.json"):
        bindings[str((PUBLIC / name).relative_to(ROOT))] = sha(PUBLIC / name)
    aliases = {"eec_ranger_x_band": "eec_ranger", "jrc_jma5200mk2": "jrc_jma5200", "garmin_gmr_fantom": "garmin_fantom"}
    history_by_id = {g["family_id"]: g for g in history["groups"]}
    family_status = []
    for g in groups:
        family = g["proposed_family_id"]
        history_id = aliases.get(family, family)
        prior = history_by_id[history_id]
        family_status.append({"family_id": family, "history_family_id": history_id, "history_status": prior["status"],
                              "pending_annotation": g["eligible_for_pending_annotation"], "evaluation_admitted": False,
                              "current_model_training_lineage_overlap": prior["current_model_training_lineage_overlap"]})
    scope = json.loads((PUBLIC / "annotation_scope.json").read_text())
    selected_pdf_pages, selected_html_sections = 0, 0
    source_by_id = {r["source_id"]: r for r in rows}
    for item in scope["page_scope"]:
        row = source_by_id[item["source_id"]]
        group = next(g for g in groups if g["proposed_family_id"] == item["family_id"])
        primary = group.get("primary_source_ids") or [group["primary_source_id"]]
        if item["source_id"] not in primary or not group["eligible_for_pending_annotation"]:
            raise ValueError("Annotation scope outside admitted primary bytes")
        if "pages_one_based" in item:
            if any(not 1 <= n <= row["pdf"]["page_count"] for n in item["pages_one_based"]):
                raise ValueError("Annotation page outside PDF")
            selected_pdf_pages += len(item["pages_one_based"])
        else:
            if not item["whole_HTML_section"] or row["archive_role"] != "primary_manual_section":
                raise ValueError("Annotation HTML is not a primary section")
            selected_html_sections += 1
    if (selected_pdf_pages, selected_html_sections) != (scope["counts"]["selected_PDF_pages"], scope["counts"]["selected_HTML_content_sections"]):
        raise ValueError("Annotation scope count mismatch")
    review = json.loads((PUBLIC / "source_preparation_review.json").read_text())
    for declaration in (scope, review):
        for path, expected in declaration["inputs_sha256"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Changed scope/review input: {path}")
    eligible = [g for g in groups if g["eligible_for_pending_annotation"]]
    failed = [r for r in rows if r["archive_role"] == "access_failure_evidence"]
    pdfs = [r for r in rows if "pdf" in r]
    result = {
        "version": "radar_source_preparation_summary_v2", "recorded_on": "2026-10-07",
        "status": "archive_verified_annotation_pending", "model_run_ready": False,
        "evaluation_frozen": False, "human_gold": False,
        "counts": {
            "screened_candidate_families": len(groups), "families_with_primary_material": len(eligible),
            "families_pending_official_access": len(groups) - len(eligible),
            "publishers_with_primary_material": len({g["publisher"] for g in eligible}),
            "application_groups": dict(sorted(Counter(g["application"] for g in eligible).items())),
            "named_main_models": sum(len(g["document_models"]) for g in eligible),
            "original_responses": len(rows), "successful_original_snapshots": len(rows)-len(failed),
            "failed_access_responses_retained": len(failed), "primary_PDF_documents": len(pdfs),
            "primary_PDF_pages": sum(r["pdf"]["page_count"] for r in pdfs),
            "primary_manual_HTML_sections": sum(r["archive_role"] == "primary_manual_section" for r in rows),
            "auxiliary_HTML_snapshots": sum(r["archive_role"] == "identity_or_entry_support_only" for r in rows),
            "selected_PDF_pages_for_annotation": selected_pdf_pages,
            "selected_HTML_sections_for_annotation": selected_html_sections,
            "verified_local_files": len(verified), "new_curated_readings": 0, "new_questions": 0,
            "new_model_runs": 0, "new_training_runs": 0, "new_GPU_hours": 0,
        },
        "family_status": family_status,
        "exact_known_debug_source_hash_or_URI_overlap": overlaps,
        "historical_family_mentions": [g["family_id"] for g in history["groups"] if g["status"] == "historical_family_mentions_identified"],
        "necessity_gate": "not_tested_no_questions_or_fixed_E_q_yet",
        "next_action": "Declare uniform source reading scope; annotate complete eligible page readings; independent authorship and review before CPU and execution freeze",
        "limits": [
            "Source families are declared review units, not certified statistically independent samples.",
            "No exact hash/URI match does not exclude shared lineage or historical content; see history screening.",
            "Primary PDF page totals include unannotated pages; model and file counts are not verified fact counts.",
            "HTML sections of one manual and two METEK documents do not create extra family groups.",
            "Old corpus and QA/SFT overlap is disclosed; actual checkpoint training lineage and pretraining remain unknown.",
            "Byte verification is not independent semantic reference or proof the planned representation improves answers.",
        ],
        "inputs_sha256": dict(sorted(bindings.items())),
        "audit_code_sha256": sha(Path(__file__)),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build()
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        with OUT.open("x") as stream:
            stream.write(content)
    elif OUT.read_text() != content:
        raise ValueError("Source preparation summary changed")
    print(json.dumps({"status": "written" if args.write else "checked", **result["counts"], "model_run_ready": False}))
