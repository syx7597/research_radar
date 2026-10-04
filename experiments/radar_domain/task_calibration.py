"""Prepare AI-rewritten bilingual lookup questions; validate programs on CPU only.

These are derivatives of exposed development questions, permanently excluded
from final evaluation. Program validation is not model performance. No inference
or training is launched, and the original task, runtime and results stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .development_environment import DevelopmentKB, execute_program, render_records

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/thesis_direction_review/radar_development_v2"
PROBE = ROOT / "results/radar_domain/development_probe_v1"
DEST = ROOT / "artifacts/thesis_direction_review/radar_lookup_calibration_v1"
VERSION = "radar_lookup_calibration_v1"
QUESTIONS = [
    ("请取出所给来源快照的 Type 字段记录。", "Retrieve the Type field record from the given source snapshot."),
    ("请取出所给来源快照中 Since 所对应的事件记录。", "Retrieve the event record indicated by Since in the given source snapshot."),
    ("请取出所给来源快照中 initial operational capacity 对应的事件记录。", "Retrieve the initial operational capacity event record from the given source snapshot."),
    ("请取出所给来源快照的 Frequency 字段记录。", "Retrieve the Frequency field record from the given source snapshot."),
    ("请取出所给来源快照中带有 Illuminator 标签的频率记录。", "Retrieve the frequency record labeled Illuminator from the given source snapshot."),
    ("请取出所给来源快照中带有 Tracking Radar 标签的频率记录。", "Retrieve the frequency record labeled Tracking Radar from the given source snapshot."),
    ("请取出所给来源快照中带有 surface 标签的 PRF 记录。", "Retrieve the PRF record labeled surface from the given source snapshot."),
    ("请取出所给来源快照中带有 air 标签的 PRF 记录。", "Retrieve the PRF record labeled air from the given source snapshot."),
    ("请取出所给来源快照的 Frequency 字段记录。", "Retrieve the Frequency field record from the given source snapshot."),
    ("请取出所给来源快照的 Pulsewidth 字段记录。", "Retrieve the Pulsewidth field record from the given source snapshot."),
    ("请取出所给来源快照的取消事件记录。", "Retrieve the cancellation event record from the given source snapshot."),
    ("请取出所给来源快照的 Power 字段记录。", "Retrieve the Power field record from the given source snapshot."),
]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build():
    kb = DevelopmentKB(DATA / "readings.json")
    records = sorted(kb.records, key=lambda item: item["fact_id"])
    originals = json.loads((DATA / "references.json").read_bytes())["records"]
    failure = json.loads((PROBE / "summary.json").read_bytes())
    if len(records) != 12 or len(originals) != 12 or failure["cpu_replay"] != dict(failure["cpu_replay"], rows=60, mismatches=0):
        raise ValueError("Calibration requires the original twelve records and completed 60-row probe replay")
    questions = {"zh": [], "en": []}
    references, validation = [], []
    for number, (record, original, pair) in enumerate(zip(records, originals, QUESTIONS), start=1):
        if original["expected_fact_ids"] != [record["fact_id"]]:
            raise ValueError("Original question/record pairing changed")
        qid = f"radar-lookup-dev-{number:02}"
        context = (f"source_context_id={record['source_context_id']}; "
                   f"source_uri={record['citation']['source_uri']}; "
                   f"entity_anchor={record['entity_name']}.")
        zh_context = "来源上下文（实体锚点为历史入库标签）：" + context
        en_context = "Source context (the entity anchor is a historical ingestion label): " + context
        for language, text, suffix in zip(("zh", "en"), pair, (zh_context, en_context)):
            questions[language].append({"id": qid, "question": text + "\n" + suffix})
        qualifiers = [(key, record[key]) for key in ("event_type", "condition_raw") if record[key] is not None]
        if len(qualifiers) > 1:
            raise ValueError("The initial lookup calibration admits at most one qualifier")
        program = "Find<arg>" + record["entity_name"] + "<func>"
        if qualifiers:
            key, value = qualifiers[0]
            program += "QueryAttrUnderCondition<arg>" + "<arg>".join((record["attribute"], key, value))
        else:
            program += "QueryAttr<arg>" + record["attribute"]
        episode = execute_program(kb, program)
        if episode.prediction != [record] or not episode.done or any(not event["observation"]["ok"] for event in episode.events):
            raise ValueError(f"Reference program does not select exactly the complete source record: {qid}")
        rendered = render_records(episode.prediction)
        if rendered != render_records([record]) or record["citation"]["source_uri"] not in rendered:
            raise ValueError(f"Source-preserving rendering failed: {qid}")
        references.append({"id": qid, "derived_from_question_id": original["id"],
                           "expected_fact_ids": [record["fact_id"]], "canonical_program": program,
                           "languages": ["zh", "en"], "reference_scope": "same bound source record in both languages"})
        validation.append({"id": qid, "exact_source_record_selected": True, "source_rendering_preserved": True,
                           "executed_calls": episode.calls, "rendered_evidence_sha256": digest(rendered.encode("utf-8"))})
    outputs = {f"questions.{language}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
               for language, rows in questions.items()}
    outputs["references.json"] = encoded({"version": VERSION, "split": "development_seen_derivative",
                                          "reference_kind": "AI_authored_programs_and_bilingual_rewrites",
                                          "human_verified": False, "independent_gold": False,
                                          "excluded_from_final_evaluation": True, "must_not_be_loaded_by_inference": True,
                                          "records": references})
    inputs = [DATA / name for name in ("readings.json", "questions.jsonl", "references.json", "manifest.json")]
    inputs += [PROBE / name for name in ("protocol.json", "summary.json", "per_question_diagnosis.json", "execution_summary.json")]
    inputs += [Path(__file__), Path(__file__).with_name("development_environment.py"), Path(__file__).with_name("development_probe.py")]
    outputs["manifest.json"] = encoded({
        "version": VERSION, "purpose": "Offline material for one future fixed lookup/language diagnostic; not launched here.",
        "split": "development_seen_derivative", "permanently_excluded_from_final_evaluation": True,
        "rewrite_and_translation_by": "AI", "bilingual_semantic_equivalence": "AI intended and reviewed; no independent expert verification",
        "human_verified": False, "independent_gold": False, "question_count_per_language": 12,
        "model_inference_runs": 0, "new_training_started": False, "gpu_used": False,
        "input_files_only": ["questions.zh.jsonl", "questions.en.jsonl"], "reference_file_excluded_from_inference": "references.json",
        "cpu_validation": {"programs_checked": 12, "exact_record_matches": 12, "renderer_checks": 12,
                           "is_model_performance": False, "records": validation},
        "limits": ["Rewritten lookup tasks do not measure semantic adjudication or equipment truth.",
                   "These twelve exposed derivatives cannot become held-out radar QA.",
                   "The two language versions share the same targets and reference programs; CPU success proves only executability.",
                   "Any later comparison with the original probe also changes task wording and must not be attributed solely to language."],
        "inputs_sha256": {str(item.relative_to(ROOT)): digest(item.read_bytes()) for item in inputs},
        "outputs_sha256": {name: digest(value) for name, value in outputs.items()},
    })
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    artifacts = build()
    if args.check:
        if any((DEST / name).read_bytes() != value for name, value in artifacts.items()):
            raise ValueError("Calibration artifacts differ from their deterministic construction")
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        if any((DEST / name).exists() for name in artifacts):
            raise FileExistsError("Calibration already exists; use --check")
        for name, value in artifacts.items():
            with (DEST / name).open("xb") as stream:
                stream.write(value)
    print(json.dumps({"mode": "verified" if args.check else "created", "cpu_programs_validated": 12,
                      "model_inference_runs": 0, "new_training_started": False}))


if __name__ == "__main__":
    main()
