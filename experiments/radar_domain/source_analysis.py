"""Verify every source-evaluation run before scoring or preparing AI review.

Exact reference-field agreement is a strict, conservative mechanical diagnostic.
It is never described as free-text semantic correctness. Natural answers are
exported with arm identities hidden for a separate AI source-supported review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from . import source_probe as probe
from . import source_answers as answers
from . import source_rag as rag
from .development_analysis import require, read_jsonl, replay_row, TOTAL_KEYS
from .development_environment import DevelopmentKB, execute_program


def verify_complete():
    protocol = probe.verify("P", weights=False)
    completion = json.loads((probe.OUT / "pipeline_completed.json").read_text())
    require(completion["status"] == "completed" and completion["jobs_returned"] == 11, "All eleven GPU jobs must complete")
    require(completion["protocol_sha256"] == probe.sha(probe.PROTOCOL), "Controller protocol mismatch")
    final_base = completion["base_files_final"]
    for item in protocol["base_weights"]["files"]:
        name = str(Path(protocol["model"]) / item["file"])
        require(final_base[name]["sha256"] == item["sha256"] and final_base[name]["bytes"] == item["bytes"], "Final base identity mismatch")
    questions = probe.questions()
    kb = DevelopmentKB(probe.DATA / "kb.json")
    selections, generated, runtimes, hashes = {}, {}, {}, {}
    # Do not replay or open references until all five + six completion manifests exist.
    for label in probe.MODELS:
        path = probe.OUT / f"{label}.selection.jsonl"
        rp = path.with_suffix(".runtime.json")
        runtime = json.loads(rp.read_text())
        require(runtime["status"] == "completed" and runtime["label"] == label, "Incomplete selector")
        require(runtime["protocol_sha256"] == probe.sha(probe.PROTOCOL) and runtime["output_sha256"] == probe.sha(path), "Selection byte binding mismatch")
        require(runtime["model"] == protocol["models"][label] and runtime["config"] == protocol["inference"], "Selection model/config mismatch")
        selections[label], runtimes[label, "selection"] = read_jsonl(path), runtime
        hashes.update({str(path): probe.sha(path), str(rp): probe.sha(rp)})
    for arm in probe.ARMS:
        path = probe.OUT / f"{arm}.answers.jsonl"
        rp = path.with_suffix(".runtime.json")
        runtime = json.loads(rp.read_text())
        require(runtime["status"] == "completed" and runtime["arm"] == arm, "Incomplete common-generator answer run")
        require(runtime["protocol_sha256"] == probe.sha(probe.PROTOCOL) and runtime["output_sha256"] == probe.sha(path), "Answer byte binding mismatch")
        require(runtime["model"] == protocol["model"] and runtime["adapter"] is None and runtime["config"] == protocol["answer_inference"], "Common answer generator changed")
        generated[arm], runtimes[arm, "answer"] = read_jsonl(path), runtime
        hashes.update({str(path): probe.sha(path), str(rp): probe.sha(rp)})
    qids = [q["id"] for q in questions]
    replayed = 0
    for label, rows in selections.items():
        require([r["id"] for r in rows] == qids, "Selection ID order mismatch")
        totals = Counter(questions=len(rows))
        for row, q in zip(rows, questions):
            require(row["label"] == label, "Wrong selector row label")
            replay_row(kb, row, q, protocol, label)
            totals.update({key: row[key] for key in TOTAL_KEYS})
            replayed += 1
        require(dict(totals) == runtimes[label, "selection"]["totals"], "Selection totals mismatch")
    for arm, rows in generated.items():
        expected, evidence_hashes = answers.evidence_rows(arm, protocol)
        require(runtimes[arm, "answer"]["evidence_inputs_sha256"] == evidence_hashes, "Wrong answer evidence binding")
        require([r["id"] for r in rows] == qids, "Answer ID order mismatch")
        totals = Counter()
        for row, expected_row in zip(rows, expected):
            require(row["arm"] == arm and row["question"] == expected_row["question"], "Wrong answer row")
            require(row["evidence"] == expected_row["evidence"], "Answer evidence differs from fixed retrieval/selection")
            require(row["messages"] == rag.answer_messages(expected_row["question"], expected_row["evidence"]), "Shared answer prompt drift")
            parsed, error = None, None
            try:
                parsed = rag.parse_answer(row["generated_text"], row["evidence"])
            except (ValueError, TypeError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            require(parsed == row["parsed"] and error == row["parse_error"], "Answer parser replay mismatch")
            require(all(type(row[k]) is int and row[k] >= 0 for k in ("input_tokens", "generated_tokens", "total_tokens")), "Invalid token counts")
            require(row["generated_tokens"] <= protocol["answer_inference"]["max_generated"] and
                    row["input_tokens"] + protocol["answer_inference"]["max_generated"] <= protocol["answer_inference"]["max_context"], "Answer budget violation")
            require(row["total_tokens"] == row["input_tokens"] + row["generated_tokens"], "Answer token sum mismatch")
            totals["questions"] += 1
            totals["parsed"] += int(parsed is not None)
            for key in ("input_tokens", "generated_tokens", "total_tokens"):
                totals[key] += row[key]
        require(dict(totals) == runtimes[arm, "answer"]["totals"], "Answer aggregate mismatch")
    probe.previous_training.hash_mapping(hashes)
    probe.previous_training.hash_mapping(protocol["inputs_sha256"])
    return protocol, questions, kb, selections, generated, runtimes, hashes, replayed


def load_references(protocol, questions, kb):
    probe.previous_training.hash_mapping(protocol["reference_sha256"])
    rows = read_jsonl(probe.DATA / "references.jsonl")
    require([r["id"] for r in rows] == [q["id"] for q in questions], "Reference ID coverage mismatch")
    for row in rows:
        require(row["review_status"] == "AI_only_development", "Reference falsely claims human gold")
        episode = execute_program(kb, row["canonical_program"])
        require(episode.prediction is not None and {r["fact_id"] for r in episode.prediction} == set(row["expected_fact_ids"]), "Reference program is inconsistent")
        require(row["semantic_claims"], "Semantic reference claims are missing")
    return {row["id"]: row for row in rows}


def norm_text(value):
    if value is None:
        return None
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).casefold()).replace("μ", "u").replace("µ", "u")


def claim_signature(claim):
    """Strict normalized field signature; aliases/paraphrases need separate review."""
    # SI prefixes are case-sensitive: mHz/MHz and ms/Ms must not collapse.
    raw_unit = (None if claim["unit"] is None else re.sub(r"\s+", "", unicodedata.normalize("NFKC", claim["unit"])).replace("μ", "u").replace("µ", "u"))
    conversions = {"Hz": ("Hz", "1"), "mHz": ("Hz", ".001"), "kHz": ("Hz", "1000"),
                   "MHz": ("Hz", "1000000"), "GHz": ("Hz", "1000000000"),
                   "W": ("W", "1"), "mW": ("W", ".001"), "kW": ("W", "1000"), "MW": ("W", "1000000"),
                   "s": ("s", "1"), "ms": ("s", ".001"), "us": ("s", ".000001"), "ns": ("s", ".000000001"), "Ms": ("s", "1000000"),
                   "km": ("m", "1000"), "m": ("m", "1"), "°C": ("degC", "1"), "degC": ("degC", "1"),
                   "°": ("degree", "1"), "deg": ("degree", "1"), "degree": ("degree", "1"), "degrees": ("degree", "1")}
    canonical, scale = conversions.get(raw_unit, (raw_unit, "1"))
    def number(value):
        if value is None:
            return None
        return str((Decimal(str(value)) * Decimal(scale)).normalize())
    kind = claim["value_kind"]
    if kind in {"scalar", "approximate"}:
        values = (number(claim["value"]), None, None)
    elif kind in {"range", "lower_bound", "upper_bound"}:
        values = (None, number(claim["min_value"]), number(claim["max_value"]))
    elif kind == "options":
        try:
            value = tuple(sorted(number(v) for v in claim["value"]))
        except Exception:
            value = tuple(sorted(norm_text(v) for v in claim["value"]))
        values = (value, None, None)
    else:
        values = (norm_text(claim["value"]), None, None)
    return (norm_text(claim["subject"]), norm_text(claim["attribute"]), kind, values, canonical,
            norm_text(claim["condition"]), norm_text(claim["event"]), claim["unknown_scope"], claim["bound_inclusive"])


def citation_pairs(claims):
    return {(citation["source_id"], citation["chunk_id"]) for claim in claims for citation in claim["citations"]}


def strict_claim_match(predicted, expected):
    return (claim_signature(predicted) == claim_signature(expected) and
            bool(citation_pairs([predicted]) & citation_pairs([expected])))


def build():
    protocol, questions, kb, selections, generated, runtimes, hashes, replayed = verify_complete()
    references = load_references(protocol, questions, kb)
    selection_index = {(label, row["id"]): row for label, rows in selections.items() for row in rows}
    summaries, details = {}, []
    for arm, rows in generated.items():
        counts, by_source = Counter(), {}
        for row in rows:
            expected = references[row["id"]]["semantic_claims"]
            required = citation_pairs(expected)
            available = {(chunk["source_id"], chunk["chunk_id"]) for chunk in row["evidence"]["chunks"]}
            parsed = row["parsed"]
            claims = parsed["claims"] if parsed is not None else []
            matched = [any(strict_claim_match(c, e) for c in claims) for e in expected]
            flags = {"json_contract_valid": parsed is not None,
                     "all_required_chunks_available": required <= available,
                     "strict_reference_claims_recalled": all(matched),
                     "strict_reference_claim_set_equal": all(matched) and len(claims) == len(expected)}
            if arm != "RAG":
                selected = selection_index[arm, row["id"]]
                flags["exact_record_selection"] = selected["prediction"] is not None and {r["fact_id"] for r in selected["prediction"]} == set(references[row["id"]]["expected_fact_ids"])
                flags["finished_selection"] = selected["prediction"] is not None
            counts.update({key: int(value) for key, value in flags.items()})
            sources = sorted({source for source, _ in required})
            for source in sources:
                group = by_source.setdefault(source, Counter())
                group["questions"] += 1
                group.update({key: int(value) for key, value in flags.items()})
            details.append({"arm": arm, "id": row["id"], "source_ids": sources, "metrics": flags,
                            "parse_error": row["parse_error"], "semantic_answer_reviewed": False})
        summaries[arm] = {"questions": len(rows), "counts": dict(counts), "by_source": {k: dict(v) for k, v in by_source.items()},
                          "answer_runtime": runtimes[arm, "answer"],
                          "selection_runtime": runtimes.get((arm, "selection"))}
    hashes.update(protocol["inputs_sha256"])
    hashes.update(protocol["reference_sha256"])
    hashes[str(probe.OUT / "pipeline_completed.json")] = probe.sha(probe.OUT / "pipeline_completed.json")
    summary = {"version": "radar_sources_mechanical_analysis_v1", "scope": protocol["scope"], "runs": summaries,
               "selection_replay": {"rows": replayed, "mismatches": 0},
               "answer_input_and_parser_replay": {"rows": sum(len(r) for r in generated.values()), "mismatches": 0},
               "all_outputs_verified_before_reference_access": True,
               "natural_language_answer_accuracy_scored": False,
               "strict_match_limit": "Strict signatures normalize case/spacing and basic units, but not synonyms, paraphrases, alternative decompositions or source semantics. Non-match is not proof of a wrong natural answer.",
               "source_granularity_limit": "Whole-page retrieval can include the target despite wrong record selection; these are separate metrics.",
               "human_gold": False, "statistical_significance_tested": False, "inputs_sha256": hashes}
    return summary, {"version": summary["version"], "records": details}, questions, references, generated


def prepare_blind(questions, references, generated):
    """Hide arm labels and deduplicate identical question/evidence/answer triples."""
    by_q = {q["id"]: q for q in questions}
    unique, assignments = {}, []
    for arm, rows in generated.items():
        for row in rows:
            visible = {"question": by_q[row["id"]]["question"], "source_chunks": row["evidence"]["chunks"],
                       "generated_text": row["generated_text"], "parsed": row["parsed"],
                       "reference_claims": references[row["id"]]["semantic_claims"]}
            digest = hashlib.sha256(json.dumps(visible, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            blind_id = "answer-review-" + digest[:16]
            unique[blind_id] = {"blind_id": blind_id, **visible}
            assignments.append({"arm": arm, "id": row["id"], "blind_id": blind_id})
    rows = list(unique.values())
    random.Random(20261005).shuffle(rows)
    packet = {"version": "radar_sources_blind_ai_review_v1", "reviewer_kind": "AI", "human_gold": False,
              "arm_names_hidden": True, "deduplication": "Identical question, supplied evidence and generated answer reviewed once; map back to every original arm afterward",
              "rubric": {"answer_correct": "Natural answer directly answers all requested items and preserves source subject, quantity form, units and required conditions. Missing/incorrect answers are false.",
                         "answer_supported": "Every substantive statement in the natural answer is supported by the supplied chunks, not external memory. An appropriate lack-of-evidence statement can be supported yet not correct for an answerable question.",
                         "citations_support_answer": "The structured citations identify supplied chunks supporting the substantive natural answer; missing/invalid citations are false.",
                         "required_conditions_preserved": "True/false for a question requiring a condition; null if no required condition.",
                         "notes": "Give a concise source-grounded reason; distinguish unit conversion, discrete choices/ranges, peak/average, and evidence insufficiency."},
              "records": rows}
    return packet, {"version": packet["version"], "records": assignments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, detail, questions, references, generated = build()
    blind, key = prepare_blind(questions, references, generated)
    outputs = {probe.OUT / "mechanical_summary.json": summary, probe.OUT / "per_question_diagnosis.json": detail,
               probe.DATA / "blind_answer_review.json": blind, probe.DATA / "blind_answer_key.json": key}
    for path, value in outputs.items():
        if args.check:
            require(json.loads(path.read_text()) == value, f"Analysis drift: {path}")
        else:
            probe.write_new(path, value)
    print(json.dumps({"status": "verified" if args.check else "written", "selection_replay": summary["selection_replay"],
                      "answer_replay": summary["answer_input_and_parser_replay"], "blind_unique_answers": len(blind["records"])}))


if __name__ == "__main__":
    main()
