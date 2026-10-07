"""One registered semantic-payload context check, without model execution.

The complete archive remains in a bound sidecar. The two structured LLM views
are lossless over E_sem only; original quotation text is not secretly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .coverage_preflight import LOCAL, compact, inputs, messages, tokenizer, MAX_GENERATED, SYSTEM_PROMPT
from .coverage_representation import encode_flat, encode_bound, validate_equal_information
from .source_readings import ROOT, sha, write_new

REVISION = ROOT / "artifacts/thesis_direction_review/radar_representation_v1/input_contract_revision.json"


def context_choice(lengths, original=8192, native=32768, generated=768):
    if not lengths:
        raise ValueError("No prompt measurements")
    maximum = max(lengths) + generated
    return original if maximum <= original else native if maximum <= native else None


def build(output, tokenizer_path):
    from .coverage_payload import project_records, restore_records
    revision = json.loads(REVISION.read_text())
    if revision["new_question_outputs_seen"] or revision["model_run_ready"]:
        raise ValueError("Expected a pre-output candidate revision")
    for rel, expected in revision["inputs_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Registered revision input changed")
    questions, retrieved, records, chunks, bindings = inputs()
    tok, token_hashes = tokenizer(tokenizer_path)
    native = json.loads((tokenizer_path / "config.json").read_text())["max_position_embeddings"]
    if native != revision["context_rule_before_measurement"]["model_native_max_context"]:
        raise ValueError("Native context identity changed")
    if output.exists() or not output.resolve().is_relative_to(LOCAL):
        raise ValueError("Use a new local output directory")
    all_records = list(records.values())
    semantic, sidecar = project_records(all_records)
    expected_semantic = [{**r, "citation": {"chunk_id": r["citation"]["chunk_id"]}} for r in all_records]
    validate_equal_information(expected_semantic, encode_flat(semantic), encode_bound(semantic))
    reconstructed = restore_records(semantic, sidecar)
    validate_equal_information(all_records, encode_flat(reconstructed), encode_bound(reconstructed))
    sem = {r["record_id"]: r for r in semantic}
    if list(sem) != list(records) or len(semantic) != len(all_records):
        raise ValueError("Projection changed corpus membership/order")
    output.mkdir(parents=True)
    write_new(output / "semantic_corpus_and_audit.json", {"semantic_records": semantic, "audit_sidecar": sidecar})
    old_path = LOCAL / "full_record_v1/preflight.json"
    old = {r["question_id"]: r for r in json.loads(old_path.read_text())["rows"]}
    rows = []
    with (output / "inputs.jsonl").open("x") as stream:
        for q, found in zip(questions, retrieved):
            if q["question_id"] != found["question_id"]:
                raise ValueError("Question order changed")
            selected = [sem[r] for r in found["all_reading_ids"]]
            flat, bound = encode_flat(selected), encode_bound(selected)
            validate_equal_information(selected, flat, bound)
            raw = {"chunks": [{k: chunks[cid][k] for k in ("source_id", "chunk_id", "title", "text")}
                              for cid in found["retrieved_chunks"]]}
            lengths = {}
            for arm, supplied in (("raw", raw), ("flat", json.loads(flat)), ("bound", json.loads(bound))):
                msg = messages(q["question"], supplied)
                length = len(tok.apply_chat_template(msg, tokenize=True, return_dict=False, add_generation_prompt=True))
                lengths[arm] = length
                stream.write(compact({"question_id": q["question_id"], "arm": arm, "messages": msg,
                                      "input_tokens": length, "truncated": False}) + "\n")
            if lengths["raw"] != old[q["question_id"]]["input_tokens"]["raw"]:
                raise ValueError("Common prompt or raw evidence changed")
            rows.append({"question_id": q["question_id"], "record_count": len(selected),
                         "record_ids": found["all_reading_ids"], "input_tokens": lengths,
                         "semantic_roundtrip": True, "selected_evidence_sha256": hashlib.sha256(compact(selected).encode()).hexdigest()})
    cap = context_choice([n for r in rows for n in r["input_tokens"].values()], native=native)
    summary = {a: {"min_input_tokens": min(r["input_tokens"][a] for r in rows),
                   "max_input_tokens": max(r["input_tokens"][a] for r in rows),
                   "total_input_tokens": sum(r["input_tokens"][a] for r in rows),
                   "overflow_at_8192": sum(r["input_tokens"][a] + MAX_GENERATED > 8192 for r in rows),
                   "overflow_at_native_context": sum(r["input_tokens"][a] + MAX_GENERATED > native for r in rows)}
               for a in ("raw", "flat", "bound")}
    for p in [REVISION, old_path, *[Path(__file__).with_name(n) for n in
              ("coverage_payload.py", "coverage_representation.py", "coverage_preflight.py", "coverage_semantic_preflight.py")]]:
        bindings[str(p.relative_to(ROOT))] = sha(p)
    result = {"schema": "radar_semantic_payload_context_preflight_v1", "inputs_sha256": bindings,
              "question_count": len(rows), "corpus_records": len(semantic),
              "semantic_roundtrips": len(rows) * 2, "archive_roundtrip_records_with_sidecar": len(reconstructed),
              "retained_fields_verified_over_all_source_records": len(all_records),
              "represented_record_occurrences": sum(r["record_count"] for r in rows),
              "selected_common_max_context": cap, "reserved_generated_tokens": MAX_GENERATED,
              "native_model_max_context": native, "fits_registered_candidate": cap is not None,
              "arms": summary, "tokenizer_files_sha256": token_hashes,
              "model_config_sha256": sha(tokenizer_path / "config.json"),
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "local_inputs_sha256": sha(output / "inputs.jsonl"),
              "local_semantic_corpus_and_audit_sha256": sha(output / "semantic_corpus_and_audit.json"),
              "raw_inputs_unchanged_from_full_record_check": True, "rows": rows,
              "model_run_ready": False, "new_model_runs": 0, "new_GPU_hours": 0,
              "next_action": "old_development_prompt_memory_latency_probe_then_freeze" if cap else "stop_representation_candidate",
              "limits": "E_sem excludes quotation text/byte spans/hash; only both structured arms share identical declared semantic fields. Sidecar reconstruction is not LLM access to the full original archive."}
    write_new(output / "preflight.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--tokenizer", type=Path, required=True)
    a = p.parse_args(); d = build(a.output, a.tokenizer)
    print(json.dumps({"questions": d["question_count"], "roundtrips": d["semantic_roundtrips"],
                      "arms": d["arms"], "selected_common_max_context": d["selected_common_max_context"],
                      "next_action": d["next_action"], "model_run_ready": False}))


if __name__ == "__main__":
    main()
