"""CPU-only source-packet/tokenizer/weight checks before a bounded evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from . import source_probe as probe
from . import source_rag as rag
from .development_analysis import require


def fingerprint(path):
    s = Path(path).stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def hash_base(protocol):
    found = {}
    for item in protocol["base_weights"]["files"]:
        path = Path(protocol["model"]) / item["file"]
        before = fingerprint(path)
        require(before["bytes"] == item["bytes"] and probe.sha(path) == item["sha256"], "Base weight identity changed")
        require(fingerprint(path) == before, "Base weights changed during hashing")
        found[str(path)] = {**before, "sha256": item["sha256"]}
    return found


def verify_live_base(protocol):
    attestation = json.loads((probe.OUT / "preflight.json").read_text())
    require(attestation["status"] == "passed" and attestation["base_weights"] == protocol["base_weights"], "Wrong base attestation")
    probe.previous_training.hash_mapping(attestation["base_config_sha256"])
    for name, bound in attestation["base_files"].items():
        require(fingerprint(name) == {k: bound[k] for k in ("bytes", "mtime_ns", "ctime_ns")}, "Attested base changed since CPU preflight")


def main():
    out = probe.OUT / "preflight.json"
    if out.exists():
        raise FileExistsError("Preflight already frozen; do not replace it")
    old = json.loads(probe.previous_training.PROTOCOL.read_text())
    inputs = [*probe.CODE, str(probe.DATA / "questions.jsonl"), str(probe.DATA / "kb.json"),
              str(probe.DATA / "chunks.jsonl"), str(probe.ARTIFACTS / "manifest.json")]
    hashes = {name: probe.sha(name) for name in inputs}
    model = old["model"]
    require(probe.previous_training.tokenizer_identity(model) == old["tokenizer"]["files_sha256"], "Tokenizer/template file roster changed")
    for name, digest in old["tokenizer"]["files_sha256"].items():
        require(probe.sha(Path(model) / name) == digest, "Pinned tokenizer drift")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    index = rag.BM25Index.from_path(probe.DATA / "chunks.jsonl")
    questions = probe.questions()
    # Token lengths are checked without references and without any model forward pass.
    longest = sorted(index.chunks, key=lambda chunk: len(tokenizer.encode(json.dumps({k: chunk[k] for k in rag.EVIDENCE_KEYS}, ensure_ascii=False))), reverse=True)[:rag.TOP_K]
    max_len = 0
    for q in questions:
        variants = [rag.evidence_for_retrieval(index.retrieve(q["question"])),
                    rag.evidence_for_retrieval(longest), {"chunks": []}]
        for evidence in variants:
            length = len(tokenizer.apply_chat_template(rag.answer_messages(q["question"], evidence), tokenize=True,
                                                      return_dict=False, add_generation_prompt=True))
            max_len = max(max_len, length)
    # Keep generous room for concatenation boundary variation in other chunk combinations.
    require(max_len + 768 + 64 <= 8192, "Four whole-page chunks exceed the unchanged answer budget")
    base_protocol = {"model": model, "base_weights": old["base_weights"]}
    base_files = hash_base(base_protocol)
    config_hashes = {str(Path(model) / name): probe.sha(Path(model) / name)
                     for name in ("config.json", "generation_config.json") if (Path(model) / name).exists()}
    require(str(Path(model) / "config.json") in config_hashes, "Base config is missing")
    probe.previous_training.hash_mapping(hashes)
    result = {"version": "radar_sources_cpu_preflight_v1", "status": "passed", "cpu_only": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "questions": len(questions),
              "chunks": len(index.chunks), "max_checked_answer_prompt_tokens": max_len,
              "max_generated_tokens": 768, "max_context_tokens": 8192, "no_truncation": True,
              "base_weights": old["base_weights"], "base_files": base_files,
              "base_config_sha256": config_hashes,
              "inputs_sha256": hashes}
    probe.write_new(out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
