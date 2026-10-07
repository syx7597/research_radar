"""Bind every frozen prompt's token IDs before cross-version remote execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .coverage_runtime_probe import verify_tokenizer
from .source_readings import ROOT, sha, write_new


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def prepare(output, tokenizer_path):
    public = ROOT / "artifacts/thesis_direction_review/radar_representation_v1/semantic_preflight.json"
    source = ROOT / "data/radar_sources_v2/representation_v1/semantic_payload_v1/inputs.jsonl"
    preflight = json.loads(public.read_text())
    if sha(source) != preflight["local_inputs_sha256"]:
        raise ValueError("Frozen three-arm input file changed")
    output = Path(output).resolve()
    expected_root = ROOT / "data/radar_sources_v2/evaluation_preparation_v1"
    if output.exists() or not output.is_relative_to(expected_root):
        raise ValueError("Use a new local token-identity file")
    tokenizer = verify_tokenizer(tokenizer_path, {"tokenizer_sha256": preflight["tokenizer_files_sha256"]})
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    identities, seen = [], set()
    for row in rows:
        key = row["question_id"], row["arm"]
        if key in seen or row["arm"] not in ("raw", "flat", "bound") or row["truncated"] is not False:
            raise ValueError("Wrong question/arm roster or truncated input")
        seen.add(key)
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False,
                                            add_generation_prompt=True, truncation=False)
        if len(ids) != row["input_tokens"] or len(ids) + 768 > 32768:
            raise ValueError("Frozen prompt tokenization changed")
        identities.append({"question_id": row["question_id"], "arm": row["arm"],
                           "token_count": len(ids), "token_ids_sha256": token_sha(ids)})
    qids = {r["question_id"] for r in rows}
    if len(qids) != 96 or seen != {(q, a) for q in qids for a in ("raw", "flat", "bound")}:
        raise ValueError("Expected exactly the fixed 96 by three roster")
    write_new(output, identities)
    return {"prompt_count": len(identities), "path": str(output.relative_to(ROOT)), "sha256": sha(output),
            "input_sha256": sha(source), "code_sha256": sha(Path(__file__)),
            "model_runs": 0, "schema": "compact_ASCII_JSON_token_IDs_SHA256"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.tokenizer)))


if __name__ == "__main__":
    main()
