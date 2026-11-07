"""Verify saved hand fixtures, replayed pairs, and actual token-budget matching."""
import hashlib
import json
from pathlib import Path

from .table_program import TableExecutor
from .table_recovery_data import make_pair
from .table_training_data import balanced_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    summary_path = Path("results/radar_domain/recovery_contract_20261107/summary.json")
    summary = json.loads(summary_path.read_text())
    for path, expected in summary["inputs_sha256"].items():
        assert sha(path) == expected, path
    private = summary["private_fixtures"]
    assert sha(private["path"]) == private["sha256"]
    pairs = json.loads(Path(private["path"]).read_text())
    base = Path("data/radar_domain_recovery_v2/frozen_20261107/dev")
    refs = {r["id"]: r for r in (json.loads(s) for s in (base / "references.jsonl").read_text().splitlines())}
    executor = TableExecutor(base / "catalog.json")
    tokenizer_path = Path("cache/qwen2.5-3b-tokenizer")
    tokens = json.loads(Path("results/radar_domain/frozen_20261107/token_preflight.json").read_text())
    for name, expected in tokens["tokenizer_files_sha256"].items():
        assert sha(tokenizer_path / name) == expected
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    checks = []
    for item in pairs:
        assert not item["model_generated"] and not item["training_eligible"]
        ident, kind = item["reference_id"], item["fixture_kind"]
        new = make_pair(executor, refs[ident], item["rollout"])
        assert new == item["pair"], (ident, kind)
        clean, recovery = balanced_pair(tokenizer, new["clean"], new["recovery"])
        assert not clean["training_eligible"] and not recovery["training_eligible"]
        assert clean["supervised_tokens"] == recovery["supervised_tokens"]
        checks.append({"id": ident, "fixture_kind": kind,
            "calls_clean": new["clean"]["calls"], "calls_recovery": new["recovery"]["calls"],
            "context_clean": len(clean["input_ids"]), "context_recovery": len(recovery["input_ids"]),
            "matched_supervised_tokens": clean["supervised_tokens"],
            "masked_clean": clean["budget_masked_tokens"], "masked_recovery": recovery["budget_masked_tokens"]})
    assert checks == summary["checks"]
    assert len(checks) == summary["fixture_pairs"] == 24
    print(json.dumps({"status": "passed", "constructed_fixture_pairs": len(checks),
                      "real_model_failures": 0, "formal_recovery_pairs": 0,
                      "all_replay_and_token_counts_match_saved_report": True}))


if __name__ == "__main__":
    main()
