"""Exercise the new runner/scorer with scripted real-development references.

No weights or model are loaded. Reference-playing controls remain cpu_fixture
and never become model failures, new evaluation predictions or training data.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.agent_feedback.recovery import Excluded

from .table_inference import load_partition, run_episode
from .table_evaluation import evaluate_records
from .table_runtime_executor import TableRuntimeExecutor as TableExecutor
from .table_recovery_data import action_text
from .table_runtime import sha, verify_files


class ReferenceControl:
    def __init__(self, tokenizer, actions):
        self.tokenizer, self.actions, self.index = tokenizer, actions, 0

    def generate(self, prompt_ids, max_new_tokens):
        if self.index >= len(self.actions):
            raise ValueError("Reference control exhausted before submission")
        tokens = self.tokenizer.encode(action_text(self.actions[self.index]), add_special_tokens=False)
        tokens += [self.tokenizer.eos_token_id]
        if len(tokens) > max_new_tokens:
            raise ValueError("Reference control does not fit the fixed per-turn budget")
        self.index += 1
        return tokens


def run(tokenizer_dir):
    import transformers, tokenizers
    from transformers import AutoTokenizer
    report_path = Path("results/radar_domain/frozen_20261107/token_preflight.json")
    identity = json.loads(report_path.read_text())
    if (transformers.__version__, tokenizers.__version__) != ("5.18.0", "0.23.2"):
        raise ValueError("Use the pinned tokenizer-only CPU environment")
    verify_files({str(Path(tokenizer_dir)/name): h for name, h in identity["tokenizer_files_sha256"].items()})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    part = load_partition(split="dev")
    ref_path, meta_path = [Path(part["descriptor"][k]["path"]) for k in ("references", "metadata")]
    refs = [json.loads(s) for s in ref_path.read_text().splitlines() if json.loads(s)["id"].startswith("RDR-DEV-")]
    metadata = [json.loads(s) for s in meta_path.read_text().splitlines() if json.loads(s)["id"].startswith("RDR-DEV-")]
    assert len(refs) == len(metadata) == 12
    executor = TableExecutor(part["catalog"])
    generation_id = "cpu_reference_control_20261108"
    predictions = [run_episode({"id": ref["id"], "question": ref["question"]}, executor, tokenizer,
                    ReferenceControl(tokenizer, ref["actions"]), generation_id=generation_id) for ref in refs]
    scored = evaluate_records(predictions, refs, metadata, part["catalog"], "dev", tokenizer=tokenizer,
                              expected_origin="cpu_fixture", generation_id=generation_id)
    if not all(r["correct"] for r in scored):
        raise ValueError("Runner changed reference-controlled execution")
    changed = deepcopy(predictions)
    changed[0]["messages"][3]["content"] = '{"ok":false,"detail":"tampered observation"}'
    try:
        evaluate_records(changed, refs, metadata, part["catalog"], "dev", tokenizer=tokenizer,
                         expected_origin="cpu_fixture", generation_id=generation_id)
    except (ValueError, RuntimeError, Excluded) as exc:
        tamper_rejected = type(exc).__name__
    else:
        raise ValueError("Scorer accepted changed execution feedback")
    try:
        evaluate_records(predictions[:-1], refs, metadata, part["catalog"], "dev", tokenizer=tokenizer,
                         expected_origin="cpu_fixture", generation_id=generation_id)
    except ValueError:
        incomplete_rejected = True
    else:
        raise ValueError("Scorer accepted a missing question")
    paths = [Path(__file__), ref_path, meta_path, report_path,
        *[Path("experiments/radar_domain")/(name+".py") for name in (
            "table_inference", "table_evaluation", "table_program", "table_recovery_data", "table_scoring", "table_runtime", "table_runtime_executor")]]
    result = {"schema": "radar_table_runner_cpu_smoke_v1", "date": "2026-11-08", "status": "passed",
        "origin": "scripted_reference_control", "questions": 12, "model_runs": 0, "model_weights_loaded": False,
        "server_accessed": False, "training_eligible": False,
        "reference_control_outputs_preserved": 12, "full_token_accounting_recomputed": True,
        "changed_observation_rejected": tamper_rejected, "incomplete_denominator_rejected": incomplete_rejected,
        "cases": [{"id": r["id"], "calls": r["calls"], "input_tokens": r["input_tokens"],
                   "generated_tokens": r["generated_tokens"], "stop_reason": r["stop_reason"]} for r in predictions],
        "code_and_input_sha256": {str(p): sha(p) for p in paths}}
    return result, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default="cache/qwen2.5-3b-tokenizer")
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    if (args.private_output.exists() or args.summary.exists()
            or not args.private_output.resolve().is_relative_to(Path("data").resolve())):
        raise ValueError("Use new private control output and summary paths")
    result, predictions = run(args.tokenizer)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in predictions))
    result["private_controls"] = {"path": str(args.private_output), "sha256": sha(args.private_output)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": "passed", "reference_controls": 12, "model_runs": 0}))


if __name__ == "__main__":
    main()
