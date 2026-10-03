"""Freeze the agent split once and replay training programs as real tool calls.

Project holdout gold is copied, never executed or summarized during preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

from experiments.condition_consistency.prepare_data import EXPECTED_SHA256, make_splits, sha256
from experiments.condition_consistency.prepare_repair_splits import frozen_write, jsonl, require
from experiments.condition_consistency.executor import KoPLExecutor
from .environment import MAX_CALLS, SYSTEM_PROMPT, gold_trajectory

SEED = 20261003
DATA = Path("data/agent_feedback")
RESULTS = Path("results/agent_feedback")


def freeze():
    original = Path("results/condition_consistency/data_check/split_manifest.json")
    historical = Path("results/condition_consistency/query_repair/split_manifest.json")
    pilot = json.loads(original.read_text())
    old = json.loads(historical.read_text())
    for name, digest in EXPECTED_SHA256.items():
        require(sha256(Path("datasets/kqa_pro") / name) == digest, f"Source changed: {name}")
    rows = json.loads(Path("datasets/kqa_pro/train.json").read_text())
    val = json.loads(Path("datasets/kqa_pro/val.json").read_text())
    size = pilot["deduplication"]["eligible_component_representatives"]
    pool, dedup = make_splits(rows, val, pilot["seed"], {"all": size})
    used = {i for split in pilot["splits"].values() for i in split["indices"]}
    used.update(int(r.split(":")[1]) for split in old["splits"].values() for r in split["ids"])
    require(used <= set(pool["all"]), "Historical IDs are not component representatives")
    remaining = sorted(set(pool["all"]) - used)
    random.Random(SEED).shuffle(remaining)
    splits = {"train": pilot["splits"]["generator_train"]["indices"],
              "dev": pilot["splits"]["dev"]["indices"], "holdout": remaining[:2000]}
    require(len(splits["holdout"]) == 2000, "Insufficient unused components")
    manifest = {"seed": SEED, "source_sha256": EXPECTED_SHA256,
                "history": {str(original): sha256(original), str(historical): sha256(historical)},
                "excluded_historical_components": len(used), "deduplication": dedup,
                "holdout_usage": "frozen by ID only; gold copied, not executed, analyzed or scored",
                "validation_history": "official validation and old 2000 holdout already seen; reference only",
                "selection": "same exact question-or-program components as original; exclude all historical components, sorted seeded shuffle",
                "splits": {}}
    for name, ids in splits.items():
        gold = [{"id": f"train:{i}", "question": rows[i]["question"],
                 "program": rows[i]["program"], "answer": rows[i]["answer"]} for i in ids]
        qpath = DATA / f"{name}.questions.jsonl"
        gpath = DATA / f"{name}.gold.jsonl"
        frozen_write(qpath, jsonl([{k: row[k] for k in ("id", "question")} for row in gold]))
        frozen_write(gpath, jsonl(gold))
        manifest["splits"][name] = {"count": len(ids), "ids": [f"train:{i}" for i in ids],
                                    "questions": {"path": str(qpath), "sha256": sha256(qpath)},
                                    "gold": {"path": str(gpath), "sha256": sha256(gpath)}}
    frozen_write(RESULTS / "split_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    protocol = {
        "version": "agent_feedback_v1", "frozen_date": "2026-10-03", "seed": SEED,
        "model": "Qwen/Qwen2.5-3B-Instruct", "model_revision": "pin_download_before_training",
        "adapter": {"type": "LoRA", "rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "sft": {"learning_rate": 1e-4, "warmup_ratio": 0.03, "effective_batch_size": 16,
                "initial_epochs": 2, "continuation_epochs_equivalent": 1, "max_context": 8192,
                "supervision": "assistant action content and end token only; erroneous actions/observations masked",
                "comparison": "shared initial adapter; A/C same unique questions and supervised-token budget; input tokens/compute separately reported"},
        "arms": {"P": "same backbone one-shot program SFT, 3 epochs",
                 "A": "initial agent SFT + clean continuation",
                 "C": "initial agent SFT + mixed recovery continuation",
                 "B": "A + outcome-only GRPO", "D": "C + identical GRPO"},
        "recovery": {"target_supervised_fraction": 0.5, "prefer": "initial model real training errors",
                     "synthetic": "supplement only, report separately; do not infer error from empty result",
                     "quality": "verify executed suffix, dependencies and conditions; do not expose gold to inference"},
        "rl": {"reward": "official exact answer comparison, correct 1 else 0", "group_size": 4,
               "max_updates": 200, "learning_rate": 5e-6, "beta": 0.0, "num_iterations": 1,
               "max_calls": MAX_CALLS, "per_turn_max_new_tokens": 192, "total_generated_token_cap": 2048,
               "backend": "mature GRPO implementation; exact installed version recorded before run"},
        "inference": {"max_calls": MAX_CALLS, "max_context": 8192,
                      "per_turn_max_new_tokens": 192, "total_generated_token_cap": 2048,
                      "decoding": "greedy", "observations": "same bounded actual results for all agent arms"},
        "pilot_gpu_hours_cap": 72, "max_mechanism_revisions": 1,
        "go_rule": {"accuracy_gain_pp": 2.0, "or_accuracy_loss_at_most_pp": 1.0,
                    "and_total_inference_token_reduction_fraction": 0.20,
                    "interpretation": "development investment decision, not statistical significance"},
        "formal_evaluation": "freeze method, then same fixed holdout; two seeds for key configurations; official val reference only",
        "prompt": SYSTEM_PROMPT,
    }
    frozen_write(RESULTS / "protocol.json", json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"frozen": True, "excluded_historical": len(used),
                      "sizes": {k: len(v) for k, v in splits.items()}}), flush=True)


def replay():
    executor = KoPLExecutor("datasets/kqa_pro/kb.json")
    start = time.monotonic()
    report = {"purpose": "infrastructure and SFT data, not model accuracy", "splits": {}}
    for name in ("train", "dev"):
        rows = [json.loads(x) for x in (DATA / f"{name}.gold.jsonl").read_text().splitlines()]
        counts, mismatches, trajectories = Counter(), [], []
        for row in rows:
            trajectory = gold_trajectory(executor, row)
            direct = executor.execute(row["program"])
            counts["total"] += 1
            counts["step_vs_direct_agreement"] += trajectory["prediction"] == direct["prediction"]
            counts["correct"] += trajectory["correct"]
            if not trajectory["correct"]:
                mismatches.append({"id": row["id"], "prediction": trajectory["prediction"], "gold": str(row["answer"])})
            # Keep all rows in evaluation; do not train on a contradictory target.
            if name == "train" and trajectory["correct"]:
                trajectories.append(trajectory)
            if counts["total"] % 500 == 0:
                print(json.dumps({"split": name, **counts}), flush=True)
        require(counts["step_vs_direct_agreement"] == counts["total"], "Step execution differs from existing executor")
        report["splits"][name] = {**counts, "mismatches": mismatches}
        if name == "train":
            frozen_write(DATA / "train.success.jsonl", jsonl(trajectories))
            report["training_trajectories"] = {"count": len(trajectories), "sha256": sha256(DATA / "train.success.jsonl")}
    report["seconds"] = time.monotonic() - start
    report["environment_sha256"] = sha256(Path(__file__).with_name("environment.py"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gold_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "replay"])
    args = parser.parse_args()
    freeze() if args.command == "freeze" else replay()
