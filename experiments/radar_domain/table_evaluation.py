"""Offline replay/scoring and paired comparison of complete typed-table runs."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random

from .table_inference import BUDGET, SPLIT, SUPPLEMENT, load_partition, rows, read, encode_prompt, trim_tokens
from .table_runtime_executor import TableRuntimeExecutor as TableExecutor
from .table_program import digest
from .table_recovery_data import replay
from .table_runtime import sha, verify_files
from .table_scoring import CatalogIndex, score_answer


def validate_token_accounting(row, question, tokenizer=None):
    turns = row["turn_tokens"]
    if len(turns) != row["calls"] or row["budgets"] != BUDGET:
        raise ValueError("Call/token inventory differs from the registered budget")
    generated, inputs = 0, 0
    for i, item in enumerate(turns):
        counted = item["generated_token_ids"]
        if any(type(t) is not int or t < 0 for t in counted):
            raise ValueError("Actual generated token IDs required")
        if item["generated_tokens"] != len(counted):
            raise ValueError("Generated token count differs from IDs")
        allowance = min(BUDGET["per_turn_generated_tokens"], BUDGET["total_generated_tokens"]-generated,
                        BUDGET["context_tokens"]-item["input_tokens"])
        if allowance != item["granted_tokens"] or not 0 <= len(counted) <= allowance:
            raise ValueError("Generation exceeded its allowance")
        if tokenizer is not None:
            prompt = encode_prompt(tokenizer, row["messages"][:2+2*i])
            after = encode_prompt(tokenizer, row["messages"][:4+2*i])
            if len(prompt) != item["input_tokens"] or len(after) != item["context_tokens_after_observation"]:
                raise ValueError("Full-history token count mismatch")
            trimmed, content = trim_tokens(counted, tokenizer)
            if trimmed != counted or tokenizer.decode(content, skip_special_tokens=False) != row["messages"][2+2*i]["content"]:
                raise ValueError("Generated text differs from its recorded actual token IDs")
        if i < len(turns)-1 and item["context_tokens_after_observation"] >= BUDGET["context_tokens"]:
            raise ValueError("Generation continued after full history exhausted context")
        inputs += item["input_tokens"]; generated += len(counted)
    if (inputs != row["input_tokens"] or generated != row["generated_tokens"]
            or inputs+generated != row["total_tokens"] or generated > BUDGET["total_generated_tokens"]):
        raise ValueError("Cumulative token totals differ")
    if row["stop_reason"] not in {"finished", "call_budget", "context_budget", "generation_budget"}:
        raise ValueError("Unexpected/incomplete run stop reason")
    if row["stop_reason"] == "finished" and row["answer"] is None:
        raise ValueError("Finished marker lacks submitted answer")
    if row["answer"] is not None and row["stop_reason"] != "finished":
        raise ValueError("Submitted answer has a contradictory stop reason")
    if row["stop_reason"] == "call_budget" and row["calls"] != BUDGET["max_calls"]:
        raise ValueError("Premature call-budget termination")
    if row["stop_reason"] == "generation_budget" and generated != BUDGET["total_generated_tokens"]:
        raise ValueError("Premature generation-budget termination")
    if tokenizer is not None and row["stop_reason"] == "context_budget":
        if len(encode_prompt(tokenizer, row["messages"])) < BUDGET["context_tokens"]:
            raise ValueError("Premature context-budget termination")


def population(split, meta):
    if split == "dev":
        return "synthetic_development" if meta["origin"] == "synthetic" else "real_development"
    return {"train": "synthetic_train", "holdout": "new_source_basic_24",
            "supplement": "new_source_composition_12"}[split]


def aggregate(scored):
    if not scored:
        raise ValueError("Empty complete population")
    n = len(scored)
    return {"questions": n, "correct": sum(r["correct"] for r in scored),
            "accuracy": sum(r["correct"] for r in scored)/n,
            "completed": sum(r["completed"] for r in scored),
            "well_formed": sum(r["well_formed"] for r in scored),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in scored)),
            "totals": {k: sum(r[k] for r in scored) for k in
                       ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens")}}


def evaluate_records(predictions, references, metadata, catalog, split, *, tokenizer=None,
                     expected_origin="cpu_fixture", generation_id=None):
    ids = [r["id"] for r in references]
    if len(ids) != len(set(ids)) or [r["id"] for r in predictions] != ids or [r["id"] for r in metadata] != ids:
        raise ValueError("Require complete frozen IDs in exact order, including failed episodes")
    executor, index, scored = TableExecutor(catalog), CatalogIndex(catalog), []
    for row, ref, meta in zip(predictions, references, metadata):
        if row["provenance"] != {"kind": expected_origin, "generation_id": generation_id}:
            raise ValueError("Rollout origin/generation identity differs")
        episode = replay(executor, ref, row, BUDGET["max_calls"])
        validate_token_accounting(row, ref["question"], tokenizer)
        if row["invalid_calls"] != sum(not e["observation"]["ok"] for e in episode.events):
            raise ValueError("Invalid-call count differs from actual replay")
        if row["selected_handle"] != episode.selected:
            raise ValueError("Submitted handle differs from replay")
        judgment = score_answer(episode.answer, ref["answer"], index, meta["answer_spec"])
        scored.append({"id": row["id"], **judgment, "completed": episode.answer is not None,
            "population": population(split, meta), "source_group": meta.get("source_group", meta.get("family_id")),
            "task_type": meta["task_type"], "stop_reason": row["stop_reason"],
            **{k: row[k] for k in ("calls", "invalid_calls", "input_tokens", "generated_tokens", "total_tokens")}})
    return scored


def score_manifest(generation_path, *, tokenizer, allow_cpu_fixtures=False):
    generation = read(generation_path)
    origin = generation.get("origin")
    if tokenizer is None:
        raise ValueError("Offline evaluation must verify actual full-history tokenization")
    if (generation.get("schema") != "radar_table_rollout_manifest_v1" or generation.get("status") != "completed"
            or origin not in ({"model_generated", "cpu_fixture"} if allow_cpu_fixtures else {"model_generated"})):
        raise ValueError("A complete admitted generation manifest is required")
    if origin == "model_generated" and (not generation.get("model_gpu_smoke", {}).get("finite_logits")
                                         or not generation.get("model_files_sha256")):
        raise ValueError("Real model identity and completed model-level smoke required")
    if origin == "model_generated":
        budget_completion = generation["gpu_budget_completion"]
        from .table_job_budget import require_round_ledger
        require_round_ledger(budget_completion["ledger_path"])
        ledger = read(budget_completion["ledger_path"])
        jobs = [j for j in ledger["jobs"] if j["job_id"] == generation["generation_id"]]
        if (len(jobs) != 1 or jobs[0]["status"] != "completed" or budget_completion["status"] != "completed"
                or budget_completion["job_id"] != generation["generation_id"] or digest(jobs[0]) != budget_completion["job_sha256"]):
            raise ValueError("Model generation and GPU budget terminal states disagree")
    split = generation["split"]
    parent = generation.get("parent_split_manifest", str(SPLIT))
    selected = generation["split_manifest"]
    data = load_partition(parent if split == "supplement" else selected, split,
                          selected if split == "supplement" else SUPPLEMENT)
    if data["manifest_sha256"] != generation["split_sha256"]:
        raise ValueError("Frozen split identity differs")
    desc = data["descriptor"]
    for name in ("questions", "references", "metadata", "catalog"):
        if generation[name] != desc[name]:
            raise ValueError("Generation source binding changed: " + name)
        verify_files({desc[name]["path"]: desc[name]["sha256"]})
    if generation["implementation_sha256"].get("experiments/radar_domain/table_program.py") != sha("experiments/radar_domain/table_program.py"):
        raise ValueError("Replay executor identity differs")
    runtime_adapter = "experiments/radar_domain/table_runtime_executor.py"
    if generation["implementation_sha256"].get(runtime_adapter) != sha(runtime_adapter):
        raise ValueError("Runtime error-feedback adapter identity differs")
    spec = generation["rollouts"]
    verify_files({spec["path"]: spec["sha256"]})
    predictions = rows(spec["path"])
    if spec["rows"] != len(predictions) or len(predictions) != len(data["questions"]):
        raise ValueError("Incomplete generation coverage")
    scored = evaluate_records(predictions, rows(desc["references"]["path"]), rows(desc["metadata"]["path"]),
                              data["catalog"], split, tokenizer=tokenizer, expected_origin=origin,
                              generation_id=generation["generation_id"])
    grouped = {}
    for field in ("population", "source_group", "task_type"):
        groups = defaultdict(list)
        for row in scored:
            groups[row[field]].append(row)
        grouped["by_" + field] = {k: aggregate(v) for k, v in sorted(groups.items())}
    summary = {"schema": "radar_table_model_evaluation_v1", "status": "complete_replay_and_score",
        "split": split, "origin": origin, "model_performance_result": origin == "model_generated",
        "generation": {"path": str(generation_path), "sha256": sha(generation_path)},
        "split_sha256": generation["split_sha256"], "catalog_semantic_sha256": CatalogIndex(data["catalog"]).sha256,
        "overall": aggregate(scored), **grouped, "failed_episodes_retained": True,
        "reference_answers_exposed_to_generation": False, "token_accounting_verified": True,
        "budgets": BUDGET, "decoding": generation.get("decoding"), "model_role": generation.get("collector_role"),
        "batch_size": generation.get("batch_size"), "seed": generation.get("seed"),
        "implementation_sha256": generation["implementation_sha256"],
        "model_files_sha256": generation.get("model_files_sha256", {}),
        "base_model_identity": generation.get("base_model_identity"),
        "runtime_versions": generation.get("runtime_versions"),
        "collector_training_result": generation.get("collector_training_result"),
        "overall_role": "Development mixture inventory; use the two separate population reports" if split == "dev" else "Complete fixed split",
        "scorer_sha256": sha(__file__), "answer_scoring_sha256": sha("experiments/radar_domain/table_scoring.py")}
    return summary, scored


def paired_counts(a, c):
    if [r["id"] for r in a] != [r["id"] for r in c] or len({r["id"] for r in a}) != len(a) or not a:
        raise ValueError("Paired comparison requires identical complete ordered IDs")
    corrected = sum(not x["correct"] and y["correct"] for x, y in zip(a, c))
    regressed = sum(x["correct"] and not y["correct"] for x, y in zip(a, c))
    return {"questions": len(a), "A_correct": sum(r["correct"] for r in a), "C_correct": sum(r["correct"] for r in c),
            "corrected": corrected, "regressed": regressed, "net": corrected-regressed,
            "delta_percentage_points": 100*(corrected-regressed)/len(a)}


def paired_bootstrap(a, c, *, replicates=5000, seed=20261107):
    paired_counts(a, c)  # Validate complete pairing before sampling.
    differences = [int(y["correct"])-int(x["correct"]) for x, y in zip(a, c)]
    rng, n = random.Random(seed), len(differences)
    samples = sorted(100 * sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(replicates))
    def percentile(probability):
        position = (len(samples)-1)*probability
        index = int(position)
        fraction = position-index
        return samples[index]*(1-fraction)+samples[min(index+1,len(samples)-1)]*fraction
    return {"method": "paired question percentile bootstrap", "replicates": replicates, "seed": seed,
            "confidence_level": .95, "delta_percentage_points": 100*sum(differences)/n,
            "ci_percentage_points": [percentile(.025), percentile(.975)],
            "scope": "Conditional on these fixed model checkpoints and the 24-question collection; source-family dependence remains and is reported separately"}


def compare(a_manifest, c_manifest, *, tokenizer, allow_cpu_fixtures=False):
    a, ar = score_manifest(a_manifest, tokenizer=tokenizer, allow_cpu_fixtures=allow_cpu_fixtures)
    c, cr = score_manifest(c_manifest, tokenizer=tokenizer, allow_cpu_fixtures=allow_cpu_fixtures)
    for key in ("split", "split_sha256", "catalog_semantic_sha256", "budgets", "decoding", "origin",
                "batch_size", "seed", "implementation_sha256"):
        if a[key] != c[key]:
            raise ValueError("Unmatched comparison setting: " + key)
    if a["origin"] == "model_generated" and (a["model_role"], c["model_role"]) != ("A_domain", "C_domain"):
        raise ValueError("Primary comparison must identify ordinary A and recovery C")
    if a["origin"] == "model_generated":
        for key in ("base_model_identity", "runtime_versions"):
            if not a[key] or a[key] != c[key]:
                raise ValueError("Unmatched base or runtime identity: " + key)
        training = []
        for arm in (a, c):
            bound = arm["collector_training_result"]
            verify_files({bound["path"]: bound["sha256"]})
            training.append(read(bound["path"]))
        for key in ("preparation", "base_and_starting_model_files_sha256", "paired_settings", "seed",
                    "actual_supervised_tokens", "actual_sampled_order_sha256", "training_environment", "global_steps"):
            if key not in training[0] or training[0][key] != training[1].get(key):
                raise ValueError("A/C training was not matched: " + key)
        settings = training[0]["paired_settings"]
        verify_files({settings["path"]: settings["sha256"]})
        settings_doc = read(settings["path"])
        if (settings_doc.get("schema") != "radar_table_paired_training_settings_v1"
                or settings_doc["preparation"] != training[0]["preparation"] or not settings_doc.get("common_config")
                or settings_doc.get("training_environment") != training[0]["training_environment"]):
            raise ValueError("Paired training settings are unbound or incomplete")
    groups = {}
    for field in ("population", "source_group", "task_type"):
        keys = sorted({r[field] for r in ar})
        groups["by_" + field] = {k: paired_counts([r for r in ar if r[field] == k], [r for r in cr if r[field] == k]) for k in keys}
    result = {"schema": "radar_table_paired_evaluation_v1", "split": a["split"],
        "model_performance_result": a["model_performance_result"], "A_generation": a["generation"],
        "C_generation": c["generation"], "paired": paired_counts(ar, cr), **groups,
        "source_relation": "Basic24 and composition12 are separate populations sharing four source families; never pooled as independent sources",
        "uncertainty_scope": "Exact paired and source-family counts; source correlation retained in interpretation"}
    if a["split"] == "holdout":
        parent = read(SPLIT)
        contract = parent["task_contract"]
        verify_files({contract["path"]: contract["sha256"]})
        stats = read(contract["path"])["statistics"]
        if len(ar) != 24:
            raise ValueError("The registered primary bootstrap population has 24 questions")
        result["registered_primary_bootstrap"] = paired_bootstrap(ar, cr,
            replicates=stats["paired_question_bootstrap_replicates"], seed=stats["seed"])
        result["statistics_contract"] = contract
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default="cache/qwen2.5-3b-tokenizer")
    sub = p.add_subparsers(dest="command", required=True)
    one = sub.add_parser("score"); one.add_argument("--generation", required=True)
    one.add_argument("--summary", required=True); one.add_argument("--scored-rows", required=True)
    pair = sub.add_parser("compare"); pair.add_argument("--a", required=True); pair.add_argument("--c", required=True)
    pair.add_argument("--summary", required=True)
    args = p.parse_args()
    destination = Path(args.summary)
    if destination.exists():
        raise FileExistsError("Never overwrite an evaluation report")
    expected = read("results/radar_domain/frozen_20261107/token_preflight.json")["tokenizer_files_sha256"]
    verify_files({str(Path(args.tokenizer)/n): h for n, h in expected.items()})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.command == "score":
        private = Path(args.scored_rows)
        if private.exists() or not private.resolve().is_relative_to(Path("data").resolve()):
            raise ValueError("Use a new private data path for per-question scored rows")
        report, scored = score_manifest(args.generation, tokenizer=tokenizer)
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in scored))
        report["private_scored_rows"] = {"path": str(private), "sha256": sha(private)}
    else:
        report = compare(args.a, args.c, tokenizer=tokenizer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": "completed", "summary": str(destination)}))


if __name__ == "__main__":
    main()
