"""Offline, outcome-blind cohort construction for the fixed recovery diagnostic.

Inputs are fixed-I rollouts and references only. No A/C outcomes are accepted.
Every question gets a ledger entry. If either condition fails a token budget,
exclude the entire pair, never only the harder condition.
"""
from collections import Counter
from copy import deepcopy

from .same_state import build_states, token_check
from .same_state_inference import validate_inventory


def select_budgeted_pairs(states, audits, checks):
    if states:
        validate_inventory(states)
    ids = [row["id"] for row in audits]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A complete, unique source-question ledger is required")
    accepted = [row["id"] for row in audits if row["status"] == "accepted"]
    if [row["id"] for row in states[::2]] != accepted:
        raise ValueError("Replay-accepted states differ from the complete ordered ledger")
    if [(c["id"], c["condition"]) for c in checks] != [(r["id"], r["condition"]) for r in states]:
        raise ValueError("Token preflight must cover every state in the same order")
    for check in checks:
        if (type(check.get("within_budgets")) is not bool
                or any(type(check.get(k)) is not int or check[k] < 0 for k in (
                    "initial_prompt_tokens", "max_reference_prompt_tokens",
                    "reference_generated_tokens", "max_reference_turn_tokens"))
                or not isinstance(check.get("initial_prompt_ids_sha256"), str)
                or len(check["initial_prompt_ids_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in check["initial_prompt_ids_sha256"])):
            raise ValueError("Malformed token-budget evidence")
    # Exact combined per-turn context feasibility was computed by token_check;
    # these redundant limits also reject inconsistent passing summaries.
    for check in checks:
        if check["within_budgets"] and (check["initial_prompt_tokens"] >= 8192
                or check["max_reference_prompt_tokens"] >= 8192
                or check["reference_generated_tokens"] > 2048 or check["max_reference_turn_tokens"] > 192):
            raise ValueError("Passing token-budget flag contradicts recorded counters")
    selected, selected_checks, ledger = [], [], []
    pair_index = 0
    for audit in audits:
        if audit["status"] not in {"accepted", "excluded"}:
            raise ValueError("Unknown construction status")
        entry = {"id": audit["id"], "construction": deepcopy(audit)}
        if audit["status"] != "accepted":
            entry.update(eligible=False, reason=audit["reason"])
        else:
            start = 2 * pair_index
            pair_checks = checks[start:start + 2]
            entry["token_checks"] = deepcopy(pair_checks)
            passing = all(c["within_budgets"] for c in pair_checks)
            entry.update(eligible=passing, reason="accepted" if passing else "reference_token_budget")
            if passing:
                selected.extend(states[start:start + 2])
                selected_checks.extend(pair_checks)
            pair_index += 1
        ledger.append(entry)
    counts = dict(Counter(row["reason"] for row in ledger))
    if sum(counts.values()) != len(audits):
        raise AssertionError("Cohort exclusions do not partition the fixed pool")
    report = {"schema": "same_state_cohort_v1", "source_questions": len(audits),
              "replay_accepted_pairs": len(accepted), "eligible_pairs": len(selected) // 2,
              "counts": counts, "eligible_ids": [s["id"] for s in selected[::2]],
              "all_source_questions_accounted_for": True, "AC_predictions_used": False,
              "state_replay_passed": True, "token_budget_passed": True,
              "evaluation_possible": bool(selected),
              "zero_eligible_policy": "Report non-estimable; no automatic pool expansion",
              "initial_prompt_ids_sha256": {s["id"]: {
                  "clean": selected_checks[i]["initial_prompt_ids_sha256"],
                  "divergent": selected_checks[i+1]["initial_prompt_ids_sha256"]}
                  for i, s in enumerate(selected) if i % 2 == 0}}
    return selected, ledger, report


def construct_cohort(executor, tokenizer, rollouts, gold):
    """Call only after collection completion and original file hashes are checked."""
    states, audits, _ = build_states(executor, rollouts, gold)
    checks = token_check(executor, tokenizer, states, gold, rollouts)
    return select_budgeted_pairs(states, audits, checks)
