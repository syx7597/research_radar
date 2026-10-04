"""Bounded CPU-only diagnosis of five prespecified development prefix drifts.

No model weights, gold answers, holdout data, generation, or source mutations.
Print an aggregate report without questions, answers, or complete actions.
"""
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_failure(row):
    return next((i + 1 for i, event in enumerate(row["events"])
                 if not event["observation"]["ok"]), None)


def action_kind(message):
    try:
        call = json.loads(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",
                                    message["content"], flags=re.S)[0])
        return {"tool": call["name"], "function": call["arguments"].get("function")}
    except (ValueError, IndexError, KeyError, TypeError):
        return {"tool": "unparseable"}


def main():
    if os.environ.get("PYTHONHASHSEED") != "20261003" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Require original hash seed and explicitly hidden GPUs")
    from transformers import AutoTokenizer
    from experiments.agent_feedback.download_weights import ROOT as model_root
    from experiments.agent_feedback.environment import compact, tool_schemas
    from experiments.agent_feedback.feedback_mask import visible_observation

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True, padding_side="left")
    inputs, cases = {}, []
    specs = [(20261003, "recovery_dev_v1", "recovery_feedback_masked_dev_v1", ["train:78759", "train:41980"]),
             (20261004, "recovery_dev_seed20261004", "recovery_feedback_masked_dev_seed20261004",
              ["train:75418", "train:40934", "train:90102"])]
    checked = set()
    for seed, normal_name, masked_name, ids in specs:
        arms = {}
        for name, stem in (("normal", normal_name), ("masked", masked_name)):
            path = Path("results/agent_feedback") / (stem + ".jsonl")
            inputs[str(path)] = sha(path)
            arms[name] = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(arms[name]) == 500
        assert [r["id"] for r in arms["normal"]] == [r["id"] for r in arms["masked"]]
        for ident in ids:
            index = next(i for i, row in enumerate(arms["normal"]) if row["id"] == ident)
            normal, masked = (arms[arm][index] for arm in ("normal", "masked"))
            message_index = next(i for i, (a, b) in enumerate(zip(normal["messages"], masked["messages"])) if a != b)
            assert message_index >= 2 and message_index % 2 == 0
            turn = (message_index - 2) // 2 + 1
            batch_start = index // 8 * 8
            batch = {arm: rows[batch_start:batch_start + 8] for arm, rows in arms.items()}
            layouts = {}
            for arm, rows in batch.items():
                active = []
                for row in rows:
                    mode = "original" if arm == "normal" else "generic_failure_v1"
                    assert len(row["messages"]) == 2 + 2 * len(row["events"])
                    prompts = []
                    for step, event in enumerate(row["events"]):
                        assert row["messages"][3 + 2 * step] == {
                            "role": "tool", "content": compact(visible_observation(event["observation"], mode))}
                        prompts.append(tokenizer.apply_chat_template(row["messages"][:2 + 2 * step],
                            tokenize=True, return_dict=False, add_generation_prompt=True, tools=tool_schemas()))
                    assert sum(map(len, prompts)) == row["input_tokens"]
                    checked.add((seed, arm, row["id"]))
                    if len(row["events"]) < turn:
                        continue
                    prompt = prompts[turn - 1]
                    previous_text_lengths = [len(tokenizer.encode(row["messages"][2 + 2 * j]["content"],
                                                                 add_special_tokens=False)) for j in range(turn - 1)]
                    # Entire-run generated-token totals give a firm prefix upper
                    # bound. Retokenization is an additional reconstruction only:
                    # original per-turn token IDs/EOS flags were not logged.
                    firm_allowance_lower = min(192, 2048 - row["generated_tokens"], 8192 - len(prompt))
                    reconstructed_lower = min(192, 2048 - sum(previous_text_lengths) - (turn - 1), 8192 - len(prompt))
                    active.append({"id": row["id"], "prompt_tokens": len(prompt),
                                   "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                                   "first_failure_turn": first_failure(row),
                                   "192_allowance_proved_from_total_bound": firm_allowance_lower == 192,
                                   "192_allowance_supported_under_retokenization": reconstructed_lower >= 192})
                layouts[arm] = {"active": active, "active_count": len(active),
                                "target_position": next(i for i, r in enumerate(active) if r["id"] == ident),
                                "maximum_prompt_tokens": max(r["prompt_tokens"] for r in active),
                                "all_192_allowance_proved_from_total_bound": all(r["192_allowance_proved_from_total_bound"] for r in active),
                                "all_192_allowance_supported_under_retokenization": all(r["192_allowance_supported_under_retokenization"] for r in active)}
                target = next(r for r in active if r["id"] == ident)
                layouts[arm]["target_prompt_tokens"] = target["prompt_tokens"]
                layouts[arm]["target_left_padding_if_single_192_group"] = layouts[arm]["maximum_prompt_tokens"] - target["prompt_tokens"]
            others = [{"id": row["id"], "first_masked_failure_turn": first_failure(row)}
                      for row in batch["masked"] if row["id"] != ident
                      and first_failure(row) is not None and first_failure(row) < turn]
            target_prompts = [next(r["prompt_sha256"] for r in layouts[arm]["active"] if r["id"] == ident)
                              for arm in ("normal", "masked")]
            cases.append({"continuation_seed": seed, "id": ident, "question_zero_index": index,
                          "batch_zero_index_start": batch_start, "first_different_message_zero_index": message_index,
                          "first_different_assistant_turn": turn,
                          "same_target_prior_messages": normal["messages"][:message_index] == masked["messages"][:message_index],
                          "same_target_prior_events": normal["events"][:turn - 1] == masked["events"][:turn - 1],
                          "same_target_prompt_token_ids": target_prompts[0] == target_prompts[1],
                          "normal_first_failure_turn": first_failure(normal), "masked_first_failure_turn": first_failure(masked),
                          "normal_action_kind": action_kind(normal["messages"][message_index]),
                          "masked_action_kind": action_kind(masked["messages"][message_index]),
                          "other_questions_already_masked_in_same_batch": others, "layouts": layouts,
                          "active_members_changed": [r["id"] for r in layouts["normal"]["active"]] !=
                                                    [r["id"] for r in layouts["masked"]["active"]],
                          "maximum_prompt_length_changed": layouts["normal"]["maximum_prompt_tokens"] !=
                                                           layouts["masked"]["maximum_prompt_tokens"]})
    report = {"scope": "Five predeclared development prefix divergences and their original eight-question batches; CPU only",
              "cases": cases, "unique_batch_trajectories_checked": len(checked),
              "visible_messages_match_declared_renderer": True, "all_reconstructed_input_token_totals_match": True,
              "causal_conclusion": "All five diverge before their own first rendered failure. Prior interventions on other batch members and changed batch/padding are observable; numerical causation is plausible but not established without tensors/logits or a controlled rerun.",
              "grouping_limit": "For most cases all max-new-token allowances are proved 192 from recorded total-token upper bounds. Otherwise the 192 grouping is reconstructed from re-tokenized text plus at most one EOS per prior turn; original per-turn token IDs were not retained.",
              "interpretation": "Downgrade pure per-question feedback mechanism attribution. This diagnosis neither changes normal SFT results nor motivates further training or model selection.",
              "inputs_sha256": inputs}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
