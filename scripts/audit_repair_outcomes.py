#!/usr/bin/env python3
"""Post-hoc audit of an already-scored, frozen query-repair experiment.

Never run this script on holdout before final scoring is authorized and complete.
It reads gold only after validating the final metrics/cache hashes. It does not
select candidates, tune policies, execute repairs, or change any existing file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.condition_consistency.executor import compare_answers, validate_program
from experiments.condition_consistency.repair import field_slots
from experiments.condition_consistency.evaluate_repairs import baseline_index, choose_repair, outcome


SUBSET_FUNCTIONS = {
    "qualifier": {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate", "QueryAttrQualifier",
                  "QueryRelationQualifier", "QueryAttrUnderCondition"},
    "numeric_or_selection": {"FilterNum", "QFilterNum", "VerifyNum", "SelectBetween", "SelectAmong"},
    "temporal": {"FilterYear", "FilterDate", "QFilterYear", "QFilterDate", "VerifyYear", "VerifyDate"},
    "set_operations": {"And", "Or"},
    "count": {"Count"},
}
SUBSET_LABELS = {"qualifier": "限定符", "numeric_or_selection": "数值或极值选择",
                 "temporal": "时间条件", "set_operations": "集合操作", "count": "计数",
                 "two_or_more_relate": "至少两步 Relate"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["id"])
        if qid in result:
            raise ValueError(f"Duplicate ID in {path}: {qid}")
        result[qid] = row
    if not result:
        raise ValueError(f"Empty input: {path}")
    return result


def normalized(program):
    if not isinstance(program, list):
        return None
    try:
        return validate_program(program)
    except (ValueError, TypeError, KeyError):
        return None


def exact_program(candidate, gold_program):
    program = normalized(candidate.get("program")) if candidate else None
    return bool(program is not None and program == normalized(gold_program))


def checked_candidate(row, index):
    if index is None:
        return None
    if type(index) is not int or not 0 <= index < len(row["candidates"]):
        raise ValueError(f"Invalid candidate index for {row['id']}: {index}")
    return row["candidates"][index]


def correct(candidate, answer):
    return bool(candidate and candidate["valid"] and compare_answers(answer, candidate["prediction"]))


def verified_edit(row, candidate):
    """Check the actual programs, not just the proposal's preserved-field claim."""
    if candidate.get("origin") != "repair":
        raise ValueError(f"An outcome-changing candidate is not a repair: {row['id']}")
    parent_index = candidate["parent_index"]
    parent = checked_candidate(row, parent_index)
    if parent.get("origin") != "original":
        raise ValueError(f"Repair parent is not an original: {row['id']}")
    before, after = normalized(parent.get("program")), normalized(candidate.get("program"))
    if before is None or after is None or len(before) != len(after):
        raise ValueError(f"Repair altered structure or is unparseable: {row['id']}")
    diffs = []
    for step, (old, new) in enumerate(zip(before, after)):
        if old["function"] != new["function"] or old["dependencies"] != new["dependencies"] or len(old["inputs"]) != len(new["inputs"]):
            raise ValueError(f"Repair altered functions/dependencies: {row['id']}")
        for arg, (left, right) in enumerate(zip(old["inputs"], new["inputs"])):
            if left != right:
                diffs.append((step, arg, left, right))
    edit = candidate["edit"]
    expected = (edit["step"], edit["input"], edit["from"], edit["to"])
    if diffs != [expected]:
        raise ValueError(f"Repair is not the claimed single edit: {row['id']}")
    slots = field_slots(before, edit["step"])
    if not any(slot["input"] == edit["input"] and slot["role"] == edit["role"] for slot in slots):
        raise ValueError(f"Changed argument is not the declared schema field: {row['id']}")
    return {**edit, "function": before[edit["step"]]["function"]}


def clipped(value, limit=120):
    if value is None:
        return None
    value = str(value)
    return value if len(value) <= limit else value[:limit - 1] + "…"


def parent_position(parent, baseline):
    return "no_baseline" if baseline is None else "earlier" if parent < baseline else "later" if parent > baseline else "same"


def transition_profile(cases):
    return {change: {
        "questions": len(group := [case for case in cases if case["change"] == change]),
        "baseline_schema_clean": dict(Counter("no_baseline" if c["first_valid_index"] is None else "clean" if c["baseline_schema_clean"] else "conflicted" for c in group)),
        "parent_relative_to_first_valid": dict(Counter(c["parent_relative_to_first_valid"] for c in group)),
        "parent_indices": dict(Counter(c["parent_index"] for c in group)),
        "clean_baseline_replaced_by_other_parent_ids": [c["id"] for c in group if c["baseline_schema_clean"] and not c["parent_is_first_valid"]],
    } for change in ("corrected", "regressed")}


def subset_names(program):
    functions = [step["function"] for step in program]
    groups = [name for name, members in SUBSET_FUNCTIONS.items() if set(functions) & members]
    if functions.count("Relate") >= 2:
        groups.append("two_or_more_relate")
    return groups


def audit(metrics, cache, gold):
    metric_rows = {str(row["id"]): row for row in metrics["per_question"]}
    if len(metric_rows) != len(metrics["per_question"]) or set(metric_rows) != set(cache) or set(cache) != set(gold):
        raise ValueError("Final metrics, local cache and gold IDs must match exactly")
    if metrics["questions"] != len(cache):
        raise ValueError("Final metric denominator differs")
    cases, switched_same_answer_status = [], []
    subset_ids = {**{name: [] for name in SUBSET_FUNCTIONS}, "two_or_more_relate": []}
    all_local_exact, all_baseline_exact, accepted_count = 0, 0, 0
    for qid, detail in metric_rows.items():
        row, target = cache[qid], gold[qid]
        if row["question"] != target["question"]:
            raise ValueError(f"Question mismatch: {qid}")
        if "answer" in row or "program" in row:
            raise ValueError("Scored repair cache unexpectedly contains gold metadata")
        baseline_idx, selected_idx = detail["selected"]["first_valid"], detail["selected"]["local_repair"]
        baseline, selected = checked_candidate(row, baseline_idx), checked_candidate(row, selected_idx)
        expected_baseline = next((i for i, c in enumerate(row["candidates"]) if c.get("origin") == "original" and c["valid"]), None)
        if baseline_idx != expected_baseline:
            raise ValueError(f"Stored first-valid index does not match cache: {qid}")
        before, after = correct(baseline, target["answer"]), correct(selected, target["answer"])
        if before != detail["correct"]["first_valid"] or after != detail["correct"]["local_repair"]:
            raise ValueError(f"Recomputed outcome differs from frozen final metrics: {qid}")
        for name in subset_names(target["program"]):
            subset_ids[name].append(qid)
        selected_exact, baseline_exact = exact_program(selected, target["program"]), exact_program(baseline, target["program"])
        all_local_exact += selected_exact
        all_baseline_exact += baseline_exact
        if selected_idx != baseline_idx:
            accepted_count += 1
            actual_edit = verified_edit(row, selected)
            if before == after:
                switched_same_answer_status.append({"id": qid, "correct_before_and_after": before,
                                                    "selected_program_exact_gold": selected_exact})
        if before == after:
            continue
        gold_functions = sorted({step["function"] for step in target["program"]})
        originals_have_correct = any(correct(c, target["answer"]) for c in row["candidates"] if c.get("origin") == "original")
        cases.append({
            "id": qid, "change": "corrected" if after else "regressed",
            "first_valid_index": baseline_idx, "selected_index": selected_idx,
            "parent_index": selected["parent_index"], "parent_is_first_valid": selected["parent_index"] == baseline_idx,
            "parent_relative_to_first_valid": parent_position(selected["parent_index"], baseline_idx),
            "baseline_schema_clean": baseline.get("schema_clean") if baseline else None,
            "edit": actual_edit, "single_field_edit_verified": True,
            "trigger": selected["trigger"], "schema_clean": selected.get("schema_clean"),
            "match_score": selected["match_score"], "tf_mean_logprob": selected.get("tf_mean_logprob"),
            "gold_program_exact_match": selected_exact, "baseline_program_exact_gold": baseline_exact,
            "gold_function_set": gold_functions, "posthoc_gold_subsets": subset_names(target["program"]),
            "original_pool_has_correct_answer": originals_have_correct,
            "baseline_valid": bool(baseline and baseline["valid"]),
            "baseline_empty_result": baseline.get("empty_result") if baseline else None,
            "selected_empty_result": selected.get("empty_result"),
            "baseline_prediction_excerpt": clipped(baseline.get("prediction")) if baseline else None,
            "selected_prediction_excerpt": clipped(selected.get("prediction")),
            "gold_answer_excerpt": clipped(target["answer"]),
            "context": {key: selected.get("context", {}).get(key) for key in
                        ("scope", "trusted", "semantic_correctness_guaranteed", "prefix_steps",
                         "entity_counts", "attached_fact_counts", "compatible_field_count", "qualifier_fact_binding")},
        })
    counts = Counter(case["change"] for case in cases)
    paired = metrics["comparisons"]["local_repair_vs_first_valid"]
    if counts["corrected"] != paired["corrected"] or counts["regressed"] != paired["regressed"]:
        raise ValueError("Audit transitions differ from frozen paired-comparison counts")
    subsets = {}
    for name, ids in subset_ids.items():
        methods = {}
        for method in metrics["metrics"]:
            total = sum(metric_rows[qid]["correct"][method] for qid in ids)
            methods[method] = {"correct": total, "questions": len(ids), "accuracy": total / len(ids) if ids else None}
        subsets[name] = {"questions": len(ids), "metrics": methods,
                         "local_corrected_vs_first_valid": sum(not metric_rows[qid]["correct"]["first_valid"] and metric_rows[qid]["correct"]["local_repair"] for qid in ids),
                         "local_regressed_vs_first_valid": sum(metric_rows[qid]["correct"]["first_valid"] and not metric_rows[qid]["correct"]["local_repair"] for qid in ids)}
    return {
        "purpose": "Post-hoc mechanical audit of frozen results, not independent human semantic annotation",
        "split": metrics["split"], "questions": len(cache), "metrics": metrics["metrics"],
        "local_vs_first_valid": paired,
        "program_match_definition": "Exact functions, normalized/inferred dependencies and input strings; alternate correct programs may differ",
        "semantic_warning": "Answer equality does not imply semantic correctness. Gold-program mismatch does not prove semantic incorrectness. Gold labels may be inconsistent.",
        "method_changed": False, "labels_changed": False, "cases_selected": "All corrected/regressed local vs first-valid questions, no case sampling",
        "accepted_repairs": accepted_count,
        "accepted_with_unchanged_correctness": switched_same_answer_status,
        "exact_gold_program_counts": {"first_valid": all_baseline_exact, "local_repair": all_local_exact,
                                      "corrected_selected_exact_gold": sum(c["change"] == "corrected" and c["gold_program_exact_match"] for c in cases),
                                      "regressed_selected_exact_gold": sum(c["change"] == "regressed" and c["gold_program_exact_match"] for c in cases)},
        "corrected_without_correct_original_candidate": sum(c["change"] == "corrected" and not c["original_pool_has_correct_answer"] for c in cases),
        "transition_counts_by_edit_role": {change: dict(Counter(c["edit"]["role"] for c in cases if c["change"] == change)) for change in ("corrected", "regressed")},
        "transition_counts_by_trigger": {change: dict(Counter(c["trigger"] for c in cases if c["change"] == change)) for change in ("corrected", "regressed")},
        "transition_profile": transition_profile(cases),
        "posthoc_subsets_warning": "Overlapping groups defined using gold functions after final scoring; descriptive diagnostics only, not preregistered subgroup efficacy claims",
        "posthoc_subset_definitions": {**{name: sorted(functions) for name, functions in SUBSET_FUNCTIONS.items()}, "two_or_more_relate": "At least two Relate steps in the gold program"},
        "posthoc_subsets": subsets, "cases": cases,
    }


def development_diagnostics(run_dir, split_dir):
    """Replay only the already-frozen v1 selector; never evaluate a new gate."""
    policy_path = run_dir / "frozen_policy.json"
    policy = json.loads(policy_path.read_text())
    result = {"warning": "Post-hoc diagnosis with the unchanged frozen v1 policy, not evaluation of a new baseline-protection rule", "policy_sha256": sha256(policy_path), "splits": {}}
    for split, prefix in (("dev_diagnostic", "dev"), ("calibration", "calibration")):
        cache_path, gold_path = run_dir / f"{prefix}_local_scored.jsonl", split_dir / f"{split}.gold.jsonl"
        cache, gold = records(cache_path), records(gold_path)
        if set(cache) != set(gold):
            raise ValueError(f"Development diagnostic ID mismatch: {split}")
        if split == "calibration" and (sha256(gold_path) != policy["calibration_gold_sha256"] or sha256(cache_path) != policy["calibration"]["local"]["input_sha256"]):
            raise ValueError("Calibration inputs differ from frozen policy")
        cases, baseline_correct, local_correct = [], 0, 0
        for qid, row in cache.items():
            if row["question"] != gold[qid]["question"]:
                raise ValueError(f"Development question mismatch: {qid}")
            baseline = baseline_index(row)
            selected = choose_repair(row, policy["policies"]["local"])
            before, after = outcome(row, baseline, gold[qid]["answer"]), outcome(row, selected, gold[qid]["answer"])
            baseline_correct += before
            local_correct += after
            if before == after:
                continue
            parent = row["candidates"][selected]["parent_index"]
            cases.append({"id": qid, "change": "corrected" if after else "regressed",
                          "first_valid_index": baseline, "parent_index": parent,
                          "parent_is_first_valid": parent == baseline,
                          "parent_relative_to_first_valid": parent_position(parent, baseline),
                          "baseline_schema_clean": row["candidates"][baseline]["schema_clean"] if baseline is not None else None})
        result["splits"][split] = {"questions": len(cache), "baseline_correct": baseline_correct, "local_correct": local_correct,
                                   "transition_profile": transition_profile(cases), "cases": cases,
                                   "cache_sha256": sha256(cache_path), "gold_sha256": sha256(gold_path)}
    return result


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def markdown(result):
    metrics, pair = result["metrics"], result["local_vs_first_valid"]
    lines = ["# 查询修复结果案例审计", "",
             f"对象：已冻结评分的 `{result['split']}`，共 {result['questions']} 题。此报告由脚本机械核对全部纠正和改错案例，**不是人工语义评审，也不是新一轮算法实验**。没有改动方法、阈值、标签或选取评测子集。", "",
             "## 整体转换与程序匹配", "",
             f"first_valid 正确 {metrics['first_valid']['correct']} 题，局部修复正确 {metrics['local_repair']['correct']} 题；纠正 {pair['corrected']} 题、改错 {pair['regressed']} 题，净变化 {pair['net_corrected']:+d} 题。共接受 {result['accepted_repairs']} 次修复，其中 {len(result['accepted_with_unchanged_correctness'])} 次未改变答案正确与否。", "",
             f"纠正案例中，{result['exact_gold_program_counts']['corrected_selected_exact_gold']} 个被选程序与规范化金标程序完全相同；{result['corrected_without_correct_original_candidate']} 个案例的原四候选均无正确答案。程序匹配仅比较函数、依赖和参数，不做语义等价证明。", "",
             "所有纠正和改错均实际核验了父程序到修复程序的单字段限制。基线是否干净、父候选相对顺序等汇总见结构化结果的 `transition_profile`。", "",
             "**答案相同不保证程序语义正确；程序与金标不同也不必然错误。** 数据金标可能有问题。本报告不从少量成功案例推导整个方法的语义可靠性，仍需独立人工抽查。单字段限制是相对于修复的父候选；父候选未必等于 first_valid，因此最终输出与基线的差异可能不止一个字段。", "",
             "## 全部纠正与改错案例", "",
             "只公开 ID、必要字段片段和机械诊断，不公开完整题目或程序。表中“精确金标”是规范依赖后完整程序一致，“原池正确”表示原四候选已含正确答案。", "",
             "| ID | 转换 | 编辑字段 | 触发 | 父候选为基线 | 基线schema干净 | 精确金标 | 原池正确 |",
             "|---|---|---|---|---|---|---|---|"]
    for case in result["cases"]:
        edit = case["edit"]
        change = "纠正" if case["change"] == "corrected" else "改错"
        trigger = "角色非法" if case["trigger"] == "globally_illegal_role_field" else "显式类型不符"
        description = f"{edit['role']} / {edit['function']}: `{md_cell(edit['from'])}` → `{md_cell(edit['to'])}`"
        yes = lambda value: "是" if value else "否"
        clean = "无输出" if case["first_valid_index"] is None else yes(case["baseline_schema_clean"])
        lines.append(f"| {case['id']} | {change} | {description} | {trigger} | {yes(case['parent_is_first_valid'])} | {clean} | {yes(case['gold_program_exact_match'])} | {yes(case['original_pool_has_correct_answer'])} |")
    if not result["cases"]:
        lines.append("| — | 无答案正确性变化 | — | — | — | — | — | — |")
    lines.extend(["", "## 按金标函数划分的事后诊断", "",
                  "以下类别相互重叠，只用于解释结果，不能相加，也不是预注册的子群有效性结论。数值组包括数量过滤/验证及极值选择，时间组单独列出；限定符组可与两者重叠。", "",
                  "| 子集 | 题数 | first_valid 正确 | 全局修复正确 | 局部修复正确 | beam8 正确 | 局部纠正 / 改错 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, subset in result["posthoc_subsets"].items():
        sm = subset["metrics"]
        lines.append(f"| {SUBSET_LABELS[name]} | {subset['questions']} | {sm['first_valid']['correct']} | {sm.get('global_repair', {}).get('correct', '—')} | {sm['local_repair']['correct']} | {sm.get('beam8_first_valid', {}).get('correct', '—')} | {subset['local_corrected_vs_first_valid']} / {subset['local_regressed_vs_first_valid']} |")
    if "development_diagnostics" in result:
        lines.extend(["", "## 旧开发与校准数据上的同类现象", "",
                      "下表仍使用已经冻结的 v1 选择策略，只统计其行为，**没有运行新的基线保护规则，也没有重新调参**。机制建议已受到当前检查集的观察启发，当前检查集此后必须标记为已见。", "",
                      "| 集合 | first_valid / 局部正确 | 纠正 / 改错 | 纠正中基线干净 | 改错中基线干净 |",
                      "|---|---|---|---|---|"])
        for name, diag in result["development_diagnostics"]["splits"].items():
            profile = diag["transition_profile"]
            lines.append(f"| {name} | {diag['baseline_correct']} / {diag['local_correct']} | {profile['corrected']['questions']} / {profile['regressed']['questions']} | {profile['corrected']['baseline_schema_clean'].get('clean', 0)} | {profile['regressed']['baseline_schema_clean'].get('clean', 0)} |")
    lines.extend(["", "## 复现与来源", "",
                  "脚本：[audit_repair_outcomes.py](../../scripts/audit_repair_outcomes.py)。结构化结果：[outcome_audit.json](../../results/condition_consistency/query_repair/outcome_audit.json)。脚本先核对冻结评测、缓存和金标 SHA256/ID/题目，再重算已选答案正确性；不重新选择候选。", "",
                  "完整输入及脚本哈希、逐例编辑核查和子集定义见结构化结果。原数据来源和许可见 [DATA.md](../../experiments/condition_consistency/DATA.md)。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/condition_consistency/query_repair/holdout_metrics.json"))
    parser.add_argument("--cache", type=Path, default=Path("results/condition_consistency/query_repair/holdout_local_scored.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/condition_consistency/repair_splits/holdout.gold.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/condition_consistency/query_repair/outcome_audit.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/research/QUERY_REPAIR_CASE_AUDIT.md"))
    parser.add_argument("--development-diagnostics", action="store_true", help="Also inspect unchanged v1 decisions on old dev/calibration")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate this descriptive audit only; frozen input metrics stay untouched")
    args = parser.parse_args()
    if (args.output.exists() or args.report.exists()) and not args.overwrite:
        raise ValueError("Refusing to overwrite a previous audit/report; choose new paths")
    # Read final metrics first. Before final scoring there is no valid entry point.
    metrics = json.loads(args.metrics.read_text())
    if metrics["input_sha256"]["local"] != sha256(args.cache) or metrics["gold_sha256"] != sha256(args.gold):
        raise ValueError("Cache/gold hashes do not match the frozen final metrics")
    result = audit(metrics, records(args.cache), records(args.gold))
    if args.development_diagnostics:
        result["development_diagnostics"] = development_diagnostics(args.cache.parent, args.gold.parent)
    result["source_sha256"] = {"metrics": sha256(args.metrics), "local_scored_cache": sha256(args.cache),
                               "gold": sha256(args.gold), "audit_script": sha256(Path(__file__)),
                               "executor_adapter": sha256(ROOT / "experiments/condition_consistency/executor.py"),
                               "repair_source": sha256(ROOT / "experiments/condition_consistency/repair.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.report.write_text(markdown(result))
    print(json.dumps({"split": result["split"], "questions": result["questions"],
                      "cases": len(result["cases"]), "local_vs_first_valid": result["local_vs_first_valid"],
                      "exact_gold_program_counts": result["exact_gold_program_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
