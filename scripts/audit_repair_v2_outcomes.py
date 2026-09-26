#!/usr/bin/env python3
"""Post-hoc audit of the complete, already-scored official validation run.

Run only after final v2 official-val scoring is complete and inspection is
authorized. The v1 audit is reused via an in-memory method-name mapping; no v1
code, cached predictions, frozen metrics, policy or dataset label is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_repair_outcomes as v1_audit


ALIASES = {"v2_local_repair": "local_repair", "v2_global_repair": "global_repair"}
ROLE_ORDER = ("attribute", "relation", "qualifier")


def alias_key(key):
    for original, alias in ALIASES.items():
        key = key.replace(original, alias)
    return key


def adapt_metrics(raw):
    """Use only a deep copy; preserve original disk hashes and original labels."""
    if raw.get("version") != "query_repair_v2":
        raise ValueError("Expected a frozen v2 report")
    result = copy.deepcopy(raw)
    result["metrics"] = {alias_key(key): value for key, value in raw["metrics"].items()}
    result["comparisons"] = {alias_key(key): value for key, value in raw["comparisons"].items()}
    for row in result["per_question"]:
        for field in ("correct", "selected"):
            row[field] = {alias_key(key): value for key, value in row[field].items()}
    return result


def representative_corrections(cases, limit=10):
    """Fixed role round-robin, retaining original metric order inside each role."""
    groups = {role: [case for case in cases if case["change"] == "corrected" and case["edit"]["role"] == role]
              for role in ROLE_ORDER}
    selected = []
    while len(selected) < limit and any(groups.values()):
        for role in ROLE_ORDER:
            if groups[role] and len(selected) < limit:
                selected.append(groups[role].pop(0))
    return selected


def verify_guard(raw, cache):
    retained = 0
    retained_wrong = 0
    for row in raw["per_question"]:
        qid = str(row["id"])
        baseline = row["selected"]["first_valid"]
        candidate = v1_audit.checked_candidate(cache[qid], baseline)
        guarded = bool(candidate and candidate["schema_clean"] is True)
        if row["guard_retained"] != guarded:
            raise ValueError(f"Guard indicator differs from cache: {qid}")
        if guarded and row["selected"]["v2_local_repair"] != baseline:
            raise ValueError(f"v2 local selection violated the frozen guard: {qid}")
        retained += guarded
        retained_wrong += guarded and not row["correct"]["first_valid"]
    if retained != raw["guard"]["retained_questions"]:
        raise ValueError("Guard total differs from frozen metrics")
    return {"rule": raw["guard"]["rule"], "retained_questions": retained,
            "all_guarded_local_indices_equal_baseline": True,
            "retained_wrong_answers": retained_wrong,
            "warning": "Schema-clean does not mean semantically correct; the guard can preserve wrong original answers"}


def case_table(cases):
    lines = ["| ID | 编辑字段 | 触发 | 父排名 / 基线排名 | 基线schema干净 | 修复精确金标 | 基线精确金标 |",
             "|---|---|---|---|---|---|---|"]
    for case in cases:
        edit = case["edit"]
        field = f"{edit['role']} / {edit['function']}: `{v1_audit.md_cell(edit['from'])}` → `{v1_audit.md_cell(edit['to'])}`"
        trigger = "角色非法" if case["trigger"] == "globally_illegal_role_field" else "显式类型不符"
        base_index = "无" if case["first_valid_index"] is None else str(case["first_valid_index"])
        clean = "无输出" if case["first_valid_index"] is None else "是" if case["baseline_schema_clean"] else "否"
        match = "是" if case["gold_program_exact_match"] else "否"
        baseline_match = "是" if case["baseline_program_exact_gold"] else "否"
        lines.append(f"| {case['id']} | {field} | {trigger} | {case['parent_index']} / {base_index} | {clean} | {match} | {baseline_match} |")
    if not cases:
        lines.append("| — | 无此类案例 | — | — | — | — | — |")
    return lines


def markdown(result):
    metrics = result["metrics"]
    pair = result["local_vs_first_valid"]
    exact = result["exact_gold_program_counts"]
    corrected = [case for case in result["cases"] if case["change"] == "corrected"]
    regressed = [case for case in result["cases"] if case["change"] == "regressed"]
    examples = representative_corrections(result["cases"])
    lines = ["# v2 官方验证集修复案例审计", "",
             f"对象：完整 KQA Pro 官方验证集 {result['questions']} 题，已经完成冻结策略评分。本报告是**事后机械诊断，不是独立人工语义标注，也没有进行新一轮调参**。v2 策略冻结后才运行本次评测，但早期项目曾使用官方 val，不能把它宣称为项目从未接触的隐藏测试集。", "",
             "## 答案转换与程序匹配", "",
             f"first_valid 正确 {metrics['first_valid']['correct']} 题，v2 局部修复正确 {metrics['local_repair']['correct']} 题。纠正 {pair['corrected']} 题、改错 {pair['regressed']} 题，净变化 {pair['net_corrected']:+d} 题。共接受 {result['accepted_repairs']} 次修复，{len(result['accepted_with_unchanged_correctness'])} 次未改变答案正确与否。", "",
             f"纠正中的 {exact['corrected_selected_exact_gold']} 个修复程序与规范化金标完全一致；{result['corrected_without_correct_original_candidate']} 个纠正案例原四候选均无正确答案。所有变化案例均核对了相对父候选的真实单字段修改。", "",
             "规范化仅统一推断出的依赖与程序结构，不判断语义等价。**答案相同不保证程序语义正确；程序不等于金标也不必然错误。** 单字段保护相对于父候选，父候选可能不同于 first_valid；这不是最终答案只发生单一语义变化的保证。", "",
             f"保护规则保留了 {result['guard_verification']['retained_questions']} 个已有干净基线，逐题确认均未被替换；其中 {result['guard_verification']['retained_wrong_answers']} 个基线答案错误，说明 schema 干净只是一项保守门控证据。", "",
             "| 编辑角色 | 纠正 | 改错 |", "|---|---:|---:|"]
    for role in ROLE_ORDER:
        lines.append(f"| {role} | {sum(case['edit']['role'] == role for case in corrected)} | {sum(case['edit']['role'] == role for case in regressed)} |")
    lines.extend(["", "## 全部改错案例", "",
                  "以下列出全部回退，不挑选或省略失败案例。排名从 0 开始，缺少基线显示为“无”。"])
    lines.extend(["", *case_table(regressed), "", "## 最多十个纠正示例", "",
                  "按属性、关系、限定符轮流取例，每类保持冻结评测中的原顺序，最多十例；不按匹配程度、答案好看与否或人工偏好选取。示例分布不代表总体比例，全部纠正 ID 和机械诊断保存在 JSON。", ""])
    lines.extend(case_table(examples))
    lines.extend(["", "## 金标函数子集的事后统计", "",
                  "类别相互重叠，只用于诊断，不得相加或当作预注册的子群优越性结论。数值组包含数量过滤/验证及极值选择，时间组另列。", "",
                  "| 子集 | 题数 | first_valid 正确 | v2全局正确 | v2局部正确 | beam8 正确 | 局部纠正 / 改错 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, subset in result["posthoc_subsets"].items():
        sm = subset["metrics"]
        lines.append(f"| {v1_audit.SUBSET_LABELS[name]} | {subset['questions']} | {sm['first_valid']['correct']} | {sm['global_repair']['correct']} | {sm['local_repair']['correct']} | {sm['beam8_first_valid']['correct']} | {subset['local_corrected_vs_first_valid']} / {subset['local_regressed_vs_first_valid']} |")
    lines.extend(["", "## 来源与复现边界", "",
                  "脚本：[audit_repair_v2_outcomes.py](../../scripts/audit_repair_v2_outcomes.py)。结构化结果：[v2/outcome_audit.json](../../results/condition_consistency/query_repair/v2/outcome_audit.json)。", "",
                  "脚本读取最终评分后，先核验原始 metrics、金标、缓存与冻结策略的哈希；仅在内存将 v2 方法名映射到既有审计函数需要的别名。v1/v2 指标文件、金标、阈值及算法文件均未修改。JSON 的 `method_aliases` 明确说明 `local_repair/global_repair` 在本审计中分别代表 v2 局部/全局方法。", "",
                  "所有案例、输入和源码哈希及子集定义见结构化结果。原数据来源与许可见 [DATA.md](../../experiments/condition_consistency/DATA.md)。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run = Path("results/condition_consistency/query_repair/v2")
    parser.add_argument("--metrics", type=Path, default=run / "official_val_metrics.json")
    parser.add_argument("--cache", type=Path, default=run / "val_local_scored.jsonl")
    parser.add_argument("--gold", type=Path, default=Path("data/condition_consistency/repair_splits/official_val.gold.jsonl"))
    parser.add_argument("--policy", type=Path, default=run / "frozen_policy.json")
    parser.add_argument("--output", type=Path, default=run / "outcome_audit.json")
    parser.add_argument("--report", type=Path, default=Path("docs/research/QUERY_REPAIR_V2_CASE_AUDIT.md"))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("Refusing to overwrite an existing v2 audit/report")
    raw_bytes = args.metrics.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("split") != "official_val" or raw.get("questions") != 11797:
        raise ValueError("Expected final scoring of the complete 11797-question official validation set")
    if (raw["input_sha256"]["local"] != v1_audit.sha256(args.cache)
            or raw["gold_sha256"] != v1_audit.sha256(args.gold)
            or raw["policy_sha256"] != v1_audit.sha256(args.policy)):
        raise ValueError("Audit input hashes differ from the frozen original metrics")
    cache, gold = v1_audit.records(args.cache), v1_audit.records(args.gold)
    result = v1_audit.audit(adapt_metrics(raw), cache, gold)
    result["version"] = "query_repair_v2_posthoc_audit"
    result["method_aliases"] = {alias: original for original, alias in ALIASES.items()}
    result["original_split_role"] = raw["split_role"]
    result["guard_verification"] = verify_guard(raw, cache)
    result["representative_corrected_ids"] = [case["id"] for case in representative_corrections(result["cases"])]
    result["representative_selection"] = "At most 10 corrections; role round-robin attribute/relation/qualifier, preserving original metric order within each role"
    result["source_sha256"] = {"original_metrics": v1_audit.sha256(args.metrics),
                               "local_scored_cache": v1_audit.sha256(args.cache), "gold": v1_audit.sha256(args.gold),
                               "frozen_v2_policy": v1_audit.sha256(args.policy), "v2_audit_script": v1_audit.sha256(Path(__file__)),
                               "reused_v1_audit_script": v1_audit.sha256(ROOT / "scripts/audit_repair_outcomes.py")}
    if args.metrics.read_bytes() != raw_bytes:
        raise ValueError("Frozen metrics changed during audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.report.write_text(markdown(result))
    print(json.dumps({"questions": result["questions"], "local_vs_first_valid": result["local_vs_first_valid"],
                      "exact_gold_program_counts": result["exact_gold_program_counts"],
                      "transition_counts_by_edit_role": result["transition_counts_by_edit_role"],
                      "guard": result["guard_verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
