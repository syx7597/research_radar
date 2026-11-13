"""One numeric source for Chinese/English short-paper tables."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(number, places=2):
    return str(Decimal(str(number)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def table(command, headers, body, columns):
    lines = ["\\newcommand{\\"+command+"}{%", "\\begin{tabular}{"+columns+"}",
             "\\toprule", " & ".join(headers)+r" \\", "\\midrule"]
    lines += [" & ".join(map(str, row))+r" \\" for row in body]
    return "\n".join(lines+["\\bottomrule", "\\end{tabular}%", "}"])+"\n"


def generate():
    source = HERE / "tables_data.json"; data = json.loads(source.read_text())
    for item in data["sources"].values():
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Evidence source changed: "+item["path"])
    metrics = {r["model"]: r for r in data["main_results"]}
    effects = {r["seed"]: r for r in data["paired_effects"]}
    output, specifications = [], {}
    for lang in ("ZH", "EN"):
        main = []
        for key in ("P", "A1", "C1", "A2", "C2"):
            row = metrics[key]
            effect = effects[row["seed"]] if key.startswith("C") else None
            label = "$"+key[0]+("_{"+key[1]+"}" if len(key)>1 else "")+"$"
            main.append([label, str(row["correct"]), fmt(Decimal(row["correct"])*100/row["questions"]),
                         "+"+fmt(effect["delta_pp"]) if effect else "--",
                         "["+", ".join(effect["ci95_display"])+"]" if effect else "--"])
        primary = data["primary_estimate"]
        main += [["A均值" if lang=="ZH" else "A mean", "--", fmt(primary["A_mean_accuracy"]*100), "--", "--"],
                 ["C均值" if lang=="ZH" else "C mean", "--", fmt(primary["C_mean_accuracy"]*100),
                  "+"+fmt(primary["accuracy_delta_pp"]), "["+", ".join(primary["ci95_display"])+"]"]]
        headers = ["模型", "正确数", "准确率(\\%)", "C减A", "95\\%区间"] if lang=="ZH" else ["Model", "Correct", "Acc. (\\%)", "$C-A$", "95\\% CI"]
        name = "RecoveryMainTable"+lang
        output.append(table(name, headers, main, "@{}lrrrl@{}"))
        specifications[name] = {"headers": headers, "rows": main}
        cost = []
        for key in ("A1", "C1", "A2", "C2"):
            row = metrics[key]
            cost.append(["$"+key[0]+"_{"+key[1]+"}$", row["failed_to_finish"], row["invalid_calls"],
                         fmt(row["input_tokens"]/1e6, 3), fmt(row["generated_tokens"]/1e3, 3), fmt(row["total_tokens"]/1e6, 3)])
        headers = ["模型", "未完成", "无效调用", "输入(百万)", "生成(千)", "合计(百万)"] if lang=="ZH" else ["Model", "Unfinished", "Invalid", "Input (M)", "Output (k)", "Total (M)"]
        name = "RecoveryCostTable"+lang
        output.append(table(name, headers, cost, "@{}lrrrrr@{}"))
        specifications[name] = {"headers": headers, "rows": cost}
        source_diag = json.loads((ROOT/data["sources"]["construction"]["path"]).read_text())
        body = []
        labels = [("suffix_all_calls_execute", "后缀全部调用可执行", "All suffix calls execute"),
                  ("submitted_answer", "成功提交答案", "Answer submitted"),
                  ("final_answer_correct", "答案与参考一致", "Answer matches reference"),
                  ("reference_observations_match", "全部参考观察一致", "All reference observations match"),
                  ("full_gate_pass", "完整构造验收通过", "Full construction gate passes")]
        for metric, zh, en in labels:
            body.append([zh if lang=="ZH" else en, source_diag["arms"]["direct"][metric], source_diag["arms"]["mapped_unfiltered"][metric]])
        headers = ["构造指标", "原编号拼接", "依赖映射"] if lang=="ZH" else ["Construction check", "Direct splice", "Remapped"]
        name = "RecoveryConstructionTable"+lang
        output.append(table(name, headers, body, "@{}lrr@{}"))
        specifications[name] = {"headers": headers, "rows": body}
    related = {
        "ZH": [["DAgger", "策略访问状态上的专家动作", "专家标注与迭代数据聚合"],
               ["Agent-R", "搜索分支与模型指导修订", "MCTS评价与迭代自训练"],
               ["STeP", "教师反思与纠正", "保留学生历史，屏蔽错误步骤"],
               ["OEC", "学生历史上的专家续写", "环境筛选与续写部分监督"],
               ["ReTerm", "失败终态与历史上的教师恢复", "恢复SFT及进一步在线RL"],
               ["本文", "已有参考程序的后缀", "句柄映射、只读重放与后缀监督"]],
        "EN": [["DAgger", "Expert actions on visited states", "Expert labeling and iterative aggregation"],
               ["Agent-R", "Search branches and model-guided revision", "MCTS outcomes and iterative self-training"],
               ["STeP", "Teacher reflections and corrections", "Student context with masked erroneous steps"],
               ["OEC", "Expert continuation of student history", "Environment filtering and continuation loss"],
               ["ReTerm", "Teacher recovery from failed terminal histories", "Recovery SFT followed by online recovery RL"],
               ["This work", "Existing reference-program suffix", "Handle remapping, read-only replay, suffix loss"]]}
    for lang, body in related.items():
        headers = ["方法", "后续目标来源", "构造与监督条件"] if lang=="ZH" else ["Method", "Continuation source", "Construction and supervision"]
        name = "RecoveryRelatedTable"+lang
        output.append(table(name, headers, body, "@{}p{0.14\\linewidth}p{0.37\\linewidth}p{0.42\\linewidth}@{}"))
        specifications[name] = {"headers": headers, "rows": body}
    tex = "% Generated from one audited data file; no external-paper scores are compared.\n"+"\n".join(output)
    manifest = {"schema": "bilingual_recovery_table_export_v1", "date": "2026-11-14",
        "data_sha256": sha(source), "code_sha256": sha(__file__), "source_bindings_verified": len(data["sources"]),
        "output_sha256": hashlib.sha256(tex.encode()).hexdigest(), "both_languages_share_numeric_rows": True,
        "rounding": "Decimal ROUND_HALF_UP; primary interval display fixed to audited 3.53/5.98",
        "external_comparison": "Method/resource comparison only; no cross-paper score ranking"}
    return {"generated_tables.tex": tex, "table_specs.json": json.dumps(specifications, ensure_ascii=False, indent=2)+"\n",
            "table_export_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2)+"\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = generate()
    for name, text in outputs.items():
        path = HERE/name
        if args.check:
            if not path.exists() or path.read_text()!=text:
                raise ValueError("Stale table export: "+name)
        else:
            path.write_text(text)
    print(json.dumps({"checked" if args.check else "written": list(outputs)}))


if __name__ == "__main__":
    main()
