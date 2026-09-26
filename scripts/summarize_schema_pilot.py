"""Post-hoc development error inventory; no new inference or parameter choice."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_query_repair import difference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["id"]: row for row in data}
    if len(result) != len(data):
        raise ValueError("Duplicate IDs")
    return result


def main():
    run = ROOT / "results/condition_consistency/schema_pilot"
    output = run / "dev500/residual_errors.json"
    review = run / "dev500/legitimate_field_review.csv"
    if output.exists() or review.exists():
        raise ValueError("Keep original diagnostic outputs; no overwrite")
    paths = {"metrics": run / "dev500/metrics.json", "protocol": run / "protocol.json",
             "generated": run / "dev500/constrained4_generated.jsonl",
             "executed": run / "dev500/constrained4_executed.jsonl",
             "gold": ROOT / "data/condition_consistency/splits/dev.jsonl",
             "schema": ROOT / "results/condition_consistency/query_repair_audit/schema.json"}
    metrics, protocol = (json.loads(paths[name].read_text()) for name in ("metrics", "protocol"))
    for name in ("gold", "schema"):
        if sha(paths[name]) != protocol[name]["sha256"]:
            raise ValueError("Data differs from frozen protocol")
    if (sha(paths["protocol"]) != metrics["protocol_sha256"]
            or sha(paths["executed"]) != metrics["provenance"]["constrained4"]["executed_sha256"]
            or sha(paths["generated"]) != metrics["provenance"]["constrained4"]["generation_meta"]["output_sha256"]):
        raise ValueError("Candidate or report bytes differ")
    gold, predictions = rows(paths["gold"]), rows(paths["executed"])
    details = {row["id"]: row for row in metrics["per_question"]["constrained4"]}
    if set(gold) != set(predictions) or set(gold) != set(details) or len(gold) != 500:
        raise ValueError("Expected all 500 development questions")
    schema = json.loads(paths["schema"].read_text())["names_by_role"]
    counts, categories, field_roles, errors, review_rows = Counter(), Counter(), Counter(), [], []
    for qid, detail in details.items():
        if detail["correct"]:
            continue
        counts["remaining_errors"] += 1
        counts["errors_with_correct_candidate"] += detail["oracle_correct"]
        counts["errors_without_correct_candidate"] += not detail["oracle_correct"]
        selected = detail["selected"]
        error = {"id": qid, "selected": selected, "oracle_correct": detail["oracle_correct"]}
        if selected is None:
            categories["no_executable_output"] += 1
            errors.append(error)
            continue
        candidate = predictions[qid]["candidates"][selected]
        if not candidate["valid"] or candidate["schema_issues"]:
            raise ValueError("Selected constrained candidate unexpectedly invalid")
        diff = difference(candidate["program"], gold[qid]["program"])
        categories[diff["category"]] += 1
        error["difference_from_reference_gold"] = diff
        if diff["category"] == "single_field":
            edit = diff["differences"][0]
            field_roles[edit["role"]] += 1
            legal = edit["predicted"] in schema[edit["role"]]
            error["predicted_field_globally_legal"] = legal
            if legal and edit["role"] in ("attribute", "relation", "qualifier"):
                counts["single_legal_predicate_reference_difference"] += 1
                review_rows.append({"id": qid, "question": gold[qid]["question"], "role": edit["role"],
                    "predicted_program": candidate["program_text"],
                    "reference_program": json.dumps(gold[qid]["program"], ensure_ascii=False),
                    "mechanical_difference": json.dumps(edit, ensure_ascii=False),
                    "human_review_status": "pending", "reviewer": ""})
        errors.append(error)
    result = {"scope": "Post-hoc mechanical inventory on previously seen development data; not human error attribution or training evidence",
              "inputs_sha256": {k: sha(p) for k, p in paths.items()},
              "script_sha256": sha(Path(__file__)), "counts": dict(counts),
              "selected_error_categories": dict(categories), "single_field_roles": dict(field_roles), "errors": errors,
              "limits": ["Differences from one reference program do not prove semantic causes or recoverable gains",
                         "No synthetic negatives, training, new predictions or gold execution",
                         "All failed questions remain; missing outputs are not assigned a fabricated program"]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = ["id", "question", "role", "predicted_program", "reference_program", "mechanical_difference",
              "human_review_status", "reviewer"]
    with review.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps({k: result[k] for k in ("counts", "selected_error_categories", "single_field_roles")}, indent=2))


if __name__ == "__main__":
    main()
