"""Profile task structure only; does not train, generate negatives, or evaluate a model.

python3 -B scripts/profile_public_benchmark.py --write
Requires the local KQA Pro kb/val and RadarKG entity snapshot.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "qualifier": {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate",
                  "QueryAttrQualifier", "QueryRelationQualifier", "QueryAttrUnderCondition"},
    "numeric_or_comparison": {"FilterNum", "QFilterNum", "VerifyNum", "SelectBetween", "SelectAmong"},
    "set": {"And", "Or"},
    "count": {"Count"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    paths = {"kqa_val": ROOT / "datasets/kqa_pro/val.json",
             "kqa_kb": ROOT / "datasets/kqa_pro/kb.json",
             "radar_entities": ROOT / "kg_v3/entities.json"}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit("Restore local inputs first: " + ", ".join(missing))
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    rows, kb, entities = data["kqa_val"], data["kqa_kb"], data["radar_entities"]
    functions, lengths, families = Counter(), Counter(), Counter()
    for row in rows:
        used = {n["function"] for n in row["program"]}
        functions.update(used)
        lengths[len(row["program"])] += 1
        for family, members in FAMILIES.items():
            families[family] += bool(used & members)
    attrs = [a for e in entities for a in e.get("attributes", [])]
    profile = {
        "purpose": "Task and local-data structure, not observed model errors or method effectiveness",
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
        "kqa_pro": {
            "validation_questions": len(rows),
            "local_train_present": (ROOT / "datasets/kqa_pro/train.json").is_file(),
            "family_definition": {k: sorted(v) for k, v in FAMILIES.items()},
            "family_counts_overlapping": dict(families),
            "question_counts_by_function": dict(sorted(functions.items())),
            "program_length_counts": dict(sorted(lengths.items())),
            "entities": len(kb["entities"]), "concepts": len(kb["concepts"]),
        },
        "radar": {
            "entities": len(entities), "attribute_records": len(attrs),
            "with_numeric_value_and_unit": sum(isinstance(a.get("value"), (int, float))
                                                and not isinstance(a.get("value"), bool)
                                                and bool(a.get("unit")) for a in attrs),
            "with_condition_or_qualifier_field": sum(any(k in a for k in
                ("condition", "conditions", "qualifier", "qualifiers")) for a in attrs),
            "attribute_counts": dict(Counter(a.get("attr", "unknown") for a in attrs)),
        },
    }
    if args.write:
        out = ROOT / "artifacts/public_benchmark_profile.json"
        out.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        print("Wrote", out.relative_to(ROOT))
    print(json.dumps({"kqa_questions": len(rows), "families": dict(families),
                      "radar_attributes": len(attrs),
                      "radar_numeric_attributes": profile["radar"]["with_numeric_value_and_unit"],
                      "radar_structured_conditions": profile["radar"]["with_condition_or_qualifier_field"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
