"""Post-result descriptive decomposition; no new scoring, selection or model run."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .coverage_review_summary import joint, require
from .source_readings import ROOT, sha, write_new


def build(output):
    package = ROOT / "data/radar_sources_v2/semantic_review_v1"
    summary_path = ROOT / "results/radar_domain/coverage_v2/semantic_summary.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((package / "manifest.json").read_text())
    mapping_path, verdicts_path = package / "mapping_for_adjudicator_only.json", package / "final_verdicts.json"
    require(sha(verdicts_path) == summary["private_verdicts_sha256"] and sha(mapping_path) == manifest["mapping_sha256"], "Frozen scoring inputs changed")
    mapping = {r["review_id"]: r for r in json.loads(mapping_path.read_text())["rows"]}
    arms = {a: {} for a in ("raw", "flat", "bound")}
    for row in json.loads(verdicts_path.read_text())["rows"]:
        identity = mapping[row["review_id"]]
        arms[identity["arm"]][identity["question_id"]] = row
    require(all(len(rows) == 96 for rows in arms.values()) and all(set(rows) == set(arms["flat"]) for rows in arms.values()), "Incomplete paired roster")
    wins, losses, joint_win_strata = 0, 0, Counter()
    for qid, flat in arms["flat"].items():
        bound = arms["bound"][qid]
        wins += bound["answer_correct"] and not flat["answer_correct"]
        losses += flat["answer_correct"] and not bound["answer_correct"]
        if joint(bound) and not joint(flat):
            joint_win_strata["both_answers_correct" if flat["answer_correct"] else "flat_answer_incorrect"] += 1
    result = {"schema": "radar_coverage_posthoc_diagnosis_v1", "posthoc": True,
              "semantic_summary_sha256": sha(summary_path), "code_sha256": sha(Path(__file__)),
              "answer_correct_bound_minus_flat": {"wins": wins, "losses": losses, "net": wins - losses},
              "joint_win_strata": dict(joint_win_strata),
              "arms": {a: {"correct_answer_but_citation_failure": sum(r["answer_correct"] and not r["citations_support"] for r in rows.values()),
                            "correct_answer_but_supplied_support_failure": sum(r["answer_correct"] and not r["supported_by_supplied_evidence"] for r in rows.values())}
                       for a, rows in arms.items()},
              "meaning": "Most paired joint wins can occur with both answer texts already correct; do not attribute the whole joint gain to better factual reasoning or conditional binding.",
              "limits": "Descriptive analysis after outputs, not a registered mechanistic ablation; original scoring, denominator and continuation gate unchanged."}
    require(wins - losses == summary["adjudicated"]["overall"]["bound"]["answer_correct"] - summary["adjudicated"]["overall"]["flat"]["answer_correct"], "Answer decomposition differs")
    require(sum(joint_win_strata.values()) == summary["adjudicated"]["overall"]["paired_bound_minus_flat"]["wins"], "Joint decomposition differs")
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / "results") and not output.exists(), "Use a new public aggregate file")
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))


if __name__ == "__main__":
    main()
