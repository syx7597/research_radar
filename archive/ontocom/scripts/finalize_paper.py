"""
Finalize paper tables once WN18RR OntoCom results are available.

Usage:
    python scripts/finalize_paper.py
"""

import json
import re
from pathlib import Path

RESULTS_PATH = Path("results/link_prediction/ontocom_wn18rr.json")
TABLES_PATH = Path("paper/tables.tex")
PAPER_PATH = Path("paper/ontocom_paper.tex")


def format_val(v):
    return f"{v:.4f}"


def update_tables(metrics):
    """Update Table 2 in tables.tex with WN18RR OntoCom results."""
    mrr = format_val(metrics["mrr"])
    h1 = format_val(metrics["hits_at_1"])
    h3 = format_val(metrics.get("hits_at_3", 0.0))
    h10 = format_val(metrics["hits_at_10"])

    content = TABLES_PATH.read_text(encoding="utf-8")

    # Replace placeholder row
    old = r"OntoCom (ours) & \textbf{---} & \textbf{---} & \textbf{---} & \textbf{---} \\"
    new = (f"OntoCom (ours) & \\textbf{{{mrr}}} & \\textbf{{{h1}}} & "
           f"\\textbf{{{h3}}} & \\textbf{{{h10}}} \\\\")

    if old in content:
        content = content.replace(old, new)
        TABLES_PATH.write_text(content, encoding="utf-8")
        print(f"[OK] Updated Table 2 with OntoCom WN18RR: MRR={mrr} H@1={h1} H@3={h3} H@10={h10}")
    else:
        print("[WARN] Placeholder not found in tables.tex — may already be updated")
        print(f"       New values: MRR={mrr} H@1={h1} H@3={h3} H@10={h10}")


def update_paper_text(metrics):
    """Update WN18RR section in ontocom_paper.tex."""
    mrr = format_val(metrics["mrr"])
    h1 = format_val(metrics["hits_at_1"])
    h10 = format_val(metrics["hits_at_10"])

    content = PAPER_PATH.read_text(encoding="utf-8")

    # Replace the "pending" paragraph with actual results
    old_snippet = "OntoCom training is ongoing\n(epoch 100/200 at this writing; loss 0.1243, steadily decreasing from 0.1523\nat epoch 50)."
    new_snippet = (f"OntoCom achieves MRR~=~{mrr}, Hits@1~=~{h1}, Hits@10~=~{h10},\n"
                   f"outperforming RotatE (MRR~=~0.4675) by "
                   f"{(metrics['mrr']/0.4675 - 1)*100:+.1f}\\%.")

    if old_snippet in content:
        content = content.replace(old_snippet, new_snippet)
        PAPER_PATH.write_text(content, encoding="utf-8")
        print(f"[OK] Updated WN18RR text section")
    else:
        print("[WARN] Could not find WN18RR text snippet; manual update needed")

    # Update conclusion if it mentions "pending"
    if "pending" in content:
        print("[INFO] 'pending' still in paper — check for other placeholders")


def main():
    if not RESULTS_PATH.exists():
        print(f"WN18RR OntoCom results not yet available: {RESULTS_PATH}")
        print("Re-run this script once run_wn18rr.py completes.")
        return

    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n=== WN18RR OntoCom Results ===")
    print(f"  MRR    = {data['mrr']:.4f}")
    print(f"  H@1    = {data['hits_at_1']:.4f}")
    print(f"  H@3    = {data.get('hits_at_3', 0.0):.4f}")
    print(f"  H@10   = {data['hits_at_10']:.4f}")
    print(f"  Time   = {data['training_time']}s")
    print()

    update_tables(data)
    update_paper_text(data)

    print("\nFinalization complete. Check paper/ directory.")


if __name__ == "__main__":
    main()
