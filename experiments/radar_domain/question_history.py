"""Bounded old-QA screening for a separate reviewer, never the new author.

Lexical matches and similarities generate review candidates, not a semantic
isolation certificate. Raw questions/answers and matches remain local-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from experiments.radar_domain.coverage_design import QUESTION_FILES, question_sha
from experiments.radar_domain.source_history_screening import PATTERNS, normalize
from experiments.radar_domain.source_readings import ROOT, sha, write_new
from experiments.radar_domain.source_rag import tokenize

LOCAL = ROOT / "data/radar_sources_v2/qa_v1"
INDEX = LOCAL / "history_index.json"
FILES = [f"ca_agraphrag/data/{n}.jsonl" for n in ("train", "dev", "test", "dev_polished", "test_polished", "sft_train")] + list(QUESTION_FILES)
FAMILY_KEYS = {"eec_ranger_x_band": "eec_ranger", "jrc_jma5200mk2": "jrc_jma5200", "garmin_gmr_fantom": "garmin_fantom"}


def build_index():
    rows, inputs, counts = [], {}, {}
    for name in FILES:
        p = ROOT / name; inputs[name] = sha(p)
        if p.suffix == ".csv":
            with p.open() as f:
                raw = list(csv.DictReader(f))
        else:
            raw = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        counts[name] = len(raw)
        for i, r in enumerate(raw, 1):
            if name.endswith("sft_train.jsonl"):
                qs = [m["content"] for m in r["messages"] if m["role"] == "user"]
                if len(qs) != 1:
                    raise ValueError("Unexpected SFT user-question structure")
                question = qs[0]
            else:
                question = r["question"]
            payload = normalize(json.dumps(r, ensure_ascii=False))
            matches = {f: [width for width, pat in pats.items() if re.search(pat, payload)]
                       for f, pats in PATTERNS.items()}
            rows.append({"history_id": f"{name}:row:{i}", "source_file": name,
                         "old_question_id": r.get("qid") or r.get("id"), "question": question,
                         "question_sha256": question_sha(question),
                         "family_alias_matches": {k: v for k, v in matches.items() if v},
                         "original_row": r})
    return {"schema": "radar_old_question_index_v1", "inputs_sha256": inputs,
            "counts_by_file": counts, "rows": rows,
            "reference_scope": "5401 original QA plus 1124 polished variants, 4277 SFT traces and all registered 12/24-target question versions",
            "limits": "Includes variants/trace repeats, not independent targets; alias coverage and historical source lineage remain incomplete."}


def load_index():
    d = json.loads(INDEX.read_text())
    for rel, expected in d["inputs_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Changed historical QA input")
    # Entire index must reproduce: a removed row must not escape review silently.
    if d != build_index():
        raise ValueError("Historical index changed")
    return d


def screen(paths):
    d = load_index(); old = d["rows"]
    # Character n-gram/word similarity supplements family matches. Neither is
    # used to admit questions automatically or to rewrite the new author's work.
    tokens = [set(tokenize(r["question"])) for r in old]
    questions, inputs = [], {}
    for p in paths:
        p = Path(p).resolve(); inputs[str(p.relative_to(ROOT))] = sha(p)
        questions += json.loads(p.read_text())["questions"]
    results = []
    for q in questions:
        t = set(tokenize(q["question_zh"]))
        scores = sorted(((len(t & u) / max(1, len(t | u)), i) for i, u in enumerate(tokens)), reverse=True)
        near, known = [], set()
        # Three distinct old wordings, not three copies of one old question.
        for score, i in scores:
            r = old[i]
            if r["question_sha256"] in known:
                continue
            known.add(r["question_sha256"])
            near.append({"history_id": r["history_id"], "similarity": score})
            if len(near) == 3:
                break
        family = FAMILY_KEYS.get(q["family_id"], q["family_id"])
        results.append({"question_id": q["question_id"], "family_id": q["family_id"],
                        "question_sha256": question_sha(q["question_zh"]),
                        "exact_normalized_matches": [r["history_id"] for r in old if r["question_sha256"] == question_sha(q["question_zh"])],
                        "same_or_ambiguous_family_candidates": [r["history_id"] for r in old if family in r["family_alias_matches"]],
                        "nearest_old_wordings": near, "semantic_admission": "pending_reviewer"})
    return {"schema": "radar_question_history_screening_v1", "inputs_sha256": inputs,
            "history_index_sha256": sha(INDEX), "history_counts_by_file": d["counts_by_file"],
            "question_count": len(questions), "rows": results,
            "limits": "No exact match does not prove independent intent or unseen family. A reviewer must inspect the candidate contexts and source lineage."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("packets", nargs="*")
    a = p.parse_args()
    if a.prepare:
        d = build_index(); write_new(INDEX, d)
        print(json.dumps({"indexed_question_or_trace_rows": len(d["rows"]), "input_files": len(d["inputs_sha256"])}))
    elif a.packets:
        if a.output is None or not a.output.resolve().is_relative_to(LOCAL):
            raise ValueError("Local screening output required")
        d = screen(a.packets); write_new(a.output, d)
        print(json.dumps({"screened_questions": d["question_count"]}))
    else:
        print(json.dumps({"verified_index_rows": len(load_index()["rows"])}))


if __name__ == "__main__":
    main()
