"""Freeze a separately reviewed composition evaluation; preserve the original split.

Run with the pinned tokenizer-only CPU environment. --check reconstructs every
artifact and compares its bytes. Full questions, references and traces stay private.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from .freeze_offline_dataset import formatted_question
from .table_program import TableExecutor, execute_actions, program_answer_spec
from .table_scoring import CatalogIndex, score_answer
from .table_token_preflight import tokenize

BASE = Path("data/radar_domain_recovery_v2")
OLD = BASE / "frozen_20261107"
OUT = BASE / "compositional_supplement_20261108"
PUBLIC = Path("results/radar_domain/compositional_supplement_20261108")
AUTHORS = BASE / "composition_authors_20261108"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def encode(obj, lines=False):
    if lines:
        return "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in obj)
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def schema_only(spec):
    if set(spec) != {"outputs"} or not isinstance(spec["outputs"], list) or not spec["outputs"]:
        raise ValueError("Output format must be nonempty and value-free")
    normalized = []
    for output in spec["outputs"]:
        if output.get("kind") not in {"table", "assessment"} or not isinstance(output.get("name"), str):
            raise ValueError("Invalid output format")
        expected = {"name", "kind", "columns"} if output["kind"] == "table" else {"name", "kind"}
        if set(output) != expected:
            raise ValueError("Output format may not contain answer values or counts")
        item = deepcopy(output)
        if "columns" in item:
            columns = item["columns"]
            if not isinstance(columns, list) or not columns or any(type(c) is not str for c in columns) or len(set(columns)) != len(columns):
                raise ValueError("Explicit unique columns required")
            item["columns"] = sorted(columns)
        normalized.append(item)
    if len({o["name"] for o in normalized}) != len(normalized):
        raise ValueError("Duplicate answer output")
    return {"outputs": sorted(normalized, key=lambda o: o["name"])}


def live_operations(actions):
    """Count only operations that contribute to the submitted answer handle."""
    if not actions or actions[-1]["name"] != "finish":
        raise ValueError("A complete submitted program is required")
    visited = set()

    def visit(index):
        if type(index) is not int or not 0 <= index < len(actions)-1:
            raise ValueError("Invalid reference dependency")
        if index in visited:
            return
        visited.add(index)
        call = actions[index]
        if call["name"] != "step":
            raise ValueError("Only successful step handles may be referenced")
        for dependency in call["arguments"]["dependencies"]:
            if type(dependency) is not int or not dependency < index:
                raise ValueError("References must use earlier handles")
            visit(dependency)

    visit(actions[-1]["arguments"]["answer_handle"])
    return {actions[i]["arguments"]["function"] for i in visited}


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("Bound input changed: " + str(path))


def build():
    import transformers, tokenizers
    from transformers import AutoTokenizer
    if (transformers.__version__, tokenizers.__version__) != ("5.18.0", "0.23.2"):
        raise ValueError("Use the registered CPU tokenizer environment")
    design_path, review_path = PUBLIC / "design.json", PUBLIC / "source_review.safe.json"
    design, review = read(design_path), read(review_path)
    verify_bindings(design["inputs_sha256"])
    if review["blocking_findings"] or review.get("reference_grade") != "AI":
        raise ValueError("Resolved independent source review required")
    verify_bindings(review["author_files_sha256"])
    review_inputs = []
    for key in ("independent_answers", "seal"):
        binding = review[key]
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Independent source answer seal changed")
        review_inputs.append(Path(binding["path"]))
    old_manifest = read(OLD / "split_manifest.json")
    old_ids = {q for part in old_manifest["splits"].values() for q in part["ids"]}
    catalog_path = OLD / "holdout/catalog.json"
    catalog = read(catalog_path)
    executor, index = TableExecutor(catalog), CatalogIndex(catalog)
    if len(catalog["records"]) != 156:
        raise ValueError("Use the same complete 156-reading source catalog")
    source_families = {r["family_id"] for r in catalog["records"]}
    if source_families != set(design["families"]):
        raise ValueError("Source scope changed")
    tokenizer_path = Path("cache/qwen2.5-3b-tokenizer")
    token_original = read(Path("results/radar_domain/frozen_20261107/token_preflight.json"))
    verify_bindings({str(tokenizer_path/n): h for n, h in token_original["tokenizer_files_sha256"].items()})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    accepted, rejected = set(review["accepted_ids"]), set(review["rejected_ids"])
    if accepted & rejected or not 0 < len(accepted) <= design["question_cap"]:
        raise ValueError("Invalid independent admission decisions")
    questions, refs, metadata, replays, token_checks = [], [], [], [], []
    submitted, families, operations, profiles = set(), Counter(), Counter(), []
    submission_families = Counter()
    old_original_questions = {r["original_question"] for r in rows(OLD / "holdout/metadata.jsonl")}
    input_paths = [design_path, review_path, catalog_path, OLD / "split_manifest.json", OLD / "task_contract.json",
                   Path(__file__), Path("experiments/radar_domain/table_program.py"),
                   Path("experiments/radar_domain/table_scoring.py"), Path("experiments/radar_domain/table_token_preflight.py"),
                   Path("experiments/radar_domain/freeze_offline_dataset.py"),
                   *review_inputs, *[tokenizer_path/n for n in token_original["tokenizer_files_sha256"]]]
    for author in ("marine", "land"):
        folder = AUTHORS / author
        manifest_path = folder / "manifest.json"
        package = read(manifest_path)
        verify_bindings(package["outputs_sha256"])
        if package.get("training_allowed", True):
            raise ValueError("Evaluation source package cannot be training material")
        input_paths += [manifest_path, *[Path(p) for p in package["outputs_sha256"]]]
        authored = {q["id"]: q for q in read(folder / "questions.json")["rows"]}
        expected = {q["id"]: q for q in read(folder / "author_reference.json")["rows"]}
        specs = {q["id"]: q["answer_spec"] for q in read(folder / "answer_specs.json")["rows"]}
        source_refs = rows(folder / "references.jsonl")
        if set(authored) != set(expected) or set(authored) != set(specs) or {r["id"] for r in source_refs} != set(authored):
            raise ValueError("Incomplete author package")
        for original in source_refs:
            ident = original["id"]
            if ident in submitted or ident in old_ids:
                raise ValueError("Evaluation question ID overlap")
            submitted.add(ident)
            submission_families[authored[ident]["family_id"]] += 1
            if authored[ident]["question"] in old_original_questions:
                raise ValueError("Exact duplicate of an original evaluation question")
            if ident in rejected:
                continue
            if ident not in accepted:
                raise ValueError("Question has no independent admission decision")
            q, source = authored[ident], expected[ident]
            if q["question"] != original["question"] or q["family_id"] not in source_families:
                raise ValueError("Authored question/reference differs")
            spec = schema_only(specs[ident])
            if spec != schema_only(program_answer_spec(original["actions"])):
                raise ValueError("Answer format differs from operations")
            live = live_operations(original["actions"])
            if not live & {"Join", "Filter", "Derive", "Assess", "Expand"}:
                raise ValueError("Only lookup/assembly; no live compositional operation")
            required = set(source["required_record_ids"])
            if len(required) < 2 or any(executor.by_id[r]["family_id"] != q["family_id"] for r in required):
                raise ValueError("Necessary source records are missing or cross family")
            question = formatted_question(q["question"], spec)
            ep = execute_actions(executor, question, original["actions"])
            if not score_answer(ep.answer, original["answer"], index, spec)["correct"]:
                raise ValueError("Frozen replay changed source answer: " + ident)
            if not required <= set(ep.answer["source_records"]):
                raise ValueError("Reference omitted necessary source support")
            trace = {"id": ident, "messages": ep.messages + [{"role": "assistant", "content": "Done."}],
                     "training_eligible": False}
            trace["supervise"] = [m["role"] == "assistant" for m in trace["messages"]]
            ids, spans = tokenize(tokenizer, trace, design["evaluation"]["budgets"]["context_tokens"])
            action_spans = [s for s in spans if not s["done"]]
            max_step = max(s["tokens"] for s in action_spans)
            total = sum(s["tokens"] for s in action_spans)
            budgets = design["evaluation"]["budgets"]
            if max_step > budgets["per_turn_generated_tokens"] or total > budgets["total_generated_tokens"] or ep.calls > budgets["max_calls"]:
                raise ValueError("Reference exceeds pre-registered budget")
            questions.append({"id": ident, "question": question})
            refs.append({"id": ident, "question": question, "actions": original["actions"], "answer": ep.answer})
            metadata.append({"id": ident, "family_id": q["family_id"], "task_type": q["task_type"],
                "source_pages": q["source_pages"], "original_question": q["question"],
                "original_question_sha256": hashlib.sha256(q["question"].encode()).hexdigest(),
                "answer_spec": spec, "training_allowed": False, "reference_grade": "AI"})
            replays.append({"id": ident, "messages": ep.messages, "events": ep.events, "calls": ep.calls,
                            "model_generated": False, "training_eligible": False})
            token_checks.append({"id": ident, "context_tokens": len(ids), "max_action_tokens": max_step,
                                 "total_action_tokens": total, "calls": ep.calls, "training_eligible": False})
            families[q["family_id"]] += 1
            operations.update(a["arguments"].get("function", a["name"]) for a in original["actions"])
            profiles.append({"id": ident, "family_id": q["family_id"], "calls": ep.calls,
                             "live_operations": sorted(live), "task_type": q["task_type"]})
    if (submitted != accepted | rejected or len(submitted) > design["question_cap"]
            or any(n > design["target_per_family"] for n in submission_families.values())):
        raise ValueError("Question cap/admission mismatch")
    verify_bindings(design["inputs_sha256"])
    input_hashes = {str(p): sha(p) for p in sorted(set(input_paths))}
    outputs = {OUT / "catalog.json": catalog_path.read_text(), OUT / "questions.jsonl": encode(questions, True),
               OUT / "references.jsonl": encode(refs, True), OUT / "metadata.jsonl": encode(metadata, True),
               OUT / "reference_replays.jsonl": encode(replays, True), OUT / "token_checks.jsonl": encode(token_checks, True)}
    manifest = {"schema": "radar_table_evaluation_supplement_v1", "date": "2026-11-08", "status": "frozen_before_model_runs",
        "role": "separate_compositional_evaluation", "ids": [q["id"] for q in questions],
        "group_by_id": {m["id"]: m["family_id"] for m in metadata}, "question_count": len(questions),
        "training_allowed": False, "human_gold": False, "model_outputs_seen": False, "model_runs": 0,
        "parent_split": {"path": str(OLD / "split_manifest.json"), "sha256": sha(OLD / "split_manifest.json")},
        "original_24_unchanged": True, "same_complete_holdout_catalog": True,
        "source_family_relation": "Same four source families as original24; separate question set, not independent new sources",
        "reporting": "Original24 primary remains unchanged. Supplement uses its own fixed denominator, paired correctness changes and source-family counts; failures stay in denominator.",
        "evaluation_budget": design["evaluation"]["budgets"], "reference_review": {"path": str(review_path), "sha256": sha(review_path)},
        "inputs_sha256": input_hashes,
        "outputs": {p.name: {"path": str(p), "sha256": hashlib.sha256(s.encode()).hexdigest()} for p, s in outputs.items()}}
    outputs[OUT / "manifest.json"] = encode(manifest)
    summary = {"schema": "radar_composition_supplement_public_v1", "date": "2026-11-08", "status": "frozen_before_model_runs",
        "question_count": len(questions), "family_counts": dict(families), "source_catalog_records": 156,
        "original_24_unchanged": True, "training_questions_unchanged": 576, "development_questions_unchanged": 108,
        "reference_replays_passed": len(refs), "reference_calls": sum(x["calls"] for x in profiles),
        "call_range": [min(x["calls"] for x in profiles), max(x["calls"] for x in profiles)],
        "operation_counts": dict(operations), "question_profiles": profiles,
        "token_preflight": {"references": len(token_checks), "max_context_tokens": max(x["context_tokens"] for x in token_checks),
            "max_action_tokens": max(x["max_action_tokens"] for x in token_checks),
            "max_total_action_tokens": max(x["total_action_tokens"] for x in token_checks), "budget": design["evaluation"]["budgets"],
            "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
            "no_truncation": True, "tokenizer_files_sha256": token_original["tokenizer_files_sha256"]},
        "source_review_grade": "AI", "training_allowed": False, "model_runs": 0, "server_accessed": False,
        "admission_note": "Source reading counts are not independent fact counts; OTT-01 comprises two state fact groups with correlated power/current presentations and redundant classification predicates.",
        "manifest": {"path": str(OUT / "manifest.json"), "sha256": hashlib.sha256(outputs[OUT / "manifest.json"].encode()).hexdigest()},
        "design": {"path": str(design_path), "sha256": sha(design_path)},
        "independent_review": {"path": str(review_path), "sha256": sha(review_path)},
        "code_sha256": sha(__file__)}
    outputs[PUBLIC / "summary.json"] = encode(summary)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if not args.check and (OUT.exists() or (PUBLIC / "summary.json").exists()):
        raise FileExistsError("Preserve frozen/partial artifacts; inspect before writing")
    outputs = build()
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError("Frozen artifact changed: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(json.dumps({"status": "checked" if args.check else "frozen", "files": len(outputs),
                      "summary": str(PUBLIC / "summary.json")}))


if __name__ == "__main__":
    main()
