"""Reproduce bounded historical alias screening; never certify unseen data.

Reads only explicit legacy selectors below and known-exposure metadata. Reports
string matches, context classes and file hashes, not raw source/QA text.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts/thesis_direction_review/radar_sources_v2/history_screening.json"
INPUTS = ROOT / "artifacts/thesis_direction_review/radar_sources_v2/history_inputs.json"
REGISTRY = "artifacts/thesis_direction_review/radar_coverage_v2/exposure_registry.json"
PATTERNS = {'vaisala_wrs300': {'narrow': '(?<![a-z0-9])wrs[ _-]*300(?![0-9])', 'broad': '(?<![a-z0-9])wrs[ _-]*\\d+'}, 'eec_ranger': {'narrow': '(?<![a-z0-9])ranger[ _-]*x[ _-]*(?:1|5)(?![0-9])', 'broad': '(?<![a-z0-9])rangers?(?![a-z])'}, 'gamic_gmwr': {'narrow': '(?<![a-z0-9])gmwr[ _-]*25[ _-]*ws(?![a-z0-9])', 'broad': '(?<![a-z0-9])gmwr(?![a-z])'}, 'furuno_drs_nxt': {'narrow': '(?<![a-z0-9])drs[ _-]*(?:(?:6|12|25)[ _-]*a?[ _-]*)?nxt(?![a-z0-9])', 'broad': '(?<![a-z0-9])drs[ _-]*(?:\\d+[a-z]*)?|-?\\bnxt\\b'}, 'jrc_jma5200': {'narrow': '(?<![a-z0-9])(?:jma[ _-]*52(?:00|12|22)|nke[ _-]*(?:2103|2254))(?![0-9])', 'broad': '(?<![a-z0-9])jma[ _-]*\\d{3,4}'}, 'garmin_fantom': {'narrow': '(?<![a-z0-9])fantom(?![a-z])', 'broad': '(?<![a-z0-9])(?:gmr|fantom)(?![a-z])'}, 'simrad_halo': {'narrow': '(?<![a-z0-9])halo[ _-]*(?:3|4|6|20|20\\+|24|2000|3000)(?![0-9])', 'broad': '(?<![a-z0-9])halo(?![a-z])'}, 'raymarine_cyclone': {'narrow': '(?<![a-z0-9])(?:cyclone[ _-]*pro|e[ _-]*7062[01])(?![a-z0-9])', 'broad': '(?<![a-z0-9])cyclones?(?![a-z])'}}
SELECTORS = ['kg_v3/entities.json', 'kg_v3/edges.json', 'radar_corpus/corpus.json', 'radar_corpus/raw/*.json', 'radar_corpus/raw_v1/*.json', 'radar_corpus/filtered.json', 'radar_corpus/discovered.json', 'graphrag_index/merged_triples*.json', 'pipeline/v3/work/chunks.jsonl', 'pipeline/v3/work/manual_body_chunks.jsonl', 'pipeline/v3/work/naval_body_chunks.jsonl', 'pipeline/v3/work/manual*attrs.jsonl', 'pipeline/v3/work/naval*attrs.jsonl', 'data/v2/*.json', 'ca_agraphrag/data/train.jsonl', 'ca_agraphrag/data/dev.jsonl', 'ca_agraphrag/data/test.jsonl', 'ca_agraphrag/data/*polished.jsonl', 'ca_agraphrag/data/sft_train.jsonl']


# Candidate extension registered by root before any new questions/model outputs.
PATTERNS["metek_mrr"] = {
    "narrow": r"(?<![a-z0-9])mrr[ _-]*(?:2|pro)(?![a-z0-9])|\bmetek\b[\s\S]{0,60}(?:\bmrr\b|micro[ _-]*rain[ _-]*radar)",
    "broad": r"(?<![a-z0-9])mrr(?![a-z])|micro[ _-]*rain[ _-]*radar|\bmetek\b",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))


def leaves(value, pointer=""):
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, pointer + "/" + str(i))
    elif isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield pointer + "/@key:" + escaped, str(key)
            yield from leaves(child, pointer + "/" + escaped)


def context_class(family, context):
    """AI-reviewed lexical context classes, conservative for bare abbreviations.

    These classes are lineage screening, not source fact annotation. Unknown
    contexts stay unresolved. Same-publisher models are not made synonyms.
    """
    if family == "eec_ranger":
        if re.search(r"ranger[ _-]*x(?:[ _-]*(?:series|1|5))?", context):
            return "candidate_family_mention"
        if any(x in context for x in ("uss ranger", "ranger r5", "ranger_r5", "ranger r20ss", "ranger_r20ss", "ranger® r6ss", "ranger r6ss")):
            return "other_named_entity"
    elif family == "gamic_gmwr":
        if re.search(r"gmwr[ _-]*(?:25|400|1000)", context):
            return "candidate_family_mention"
    elif family == "furuno_drs_nxt":
        if re.search(r"drs[ _-]*\d+[a-z]?[ _-]*nxt", context):
            return "candidate_family_mention"
        if any(x in context for x in ("sustainment", "ew", "technologies", "drs等公司", "aic、drs 和")):
            return "other_named_entity"
    elif family == "jrc_jma5200":
        if re.search(PATTERNS[family]["narrow"], context):
            return "candidate_family_mention"
        if re.search(r"jma[ _-]*(?:1596|3000|1576|912|610|540|254)(?!\d)", context):
            return "same_publisher_other_model_lineage_unresolved"
    elif family == "garmin_fantom":
        if "fantom" in context or "garmin" in context:
            return "candidate_family_mention"
        if any(x in context for x in ("tfr/gmr", "tfr", "地形", "tornado", "super searcher")):
            return "other_radar_abbreviation"
        # A bare GMR token in a query/list has insufficient identity evidence.
        return "ambiguous_abbreviation_unresolved"
    elif family == "simrad_halo":
        if re.search(r"halo[ _-]*(?:20|24|3|4|6)", context) or any(x in context for x in ("halo__rt", "navigation radar", "pulsed radars", "type halo", "simrad")) or context == "halo":
            # Bare HALO entity was reviewed at kg_v3/entities.json index 7804,
            # whose attributes cite halo__rt; the source chunk lists HALO-3/4/6.
            return "candidate_family_mention"
    elif family == "raymarine_cyclone":
        if "cyclone 级" in context:
            return "other_named_entity_ship_class"
        if "raymarine" in context or "e70620" in context or "e70621" in context:
            return "candidate_family_mention"
    elif family == "metek_mrr":
        if re.search(PATTERNS[family]["narrow"], context) or "micro rain radar" in context:
            return "candidate_family_mention"
        if any(x in context for x in ("tps-77", "tps_77", "lanza", "mrr-3d", "三坐标", "三维", "对空", "舰艇", "行波管", "自卫", "multi-role", "发射机/接收机", "电子稳定", "mrr 进行自动波形", "mrr comes in two forms")):
            return "other_radar_abbreviation"
        return "ambiguous_abbreviation_unresolved"
    elif family == "vaisala_wrs300" and re.search(PATTERNS[family]["narrow"], context):
        return "candidate_family_mention"
    return "unresolved_context"


def parse_rows(path, raw):
    if path.suffix == ".jsonl":
        return [(f"line:{i}", json.loads(line)) for i, line in enumerate(raw.decode("utf-8").splitlines(), 1) if line.strip()]
    obj = json.loads(raw)
    # Preserve JSON pointers while counting top-level records separately.
    return [("", obj)]


def scope_for(path):
    if path.startswith("ca_agraphrag/data/"):
        if path.endswith("sft_train.jsonl"):
            return "legacy_sft_file"
        if "polished" in path:
            return "legacy_qa_polished_file"
        return "legacy_qa_original_file"
    return "legacy_corpus_kg_or_intermediate"


def build():
    paths = set()
    selector_status = []
    for selector in SELECTORS:
        found = sorted(p for p in ROOT.glob(selector) if p.is_file())
        selector_status.append({"selector": selector, "matched_files": len(found), "missing": not bool(found)})
        paths.update(found)
    inventory, hits, row_counts = [], [], {}
    compiled = {family: {width: re.compile(pattern) for width, pattern in pats.items()} for family, pats in PATTERNS.items()}
    for path in sorted(paths):
        rel = str(path.relative_to(ROOT))
        assert not rel.startswith(("data/agent_feedback", "data/condition_consistency"))
        raw = path.read_bytes()
        rows = parse_rows(path, raw)
        count = len(rows) if path.suffix == ".jsonl" else (len(rows[0][1]) if isinstance(rows[0][1], list) else None)
        row_counts[rel] = count
        inventory.append({"path": rel, "sha256": digest(raw), "size_bytes": len(raw), "record_count": count, "record_count_unit": "nonempty_jsonl_lines" if path.suffix == ".jsonl" else "top_level_array_items" if count is not None else "not_a_top_level_array", "scope": scope_for(rel)})
        for prefix, obj in rows:
            for pointer, text in leaves(obj):
                normalized = normalize(text)
                for family, widths in compiled.items():
                    for width, regex in widths.items():
                        for match in regex.finditer(normalized):
                            context = normalized[max(0, match.start()-180):match.end()+180]
                            position = prefix + pointer
                            hit = {"file": rel, "pointer": position, "family": family, "width": width, "normalized_offset": match.start(), "context_sha256": digest(context.encode()), "classification": context_class(family, context), "scope": scope_for(rel), "record_locator": prefix or "/" + pointer.lstrip("/").split("/")[0], "record_identifier": obj.get("qid") if prefix and isinstance(obj, dict) else None}
                            hit["hit_id"] = digest(json.dumps(hit, sort_keys=True).encode())[:20]
                            hits.append(hit)
    groups = []
    for family, pats in PATTERNS.items():
        group_hits = [h for h in hits if h["family"] == family]
        by_width = {}
        for width in ("narrow", "broad"):
            hs = [h for h in group_hits if h["width"] == width]
            scopes = {}
            for scope in ("legacy_corpus_kg_or_intermediate", "legacy_qa_original_file", "legacy_qa_polished_file", "legacy_sft_file"):
                scoped = [h for h in hs if h["scope"] == scope]
                scopes[scope] = {"match_occurrences": len(scoped), "matching_files": len({h["file"] for h in scoped}), "unique_field_locations": len({(h["file"], h["pointer"]) for h in scoped}), "unique_jsonl_records": len({(h["file"], h["record_locator"]) for h in scoped if h["file"].endswith(".jsonl")}), "class_counts": dict(sorted(Counter(h["classification"] for h in scoped).items()))}
            by_width[width] = {"regex": pats[width], "match_occurrences": len(hs), "matching_files": len({h["file"] for h in hs}), "class_counts": dict(sorted(Counter(h["classification"] for h in hs).items())), "by_scope": scopes}
        confirmed = [h for h in group_hits if h["classification"] == "candidate_family_mention"]
        samples = []
        for classification in sorted({h["classification"] for h in group_hits}):
            distinct = {(h["file"], h["pointer"]): h for h in group_hits if h["classification"] == classification}
            ordered = sorted(distinct.values(), key=lambda h:(not h["file"].startswith("kg_v3/entities"), h["file"],h["pointer"]))
            samples.extend(ordered[:3])
        by_file = []
        for filename in sorted({h["file"] for h in confirmed}):
            hs = [h for h in confirmed if h["file"] == filename]
            by_file.append({"path": filename, "scope": scope_for(filename), "unique_field_locations": len({h["pointer"] for h in hs}), "unique_jsonl_records": len({h["record_locator"] for h in hs}) if filename.endswith(".jsonl") else None, "record_identifier_examples": sorted({h["record_identifier"] for h in hs if h["record_identifier"]})[:5], "narrow_occurrences": sum(h["width"] == "narrow" for h in hs), "broad_occurrences": sum(h["width"] == "broad" for h in hs), "locator_examples": sorted({h["pointer"] for h in hs})[:3]})
        groups.append({"family_id": family, "status": "historical_family_mentions_identified" if confirmed else "no_resolved_candidate_family_mention_in_scanned_aliases", "matching": by_width, "resolved_family_files": by_file, "sample_locations_without_source_text": samples, "current_model_training_lineage_overlap": "unknown_not_inferred_from_file_presence", "evaluation_admitted": False})
    registry_raw = (ROOT / REGISTRY).read_bytes(); registry = json.loads(registry_raw)
    registry_matches = []
    for family, widths in compiled.items():
        for field in ("family_ids", "canonical_document_ids", "source_uris"):
            for i, value in enumerate(registry[field]):
                for width, pattern in widths.items():
                    if pattern.search(normalize(value)):
                        registry_matches.append({"family_id": family, "width": width, "field": field, "index": i, "registry_value": value, "interpretation": "screen_hit_not_automatic_same_family"})
    inputs = {"version": "radar_history_screening_inputs_v2", "selection": selector_status, "files": inventory, "file_count": len(inventory), "total_bytes": sum(f["size_bytes"] for f in inventory), "registry_metadata": {"path":REGISTRY,"sha256":digest(registry_raw),"size_bytes":len(registry_raw)}, "excluded_untouched_prefixes":["data/agent_feedback", "data/condition_consistency"], "scope_limit":"Explicit selectors only; not recursive full-workspace or model-training provenance certification"}
    contexts = defaultdict(set)
    for hit in hits:
        contexts[(hit["family"],hit["classification"])].add(hit["context_sha256"])
    original = {name: row_counts[f"ca_agraphrag/data/{name}.jsonl"] for name in ("train","dev","test")}
    report = {"version":"radar_history_alias_screening_v2", "recorded_on":"2026-10-07", "reviewer":"AI /root/coverage_weather_sources", "human_review":False, "status":"bounded_source_lineage_screening_not_gold_or_training_contamination_certification", "new_questions":0,"new_model_runs":0,"model_run_ready":False,"candidate_count":len(PATTERNS),"candidate_extension_note":"Original eight retained; METEK MRR family added by root before QA/model outputs as a source-access replacement candidate. No default unseen designation.", "normalization":"NFKC, casefold, Unicode dash folding; JSON string values AND object keys; embedded strings scanned lexically, no semantic paraphrase matching", "counting":"Occurrences per regex in normalized string fields/keys; narrow and broad may overlap and must not be added as independent mentions. File/field counts and scope breakdown retained; repeated exports are not independent source evidence.", "input_manifest_path":str(INPUTS.relative_to(ROOT)), "input_manifest_content_sha256":digest((json.dumps(inputs,ensure_ascii=False,indent=2)+"\n").encode()), "scanner_sha256":digest(Path(__file__).read_bytes()), "legacy_qa_counts":{"original_splits":original,"original_total":sum(original.values()),"polished_splits":{name:row_counts[f"ca_agraphrag/data/{name}_polished.jsonl"] for name in ("dev","test")},"sft_train_records":row_counts["ca_agraphrag/data/sft_train.jsonl"],"interpretation":"5401 original QA target records; polished variants and SFT are derivatives, not extra independent questions; existing file is not proof a current checkpoint trained on it."}, "groups":groups, "known_debug_exposure_registry_comparison":{"registry_documents":registry["source_document_count"],"declared_family_groups":registry["declared_family_group_count"],"match_details":registry_matches,"exact_candidate_family_identifier_matches":sorted(set(PATTERNS) & set(registry["family_ids"])),"interpretation":"Metadata alias scan of 10 known debug source groups only. JMA-1030 broad-prefix hits refer to the old JMA1030 group, not automatically JMA5200. Same publisher does not establish same family; no exact string overlap does not prove source isolation. Candidate byte/document identity comparison remains a separate root task."}, "context_review":{"type":"AI contextual classification with explicit conservative rules in scanner", "context_hashes_by_class":[{"family_id":family,"classification":classification,"unique_contexts":len(values),"sha256":sorted(values)} for (family,classification),values in sorted(contexts.items())], "supporting_identity_locators":[{"path":"kg_v3/entities.json","pointer":"/847","entity":"TFR/GMR","purpose":"Non-Garmin ground-mapping abbreviation lineage"},{"path":"kg_v3/entities.json","pointer":"/1584","entity":"GMR","purpose":"Bare abbreviation has limited attributes; do not upgrade all bare GMR matches to resolved Garmin absence"},{"path":"kg_v3/entities.json","pointer":"/7804","entity":"HALO","purpose":"Bare HALO entity cites the maritime halo__rt lineage"}], "limits":"Classes are AI lineage judgments, not human annotations. Bare GMR and MRR list/query matches remain unresolved. Repeated identical contexts are grouped; full raw passages and QA are not exported."}, "limits":["No candidate alias hit is not proof of complete project or pretraining non-exposure; transliterations, unnamed derivatives and unscanned backups can be missed.","Historical family overlap does not imply the newly archived official document bytes were used before; old corpus and old QA/SFT file presence are reported separately.","No relationship between these legacy files and the currently evaluated model checkpoint was certified; actual training-lineage overlap is unknown.","Old 5401 automatic QA and their rewrites remain ineligible as newly independent questions regardless of match count.","Four resolved historical families require an explicit source/QA lineage decision; do not silently rename them or treat later official versions as fully unseen families.","No public agent_feedback or condition_consistency benchmark questions were opened."]}
    return inputs, report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    inputs,report=build()
    for path,obj in ((INPUTS,inputs),(REPORT,report)):
        text=json.dumps(obj,ensure_ascii=False,indent=2)+"\n"
        if args.check:
            if not path.exists() or path.read_text()!=text:
                raise SystemExit(f"check mismatch: {path.relative_to(ROOT)}")
        else:
            if path.exists():raise SystemExit(f"refuse overwrite: {path.relative_to(ROOT)}")
            path.write_text(text)
    print(json.dumps({"status":"check_passed" if args.check else "written","files":inputs["file_count"],"legacy_qa_total":report["legacy_qa_counts"]["original_total"],"families_with_resolved_mentions":[g["family_id"] for g in report["groups"] if g["status"]=="historical_family_mentions_identified"]}))


if __name__=="__main__":
    main()
