"""Read-only development-cache audit for thesis scoping, not method evaluation.

Uses gold solely for post-hoc structural differences and cached-answer scoring.
Never generates, executes a query, trains, tunes, or reads official validation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_query_repair import difference, field_violations
from experiments.condition_consistency.executor import compare_answers


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        data = [json.loads(line) for line in stream if line.strip()]
    result = {row['id']: row for row in data}
    if len(result) != len(data):
        raise ValueError(f'Duplicate IDs: {path}')
    return result


def first_valid(candidates):
    return next((i for i, candidate in enumerate(candidates) if candidate['valid']), None)


def correct(candidate, gold):
    return bool(candidate is not None and candidate['valid']
                and compare_answers(gold['answer'], candidate['prediction']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/thesis_direction_review/development_evidence.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the first audit; supply a different output path')
    base = ROOT/'results/condition_consistency/query_repair'
    paths = {'metrics': base/'v2/dev_metrics.json',
             'gold': ROOT/'data/condition_consistency/repair_splits/dev_diagnostic.gold.jsonl',
             'global': base/'dev_global_scored.jsonl',
             'beam8': base/'dev_beam8_executed.jsonl',
             'schema': ROOT/'results/condition_consistency/query_repair_audit/schema.json'}
    hashes = {key: digest(path) for key, path in paths.items()}
    metrics = json.loads(paths['metrics'].read_text())
    assert metrics['gold_sha256'] == hashes['gold']
    for name in ('global', 'beam8'):
        assert metrics['input_sha256'][name] == hashes[name]
    gold, global_rows, beam_rows = (rows(paths[name]) for name in ('gold', 'global', 'beam8'))
    details = {row['id']: row for row in metrics['per_question']}
    assert len(gold) == 500 and set(gold) == set(global_rows) == set(beam_rows) == set(details)
    schema_data = json.loads(paths['schema'].read_text())
    schema = {role: set(values) for role, values in schema_data['names_by_role'].items()}
    counters, categories, errors = Counter(), Counter(), []
    for qid, reference_gold in gold.items():
        originals = [c for c in global_rows[qid]['candidates'] if c['origin'] == 'original']
        beam8 = beam_rows[qid]['candidates']
        assert len(originals) == 4 and len(beam8) == 8
        i4, i8 = first_valid(originals), first_valid(beam8)
        c4 = None if i4 is None else originals[i4]
        c8 = None if i8 is None else beam8[i8]
        guard = c4 is not None and c4['schema_clean'] is True
        assert guard == details[qid]['guard_retained']
        assert i4 == details[qid]['selected']['first_valid']
        assert i8 == details[qid]['selected']['beam8_first_valid']
        assert correct(c4, reference_gold) == details[qid]['correct']['first_valid']
        assert correct(c8, reference_gold) == details[qid]['correct']['beam8_first_valid']
        chosen = c4 if guard else c8
        counters['questions'] += 1
        counters['guarded'] += guard
        counters['first_valid_correct'] += correct(c4, reference_gold)
        counters['beam8_correct'] += correct(c8, reference_gold)
        counters['guarded_beam8_correct'] += correct(chosen, reference_gold)
        if correct(chosen, reference_gold):
            continue
        counters['remaining_errors'] += 1
        counters['errors_without_valid_output'] += chosen is None
        diagnostic = chosen if chosen is not None else originals[0]
        diff = difference(diagnostic.get('program'), reference_gold['program'])
        categories[diff['category']] += 1
        pool = originals + beam8
        has_correct = any(correct(candidate, reference_gold) for candidate in pool)
        counters['errors_with_correct_in_original_union_pool'] += has_correct
        counters['errors_without_correct_in_original_union_pool'] += not has_correct
        violations = field_violations(diagnostic.get('program'), schema)
        counters['error_reference_with_global_schema_violation'] += bool(violations)
        if diff['category'] == 'single_field':
            edit = diff['differences'][0]
            legal = edit['predicted'] in schema[edit['role']]
            counters['single_field_' + ('globally_legal' if legal else 'globally_illegal')] += 1
        single_parameter_candidates = []
        for rank, candidate in enumerate(pool):
            if correct(candidate, reference_gold):
                continue
            change = difference(candidate.get('program'), reference_gold['program'])
            if len(change['differences']) == 1:
                single_parameter_candidates.append({'pool_index': rank,
                    'role': change['differences'][0]['role']})
        counters['errors_with_wrong_pool_candidate_one_argument_from_gold'] += bool(single_parameter_candidates)
        errors.append({'id': qid, 'reference': 'selected' if chosen is not None else 'beam4_top1_diagnostic',
                       'guard_retained': guard, 'has_correct_in_original_union_pool': has_correct,
                       'reference_diff': diff, 'reference_schema_violations': violations,
                       'one_argument_from_gold_candidates': single_parameter_candidates})
    report = {'purpose': 'Post-hoc thesis scoping on previously inspected 500-question development cache',
              'selection_rule': 'Keep schema-clean beam4 first-valid, otherwise beam8 first-valid',
              'source_sha256': hashes, 'script_sha256': digest(Path(__file__)),
              'counts': dict(counters), 'reference_difference_categories': dict(categories),
              'errors': errors,
              'limits': ['Gold structural differences are not independently verified semantic error causes.',
                         'One-argument distance to gold is a reference-program diagnostic, not achievable gain.',
                         'The union pool includes 4+8 original candidates, potentially duplicates; it is not beam12.',
                         'Known label inconsistencies and every question are retained.',
                         'This development set has been inspected repeatedly and supports no new generalization claim.',
                         'No new generation, query execution, training or official-val analysis was performed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({key: report[key] for key in ('counts', 'reference_difference_categories')}, indent=2))


if __name__ == '__main__':
    main()
