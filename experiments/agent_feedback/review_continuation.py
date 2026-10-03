"""CPU-only, frozen-development audit before conditional seed replication.

Replays saved actions, never invokes a model, changes labels, or opens holdout.
The public report contains aggregates and IDs; full changed cases stay JSONL.
"""
from collections import Counter, defaultdict
import datetime
import json
import math
import os
from pathlib import Path

from .compare import compare_predictions, digest, indexed, validate_frozen_dev
from .continuation_runs import validate_training
from .environment import MAX_CALLS, TERMINALS
from .recovery import Excluded, replay_rollout
from .training_data import read_jsonl
from experiments.condition_consistency.executor import KoPLExecutor, compare_answers

ROOT = Path('results/agent_feedback')
OUTPUT = ROOT / 'continuation_review.json'
CASES = ROOT / 'continuation_review_cases.jsonl'


def outcome(row, answer):
    if row['prediction'] is None:
        return 'unfinished'
    return 'correct' if compare_answers(answer, row['prediction']) else 'finished_wrong'


def errors(row):
    return Counter(e['observation'].get('error', 'unknown') + ':' +
                   e['observation'].get('detail', '')
                   for e in row['events'] if not e['observation']['ok'])


def signature(event):
    return json.dumps([event['tool'], event['arguments']], sort_keys=True, ensure_ascii=False)


def repeated_invalid(row):
    seen, repeats = set(), 0
    for event in row['events']:
        key = signature(event)
        if not event['observation']['ok']:
            repeats += key in seen
            seen.add(key)
    return repeats


def main():
    if os.environ.get('PYTHONHASHSEED') != '20261003':
        raise RuntimeError('Replay requires the original inference PYTHONHASHSEED=20261003')
    if OUTPUT.exists() or CASES.exists():
        raise FileExistsError('Review already exists; never overwrite evidence')
    split = json.loads((ROOT / 'split_manifest.json').read_text())
    gold_path = Path('data/agent_feedback/dev.gold.jsonl')
    gold = read_jsonl(gold_path)
    validate_frozen_dev(gold, gold_path, split)
    by_gold = indexed(gold, 'gold')
    manifest_path = ROOT / 'continuation_data.json'
    manifest = json.loads(manifest_path.read_text())
    adapter = ROOT / 'initial_sft_v1/model'
    adapter_hash = digest(adapter / 'adapter_model.safetensors')
    train_names = {'A': 'clean_sft_v1', 'C': 'recovery_sft_v1'}
    pred_names = {'A': 'clean_dev_v1', 'C': 'recovery_dev_v1'}
    paths = [manifest_path, ROOT / 'split_manifest.json', ROOT / 'protocol.json',
             ROOT / 'continuation_code_manifest_v1.json',
             ROOT / 'recovery_vs_clean_dev.json', ROOT / 'recovery_vs_program_dev.json',
             adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors',
             gold_path, Path('data/agent_feedback/dev.questions.jsonl'),
             Path('datasets/kqa_pro/kb.json'), ROOT / 'program_dev_v1.jsonl']
    data, training, issue_counts = {}, {}, Counter()
    for arm, name in train_names.items():
        validate_training(arm, name, manifest, digest(manifest_path), adapter_hash)
        inp = json.loads((ROOT / name / 'input_manifest.json').read_text())
        result = json.loads((ROOT / name / 'run_result.json').read_text())
        training[arm] = {'config': inp['config'], 'versions': inp['versions'],
                         'actual_supervised_tokens': result['actual_supervised_tokens'],
                         'actual_input_tokens': result['actual_input_tokens'],
                         'microbatches': result['microbatches'], 'seconds': result['seconds'],
                         'seed': result['seed'], 'initial_adapter_sha256':
                         inp['continuation']['adapter_weights_sha256']}
        if result['actual_input_tokens'] != manifest['arms'][arm]['input_tokens']:
            issue_counts['training_input_accounting'] += 1
        if digest(manifest['arms'][arm]['path']) != manifest['arms'][arm]['sha256']:
            issue_counts['changed_training_artifact'] += 1
        for f in ['input_manifest.json', 'run_result.json']:
            paths.append(ROOT / name / f)
        paths.append(Path(manifest['arms'][arm]['path']))
        paths.append(ROOT / f'{pred_names[arm]}.jsonl')
        data[arm] = read_jsonl(paths[-1])
        for suffix in ['.metrics.json', '.runtime.json']:
            paths.append(ROOT / f'{pred_names[arm]}{suffix}')
    config_a = {k: v for k, v in training['A']['config'].items() if k not in {'data', 'output'}}
    config_c = {k: v for k, v in training['C']['config'].items() if k not in {'data', 'output'}}
    if config_a != config_c or training['A']['versions'] != training['C']['versions']:
        issue_counts['training_configuration_mismatch'] += 1
    code = json.loads((ROOT / 'continuation_code_manifest_v1.json').read_text())['files']
    changed_code = [p for p, sha in code.items() if digest(p) != sha]
    if changed_code:
        issue_counts['changed_original_code'] += len(changed_code)
    paths.extend(Path(p) for p in code)
    paths.append(Path('external/kqa_pro_baselines/evaluate.py'))
    paths.append(Path('external/kqa_pro_baselines/Program/executor_rule.py'))
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    reproduced = compare_predictions(data['A'], data['C'], gold, go_rule=protocol['go_rule'])
    original = json.loads((ROOT / 'recovery_vs_clean_dev.json').read_text())
    comparison_matches = all(original[k] == v for k, v in reproduced.items())
    if not comparison_matches:
        issue_counts['comparison_mismatch'] += 1
    executor = KoPLExecutor('datasets/kqa_pro/kb.json')
    replay, per_arm, reviewed = {}, {}, {}
    for arm in ('A', 'C'):
        checked, mismatches = 0, []
        error_types, error_details = Counter(), Counter()
        counts = Counter()
        rows = indexed(data[arm], arm)
        reviewed[arm] = {}
        for row in data[arm]:
            ref = by_gold[row['id']]
            record = {'id': row['id'], 'outcome': outcome(row, ref['answer']),
                      'calls': row['calls'], 'invalid_calls': row['invalid_calls'],
                      'repeated_invalid_calls': repeated_invalid(row),
                      'invalid_details': dict(errors(row))}
            try:
                _, ep = replay_rollout(executor, row, ref, MAX_CALLS)
                if (ep.selected != row['selected_handle'] or
                        sum(not e['observation']['ok'] for e in ep.events) != row['invalid_calls']):
                    raise Excluded('selected_handle_or_invalid_count_mismatch')
                checked += 1
                record['selected_function'] = (ep.handles[ep.selected].function
                                                if ep.selected is not None else None)
                matches = []
                for i, h in enumerate(ep.handles):
                    if h.function not in TERMINALS:
                        continue
                    raw = h.value
                    values = [] if raw is None else ([str(x) for x in raw] if isinstance(raw, list) else [str(raw)])
                    prediction = values[0] if values else 'None'
                    if compare_answers(ref['answer'], prediction):
                        matches.append(i)
                record['handles_with_gold_matching_value'] = matches
            except Excluded as exc:
                mismatches.append({'id': row['id'], 'reason': exc.reason, 'detail': exc.detail})
            counts[record['outcome']] += 1
            counts['calls'] += row['calls']
            counts['invalid_calls'] += row['invalid_calls']
            counts['repeated_invalid_calls'] += record['repeated_invalid_calls']
            counts['episodes_with_invalid_call'] += row['invalid_calls'] > 0
            counts['correct_with_invalid_call'] += record['outcome'] == 'correct' and row['invalid_calls'] > 0
            counts['unfinished_with_gold_matching_handle'] += (record['outcome'] == 'unfinished' and
                                                               bool(record.get('handles_with_gold_matching_value')))
            for e in row['events']:
                if not e['observation']['ok']:
                    error_types[e['observation']['error']] += 1
            error_details.update(errors(row))
            reviewed[arm][row['id']] = record
            if len(reviewed[arm]) % 100 == 0:
                print(json.dumps({'arm': arm, 'replayed': len(reviewed[arm]), 'mismatches': len(mismatches)}), flush=True)
        replay[arm] = {'checked': checked, 'mismatches': mismatches}
        issue_counts['replay_mismatches'] += len(mismatches)
        runtime = json.loads((ROOT / f'{pred_names[arm]}.runtime.json').read_text())
        metrics = json.loads((ROOT / f'{pred_names[arm]}.metrics.json').read_text())
        summ = reproduced['arms']['baseline' if arm == 'A' else 'candidate']
        if (metrics['correct'] != counts['correct'] or metrics['total'] != len(gold) or
                runtime['totals'] != {**summ['totals'], 'questions': len(gold)}):
            issue_counts['metrics_or_runtime_mismatch'] += 1
        per_arm[arm] = {**dict(counts), 'invalid_call_fraction': counts['invalid_calls'] / counts['calls'],
                        'valid_calls': counts['calls'] - counts['invalid_calls'],
                        'error_types': dict(error_types), 'common_errors': error_details.most_common(12),
                        'generation_seconds': runtime['seconds'], **summ['totals']}
    matrix, changed, groups = Counter(), [], defaultdict(lambda: Counter())
    for ref in gold:
        id = ref['id']
        a, c = reviewed['A'][id], reviewed['C'][id]
        matrix[a['outcome'] + ' -> ' + c['outcome']] += 1
        difference = int(c['outcome'] == 'correct') - int(a['outcome'] == 'correct')
        functions = [step['function'] for step in ref['program']]
        # Overlapping, descriptive groups fixed from reference functions, not model outcomes.
        labels = ['all', 'reference_steps_le_5' if len(functions) <= 5 else 'reference_steps_gt_5']
        if any(fn.startswith('QFilter') or fn in {'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'} for fn in functions):
            labels.append('qualifier')
        if any(fn in {'And', 'Or'} for fn in functions):
            labels.append('set_operation')
        if any(fn in {'SelectAmong', 'SelectBetween', 'FilterNum', 'FilterYear', 'FilterDate',
                      'VerifyNum', 'VerifyYear', 'VerifyDate', 'Count'} for fn in functions):
            labels.append('comparison_numeric_or_count')
        for label in labels:
            groups[label]['questions'] += 1
            groups[label]['A_correct'] += a['outcome'] == 'correct'
            groups[label]['C_correct'] += c['outcome'] == 'correct'
            groups[label]['net_corrected'] += difference
        if difference:
            changed.append({'id': id, 'change': 'corrected' if difference > 0 else 'regressed',
                            'A': a, 'C': c, 'reference_functions': functions,
                            'question': ref['question'], 'answer': ref['answer'], 'program': ref['program'],
                            'A_prediction': indexed(data['A'], 'A')[id]['prediction'],
                            'C_prediction': indexed(data['C'], 'C')[id]['prediction']})
    known = json.loads((ROOT / 'gold_replay.json').read_text())['splits']['dev']['mismatches']
    known_ids = [r['id'] for r in known]
    known_status = {id: {'A': reviewed['A'][id]['outcome'], 'C': reviewed['C'][id]['outcome']} for id in known_ids}
    # Repeating comparisons on the same questions does not create new independent test examples.
    gate = not any(issue_counts.values()) and reproduced['pilot_investment_decision']['continue_investment']
    report = {'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'CPU replay of all 1000 frozen dev episodes; no new inference, training, or holdout',
              'pythonhashseed': os.environ['PYTHONHASHSEED'], 'training': training,
              'comparison_exactly_reproduced': comparison_matches, 'comparison': reproduced,
              'replay': replay, 'arms': per_arm, 'outcome_transition_matrix': dict(matrix),
              'changed_case_ids': {kind: [r['id'] for r in changed if r['change'] == kind]
                                   for kind in ('corrected', 'regressed')},
              'descriptive_overlapping_groups': {k: dict(v) for k, v in groups.items()},
              'known_reference_answer_mismatches_retained': known_status,
              'issues': dict(issue_counts), 'changed_original_code': changed_code,
              'gate': {'allow_paired_replication': gate,
                       'scope': 'same initial adapter and frozen corpus; only continuation training RNG changes',
                       'allow_rl': False, 'allow_semantic_preference_training': False},
              'limitations': ['Post-hoc descriptive groups are not independent significance claims.',
                             'A gold-matching intermediate value can occur by chance; not proof of semantic correctness.',
                             'Fewer invalid calls does not alone establish causal use of diagnostic feedback.',
                             'No full initial-model or trajectory-resampling seed variation is tested.'],
              'inputs_sha256': {str(p): digest(p) for p in paths},
              'review_script_sha256': digest(__file__)}
    with CASES.open('x') as f:
        for row in changed:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report['private_changed_cases'] = {'path': str(CASES), 'sha256': digest(CASES), 'count': len(changed)}
    with OUTPUT.open('x') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'gate': report['gate'], 'issues': report['issues'], 'matrix': dict(matrix),
                      'arms': per_arm, 'groups': report['descriptive_overlapping_groups']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
