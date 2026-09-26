"""Calibrate on declared training-internal labels; evaluate a frozen policy once.

The selector only sees question/candidates. Gold is joined outside the selector
for calibration or final scoring, never to generate or choose repair positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import read_records, sha256, write_json
from .executor import compare_answers
from .evaluate import paired_comparison


def baseline_index(row):
    return next((i for i, c in enumerate(row['candidates'])
                 if c.get('origin', 'original') == 'original' and c['valid']), None)


def choose_repair(row, policy, *, unconditional=False):
    """No gold, answer match, emptiness preference or runtime outcome oracle."""
    candidates = row['candidates']
    fallback = baseline_index(row)
    if policy.get('disabled') and not unconditional:
        return fallback
    eligible = [(i, c) for i, c in enumerate(candidates)
                if c.get('origin') == 'repair' and c['valid'] and c.get('schema_clean', False)
                and (unconditional or c['match_score'] >= policy['min_match'])]
    if not eligible:
        return fallback
    def score(c):
        return c['tf_mean_logprob'] + policy['alpha'] * c['match_score']
    selected, candidate = max(eligible, key=lambda pair: (score(pair[1]), -pair[0]))
    if unconditional:
        return selected
    reference = fallback if fallback is not None else candidate['parent_index']
    # Add match evidence to the proposed repair; original remains the fixed
    # first-valid fallback, not a newly reranked original candidate pool.
    delta = score(candidate) - candidates[reference]['tf_mean_logprob']
    return selected if delta >= policy['margin'] else fallback


def indexed(rows):
    result = {str(r['id']): r for r in rows}
    if len(result) != len(rows) or not rows:
        raise ValueError('Empty input or duplicate IDs')
    return result


def load_aligned(paths, gold_path):
    gold = indexed(read_records(gold_path))
    inputs = {}
    for name, path in paths.items():
        cache = indexed(read_records(path))
        if set(cache) != set(gold):
            raise ValueError(f'{name}: mismatched ID sets')
        for qid, row in cache.items():
            if row['question'] != gold[qid]['question']:
                raise ValueError(f'Question mismatch: {qid}')
            if 'answer' in row or 'program' in row:
                raise ValueError('Repair selection caches must contain no gold')
            if name == 'beam8':
                if len(row['candidates']) != 8:
                    raise ValueError('Expected exactly 8 beam candidates')
            else:
                origins = [c.get('origin') for c in row['candidates']]
                if origins[:4] != ['original'] * 4 or any(o != 'repair' for o in origins[4:]) or len(origins) > 8:
                    raise ValueError('Expected 4 originals followed by at most 4 repairs')
            for c in row['candidates']:
                if not isinstance(c.get('valid'), bool):
                    raise ValueError('Official execution required for every candidate')
        inputs[name] = cache
    return gold, inputs


def verify_split(args, gold, split):
    manifest = json.loads(args.manifest.read_text())
    expected = manifest['splits'][split]
    if sha256(args.gold) != expected['gold']['sha256'] or set(gold) != set(expected['ids']):
        raise ValueError(f'Gold input does not match frozen {split} split')
    return manifest


def verify_cache_provenance(args):
    split_manifest = json.loads(args.manifest.read_text())
    checkpoint = json.loads(args.checkpoint_manifest.read_text())['files']
    expected_weights = {k: v['sha256'] for k, v in checkpoint.items() if k.endswith('.safetensors')}
    expected_config = {k: v['sha256'] for k, v in checkpoint.items() if k.endswith('.json')}
    identity = {}
    for mode in ('global', 'local'):
        path = getattr(args, mode + '_cache')
        meta = json.loads(path.with_suffix('.jsonl.meta.json').read_text())
        repair_meta = meta['repair_metadata']
        if (meta['output_sha256'] != sha256(path)
                or meta['input_sha256'] != repair_meta['output_sha256']
                or meta['scorer_sha256'] != sha256(Path(__file__).with_name('score_repairs.py'))
                or repair_meta['repair_source_sha256'] != sha256(Path(__file__).with_name('repair.py'))
                or repair_meta['kb_sha256'] != split_manifest['source_sha256']['kb.json']
                or repair_meta.get('python_hash_seed') != '20260926'
                or repair_meta.get('adapter_source_sha256') != sha256(Path(__file__).with_name('executor.py'))
                or meta['model_weights'] != expected_weights or meta['model_config'] != expected_config
                or repair_meta['config']['mode'] != mode
                or repair_meta['config']['max_original'] != 4
                or repair_meta['config']['max_repairs'] != 4):
            raise ValueError(f'{mode}: cache/model/code/budget provenance mismatch')
        identity[mode] = dict(repair_config_sha256=repair_meta['config_sha256'],
                              executor_source_sha256=repair_meta['executor_source_sha256'],
                              adapter_source_sha256=repair_meta['adapter_source_sha256'],
                              executor_source_commit=repair_meta['executor_source_commit'])
    return identity


def outcome(row, index, answer):
    return bool(index is not None and row['candidates'][index]['valid']
                and compare_answers(answer, row['candidates'][index]['prediction']))


def calibrate(args):
    gold, inputs = load_aligned({'global': args.global_cache, 'local': args.local_cache}, args.gold)
    verify_split(args, gold, 'calibration')
    identity = verify_cache_provenance(args)
    config = {'stage': 'frozen_on_training_internal_calibration',
              'split_manifest_sha256': sha256(args.manifest),
              'checkpoint_manifest_sha256': sha256(args.checkpoint_manifest),
              'cache_identity': identity,
              'calibration_ids': sorted(gold), 'calibration_gold_sha256': sha256(args.gold),
              'policy_family': 'mean teacher-forced logprob + alpha*field_match; fixed first-valid fallback',
              'tie_break': 'highest correct count, fewest regressions, fewest switches, simplest scoring',
              'selector_sha256': sha256(Path(__file__)),
              'repair_sha256': sha256(Path(__file__).with_name('repair.py')),
              'scorer_sha256': sha256(Path(__file__).with_name('score_repairs.py')),
              'policies': {}, 'calibration': {}}
    for mode, rows in inputs.items():
        base = {qid: outcome(row, baseline_index(row), gold[qid]['answer']) for qid, row in rows.items()}
        grid = [dict(disabled=True, alpha=0.0, min_match=0.0, margin=0.0)]
        grid.extend(dict(disabled=False, alpha=a, min_match=m, margin=t)
                    for a in (0.0, 0.25, 0.5)
                    for m in (0.0, 0.4, 0.6, 0.8)
                    for t in (-0.25, 0.0, 0.1, 0.25, 0.5, 1.0))
        trials = []
        for policy in grid:
            correct = corrected = regressed = switches = 0
            for qid, row in rows.items():
                pick = choose_repair(row, policy)
                ok = outcome(row, pick, gold[qid]['answer'])
                correct += ok
                corrected += ok and not base[qid]
                regressed += base[qid] and not ok
                switches += pick != baseline_index(row)
            trials.append(dict(policy=policy, correct=correct, corrected=corrected,
                               regressed=regressed, switches=switches))
        best = max(trials, key=lambda t: (t['correct'], -t['regressed'], -t['switches'],
                                          -t['policy']['alpha']))
        config['policies'][mode] = best['policy']
        config['calibration'][mode] = dict(questions=len(rows), baseline_correct=sum(base.values()),
                                         selected=best, trials=trials,
                                         input_sha256=sha256(getattr(args, mode + '_cache')))
    write_json(args.output, config)
    print(json.dumps({m: c['selected'] for m, c in config['calibration'].items()}, indent=2))


def evaluate(args):
    paths = {'global': args.global_cache, 'local': args.local_cache}
    if args.beam8_cache:
        paths['beam8'] = args.beam8_cache
    gold, caches = load_aligned(paths, args.gold)
    verify_split(args, gold, args.split)
    config = json.loads(args.policy.read_text())
    if (verify_cache_provenance(args) != config['cache_identity']
            or sha256(args.checkpoint_manifest) != config['checkpoint_manifest_sha256']):
        raise ValueError('Cache identity changed after calibration')
    if args.beam8_cache:
        beam_meta = json.loads(args.beam8_cache.with_suffix('.jsonl.meta.json').read_text())
        if (beam_meta['output_sha256'] != sha256(args.beam8_cache)
                or beam_meta.get('python_hash_seed') != '20260926'
                or beam_meta.get('adapter_source_sha256') != sha256(Path(__file__).with_name('executor.py'))
                or beam_meta['kb_sha256'] != json.loads(args.manifest.read_text())['source_sha256']['kb.json']):
            raise ValueError('Beam8 execution provenance/hash seed mismatch')
    if config['split_manifest_sha256'] != sha256(args.manifest):
        raise ValueError('Split manifest changed after policy freeze')
    if set(gold) & set(config['calibration_ids']):
        raise ValueError('Evaluation overlaps calibration IDs')
    if (config['selector_sha256'] != sha256(Path(__file__))
            or config['repair_sha256'] != sha256(Path(__file__).with_name('repair.py'))
            or config['scorer_sha256'] != sha256(Path(__file__).with_name('score_repairs.py'))):
        raise ValueError('Method code changed after calibration freeze')
    correctness, details = {}, []
    for qid, target in gold.items():
        row = caches['global'][qid]
        other = caches['local'][qid]
        originals = [c for c in row['candidates'] if c['origin'] == 'original']
        local_originals = [c for c in other['candidates'] if c['origin'] == 'original']
        if [(c['program_text'], c['prediction'], c['valid']) for c in originals] != [(c['program_text'], c['prediction'], c['valid']) for c in local_originals]:
            raise ValueError('Global/local originals disagree')
        picks = {'top1': (row, 0 if originals else None), 'first_valid': (row, baseline_index(row))}
        for mode in ('global', 'local'):
            r = caches[mode][qid]
            picks[mode + '_repair'] = (r, choose_repair(r, config['policies'][mode]))
            picks[mode + '_no_acceptance'] = (r, choose_repair(r, config['policies'][mode], unconditional=True))
        if 'beam8' in caches:
            b = caches['beam8'][qid]
            picks['beam8_first_valid'] = (b, baseline_index(b))
        correct = {method: outcome(r, idx, target['answer']) for method, (r, idx) in picks.items()}
        for mode in ('global', 'local'):
            correct[mode + '_oracle'] = any(c['valid'] and compare_answers(target['answer'], c['prediction'])
                                           for c in caches[mode][qid]['candidates'])
        for method, ok in correct.items():
            correctness.setdefault(method, []).append(ok)
        details.append(dict(id=qid, correct=correct,
                            selected={m: i for m, (_, i) in picks.items()},
                            repairs={m: sum(c.get('origin') == 'repair' for c in caches[m][qid]['candidates'])
                                     for m in ('global', 'local')}))
    n = len(gold)
    report = dict(split=args.split, questions=n, gold_sha256=sha256(args.gold), policy_sha256=sha256(args.policy),
                  input_sha256={m: sha256(p) for m, p in paths.items()},
                  metrics={m: dict(correct=sum(v), questions=n, accuracy=sum(v)/n) for m, v in correctness.items()},
                  comparisons={}, per_question=details)
    for method in correctness:
        if method in ('first_valid', 'top1') or method.endswith('_oracle'):
            continue
        report['comparisons'][method + '_vs_first_valid'] = paired_comparison(correctness['first_valid'], correctness[method], seed=20260926)
    for control in ('global_repair', 'beam8_first_valid'):
        if control in correctness:
            report['comparisons']['local_repair_vs_' + control] = paired_comparison(correctness[control], correctness['local_repair'], seed=20260926)
    report['coverage'] = {m: dict(questions_with_repairs=sum(d['repairs'][m] > 0 for d in details),
                                  new_candidates=sum(d['repairs'][m] for d in details)) for m in ('global', 'local')}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_question'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['calibrate', 'evaluate'])
    p.add_argument('--global-cache', type=Path, required=True)
    p.add_argument('--local-cache', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path)
    p.add_argument('--beam8-cache', type=Path)
    p.add_argument('--manifest', type=Path, default=Path('results/condition_consistency/query_repair/split_manifest.json'))
    p.add_argument('--checkpoint-manifest', type=Path, default=Path('results/condition_consistency/pilot_bart5k_seed20260926/checkpoint_manifest.json'))
    p.add_argument('--split', choices=['design', 'dev_diagnostic', 'holdout'], default='holdout')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing decision/report')
    if args.command == 'evaluate' and not args.policy:
        p.error('--policy required for evaluate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (calibrate if args.command == 'calibrate' else evaluate)(args)


if __name__ == '__main__':
    main()
