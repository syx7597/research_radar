"""Build a compact, hash-linked summary after both frozen repair rounds finish."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=Path('results/condition_consistency/query_repair'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.run_dir
    output = args.output or root/'summary.json'
    if output.exists():
        raise ValueError('Preserve the first summary; use another output path')
    sources = {
        'v1_dev': root/'dev_metrics.json',
        'v1_first_holdout': root/'holdout_metrics.json',
        'v2_dev': root/'v2/dev_metrics.json',
        'v2_seen_2000_diagnostic': root/'v2/holdout_seen_metrics.json',
        'v2_official_validation': root/'v2/official_val_metrics.json'}
    summary = {'status': 'complete', 'generator_training_examples': 5000,
               'generator_training_seeds': [20260926], 'new_generator_training_in_this_run': False,
               'bounded_development_rounds': 2, 'results': {}, 'costs': {}}
    for name, path in sources.items():
        data = json.loads(path.read_text())
        summary['results'][name] = {'source': str(path), 'sha256': digest(path),
                                   'questions': data['questions'], 'metrics': data['metrics'],
                                   'comparisons': data['comparisons']}
        if 'split_role' in data:
            summary['results'][name]['split_role'] = data['split_role']
    for name in ('frozen_policy.json', 'freeze_receipt.json', 'v2/frozen_policy.json', 'v2/freeze_receipt.json'):
        summary.setdefault('freeze_sha256', {})[name] = digest(root/name)
    for mode in ('global', 'local'):
        raw = json.loads((root/f'v2/val_{mode}.jsonl.meta.json').read_text())
        score = json.loads((root/f'v2/val_{mode}_scored.jsonl.meta.json').read_text())
        summary['costs'][mode] = {'counts': raw['counts'], 'cpu_counters': raw['cost'],
                                'cpu_stage_wall_seconds': raw['elapsed_seconds'],
                                'sum_shard_seconds': raw['sum_shard_elapsed_seconds'],
                                'scored_programs': score['programs'],
                                'rescored_source_tokens': score['source_tokens'],
                                'rescored_target_tokens': score['target_tokens'],
                                'score_stage_seconds': score['elapsed_seconds']}
    summary['costs']['limitations'] = [
        'Repair CPU elapsed is the full concurrent CPU stage, not isolated per-method latency.',
        'Generation and rescoring have different batch/parallelization patterns; counts are not equal compute.',
        'v2 guard acts at selection after all caches are built; no runtime cost saving is claimed.',
        'BF16 batch shapes can change scores slightly; same checkpoint/scoring specification does not imply bitwise identical scores.']
    summary['evaluation_limits'] = [
        'v1 2000-question holdout was inspected to design v2; v2 scores there are diagnostic only.',
        'Official val was accessed in earlier project work; this is public validation, not hidden test.',
        'One trained generator seed; paired question bootstrap does not establish training-seed stability.',
        'Answer equality does not guarantee program semantic equivalence.',
        'Original labels and all questions are retained, including known data inconsistencies.']
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summary['results']['v2_official_validation']['metrics'], indent=2))


if __name__ == '__main__':
    main()
