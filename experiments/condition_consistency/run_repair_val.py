"""Four-GPU official-val generation, then sharded gold-free execution/scoring.

Run `generate` and `execute` separately. Execute requires a frozen v2 policy.
Raw gold is opened only by the separate frozen evaluator.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .baseline import read_records, sha256, write_json


def run_module(module, args, log, gpu=None):
    env = dict(os.environ, PYTHONHASHSEED='20260926', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log.open('w') as out:
        subprocess.run([sys.executable, '-m', 'experiments.condition_consistency.' + module,
                        *map(str, args)], env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def merge(paths, output):
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    seen = set()
    with output.open('w') as out:
        for path in paths:
            for row in read_records(path):
                if row['id'] in seen:
                    raise ValueError('Overlapping shards')
                seen.add(row['id'])
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    if seen != {f'val:{i}' for i in range(11797)}:
        raise ValueError('Incomplete official val')
    return len(seen)


def merge_execution_metadata(paths, output, generated, elapsed):
    metas = [json.loads(p.with_suffix('.jsonl.meta.json').read_text()) for p in paths]
    identity_keys = ('config', 'config_sha256', 'python_hash_seed', 'kb_sha256',
                     'repair_source_sha256', 'adapter_source_sha256',
                     'executor_source_commit', 'executor_source_sha256', 'source_sha256', 'timeout_seconds')
    for key in identity_keys:
        if any(m.get(key) != metas[0].get(key) for m in metas):
            raise ValueError(f'Inconsistent shard metadata: {key}')
    result = {k: metas[0][k] for k in identity_keys if k in metas[0]}
    for key in ('counts', 'triggers', 'rejections', 'cost'):
        if key in metas[0]:
            total = Counter()
            for meta in metas:
                total.update(meta[key])
            result[key] = dict(total)
    for key in ('questions', 'total_candidates', 'valid_candidates'):
        if key in metas[0]:
            result[key] = sum(m[key] for m in metas)
    result.update(output_sha256=sha256(output), predictions_sha256=sha256(generated),
                  input_sha256=sha256(generated), elapsed_seconds=elapsed,
                  sum_shard_elapsed_seconds=sum(m['elapsed_seconds'] for m in metas),
                  shard_metadata_sha256={p.name: sha256(p.with_suffix('.jsonl.meta.json')) for p in paths},
                  gold_usage='none', timing='elapsed is entire concurrent CPU stage, not isolated method latency')
    write_json(output.with_suffix('.jsonl.meta.json'), result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['generate', 'execute'])
    p.add_argument('--run-dir', type=Path, default=Path('results/condition_consistency/query_repair/v2'))
    p.add_argument('--input', type=Path, default=Path('data/condition_consistency/repair_splits/official_val.questions.jsonl'))
    p.add_argument('--model', default='results/condition_consistency/pilot_bart5k_seed20260926/generator/final')
    args = p.parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    rows = read_records(args.input)
    if len(rows) != 11797 or any(set(r) != {'id', 'question'} for r in rows):
        raise ValueError('Only complete question-only official val is allowed')
    started = time.perf_counter()
    if args.stage == 'generate':
        def generate_shard(shard):
            for beams in (4, 8):
                run_module('baseline', ['generate', '--input', args.input,
                    '--output', run/f'val_beam{beams}_part{shard}.jsonl', '--model', args.model,
                    '--local-files-only', '--start-index', shard*3000, '--limit', 3000,
                    '--beams', beams, '--candidates', beams, '--batch-size', 8, '--precision', 'bf16'],
                    run/f'val_beam{beams}_part{shard}_generate.log', gpu=shard)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(generate_shard, range(4)))
        for beams in (4, 8):
            merge([run/f'val_beam{beams}_part{s}.jsonl' for s in range(4)], run/f'val_beam{beams}_generated.jsonl')
    else:
        if not (run/'frozen_policy.json').exists():
            raise ValueError('Freeze v2 before executing official val')
        def execute_shard(item):
            shard, mode = item
            if mode == 'beam8':
                run_module('execute_unlabeled', ['--input', run/f'val_beam8_part{shard}.jsonl',
                    '--output', run/f'val_beam8_part{shard}_executed.jsonl'], run/f'val_beam8_part{shard}_execute.log')
            else:
                run_module('repair', ['--predictions', run/f'val_beam4_part{shard}.jsonl',
                    '--kb', 'datasets/kqa_pro/kb.json', '--mode', mode,
                    '--output', run/f'val_{mode}_part{shard}.jsonl'], run/f'val_{mode}_part{shard}_repair.log')
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(execute_shard, [(s, m) for s in range(4) for m in ('global', 'local', 'beam8')]))
        cpu_elapsed = time.perf_counter()-started
        for mode in ('global', 'local', 'beam8'):
            suffix = '_executed' if mode == 'beam8' else ''
            parts = [run/f'val_{mode}_part{s}{suffix}.jsonl' for s in range(4)]
            output = run/f'val_{mode}{suffix}.jsonl'
            merge(parts, output)
            merge_execution_metadata(parts, output, run/f'val_beam{8 if mode == "beam8" else 4}_generated.jsonl', cpu_elapsed)
        def score_mode(item):
            gpu, mode = item
            run_module('score_repairs', ['--input', run/f'val_{mode}.jsonl',
                '--output', run/f'val_{mode}_scored.jsonl', '--model', args.model],
                run/f'val_{mode}_score.log', gpu=gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(score_mode, enumerate(('global', 'local'))))
    write_json(run/f'{args.stage}_stage.json', dict(status='complete', questions=len(rows),
               input_sha256=sha256(args.input), elapsed_seconds=time.perf_counter()-started,
               driver_sha256=sha256(Path(__file__)), gold_used=False))


if __name__ == '__main__':
    main()
