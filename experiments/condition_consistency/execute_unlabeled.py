"""Execute generated candidates without opening any gold file."""
import argparse
from pathlib import Path
import json
import os
import time

from .baseline import read_records, sha256, write_json
from .executor import KoPLExecutor, BASELINES_COMMIT
from .execute_predictions import execute_candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kb', type=Path, default=Path('datasets/kqa_pro/kb.json'))
    p.add_argument('--candidates', type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output exists')
    start = time.perf_counter()
    executor = KoPLExecutor(args.kb)
    rows = read_records(args.input)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs or empty input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.jsonl.tmp')
    valid = total = 0
    with temp.open('w') as f:
        for row in rows:
            if set(row) != {'id', 'question', 'candidates'} or len(row['candidates']) != args.candidates:
                raise ValueError('Expected question-only generation and fixed candidate count')
            candidates = [execute_candidate(c, executor, 5.0) for c in row['candidates']]
            valid += sum(c['valid'] for c in candidates)
            total += len(candidates)
            f.write(json.dumps(dict(id=row['id'], question=row['question'], candidates=candidates), ensure_ascii=False) + '\n')
    temp.replace(args.output)
    meta = dict(input_sha256=sha256(args.input), output_sha256=sha256(args.output),
                kb_sha256=sha256(args.kb), executor_source_commit=BASELINES_COMMIT,
                questions=len(rows), total_candidates=total, valid_candidates=valid,
                elapsed_seconds=time.perf_counter()-start, gold_used=False,
                python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                adapter_source_sha256=sha256(Path(__file__).with_name('executor.py')),
                timeout_seconds=5.0, source_sha256=sha256(Path(__file__)))
    write_json(args.output.with_suffix('.jsonl.meta.json'), meta)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
