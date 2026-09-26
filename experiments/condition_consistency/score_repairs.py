"""Teacher-force the fixed generator over original and repaired programs alike.

Only question and candidate program text enter the model. Scores include the
tokenizer's BOS/EOS as in SFT, ignore padding, and are not beam-search scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .baseline import read_records, sha256, write_json, checkpoint_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    rows = read_records(args.input)
    source_meta_path = args.input.with_suffix('.jsonl.meta.json')
    source_meta = json.loads(source_meta_path.read_text())
    if source_meta['output_sha256'] != sha256(args.input):
        raise ValueError('Repair cache metadata/hash mismatch')
    if any('answer' in r or 'program' in r for r in rows):
        raise ValueError('Scoring input must be question-only candidate records')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32).to(args.device).eval()
    pairs = [(row['question'], c) for row in rows for c in row['candidates']]
    start = time.perf_counter()
    target_tokens = source_tokens = padded_target_tokens = 0
    with torch.inference_mode():
        for offset in range(0, len(pairs), args.batch_size):
            batch = pairs[offset:offset + args.batch_size]
            source = tokenizer([q for q, _ in batch], padding=True, return_tensors='pt').to(args.device)
            target = tokenizer(text_target=[c['program_text'] for _, c in batch], padding=True, return_tensors='pt').to(args.device)
            if source.input_ids.shape[1] > 256 or target.input_ids.shape[1] > 512:
                raise ValueError('Scoring would exceed the declared source/target limits')
            labels = target.input_ids.masked_fill(target.attention_mask == 0, -100)
            logits = model(**source, labels=labels).logits.float()
            per_token = -torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
            sums = per_token.sum(dim=1).cpu().tolist()
            counts = target.attention_mask.sum(dim=1).cpu().tolist()
            for (_, candidate), total, count in zip(batch, sums, counts):
                candidate['tf_logprob'] = total
                candidate['tf_tokens'] = count
                candidate['tf_mean_logprob'] = total / count
            target_tokens += sum(counts)
            source_tokens += int(source.attention_mask.sum().item())
            padded_target_tokens += labels.numel()
            if offset % (args.batch_size * 20) == 0:
                print(f'Scored {min(offset + len(batch), len(pairs))}/{len(pairs)}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temporary.replace(args.output)
    meta = dict(input_sha256=sha256(args.input), output_sha256=sha256(args.output),
                model_config=checkpoint_fingerprint(args.model), questions=len(rows),
                model_weights={p.name: sha256(p) for p in sorted(Path(args.model).glob('*.safetensors'))},
                repair_metadata=source_meta,
                programs=len(pairs), source_tokens=source_tokens, target_tokens=target_tokens,
                padded_target_tokens=padded_target_tokens, elapsed_seconds=time.perf_counter()-start,
                score_definition='sum/mean teacher-forced token log probabilities including BOS/EOS; padding masked',
                gold_used=False, scorer_sha256=sha256(Path(__file__)))
    write_json(args.output.with_suffix('.jsonl.meta.json'), meta)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
