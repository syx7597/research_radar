"""Question-only BART/KoPL baseline for the public-data feasibility experiment.

Heavy dependencies are imported only for train/generate. Run ``inspect`` first
without torch. Serialization follows KQAPro_Baselines/Bart_Program/preprocess.py.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Programs can repeat relation traversals and entire branches. Summarization
# checkpoints may inherit anti-repetition/minimum-length settings that corrupt
# valid KoPL. Construct a fresh GenerationConfig with neutral program settings.
PROGRAM_GENERATION_POLICY = {
    "do_sample": False, "num_beam_groups": 1, "diversity_penalty": 0.0,
    "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
    "min_length": 0, "min_new_tokens": 0,
    "bad_words_ids": None, "force_words_ids": None,
    "suppress_tokens": None, "begin_suppress_tokens": None,
    "constraints": None, "sequence_bias": None,
    "exponential_decay_length_penalty": None,
}


def program_generation_config(transformers: Any, token_config: Any,
                              args: argparse.Namespace) -> Any:
    return transformers.GenerationConfig(
        **PROGRAM_GENERATION_POLICY,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
        decoder_start_token_id=token_config.decoder_start_token_id,
        forced_bos_token_id=getattr(token_config, "forced_bos_token_id", None),
        forced_eos_token_id=getattr(token_config, "forced_eos_token_id", None),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beams, num_return_sequences=args.candidates,
        length_penalty=args.length_penalty, early_stopping=True,
        return_dict_in_generate=True, output_scores=True,
    )


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array/JSONL of objects: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Missing question at row {index}: {path}")
    return rows


def serialize_program(program: list[dict]) -> str:
    # No special-token deletion: separators are ordinary added tokens.
    if not isinstance(program, list) or not program:
        raise ValueError("A training program must be a nonempty list")
    chunks = []
    for item in program:
        function = item.get("function")
        inputs = item.get("inputs", [])
        if not isinstance(function, str) or not isinstance(inputs, list):
            raise ValueError("Invalid KoPL program step")
        if not all(isinstance(value, str) for value in inputs):
            raise ValueError("KoPL inputs must be strings")
        if any(marker in value for value in [function, *inputs] for marker in ("<func>", "<arg>")):
            raise ValueError("KoPL values contain an ambiguous sequence delimiter")
        chunks.append(function + "".join(" <arg> " + value for value in inputs))
    return " <func> ".join(chunks)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def row_id(row: dict, index: int) -> str:
    return str(row.get("id", row.get("source_index", index)))


def select_rows(path: str, start: int = 0, limit: int = 0) -> list[dict]:
    rows = read_records(path)
    if start < 0 or limit < 0:
        raise ValueError("start-index and limit must be nonnegative")
    return rows[start : start + limit if limit else None]


def inspect_data(args: argparse.Namespace) -> None:
    rows = select_rows(args.input, args.start_index, args.limit)
    targets = [serialize_program(row["program"]) for row in rows if "program" in row]
    print(json.dumps({
        "path": str(Path(args.input).resolve()), "sha256": sha256(args.input),
        "selected_rows": len(rows), "rows_with_program": len(targets),
        "max_question_characters": max((len(row["question"]) for row in rows), default=0),
        "max_program_characters": max(map(len, targets), default=0),
        "model_inputs": ["question"], "target_format": "official <func>/<arg> KoPL",
    }, ensure_ascii=False, indent=2))


class EncodedRows:
    def __init__(self, rows: list[dict], tokenizer: Any, args: argparse.Namespace):
        self.examples = []
        self.statistics = {"examples": len(rows), "source_tokens": 0, "target_tokens": 0,
                           "source_truncations": 0, "target_truncations": 0}
        for row in rows:
            source = tokenizer(row["question"], truncation=False)
            target = tokenizer(text_target=serialize_program(row["program"]), truncation=False)
            self.statistics["source_truncations"] += len(source["input_ids"]) > args.max_source_length
            self.statistics["target_truncations"] += len(target["input_ids"]) > args.max_target_length
            if len(source["input_ids"]) > args.max_source_length:
                source = tokenizer(row["question"], max_length=args.max_source_length, truncation=True)
            if len(target["input_ids"]) > args.max_target_length:
                target = tokenizer(text_target=serialize_program(row["program"]),
                                   max_length=args.max_target_length, truncation=True)
            self.statistics["source_tokens"] += len(source["input_ids"])
            self.statistics["target_tokens"] += len(target["input_ids"])
            self.examples.append({"input_ids": source["input_ids"],
                                  "attention_mask": source["attention_mask"],
                                  "labels": target["input_ids"]})
        if (self.statistics["source_truncations"] or self.statistics["target_truncations"]) and not args.allow_truncation:
            raise ValueError(f"Sequence truncation would occur: {self.statistics}. Increase limits or explicitly --allow-truncation.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise SystemExit("train/generate require torch, transformers, and accelerate; see BASELINE.md") from error
    return torch, transformers


def train(args: argparse.Namespace) -> None:
    torch, transformers = dependencies()
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(args.seed)
    train_rows = select_rows(args.train_file, limit=args.limit)
    dev_rows = read_records(args.dev_file) if args.dev_file else []
    if not train_rows:
        raise ValueError("Training split is empty")
    # A question duplicated across these files would compromise development.
    overlap = {row["question"].strip() for row in train_rows} & {row["question"].strip() for row in dev_rows}
    if overlap:
        raise ValueError(f"Train/development question overlap: {len(overlap)}")
    output_dir = Path(args.output_dir)
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    resume = last_checkpoint if args.resume == "auto" else args.resume
    if args.resume == "auto" and not last_checkpoint:
        raise ValueError("--resume auto requested, but no checkpoint-* directory exists")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("Output directory is nonempty. Use --resume or a new output directory.")
    tokenizer = AutoTokenizer.from_pretrained(resume or args.model, revision=args.revision,
                                               local_files_only=args.local_files_only)
    tokenizer.add_tokens(["<func>", "<arg>"], special_tokens=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(resume or args.model, revision=args.revision,
                                                 local_files_only=args.local_files_only)
    model.resize_token_embeddings(len(tokenizer))
    train_dataset = EncodedRows(train_rows, tokenizer, args)
    dev_dataset = EncodedRows(dev_rows, tokenizer, args) if dev_rows else None
    training_options = dict(
        output_dir=str(output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="linear", optim="adamw_torch",
        logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.save_steps,
        eval_steps=args.save_steps, save_total_limit=2, seed=args.seed, data_seed=args.seed,
        fp16=args.precision == "fp16", bf16=args.precision == "bf16",
        dataloader_num_workers=0, report_to=[], predict_with_generate=False,
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        load_best_model_at_end=False, disable_tqdm=not args.progress,
    )
    argument_parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    if "include_num_input_tokens_seen" in argument_parameters:
        training_options["include_num_input_tokens_seen"] = True
    evaluation_key = "eval_strategy" if "eval_strategy" in argument_parameters else "evaluation_strategy"
    training_options[evaluation_key] = "steps" if dev_dataset else "no"
    training_args = Seq2SeqTrainingArguments(**training_options)
    trainer_options = dict(model=model, args=training_args, train_dataset=train_dataset,
                           eval_dataset=dev_dataset,
                           data_collator=DataCollatorForSeq2Seq(tokenizer, model=model,
                                                               pad_to_multiple_of=8))
    tokenizer_key = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters else "tokenizer"
    trainer_options[tokenizer_key] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_options)
    metadata = {
        "args": vars(args), "train_sha256": sha256(args.train_file),
        "dev_sha256": sha256(args.dev_file) if args.dev_file else None,
        "train_tokens": train_dataset.statistics,
        "dev_tokens": dev_dataset.statistics if dev_dataset else None,
        "world_size": training_args.world_size,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * training_args.world_size,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "model_commit": getattr(model.config, "_commit_hash", None),
        "model_inputs": ["question"], "gold_entity_linking": False,
        "selection_rule": "final checkpoint; no validation-based checkpoint selection",
    }
    previous_metadata = output_dir / "run_config.json"
    if resume and previous_metadata.exists():
        previous = json.loads(previous_metadata.read_text(encoding="utf-8"))
        for key in ("train_sha256", "dev_sha256"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"Cannot resume: {key} differs from the original run")
    if trainer.is_world_process_zero():
        write_json(previous_metadata, metadata)
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.perf_counter() - started
    trainer.save_model(str(output_dir / "final"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir / "final")
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["invocation_elapsed_seconds"] = elapsed
    metrics["total_global_optimizer_steps"] = trainer.state.global_step
    metrics["dataset_tokens_per_epoch"] = train_dataset.statistics
    metrics["trainer_input_token_positions_seen"] = getattr(trainer.state, "num_input_tokens_seen", None)
    metrics["token_accounting_note"] = "Dataset token counts are exact; repeated sampled/processed tokens are not separately instrumented."
    if dev_dataset:
        metrics.update(trainer.evaluate())
    if trainer.is_world_process_zero():
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def checkpoint_fingerprint(model: str) -> dict:
    path = Path(model)
    if path.is_dir():
        # Hash small identity files; training artifacts and model identifier are
        # also retained. Do not pretend this hashes the model's full weights.
        return {str(file.relative_to(path)): sha256(file)
                for file in sorted(path.glob("*.json"))}
    return {"model_id": model}


def token_scores(sequence_ids: list[int], transition_scores: list[float], eos_id: int | None) -> tuple[float, int, bool]:
    """Sum actual generated positions, including the first EOS, never padding."""
    total = 0.0
    count = 0
    ended = False
    for token, score in zip(sequence_ids, transition_scores):
        total += float(score)
        count += 1
        if token == eos_id:
            ended = True
            break
    return total, count, ended


def generate(args: argparse.Namespace) -> None:
    torch, transformers = dependencies()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    rows = select_rows(args.input, args.start_index, args.limit)
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    if args.beams < args.candidates or args.candidates < 1:
        raise ValueError("Require beams >= candidates >= 1")
    protocol = {
        "input_sha256": sha256(args.input), "model": args.model,
        "checkpoint_config_sha256": checkpoint_fingerprint(args.model),
        "revision": args.revision, "start_index": args.start_index, "limit": args.limit,
        "seed": args.seed, "candidates": args.candidates, "beams": args.beams,
        "max_source_length": args.max_source_length, "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty, "precision": args.precision,
        "program_generation_policy": PROGRAM_GENERATION_POLICY,
        "model_inputs": ["question"], "gold_candidates_inserted": False,
        "candidate_order": "beam search score, descending",
    }
    completed = 0
    if output.exists():
        if not args.resume:
            raise ValueError("Output already exists; pass --resume or choose a new path")
        if not metadata_path.exists():
            raise ValueError("Cannot safely resume without output metadata")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("protocol") != protocol:
            raise ValueError("Cannot resume: generation protocol or input changed")
        existing = read_records(output)
        if len(existing) > len(rows):
            raise ValueError("Existing predictions exceed input length")
        for index, prediction in enumerate(existing):
            if (prediction["id"] != row_id(rows[index], args.start_index + index)
                    or prediction["question"] != rows[index]["question"]):
                raise ValueError(f"Resume output does not match input at row {index}")
        completed = len(existing)
    elif args.resume:
        raise ValueError("--resume requested but output does not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                               local_files_only=args.local_files_only)
    if any(token not in tokenizer.get_vocab() for token in ("<func>", "<arg>")):
        raise ValueError("Checkpoint tokenizer lacks trained KoPL delimiter tokens")
    if any(token in tokenizer.all_special_tokens for token in ("<func>", "<arg>")):
        raise ValueError("KoPL delimiters must be ordinary tokens to survive decoding")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision,
                                                 local_files_only=args.local_files_only,
                                                 torch_dtype=dtype).to(device)
    model.eval()
    # Transformers may migrate forced token IDs out of model.config on save;
    # preserve these IDs from the loaded generation config, but no other rules.
    generation_config = program_generation_config(transformers, model.generation_config, args)
    metadata = {"protocol": protocol, "status": "running", "completed_rows": completed,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "generation_config": generation_config.to_dict(),
                "model_commit": getattr(model.config, "_commit_hash", None)}
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    generated_tokens = 0
    source_tokens = 0
    truncations = 0
    with output.open("a" if args.resume else "w", encoding="utf-8") as handle, torch.inference_mode():
        for offset in range(completed, len(rows), args.batch_size):
            batch_rows = rows[offset : offset + args.batch_size]
            # Never pass gold program, answer, choices, or extracted gold entities.
            questions = [row["question"] for row in batch_rows]
            lengths = [len(ids) for ids in tokenizer(questions, truncation=False)["input_ids"]]
            batch_truncations = sum(length > args.max_source_length for length in lengths)
            if batch_truncations and not args.allow_truncation:
                raise ValueError("Input question exceeds max-source-length; increase it or explicitly allow truncation")
            truncations += batch_truncations
            encoded = tokenizer(questions, return_tensors="pt", padding=True, truncation=True,
                                max_length=args.max_source_length).to(device)
            source_tokens += int(encoded["attention_mask"].sum().item())
            outputs = model.generate(
                **encoded, generation_config=generation_config,
            )
            transitions = model.compute_transition_scores(outputs.sequences, outputs.scores,
                                                           getattr(outputs, "beam_indices", None),
                                                           normalize_logits=True).cpu().tolist()
            sequence_ids = outputs.sequences.cpu().tolist()
            decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            beam_scores = getattr(outputs, "sequences_scores", None)
            if beam_scores is not None:
                beam_scores = beam_scores.cpu().tolist()
            for within, row in enumerate(batch_rows):
                candidates = []
                for rank in range(args.candidates):
                    position = within * args.candidates + rank
                    # Encoder-decoder generation begins with one decoder start token.
                    generated = sequence_ids[position][1 : 1 + len(transitions[position])]
                    logprob, token_count, ended = token_scores(generated, transitions[position], tokenizer.eos_token_id)
                    generated_tokens += token_count
                    candidates.append({
                        "rank": rank, "program_text": decoded[position].strip(),
                        "sequence_logprob": logprob,
                        "normalized_logprob": logprob / max(1, token_count),
                        "beam_score": beam_scores[position] if beam_scores is not None else None,
                        "generated_tokens": token_count, "ended_with_eos": ended,
                        "hit_generation_limit": token_count >= args.max_new_tokens,
                    })
                record = {"id": row_id(row, args.start_index + offset + within),
                          "question": row["question"], "candidates": candidates}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            metadata.update(completed_rows=offset + len(batch_rows),
                            invocation_elapsed_seconds=time.perf_counter() - started,
                            invocation_source_tokens=source_tokens,
                            invocation_returned_candidate_tokens=generated_tokens,
                            invocation_source_truncations=truncations)
            write_json(metadata_path, metadata)
            print(f"Generated {metadata['completed_rows']}/{len(rows)} questions in {metadata['invocation_elapsed_seconds']:.1f}s", flush=True)
    metadata["status"] = "complete"
    write_json(metadata_path, metadata)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Validate input shape without ML dependencies")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--start-index", type=int, default=0)
    inspect_parser.add_argument("--limit", type=int, default=0)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--train-file", required=True)
    train_parser.add_argument("--dev-file")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model", default="facebook/bart-base")
    train_parser.add_argument("--revision", default="main")
    train_parser.add_argument("--seed", type=int, default=20260926)
    train_parser.add_argument("--limit", type=int, default=0, help="First N prepared train rows; 0 means all")
    train_parser.add_argument("--epochs", type=float, default=12)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    train_parser.add_argument("--save-steps", type=int, default=500)
    train_parser.add_argument("--logging-steps", type=int, default=25)
    train_parser.add_argument("--max-source-length", type=int, default=256)
    train_parser.add_argument("--max-target-length", type=int, default=512)
    train_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train_parser.add_argument("--resume", nargs="?", const="auto")
    train_parser.add_argument("--local-files-only", action="store_true")
    train_parser.add_argument("--allow-truncation", action="store_true")
    train_parser.add_argument("--progress", action="store_true")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--input", required=True)
    generate_parser.add_argument("--output", required=True, help="Predictions JSONL; gold fields are never copied")
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--revision", default="main")
    generate_parser.add_argument("--start-index", type=int, default=0)
    generate_parser.add_argument("--limit", type=int, default=0)
    generate_parser.add_argument("--seed", type=int, default=20260926)
    generate_parser.add_argument("--batch-size", type=int, default=8)
    generate_parser.add_argument("--beams", type=int, default=4)
    generate_parser.add_argument("--candidates", type=int, default=4)
    generate_parser.add_argument("--max-source-length", type=int, default=256)
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    generate_parser.add_argument("--length-penalty", type=float, default=1.0)
    generate_parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    generate_parser.add_argument("--device")
    generate_parser.add_argument("--resume", action="store_true")
    generate_parser.add_argument("--local-files-only", action="store_true")
    generate_parser.add_argument("--allow-truncation", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    {"inspect": inspect_data, "train": train, "generate": generate}[args.command](args)


if __name__ == "__main__":
    main()
