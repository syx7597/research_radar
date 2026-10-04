"""Render generic failure diagnostics while retaining the frozen real executor.

This separate entry reuses the frozen greedy development generator. Only the
model-visible error/detail strings change; real events and execution state do
not. Successful observations, including empty/zero/no, are untouched.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import datetime
import json
import os
from pathlib import Path

from . import inference as frozen_inference
from .compare import digest
from .training_data import read_jsonl

MODE = "generic_failure_v1"
GENERIC_ERROR = "ExecutionError"
GENERIC_DETAIL = "execution_failed"
SEED = 20261003
FROZEN_INFERENCE_SHA256 = "c16a656cb971b38c3edd723b970662b198b5805b697fa13fd05def1319fffa28"
FROZEN_ENVIRONMENT_SHA256 = "adb81075b2e0db337f06e9afeb78116e384bb18151d6d12aecbd6ab0f4594db2"


def visible_observation(observation, mode=MODE):
    """Return a fresh view, never mutate the dictionary already stored in events."""
    if mode not in (MODE, "original"):
        raise ValueError("Unknown feedback rendering mode")
    if not isinstance(observation, dict) or type(observation.get("ok")) is not bool:
        raise ValueError("An observation requires an explicit boolean success flag")
    view = deepcopy(observation)
    if mode == MODE and observation["ok"] is False:
        if not isinstance(observation.get("error"), str) or not isinstance(observation.get("detail"), str):
            raise ValueError("Frozen failure observations require error and detail strings")
        view["error"], view["detail"] = GENERIC_ERROR, GENERIC_DETAIL
    return view


@contextmanager
def rendering_intervention(mode=MODE):
    """Wrap only the return value consumed by the original message renderer."""
    if mode not in (MODE, "original"):
        raise ValueError("Unknown feedback rendering mode")
    original = frozen_inference.execute_response
    counters = {"observations": 0, "failed_observations": 0, "masked_observations": 0}

    def render(episode, text):
        observation = original(episode, text)
        counters["observations"] += 1
        counters["failed_observations"] += observation["ok"] is False
        counters["masked_observations"] += mode == MODE and observation["ok"] is False
        return visible_observation(observation, mode)

    frozen_inference.execute_response = render
    try:
        yield counters
    finally:
        frozen_inference.execute_response = original


def validate_generation_scope(args):
    if os.environ.get("PYTHONHASHSEED") != str(SEED) or args.seed != SEED:
        raise ValueError("Frozen execution and generation require seed 20261003")
    if args.program or args.offset != 0 or args.limit != 0:
        raise ValueError("Diagnostic generation requires the entire agent development split")
    if args.feedback_mode != MODE:
        raise ValueError("The fixed diagnostic only permits generic_failure_v1")
    spec = json.loads(Path("results/agent_feedback/split_manifest.json").read_text())["splits"]["dev"]
    questions = Path(args.questions)
    if (questions.resolve() != Path("data/agent_feedback/dev.questions.jsonl").resolve()
            or questions.resolve() != Path(spec["questions"]["path"]).resolve()
            or digest(questions) != spec["questions"]["sha256"]):
        raise ValueError("Only the original frozen development questions are permitted")
    rows = read_jsonl(questions)
    frozen_inference.validate_questions(rows)
    if len(rows) != 500 or spec["count"] != 500 or {r["id"] for r in rows} != set(spec["ids"]):
        raise ValueError("Expected complete frozen 500-question coverage")
    paths = {"inference": Path(frozen_inference.__file__),
             "environment": Path(__file__).with_name("environment.py"),
             "renderer": Path(__file__), "questions": questions, "kb": Path(args.kb),
             "adapter_config": Path(args.adapter) / "adapter_config.json",
             "adapter_weights": Path(args.adapter) / "adapter_model.safetensors"}
    hashes = {key: digest(path) for key, path in paths.items()}
    if hashes["inference"] != FROZEN_INFERENCE_SHA256 or hashes["environment"] != FROZEN_ENVIRONMENT_SHA256:
        raise ValueError("Original inference or execution source differs from the frozen implementation")
    return hashes


def generate(args):
    output = Path(args.output)
    sidecar, started = output.with_suffix(".feedback.json"), output.with_suffix(".feedback_started.json")
    if any(path.exists() for path in (output, sidecar, started, output.with_suffix(".runtime.json"))):
        raise FileExistsError("Diagnostic evidence exists; never overwrite or resume")
    hashes = validate_generation_scope(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"feedback_mode": MODE, "masked_fields": ["error", "detail"],
              "replacement": {"error": GENERIC_ERROR, "detail": GENERIC_DETAIL},
              "preserved": "ok flag, every other field, all successful observations, real events, handles, types, state and budgets",
              "questions": 500, "config": vars(args), "inputs_sha256": hashes,
              "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with started.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    with rendering_intervention() as counts:
        frozen_inference.generate(args)
    record.update(status="completed", rendering_counts=counts,
                  predictions_sha256=digest(output), runtime_sha256=digest(output.with_suffix(".runtime.json")))
    with sidecar.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kb", default="datasets/kqa_pro/kb.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-generated", type=int, default=2048)
    p.add_argument("--program", action="store_true")
    p.add_argument("--feedback-mode", choices=(MODE,), default=MODE)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
