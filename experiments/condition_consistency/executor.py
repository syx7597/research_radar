"""Strict data-only KoPL adapter around unmodified official implementations.

Sources are downloaded separately (MIT); no generated Python is ever evaluated.
The default is the RuleExecutor released alongside this KQA Pro dataset.
Modern KoPL is an explicit comparison backend and has different semantics.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
KOPL_COMMIT = "486c07bd83ea32268ff90fa6942cc9f7e1574ae5"
BASELINES_COMMIT = "14d87cd22eb79f702fd4ad5c09240bef126d9dce"
# (dependencies, string arguments), matching official Program/executor_rule.py.
SIGNATURES = {
    "FindAll": (0, 0), "Find": (0, 1), "FilterConcept": (1, 1),
    "FilterStr": (1, 2), "FilterNum": (1, 3), "FilterYear": (1, 3),
    "FilterDate": (1, 3), "QFilterStr": (1, 2), "QFilterNum": (1, 3),
    "QFilterYear": (1, 3), "QFilterDate": (1, 3), "Relate": (1, 2),
    "And": (2, 0), "Or": (2, 0), "What": (1, 0), "Count": (1, 0),
    "SelectBetween": (2, 2), "SelectAmong": (1, 2), "QueryAttr": (1, 1),
    "QueryAttrUnderCondition": (1, 3), "VerifyStr": (1, 1),
    "VerifyNum": (1, 2), "VerifyYear": (1, 2), "VerifyDate": (1, 2),
    "QueryRelation": (2, 0), "QueryAttrQualifier": (1, 3),
    "QueryRelationQualifier": (2, 2),
}
TERMINALS = frozenset(SIGNATURES) - {
    "FindAll", "Find", "FilterConcept", "FilterStr", "FilterNum", "FilterYear",
    "FilterDate", "QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate",
    "Relate", "And", "Or",
}
ENTITY_FUNCTIONS = frozenset(SIGNATURES) - TERMINALS
VALUE_FUNCTIONS = {"QueryAttr", "QueryAttrUnderCondition", "QueryAttrQualifier", "QueryRelationQualifier"}
FACT_FUNCTIONS = {"FilterStr", "FilterNum", "FilterYear", "FilterDate", "QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate", "Relate"}


def infer_dependencies(functions: list[str]) -> list[list[int]]:
    """The official forward() branch-stack convention, with 0-based indices."""
    stack, result = [], []
    for i, function in enumerate(functions):
        if function not in SIGNATURES:
            raise ValueError(f"Unknown KoPL function: {function!r}")
        arity = SIGNATURES[function][0]
        if arity == 0:
            stack.append(i - 1)
            deps = []
        elif arity == 2:
            if len(stack) < 2:
                raise ValueError(f"Missing branch for {function} at {i}")
            deps = [stack.pop(), i - 1]
        else:
            deps = [i - 1]
        if any(d < 0 or d >= i for d in deps):
            raise ValueError(f"Invalid dependency for {function} at {i}")
        result.append(deps)
    return result


def validate_program(program: list[dict]) -> list[dict]:
    if not isinstance(program, list) or not 1 <= len(program) <= 64:
        raise ValueError("Program must contain 1..64 steps")
    functions = [step.get("function") if isinstance(step, dict) else None for step in program]
    inferred = infer_dependencies(functions)
    normalized = []
    for i, step in enumerate(program):
        function, inputs = step["function"], step.get("inputs")
        dep_count, arg_count = SIGNATURES[function]
        if not isinstance(inputs, list) or len(inputs) != arg_count:
            raise ValueError(f"Wrong argument count at step {i}")
        if not all(isinstance(x, str) and len(x) <= 2048 for x in inputs):
            raise ValueError(f"Arguments must be bounded strings at step {i}")
        deps = step.get("dependencies", inferred[i])
        if (not isinstance(deps, list) or len(deps) != dep_count
                or any(type(d) is not int or d < 0 or d >= i for d in deps)):
            raise ValueError(f"Invalid dependency at step {i}")
        expected = VALUE_FUNCTIONS if function.startswith("Verify") else ENTITY_FUNCTIONS
        if any(functions[d] not in expected for d in deps):
            raise ValueError(f"Incompatible dependency type at step {i}")
        if function.startswith("QFilter") and functions[deps[0]] not in FACT_FUNCTIONS:
            raise ValueError(f"Qualifier filter requires attached facts at step {i}")
        if function in {"FilterNum", "FilterYear", "FilterDate", "QFilterNum", "QFilterYear", "QFilterDate", "VerifyNum", "VerifyYear", "VerifyDate"}:
            if inputs[-1] not in {"=", "!=", "<", ">"}:
                raise ValueError(f"Unsupported comparison at step {i}")
        if function == "Relate" and inputs[-1] not in {"forward", "backward"}:
            raise ValueError(f"Unsupported relation direction at step {i}")
        if function == "SelectBetween" and inputs[-1] not in {"greater", "less"}:
            raise ValueError(f"Unsupported comparison at step {i}")
        if function == "SelectAmong" and inputs[-1] not in {"largest", "smallest"}:
            raise ValueError(f"Unsupported selection at step {i}")
        normalized.append({"function": function, "inputs": inputs[:], "dependencies": deps[:]})
    if functions[-1] not in TERMINALS:
        raise ValueError("Program must end with an answer-producing function")
    return normalized


def parse_program(text: str) -> list[dict]:
    if not isinstance(text, str) or len(text) > 65536:
        raise ValueError("Program text must be a bounded string")
    steps = []
    for chunk in text.strip().split("<func>"):
        parts = [part.strip() for part in chunk.split("<arg>")]
        steps.append({"function": parts[0], "inputs": parts[1:]})
    return validate_program(steps)


def serialize_program(program: list[dict]) -> str:
    normalized = validate_program(program)
    inferred = infer_dependencies([step["function"] for step in normalized])
    if any(step["dependencies"] != deps for step, deps in zip(normalized, inferred)):
        raise ValueError("Explicit dependencies cannot be represented by the official linear format")
    if any("<func>" in arg or "<arg>" in arg for step in normalized for arg in step["inputs"]):
        raise ValueError("Argument contains a reserved delimiter")
    return " <func> ".join(" <arg> ".join([step["function"], *step["inputs"]]) for step in normalized)


def _add_dependencies() -> None:
    runtime = ROOT / "external/runtime_deps"
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))


@lru_cache(maxsize=1)
def _official_equal():
    _add_dependencies()
    path = ROOT / "external/kqa_pro_baselines/evaluate.py"
    if not path.is_file():
        raise FileNotFoundError(f"Download the official Baselines checkout first: {path}")
    spec = importlib.util.spec_from_file_location("_official_kqa_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.whether_equal


def compare_answers(answer, prediction) -> bool:
    """Use KQA Pro's official numeric/date comparison, with invalid != correct."""
    return prediction is not None and bool(_official_equal()(str(answer), str(prediction)))


class KoPLExecutor:
    def __init__(self, kb_path: str | Path, source_root: str | Path | None = None,
                 backend: str = "baseline"):
        _add_dependencies()
        if backend not in {"baseline", "modern"}:
            raise ValueError("backend must be baseline or modern")
        self.backend = backend
        if backend == "baseline":
            source = Path(source_root) if source_root else ROOT / "external/kqa_pro_baselines"
            if not (source / "Program/executor_rule.py").is_file():
                raise FileNotFoundError(f"Download the official Baselines checkout first: {source}")
            sys.path.insert(0, str(source))
            from Program.executor_rule import RuleExecutor
            with contextlib.redirect_stdout(io.StringIO()):
                self.engine = RuleExecutor({}, str(kb_path))
            return
        source = Path(source_root) if source_root else ROOT / "external/kopl"
        if not (source / "src/kopl/kopl.py").is_file():
            raise FileNotFoundError(f"Download the official KoPL checkout first: {source}")
        sys.path.insert(0, str(source / "src"))
        from kopl.kopl import KoPLEngine
        with open(kb_path, encoding="utf-8") as handle:
            kb = json.load(handle)
        # These are field renames only. The source file is never modified.
        for concept in kb["concepts"].values():
            if "subclassOf" not in concept:
                concept["subclassOf"] = concept.pop("instanceOf")
        for entity in kb["entities"].values():
            for relation in entity["relations"]:
                if "relation" not in relation:
                    relation["relation"] = relation.pop("predicate")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.engine = KoPLEngine(kb)

    def execute(self, program: list[dict] | str) -> dict:
        try:
            steps = parse_program(program) if isinstance(program, str) else validate_program(program)
            memory = []
            for step in steps:
                # Function names and dependency indices were checked above.
                deps = [memory[i] for i in step["dependencies"]]
                if self.backend == "baseline":
                    value = getattr(self.engine, step["function"])(deps, step["inputs"])
                else:
                    name = "QueryName" if step["function"] == "What" else step["function"]
                    value = getattr(self.engine, name)(*deps, *step["inputs"])
                memory.append(value)
            raw = memory[-1]
            answers = [] if raw is None else ([str(value) for value in raw] if isinstance(raw, list) else [str(raw)])
            # Match official Bart_Program/predict.py first-answer convention.
            prediction = answers[0] if answers else "None"
            return {"valid": True, "prediction": prediction, "error": None,
                    "answers": answers, "empty_result": not answers}
        except Exception as exc:
            return {"valid": False, "prediction": None, "error": f"{type(exc).__name__}: {exc}",
                    "answers": [], "empty_result": False}
