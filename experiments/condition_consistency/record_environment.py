"""Record runtime, source and model identities without credentials or host IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    root = Path("experiments/condition_consistency")
    record = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
        "source_sha256": {str(p): digest(p) for p in sorted(root.iterdir())
                          if p.is_file() and p.suffix in {".py", ".sh", ".md", ".txt"}},
        "model_files_sha256": {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
    }
    source = args.model / "source.json"
    if source.exists():
        record["model_source"] = json.loads(source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    freeze = subprocess.check_output([str(Path(__import__("sys").executable)), "-m", "pip", "freeze"], text=True)
    args.output.with_name("environment_freeze.txt").write_text(freeze)


if __name__ == "__main__":
    main()
