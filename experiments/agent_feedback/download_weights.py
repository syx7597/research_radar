"""Resume the pinned public Qwen weights in verified bounded HTTP ranges.

Used only when whole-file transfers repeatedly disconnect. Existing prefixes
are retained; only a complete SHA256-verified file becomes a model weight.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
FILES = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
ROOT = Path("models/qwen2.5-3b-instruct")
CHUNK = 8 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(CHUNK), b""):
            h.update(data)
    return h.hexdigest()


def fetch_range(url, path, start, end, size):
    if path.exists() and path.stat().st_size == end - start + 1:
        return
    tmp, headers = path.with_suffix(".tmp"), path.with_suffix(".headers")
    subprocess.run(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
        "--retry", "3", "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "15",
        "--max-time", "180", "--max-filesize", str(end - start + 1),
        "--range", f"{start}-{end}", "--dump-header", str(headers), "--output", str(tmp), url], check=True)
    ranges = re.findall(r"content-range:\s*bytes (\d+)-(\d+)/(\d+)", headers.read_text(), re.I)
    if not ranges or tuple(map(int, ranges[-1])) != (start, end, size) or tmp.stat().st_size != end - start + 1:
        raise ValueError("Server returned an unexpected byte range")
    tmp.replace(path)
    headers.unlink()


def main():
    start_time = time.monotonic()
    descriptors, tasks = [], []
    for name, expected in FILES.items():
        target = ROOT / name
        if target.exists():
            if sha(target) != expected:
                raise ValueError(f"Existing weight hash mismatch: {name}")
            continue
        url = f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/{REVISION}/{name}?download=true"
        head = subprocess.check_output(["curl", "--http1.1", "--fail", "--silent", "--show-error",
                "--head", "--max-time", "30", url], text=True)
        size = int(re.search(r"x-linked-size:\s*(\d+)", head, re.I).group(1))
        etag = re.search(r'x-linked-etag:\s*"?([a-f0-9]{64})', head, re.I).group(1)
        if etag != expected:
            raise ValueError("Pinned remote weight identity differs")
        prefix = ROOT / (name + ".curl.part")
        length = prefix.stat().st_size if prefix.exists() else 0
        folder = ROOT / (name + ".ranges")
        folder.mkdir(exist_ok=True)
        parts = []
        for offset in range(length, size, CHUNK):
            end = min(size, offset + CHUNK) - 1
            path = folder / f"{offset:012d}-{end:012d}.part"
            parts.append(path)
            tasks.append((url, path, offset, end, size))
        descriptors.append((name, target, prefix, length, parts, expected))
        print(json.dumps({"file": name, "total_bytes": size, "retained_prefix_bytes": length,
                          "remaining_ranges": len(parts)}), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_range, *args) for args in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 10 == 0 or i == len(futures):
                print(json.dumps({"ranges_completed": i, "ranges_total": len(futures),
                                  "seconds": time.monotonic() - start_time}), flush=True)
    manifest = []
    for name, target, prefix, length, parts, expected in descriptors:
        tmp = target.with_suffix(".assembled")
        with tmp.open("wb") as out:
            if length:
                if prefix.stat().st_size != length:
                    raise ValueError("Retained prefix changed during download")
                with prefix.open("rb") as f:
                    shutil.copyfileobj(f, out, CHUNK)
            for part in parts:
                with part.open("rb") as f:
                    shutil.copyfileobj(f, out, CHUNK)
        if sha(tmp) != expected:
            raise ValueError(f"Assembled weight hash mismatch: {name}")
        tmp.replace(target)
        manifest.append({"file": name, "bytes": target.stat().st_size, "sha256": expected})
    output = Path("results/agent_feedback/weights_verified.json")
    output.write_text(json.dumps({"revision": REVISION, "files": manifest,
                                 "seconds": time.monotonic() - start_time}, indent=2) + "\n")
    print("WEIGHTS_SHA256_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
