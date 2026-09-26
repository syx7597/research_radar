"""Check the staged/public Git tree without printing credential values.

This is a focused publication guard, not a full secret-scanning product.
"""
import re
import subprocess
import sys

BLOCKED = ("manuals/", "data/", "radar_corpus/", "kg_v3/", "graphrag_index/",
           "ca_agraphrag/data/", "extraction_results/", "external/", "example/",
           "datasets/kqa_pro/", "datasets/webqsp/", "datasets/mintaka/", "datasets/lc_quad/",
           "pipeline/v3/work/", "pipeline/v3/cache/", "reports/neo4j_import/")
PATTERNS = [re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
            re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
            re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]


def main():
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    failures = []
    count = 0
    total = 0
    for entry in entries:
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        path = path_bytes.decode()
        count += 1
        if mode == b"160000" or stage != b"0":
            failures.append((path, "gitlink or unresolved merge"))
            continue
        if path.startswith(BLOCKED) or path in {"apikey.txt", ".env"}:
            failures.append((path, "local-only artifact"))
        size = int(subprocess.check_output(["git", "cat-file", "-s", oid.decode()]))
        total += size
        if size > 20 * 1024 * 1024:
            failures.append((path, "file exceeds public snapshot size limit (20 MiB)"))
            continue
        content = subprocess.check_output(["git", "cat-file", "blob", oid.decode()])
        for rx in PATTERNS:
            if rx.search(content):
                failures.append((path, "credential-like content; value suppressed"))
                break
    for path, reason in failures:
        print(f"FAIL {path}: {reason}")
    print(f"Checked {count} staged files, {total:,} bytes; failures={len(failures)}")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
