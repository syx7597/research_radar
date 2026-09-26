"""Remove only reviewed PDF render outputs. Dry run by default; --apply deletes.

Original PDFs, transcripts, model outputs and indexes are never deleted here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
AIR = "manuals/机载雷达手册  第4版=AIRBORNE RADAR HANDBOOK_13872580.pdf"
NAVAL = "manuals/世界海用雷达手册.pdf"
SOURCES = {AIR: "3e6d61cb9a281abefb72a9eb8c98a25eec35f173224e232d332401cb8374a3f6",
           NAVAL: "f5d34b1b09a08231886cf268c937a895e14dec0e9ab7c360ec71348a74ddf266"}
TARGETS = {
    "pipeline/v3/manual/manual_split": (AIR, "python pipeline/v3/manual/split_by_radar.py"),
    "pipeline/v3/manual/manual_body_pages": (AIR, "python pipeline/v3/manual/split_by_radar.py --flat"),
    "pipeline/v3/manual/manual_appendix_ab_pages": (AIR, "python pipeline/v3/manual/split_by_radar.py --appendix-ab"),
    "pipeline/v3/manual/pages": (AIR, "python pipeline/v3/manual/render_manual.py 494 516"),
    "data/v3/naval_appendix_pages": (NAVAL, "python pipeline/v3/manual/naval_transcribe.py --render-only 643 727"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for rel, expected in SOURCES.items():
        p = ROOT / rel
        if not p.is_file() or digest(p) != expected:
            raise SystemExit(f"Original PDF missing or differs from reviewed source: {rel}")
    for rel in ["pipeline/v3/manual/toc_radars.json", "pipeline/v3/manual/toc_radars_p2.json",
                "pipeline/v3/manual/split_by_radar.py", "pipeline/v3/manual/render_manual.py",
                "pipeline/v3/manual/naval_transcribe.py"]:
        if not (ROOT / rel).is_file():
            raise SystemExit(f"Missing reconstruction input: {rel}")
    rows = []
    for rel, (source, command) in TARGETS.items():
        p = ROOT / rel
        if not p.exists():
            continue
        if p.is_symlink() or not p.is_dir():
            raise SystemExit(f"Refusing non-directory or symlink: {rel}")
        contents = list(p.rglob("*"))
        if any(f.is_symlink() for f in contents):
            raise SystemExit(f"Refusing directory containing symlinks: {rel}")
        files = [f for f in contents if f.is_file()]
        unexpected = [f for f in files if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}
                      and not (rel.endswith("manual_split") and f.name == "meta.json")]
        if unexpected:
            raise SystemExit(f"Unexpected output files in {rel}; review before deleting")
        rows.append({"path": rel, "files": len(files), "bytes": sum(f.stat().st_size for f in files),
                     "source": source, "rebuild": command})
    manifest = {"date": "2026-09-26", "applied": args.apply, "sources": SOURCES,
                "targets": rows, "total_bytes": sum(r["bytes"] for r in rows)}
    if args.apply and rows:
        out = ROOT / "artifacts/cleanup_manifest.json"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            raise SystemExit("Existing cleanup manifest preserved; review before another deletion")
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for row in rows:
            shutil.rmtree(ROOT / row["path"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
