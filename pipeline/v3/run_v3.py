"""
v3 流水线编排：S1 → S2 → S3 → S4 → S5。

试点: python pipeline/v3/run_v3.py --pilot        (10 wiki + 15 rt + 5 gs)
全量: python pipeline/v3/run_v3.py --full
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V3   = ROOT / "pipeline" / "v3"


def pilot_docs() -> list[str]:
    docs = json.loads((ROOT / "radar_corpus" / "corpus.json").read_text(encoding="utf-8"))
    wiki = [d["id"] for d in docs if not d["id"].endswith(("__rt", "__gs"))]
    rt   = [d["id"] for d in docs if d["id"].endswith("__rt")]
    gs   = [d["id"] for d in docs if d["id"].endswith("__gs")]
    rng  = random.Random(42)
    return rng.sample(wiki, 10) + rng.sample(rt, 15) + rng.sample(gs, 5)


def run(script: str, *args):
    cmd = [sys.executable, "-X", "utf8", str(V3 / script), *args]
    print(f"\n===== {script} {' '.join(args[:1])}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{script} failed rc={r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--skip-s1", action="store_true", help="复用已有 chunks.jsonl")
    args = ap.parse_args()

    if args.pilot:
        ids = pilot_docs()
        print(f"[pilot] {len(ids)} docs")
        if not args.skip_s1:
            run("s1_ingest.py", "--docs", ",".join(ids))
    elif not args.skip_s1:
        run("s1_ingest.py")

    run("s2_deterministic.py")
    run("s3_extract.py")
    run("s4_critic.py")
    run("s6_canonicalize.py")     # 类别 tail 归一（功能/体制/模式）+ 降级
    run("s5_fuse.py")             # 融合（含全局消歧：雷达/厂商/平台 + headquarteredIn 派生）
    run("s8_quality.py")          # 全局质量校验（垃圾/碎片/悬空/类型违规自动检测）


if __name__ == "__main__":
    main()
