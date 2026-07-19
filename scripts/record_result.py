#!/usr/bin/env python3
"""Append a reproducible experiment record to experiments/results.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception:
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--arch", required=True, choices=["sm_90", "sm_103"])
    parser.add_argument(
        "--decision", required=True, choices=["accept", "reject", "refine", "baseline"]
    )
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    environment = {
        "nvidia_smi": run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ),
        "python": run("python", "--version"),
        "torch": run("python", "-c", "import torch; print(torch.__version__, torch.version.cuda)"),
        "nvcc": run("nvcc", "--version"),
    }
    upstream = ROOT / ".upstream" / "flash-attention"
    record = {
        "exp_id": args.exp_id,
        "kernel": args.kernel,
        "arch": args.arch,
        "decision": args.decision,
        "hypothesis": args.hypothesis,
        "notes": args.notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": run("git", "rev-parse", "HEAD", cwd=ROOT) or "unversioned",
        "git_dirty": bool(run("git", "status", "--porcelain", cwd=ROOT)),
        "flash_attention_sha": run("git", "rev-parse", "HEAD", cwd=upstream)
        if upstream.exists()
        else None,
        "model_lock_sha256": sha256(ROOT / "configs/model/gemma4-31b.lock.json"),
        "environment_policy_sha256": sha256(ROOT / "configs/env/latest-compatible.env"),
        "environment": environment,
    }
    if args.bench:
        record["results"] = [
            json.loads(line) for line in args.bench.read_text().splitlines() if line.strip()
        ]
    if args.profile:
        record["profile_artifact"] = str(args.profile)
    output = ROOT / "experiments/results.jsonl"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"recorded {args.exp_id} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
