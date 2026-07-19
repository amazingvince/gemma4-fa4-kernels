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


def full_git_sha(value: str) -> str:
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("git SHA must be exactly 40 lowercase hex characters")
    return value


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


def git_dirty(path: Path) -> bool | None:
    status = run("git", "status", "--porcelain", cwd=path)
    return None if status is None else bool(status)


def environment_policy_path(arch: str) -> Path:
    if arch == "sm_90":
        return ROOT / "configs/env/h100-compatible.env"
    if arch == "sm_103":
        return ROOT / "configs/env/latest-compatible.env"
    raise ValueError(f"unsupported architecture: {arch}")


def validate_record(record: object) -> list[str]:
    """Validate against experiments/schema.json.

    Uses jsonschema when installed (dev extra); otherwise falls back to a
    minimal interpreter covering the constraints the schema actually uses.
    Returns a list of problems; empty means valid.
    """
    schema = json.loads((ROOT / "experiments/schema.json").read_text())
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        return [error.message for error in validator.iter_errors(record)]
    except ImportError:
        pass
    import re

    expected_type = schema.get("type")
    expected_python_type = {
        "string": str,
        "object": dict,
        "array": list,
    }.get(expected_type)
    if expected_python_type is not None and not isinstance(record, expected_python_type):
        article = "an" if expected_type in {"array", "object"} else "a"
        return [f"record must be {article} {expected_type}"]

    problems: list[str] = []
    for key in schema.get("required", []):
        if key not in record or record[key] is None:
            problems.append(f"missing required field {key!r}")
    properties = schema.get("properties", {})
    for key, rule in properties.items():
        if key not in record:
            continue
        expected_type = rule.get("type")
        expected_python_type = {
            "string": str,
            "object": dict,
            "array": list,
        }.get(expected_type)
        if expected_python_type is not None and not isinstance(record[key], expected_python_type):
            article = "an" if expected_type in {"array", "object"} else "a"
            problems.append(f"{key} must be {article} {expected_type}")
            continue
        if "enum" in rule and record[key] not in rule["enum"]:
            problems.append(f"{key}={record[key]!r} not in {rule['enum']}")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], str(record[key])):
            problems.append(f"{key}={record[key]!r} does not match {rule['pattern']!r}")
        item_rule = rule.get("items", {})
        if expected_type == "array" and item_rule.get("type") == "object":
            for index, item in enumerate(record[key]):
                if not isinstance(item, dict):
                    problems.append(f"{key}[{index}] must be an object")
    return problems


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
    parser.add_argument(
        "--git-sha",
        type=full_git_sha,
        help="source revision for a synchronized remote checkout without .git metadata",
    )
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
    policy_path = environment_policy_path(args.arch)
    record = {
        "exp_id": args.exp_id,
        "kernel": args.kernel,
        "arch": args.arch,
        "decision": args.decision,
        "hypothesis": args.hypothesis,
        "notes": args.notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": args.git_sha or run("git", "rev-parse", "HEAD", cwd=ROOT) or "unversioned",
        "git_dirty": git_dirty(ROOT),
        "flash_attention_sha": run("git", "rev-parse", "HEAD", cwd=upstream)
        if upstream.exists()
        else None,
        "model_lock_sha256": sha256(ROOT / "configs/model/gemma4-31b.lock.json"),
        "environment_policy_path": policy_path.relative_to(ROOT).as_posix(),
        "environment_policy_sha256": sha256(policy_path),
        "environment": environment,
    }
    if args.bench:
        record["results"] = [
            json.loads(line) for line in args.bench.read_text().splitlines() if line.strip()
        ]
    if args.profile:
        record["profile_artifact"] = str(args.profile)
    problems = validate_record(record)
    if problems:
        for problem in problems:
            print(f"schema violation: {problem}")
        print("record NOT written (fix the record or the schema, do not bypass)")
        return 1
    output = ROOT / "experiments/results.jsonl"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"recorded {args.exp_id} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
