#!/usr/bin/env python3
"""EXP-0044 fresh-process H100 production long-context soak harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENT = "EXP-0044"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_SEEDS = (44_001, 44_002, 44_003)


def _commands(
    output_dir: Path,
    *,
    global_repeats: int,
    local_seeds: tuple[int, ...],
    benchmark_warmup: int,
    benchmark_reps: int,
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    commands: list[tuple[str, list[str]]] = [
        (
            "global-s64k-fwd-bwd",
            [
                python,
                "benchmarks/bench_attention.py",
                "--ladder",
                "full",
                "--only",
                "global_s64k",
                "--impl",
                "fa4",
                "--mode",
                "fwd_bwd",
                "--warmup",
                str(benchmark_warmup),
                "--reps",
                str(benchmark_reps),
                "--l2",
                "hot",
                "--json",
                str(output_dir / "global-s64k-fwd-bwd.jsonl"),
            ],
        ),
        (
            "global-q1-k262144",
            [
                python,
                "scripts/probe_h100_global_varlen_backward.py",
                "--q-lengths",
                "1",
                "--k-lengths",
                "262144",
                "--gradient-source",
                "out_lse",
                "--repeats",
                str(global_repeats),
                "--nondefault-stream",
                "--analytic-zero",
                "--analytic-pattern",
                "final",
                "--analytic-score",
                "finite-large",
            ],
        ),
    ]
    commands.extend(
        (
            f"local-s262144-seed-{seed}",
            [
                python,
                "scripts/probe_h100_local_varlen_backward.py",
                "--q-lengths",
                "262144",
                "--gradient-source",
                "out",
                "--repeats",
                "1",
                "--nondefault-stream",
                "--seed",
                str(seed),
            ],
        )
        for seed in local_seeds
    )
    return commands


def _compute_processes() -> list[str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute query failed: {query.stderr.strip()}")
    return [line.strip() for line in query.stdout.splitlines() if line.strip()]


def _wait_for_empty_compute_processes() -> list[str]:
    processes = _compute_processes()
    for _ in range(10):
        if not processes:
            return []
        time.sleep(0.2)
        processes = _compute_processes()
    return processes


def _cache_inventory(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def _run_case(label: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    stdout_path = output_dir / f"{label}.stdout.log"
    stderr_path = output_dir / f"{label}.stderr.log"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    remaining = _wait_for_empty_compute_processes()
    cache_value = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    cache_path = Path(cache_value).resolve() if cache_value else None
    record = {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "post_child_compute_processes": remaining,
        "fa4_cache_after": _cache_inventory(cache_path),
    }
    return record


def _validate_benchmark(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get("status") != "ok":
        raise RuntimeError(f"global S64K benchmark did not complete normally: {rows}")
    row = rows[0]
    if (
        row.get("name") != "global_s64k"
        or row.get("mode") != "fwd_bwd"
        or row.get("softmax_scale") != 1.0
        or row.get("head_dim") != 512
        or row.get("q_heads") != 32
        or row.get("kv_heads") != 4
        or row.get("forward_single_launch") is not True
        or row.get("owner_computes_dkv") is not True
    ):
        raise RuntimeError(f"global S64K benchmark contract drifted: {row}")
    return row


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative integers")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-repeats", type=int, default=3)
    parser.add_argument("--local-seeds", type=_parse_seeds, default=DEFAULT_LOCAL_SEEDS)
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-reps", type=int, default=5)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.global_repeats < 1 or args.benchmark_warmup < 0 or args.benchmark_reps < 1:
        raise ValueError("repeat counts must be positive and warmup must be nonnegative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_processes = _compute_processes()
    if initial_processes:
        raise RuntimeError(f"soak requires an exclusive idle GPU: {initial_processes}")
    commands = _commands(
        output_dir,
        global_repeats=args.global_repeats,
        local_seeds=args.local_seeds,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_reps=args.benchmark_reps,
    )
    records = []
    status = "passed"
    error = None
    try:
        for label, command in commands:
            record = _run_case(label, command, output_dir)
            records.append(record)
            if record["returncode"] != 0:
                raise RuntimeError(f"{label} failed with exit {record['returncode']}")
            if record["post_child_compute_processes"]:
                raise RuntimeError(
                    f"{label} left GPU compute processes: {record['post_child_compute_processes']}"
                )
        benchmark = _validate_benchmark(output_dir / "global-s64k-fwd-bwd.jsonl")
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        benchmark = None
    cache_value = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    cache_path = Path(cache_value).resolve() if cache_value else None
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": status,
        "boundary": "h100_production_long_context_soak",
        "request": {
            "global_repeats": args.global_repeats,
            "local_seeds": list(args.local_seeds),
            "benchmark_warmup": args.benchmark_warmup,
            "benchmark_reps": args.benchmark_reps,
        },
        "commands": records,
        "global_s64k": benchmark,
        "fa4_cache": _cache_inventory(cache_path),
        "final_compute_processes": _wait_for_empty_compute_processes(),
        "error": error,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(summary_path), "status": status}, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
