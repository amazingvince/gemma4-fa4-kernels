#!/usr/bin/env python3
"""Measure local BF16 GEMM and HBM-copy denominators at fixed clocks."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch


def _time(fn, warmup: int = 10, reps: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def gemm_peak(n: int) -> float:
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    ms = _time(lambda: a @ b)
    return 2 * n**3 / (ms * 1e-3) / 1e12


def hbm_bandwidth(gib: float) -> float:
    elements = int(gib * (1 << 30) / 2)
    src = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    ms = _time(lambda: dst.copy_(src))
    return 2 * elements * 2 / (ms * 1e-3) / 1e12


def smi() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.sm,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-n", type=int, default=8192)
    parser.add_argument("--copy-gib", type=float, default=4.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    peak = gemm_peak(args.gemm_n)
    bandwidth = hbm_bandwidth(args.copy_gib)
    result = {
        "host": socket.gethostname(),
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "nvidia_smi": smi(),
        "measured_bf16_tflops": peak,
        "measured_hbm_tbs_read_plus_write": bandwidth,
        "ridge_flops_per_byte": peak / bandwidth,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
