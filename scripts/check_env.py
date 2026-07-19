#!/usr/bin/env python3
"""Capture and strictly validate an H100/B300 CuTe-DSL development environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def command(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def load_policy() -> dict[str, str]:
    policy: dict[str, str] = {}
    for line in (ROOT / "configs/env/latest-compatible.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            policy[key] = value
    return policy


def nvcc_release(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"release\s+(\d+\.\d+)", text)
    return match.group(1) if match else None


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return command(["git", "rev-parse", "HEAD"], cwd=path)


def git_dirty(path: Path) -> bool | None:
    if not (path / ".git").exists():
        return None
    status = command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=path)
    return None if status is None else bool(status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-arch", choices=["sm_90", "sm_103"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-transformers", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    policy = load_policy()
    errors: list[str] = []
    warnings: list[str] = []
    nvcc_text = command(["nvcc", "--version"])
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "policy": policy,
        "nvcc": nvcc_text,
        "nvcc_release": nvcc_release(nvcc_text),
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "architecture_env": {
            "CUTE_DSL_ARCH": __import__("os").environ.get("CUTE_DSL_ARCH"),
            "FLASH_ATTENTION_ARCH": __import__("os").environ.get("FLASH_ATTENTION_ARCH"),
        },
    }

    python_mm = ".".join(report["python"].split(".")[:2])
    if python_mm != policy["PYTHON_VERSION"]:
        errors.append(f"python {python_mm} != policy {policy['PYTHON_VERSION']}")

    if report["nvcc_release"] is None:
        if args.strict:
            errors.append("nvcc is unavailable or its release could not be parsed")
    elif report["nvcc_release"] != policy["CUDA_TOOLKIT_VERSION"]:
        errors.append(f"nvcc {report['nvcc_release']} != policy {policy['CUDA_TOOLKIT_VERSION']}")

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.__version__.startswith(policy["PYTORCH_VERSION"]):
            errors.append(f"torch {torch.__version__} != policy {policy['PYTORCH_VERSION']}")
        expected_runtime = policy["PYTORCH_CUDA_RUNTIME"]
        if torch.version.cuda is None or not torch.version.cuda.startswith(expected_runtime):
            errors.append(f"torch CUDA runtime {torch.version.cuda!r} != policy {expected_runtime}")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            report["device"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = f"{major}.{minor}"
            actual_arch = f"sm_{major}{minor}"
            report["detected_arch"] = actual_arch
            if args.expect_arch and actual_arch != args.expect_arch:
                errors.append(f"expected {args.expect_arch}, found {actual_arch}")
        elif args.strict:
            errors.append("torch.cuda.is_available() is false")
    except Exception as exc:  # pragma: no cover - depends on host environment
        report["torch_error"] = repr(exc)
        errors.append(f"torch import failed: {exc}")

    try:
        dsl_version = importlib.metadata.version("nvidia-cutlass-dsl")
        report["nvidia_cutlass_dsl"] = dsl_version
        if dsl_version != policy["CUTLASS_DSL_VERSION"]:
            errors.append(
                f"nvidia-cutlass-dsl {dsl_version} != policy {policy['CUTLASS_DSL_VERSION']}"
            )
    except importlib.metadata.PackageNotFoundError:
        report["nvidia_cutlass_dsl"] = None
        errors.append("nvidia-cutlass-dsl is not installed")

    try:
        importlib.import_module("flash_attn.cute")
        report["flash_attn_cute_import"] = True
    except Exception as exc:  # pragma: no cover - depends on GPU install
        report["flash_attn_cute_import"] = False
        if args.strict:
            errors.append(f"flash_attn.cute import failed: {exc}")
        else:
            warnings.append(f"flash_attn.cute import failed: {exc}")

    try:
        importlib.import_module("transformers.models.gemma4.modeling_gemma4")
        report["transformers_oracle_import"] = True
    except Exception as exc:
        report["transformers_oracle_import"] = False
        if args.require_transformers:
            errors.append(f"pinned Transformers oracle import failed: {exc}")
        else:
            warnings.append(f"Transformers oracle is not importable: {exc}")

    if report["nvidia_smi"]:
        first = str(report["nvidia_smi"]).splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 2 and version_tuple(parts[1]) < version_tuple(
            policy["CUDA_DRIVER_MIN_FULL"]
        ):
            errors.append(
                f"driver {parts[1]} is older than CUDA 13.3 full-feature policy "
                f"{policy['CUDA_DRIVER_MIN_FULL']}"
            )
    elif args.strict:
        errors.append("nvidia-smi is unavailable")

    required_tools = (
        "ncu",
        "nsys",
        "compute-sanitizer",
        "nvdisasm",
        "cuobjdump",
        "ptxas",
    )
    report["tools"] = {name: shutil.which(name) for name in required_tools}
    if args.strict:
        for name, location in report["tools"].items():
            if location is None:
                errors.append(f"required CUDA development tool is missing: {name}")

    upstream = ROOT / ".upstream"
    upstream_heads = {
        "flash_attention": git_head(upstream / "flash-attention"),
        "transformers": git_head(upstream / "transformers"),
    }
    upstream_dirty = {
        "flash_attention": git_dirty(upstream / "flash-attention"),
        "transformers": git_dirty(upstream / "transformers"),
    }
    report["upstream_heads"] = upstream_heads
    report["upstream_dirty"] = upstream_dirty
    if upstream_heads["flash_attention"] != policy["FLASH_ATTN_REV"]:
        errors.append(
            "FlashAttention checkout does not match policy "
            f"{policy['FLASH_ATTN_REV']}: {upstream_heads['flash_attention']}"
        )
    if upstream_dirty["flash_attention"] is None:
        errors.append("FlashAttention checkout cleanliness could not be determined")
    elif upstream_dirty["flash_attention"]:
        errors.append("FlashAttention checkout has uncommitted changes")
    if args.require_transformers and upstream_heads["transformers"] != policy["TRANSFORMERS_REV"]:
        errors.append(
            "Transformers checkout does not match policy "
            f"{policy['TRANSFORMERS_REV']}: {upstream_heads['transformers']}"
        )
    if args.require_transformers:
        if upstream_dirty["transformers"] is None:
            errors.append("Transformers checkout cleanliness could not be determined")
        elif upstream_dirty["transformers"]:
            errors.append("Transformers checkout has uncommitted changes")

    report["warnings"] = warnings
    report["errors"] = errors
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
