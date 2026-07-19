#!/usr/bin/env python3
"""Capture and strictly validate an H100/B300 CuTe-DSL development environment."""

from __future__ import annotations

import argparse
import hashlib
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


def torch_version_parts(value: str) -> tuple[str, str | None]:
    public, separator, local = value.partition("+")
    return public, local if separator else None


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


def load_policy(profile: str = "b300") -> dict[str, str]:
    policy_path = (
        ROOT / "configs/env/h100-compatible.env"
        if profile == "h100"
        else ROOT / "configs/env/latest-compatible.env"
    )
    policy: dict[str, str] = {}
    for line in policy_path.read_text().splitlines():
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


def git_status(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=path)


def git_dirty(path: Path) -> bool | None:
    status = git_status(path)
    return None if status is None else bool(status)


def git_diff(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return command(["git", "diff", "--binary", "--no-ext-diff", "--"], cwd=path)


def normalized_git_patch(text: str) -> str:
    """Ignore only Git's configurable object-ID abbreviation in index headers."""

    return "\n".join(line for line in text.strip().splitlines() if not line.startswith("index "))


def imported_module_path(module: object) -> str | None:
    value = getattr(module, "__file__", None)
    return str(Path(value).resolve()) if value else None


def path_is_within(path: str | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["h100", "b300"])
    parser.add_argument("--expect-arch", choices=["sm_90", "sm_103"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-profilers", action="store_true")
    parser.add_argument("--require-transformers", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    inferred_profile = {"sm_90": "h100", "sm_103": "b300"}.get(args.expect_arch)
    profile = args.profile or inferred_profile or "b300"
    if args.profile and inferred_profile and args.profile != inferred_profile:
        parser.error(f"--profile {args.profile} conflicts with --expect-arch {args.expect_arch}")
    policy = load_policy(profile)
    upstream = ROOT / ".upstream"
    errors: list[str] = []
    warnings: list[str] = []
    nvcc_text = command(["nvcc", "--version"])
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "profile": profile,
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

    expected_arch_env = {
        "CUTE_DSL_ARCH": policy["CUTE_DSL_ARCH"],
        "FLASH_ATTENTION_ARCH": policy["FLASH_ATTENTION_ARCH"],
    }
    for name, expected in expected_arch_env.items():
        actual = report["architecture_env"][name]
        if args.strict and actual != expected:
            errors.append(f"{name}={actual!r} != policy {expected!r}")

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

        report["torch"] = str(torch.__version__)
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        torch_public, torch_local = torch_version_parts(report["torch"])
        if torch_public != policy["PYTORCH_VERSION"]:
            errors.append(f"torch {report['torch']} != policy {policy['PYTORCH_VERSION']}")
        if torch_local != policy["PYTORCH_CUDA_WHEEL"]:
            errors.append(f"torch build +{torch_local} != policy +{policy['PYTORCH_CUDA_WHEEL']}")
        expected_runtime = policy["PYTORCH_CUDA_RUNTIME"]
        if torch.version.cuda != expected_runtime:
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
        dsl_base_version = importlib.metadata.version("nvidia-cutlass-dsl-libs-base")
    except importlib.metadata.PackageNotFoundError:
        dsl_base_version = None
    report["nvidia_cutlass_dsl_libs_base"] = dsl_base_version
    if dsl_base_version != policy["CUTLASS_DSL_VERSION"]:
        errors.append(
            "nvidia-cutlass-dsl-libs-base "
            f"{dsl_base_version!r} != policy {policy['CUTLASS_DSL_VERSION']}"
        )

    try:
        cu13_dsl_libs = importlib.metadata.version("nvidia-cutlass-dsl-libs-cu13")
    except importlib.metadata.PackageNotFoundError:
        cu13_dsl_libs = None
    report["nvidia_cutlass_dsl_libs_cu13"] = cu13_dsl_libs
    if profile == "h100" and cu13_dsl_libs is not None:
        errors.append("H100 policy forbids nvidia-cutlass-dsl-libs-cu13; use FA4 [dev]")
    if profile == "b300" and cu13_dsl_libs != policy["CUTLASS_DSL_VERSION"]:
        errors.append(
            "B300 policy requires nvidia-cutlass-dsl-libs-cu13 "
            f"{policy['CUTLASS_DSL_VERSION']}; found {cu13_dsl_libs!r}"
        )

    try:
        quack_version = importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        quack_version = None
    report["quack_kernels"] = quack_version
    if quack_version != policy["QUACK_KERNELS_VERSION"]:
        errors.append(
            f"quack-kernels {quack_version!r} != policy {policy['QUACK_KERNELS_VERSION']}"
        )

    try:
        flash_attn_cute = importlib.import_module("flash_attn.cute")
        report["flash_attn_cute_import"] = True
        report["flash_attn_cute_path"] = imported_module_path(flash_attn_cute)
        if not path_is_within(report["flash_attn_cute_path"], upstream / "flash-attention"):
            message = "flash_attn.cute was not imported from the pinned FlashAttention checkout"
            (errors if args.strict else warnings).append(message)
    except Exception as exc:  # pragma: no cover - depends on GPU install
        report["flash_attn_cute_import"] = False
        report["flash_attn_cute_path"] = None
        if args.strict:
            errors.append(f"flash_attn.cute import failed: {exc}")
        else:
            warnings.append(f"flash_attn.cute import failed: {exc}")

    try:
        transformers_gemma4 = importlib.import_module("transformers.models.gemma4.modeling_gemma4")
        report["transformers_oracle_import"] = True
        report["transformers_oracle_path"] = imported_module_path(transformers_gemma4)
        if not path_is_within(report["transformers_oracle_path"], upstream / "transformers"):
            message = "Gemma 4 oracle was not imported from the pinned Transformers checkout"
            (errors if args.require_transformers else warnings).append(message)
    except Exception as exc:
        report["transformers_oracle_import"] = False
        report["transformers_oracle_path"] = None
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
                f"driver {parts[1]} is older than CUDA {policy['CUDA_TOOLKIT_VERSION']} "
                "full-feature policy "
                f"{policy['CUDA_DRIVER_MIN_FULL']}"
            )
    elif args.strict:
        errors.append("nvidia-smi is unavailable")

    correctness_tools = ("ncu", "compute-sanitizer", "nvdisasm", "cuobjdump", "ptxas")
    profiler_tools = ("ncu", "nsys")
    all_tools = tuple(dict.fromkeys((*correctness_tools, *profiler_tools)))
    report["tools"] = {name: shutil.which(name) for name in all_tools}
    if args.strict:
        required_tools = (*correctness_tools, *(profiler_tools if args.require_profilers else ()))
        for name in dict.fromkeys(required_tools):
            location = report["tools"][name]
            if location is None:
                errors.append(f"required CUDA development tool is missing: {name}")
        if not args.require_profilers and report["tools"]["nsys"] is None:
            warnings.append(
                "Nsight Systems (nsys) is unavailable; pass --require-profilers at the benchmark gate"
            )

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
    expected_patch_path = policy.get("FLASH_ATTN_PATCH_PATH")
    patch_report = None
    if expected_patch_path:
        patch_path = ROOT / expected_patch_path
        patch_text = patch_path.read_text() if patch_path.is_file() else None
        patch_sha256 = (
            hashlib.sha256(patch_path.read_bytes()).hexdigest() if patch_path.is_file() else None
        )
        checkout_diff = git_diff(upstream / "flash-attention")
        checkout_status = git_status(upstream / "flash-attention")
        patch_applied = (
            patch_text is not None
            and checkout_diff is not None
            and normalized_git_patch(checkout_diff) == normalized_git_patch(patch_text)
            and checkout_status
            == "M flash_attn/cute/flash_bwd_postprocess.py\n"
            " M flash_attn/cute/flash_bwd_sm90.py\n"
            " M flash_attn/cute/interface.py"
        )
        patch_report = {
            "path": expected_patch_path,
            "sha256": patch_sha256,
            "checkout_status": checkout_status,
            "applied_exactly": patch_applied,
        }
        if patch_sha256 != policy.get("FLASH_ATTN_PATCH_SHA256"):
            errors.append("required FlashAttention patch file hash does not match policy")
        if not patch_applied:
            errors.append("FlashAttention checkout does not match the required patch stack")
    elif upstream_dirty["flash_attention"]:
        errors.append("FlashAttention checkout has uncommitted changes")
    report["flash_attention_patch"] = patch_report
    transformers_patch_report = None
    if args.require_transformers and upstream_heads["transformers"] != policy["TRANSFORMERS_REV"]:
        errors.append(
            "Transformers checkout does not match policy "
            f"{policy['TRANSFORMERS_REV']}: {upstream_heads['transformers']}"
        )
    if args.require_transformers:
        if upstream_dirty["transformers"] is None:
            errors.append("Transformers checkout cleanliness could not be determined")
        elif expected_transformers_patch_path := policy.get("TRANSFORMERS_PATCH_PATH"):
            transformers_patch_path = ROOT / expected_transformers_patch_path
            transformers_patch_text = (
                transformers_patch_path.read_text() if transformers_patch_path.is_file() else None
            )
            transformers_patch_sha256 = (
                hashlib.sha256(transformers_patch_path.read_bytes()).hexdigest()
                if transformers_patch_path.is_file()
                else None
            )
            transformers_checkout_diff = git_diff(upstream / "transformers")
            transformers_checkout_status = git_status(upstream / "transformers")
            transformers_patch_applied = (
                transformers_patch_text is not None
                and transformers_checkout_diff is not None
                and normalized_git_patch(transformers_checkout_diff)
                == normalized_git_patch(transformers_patch_text)
                and transformers_checkout_status
                == "M src/transformers/models/gemma4/modeling_gemma4.py"
            )
            transformers_patch_report = {
                "path": expected_transformers_patch_path,
                "sha256": transformers_patch_sha256,
                "checkout_status": transformers_checkout_status,
                "applied_exactly": transformers_patch_applied,
            }
            if transformers_patch_sha256 != policy.get("TRANSFORMERS_PATCH_SHA256"):
                errors.append("required Transformers patch file hash does not match policy")
            if not transformers_patch_applied:
                errors.append("Transformers checkout does not match the required patch stack")
        elif upstream_dirty["transformers"]:
            errors.append("Transformers checkout has uncommitted changes")
    report["transformers_patch"] = transformers_patch_report

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
