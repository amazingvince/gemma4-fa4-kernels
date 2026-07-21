#!/usr/bin/env python3
"""Fail-closed preflight for the experimental Axolotl/H100 integration stack."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

AXOLOTL_REVISION = "2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417"
TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
FLASH_ATTENTION_REVISION = "77aacb68d194ba9af1010eda5eac3e7c0df8e6f6"
MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MIN_HBM_BYTES = 70 << 30
ROOT = Path(__file__).resolve().parents[2]


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return (0, 0, 0)
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def validate_snapshot(
    snapshot: dict[str, Any],
    *,
    expected_flash_revision: str = FLASH_ATTENTION_REVISION,
    require_flash_patch: bool = True,
) -> list[str]:
    errors: list[str] = []
    if _version_tuple(str(snapshot.get("python_version", ""))) < (3, 12, 0):
        errors.append("Python 3.12 or newer is required")
    torch_version = _version_tuple(str(snapshot.get("torch_version", "")))
    if not (torch_version >= (2, 11, 0) and torch_version <= (2, 12, 1)):
        errors.append("Axolotl EXP-0036 requires PyTorch >=2.11.0 and <=2.12.1")
    if snapshot.get("cuda_available") is not True:
        errors.append("CUDA is not available")
    if snapshot.get("cuda_device_count") != 1:
        errors.append("exactly one visible H100 is required")
    if snapshot.get("cuda_capability") != [9, 0]:
        errors.append("the harness requires H100 / SM90 capability [9, 0]")
    if int(snapshot.get("total_memory_bytes") or 0) < MIN_HBM_BYTES:
        errors.append("at least 70 GiB of visible HBM is required for the 12B BF16 smoke run")
    if snapshot.get("axolotl_revision") != AXOLOTL_REVISION:
        errors.append(f"Axolotl revision must be {AXOLOTL_REVISION}")
    if snapshot.get("transformers_revision") != TRANSFORMERS_REVISION:
        errors.append(f"Transformers revision must be {TRANSFORMERS_REVISION}")
    if snapshot.get("flash_attention_revision") != expected_flash_revision:
        errors.append(f"FlashAttention revision must be {expected_flash_revision}")
    if snapshot.get("transformers_patch_applied") is not True:
        errors.append("the locked Transformers patch is not applied")
    if require_flash_patch and snapshot.get("flash_attention_patch_applied") is not True:
        errors.append("the locked FlashAttention patch is not applied")
    if not require_flash_patch and snapshot.get("flash_attention_worktree_clean") is not True:
        errors.append("the candidate FlashAttention worktree must be clean")
    if snapshot.get("model_access") is not True:
        errors.append(f"gated model access is unavailable for {MODEL_ID}@{MODEL_REVISION}")
    return errors


def _git_root(module_file: str) -> Path | None:
    for parent in (Path(module_file).resolve(), *Path(module_file).resolve().parents):
        candidate = parent if parent.is_dir() else parent.parent
        if (candidate / ".git").exists():
            return candidate
    return None


def _revision(module_file: str) -> tuple[str | None, str | None]:
    root = _git_root(module_file)
    if root is None:
        return None, None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout.strip() if result.returncode == 0 else None), str(root)


def _patch_applied(root: str | None, patch: Path) -> bool:
    if root is None or not patch.is_file():
        return False
    result = subprocess.run(
        ["git", "-C", root, "apply", "--reverse", "--check", str(patch)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _worktree_clean(root: str | None) -> bool:
    if root is None:
        return False
    result = subprocess.run(
        ["git", "-C", root, "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def collect_snapshot(
    *, check_model_access: bool = True, check_project_fa4: bool = True
) -> dict[str, Any]:
    import torch

    modules = {}
    names = ["axolotl", "transformers"]
    if check_project_fa4:
        names.extend(("flash_attn.cute", "flash_attn_2_cuda"))
    for name in names:
        module = importlib.import_module(name)
        modules[name] = module
    axolotl_rev, axolotl_root = _revision(modules["axolotl"].__file__)
    transformers_rev, transformers_root = _revision(modules["transformers"].__file__)
    flash_rev, flash_root = (
        _revision(modules["flash_attn.cute"].__file__) if check_project_fa4 else (None, None)
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if cuda_available else 0
    properties = torch.cuda.get_device_properties(0) if device_count else None
    model_access = False
    model_error = None
    if check_model_access:
        try:
            from huggingface_hub import HfApi

            HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, token=os.getenv("HF_TOKEN"))
            model_access = True
        except Exception as exc:  # pragma: no cover - requires network/auth
            model_error = f"{type(exc).__name__}: {exc}"
    snapshot = {
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "torch_version": torch.__version__.split("+")[0],
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if device_count else None,
        "device_name": properties.name if properties else None,
        "total_memory_bytes": properties.total_memory if properties else 0,
        "axolotl_revision": axolotl_rev,
        "axolotl_root": axolotl_root,
        "transformers_revision": transformers_rev,
        "transformers_root": transformers_root,
        "flash_attention_revision": flash_rev,
        "flash_attention_root": flash_root,
        "flash_attention_import_file": (
            str(Path(modules["flash_attn.cute"].__file__).resolve()) if check_project_fa4 else None
        ),
        "flash_attention_2_extension_file": (
            str(Path(modules["flash_attn_2_cuda"].__file__).resolve())
            if check_project_fa4
            else None
        ),
        "flash_attention_worktree_clean": _worktree_clean(flash_root),
        "transformers_patch_applied": _patch_applied(
            transformers_root,
            ROOT / "patches/transformers/0001-gemma4-forward-vision-block-ids.patch",
        ),
        "flash_attention_patch_applied": _patch_applied(
            flash_root,
            ROOT / "patches/flash-attention/0002-sm90-gemma4-d512-forward-backward.patch",
        ),
        "model_access": model_access,
        "model_access_error": model_error,
    }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--skip-model-access", action="store_true")
    parser.add_argument("--skip-fa4", action="store_true")
    parser.add_argument("--expected-fa4-revision", default=FLASH_ATTENTION_REVISION)
    parser.add_argument("--expected-fa4-root", type=Path)
    parser.add_argument("--expected-fa2-extension-root", type=Path)
    args = parser.parse_args()
    try:
        snapshot = collect_snapshot(
            check_model_access=not args.skip_model_access,
            check_project_fa4=not args.skip_fa4,
        )
    except Exception as exc:
        print(f"Axolotl preflight import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    candidate_fa4 = args.expected_fa4_revision != FLASH_ATTENTION_REVISION
    errors = validate_snapshot(
        snapshot,
        expected_flash_revision=args.expected_fa4_revision,
        require_flash_patch=not candidate_fa4,
    )
    if args.expected_fa4_root is not None:
        observed_root = snapshot.get("flash_attention_root")
        if (
            observed_root is None
            or Path(observed_root).resolve() != args.expected_fa4_root.resolve()
        ):
            errors.append(
                f"FlashAttention import root must be {args.expected_fa4_root.resolve()}, "
                f"got {observed_root}"
            )
    if args.expected_fa2_extension_root is not None:
        observed_extension = snapshot.get("flash_attention_2_extension_file")
        expected_extension_root = args.expected_fa2_extension_root.resolve()
        if (
            observed_extension is None
            or expected_extension_root not in Path(observed_extension).resolve().parents
        ):
            errors.append(
                f"FlashAttention-2 extension must resolve under {expected_extension_root}, "
                f"got {observed_extension}"
            )
    if args.skip_model_access:
        errors = [error for error in errors if "gated model access" not in error]
        snapshot["model_access_check_skipped"] = True
    if args.skip_fa4:
        errors = [
            error
            for error in errors
            if "FlashAttention revision" not in error and "FlashAttention patch" not in error
        ]
        snapshot["project_fa4_check_skipped"] = True
    snapshot["errors"] = errors
    snapshot["passed"] = not errors
    payload = json.dumps(snapshot, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
