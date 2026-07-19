#!/usr/bin/env python3
"""Audit H100 FA4 cache reuse through the pinned Transformers adapter.

This is a compile/correctness probe, not a benchmark.  It deliberately crosses
the three compile-time block classes in the project global backward, then proves
that runtime values, canonical legal stride orders, batch size, and packed
segment order do not create additional retained objects inside a warm class.
Long-context forward-only calls additionally prove that fixed and native-varlen
forward families are distinct but bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch

from gemma4_fa4.model_spec import GEMMA4_31B
from gemma4_fa4.transformers_integration import (
    Gemma4DispatchResult,
    gemma4_fa4_mask,
    gemma4_fa4_prepared,
)

_GLOBAL_LAYER_INDEX = 5
_TRUE_VALUES = {"1", "on", "true", "yes"}


def _require_h100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Transformers cache probe requires CUDA")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability != (9, 0) or "H100" not in name.upper():
        raise RuntimeError(
            "the Transformers cache probe requires an H100 (SM90); "
            f"found {name!r} with compute capability {capability}"
        )


def _isolated_cache_dir(parser: argparse.ArgumentParser) -> Path:
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1":
        parser.error("the cache probe requires real H100 execution, not fake-tensor mode")
    enabled = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "").lower()
    if enabled not in _TRUE_VALUES:
        parser.error("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 is required")
    raw = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    if not raw:
        parser.error("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must name an isolated empty path")
    cache_dir = Path(raw).expanduser().resolve()
    if cache_dir.exists():
        existing = sorted(path.name for path in cache_dir.iterdir())
        if existing:
            parser.error(
                f"FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must be empty; found entries={existing[:8]}"
            )
    else:
        cache_dir.mkdir(parents=True)
    return cache_dir


def _object_hashes(cache_dir: Path) -> dict[str, str]:
    objects: dict[str, str] = {}
    for path in sorted(cache_dir.rglob("*.o")):
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"retained cache object is missing or empty: {path}")
        relative = path.relative_to(cache_dir).as_posix()
        objects[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return objects


def _preserved_objects(
    before: dict[str, str],
    after: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    return added, removed, changed


def _expect_additions(
    cache_dir: Path,
    label: str,
    run: Callable[[], None],
) -> dict[str, str]:
    before = _object_hashes(cache_dir)
    run()
    after = _object_hashes(cache_dir)
    added, removed, changed = _preserved_objects(before, after)
    if removed or changed:
        raise AssertionError(
            f"{label} mutated retained objects: removed={removed}, changed={changed}"
        )
    if not added:
        raise AssertionError(f"{label} did not retain the expected new compile family/class")
    print(
        f"cache_transition label={label} before={len(before)} after={len(after)} added={len(added)}"
    )
    for relative in added:
        print(f"added_object={relative} sha256={after[relative]}")
    return after


def _expect_reuse(
    cache_dir: Path,
    label: str,
    expected: dict[str, str],
    run: Callable[[], None],
) -> None:
    before = _object_hashes(cache_dir)
    if before != expected:
        raise AssertionError(f"{label} began from an unexpected cache snapshot")
    run()
    after = _object_hashes(cache_dir)
    if after != before:
        added, removed, changed = _preserved_objects(before, after)
        raise AssertionError(
            f"{label} changed cache objects: added={added}, removed={removed}, changed={changed}"
        )
    print(f"cache_reuse label={label} objects={len(after)}")


def _global_module() -> SimpleNamespace:
    spec = GEMMA4_31B.spec_for_layer(_GLOBAL_LAYER_INDEX)
    return SimpleNamespace(
        layer_idx=_GLOBAL_LAYER_INDEX,
        layer_type=spec.kind,
        is_sliding=False,
        head_dim=spec.head_dim_qk,
        num_key_value_groups=spec.qhead_per_kvhead,
        scaling=spec.softmax_scale,
        sliding_window=spec.sliding_window,
    )


def _canonical_tensor(
    shape: tuple[int, int, int, int],
    *,
    generator: torch.Generator,
    standard_deviation: float,
    layout: str,
    requires_grad: bool,
) -> torch.Tensor:
    batch, heads, seqlen, head_dim = shape
    if layout == "bhsd-contiguous":
        tensor = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    elif layout == "bshd-backed":
        backing = torch.empty(
            batch,
            seqlen,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        tensor = backing.transpose(1, 2)
    elif layout == "aligned-batch-offset":
        backing = torch.empty(
            batch + 1,
            heads,
            seqlen,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        tensor = backing[1:]
    else:  # pragma: no cover - all callers use one of the named layouts
        raise ValueError(f"unknown canonical layout: {layout}")
    tensor.normal_(mean=0.0, std=standard_deviation, generator=generator)
    tensor.requires_grad_(requires_grad)
    if not tensor.is_leaf:
        raise AssertionError("cache-probe inputs must remain leaf tensors")
    return tensor


def _make_inputs(
    *,
    batch: int,
    seqlen: int,
    seed: int,
    layout: str,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GEMMA4_31B.spec_for_layer(_GLOBAL_LAYER_INDEX)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = _canonical_tensor(
        (batch, spec.num_q_heads, seqlen, spec.head_dim_qk),
        generator=generator,
        standard_deviation=0.03125,
        layout=layout,
        requires_grad=requires_grad,
    )
    k = _canonical_tensor(
        (batch, spec.num_kv_heads, seqlen, spec.head_dim_qk),
        generator=generator,
        standard_deviation=0.03125,
        layout=layout,
        requires_grad=requires_grad,
    )
    v = _canonical_tensor(
        (batch, spec.num_kv_heads, seqlen, spec.head_dim_v),
        generator=generator,
        standard_deviation=0.25,
        layout=layout,
        requires_grad=requires_grad,
    )
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("the cache probe requires distinct prepared K and V")
    for name, tensor in zip(("Q", "K", "V"), (q, k, v), strict=True):
        bshd = tensor.transpose(1, 2)
        if bshd.stride(-1) != 1 or any(stride % 8 for stride in bshd.stride()[:-1]):
            raise AssertionError(f"{name} does not satisfy the 16-byte BF16 stride contract")
        if torch._debug_has_internal_overlap(bshd) != 0:
            raise AssertionError(f"{name} canonical layout is overlapping or indeterminate")
        if bshd.data_ptr() % 16:
            raise AssertionError(f"{name} base pointer is not 16-byte aligned")
    return q, k, v


def _cumulative(lengths: Sequence[int]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        if length <= 0:
            raise ValueError("cache-probe packed lengths must be positive")
        values.append(values[-1] + length)
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _check_result(
    result: Gemma4DispatchResult,
    q: torch.Tensor,
    *,
    expected_path: str,
) -> None:
    expected_output = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    expected_lse = (q.shape[0], q.shape[1], q.shape[2])
    if result.path != expected_path:
        raise AssertionError(f"expected adapter path {expected_path}, got {result.path}")
    if result.output.shape != expected_output or result.output.dtype != torch.bfloat16:
        raise AssertionError(
            f"{expected_path} returned an invalid output contract: "
            f"{tuple(result.output.shape)}, {result.output.dtype}"
        )
    if result.lse is None or result.lse.shape != expected_lse or result.lse.dtype != torch.float32:
        raise AssertionError(f"{expected_path} returned an invalid FP32 LSE contract")
    if not torch.isfinite(result.output).all() or not torch.isfinite(result.lse).all():
        raise AssertionError(f"{expected_path} returned non-finite output or LSE")


def _run_adapter_case(
    *,
    batch: int,
    seqlen: int,
    seed: int,
    layout: str,
    expected_path: str,
    segment_lengths: Sequence[int] | None = None,
    backward: bool,
) -> None:
    q, k, v = _make_inputs(
        batch=batch,
        seqlen=seqlen,
        seed=seed,
        layout=layout,
        requires_grad=backward,
    )
    plan = gemma4_fa4_mask(batch_size=batch, q_length=seqlen, kv_length=seqlen)
    packed_kwargs = {}
    if segment_lengths is not None:
        if sum(segment_lengths) != batch * seqlen:
            raise ValueError("packed segment lengths must sum to the flattened BHSD total")
        cu = _cumulative(segment_lengths)
        packed_kwargs = {
            "cu_seq_lens_q": cu,
            "cu_seq_lens_k": cu.clone(),
            "max_length_q": max(segment_lengths),
            "max_length_k": max(segment_lengths),
        }

    grad_context = torch.enable_grad() if backward else torch.no_grad()
    with grad_context:
        result = gemma4_fa4_prepared(
            _global_module(),
            q,
            k,
            v,
            plan,
            dropout=0.0,
            scaling=1.0,
            sliding_window=None,
            allow_flex_fallback=False,
            **packed_kwargs,
        )
        _check_result(result, q, expected_path=expected_path)
        if backward:
            generator = torch.Generator(device="cuda").manual_seed(seed + 100_000)
            dout = torch.empty_like(result.output).normal_(generator=generator)
            grads = torch.autograd.grad(result.output, (q, k, v), dout)
            for name, grad, source in zip(("dQ", "dK", "dV"), grads, (q, k, v), strict=True):
                if grad.shape != source.shape or grad.dtype != torch.bfloat16:
                    raise AssertionError(f"{expected_path} returned an invalid {name} contract")
                if not torch.isfinite(grad).all():
                    raise AssertionError(f"{expected_path} returned non-finite {name}")
    torch.cuda.synchronize()
    print(
        f"adapter_case path={expected_path} batch={batch} seqlen={seqlen} "
        f"layout={layout} segments={list(segment_lengths) if segment_lengths else None} "
        f"backward={backward}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove bounded global FA4 cache variants through the Transformers adapter"
    )
    _require_h100()
    cache_dir = _isolated_cache_dir(parser)
    print(
        f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__} cache_dir={cache_dir}"
    )

    # The patched global backward key has exactly these three runtime-derived
    # boolean classes for M64 x N32: (single-Q, single-K) = (T,T), (T,F), (F,F).
    backward_classes = (
        ("s_le_32", 31, 29),
        ("s_33_64", 33, 47),
        ("s_ge_65", 65, 97),
    )
    snapshot: dict[str, str] = {}
    for class_index, (label, first_seqlen, replay_seqlen) in enumerate(backward_classes):
        snapshot = _expect_additions(
            cache_dir,
            f"fixed_backward_{label}",
            lambda label=label, seqlen=first_seqlen, index=class_index: _run_adapter_case(
                batch=1,
                seqlen=seqlen,
                seed=11_000 + index,
                layout="bhsd-contiguous",
                expected_path="fa4_global_fixed",
                backward=True,
            ),
        )
        _expect_reuse(
            cache_dir,
            f"fixed_backward_{label}_runtime_values_and_strides",
            snapshot,
            lambda seqlen=replay_seqlen, index=class_index: _run_adapter_case(
                batch=1,
                seqlen=seqlen,
                seed=12_000 + index,
                layout="bshd-backed",
                expected_path="fa4_global_fixed",
                backward=True,
            ),
        )

    # Host composition over three already-warm block classes must not create a
    # fourth kernel family. Reordering the segments and changing the input
    # stride order are runtime changes only.
    _expect_reuse(
        cache_dir,
        "composed_segments_all_backward_classes",
        snapshot,
        lambda: _run_adapter_case(
            batch=1,
            seqlen=129,
            seed=13_001,
            layout="bhsd-contiguous",
            expected_path="fa4_global_varlen",
            segment_lengths=(31, 33, 65),
            backward=True,
        ),
    )
    _expect_reuse(
        cache_dir,
        "composed_segment_order_values_and_strides",
        snapshot,
        lambda: _run_adapter_case(
            batch=1,
            seqlen=129,
            seed=13_002,
            layout="aligned-batch-offset",
            expected_path="fa4_global_varlen",
            segment_lengths=(65, 31, 33),
            backward=True,
        ),
    )
    _expect_reuse(
        cache_dir,
        "composed_runtime_batch",
        snapshot,
        lambda: _run_adapter_case(
            batch=3,
            seqlen=33,
            seed=13_003,
            layout="bshd-backed",
            expected_path="fa4_global_varlen",
            backward=True,
        ),
    )

    # Long, no-grad fixed/rectangular calls use the same fixed forward family.
    # They must not compile backward and must not add an S-dependent object.
    _expect_reuse(
        cache_dir,
        "long_fixed_forward_only",
        snapshot,
        lambda: _run_adapter_case(
            batch=1,
            seqlen=1025,
            seed=14_001,
            layout="bshd-backed",
            expected_path="fa4_global_forward_only",
            backward=False,
        ),
    )
    _expect_reuse(
        cache_dir,
        "long_fixed_forward_only_runtime_batch_and_strides",
        snapshot,
        lambda: _run_adapter_case(
            batch=2,
            seqlen=1033,
            seed=14_002,
            layout="aligned-batch-offset",
            expected_path="fa4_global_forward_only",
            backward=False,
        ),
    )

    # Native packed forward has cu_seqlens in its compile key and therefore
    # owns one additional bounded forward family. Runtime maxima and cumulative
    # values remain dynamic once that family is warm.
    snapshot = _expect_additions(
        cache_dir,
        "long_native_varlen_forward_family",
        lambda: _run_adapter_case(
            batch=1,
            seqlen=2058,
            seed=15_001,
            layout="bhsd-contiguous",
            expected_path="fa4_global_varlen_forward_only",
            segment_lengths=(1025, 1033),
            backward=False,
        ),
    )
    _expect_reuse(
        cache_dir,
        "long_native_varlen_segment_order_values_and_strides",
        snapshot,
        lambda: _run_adapter_case(
            batch=1,
            seqlen=2058,
            seed=15_002,
            layout="bshd-backed",
            expected_path="fa4_global_varlen_forward_only",
            segment_lengths=(1033, 1025),
            backward=False,
        ),
    )
    _expect_reuse(
        cache_dir,
        "long_native_varlen_runtime_batch",
        snapshot,
        lambda: _run_adapter_case(
            batch=2,
            seqlen=1033,
            seed=15_003,
            layout="aligned-batch-offset",
            expected_path="fa4_global_varlen_forward_only",
            segment_lengths=(1033, 1033),
            backward=False,
        ),
    )

    final = _object_hashes(cache_dir)
    if not final:
        raise AssertionError("the cache probe did not retain any compiled object files")
    retained_bytes = sum(path.stat().st_size for path in cache_dir.rglob("*.o"))
    print(
        f"cache_final objects={len(final)} unique_sha256={len(set(final.values()))} "
        f"bytes={retained_bytes}"
    )
    for relative, digest in final.items():
        print(f"object={relative} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
