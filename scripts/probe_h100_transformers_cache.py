#!/usr/bin/env python3
"""Audit H100 FA4 cache reuse through the pinned Transformers adapter.

This is a compile/correctness probe, not a benchmark. It inventories the fixed
BSHD and native packed-THD global backward application keys independently,
crosses their three valid scheduler classes, and proves that runtime totals,
packed batch size, cumulative values, segment order, and legal stride orders do
not create another class. Long-context forward-only calls retain their separate
fixed and native-varlen bounded-cache gates.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import torch

from gemma4_fa4.model_spec import GEMMA4_31B
from gemma4_fa4.transformers_integration import (
    Gemma4DispatchResult,
    gemma4_fa4_mask,
    gemma4_fa4_prepared,
)

_GLOBAL_LAYER_INDEX = 5
_TRUE_VALUES = {"1", "on", "true", "yes"}
_Q_BLOCK = 64
_K_BLOCK = 32
_MAX_BACKWARD_SEQLEN = 2048
_BACKWARD_VARIANTS = ("dkv", "dq_lo", "dq_hi")
_SCHEDULER_CLASSES = ("single_single", "single_multi", "multi_multi")
_FIXED_APPLICATION_KEY_LENGTH = 26
_NATIVE_APPLICATION_KEY_LENGTH = 28


class _PackedCase(NamedTuple):
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    layout: str


# Each replay changes totals, logical packed batch size, cumulative values,
# segment order, and legal input stride order while remaining in the warm class.
# The long replays include unequal K1025/K2048 without adding a length class.
_NATIVE_BACKWARD_CASES = (
    (
        "single_single",
        _PackedCase((31, 1), (32, 1), "bhsd-contiguous"),
        _PackedCase((1, 31, 32), (1, 32, 32), "bshd-backed"),
    ),
    (
        "single_multi",
        _PackedCase((33, 1), (64, 33), "bhsd-contiguous"),
        _PackedCase((1, 33, 63), (33, 64, 2048), "aligned-batch-offset"),
    ),
    (
        "multi_multi",
        _PackedCase((65, 33), (129, 64), "bhsd-contiguous"),
        _PackedCase((33, 129, 65), (64, 2048, 1025), "bshd-backed"),
    ),
)


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


def _scheduler_class(max_q: int, max_k: int) -> str:
    if not (1 <= max_q <= max_k <= _MAX_BACKWARD_SEQLEN):
        raise ValueError("global backward maxima must satisfy 1 <= Sq <= Sk <= 2048")
    single_q = max_q <= _Q_BLOCK
    single_k = max_k <= _K_BLOCK
    return _scheduler_class_from_flags(single_q, single_k)


def _scheduler_class_from_flags(single_q: bool, single_k: bool) -> str:
    classes = {
        (True, True): "single_single",
        (True, False): "single_multi",
        (False, False): "multi_multi",
    }
    try:
        return classes[(single_q, single_k)]
    except KeyError as exc:
        raise ValueError(
            "multi-Q/single-K is impossible when every packed segment satisfies Sq <= Sk"
        ) from exc


def _validate_packed_lengths(
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
) -> str:
    if len(q_lengths) != len(k_lengths) or not q_lengths:
        raise ValueError("packed Q/K lengths must have the same nonzero batch count")
    if any(
        not isinstance(q_length, int)
        or isinstance(q_length, bool)
        or not isinstance(k_length, int)
        or isinstance(k_length, bool)
        or not (1 <= q_length <= k_length <= _MAX_BACKWARD_SEQLEN)
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    ):
        raise ValueError("packed lengths must satisfy 1 <= Sq <= Sk <= 2048")
    return _scheduler_class(max(q_lengths), max(k_lengths))


def _decode_global_backward_application_key(key: object) -> tuple[str, str, str]:
    if not isinstance(key, tuple) or not key or key[-1] not in _BACKWARD_VARIANTS:
        raise AssertionError(f"unexpected global backward application key: {key!r}")
    variant = key[-1]
    if len(key) == _FIXED_APPLICATION_KEY_LENGTH:
        abi = "fixed_bshd"
        single_q, single_k = key[-3:-1]
    elif len(key) == _NATIVE_APPLICATION_KEY_LENGTH:
        abi = "native_thd"
        single_q, single_k = key[-5:-3]
        if key[-3:-1] != (False, False):
            raise AssertionError("native THD application key lost its cumulative-array ABI marker")
    else:
        raise AssertionError(
            "global backward application key length changed: "
            f"expected {_FIXED_APPLICATION_KEY_LENGTH} or {_NATIVE_APPLICATION_KEY_LENGTH}, "
            f"got {len(key)}"
        )
    if not isinstance(single_q, bool) or not isinstance(single_k, bool):
        raise AssertionError("global backward scheduler flags must remain booleans")
    return abi, _scheduler_class_from_flags(single_q, single_k), variant


def _application_snapshot_from_keys(
    keys: Sequence[object],
) -> dict[tuple[str, str, str], str]:
    snapshot: dict[tuple[str, str, str], str] = {}
    for key in keys:
        coordinate = _decode_global_backward_application_key(key)
        if coordinate in snapshot:
            raise AssertionError(f"duplicate global backward application coordinate: {coordinate}")
        snapshot[coordinate] = hashlib.sha256(pickle.dumps(key)).hexdigest()
    return dict(sorted(snapshot.items()))


def _global_backward_application_snapshot() -> dict[tuple[str, str, str], str]:
    from flash_attn.cute.interface import _flash_attn_bwd_gemma4_global_d512

    application_cache = _flash_attn_bwd_gemma4_global_d512.compile_cache
    backing = getattr(application_cache, "cache", None)
    if not isinstance(backing, dict):
        raise AssertionError("the pinned FA4 global backward cache no longer exposes its key map")
    return _application_snapshot_from_keys(tuple(backing))


def _application_coordinates(
    abi: str,
    scheduler_classes: Sequence[str],
) -> set[tuple[str, str, str]]:
    if abi not in {"fixed_bshd", "native_thd"}:
        raise ValueError(f"unknown global backward ABI: {abi}")
    if any(name not in _SCHEDULER_CLASSES for name in scheduler_classes):
        raise ValueError(f"unknown scheduler class in {tuple(scheduler_classes)}")
    return {
        (abi, scheduler_class, variant)
        for scheduler_class in scheduler_classes
        for variant in _BACKWARD_VARIANTS
    }


def _require_application_inventory(
    snapshot: dict[tuple[str, str, str], str],
    expected: set[tuple[str, str, str]],
    *,
    label: str,
) -> None:
    actual = set(snapshot)
    if actual != expected:
        raise AssertionError(
            f"{label} application-key inventory mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    print(f"application_inventory label={label} keys={len(snapshot)}")


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
    expected_application_additions: set[tuple[str, str, str]],
    run: Callable[[], None],
) -> tuple[dict[str, str], dict[tuple[str, str, str], str]]:
    before = _object_hashes(cache_dir)
    before_applications = _global_backward_application_snapshot()
    run()
    after = _object_hashes(cache_dir)
    after_applications = _global_backward_application_snapshot()
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
    _expect_application_additions(
        before_applications,
        after_applications,
        label=label,
        expected=expected_application_additions,
    )
    return after, after_applications


def _expect_application_additions(
    before: dict[tuple[str, str, str], str],
    after: dict[tuple[str, str, str], str],
    *,
    label: str,
    expected: set[tuple[str, str, str]],
) -> None:
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {
        coordinate
        for coordinate in set(before) & set(after)
        if before[coordinate] != after[coordinate]
    }
    if removed or changed or added != expected:
        raise AssertionError(
            f"{label} application-key delta mismatch: added={sorted(added)}, "
            f"expected={sorted(expected)}, removed={sorted(removed)}, changed={sorted(changed)}"
        )
    print(
        f"application_transition label={label} before={len(before)} after={len(after)} "
        f"added={len(added)}"
    )
    for coordinate in sorted(added):
        abi, scheduler_class, variant = coordinate
        print(
            f"application_key abi={abi} scheduler={scheduler_class} variant={variant} "
            f"sha256={after[coordinate]}"
        )


def _expect_reuse(
    cache_dir: Path,
    label: str,
    expected_objects: dict[str, str],
    expected_applications: dict[tuple[str, str, str], str],
    run: Callable[[], None],
) -> None:
    before = _object_hashes(cache_dir)
    before_applications = _global_backward_application_snapshot()
    if before != expected_objects:
        raise AssertionError(f"{label} began from an unexpected cache snapshot")
    if before_applications != expected_applications:
        raise AssertionError(f"{label} began from an unexpected application-key snapshot")
    run()
    after = _object_hashes(cache_dir)
    after_applications = _global_backward_application_snapshot()
    if after != before:
        added, removed, changed = _preserved_objects(before, after)
        raise AssertionError(
            f"{label} changed cache objects: added={added}, removed={removed}, changed={changed}"
        )
    if after_applications != before_applications:
        added = sorted(set(after_applications) - set(before_applications))
        removed = sorted(set(before_applications) - set(after_applications))
        changed = sorted(
            coordinate
            for coordinate in set(before_applications) & set(after_applications)
            if before_applications[coordinate] != after_applications[coordinate]
        )
        raise AssertionError(
            f"{label} changed application keys: added={added}, removed={removed}, changed={changed}"
        )
    print(
        f"cache_reuse label={label} objects={len(after)} application_keys={len(after_applications)}"
    )


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
    q_seqlen: int,
    k_seqlen: int,
    seed: int,
    layout: str,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GEMMA4_31B.spec_for_layer(_GLOBAL_LAYER_INDEX)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = _canonical_tensor(
        (batch, spec.num_q_heads, q_seqlen, spec.head_dim_qk),
        generator=generator,
        standard_deviation=0.03125,
        layout=layout,
        requires_grad=requires_grad,
    )
    k = _canonical_tensor(
        (batch, spec.num_kv_heads, k_seqlen, spec.head_dim_qk),
        generator=generator,
        standard_deviation=0.03125,
        layout=layout,
        requires_grad=requires_grad,
    )
    v = _canonical_tensor(
        (batch, spec.num_kv_heads, k_seqlen, spec.head_dim_v),
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
    q_seqlen: int,
    k_seqlen: int,
    seed: int,
    layout: str,
    expected_path: str,
    q_segment_lengths: Sequence[int] | None = None,
    k_segment_lengths: Sequence[int] | None = None,
    backward: bool,
) -> None:
    q, k, v = _make_inputs(
        batch=batch,
        q_seqlen=q_seqlen,
        k_seqlen=k_seqlen,
        seed=seed,
        layout=layout,
        requires_grad=backward,
    )
    plan = gemma4_fa4_mask(
        batch_size=batch,
        q_length=q_seqlen,
        kv_length=k_seqlen,
        q_offset=k_seqlen - q_seqlen,
    )
    packed_kwargs = {}
    scheduler_class = None
    if (q_segment_lengths is None) != (k_segment_lengths is None):
        raise ValueError("packed Q/K segment lengths must be provided together")
    if q_segment_lengths is not None and k_segment_lengths is not None:
        scheduler_class = _validate_packed_lengths(q_segment_lengths, k_segment_lengths)
        if sum(q_segment_lengths) != batch * q_seqlen:
            raise ValueError("packed Q lengths must sum to the flattened BHSD Q total")
        if sum(k_segment_lengths) != batch * k_seqlen:
            raise ValueError("packed K lengths must sum to the flattened BHSD K/V total")
        packed_kwargs = {
            "cu_seq_lens_q": _cumulative(q_segment_lengths),
            "cu_seq_lens_k": _cumulative(k_segment_lengths),
            "max_length_q": max(q_segment_lengths),
            "max_length_k": max(k_segment_lengths),
        }
    elif backward:
        scheduler_class = _scheduler_class(q_seqlen, k_seqlen)

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
        f"adapter_case path={expected_path} physical_batch={batch} "
        f"q_total={batch * q_seqlen} k_total={batch * k_seqlen} "
        f"packed_batch={len(q_segment_lengths) if q_segment_lengths else batch} "
        f"q_lengths={list(q_segment_lengths) if q_segment_lengths else None} "
        f"k_lengths={list(k_segment_lengths) if k_segment_lengths else None} "
        f"layout={layout} scheduler={scheduler_class} backward={backward}"
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

    print(f"scheduler_contract q_block=64 k_block=32 classes={','.join(_SCHEDULER_CLASSES)}")

    # Preserve the accepted fixed BSHD application family and its exact three
    # M64 x N32 classes before introducing any packed-THD backward keys.
    fixed_cases = (
        ("single_single", 31, 32),
        ("single_multi", 33, 64),
        ("multi_multi", 65, 129),
    )
    snapshot: dict[str, str] = {}
    applications: dict[tuple[str, str, str], str] = {}
    fixed_classes_seen: list[str] = []
    for class_index, (scheduler_class, first_seqlen, replay_seqlen) in enumerate(fixed_cases):
        if _scheduler_class(first_seqlen, first_seqlen) != scheduler_class:
            raise AssertionError(f"fixed warm case mislabeled: {scheduler_class}")
        if _scheduler_class(replay_seqlen, replay_seqlen) != scheduler_class:
            raise AssertionError(f"fixed replay case mislabeled: {scheduler_class}")
        snapshot, applications = _expect_additions(
            cache_dir,
            f"fixed_backward_{scheduler_class}",
            _application_coordinates("fixed_bshd", (scheduler_class,)),
            lambda seqlen=first_seqlen, index=class_index: _run_adapter_case(
                batch=1,
                q_seqlen=seqlen,
                k_seqlen=seqlen,
                seed=11_000 + index,
                layout="bhsd-contiguous",
                expected_path="fa4_global_fixed",
                backward=True,
            ),
        )
        fixed_classes_seen.append(scheduler_class)
        _require_application_inventory(
            applications,
            _application_coordinates("fixed_bshd", fixed_classes_seen),
            label=f"fixed_through_{scheduler_class}",
        )
        _expect_reuse(
            cache_dir,
            f"fixed_backward_{scheduler_class}_runtime_values_and_strides",
            snapshot,
            applications,
            lambda seqlen=replay_seqlen, index=class_index: _run_adapter_case(
                batch=1,
                q_seqlen=seqlen,
                k_seqlen=seqlen,
                seed=12_000 + index,
                layout="bshd-backed",
                expected_path="fa4_global_fixed",
                backward=True,
            ),
        )

    fixed_applications = dict(applications)

    # Long, no-grad fixed/rectangular calls use the same fixed forward family.
    # They must not compile backward and must not add an S-dependent object.
    _expect_reuse(
        cache_dir,
        "long_fixed_forward_only",
        snapshot,
        applications,
        lambda: _run_adapter_case(
            batch=1,
            q_seqlen=1025,
            k_seqlen=1025,
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
        applications,
        lambda: _run_adapter_case(
            batch=2,
            q_seqlen=1033,
            k_seqlen=1033,
            seed=14_002,
            layout="aligned-batch-offset",
            expected_path="fa4_global_forward_only",
            backward=False,
        ),
    )

    # Native packed forward has cu_seqlens in its compile key and therefore
    # owns one additional bounded forward family. Runtime maxima and cumulative
    # values remain dynamic once that family is warm.
    snapshot, applications = _expect_additions(
        cache_dir,
        "long_native_varlen_forward_family",
        set(),
        lambda: _run_adapter_case(
            batch=1,
            q_seqlen=2058,
            k_seqlen=2058,
            seed=15_001,
            layout="bhsd-contiguous",
            expected_path="fa4_global_varlen_forward_only",
            q_segment_lengths=(1025, 1033),
            k_segment_lengths=(1025, 1033),
            backward=False,
        ),
    )
    _expect_reuse(
        cache_dir,
        "long_native_varlen_segment_order_values_and_strides",
        snapshot,
        applications,
        lambda: _run_adapter_case(
            batch=1,
            q_seqlen=2058,
            k_seqlen=2058,
            seed=15_002,
            layout="bshd-backed",
            expected_path="fa4_global_varlen_forward_only",
            q_segment_lengths=(1033, 1025),
            k_segment_lengths=(1033, 1025),
            backward=False,
        ),
    )
    _expect_reuse(
        cache_dir,
        "long_native_varlen_runtime_batch",
        snapshot,
        applications,
        lambda: _run_adapter_case(
            batch=2,
            q_seqlen=1033,
            k_seqlen=1033,
            seed=15_003,
            layout="aligned-batch-offset",
            expected_path="fa4_global_varlen_forward_only",
            q_segment_lengths=(1033, 1033),
            k_segment_lengths=(1033, 1033),
            backward=False,
        ),
    )

    # Native packed THD backward owns exactly the same three scheduler classes
    # under a separate ABI key. Each replay changes runtime totals, logical B,
    # cumulative values, segment order, and strides. The long replays prove
    # unequal K1025/K2048 do not add an exact-length specialization.
    native_classes_seen: list[str] = []
    for class_index, (scheduler_class, warm, replay) in enumerate(_NATIVE_BACKWARD_CASES):
        if _validate_packed_lengths(warm.q_lengths, warm.k_lengths) != scheduler_class:
            raise AssertionError(f"native warm case mislabeled: {scheduler_class}")
        if _validate_packed_lengths(replay.q_lengths, replay.k_lengths) != scheduler_class:
            raise AssertionError(f"native replay case mislabeled: {scheduler_class}")
        snapshot, applications = _expect_additions(
            cache_dir,
            f"native_thd_backward_{scheduler_class}",
            _application_coordinates("native_thd", (scheduler_class,)),
            lambda case=warm, index=class_index: _run_adapter_case(
                batch=1,
                q_seqlen=sum(case.q_lengths),
                k_seqlen=sum(case.k_lengths),
                seed=16_000 + index,
                layout=case.layout,
                expected_path="fa4_global_varlen_native",
                q_segment_lengths=case.q_lengths,
                k_segment_lengths=case.k_lengths,
                backward=True,
            ),
        )
        native_classes_seen.append(scheduler_class)
        expected_applications = _application_coordinates(
            "fixed_bshd", _SCHEDULER_CLASSES
        ) | _application_coordinates("native_thd", native_classes_seen)
        _require_application_inventory(
            applications,
            expected_applications,
            label=f"native_thd_through_{scheduler_class}",
        )
        retained_fixed = {
            coordinate: digest
            for coordinate, digest in applications.items()
            if coordinate[0] == "fixed_bshd"
        }
        if retained_fixed != fixed_applications:
            raise AssertionError("native THD compilation changed the fixed BSHD application keys")
        _expect_reuse(
            cache_dir,
            f"native_thd_backward_{scheduler_class}_runtime_reuse",
            snapshot,
            applications,
            lambda case=replay, index=class_index: _run_adapter_case(
                batch=1,
                q_seqlen=sum(case.q_lengths),
                k_seqlen=sum(case.k_lengths),
                seed=17_000 + index,
                layout=case.layout,
                expected_path="fa4_global_varlen_native",
                q_segment_lengths=case.q_lengths,
                k_segment_lengths=case.k_lengths,
                backward=True,
            ),
        )

    expected_final_applications = _application_coordinates(
        "fixed_bshd", _SCHEDULER_CLASSES
    ) | _application_coordinates("native_thd", _SCHEDULER_CLASSES)
    _require_application_inventory(
        applications,
        expected_final_applications,
        label="fixed_and_native_complete",
    )
    native_applications = {
        coordinate: digest
        for coordinate, digest in applications.items()
        if coordinate[0] == "native_thd"
    }
    if set(fixed_applications.values()) & set(native_applications.values()):
        raise AssertionError("fixed BSHD and native THD application keys are not distinct")
    print(
        f"application_separation fixed={len(fixed_applications)} "
        f"native={len(native_applications)} preserved=True"
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
    for coordinate, digest in applications.items():
        abi, scheduler_class, variant = coordinate
        print(
            f"application_final abi={abi} scheduler={scheduler_class} variant={variant} "
            f"sha256={digest}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
