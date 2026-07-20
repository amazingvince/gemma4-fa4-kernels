#!/usr/bin/env python3
"""Prove local FA4 compilation is independent of runtime payload values."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import torch

from gemma4_fa4.h100 import fa4_local_varlen_forward
from gemma4_fa4.model_spec import SLIDING_ATTENTION
from gemma4_fa4.transformers_integration import (
    Gemma4DispatchResult,
    gemma4_fa4_mask,
    gemma4_fa4_prepared,
)


class _StaticPrefixCase(NamedTuple):
    physical_capacity: int
    layout: str


_STATIC_PREFIX_Q_LENGTH = 1
_STATIC_PREFIX_K_LENGTH = 33
_STATIC_PREFIX_CASES = (
    _StaticPrefixCase(65, "bhsd-contiguous"),
    _StaticPrefixCase(129, "bshd-backed"),
)
_TRUE_VALUES = {"1", "on", "true", "yes"}


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the packed-varlen cache probe requires an SM90 CUDA device")


def _require_static_prefix_h100() -> None:
    _require_h100()
    if "H100" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("the StaticCache prefix probe requires an H100")


def _make_inputs(
    q_lengths: list[int],
    k_lengths: list[int],
    seed: int,
    *,
    requires_grad: bool,
):
    if len(q_lengths) != len(k_lengths) or not q_lengths:
        raise ValueError("packed Q/K lengths must have the same nonzero batch count")
    if any(
        q_length < 0 or q_length > k_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    ):
        raise ValueError("packed cache-probe lengths must satisfy 0 <= Sq <= Sk")
    if sum(q_lengths) <= 0 or sum(k_lengths) <= 0:
        raise ValueError("mixed empty cache probes require positive packed totals")
    spec = SLIDING_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(q_lengths),
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    q_cumulative = [0]
    k_cumulative = [0]
    for q_length, k_length in zip(q_lengths, k_lengths, strict=True):
        q_cumulative.append(q_cumulative[-1] + q_length)
        k_cumulative.append(k_cumulative[-1] + k_length)
    cu_q = torch.tensor(q_cumulative, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_cumulative, device="cuda", dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _metadata(k_lengths: list[int], *, reverse: bool):
    vision_parts = []
    document_parts = []
    for seqlen in k_lengths:
        positions = torch.arange(seqlen, device="cuda", dtype=torch.int32)
        vision = torch.where(positions % 5 < 3, positions // 5, -1)
        documents = (positions >= seqlen // 2).to(torch.int32)
        if reverse:
            vision = torch.flip(vision, dims=(0,))
            documents = 1 - documents
        vision_parts.append(vision)
        document_parts.append(documents)
    return torch.cat(vision_parts), torch.cat(document_parts)


def _run_payload(
    q_lengths: list[int],
    k_lengths: list[int],
    *,
    seed: int,
    custom: bool,
    reverse_metadata: bool,
    backward: bool,
) -> None:
    q, k, v, cu_q, cu_k = _make_inputs(
        q_lengths,
        k_lengths,
        seed,
        requires_grad=backward,
    )
    kwargs = {}
    if custom:
        vision, documents = _metadata(k_lengths, reverse=reverse_metadata)
        kwargs = {"vision_block_ids": vision, "document_ids": documents}
    out, lse = fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        **kwargs,
    )
    if backward:
        torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
    torch.cuda.synchronize()
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        raise AssertionError("packed-varlen cache payload produced non-finite output")


def _objects(cache_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(cache_dir.rglob("*.o")):
        relative = path.relative_to(cache_dir).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _application_key_digests(keys: Sequence[object]) -> tuple[str, ...]:
    digests = tuple(sorted(hashlib.sha256(pickle.dumps(key)).hexdigest() for key in keys))
    if len(set(digests)) != len(digests):
        raise AssertionError("distinct FA4 forward application keys produced duplicate digests")
    return digests


def _forward_application_snapshot() -> tuple[str, ...]:
    from flash_attn.cute.interface import _flash_attn_fwd

    backing = getattr(_flash_attn_fwd.compile_cache, "cache", None)
    if not isinstance(backing, dict):
        raise AssertionError("the pinned _flash_attn_fwd cache no longer exposes its key map")
    return _application_key_digests(tuple(backing))


def _validate_static_prefix_cases(cases: Sequence[_StaticPrefixCase]) -> None:
    if len(cases) < 2:
        raise ValueError("the StaticCache replay requires at least two physical capacities")
    if len({case.physical_capacity for case in cases}) != len(cases):
        raise ValueError("the StaticCache replay requires distinct physical capacities")
    if any(case.physical_capacity <= _STATIC_PREFIX_K_LENGTH for case in cases):
        raise ValueError("each physical capacity must exceed the logical K prefix")
    if len({case.layout for case in cases}) < 2:
        raise ValueError("the StaticCache replay requires distinct outer-stride layouts")
    if any(case.layout not in {"bhsd-contiguous", "bshd-backed"} for case in cases):
        raise ValueError("the StaticCache replay received an unknown physical layout")


def _empty_bhsd(
    shape: tuple[int, int, int, int],
    *,
    layout: str,
) -> torch.Tensor:
    batch, heads, seqlen, head_dim = shape
    if layout == "bhsd-contiguous":
        tensor = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    elif layout == "bshd-backed":
        backing = torch.empty(
            (batch, seqlen, heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        tensor = backing.transpose(1, 2)
    else:
        raise ValueError(f"unknown StaticCache physical layout: {layout}")
    bshd = tensor.transpose(1, 2)
    if bshd.stride(-1) != 1 or any(stride % 8 for stride in bshd.stride()[:-1]):
        raise AssertionError("StaticCache operand violates the 16-byte BF16 stride contract")
    if torch._debug_has_internal_overlap(bshd) != 0 or bshd.data_ptr() % 16:
        raise AssertionError("StaticCache operand is overlapping or misaligned")
    return tensor


def _make_static_logical_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = SLIDING_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = (
        (1, spec.num_q_heads, _STATIC_PREFIX_Q_LENGTH, spec.head_dim_qk),
        (1, spec.num_kv_heads, _STATIC_PREFIX_K_LENGTH, spec.head_dim_qk),
        (1, spec.num_kv_heads, _STATIC_PREFIX_K_LENGTH, spec.head_dim_v),
    )
    tensors = tuple(_empty_bhsd(shape, layout="bhsd-contiguous") for shape in shapes)
    for tensor in tensors:
        tensor.normal_(mean=0.0, std=0.125, generator=generator)
    return tensors


def _physical_static_operand(
    logical: torch.Tensor,
    case: _StaticPrefixCase,
) -> torch.Tensor:
    physical = _empty_bhsd(
        (logical.shape[0], logical.shape[1], case.physical_capacity, logical.shape[3]),
        layout=case.layout,
    )
    physical[:, :, : logical.shape[2], :].copy_(logical)
    physical[:, :, logical.shape[2] :, :].fill_(float("nan"))
    return physical


def _local_module() -> SimpleNamespace:
    spec = SLIDING_ATTENTION
    return SimpleNamespace(
        layer_idx=0,
        layer_type=spec.kind,
        is_sliding=True,
        head_dim=spec.head_dim_qk,
        num_key_value_groups=spec.qhead_per_kvhead,
        scaling=spec.softmax_scale,
        sliding_window=spec.sliding_window,
    )


def _run_static_prefix_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Gemma4DispatchResult:
    spec = SLIDING_ATTENTION
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise AssertionError("the StaticCache prefix probe accepts B1 only")
    if k.shape[2] != v.shape[2] or k.shape[2] < _STATIC_PREFIX_K_LENGTH:
        raise AssertionError("the StaticCache prefix probe received an invalid physical capacity")
    attention_mask = torch.zeros((1, k.shape[2]), dtype=torch.int32, device=q.device)
    attention_mask[:, :_STATIC_PREFIX_K_LENGTH] = 1
    plan = gemma4_fa4_mask(
        batch_size=1,
        q_length=q.shape[2],
        kv_length=k.shape[2],
        q_offset=_STATIC_PREFIX_K_LENGTH - q.shape[2],
        attention_mask=attention_mask,
    )
    with torch.inference_mode():
        result = gemma4_fa4_prepared(
            _local_module(),
            q,
            k,
            v,
            plan,
            dropout=0.0,
            scaling=1.0,
            sliding_window=spec.sliding_window,
            allow_flex_fallback=False,
        )
    expected_output = (1, _STATIC_PREFIX_Q_LENGTH, spec.num_q_heads, spec.head_dim_v)
    expected_lse = (1, spec.num_q_heads, _STATIC_PREFIX_Q_LENGTH)
    if result.path != "fa4_local_varlen":
        raise AssertionError(f"StaticCache prefix routed to {result.path!r}")
    if result.output.shape != expected_output or result.output.dtype != torch.bfloat16:
        raise AssertionError("StaticCache prefix returned an invalid output contract")
    if result.lse is None or result.lse.shape != expected_lse or result.lse.dtype != torch.float32:
        raise AssertionError("StaticCache prefix returned an invalid FP32 LSE contract")
    if result.output.requires_grad or result.output.grad_fn is not None:
        raise AssertionError("StaticCache prefix unexpectedly retained autograd state")
    if not torch.isfinite(result.output).all() or not torch.isfinite(result.lse).all():
        raise AssertionError("StaticCache prefix returned non-finite output or LSE")
    torch.cuda.synchronize()
    return result


def _require_exact_result(
    reference: Gemma4DispatchResult,
    candidate: Gemma4DispatchResult,
    *,
    label: str,
) -> None:
    if candidate.path != reference.path:
        raise AssertionError(f"{label} changed the adapter path")
    if not torch.equal(candidate.output, reference.output):
        raise AssertionError(f"{label} changed the logical-prefix output")
    if (
        reference.lse is None
        or candidate.lse is None
        or not torch.equal(candidate.lse, reference.lse)
    ):
        raise AssertionError(f"{label} changed the logical-prefix LSE")


def _require_cache_identity(
    expected_objects: dict[str, str],
    observed_objects: dict[str, str],
    expected_applications: tuple[str, ...],
    observed_applications: tuple[str, ...],
    *,
    label: str,
) -> None:
    if observed_objects != expected_objects:
        added = sorted(set(observed_objects) - set(expected_objects))
        removed = sorted(set(expected_objects) - set(observed_objects))
        changed = sorted(
            path
            for path in set(expected_objects) & set(observed_objects)
            if expected_objects[path] != observed_objects[path]
        )
        raise AssertionError(
            f"{label} changed persistent cache objects: "
            f"added={added}, removed={removed}, changed={changed}"
        )
    if observed_applications != expected_applications:
        added = sorted(set(observed_applications) - set(expected_applications))
        removed = sorted(set(expected_applications) - set(observed_applications))
        raise AssertionError(
            f"{label} changed _flash_attn_fwd application keys: added={added}, removed={removed}"
        )


def _run_static_prefix_replay(cache_dir: Path) -> None:
    _require_static_prefix_h100()
    _validate_static_prefix_cases(_STATIC_PREFIX_CASES)
    q, logical_k, logical_v = _make_static_logical_inputs(seed=20_101)

    reference = _run_static_prefix_forward(q, logical_k, logical_v)
    expected_objects = _objects(cache_dir)
    expected_applications = _forward_application_snapshot()
    if not expected_objects or not expected_applications:
        raise AssertionError("dense local Q1/K33 warmup did not retain both cache inventories")
    print(
        f"static_prefix_warm path={reference.path} logical_q={_STATIC_PREFIX_Q_LENGTH} "
        f"logical_k={_STATIC_PREFIX_K_LENGTH} objects={len(expected_objects)} "
        f"forward_application_keys={len(expected_applications)}"
    )

    outer_strides: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for case in _STATIC_PREFIX_CASES:
        k = _physical_static_operand(logical_k, case)
        v = _physical_static_operand(logical_v, case)
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise AssertionError("the StaticCache replay requires distinct K/V storage")
        if not torch.equal(k[:, :, :_STATIC_PREFIX_K_LENGTH], logical_k):
            raise AssertionError("the physical K backing changed the shared logical prefix")
        if not torch.equal(v[:, :, :_STATIC_PREFIX_K_LENGTH], logical_v):
            raise AssertionError("the physical V backing changed the shared logical prefix")
        strides = (tuple(k.stride()[:-1]), tuple(v.stride()[:-1]))
        outer_strides.add(strides)
        label = f"capacity_{case.physical_capacity}_{case.layout}"
        candidate = _run_static_prefix_forward(q, k, v)
        _require_exact_result(reference, candidate, label=label)
        _require_cache_identity(
            expected_objects,
            _objects(cache_dir),
            expected_applications,
            _forward_application_snapshot(),
            label=label,
        )
        print(
            f"static_prefix_replay label={label} physical_k={case.physical_capacity} "
            f"k_outer_strides={strides[0]} v_outer_strides={strides[1]} "
            "output_exact=True lse_exact=True cache_reuse=True"
        )

    if len(outer_strides) != len(_STATIC_PREFIX_CASES):
        raise AssertionError("StaticCache cases did not produce distinct legal outer strides")
    print(
        f"cache_reuse mode=static-prefix-replay objects={len(expected_objects)} "
        f"forward_application_keys={len(expected_applications)} inference_mode=True"
    )
    for relative, digest in expected_objects.items():
        print(f"object={relative} sha256={digest}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true", help="exercise packed metadata callable")
    parser.add_argument("--backward", action="store_true", help="include the public backward path")
    parser.add_argument(
        "--long-text",
        action="store_true",
        help="vary runtime maxima above EXP-0008; combine with --custom for sparse metadata",
    )
    parser.add_argument(
        "--empty-replay",
        action="store_true",
        help="make the cache-reuse payload include leading/middle/trailing empty segments",
    )
    parser.add_argument(
        "--static-prefix-replay",
        action="store_true",
        help="replay one eager B1 local Q1/K33 prefix across physical StaticCache capacities",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _require_h100()
    raw_cache_dir = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    if not raw_cache_dir:
        parser.error("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must name an isolated cache directory")
    cache_dir = Path(raw_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _objects(cache_dir):
        parser.error("the cache directory must not contain pre-existing object files")

    if args.static_prefix_replay:
        conflicting = [
            flag
            for flag, enabled in (
                ("--custom", args.custom),
                ("--backward", args.backward),
                ("--long-text", args.long_text),
                ("--empty-replay", args.empty_replay),
            )
            if enabled
        ]
        if conflicting:
            parser.error(
                "--static-prefix-replay is inference-only and cannot be combined with "
                + ", ".join(conflicting)
            )
        enabled = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "").lower()
        if enabled not in _TRUE_VALUES:
            parser.error("--static-prefix-replay requires persistent CuTe caching")
        _run_static_prefix_replay(cache_dir)
        return 0

    if args.long_text:
        first_q, first_k = [64, 65], [2048, 4097]
        if args.empty_replay:
            second_q, second_k = [0, 65, 0, 33, 0], [1, 4097, 0, 2049, 0]
        else:
            second_q, second_k = [129, 33], [8193, 2049]
    else:
        first_q, first_k = [33, 65], [64, 65]
        if args.empty_replay:
            second_q, second_k = [0, 65, 0, 34, 0], [1, 65, 0, 63, 0]
        else:
            second_q, second_k = [65, 34], [65, 63]

    # Both calls retain the same one/multi-block selectors. Their packed totals,
    # cumulative payloads, segment order, tensor contents, logical batch count
    # in empty-replay mode, and (in long mode) runtime maxima differ. Custom mode
    # also changes metadata contents. None of those runtime values may specialize
    # code.
    _run_payload(
        first_q,
        first_k,
        seed=8801,
        custom=args.custom,
        reverse_metadata=False,
        backward=args.backward,
    )
    first = _objects(cache_dir)
    if not first:
        raise AssertionError("first payload did not retain a compiled object")
    _run_payload(
        second_q,
        second_k,
        seed=8802,
        custom=args.custom,
        reverse_metadata=True,
        backward=args.backward,
    )
    second = _objects(cache_dir)
    if second != first:
        added = sorted(set(second) - set(first))
        removed = sorted(set(first) - set(second))
        raise AssertionError(
            f"runtime payload changed cache objects: added={added}, removed={removed}"
        )

    mode = ("custom" if args.custom else "native") + ("-backward" if args.backward else "-forward")
    if args.long_text:
        mode += "-long-metadata" if args.custom else "-long-text"
    if args.empty_replay:
        mode += "-empty-replay"
    print(f"cache_reuse mode={mode} objects={len(second)}")
    for relative, digest in second.items():
        print(f"object={relative} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
