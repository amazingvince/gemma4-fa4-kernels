#!/usr/bin/env python3
"""Correctness and resource probe for native H100 global-varlen backward."""

from __future__ import annotations

import argparse
from itertools import accumulate
from typing import Literal

import torch

from gemma4_fa4.h100 import (
    _global_varlen_backward_additional_bytes,
    fa4_global_text_forward,
    fa4_global_varlen_forward,
)
from gemma4_fa4.model_spec import GLOBAL_ATTENTION
from gemma4_fa4.reference import reference_attention_varlen

OUT_ATOL = 0.0625
OUT_RTOL = 0.03
LSE_ATOL = 0.25
UPSTREAM_ERROR_MULTIPLIER = 2.0
GRAD_NAMES = ("dQ", "dK", "dV")
GRADIENT_SOURCES = ("out", "lse", "out_lse")
MAX_SEQLEN = 2048

DEFAULT_Q_LENGTHS = (33, 65)
DEFAULT_K_LENGTHS = (65, 129)
PRESET_LENGTHS = {
    "b33-tiny": ((1,) * 33, (1,) * 33),
    "mixed": ((1, 33, 65), (33, 65, 129)),
    "reversed": ((65, 33, 1), (129, 65, 33)),
}

GradientSource = Literal["out", "lse", "out_lse"]


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the global-varlen backward probe requires an SM90 CUDA device")


def _parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must contain positive integers")
    return lengths


def _validate_lengths(q_lengths: tuple[int, ...], k_lengths: tuple[int, ...]) -> None:
    if len(q_lengths) != len(k_lengths):
        raise ValueError("Q and K length lists must have the same batch count")
    for index, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths, strict=True)):
        if not (1 <= q_length <= k_length <= MAX_SEQLEN):
            raise ValueError(f"packed segment {index} must satisfy 1 <= Sq <= Sk <= {MAX_SEQLEN}")


def _resolve_lengths(
    preset: str | None,
    q_lengths: tuple[int, ...] | None,
    k_lengths: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if preset is not None:
        if q_lengths is not None or k_lengths is not None:
            raise ValueError("--case cannot be combined with --q-lengths or --k-lengths")
        return PRESET_LENGTHS[preset]
    if q_lengths is None:
        if k_lengths is not None:
            raise ValueError("--k-lengths requires --q-lengths")
        return DEFAULT_Q_LENGTHS, DEFAULT_K_LENGTHS
    return q_lengths, q_lengths if k_lengths is None else k_lengths


def _cumulative_tensor(
    lengths: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor((0, *accumulate(lengths)), device=device, dtype=torch.int32)


def _segment_bounds(lengths: tuple[int, ...], index: int) -> tuple[int, int]:
    if not 0 <= index < len(lengths):
        raise IndexError("packed segment index is out of range")
    start = sum(lengths[:index])
    return start, start + lengths[index]


def _make_inputs(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    seed: int,
):
    _validate_lengths(q_lengths, k_lengths)
    spec = GLOBAL_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_shape = (sum(q_lengths), spec.num_q_heads, spec.head_dim_qk)
    kv_shape = (sum(k_lengths), spec.num_kv_heads, spec.head_dim_qk)
    q = torch.randn(
        q_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    k = torch.randn(
        kv_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    v = torch.randn(
        kv_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    do = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    dlse = torch.randn(
        (spec.num_q_heads, sum(q_lengths)),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return (
        q,
        k,
        v,
        do,
        dlse,
        _cumulative_tensor(q_lengths, device="cuda"),
        _cumulative_tensor(k_lengths, device="cuda"),
    )


def _differentiate(
    out: torch.Tensor,
    lse: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    do: torch.Tensor,
    dlse: torch.Tensor | None,
    gradient_source: GradientSource,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gradient_source == "out":
        return torch.autograd.grad(out, inputs, do)
    if dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    if gradient_source == "lse":
        grads = torch.autograd.grad(lse, inputs, dlse, allow_unused=True)
        return tuple(
            torch.zeros_like(tensor) if grad is None else grad
            for tensor, grad in zip(inputs, grads, strict=True)
        )
    return torch.autograd.grad((out, lse), inputs, (do, dlse))


def _run_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dlse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    gradient_source: GradientSource,
):
    out, lse = fa4_global_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
    )
    grads = _differentiate(out, lse, (q, k, v), do, dlse, gradient_source)
    return out, lse, grads


def _run_fixed_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dlse: torch.Tensor,
    gradient_source: GradientSource,
):
    """Run the fixed BSHD ABI and normalize its result to native THD shapes."""

    if q.shape[0] != k.shape[0]:
        raise ValueError("fixed parity requires equal Q and K lengths")
    q_fixed, k_fixed, v_fixed = (
        tensor.detach().clone().unsqueeze(0).requires_grad_(True) for tensor in (q, k, v)
    )
    out, lse = fa4_global_text_forward(q_fixed, k_fixed, v_fixed)
    grads = _differentiate(
        out,
        lse,
        (q_fixed, k_fixed, v_fixed),
        do.unsqueeze(0),
        dlse.unsqueeze(0),
        gradient_source,
    )
    return out.squeeze(0), lse.squeeze(0), tuple(grad.squeeze(0) for grad in grads)


def _run_fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dlse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    gradient_source: GradientSource,
):
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    out, lse = reference_attention_varlen(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        softmax_scale=1.0,
        sliding_window=None,
        allow_vision_bidirectional=False,
        upcast=torch.float32,
        return_lse=True,
    )
    grads = _differentiate(out, lse, (q_ref, k_ref, v_ref), do, dlse, gradient_source)
    return out, lse, grads


def _expand_kv(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if num_q_heads % x.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    return torch.repeat_interleave(x, num_q_heads // x.shape[2], dim=2)


def _run_bf16_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dlse: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    gradient_source: GradientSource,
):
    """Mirror the pinned upstream BF16 reference policy per packed segment."""

    q_pt, k_pt, v_pt = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_start = 0
    k_start = 0
    for q_length, k_length in zip(q_lengths, k_lengths, strict=True):
        q_i = q_pt[q_start : q_start + q_length].unsqueeze(0)
        k_i = k_pt[k_start : k_start + k_length].unsqueeze(0)
        v_i = v_pt[k_start : k_start + k_length].unsqueeze(0)
        k_expanded = _expand_kv(k_i, q_i.shape[2])
        v_expanded = _expand_kv(v_i, q_i.shape[2])
        scores = torch.einsum("bthd,bshd->bhts", q_i, k_expanded * 1.0)
        q_positions = torch.arange(q_length, device=q.device) + k_length - q_length
        k_positions = torch.arange(k_length, device=q.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        scores.masked_fill_(~allowed.view(1, 1, q_length, k_length), float("-inf"))
        lse_i = torch.logsumexp(scores.float(), dim=-1)
        probabilities = torch.softmax(scores, dim=-1).to(v_pt.dtype)
        outputs.append(torch.einsum("bhts,bshd->bthd", probabilities, v_expanded).squeeze(0))
        lses.append(lse_i.squeeze(0))
        q_start += q_length
        k_start += k_length
    out = torch.cat(outputs, dim=0)
    lse = torch.cat(lses, dim=1)
    grads = _differentiate(out, lse, (q_pt, k_pt, v_pt), do, dlse, gradient_source)
    return out, grads


def _quantization_atol(reference: torch.Tensor) -> float:
    roundtrip_error = ((reference + 0.3 - 0.3) - reference).abs()
    return 2.0 * roundtrip_error.max().item()


def _check_gradient_policy(
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    refs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bf16_refs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    run_label: str,
) -> list[str]:
    failures: list[str] = []
    for name, grad, ref, bf16_ref in zip(GRAD_NAMES, grads, refs, bf16_refs, strict=True):
        if not torch.isfinite(grad).all():
            failures.append(f"{run_label} {name} contains non-finite values")
            continue
        absolute_error = (grad.float() - ref.float()).abs()
        baseline_error = (bf16_ref.float() - ref.float()).abs()
        max_error = absolute_error.max().item()
        baseline_max = baseline_error.max().item()
        quantization_atol = _quantization_atol(ref)
        limit = UPSTREAM_ERROR_MULTIPLIER * baseline_max + quantization_atol
        result = "passed" if max_error <= limit else "failed"
        if result == "failed":
            failures.append(
                f"{run_label} {name} max_abs={max_error:.8g} exceeds "
                f"upstream-relative limit={limit:.8g}"
            )
        print(
            f"{result} {name} {run_label} max_abs={max_error:.8g} "
            f"mean_abs={absolute_error.mean().item():.8g} baseline_max={baseline_max:.8g} "
            f"baseline_mean={baseline_error.mean().item():.8g} "
            f"quantization_atol={quantization_atol:.8g} limit={limit:.8g}"
        )
    return failures


def _check_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    gradient_source: GradientSource,
) -> None:
    spec = GLOBAL_ATTENTION
    if q.shape[1:] != (spec.num_q_heads, spec.head_dim_qk):
        raise AssertionError(f"invalid exact global Q geometry: {q.shape}")
    if k.shape[1:] != (spec.num_kv_heads, spec.head_dim_qk):
        raise AssertionError(f"invalid exact global K geometry: {k.shape}")
    if v.shape != (k.shape[0], spec.num_kv_heads, spec.head_dim_v):
        raise AssertionError(f"invalid exact global V geometry: {v.shape}")
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise AssertionError(f"invalid O contract: {out.shape}, {out.dtype}")
    if lse.shape != (spec.num_q_heads, q.shape[0]) or lse.dtype != torch.float32:
        raise AssertionError(f"invalid LSE contract: {lse.shape}, {lse.dtype}")
    for name, grad, tensor in zip(GRAD_NAMES, grads, (q, k, v), strict=True):
        if grad.shape != tensor.shape or grad.dtype != torch.bfloat16:
            raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
        if not torch.isfinite(grad).all():
            raise AssertionError(f"{name} contains non-finite values")
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        raise AssertionError("O or LSE contains non-finite values")
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("global K and V must use distinct storage")
    if len({grad.untyped_storage().data_ptr() for grad in grads}) != 3:
        raise AssertionError("dQ, dK, and dV must use distinct storage")
    if gradient_source == "lse" and torch.count_nonzero(grads[2]).item() != 0:
        raise AssertionError("LSE-only differentiation must produce exact-zero dV")


def _check_reference(
    result,
    refs,
    bf16_refs,
    *,
    run_label: str,
) -> list[str]:
    out, lse, grads = result
    out_ref, lse_ref, grad_refs = refs
    bf16_out, bf16_grad_refs = bf16_refs
    failures: list[str] = []
    try:
        torch.testing.assert_close(
            out,
            out_ref,
            atol=OUT_ATOL,
            rtol=OUT_RTOL,
            msg="O exceeds the frozen global forward envelope",
        )
    except AssertionError as exc:
        failures.append(f"{run_label} {exc}")
    try:
        torch.testing.assert_close(
            lse,
            lse_ref,
            atol=LSE_ATOL,
            rtol=0.0,
            msg="LSE exceeds the frozen global forward envelope",
        )
    except AssertionError as exc:
        failures.append(f"{run_label} {exc}")
    out_error = (out.float() - out_ref.float()).abs()
    lse_error = (lse - lse_ref).abs()
    baseline_out_error = (bf16_out.float() - out_ref.float()).abs()
    print(
        f"forward {run_label} max_abs_O={out_error.max().item():.8g} "
        f"mean_abs_O={out_error.mean().item():.8g} "
        f"bf16_baseline_max_O={baseline_out_error.max().item():.8g} "
        f"max_abs_LSE={lse_error.max().item():.8g}"
    )
    failures.extend(_check_gradient_policy(grads, grad_refs, bf16_grad_refs, run_label=run_label))
    return failures


def _check_fixed_parity(
    native,
    fixed,
    refs,
    bf16_refs,
) -> list[str]:
    """Compare the native and fixed ABIs under the frozen numerical policy."""

    native_out, native_lse, native_grads = native
    fixed_out, fixed_lse, fixed_grads = fixed
    _ref_out, _ref_lse, grad_refs = refs
    _bf16_out, bf16_grad_refs = bf16_refs
    failures: list[str] = []
    try:
        torch.testing.assert_close(
            native_out,
            fixed_out,
            atol=OUT_ATOL,
            rtol=OUT_RTOL,
            msg="native/fixed O parity exceeds the frozen forward envelope",
        )
    except AssertionError as exc:
        failures.append(str(exc))
    try:
        torch.testing.assert_close(
            native_lse,
            fixed_lse,
            atol=LSE_ATOL,
            rtol=0.0,
            msg="native/fixed LSE parity exceeds the frozen forward envelope",
        )
    except AssertionError as exc:
        failures.append(str(exc))

    observations = [
        f"O_max_abs={(native_out.float() - fixed_out.float()).abs().max().item():.8g}",
        f"LSE_max_abs={(native_lse - fixed_lse).abs().max().item():.8g}",
    ]
    for name, native_grad, fixed_grad, ref, bf16_ref in zip(
        GRAD_NAMES,
        native_grads,
        fixed_grads,
        grad_refs,
        bf16_grad_refs,
        strict=True,
    ):
        delta = (native_grad.float() - fixed_grad.float()).abs().max().item()
        baseline_max = (bf16_ref.float() - ref.float()).abs().max().item()
        single_route_limit = UPSTREAM_ERROR_MULTIPLIER * baseline_max + _quantization_atol(ref)
        pairwise_limit = 2.0 * single_route_limit
        observations.append(f"{name}_max_abs={delta:.8g}")
        observations.append(f"{name}_limit={pairwise_limit:.8g}")
        if delta > pairwise_limit:
            failures.append(
                f"native/fixed {name} max_abs={delta:.8g} exceeds "
                f"pairwise frozen-policy limit={pairwise_limit:.8g}"
            )
    print("fixed_native_parity " + " ".join(observations))
    return failures


def _max_pairwise_abs(first: torch.Tensor, others: list[torch.Tensor]) -> float:
    return max(
        ((first.float() - other.float()).abs().max().item() for other in others),
        default=0.0,
    )


def _report_repeats(candidate_runs: list[tuple[torch.Tensor, torch.Tensor, tuple]]) -> None:
    if len(candidate_runs) <= 1:
        return
    first_out, first_lse, first_grads = candidate_runs[0]
    names_and_tensors = (
        ("O", first_out, [run[0] for run in candidate_runs[1:]]),
        ("LSE", first_lse, [run[1] for run in candidate_runs[1:]]),
        *(
            (
                name,
                first_grads[index],
                [run[2][index] for run in candidate_runs[1:]],
            )
            for index, name in enumerate(GRAD_NAMES)
        ),
    )
    observations = []
    for name, first, others in names_and_tensors:
        bitwise = all(torch.equal(first, other) for other in others)
        observations.append(
            f"{name}_bitwise={bitwise} {name}_max_pairwise_abs={_max_pairwise_abs(first, others):.8g}"
        )
    print("repeat_observation " + " ".join(observations))
    print("repeat_policy dQ_dK_determinism_claim=False each_run_checked_independently=True")


def _assert_inactive_gradients_zero(
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    q_bounds: tuple[int, int],
    k_bounds: tuple[int, int],
) -> None:
    q_start, q_end = q_bounds
    k_start, k_end = k_bounds
    q_active = torch.zeros(grads[0].shape[0], dtype=torch.bool, device=grads[0].device)
    k_active = torch.zeros(grads[1].shape[0], dtype=torch.bool, device=grads[1].device)
    q_active[q_start:q_end] = True
    k_active[k_start:k_end] = True
    for name, grad, active in (
        ("dQ", grads[0], q_active),
        ("dK", grads[1], k_active),
        ("dV", grads[2], k_active),
    ):
        if torch.count_nonzero(grad[~active]).item() != 0:
            raise AssertionError(f"inactive packed segments received nonzero {name}")


def _run_isolation(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dlse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    gradient_source: GradientSource,
) -> None:
    if len(q_lengths) < 2:
        raise ValueError("isolation requires at least two packed segments")
    q_bounds = _segment_bounds(q_lengths, 0)
    k_bounds = _segment_bounds(k_lengths, 0)
    q_start, q_end = q_bounds
    k_start, k_end = k_bounds
    do_isolated = torch.zeros_like(do)
    dlse_isolated = torch.zeros_like(dlse)
    do_isolated[q_start:q_end] = do[q_start:q_end]
    dlse_isolated[:, q_start:q_end] = dlse[:, q_start:q_end]

    def clone_inputs():
        return tuple(tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))

    q_base, k_base, v_base = clone_inputs()
    base = _run_candidate(
        q_base,
        k_base,
        v_base,
        do_isolated,
        dlse_isolated,
        cu_q,
        cu_k,
        q_lengths,
        k_lengths,
        gradient_source,
    )
    _assert_inactive_gradients_zero(base[2], q_bounds=q_bounds, k_bounds=k_bounds)
    reference_failures = _check_reference(
        base,
        _run_fp32_reference(
            q_base,
            k_base,
            v_base,
            do_isolated,
            dlse_isolated,
            cu_q,
            cu_k,
            gradient_source,
        ),
        _run_bf16_reference(
            q_base,
            k_base,
            v_base,
            do_isolated,
            dlse_isolated,
            q_lengths,
            k_lengths,
            gradient_source,
        ),
        run_label="packed_isolation_base",
    )

    q_mutated, k_mutated, v_mutated = clone_inputs()
    with torch.no_grad():
        q_mutated[q_end:] = 7.0
        k_mutated[k_end:] = -9.0
        v_mutated[k_end:] = 11.0
    mutated = _run_candidate(
        q_mutated,
        k_mutated,
        v_mutated,
        do_isolated,
        dlse_isolated,
        cu_q,
        cu_k,
        q_lengths,
        k_lengths,
        gradient_source,
    )
    _assert_inactive_gradients_zero(mutated[2], q_bounds=q_bounds, k_bounds=k_bounds)
    reference_failures.extend(
        _check_reference(
            mutated,
            _run_fp32_reference(
                q_mutated,
                k_mutated,
                v_mutated,
                do_isolated,
                dlse_isolated,
                cu_q,
                cu_k,
                gradient_source,
            ),
            _run_bf16_reference(
                q_mutated,
                k_mutated,
                v_mutated,
                do_isolated,
                dlse_isolated,
                q_lengths,
                k_lengths,
                gradient_source,
            ),
            run_label="packed_isolation_hostile_mutation",
        )
    )
    if not torch.equal(base[0][q_start:q_end], mutated[0][q_start:q_end]):
        raise AssertionError("inactive-segment mutation changed active-segment O")
    if not torch.equal(base[1][:, q_start:q_end], mutated[1][:, q_start:q_end]):
        raise AssertionError("inactive-segment mutation changed active-segment LSE")
    gradient_drift = {
        name: (base_grad.float() - mutated_grad.float()).abs().max().item()
        for name, base_grad, mutated_grad in zip(GRAD_NAMES, base[2], mutated[2], strict=True)
    }
    if reference_failures:
        raise AssertionError(
            "packed-segment isolation reference failure:\n\n" + "\n\n".join(reference_failures)
        )
    print(
        "passed packed_segment_isolation active_segment=0 "
        f"q_bounds={q_bounds} k_bounds={k_bounds} inactive_gradients_exact_zero=True "
        + " ".join(f"{name}_mutation_max_abs={value:.8g}" for name, value in gradient_drift.items())
    )


def _format_lengths(lengths: tuple[int, ...]) -> str:
    return ",".join(str(length) for length in lengths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-lengths", type=_parse_lengths)
    parser.add_argument(
        "--k-lengths",
        type=_parse_lengths,
        help="comma-separated K lengths; defaults to --q-lengths when Q is explicit",
    )
    parser.add_argument(
        "--case",
        choices=tuple(PRESET_LENGTHS),
        help="named scheduler case; cannot be combined with explicit lengths",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument("--reference", dest="reference", action="store_true")
    reference_group.add_argument("--compile-only", dest="reference", action="store_false")
    parser.set_defaults(reference=None)
    parser.add_argument(
        "--gradient-source",
        choices=GRADIENT_SOURCES,
        default="out_lse",
        help="differentiate O, LSE, or both outputs",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nondefault-stream", action="store_true")
    parser.add_argument("--record-memory", action="store_true")
    parser.add_argument("--isolation", action="store_true")
    parser.add_argument(
        "--fixed-parity",
        action="store_true",
        help="compare one equal-length native segment directly with the fixed ABI",
    )
    parser.add_argument("--seed", type=int, default=13013)
    args = parser.parse_args()
    try:
        q_lengths, k_lengths = _resolve_lengths(args.case, args.q_lengths, args.k_lengths)
        _validate_lengths(q_lengths, k_lengths)
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.record_memory and args.repeats != 1:
        parser.error("--record-memory requires --repeats 1")
    if args.isolation and len(q_lengths) < 2:
        parser.error("--isolation requires at least two packed segments")
    reference = True if args.reference is None else args.reference
    if args.fixed_parity and (len(q_lengths) != 1 or q_lengths != k_lengths):
        parser.error("--fixed-parity requires one equal-length Q/K segment")
    if args.fixed_parity and not reference:
        parser.error("--fixed-parity requires reference checks")
    _require_h100()

    q, k, v, do, dlse, cu_q, cu_k = _make_inputs(q_lengths, k_lengths, args.seed)
    if args.record_memory:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(q.device)
        baseline_allocated = torch.cuda.memory_allocated(q.device)
    stream = torch.cuda.Stream() if args.nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())

    candidate_runs = []
    for _ in range(args.repeats):
        if stream is None:
            result = _run_candidate(
                q,
                k,
                v,
                do,
                dlse,
                cu_q,
                cu_k,
                q_lengths,
                k_lengths,
                args.gradient_source,
            )
        else:
            with torch.cuda.stream(stream):
                result = _run_candidate(
                    q,
                    k,
                    v,
                    do,
                    dlse,
                    cu_q,
                    cu_k,
                    q_lengths,
                    k_lengths,
                    args.gradient_source,
                )
                out, lse, grads = result
                _ = out.float().sum() + lse.sum() + sum(grad.float().sum() for grad in grads)
            stream.synchronize()
        _check_contract(q, k, v, *result, gradient_source=args.gradient_source)
        candidate_runs.append(result)

    if args.record_memory:
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated(q.device) - baseline_allocated
        estimated = _global_varlen_backward_additional_bytes(q, k, len(q_lengths))
        if peak_delta > estimated:
            raise AssertionError(f"measured peak delta {peak_delta} exceeds estimator {estimated}")
        print(
            f"memory batch={len(q_lengths)} total_q={q.shape[0]} total_k={k.shape[0]} "
            f"peak_delta={peak_delta} estimated_additional={estimated} bounded=True"
        )

    label = (
        f"q_lengths={_format_lengths(q_lengths)} k_lengths={_format_lengths(k_lengths)} "
        f"source={args.gradient_source}"
    )
    if reference:
        refs = _run_fp32_reference(
            q,
            k,
            v,
            do,
            dlse,
            cu_q,
            cu_k,
            args.gradient_source,
        )
        bf16_refs = _run_bf16_reference(
            q,
            k,
            v,
            do,
            dlse,
            q_lengths,
            k_lengths,
            args.gradient_source,
        )
        failures: list[str] = []
        for run_index, result in enumerate(candidate_runs):
            failures.extend(
                _check_reference(
                    result,
                    refs,
                    bf16_refs,
                    run_label=f"{label} run={run_index}",
                )
            )
        if args.fixed_parity:
            fixed = _run_fixed_candidate(q, k, v, do, dlse, args.gradient_source)
            _check_contract(q, k, v, *fixed, gradient_source=args.gradient_source)
            failures.extend(
                _check_reference(
                    fixed,
                    refs,
                    bf16_refs,
                    run_label=f"fixed_parity {label}",
                )
            )
            failures.extend(_check_fixed_parity(candidate_runs[0], fixed, refs, bf16_refs))
        if failures:
            raise AssertionError("\n\n".join(failures))
    else:
        out, lse, grads = candidate_runs[-1]
        print(
            f"compiled {label} O={tuple(out.shape)} LSE={tuple(lse.shape)} "
            + " ".join(
                f"{name}={tuple(grad.shape)}" for name, grad in zip(GRAD_NAMES, grads, strict=True)
            )
        )

    _report_repeats(candidate_runs)
    if args.isolation:
        _run_isolation(
            q,
            k,
            v,
            do,
            dlse,
            cu_q,
            cu_k,
            q_lengths,
            k_lengths,
            args.gradient_source,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
