#!/usr/bin/env python3
"""Compile/correctness probe for exact H100 composed global d512 backward."""

from __future__ import annotations

import argparse
import os

import torch

from gemma4_fa4.h100 import fa4_global_text_forward
from gemma4_fa4.model_spec import GLOBAL_ATTENTION

OUT_ATOL = 0.0625
OUT_RTOL = 0.03
LSE_ATOL = 0.25
UPSTREAM_ERROR_MULTIPLIER = 2.0
DEFAULT_SEQLENS = (1, 31, 32, 33, 63, 64, 65, 127, 128, 129)
GRAD_NAMES = ("dQ", "dK", "dV")
STRUCTURED_ATOL = 1.0
STRUCTURED_RTOL = 0.02


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the global backward probe requires an SM90 CUDA device")


def _make_inputs(seqlen: int, seed: int):
    spec = GLOBAL_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = (
        (1, seqlen, spec.num_q_heads, spec.head_dim_qk),
        (1, seqlen, spec.num_kv_heads, spec.head_dim_qk),
        (1, seqlen, spec.num_kv_heads, spec.head_dim_v),
    )
    q, k, v = (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for shape in shapes
    )
    do = torch.randn(shapes[0], device="cuda", dtype=torch.bfloat16, generator=generator)
    return q, k, v, do


def _causal_mask(seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, 1, seqlen, seqlen), dtype=torch.bool, device=device).tril()


def _expand_kv(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if num_q_heads % x.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    return torch.repeat_interleave(x, num_q_heads // x.shape[2], dim=2)


def _run_fp32_reference(q, k, v, do):
    """Independent prepared-Q/K/V reference with FP32 score/softmax math."""

    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    k_expanded = _expand_kv(k_ref, q_ref.shape[2])
    v_expanded = _expand_kv(v_ref, q_ref.shape[2])
    scores = torch.einsum("bthd,bshd->bhts", q_ref.float(), k_expanded.float()) * 1.0
    scores.masked_fill_(~_causal_mask(q_ref.shape[1], q_ref.device), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1).to(torch.float32)
    probabilities = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", probabilities, v_expanded.float()).to(torch.bfloat16)
    grads = torch.autograd.grad(out, (q_ref, k_ref, v_ref), do)
    return out, lse, grads


def _run_bf16_reference(q, k, v, do):
    """Mirror pinned upstream's independent BF16 PyTorch attention policy."""

    q_pt, k_pt, v_pt = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    k_expanded = _expand_kv(k_pt, q_pt.shape[2])
    v_expanded = _expand_kv(v_pt, q_pt.shape[2])
    scores = torch.einsum("bthd,bshd->bhts", q_pt, k_expanded * 1.0)
    scores.masked_fill_(~_causal_mask(q_pt.shape[1], q_pt.device), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(v_pt.dtype)
    out = torch.einsum("bhts,bshd->bthd", probabilities, v_expanded)
    grads = torch.autograd.grad(out, (q_pt, k_pt, v_pt), do)
    return out, grads


def _run_candidate(q, k, v, do, *, expand_kv_heads: bool):
    if expand_kv_heads:
        from flash_attn.cute import flash_attn_func

        k_expanded = _expand_kv(k, q.shape[2])
        outputs = []
        lses = []
        for v_slab in v.split(256, dim=-1):
            out_slab, lse = flash_attn_func(
                q,
                k_expanded,
                _expand_kv(v_slab, q.shape[2]),
                causal=True,
                softmax_scale=1.0,
                num_splits=1,
                pack_gqa=False,
                return_lse=True,
            )
            outputs.append(out_slab)
            lses.append(lse)
        if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1" and not torch.equal(
            lses[0], lses[1]
        ):
            raise AssertionError("expanded-head V-slab launches returned different LSE values")
        out, lse = torch.cat(outputs, dim=-1), lses[0]
    else:
        out, lse = fa4_global_text_forward(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), do)
    return out, lse, grads


def _quantization_atol(reference: torch.Tensor) -> float:
    roundtrip_error = ((reference + 0.3 - 0.3) - reference).abs()
    return 2.0 * roundtrip_error.max().item()


def _check_gradient_policy(grads, refs, bf16_refs, *, run_label: str) -> list[str]:
    failures = []
    for name, grad, ref, bf16_ref in zip(GRAD_NAMES, grads, refs, bf16_refs, strict=True):
        if not torch.isfinite(grad).all():
            failures.append(f"{run_label} {name} contains non-finite values")
            continue
        abs_error = (grad.float() - ref.float()).abs()
        baseline_error = (bf16_ref.float() - ref.float()).abs()
        max_error = abs_error.max().item()
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
            f"mean_abs={abs_error.mean().item():.8g} baseline_max={baseline_max:.8g} "
            f"baseline_mean={baseline_error.mean().item():.8g} "
            f"quantization_atol={quantization_atol:.8g} limit={limit:.8g}"
        )
    return failures


def _check_contract(q, k, v, out, lse, grads, *, check_storage: bool) -> None:
    spec = GLOBAL_ATTENTION
    expected_shapes = (q.shape, k.shape, v.shape)
    if q.shape[0] != 1 or q.shape[2:] != (spec.num_q_heads, spec.head_dim_qk):
        raise AssertionError(f"invalid exact global Q geometry: {q.shape}")
    if k.shape[2:] != (spec.num_kv_heads, spec.head_dim_qk):
        raise AssertionError(f"invalid exact global K geometry: {k.shape}")
    if v.shape[2:] != (spec.num_kv_heads, spec.head_dim_v):
        raise AssertionError(f"invalid exact global V geometry: {v.shape}")
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise AssertionError(f"invalid O contract: {out.shape}, {out.dtype}")
    expected_lse_shape = (q.shape[0], spec.num_q_heads, q.shape[1])
    if lse.shape != expected_lse_shape or lse.dtype != torch.float32:
        raise AssertionError(f"invalid LSE contract: {lse.shape}, {lse.dtype}")
    for name, grad, shape in zip(GRAD_NAMES, grads, expected_shapes, strict=True):
        if grad.shape != shape or grad.dtype != torch.bfloat16:
            raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
    if check_storage:
        for name, tensor in (("O", out), ("LSE", lse), *zip(GRAD_NAMES, grads, strict=True)):
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{name} contains non-finite values")
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise AssertionError("global K and V must use distinct storage")
        if len({grad.untyped_storage().data_ptr() for grad in grads}) != 3:
            raise AssertionError("dQ, dK, and dV must use distinct storage")


def _check_reference(out, lse, grads, refs, bf16_refs, *, run_label: str) -> list[str]:
    out_ref, lse_ref, grad_refs = refs
    bf16_out, bf16_grad_refs = bf16_refs
    failures = []
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
    bf16_out_error = (bf16_out.float() - out_ref.float()).abs()
    print(
        f"forward {run_label} max_abs_O={out_error.max().item():.8g} "
        f"mean_abs_O={out_error.mean().item():.8g} "
        f"bf16_baseline_max_O={bf16_out_error.max().item():.8g} "
        f"max_abs_LSE={lse_error.max().item():.8g}"
    )
    failures.extend(_check_gradient_policy(grads, grad_refs, bf16_grad_refs, run_label=run_label))
    return failures


def _run_case(
    seqlen: int,
    *,
    expand_kv_heads: bool,
    reference: bool,
    repeats: int,
    nondefault_stream: bool,
) -> None:
    q, k, v, do = _make_inputs(seqlen, seed=5000 + seqlen)
    fake_mode = os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1"
    stream = torch.cuda.Stream() if nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    candidate_runs = []
    for _ in range(repeats):
        if stream is None:
            result = _run_candidate(q, k, v, do, expand_kv_heads=expand_kv_heads)
        else:
            with torch.cuda.stream(stream):
                result = _run_candidate(q, k, v, do, expand_kv_heads=expand_kv_heads)
            stream.synchronize()
        _check_contract(q, k, v, *result, check_storage=not fake_mode)
        candidate_runs.append(result)

    if fake_mode or not reference:
        out, lse, grads = candidate_runs[-1]
        print(
            f"compiled seqlen={seqlen} O={tuple(out.shape)} LSE={tuple(lse.shape)} "
            + " ".join(
                f"{name}={tuple(grad.shape)}" for name, grad in zip(GRAD_NAMES, grads, strict=True)
            )
        )
        return

    refs = _run_fp32_reference(q, k, v, do)
    bf16_refs = _run_bf16_reference(q, k, v, do)
    failures = []
    for run_idx, (out, lse, grads) in enumerate(candidate_runs):
        failures.extend(
            _check_reference(
                out,
                lse,
                grads,
                refs,
                bf16_refs,
                run_label=f"seqlen={seqlen} run={run_idx}",
            )
        )
    if repeats > 1:
        first_out, first_lse, first_grads = candidate_runs[0]
        equality = {
            "O": all(torch.equal(first_out, run[0]) for run in candidate_runs[1:]),
            "LSE": all(torch.equal(first_lse, run[1]) for run in candidate_runs[1:]),
            **{
                name: all(torch.equal(base, run[2][idx]) for run in candidate_runs[1:])
                for idx, (name, base) in enumerate(zip(GRAD_NAMES, first_grads, strict=True))
            },
        }
        print("repeat_exact " + " ".join(f"{name}={value}" for name, value in equality.items()))
    if failures:
        raise AssertionError("\n\n".join(failures))


def _run_structured_case(seqlen: int) -> None:
    """Prove V-slab composition and GQA head ownership with sparse dO."""

    q, k, v, do = _make_inputs(seqlen, seed=7000 + seqlen)
    do_low = do.clone()
    do_low[..., 256:] = 0
    do_high = do.clone()
    do_high[..., :256] = 0

    full = _run_candidate(q, k, v, do, expand_kv_heads=False)
    low = _run_candidate(q, k, v, do_low, expand_kv_heads=False)
    high = _run_candidate(q, k, v, do_high, expand_kv_heads=False)
    for result in (full, low, high):
        _check_contract(q, k, v, *result, check_storage=True)

    failures = []
    for name, full_grad, low_grad, high_grad in zip(
        GRAD_NAMES, full[2], low[2], high[2], strict=True
    ):
        combined = low_grad.float() + high_grad.float()
        error = (full_grad.float() - combined).abs()
        allowed = STRUCTURED_ATOL + STRUCTURED_RTOL * combined.abs()
        if not bool(torch.all(error <= allowed)):
            failures.append(
                f"structured {name} slab superposition max_abs={error.max().item():.8g}"
            )
        print(f"structured_superposition {name} seqlen={seqlen} max_abs={error.max().item():.8g}")

    if torch.count_nonzero(low[2][2][..., 256:]).item() != 0:
        failures.append("low-only dO produced nonzero high-slab dV")
    if torch.count_nonzero(high[2][2][..., :256]).item() != 0:
        failures.append("high-only dO produced nonzero low-slab dV")

    isolated_qhead = 9
    gqa_ratio = GLOBAL_ATTENTION.num_q_heads // GLOBAL_ATTENTION.num_kv_heads
    isolated_kvhead = isolated_qhead // gqa_ratio
    do_isolated = torch.zeros_like(do)
    do_isolated[:, :, isolated_qhead] = do[:, :, isolated_qhead]
    isolated = _run_candidate(q, k, v, do_isolated, expand_kv_heads=False)
    _check_contract(q, k, v, *isolated, check_storage=True)
    isolated_refs = _run_fp32_reference(q, k, v, do_isolated)
    isolated_bf16_refs = _run_bf16_reference(q, k, v, do_isolated)
    failures.extend(
        _check_reference(
            *isolated,
            isolated_refs,
            isolated_bf16_refs,
            run_label=f"isolated-qhead={isolated_qhead} seqlen={seqlen}",
        )
    )

    inactive_qheads = [head for head in range(32) if head != isolated_qhead]
    inactive_kvheads = [head for head in range(4) if head != isolated_kvhead]
    if torch.count_nonzero(isolated[2][0][:, :, inactive_qheads]).item() != 0:
        failures.append("isolated dO leaked into an inactive query-head dQ")
    for name, grad in zip(("dK", "dV"), isolated[2][1:], strict=True):
        if torch.count_nonzero(grad[:, :, inactive_kvheads]).item() != 0:
            failures.append(f"isolated dO leaked into an inactive KV-head {name}")
        if torch.count_nonzero(grad[:, :, isolated_kvhead]).item() == 0:
            failures.append(f"isolated dO did not reach its owning KV-head {name}")

    print(
        f"structured_ownership seqlen={seqlen} qhead={isolated_qhead} "
        f"kvhead={isolated_kvhead} inactive_zero=True"
    )
    if failures:
        raise AssertionError("\n\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, action="append", dest="seqlens")
    parser.add_argument(
        "--expand-kv-heads",
        action="store_true",
        help="run the historical head-expanded diagnostic instead of the project adapter",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument("--reference", dest="reference", action="store_true")
    reference_group.add_argument("--compile-only", dest="reference", action="store_false")
    parser.set_defaults(reference=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nondefault-stream", action="store_true")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="also test V-slab superposition and one-Q-head GQA ownership",
    )
    args = parser.parse_args()
    seqlens = args.seqlens or list(DEFAULT_SEQLENS)
    if any(value <= 0 or value > 1024 for value in seqlens):
        parser.error("--seqlen must be in the proven range 1..1024")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    fake_mode = os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1"
    reference = not fake_mode if args.reference is None else args.reference
    if fake_mode and reference:
        parser.error("--reference cannot run in fake-tensor mode")
    if fake_mode and args.nondefault_stream:
        parser.error("--nondefault-stream cannot run in fake-tensor mode")
    if fake_mode and args.structured:
        parser.error("--structured cannot run in fake-tensor mode")
    _require_h100()
    run_case = _run_case
    if fake_mode:
        from flash_attn.cute.testing import maybe_fake_tensor_mode

        run_case = maybe_fake_tensor_mode(True)(run_case)
    for seqlen in seqlens:
        run_case(
            seqlen,
            expand_kv_heads=args.expand_kv_heads,
            reference=reference,
            repeats=args.repeats,
            nondefault_stream=args.nondefault_stream,
        )
    if args.structured:
        _run_structured_case(seqlens[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
