#!/usr/bin/env python3
"""Compile/correctness probe for exact H100 local d256 backward."""

from __future__ import annotations

import argparse
import os
from typing import Literal

import torch

from gemma4_fa4.h100 import fa4_local_text_forward
from gemma4_fa4.masks import gemma4_attention_mask
from gemma4_fa4.model_spec import SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention

GRAD_ATOL = 0.125
GRAD_RTOL = 0.05
UPSTREAM_ERROR_MULTIPLIER = 2.0

ComparisonPolicy = Literal["frozen", "upstream-relative"]


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the local backward probe requires an SM90 CUDA device")


def _make_inputs(seqlen: int, seed: int):
    spec = SLIDING_ATTENTION
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
    do = torch.randn(
        shapes[0],
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return q, k, v, do


def _run_candidate(q, k, v, do):
    out, _ = fa4_local_text_forward(q, k, v)
    return torch.autograd.grad(out, (q, k, v), do)


def _run_reference(q, k, v, do):
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    out_ref = reference_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        softmax_scale=1.0,
        sliding_window=1024,
        upcast=torch.float32,
    ).transpose(1, 2)
    return torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do)


def _run_upstream_style_bf16_baseline(q, k, v, do):
    """Mirror pinned ``attention_ref(upcast=False, reorder_ops=True)`` at scale 1."""

    q_pt, k_pt, v_pt = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    gqa_ratio = q_pt.shape[2] // k_pt.shape[2]
    k_expanded = torch.repeat_interleave(k_pt, gqa_ratio, dim=2)
    v_expanded = torch.repeat_interleave(v_pt, gqa_ratio, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q_pt, k_expanded * 1.0)
    allowed = gemma4_attention_mask(
        batch_size=q_pt.shape[0],
        q_len=q_pt.shape[1],
        kv_len=k_pt.shape[1],
        device=q_pt.device,
        sliding_window=1024,
    )
    scores.masked_fill_(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(v_pt.dtype)
    out_pt = torch.einsum("bhts,bshd->bthd", probabilities, v_expanded)
    return torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), do)


def _quantization_atol(reference: torch.Tensor) -> float:
    roundtrip_error = ((reference + 0.3 - 0.3) - reference).abs()
    return 2.0 * roundtrip_error.max().item()


def _check_gradients(
    grads,
    refs,
    *,
    policy: ComparisonPolicy,
    bf16_refs=None,
    run_label: str,
) -> list[str]:
    failures = []
    for name, grad, ref in zip(("dQ", "dK", "dV"), grads, refs, strict=True):
        if not torch.isfinite(grad).all():
            failures.append(f"{run_label} {name} contains non-finite values")
            continue
        abs_error = (grad.float() - ref.float()).abs()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        if policy == "frozen":
            try:
                torch.testing.assert_close(
                    grad,
                    ref,
                    atol=GRAD_ATOL,
                    rtol=GRAD_RTOL,
                    msg=f"{name} exceeds the frozen gradient envelope",
                )
            except AssertionError as exc:
                failures.append(str(exc))
                result = "failed"
            else:
                result = "passed"
            print(f"{result} {name} {run_label} max_abs={max_error:.8g} mean_abs={mean_error:.8g}")
            continue

        if bf16_refs is None:
            raise AssertionError("upstream-relative comparison requires BF16 references")
        bf16_ref = bf16_refs[("dQ", "dK", "dV").index(name)]
        baseline_error = (bf16_ref.float() - ref.float()).abs()
        baseline_max = baseline_error.max().item()
        baseline_mean = baseline_error.mean().item()
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
            f"mean_abs={mean_error:.8g} baseline_max={baseline_max:.8g} "
            f"baseline_mean={baseline_mean:.8g} quantization_atol={quantization_atol:.8g} "
            f"limit={limit:.8g}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--comparison-policy",
        choices=("frozen", "upstream-relative"),
        default="frozen",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nondefault-stream", action="store_true")
    args = parser.parse_args()
    if args.seqlen <= 0:
        parser.error("--seqlen must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.comparison_policy != "frozen" and not args.reference:
        parser.error("--comparison-policy upstream-relative requires --reference")
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1" and args.reference:
        parser.error("--reference cannot run in fake-tensor mode")
    _require_h100()
    q, k, v, do = _make_inputs(args.seqlen, seed=3000 + args.seqlen)
    stream = torch.cuda.Stream() if args.nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    candidate_runs = []
    for _ in range(args.repeats):
        if stream is None:
            grads = _run_candidate(q, k, v, do)
        else:
            with torch.cuda.stream(stream):
                grads = _run_candidate(q, k, v, do)
            stream.synchronize()
        candidate_runs.append(grads)
    expected_shapes = (q.shape, k.shape, v.shape)
    for run_idx, grads in enumerate(candidate_runs):
        for name, grad, shape in zip(("dQ", "dK", "dV"), grads, expected_shapes, strict=True):
            if grad.shape != shape or grad.dtype != torch.bfloat16:
                raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
        grad_storage = [grad.untyped_storage().data_ptr() for grad in grads]
        if len(set(grad_storage)) != 3:
            raise AssertionError(f"run {run_idx} dQ, dK, and dV must use distinct storage")
    if args.reference:
        refs = _run_reference(q, k, v, do)
        bf16_refs = (
            _run_upstream_style_bf16_baseline(q, k, v, do)
            if args.comparison_policy == "upstream-relative"
            else None
        )
        failures = []
        for run_idx, grads in enumerate(candidate_runs):
            failures.extend(
                _check_gradients(
                    grads,
                    refs,
                    policy=args.comparison_policy,
                    bf16_refs=bf16_refs,
                    run_label=f"seqlen={args.seqlen} run={run_idx}",
                )
            )
        if args.repeats > 1:
            first = candidate_runs[0]
            equality = {
                name: all(torch.equal(base, other[idx]) for other in candidate_runs[1:])
                for idx, (name, base) in enumerate(zip(("dQ", "dK", "dV"), first, strict=True))
            }
            print("repeat_exact " + " ".join(f"{name}={value}" for name, value in equality.items()))
        if failures:
            raise AssertionError("\n\n".join(failures))
    else:
        grads = candidate_runs[-1]
        print(
            "compiled "
            + " ".join(
                f"{name}={tuple(grad.shape)}"
                for name, grad in zip(("dQ", "dK", "dV"), grads, strict=True)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
