#!/usr/bin/env python3
"""Compile/correctness probe for exact H100 local d256 backward."""

from __future__ import annotations

import argparse
import os

import torch

from gemma4_fa4.h100 import fa4_local_text_forward
from gemma4_fa4.model_spec import SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention

GRAD_ATOL = 0.125
GRAD_RTOL = 0.05


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.seqlen <= 0:
        parser.error("--seqlen must be positive")
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1" and args.reference:
        parser.error("--reference cannot run in fake-tensor mode")
    _require_h100()
    q, k, v, do = _make_inputs(args.seqlen, seed=3000 + args.seqlen)
    grads = _run_candidate(q, k, v, do)
    expected_shapes = (q.shape, k.shape, v.shape)
    for name, grad, shape in zip(("dQ", "dK", "dV"), grads, expected_shapes, strict=True):
        if grad.shape != shape or grad.dtype != torch.bfloat16:
            raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
    grad_storage = [grad.untyped_storage().data_ptr() for grad in grads]
    if len(set(grad_storage)) != 3:
        raise AssertionError("dQ, dK, and dV must use distinct storage")
    if args.reference:
        refs = _run_reference(q, k, v, do)
        failures = []
        for name, grad, ref in zip(("dQ", "dK", "dV"), grads, refs, strict=True):
            abs_error = (grad.float() - ref.float()).abs()
            max_error = abs_error.max().item()
            mean_error = abs_error.mean().item()
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
                print(
                    f"failed {name} seqlen={args.seqlen} "
                    f"max_abs={max_error:.8g} mean_abs={mean_error:.8g}"
                )
            else:
                print(
                    f"passed {name} seqlen={args.seqlen} "
                    f"max_abs={max_error:.8g} mean_abs={mean_error:.8g}"
                )
        if failures:
            raise AssertionError("\n\n".join(failures))
    else:
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
