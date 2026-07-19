#!/usr/bin/env python3
"""Compile probe for exact H100 composed global d512 backward."""

from __future__ import annotations

import argparse
import os

import torch

from gemma4_fa4.h100 import fa4_global_text_forward
from gemma4_fa4.model_spec import GLOBAL_ATTENTION


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


def _run(seqlen: int, *, expand_kv_heads: bool) -> None:
    q, k, v, do = _make_inputs(seqlen, seed=5000 + seqlen)
    if expand_kv_heads:
        from flash_attn.cute import flash_attn_func

        gqa_ratio = q.shape[2] // k.shape[2]
        k_expanded = torch.repeat_interleave(k, gqa_ratio, dim=2)
        outputs = [
            flash_attn_func(
                q,
                k_expanded,
                torch.repeat_interleave(v_slab, gqa_ratio, dim=2),
                causal=True,
                softmax_scale=1.0,
                num_splits=1,
                pack_gqa=False,
            )[0]
            for v_slab in v.split(256, dim=-1)
        ]
        out = torch.cat(outputs, dim=-1)
    else:
        out, _ = fa4_global_text_forward(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), do)
    for name, grad, expected in zip(("dQ", "dK", "dV"), grads, (q, k, v), strict=True):
        if grad.shape != expected.shape or grad.dtype != torch.bfloat16:
            raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1":
        if len({grad.untyped_storage().data_ptr() for grad in grads}) != 3:
            raise AssertionError("dQ, dK, and dV must use distinct storage")
    print(
        "compiled "
        + " ".join(
            f"{name}={tuple(grad.shape)}"
            for name, grad in zip(("dQ", "dK", "dV"), grads, strict=True)
        )
    )
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1":
        print("warning: compile-only probe ran on real tensors without a reference check")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--expand-kv-heads", action="store_true")
    args = parser.parse_args()
    if args.seqlen <= 0:
        parser.error("--seqlen must be positive")
    _require_h100()
    run = _run
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1":
        from flash_attn.cute.testing import maybe_fake_tensor_mode

        run = maybe_fake_tensor_mode(True)(run)
    run(args.seqlen, expand_kv_heads=args.expand_kv_heads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
