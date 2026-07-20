#!/usr/bin/env python3
"""Compile/correctness probe for the patched SM90 global d512 adapter."""

from __future__ import annotations

import argparse
import os

import torch

from gemma4_fa4.h100 import fa4_global_text_forward
from gemma4_fa4.model_spec import GLOBAL_ATTENTION
from gemma4_fa4.reference import reference_attention

OUT_ATOL = 0.0625
OUT_RTOL = 0.03
LSE_ATOL = 0.25


def _require_h100() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the global d512 probe requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"the global d512 probe requires SM90, found {capability}")


def _make_qkv(seqlen: int, seed: int) -> tuple[torch.Tensor, ...]:
    spec = GLOBAL_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        1,
        seqlen,
        spec.num_q_heads,
        spec.head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        1,
        seqlen,
        spec.num_kv_heads,
        spec.head_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        1,
        seqlen,
        spec.num_kv_heads,
        spec.head_dim_v,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise RuntimeError("probe requires distinct K and V storage")
    return q, k, v


def _check_case(seqlen: int, *, reference: bool) -> None:
    q, k, v = _make_qkv(seqlen, seed=2000 + seqlen)
    out, lse = fa4_global_text_forward(q, k, v)
    expected_lse_shape = (1, GLOBAL_ATTENTION.num_q_heads, seqlen)
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise AssertionError(f"invalid O contract: {out.shape}, {out.dtype}")
    if lse.shape != expected_lse_shape or lse.dtype != torch.float32:
        raise AssertionError(f"invalid LSE contract: {lse.shape}, {lse.dtype}")
    if not reference:
        print(f"compiled seqlen={seqlen} O={tuple(out.shape)} LSE={tuple(lse.shape)}")
        return

    out_ref, lse_ref = reference_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=1.0,
        sliding_window=None,
        upcast=torch.float32,
        return_lse=True,
    )
    out_ref = out_ref.transpose(1, 2)
    torch.testing.assert_close(out, out_ref, atol=OUT_ATOL, rtol=OUT_RTOL)
    torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=0.0)
    out_err = (out.float() - out_ref.float()).abs().max().item()
    lse_err = (lse - lse_ref).abs().max().item()
    print(f"passed seqlen={seqlen} max_abs_O={out_err:.8g} max_abs_LSE={lse_err:.8g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, action="append", dest="seqlens")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="execute and compare with the locked FP32 reference",
    )
    args = parser.parse_args()
    seqlens = args.seqlens or [128]
    if any(value <= 0 for value in seqlens):
        parser.error("--seqlen must be positive")
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1" and args.reference:
        parser.error("--reference cannot run with FLASH_ATTENTION_FAKE_TENSOR=1")
    _require_h100()
    for seqlen in seqlens:
        _check_case(seqlen, reference=args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
