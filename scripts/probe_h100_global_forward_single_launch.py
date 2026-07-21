#!/usr/bin/env python3
"""Compile/correctness discriminator for EXP-0041's full-D SM90 forward."""

from __future__ import annotations

import argparse
import json
import os

import torch

from gemma4_fa4.h100 import (
    _load_flash_attn_func,
    fa4_global_forward_only,
    fa4_global_varlen_forward_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--packed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("EXP-0041 requires an SM90 H100")
    if args.seqlen < 1:
        raise ValueError("--seqlen must be positive")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    flag = "FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH"
    if args.packed:
        q_lengths = (0, 33, 65)
        k_lengths = (1, 65, 129)
        q = torch.randn(
            sum(q_lengths),
            32,
            512,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        k = torch.randn(
            sum(k_lengths),
            4,
            512,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        v = torch.randn(
            sum(k_lengths),
            4,
            512,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        cu_q = torch.tensor((0, 0, 33, 98), device="cuda", dtype=torch.int32)
        cu_k = torch.tensor((0, 1, 66, 195), device="cuda", dtype=torch.int32)
        kwargs = {
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
        with torch.no_grad():
            os.environ[flag] = "0"
            rollback_out, rollback_lse = fa4_global_varlen_forward_only(
                q, k, v, cu_q, cu_k, **kwargs
            )
            os.environ[flag] = "1"
            candidate_out, candidate_lse = fa4_global_varlen_forward_only(
                q, k, v, cu_q, cu_k, **kwargs
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(candidate_out, rollback_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(candidate_lse, rollback_lse, atol=0.0, rtol=0.0)
        print(
            json.dumps(
                {
                    "candidate_launches": 1,
                    "device": torch.cuda.get_device_name(),
                    "lse_bitwise": bool(torch.equal(candidate_lse, rollback_lse)),
                    "out_bitwise": bool(torch.equal(candidate_out, rollback_out)),
                    "packed": True,
                    "q_lengths": q_lengths,
                    "k_lengths": k_lengths,
                    "shape": list(candidate_out.shape),
                    "softmax_scale": 1.0,
                },
                sort_keys=True,
            )
        )
        return

    q = torch.randn(
        1, args.seqlen, 32, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        1, args.seqlen, 4, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    v = torch.randn(
        1, args.seqlen, 4, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    # The accepted EXP-0002 composition is the exact rollback oracle.
    with torch.no_grad():
        os.environ[flag] = "0"
        rollback_out, rollback_lse = fa4_global_forward_only(q, k, v)
        os.environ[flag] = "1"
        candidate_out, candidate_lse = _load_flash_attn_func()(
            q,
            k,
            v,
            causal=True,
            window_size=(None, None),
            softmax_scale=1.0,
            num_splits=1,
            pack_gqa=False,
            return_lse=True,
        )
    torch.cuda.synchronize()

    out_abs = (candidate_out.float() - rollback_out.float()).abs()
    lse_abs = (candidate_lse - rollback_lse).abs()
    torch.testing.assert_close(candidate_out, rollback_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_lse, rollback_lse, atol=0.0, rtol=0.0)
    print(
        json.dumps(
            {
                "candidate_launches": 1,
                "device": torch.cuda.get_device_name(),
                "lse_bitwise": bool(torch.equal(candidate_lse, rollback_lse)),
                "lse_max_abs": float(lse_abs.max().item()),
                "out_bitwise": bool(torch.equal(candidate_out, rollback_out)),
                "out_max_abs": float(out_abs.max().item()),
                "seqlen": args.seqlen,
                "shape": list(candidate_out.shape),
                "softmax_scale": 1.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH", "1")
    main()
