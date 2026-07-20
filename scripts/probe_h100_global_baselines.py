#!/usr/bin/env python3
"""Numerically admit semantically equivalent global H100 benchmark baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

BENCH_SPEC = importlib.util.spec_from_file_location(
    "gemma4_bench_attention", ROOT / "benchmarks/bench_attention.py"
)
assert BENCH_SPEC is not None and BENCH_SPEC.loader is not None
BENCH = importlib.util.module_from_spec(BENCH_SPEC)
sys.modules[BENCH_SPEC.name] = BENCH
BENCH_SPEC.loader.exec_module(BENCH)

from scripts.probe_h100_global_backward import (  # noqa: E402
    OUT_ATOL,
    OUT_RTOL,
    _check_gradient_policy,
    _make_inputs,
    _run_bf16_reference,
    _run_fp32_reference,
)


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the global baseline probe requires an SM90 CUDA device")


def _run_candidate(impl: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do):
    q_bhsd, k_bhsd, v_bhsd = (tensor.transpose(1, 2) for tensor in (q, k, v))
    do_bhsd = do.transpose(1, 2)
    out_bhsd = BENCH._run(impl, q_bhsd, k_bhsd, v_bhsd, BENCH.GLOBAL_ATTENTION)
    grads_bhsd = torch.autograd.grad(out_bhsd, (q_bhsd, k_bhsd, v_bhsd), do_bhsd)
    return out_bhsd.transpose(1, 2), tuple(grad.transpose(1, 2) for grad in grads_bhsd)


def _admit(impl: str, seqlen: int, seed: int) -> dict:
    q, k, v, do, _dlse = _make_inputs(seqlen, seed=seed)
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise AssertionError("global K and V must use distinct storage")
    out_ref, _lse_ref, grad_refs = _run_fp32_reference(q, k, v, do)
    bf16_out, bf16_grad_refs = _run_bf16_reference(q, k, v, do)
    out, grads = _run_candidate(impl, q, k, v, do)

    torch.testing.assert_close(
        out,
        out_ref,
        atol=OUT_ATOL,
        rtol=OUT_RTOL,
        msg=f"{impl} O exceeds the frozen global envelope",
    )
    failures = _check_gradient_policy(
        grads,
        grad_refs,
        bf16_grad_refs,
        run_label=impl,
    )
    if failures:
        raise AssertionError("\n".join(failures))
    if len({grad.untyped_storage().data_ptr() for grad in grads}) != 3:
        raise AssertionError(f"{impl} did not return distinct dQ/dK/dV storage")

    out_error = (out.float() - out_ref.float()).abs()
    bf16_error = (bf16_out.float() - out_ref.float()).abs()
    result = {
        "status": "ok",
        "impl": impl,
        "device": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "dtype": "bfloat16",
        "seqlen": seqlen,
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 512,
        "softmax_scale": 1.0,
        "mask_semantics": "global_causal",
        "max_abs_o": out_error.max().item(),
        "mean_abs_o": out_error.mean().item(),
        "bf16_baseline_max_abs_o": bf16_error.max().item(),
        "max_abs_gradients": {
            name: (grad.float() - ref.float()).abs().max().item()
            for name, grad, ref in zip(("dQ", "dK", "dV"), grads, grad_refs, strict=True)
        },
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        action="append",
        choices=("fa4", "sdpa", "sdpa_expanded"),
        help="implementation to admit; repeat for multiple implementations",
    )
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seed", type=int, default=34001)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    _require_h100()
    implementations = args.impl or ["fa4", "sdpa", "sdpa_expanded"]
    results = [_admit(impl, args.seqlen, args.seed) for impl in implementations]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
