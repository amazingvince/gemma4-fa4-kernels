#!/usr/bin/env python3
"""Semantics-aware Gemma 4 attention microbenchmark.

This is a research ruler, not a dispatcher. It refuses to label a baseline as
comparable when that implementation cannot express the requested mask.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_fa4.h100 import (  # noqa: E402
    fa4_global_forward_only,
    fa4_global_text_forward,
    fa4_global_varlen_forward,
    fa4_local_text_forward,
    fa4_local_varlen_forward,
)
from gemma4_fa4.model_spec import (  # noqa: E402
    GLOBAL_ATTENTION,
    SLIDING_ATTENTION,
    AttentionLayerSpec,
)
from gemma4_fa4.reference import attention_flops, make_qkv, reference_layer  # noqa: E402


@dataclass(frozen=True)
class BenchCase:
    name: str
    spec: AttentionLayerSpec
    batch: int
    seqlen: int
    mode: str


class UnsupportedSemanticBaseline(RuntimeError):
    pass


def _load_ladder(name: str) -> list[BenchCase]:
    data = json.loads((ROOT / f"configs/benchmarks/{name}.json").read_text())
    cases = []
    for item in data["entries"]:
        base_spec = SLIDING_ATTENTION if item["layer"] == "sliding_attention" else GLOBAL_ATTENTION
        spec = replace(
            base_spec,
            num_q_heads=item.get("q_heads", base_spec.num_q_heads),
            num_kv_heads=item.get("kv_heads", base_spec.num_kv_heads),
        )
        if spec.num_q_heads % spec.num_kv_heads:
            raise ValueError(f"invalid GQA geometry in {item['name']}: {spec}")
        cases.append(BenchCase(item["name"], spec, item["batch"], item["seqlen"], item["mode"]))
    return cases


def _packed_bshd(tensor: torch.Tensor) -> torch.Tensor:
    bshd = tensor.transpose(1, 2)
    if tensor.shape[0] == 1:
        return bshd.squeeze(0)
    return bshd.reshape(-1, tensor.shape[1], tensor.shape[3])


def _unpack_thd(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return output.reshape(
        source.shape[0], source.shape[2], source.shape[1], source.shape[3]
    ).transpose(1, 2)


def _fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
    *,
    cu_seqlens: torch.Tensor | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Run the exact accepted project FA4 route and return BHSD output."""

    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)
    requires_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )

    if spec.sliding_window is not None:
        if deterministic:
            raise UnsupportedSemanticBaseline(
                "deterministic benchmarking is implemented only for global FA4 backward"
            )
        if q.shape[0] == 1 and q.shape[2] <= 1025:
            output, _lse = fa4_local_text_forward(q_bshd, k_bshd, v_bshd, spec=spec)
            return output.transpose(1, 2)
        if cu_seqlens is None:
            raise ValueError("long or batched local FA4 benchmarking requires cu_seqlens")
        output, _lse = fa4_local_varlen_forward(
            _packed_bshd(q),
            _packed_bshd(k),
            _packed_bshd(v),
            cu_seqlens,
            cu_seqlens,
            max_seqlen_q=q.shape[2],
            max_seqlen_k=k.shape[2],
            spec=spec,
        )
        return _unpack_thd(output, q)

    if not requires_backward:
        output, _lse = fa4_global_forward_only(q_bshd, k_bshd, v_bshd, spec=spec)
        return output.transpose(1, 2)
    if q.shape[0] == 1 and q.shape[2] <= 2048:
        output, _lse = fa4_global_text_forward(
            q_bshd,
            k_bshd,
            v_bshd,
            spec=spec,
            deterministic=deterministic,
        )
        return output.transpose(1, 2)
    if cu_seqlens is None:
        raise ValueError("long or batched global FA4 backward benchmarking requires cu_seqlens")
    output, _lse = fa4_global_varlen_forward(
        _packed_bshd(q),
        _packed_bshd(k),
        _packed_bshd(v),
        cu_seqlens,
        cu_seqlens,
        max_seqlen_q=q.shape[2],
        max_seqlen_k=k.shape[2],
        spec=spec,
        deterministic=deterministic,
    )
    return _unpack_thd(output, q)


def _sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: AttentionLayerSpec
) -> torch.Tensor:
    if spec.sliding_window is not None:
        raise UnsupportedSemanticBaseline(
            "plain SDPA full-causal attention is not a sliding-window baseline; use FA4/FlexAttention"
        )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        scale=1.0,
        enable_gqa=spec.qhead_per_kvhead > 1,
    )


def _sdpa_expanded(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: AttentionLayerSpec
) -> torch.Tensor:
    """Run a semantically equivalent composite after explicit KV-head expansion.

    Expansion remains inside the measured operation. During backward, autograd
    reduces the repeated-head contributions into the original distinct K and V
    tensors, preserving the public GQA gradient contract.
    """

    if spec.sliding_window is not None:
        raise UnsupportedSemanticBaseline(
            "plain SDPA full-causal attention is not a sliding-window baseline; use FA4/FlexAttention"
        )
    repeats = spec.qhead_per_kvhead
    k_expanded = torch.repeat_interleave(k, repeats, dim=1)
    v_expanded = torch.repeat_interleave(v, repeats, dim=1)
    return F.scaled_dot_product_attention(
        q,
        k_expanded,
        v_expanded,
        is_causal=True,
        scale=1.0,
    )


def _run(
    impl: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
    *,
    cu_seqlens: torch.Tensor | None = None,
    deterministic: bool = False,
):
    if impl == "fa4":
        return _fa4(
            q,
            k,
            v,
            spec,
            cu_seqlens=cu_seqlens,
            deterministic=deterministic,
        )
    if deterministic:
        raise UnsupportedSemanticBaseline(
            "--deterministic selects the global FA4 backward route and cannot label another implementation"
        )
    if impl == "sdpa":
        return _sdpa(q, k, v, spec)
    if impl == "sdpa_expanded":
        return _sdpa_expanded(q, k, v, spec)
    if impl == "reference":
        if q.shape[2] > 2048:
            raise UnsupportedSemanticBaseline("dense reference is restricted to seqlen <= 2048")
        return reference_layer(spec, q, k, v)
    raise ValueError(impl)


def _estimated_bytes(case: BenchCase, dtype_bytes: int) -> int:
    s, b, spec = case.seqlen, case.batch, case.spec
    q = b * s * spec.num_q_heads * spec.head_dim_qk * dtype_bytes
    k = b * s * spec.num_kv_heads * spec.head_dim_qk * dtype_bytes
    v = b * s * spec.num_kv_heads * spec.head_dim_v * dtype_bytes
    o = b * s * spec.num_q_heads * spec.head_dim_v * dtype_bytes
    if case.mode == "fwd":
        return q + k + v + o
    # Inputs, output, upstream grad, three grads, and conservative workspace headroom.
    return 3 * (q + k + v + o)


def _implementation_extra_bytes(case: BenchCase, dtype_bytes: int, impl: str) -> int:
    if impl != "sdpa_expanded":
        return 0
    spec = case.spec
    expanded_kv = (
        case.batch
        * case.seqlen
        * spec.num_q_heads
        * (spec.head_dim_qk + spec.head_dim_v)
        * dtype_bytes
    )
    # Backward retains expanded K/V and materializes their gradients before
    # repeat_interleave reduces them into the original four-head operands.
    return expanded_kv if case.mode == "fwd" else 2 * expanded_kv


def _memory_plan(
    case: BenchCase,
    dtype_bytes: int,
    l2_mode: str,
    l2_bytes: int,
    *,
    impl: str = "fa4",
) -> tuple[int, int]:
    if l2_mode == "hot":
        thrash_bytes = 0
    elif l2_mode == "cold":
        thrash_bytes = max(4 * l2_bytes, 64 << 20)
    else:
        raise ValueError(l2_mode)
    return (
        _estimated_bytes(case, dtype_bytes)
        + _implementation_extra_bytes(case, dtype_bytes, impl)
        + thrash_bytes,
        thrash_bytes,
    )


def _make_grad_out(
    case: BenchCase, dtype: torch.dtype, device: torch.device | str = "cuda"
) -> torch.Tensor | None:
    if case.mode == "fwd":
        return None
    return torch.randn(
        case.batch,
        case.spec.num_q_heads,
        case.seqlen,
        case.spec.head_dim_v,
        dtype=dtype,
        device=device,
    )


def _mask_semantics(spec: AttentionLayerSpec) -> str:
    return "local_text_causal_window" if spec.sliding_window is not None else "global_causal"


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    values = sorted(values)
    if len(values) < 4:
        median = statistics.median(values)
        return median, median, median
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return cuts[0], statistics.median(values), cuts[2]


def _time_case(
    case: BenchCase,
    *,
    impl: str,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
    l2_mode: str,
    max_memory_fraction: float,
    deterministic: bool = False,
) -> dict:
    free, _ = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    l2_bytes = int(getattr(props, "L2_cache_size", 64 << 20))
    estimate, thrash_bytes = _memory_plan(
        case,
        torch.tensor([], dtype=dtype).element_size(),
        l2_mode,
        l2_bytes,
        impl=impl,
    )
    if estimate > free * max_memory_fraction:
        return {
            "name": case.name,
            "status": "skipped_memory",
            "estimated_bytes": estimate,
            "thrash_bytes": thrash_bytes,
            "free_bytes": free,
            "l2_mode": l2_mode,
        }
    requires_grad = case.mode != "fwd"
    q, k, v = make_qkv(
        case.spec,
        batch=case.batch,
        seqlen=case.seqlen,
        dtype=dtype,
        device="cuda",
        requires_grad=requires_grad,
    )
    cu_seqlens = torch.arange(
        0,
        (case.batch + 1) * case.seqlen,
        case.seqlen,
        dtype=torch.int32,
        device="cuda",
    )
    grad_out = _make_grad_out(case, dtype)
    thrash = torch.zeros(thrash_bytes, dtype=torch.uint8, device="cuda") if thrash_bytes else None

    def evict():
        if thrash is not None:
            thrash.add_(1)

    def one() -> float:
        nonlocal q, k, v
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if case.mode == "fwd":
            evict()
            start.record()
            _run(
                impl,
                q,
                k,
                v,
                case.spec,
                cu_seqlens=cu_seqlens,
                deterministic=deterministic,
            )
            end.record()
        elif case.mode == "bwd":
            qi = q.detach().requires_grad_(True)
            ki = k.detach().requires_grad_(True)
            vi = v.detach().requires_grad_(True)
            out = _run(
                impl,
                qi,
                ki,
                vi,
                case.spec,
                cu_seqlens=cu_seqlens,
                deterministic=deterministic,
            )
            evict()
            start.record()
            assert grad_out is not None
            torch.autograd.backward(out, grad_out)
            end.record()
        elif case.mode == "fwd_bwd":
            qi = q.detach().requires_grad_(True)
            ki = k.detach().requires_grad_(True)
            vi = v.detach().requires_grad_(True)
            evict()
            start.record()
            out = _run(
                impl,
                qi,
                ki,
                vi,
                case.spec,
                cu_seqlens=cu_seqlens,
                deterministic=deterministic,
            )
            assert grad_out is not None
            torch.autograd.backward(out, grad_out)
            end.record()
        else:
            raise ValueError(case.mode)
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        one()
    samples = [one() for _ in range(reps)]
    q1, median, q3 = _quartiles(samples)
    flops = attention_flops(case.spec, batch=case.batch, seqlen=case.seqlen, mode=case.mode)
    return {
        "name": case.name,
        "status": "ok",
        "layer": case.spec.kind,
        "batch": case.batch,
        "seqlen": case.seqlen,
        "mode": case.mode,
        "impl": impl,
        "deterministic": deterministic,
        "dtype": str(dtype).removeprefix("torch."),
        "softmax_scale": case.spec.softmax_scale,
        "q_heads": case.spec.num_q_heads,
        "kv_heads": case.spec.num_kv_heads,
        "head_dim": case.spec.head_dim_qk,
        "window": case.spec.sliding_window,
        "mask_semantics": _mask_semantics(case.spec),
        "l2_mode": l2_mode,
        "thrash_bytes": thrash_bytes,
        "q1_ms": q1,
        "median_ms": median,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "tflops": flops / (median * 1e-3) / 1e12,
        "estimated_bytes": estimate,
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", default="smoke")
    parser.add_argument(
        "--impl",
        choices=["fa4", "sdpa", "sdpa_expanded", "reference"],
        default="fa4",
    )
    parser.add_argument("--mode", choices=["fwd", "bwd", "fwd_bwd"])
    parser.add_argument("--only")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--l2", choices=["hot", "cold"], default="hot")
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="select the opt-in deterministic global FA4 backward route",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cases = _load_ladder(args.ladder)
    if args.only:
        cases = [case for case in cases if case.name == args.only]
    if args.mode:
        cases = [replace(case, mode=args.mode) for case in cases]
    results = []
    for case in cases:
        try:
            result = _time_case(
                case,
                impl=args.impl,
                dtype=dtype,
                warmup=args.warmup,
                reps=args.reps,
                l2_mode=args.l2,
                max_memory_fraction=args.max_memory_fraction,
                deterministic=args.deterministic,
            )
        except UnsupportedSemanticBaseline as exc:
            result = {"name": case.name, "status": "unsupported_semantics", "reason": str(exc)}
        except (RuntimeError, AssertionError) as exc:
            result = {"name": case.name, "status": "error", "reason": str(exc)}
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    return 1 if any(row["status"] == "error" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
