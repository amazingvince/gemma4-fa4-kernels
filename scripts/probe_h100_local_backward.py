#!/usr/bin/env python3
"""Compile/correctness probe for exact H100 local d256 backward."""

from __future__ import annotations

import argparse
import os
from typing import Literal

import torch

from gemma4_fa4.h100 import fa4_local_forward, fa4_local_text_forward
from gemma4_fa4.masks import gemma4_attention_mask
from gemma4_fa4.model_spec import SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention

GRAD_ATOL = 0.125
GRAD_RTOL = 0.05
UPSTREAM_ERROR_MULTIPLIER = 2.0

ComparisonPolicy = Literal["frozen", "upstream-relative"]
GradientSource = Literal["out", "lse", "out_lse"]
VisionPattern = Literal["none", "text", "mixed", "adjacent", "all"]


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
    dlse = torch.randn(
        (1, spec.num_q_heads, seqlen),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return q, k, v, do, dlse


def _differentiate(out, lse, inputs, do, dlse, gradient_source: GradientSource):
    if gradient_source == "out":
        return torch.autograd.grad(out, inputs, do)
    if gradient_source == "lse":
        return torch.autograd.grad((out, lse), inputs, (torch.zeros_like(out), dlse))
    return torch.autograd.grad((out, lse), inputs, (do, dlse))


def _run_candidate(
    q,
    k,
    v,
    do,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
):
    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    if vision_block_ids is None:
        out, lse = fa4_local_text_forward(q, k, v)
    else:
        out, lse = fa4_local_forward(q, k, v, vision_block_ids=vision_block_ids)
    return _differentiate(out, lse, (q, k, v), do, dlse, gradient_source)


def _run_reference(
    q,
    k,
    v,
    do,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
):
    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    out_ref, lse_ref = reference_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        softmax_scale=1.0,
        sliding_window=1024,
        vision_block_ids=vision_block_ids,
        allow_vision_bidirectional=vision_block_ids is not None,
        upcast=torch.float32,
        return_lse=True,
    )
    return _differentiate(
        out_ref.transpose(1, 2),
        lse_ref,
        (q_ref, k_ref, v_ref),
        do,
        dlse,
        gradient_source,
    )


def _run_upstream_style_bf16_baseline(
    q,
    k,
    v,
    do,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
):
    """Mirror pinned ``attention_ref(upcast=False, reorder_ops=True)`` at scale 1."""

    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
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
        vision_block_ids=vision_block_ids,
        allow_vision_bidirectional=vision_block_ids is not None,
    )
    scores.masked_fill_(~allowed, float("-inf"))
    lse_pt = torch.logsumexp(scores.float(), dim=-1)
    probabilities = torch.softmax(scores, dim=-1).to(v_pt.dtype)
    out_pt = torch.einsum("bhts,bshd->bthd", probabilities, v_expanded)
    return _differentiate(
        out_pt,
        lse_pt,
        (q_pt, k_pt, v_pt),
        do,
        dlse,
        gradient_source,
    )


def _make_vision_block_ids(
    seqlen: int,
    pattern: VisionPattern,
    *,
    device: torch.device | str,
) -> torch.Tensor | None:
    if pattern == "none":
        return None
    ids = torch.full((1, seqlen), -1, device=device, dtype=torch.int64)
    if pattern == "text":
        return ids
    if pattern == "all":
        ids.zero_()
        return ids
    if pattern == "mixed":
        first_start = max(1, seqlen // 5)
        first_end = min(seqlen, first_start + max(2, min(12, seqlen // 4)))
        second_start = min(seqlen, first_end + max(1, seqlen // 8))
        second_end = min(seqlen, second_start + max(2, min(9, seqlen // 5)))
        ids[:, first_start:first_end] = 0
        ids[:, second_start:second_end] = 1
        return ids
    boundary = 64 if seqlen >= 72 else max(1, seqlen // 2)
    ids[:, max(0, boundary - 4) : min(seqlen, boundary + 1)] = 0
    ids[:, min(seqlen, boundary + 1) : min(seqlen, boundary + 8)] = 1
    return ids


def _assert_structured_ownership(
    grads,
    *,
    q_index: int,
    q_head: int,
    same_block_future: int,
    different_block_future: int,
    gqa_ratio: int,
) -> None:
    dq, dk, dv = grads
    kv_head = q_head // gqa_ratio

    inactive_dq = dq.clone()
    inactive_dq[:, q_index, q_head] = 0
    if torch.count_nonzero(inactive_dq).item() != 0:
        raise AssertionError("inactive query positions or heads received dQ")

    for name, grad in (("dK", dk), ("dV", dv)):
        inactive_heads = grad.clone()
        inactive_heads[:, :, kv_head] = 0
        if torch.count_nonzero(inactive_heads).item() != 0:
            raise AssertionError(f"inactive GQA KV heads received {name}")
        if torch.count_nonzero(grad[:, same_block_future, kv_head]).item() == 0:
            raise AssertionError(f"same-block future key did not receive {name}")
        if torch.count_nonzero(grad[:, different_block_future, kv_head]).item() != 0:
            raise AssertionError(f"different-block future key received {name}")


def _run_structured_ownership(seed: int) -> None:
    seqlen = 129
    q_index, q_head = 63, 9
    same_block_future, different_block_future = 64, 65
    q, k, v, random_do, _ = _make_inputs(seqlen, seed)
    do = torch.zeros_like(random_do)
    do[:, q_index, q_head] = random_do[:, q_index, q_head]
    vision_ids = torch.full((1, seqlen), -1, device="cuda", dtype=torch.int64)
    vision_ids[:, q_index : same_block_future + 1] = 0
    vision_ids[:, different_block_future] = 1

    grads = _run_candidate(q, k, v, do, vision_block_ids=vision_ids)
    refs = _run_reference(q, k, v, do, vision_block_ids=vision_ids)
    bf16_refs = _run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        vision_block_ids=vision_ids,
    )
    _assert_structured_ownership(
        grads,
        q_index=q_index,
        q_head=q_head,
        same_block_future=same_block_future,
        different_block_future=different_block_future,
        gqa_ratio=2,
    )
    failures = _check_gradients(
        grads,
        refs,
        policy="upstream-relative",
        bf16_refs=bf16_refs,
        run_label="structured q=63 qhead=9",
    )
    if failures:
        raise AssertionError("\n\n".join(failures))
    print("passed structured_ownership q=63 qhead=9 kvhead=4 same_future=64 different_future=65")


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
    parser.add_argument(
        "--gradient-source",
        choices=("out", "lse", "out_lse"),
        default="out",
        help="differentiate O, LSE, or both outputs",
    )
    parser.add_argument(
        "--vision-pattern",
        choices=("none", "text", "mixed", "adjacent", "all"),
        default="none",
    )
    parser.add_argument("--structured-ownership", action="store_true")
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
    q, k, v, do, dlse = _make_inputs(args.seqlen, seed=3000 + args.seqlen)
    vision_ids = _make_vision_block_ids(
        args.seqlen,
        args.vision_pattern,
        device=q.device,
    )
    stream = torch.cuda.Stream() if args.nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    candidate_runs = []
    for _ in range(args.repeats):
        if stream is None:
            grads = _run_candidate(
                q,
                k,
                v,
                do,
                dlse=dlse,
                gradient_source=args.gradient_source,
                vision_block_ids=vision_ids,
            )
        else:
            with torch.cuda.stream(stream):
                grads = _run_candidate(
                    q,
                    k,
                    v,
                    do,
                    dlse=dlse,
                    gradient_source=args.gradient_source,
                    vision_block_ids=vision_ids,
                )
                _ = sum(grad.float().sum() for grad in grads)
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
        refs = _run_reference(
            q,
            k,
            v,
            do,
            dlse=dlse,
            gradient_source=args.gradient_source,
            vision_block_ids=vision_ids,
        )
        bf16_refs = (
            _run_upstream_style_bf16_baseline(
                q,
                k,
                v,
                do,
                dlse=dlse,
                gradient_source=args.gradient_source,
                vision_block_ids=vision_ids,
            )
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
                    run_label=(
                        f"seqlen={args.seqlen} vision={args.vision_pattern} "
                        f"source={args.gradient_source} run={run_idx}"
                    ),
                )
            )
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
    if args.repeats > 1:
        first = candidate_runs[0]
        equality = {
            name: all(torch.equal(base, other[idx]) for other in candidate_runs[1:])
            for idx, (name, base) in enumerate(zip(("dQ", "dK", "dV"), first, strict=True))
        }
        print("repeat_exact " + " ".join(f"{name}={value}" for name, value in equality.items()))
        if not all(equality.values()):
            raise AssertionError("same-input gradient repeats were not bitwise equal")
    if args.structured_ownership:
        if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1":
            parser.error("--structured-ownership cannot run in fake-tensor mode")
        _run_structured_ownership(seed=9007)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
