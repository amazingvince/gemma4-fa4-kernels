#!/usr/bin/env python3
"""Compile/correctness probe for exact H100 packed local d256 backward."""

from __future__ import annotations

import argparse
import os
from itertools import accumulate
from typing import Literal

import torch

from gemma4_fa4.h100 import fa4_local_varlen_forward
from gemma4_fa4.masks import gemma4_attention_mask
from gemma4_fa4.model_spec import GEMMA4_31B, SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention_varlen

GRAD_ATOL = 0.125
GRAD_RTOL = 0.05
UPSTREAM_ERROR_MULTIPLIER = 2.0

ComparisonPolicy = Literal["frozen", "upstream-relative"]
GradientSource = Literal["out", "lse", "out_lse"]
VisionPattern = Literal["none", "text", "mixed", "adjacent", "all"]
DocumentPattern = Literal["none", "single", "split"]


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the local varlen backward probe requires an SM90 CUDA device")


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
    if any(q_len > k_len for q_len, k_len in zip(q_lengths, k_lengths, strict=True)):
        raise ValueError("every packed sequence requires Sq <= Sk")
    if max(k_lengths) > GEMMA4_31B.max_position_embeddings:
        raise ValueError("every Sk must be <= the locked model maximum 262144")


def _estimate_probe_live_bytes(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    repeats: int = 1,
    reference: bool = False,
) -> int:
    """Conservatively model retained candidate and optional dense-oracle storage."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    q_bytes = sum(q_lengths) * SLIDING_ATTENTION.num_q_heads * 256 * 2
    kv_bytes = sum(k_lengths) * SLIDING_ATTENTION.num_kv_heads * 256 * 2
    lse_bytes = SLIDING_ATTENTION.num_q_heads * sum(q_lengths) * 4
    # Q, dO, O, dQ, and an FP32 dQ-sized allowance; K/V plus dK/dV;
    # FP32 dK/dV accumulators (each twice one BF16 K/V tensor); two
    # FP32 LSE, dLSE, dP-sum, and log2-LSE buffers; and 2 GiB for
    # allocator/JIT/kernel workspaces.
    required = 6 * q_bytes + 8 * kv_bytes + 4 * lse_bytes + 2 * 1024**3
    # The repeat checker retains O, LSE, dQ, dK, and dV from every run.
    required += (repeats - 1) * (2 * q_bytes + 2 * kv_bytes + lse_bytes)
    if reference:
        # The independent oracle is intentionally dense. Sixty-four bytes per
        # score element covers its FP32 scores/probabilities/mask/autograd
        # state (and the sequential BF16 policy baseline), plus retain the
        # reference gradient triplet while the baseline is evaluated.
        score_elements = sum(
            q_len * k_len * SLIDING_ATTENTION.num_q_heads
            for q_len, k_len in zip(q_lengths, k_lengths, strict=True)
        )
        required += 64 * score_elements + q_bytes + 2 * kv_bytes
    return required


def _preflight_cuda_memory(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    repeats: int = 1,
    reference: bool = False,
) -> None:
    """Reject a probe whose conservative live-set estimate exceeds free HBM."""

    required = _estimate_probe_live_bytes(
        q_lengths,
        k_lengths,
        repeats=repeats,
        reference=reference,
    )
    free, total = torch.cuda.mem_get_info()
    print(
        f"memory_preflight required={required} free={free} total={total} "
        f"fraction={required / free:.4f}"
    )
    if required > int(free * 0.9):
        raise RuntimeError(
            f"memory preflight requires {required} bytes but only {free} bytes are free"
        )


def _cumulative_tensor(
    lengths: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(
        (0, *accumulate(lengths)),
        device=device,
        dtype=torch.int32,
    )


def _cumulative_values(cumulative: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in cumulative.detach().cpu().tolist())


def _make_inputs(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    seed: int,
):
    _validate_lengths(q_lengths, k_lengths)
    spec = SLIDING_ATTENTION
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
    do = torch.randn(
        q_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dlse = torch.randn(
        (spec.num_q_heads, sum(q_lengths)),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cu_q = _cumulative_tensor(q_lengths, device="cuda")
    cu_k = _cumulative_tensor(k_lengths, device="cuda")
    return q, k, v, do, dlse, cu_q, cu_k


def _differentiate(out, lse, inputs, do, dlse, gradient_source: GradientSource):
    if gradient_source == "out":
        return torch.autograd.grad(out, inputs, do)
    if gradient_source == "lse":
        # Exercise the public autograd path where O has no incoming gradient.
        # FA4 must synthesize a zero dO before its backward preprocess.
        grads = torch.autograd.grad(lse, inputs, dlse, allow_unused=True)
        # Pure reference LSE is algebraically independent of V, while FA4's
        # custom autograd returns an explicit zero dV. Normalize that contract.
        return tuple(
            torch.zeros_like(tensor) if grad is None else grad
            for tensor, grad in zip(inputs, grads, strict=True)
        )
    return torch.autograd.grad((out, lse), inputs, (do, dlse))


def _run_candidate_with_outputs(
    q,
    k,
    v,
    do,
    cu_q,
    cu_k,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
    document_ids=None,
):
    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    out, lse = fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(
            end - start
            for start, end in zip(
                _cumulative_values(cu_q)[:-1], _cumulative_values(cu_q)[1:], strict=True
            )
        ),
        max_seqlen_k=max(
            end - start
            for start, end in zip(
                _cumulative_values(cu_k)[:-1], _cumulative_values(cu_k)[1:], strict=True
            )
        ),
        vision_block_ids=vision_block_ids,
        document_ids=document_ids,
    )
    grads = _differentiate(out, lse, (q, k, v), do, dlse, gradient_source)
    return out, lse, grads


def _run_candidate(
    q,
    k,
    v,
    do,
    cu_q,
    cu_k,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
    document_ids=None,
):
    return _run_candidate_with_outputs(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source=gradient_source,
        vision_block_ids=vision_block_ids,
        document_ids=document_ids,
    )[2]


def _run_reference(
    q,
    k,
    v,
    do,
    cu_q,
    cu_k,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
    document_ids=None,
):
    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    out_ref, lse_ref = reference_attention_varlen(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        softmax_scale=1.0,
        sliding_window=1024,
        vision_block_ids=vision_block_ids,
        document_ids=document_ids,
        allow_vision_bidirectional=vision_block_ids is not None,
        upcast=torch.float32,
        return_lse=True,
    )
    return _differentiate(
        out_ref,
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
    cu_q,
    cu_k,
    *,
    dlse=None,
    gradient_source: GradientSource = "out",
    vision_block_ids=None,
    document_ids=None,
):
    """Mirror pinned ``attention_ref(upcast=False, reorder_ops=True)`` at scale 1."""

    if gradient_source != "out" and dlse is None:
        raise ValueError("dlse is required when differentiating LSE")
    q_pt, k_pt, v_pt = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    q_values = _cumulative_values(cu_q)
    k_values = _cumulative_values(cu_k)
    if len(q_values) != len(k_values):
        raise ValueError("packed Q and K must have the same batch count")

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    gqa_ratio = q_pt.shape[1] // k_pt.shape[1]
    for q_start, q_end, k_start, k_end in zip(
        q_values[:-1],
        q_values[1:],
        k_values[:-1],
        k_values[1:],
        strict=True,
    ):
        q_i = q_pt[q_start:q_end].unsqueeze(0)
        k_i = k_pt[k_start:k_end].unsqueeze(0)
        v_i = v_pt[k_start:k_end].unsqueeze(0)
        k_expanded = torch.repeat_interleave(k_i, gqa_ratio, dim=2)
        v_expanded = torch.repeat_interleave(v_i, gqa_ratio, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q_i, k_expanded * 1.0)
        vision_ids_i = (
            vision_block_ids[k_start:k_end].unsqueeze(0) if vision_block_ids is not None else None
        )
        document_ids_i = (
            document_ids[k_start:k_end].unsqueeze(0) if document_ids is not None else None
        )
        allowed = gemma4_attention_mask(
            batch_size=1,
            q_len=q_end - q_start,
            kv_len=k_end - k_start,
            device=q_pt.device,
            sliding_window=1024,
            vision_block_ids=vision_ids_i,
            document_ids=document_ids_i,
            q_start=(k_end - k_start) - (q_end - q_start),
            allow_vision_bidirectional=vision_ids_i is not None,
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        lse_i = torch.logsumexp(scores.float(), dim=-1)
        probabilities = torch.softmax(scores, dim=-1).to(v_pt.dtype)
        out_i = torch.einsum("bhts,bshd->bthd", probabilities, v_expanded)
        outputs.append(out_i.squeeze(0))
        lses.append(lse_i.squeeze(0))

    out_pt = torch.cat(outputs, dim=0)
    lse_pt = torch.cat(lses, dim=1)
    return _differentiate(
        out_pt,
        lse_pt,
        (q_pt, k_pt, v_pt),
        do,
        dlse,
        gradient_source,
    )


def _make_vision_block_ids(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    pattern: VisionPattern,
    *,
    device: torch.device | str,
) -> torch.Tensor | None:
    if pattern == "none":
        return None
    packed: list[torch.Tensor] = []
    for q_len, k_len in zip(q_lengths, k_lengths, strict=True):
        ids = torch.full((k_len,), -1, device=device, dtype=torch.int64)
        if pattern == "all":
            ids.zero_()
        elif pattern in ("mixed", "adjacent"):
            anchor = k_len - q_len
            if pattern == "mixed":
                width = max(1, min(8, k_len - anchor))
                ids[anchor : anchor + width] = 0
                second_start = min(k_len, anchor + width + 1)
                second_end = min(k_len, second_start + max(1, min(5, q_len)))
                ids[second_start:second_end] = 1
            else:
                ids[anchor : min(k_len, anchor + 2)] = 0
                if anchor + 2 < k_len:
                    ids[anchor + 2] = 1
        packed.append(ids)
    return torch.cat(packed)


def _make_document_ids(
    k_lengths: tuple[int, ...],
    pattern: DocumentPattern,
    *,
    device: torch.device | str,
) -> torch.Tensor | None:
    if pattern == "none":
        return None
    packed: list[torch.Tensor] = []
    for k_len in k_lengths:
        ids = torch.zeros((k_len,), device=device, dtype=torch.int64)
        if pattern == "split" and k_len > 1:
            ids[max(1, k_len // 2) :] = 1
        packed.append(ids)
    return torch.cat(packed)


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
    for index, (name, grad, ref) in enumerate(zip(("dQ", "dK", "dV"), grads, refs, strict=True)):
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
        bf16_ref = bf16_refs[index]
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


def _assert_gradient_contract(
    grads,
    q,
    k,
    v,
    *,
    gradient_source: GradientSource,
    fake_mode: bool,
) -> None:
    for name, grad, tensor in zip(("dQ", "dK", "dV"), grads, (q, k, v), strict=True):
        if grad.shape != tensor.shape or grad.dtype != torch.bfloat16:
            raise AssertionError(f"invalid {name} contract: {grad.shape}, {grad.dtype}")
        if not fake_mode and not torch.isfinite(grad).all():
            raise AssertionError(f"{name} contains non-finite values")
    if fake_mode:
        return
    grad_storage = [grad.untyped_storage().data_ptr() for grad in grads]
    if len(set(grad_storage)) != 3:
        raise AssertionError("dQ, dK, and dV must use distinct storage")
    if gradient_source == "lse" and torch.count_nonzero(grads[2]).item() != 0:
        raise AssertionError("LSE-only differentiation must produce an exact-zero dV")


def _assert_structured_ownership(
    grads,
    *,
    q_index: int,
    q_head: int,
    allowed_key_indices: tuple[int, ...],
    same_block_future: int,
    gqa_ratio: int,
) -> None:
    dq, dk, dv = grads
    kv_head = q_head // gqa_ratio

    inactive_dq = dq.clone()
    inactive_dq[q_index, q_head] = 0
    if torch.count_nonzero(inactive_dq).item() != 0:
        raise AssertionError("inactive packed query positions or heads received dQ")

    for name, grad in (("dK", dk), ("dV", dv)):
        inactive_heads = grad.clone()
        inactive_heads[:, kv_head] = 0
        if torch.count_nonzero(inactive_heads).item() != 0:
            raise AssertionError(f"inactive packed GQA KV heads received {name}")
        inactive_keys = grad[:, kv_head].clone()
        inactive_keys[list(allowed_key_indices)] = 0
        if torch.count_nonzero(inactive_keys).item() != 0:
            raise AssertionError(f"masked or cross-sequence keys received {name}")
        if torch.count_nonzero(grad[same_block_future, kv_head]).item() == 0:
            raise AssertionError(f"same-block future key did not receive {name}")


def _run_structured_ownership(seed: int) -> None:
    q_lengths, k_lengths = (2, 3), (2, 5)
    q, k, v, random_do, _, cu_q, cu_k = _make_inputs(q_lengths, k_lengths, seed)
    do = torch.zeros_like(random_do)
    q_index, q_head = q_lengths[0], 9
    do[q_index, q_head] = random_do[q_index, q_head]
    vision_ids = torch.tensor(
        [-1, -1, -1, -1, 0, 0, 1],
        device="cuda",
        dtype=torch.int64,
    )
    document_ids = torch.tensor(
        [0, 0, 3, 3, 4, 4, 4],
        device="cuda",
        dtype=torch.int64,
    )

    grads = _run_candidate(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        vision_block_ids=vision_ids,
        document_ids=document_ids,
    )
    refs = _run_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        vision_block_ids=vision_ids,
        document_ids=document_ids,
    )
    bf16_refs = _run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        vision_block_ids=vision_ids,
        document_ids=document_ids,
    )
    query_key, same_block_future = 4, 5
    _assert_structured_ownership(
        grads,
        q_index=q_index,
        q_head=q_head,
        allowed_key_indices=(query_key, same_block_future),
        same_block_future=same_block_future,
        gqa_ratio=2,
    )
    failures = _check_gradients(
        grads,
        refs,
        policy="upstream-relative",
        bf16_refs=bf16_refs,
        run_label="structured sequence=1 qlocal=0 qabs=2 qhead=9",
    )
    if failures:
        raise AssertionError("\n\n".join(failures))
    print(
        "passed structured_ownership sequence=1 qlocal=0 qabs=2 "
        "qhead=9 kvhead=4 same_future=3 document_blocked=0,1"
    )


def _assert_isolated_prefix_equal(
    base,
    mutated,
    *,
    q_prefix: int,
    k_prefix: int,
    check_dq: bool = True,
) -> None:
    out, lse, grads = base
    out_mut, lse_mut, grads_mut = mutated
    comparisons = [
        ("O", out[:q_prefix], out_mut[:q_prefix]),
        ("LSE", lse[:, :q_prefix], lse_mut[:, :q_prefix]),
        ("dK", grads[1][:k_prefix], grads_mut[1][:k_prefix]),
        ("dV", grads[2][:k_prefix], grads_mut[2][:k_prefix]),
    ]
    if check_dq:
        comparisons.insert(2, ("dQ", grads[0][:q_prefix], grads_mut[0][:q_prefix]))
    for name, expected, actual in comparisons:
        if not torch.equal(expected, actual):
            difference = (expected.float() - actual.float()).abs().max().item()
            raise AssertionError(
                f"packed-boundary isolation failed for {name}: max_abs={difference}"
            )


def _run_isolation(seed: int) -> None:
    q_lengths = k_lengths = (33, 65)
    q, k, v, do, dlse, cu_q, cu_k = _make_inputs(q_lengths, k_lengths, seed)
    vision_ids = _make_vision_block_ids(q_lengths, k_lengths, "all", device="cuda")
    document_ids = _make_document_ids(k_lengths, "single", device="cuda")
    base = _run_candidate_with_outputs(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
        vision_block_ids=vision_ids,
        document_ids=document_ids,
    )

    q_mut = q.detach().clone().requires_grad_(True)
    k_mut = k.detach().clone()
    v_mut = v.detach().clone()
    k_prefix = k_lengths[0]
    with torch.no_grad():
        k_mut[k_prefix::2].fill_(32.0)
        k_mut[k_prefix + 1 :: 2].fill_(-32.0)
        v_mut[k_prefix::2].fill_(2048.0)
        v_mut[k_prefix + 1 :: 2].fill_(-2048.0)
    k_mut.requires_grad_()
    v_mut.requires_grad_()
    mutated = _run_candidate_with_outputs(
        q_mut,
        k_mut,
        v_mut,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
        vision_block_ids=vision_ids,
        document_ids=document_ids,
    )
    _assert_isolated_prefix_equal(
        base,
        mutated,
        q_prefix=q_lengths[0],
        k_prefix=k_prefix,
    )
    print(
        "passed packed_boundary_isolation lengths=33,65 repeated_vision=0 repeated_document=0 "
        "mutated_sequence=1 outputs=O,LSE,dQ,dK,dV"
    )


def _run_long_text_isolation(seed: int) -> None:
    q_lengths, k_lengths = (33, 65), (2049, 4097)
    q, k, v, do, dlse, cu_q, cu_k = _make_inputs(q_lengths, k_lengths, seed)
    base = _run_candidate_with_outputs(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    base_repeat = _run_candidate_with_outputs(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    base_refs = _run_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    base_bf16_refs = _run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )

    q_mut = q.detach().clone().requires_grad_(True)
    k_mut = k.detach().clone()
    v_mut = v.detach().clone()
    k_prefix = k_lengths[0]
    with torch.no_grad():
        k_mut[k_prefix::2].fill_(32.0)
        k_mut[k_prefix + 1 :: 2].fill_(-32.0)
        v_mut[k_prefix::2].fill_(2048.0)
        v_mut[k_prefix + 1 :: 2].fill_(-2048.0)
    k_mut.requires_grad_()
    v_mut.requires_grad_()
    mutated = _run_candidate_with_outputs(
        q_mut,
        k_mut,
        v_mut,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    mutated_refs = _run_reference(
        q_mut,
        k_mut,
        v_mut,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    mutated_bf16_refs = _run_upstream_style_bf16_baseline(
        q_mut,
        k_mut,
        v_mut,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source="out_lse",
    )
    _assert_isolated_prefix_equal(
        base,
        mutated,
        q_prefix=q_lengths[0],
        k_prefix=k_prefix,
        check_dq=False,
    )
    for name, reference, mutated_reference, prefix in zip(
        ("dQ", "dK", "dV"),
        base_refs,
        mutated_refs,
        (q_lengths[0], k_prefix, k_prefix),
        strict=True,
    ):
        if not torch.equal(reference[:prefix], mutated_reference[:prefix]):
            raise AssertionError(f"reference packed-boundary isolation failed for {name}")

    failures = _check_gradients(
        base[2],
        base_refs,
        policy="upstream-relative",
        bf16_refs=base_bf16_refs,
        run_label="long-isolation base",
    )
    failures.extend(
        _check_gradients(
            mutated[2],
            mutated_refs,
            policy="upstream-relative",
            bf16_refs=mutated_bf16_refs,
            run_label="long-isolation hostile-mutation",
        )
    )
    if failures:
        raise AssertionError("\n\n".join(failures))

    control_dq_drift = (
        (base[2][0][: q_lengths[0]].float() - base_repeat[2][0][: q_lengths[0]].float())
        .abs()
        .max()
        .item()
    )
    mutation_dq_delta = (
        (base[2][0][: q_lengths[0]].float() - mutated[2][0][: q_lengths[0]].float())
        .abs()
        .max()
        .item()
    )
    print(
        "passed long_text_packed_boundary_isolation "
        "q_lengths=33,65 k_lengths=2049,4097 mutated_sequence=1 "
        "exact=O,LSE,dK,dV reference_exact=dQ,dK,dV "
        f"control_dQ_drift={control_dq_drift:.8g} mutation_dQ_delta={mutation_dq_delta:.8g}"
    )


def _format_lengths(lengths: tuple[int, ...]) -> str:
    return ",".join(str(length) for length in lengths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-lengths", type=_parse_lengths, default=(128,))
    parser.add_argument(
        "--k-lengths",
        type=_parse_lengths,
        help="comma-separated K lengths; defaults to --q-lengths",
    )
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
    parser.add_argument(
        "--document-pattern",
        choices=("none", "single", "split"),
        default="none",
    )
    parser.add_argument("--structured-ownership", action="store_true")
    parser.add_argument("--isolation", action="store_true")
    parser.add_argument("--long-text-isolation", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--allow-nondeterministic-dq",
        action="store_true",
        help="allow only dQ to differ bitwise; establish numerical correctness separately",
    )
    parser.add_argument("--nondefault-stream", action="store_true")
    parser.add_argument("--seed", type=int, default=8008)
    args = parser.parse_args()
    k_lengths = args.q_lengths if args.k_lengths is None else args.k_lengths
    try:
        _validate_lengths(args.q_lengths, k_lengths)
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.comparison_policy != "frozen" and not args.reference:
        parser.error("--comparison-policy upstream-relative requires --reference")
    if args.allow_nondeterministic_dq and (
        not args.reference or args.comparison_policy != "upstream-relative"
    ):
        parser.error(
            "--allow-nondeterministic-dq requires --reference --comparison-policy upstream-relative"
        )
    if max(k_lengths) > 1025 and (args.vision_pattern != "none" or args.document_pattern != "none"):
        parser.error("metadata-bearing lengths above 1025 require the later sparse schedule")

    fake_mode = os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1"
    if fake_mode and args.reference:
        parser.error("--reference cannot run in fake-tensor mode")
    if fake_mode and (args.structured_ownership or args.isolation or args.long_text_isolation):
        parser.error("structured ownership and isolation require real execution")
    _require_h100()
    if not fake_mode:
        _preflight_cuda_memory(
            args.q_lengths,
            k_lengths,
            repeats=args.repeats,
            reference=args.reference,
        )

    q, k, v, do, dlse, cu_q, cu_k = _make_inputs(args.q_lengths, k_lengths, args.seed)
    vision_ids = _make_vision_block_ids(
        args.q_lengths,
        k_lengths,
        args.vision_pattern,
        device=q.device,
    )
    document_ids = _make_document_ids(
        k_lengths,
        args.document_pattern,
        device=q.device,
    )
    stream = torch.cuda.Stream() if args.nondefault_stream else None
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())

    candidate_runs = []
    for _ in range(args.repeats):
        if stream is None:
            out, lse, grads = _run_candidate_with_outputs(
                q,
                k,
                v,
                do,
                cu_q,
                cu_k,
                dlse=dlse,
                gradient_source=args.gradient_source,
                vision_block_ids=vision_ids,
                document_ids=document_ids,
            )
        else:
            with torch.cuda.stream(stream):
                out, lse, grads = _run_candidate_with_outputs(
                    q,
                    k,
                    v,
                    do,
                    cu_q,
                    cu_k,
                    dlse=dlse,
                    gradient_source=args.gradient_source,
                    vision_block_ids=vision_ids,
                    document_ids=document_ids,
                )
                _ = out.float().sum() + lse.sum() + sum(grad.float().sum() for grad in grads)
            stream.synchronize()
        _assert_gradient_contract(
            grads,
            q,
            k,
            v,
            gradient_source=args.gradient_source,
            fake_mode=fake_mode,
        )
        candidate_runs.append((out, lse, grads))

    label = (
        f"q_lengths={_format_lengths(args.q_lengths)} "
        f"k_lengths={_format_lengths(k_lengths)} vision={args.vision_pattern} "
        f"documents={args.document_pattern} source={args.gradient_source}"
    )
    if args.reference:
        refs = _run_reference(
            q,
            k,
            v,
            do,
            cu_q,
            cu_k,
            dlse=dlse,
            gradient_source=args.gradient_source,
            vision_block_ids=vision_ids,
            document_ids=document_ids,
        )
        bf16_refs = (
            _run_upstream_style_bf16_baseline(
                q,
                k,
                v,
                do,
                cu_q,
                cu_k,
                dlse=dlse,
                gradient_source=args.gradient_source,
                vision_block_ids=vision_ids,
                document_ids=document_ids,
            )
            if args.comparison_policy == "upstream-relative"
            else None
        )
        failures = []
        for run_idx, (_, _, grads) in enumerate(candidate_runs):
            failures.extend(
                _check_gradients(
                    grads,
                    refs,
                    policy=args.comparison_policy,
                    bf16_refs=bf16_refs,
                    run_label=f"{label} run={run_idx}",
                )
            )
        if failures:
            raise AssertionError("\n\n".join(failures))
    else:
        grads = candidate_runs[-1][2]
        print(
            "compiled "
            + label
            + " "
            + " ".join(
                f"{name}={tuple(grad.shape)}"
                for name, grad in zip(("dQ", "dK", "dV"), grads, strict=True)
            )
        )

    if args.repeats > 1:
        first_out, first_lse, first_grads = candidate_runs[0]
        equality = {
            "O": all(torch.equal(first_out, other[0]) for other in candidate_runs[1:]),
            "LSE": all(torch.equal(first_lse, other[1]) for other in candidate_runs[1:]),
        }
        equality.update(
            {
                name: all(torch.equal(base, other[2][index]) for other in candidate_runs[1:])
                for index, (name, base) in enumerate(
                    zip(("dQ", "dK", "dV"), first_grads, strict=True)
                )
            }
        )
        print("repeat_exact " + " ".join(f"{name}={value}" for name, value in equality.items()))
        required_exact = {
            name: exact
            for name, exact in equality.items()
            if name != "dQ" or not args.allow_nondeterministic_dq
        }
        if not all(required_exact.values()):
            raise AssertionError("same-input packed output/gradient repeats were not bitwise equal")
        if args.allow_nondeterministic_dq and not equality["dQ"]:
            dq_max_abs = max(
                (first_grads[0].float() - other[2][0].float()).abs().max().item()
                for other in candidate_runs[1:]
            )
            print(f"repeat_nondeterministic_dQ max_pairwise_abs={dq_max_abs:.8g}")

    if args.structured_ownership:
        _run_structured_ownership(seed=args.seed + 1)
    if args.isolation:
        _run_isolation(seed=args.seed + 2)
    if args.long_text_isolation:
        _run_long_text_isolation(seed=args.seed + 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
