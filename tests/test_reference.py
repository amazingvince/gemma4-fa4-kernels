import math

import pytest
import torch

from gemma4_fa4.model_spec import AttentionLayerSpec
from gemma4_fa4.reference import (
    attention_flops,
    make_qkv,
    prepare_global_kv_from_projection_source,
    reference_attention,
    reference_attention_varlen,
)


def tiny_spec(kind="sliding_attention", q_heads=4, kv_heads=2, window=4):
    return AttentionLayerSpec(
        kind=kind,
        head_dim_qk=8,
        head_dim_v=8,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        softmax_scale=1.0,
        is_causal=True,
        vision_bidirectional_within_block=kind == "sliding_attention",
        sliding_window=window if kind == "sliding_attention" else None,
        rope_type="default",
        rope_theta=10_000.0,
        partial_rotary_factor=1.0,
        projection_source_shared_between_k_and_v=False,
    )


@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    [(8, 8), (8, 4), (8, 2), (8, 1), (32, 16), (32, 4)],
)
def test_gqa_matches_materialized_mha_forward_and_backward(q_heads, kv_heads):
    generator = torch.Generator().manual_seed(17 + q_heads + kv_heads)
    q = torch.randn(1, q_heads, 5, 8, dtype=torch.float64, generator=generator)
    k = torch.randn(1, kv_heads, 5, 8, dtype=torch.float64, generator=generator)
    v = torch.randn(1, kv_heads, 5, 8, dtype=torch.float64, generator=generator)
    grouped = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    materialized = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)]
    ratio = q_heads // kv_heads

    grouped_out = reference_attention(*grouped)
    materialized_out = reference_attention(
        materialized[0],
        materialized[1].repeat_interleave(ratio, dim=1),
        materialized[2].repeat_interleave(ratio, dim=1),
    )
    torch.testing.assert_close(grouped_out, materialized_out)

    grad_out = torch.linspace(-0.75, 1.25, grouped_out.numel(), dtype=torch.float64).reshape_as(
        grouped_out
    )
    grouped_out.backward(grad_out)
    materialized_out.backward(grad_out)
    for grouped_tensor, materialized_tensor in zip(grouped, materialized, strict=True):
        torch.testing.assert_close(grouped_tensor.grad, materialized_tensor.grad)


def test_window_one_returns_current_v_per_query_head():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = reference_attention(q, k, v, sliding_window=1)
    torch.testing.assert_close(out, v, atol=1e-6, rtol=1e-6)


def test_gemma_scale_is_not_generic_scaled_dot_product():
    q = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    gemma = reference_attention(q, k, v, softmax_scale=1.0)
    generic = reference_attention(q, k, v, softmax_scale=1 / math.sqrt(8))
    assert not torch.allclose(gemma, generic)


def test_returns_fp32_lse():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out, lse = reference_attention(q, k, v, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (1, 2, 4)
    assert lse.dtype == torch.float32


def test_bf16_inputs_return_bf16_output_and_fp32_lse():
    q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
    out, lse = reference_attention(q, k, v, return_lse=True)
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 2, 4)
    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


def test_packed_varlen_reference_matches_individual_lower_right_sequences():
    generator = torch.Generator().manual_seed(81)
    q = torch.randn(3, 2, 4, dtype=torch.float64, generator=generator)
    k = torch.randn(5, 1, 4, dtype=torch.float64, generator=generator)
    v = torch.randn(5, 1, 4, dtype=torch.float64, generator=generator)
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)

    out, lse = reference_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sliding_window=4,
        return_lse=True,
    )

    expected_out = []
    expected_lse = []
    for q_start, q_end, k_start, k_end in ((0, 2, 0, 3), (2, 3, 3, 5)):
        out_i, lse_i = reference_attention(
            q[q_start:q_end].transpose(0, 1).unsqueeze(0),
            k[k_start:k_end].transpose(0, 1).unsqueeze(0),
            v[k_start:k_end].transpose(0, 1).unsqueeze(0),
            sliding_window=4,
            q_start=(k_end - k_start) - (q_end - q_start),
            return_lse=True,
        )
        expected_out.append(out_i.squeeze(0).transpose(0, 1))
        expected_lse.append(lse_i.squeeze(0))
    torch.testing.assert_close(out, torch.cat(expected_out))
    torch.testing.assert_close(lse, torch.cat(expected_lse, dim=1))
    assert lse.dtype == torch.float32


def test_packed_varlen_reference_accepts_mixed_empty_segments_and_preserves_gradients():
    generator = torch.Generator().manual_seed(82)
    q = torch.randn(3, 2, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    k = torch.randn(5, 1, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    v = torch.randn(5, 1, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    # Segment lengths are Q=[0,2,0,0,1,0], K=[0,3,1,0,1,0]. This covers
    # leading/middle/trailing paired empties plus a Q-empty/K-nonempty row.
    cu_q = torch.tensor([0, 0, 2, 2, 2, 3, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 3, 4, 4, 5, 5], dtype=torch.int32)

    out, lse = reference_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sliding_window=4,
        return_lse=True,
    )

    expected_out = []
    expected_lse = []
    for q_start, q_end, k_start, k_end in ((0, 2, 0, 3), (2, 3, 4, 5)):
        out_i, lse_i = reference_attention(
            q[q_start:q_end].transpose(0, 1).unsqueeze(0),
            k[k_start:k_end].transpose(0, 1).unsqueeze(0),
            v[k_start:k_end].transpose(0, 1).unsqueeze(0),
            sliding_window=4,
            q_start=(k_end - k_start) - (q_end - q_start),
            return_lse=True,
        )
        expected_out.append(out_i.squeeze(0).transpose(0, 1))
        expected_lse.append(lse_i.squeeze(0))
    torch.testing.assert_close(out, torch.cat(expected_out))
    torch.testing.assert_close(lse, torch.cat(expected_lse, dim=1))

    (out.square().sum() + lse.sum()).backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    torch.testing.assert_close(k.grad[3], torch.zeros_like(k.grad[3]), atol=0, rtol=0)
    torch.testing.assert_close(v.grad[3], torch.zeros_like(v.grad[3]), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("cu_q", "cu_k", "match"),
    [
        (
            torch.tensor([0, 2, 1, 3], dtype=torch.int32),
            torch.tensor([0, 2, 2, 4], dtype=torch.int32),
            "nondecreasing",
        ),
        (
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 4], dtype=torch.int32),
            "Sq greater than Sk",
        ),
    ],
)
def test_packed_varlen_reference_rejects_invalid_mixed_empty_contracts(cu_q, cu_k, match):
    q = torch.zeros(3, 2, 4)
    k = torch.zeros(4, 1, 4)
    v = torch.zeros_like(k)

    with pytest.raises(ValueError, match=match):
        reference_attention_varlen(q, k, v, cu_q, cu_k)


def test_packed_varlen_reference_rejects_nonpositive_packed_totals():
    q = torch.empty(0, 2, 4)
    k = torch.empty(0, 1, 4)
    v = torch.empty_like(k)
    cumulative = torch.tensor([0, 0, 0], dtype=torch.int32)

    with pytest.raises(ValueError, match="totals must be positive"):
        reference_attention_varlen(q, k, v, cumulative, cumulative)


def test_packed_varlen_reference_composes_vision_and_document_masks():
    q = torch.zeros(4, 2, 4, dtype=torch.float64)
    k = torch.zeros(4, 1, 4, dtype=torch.float64)
    v = torch.zeros(4, 1, 4, dtype=torch.float64)
    v[2] = 16
    cumulative = torch.tensor([0, 4], dtype=torch.int32)
    vision = torch.zeros(4, dtype=torch.int32)
    documents = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    with_documents = reference_attention_varlen(
        q,
        k,
        v,
        cumulative,
        cumulative,
        sliding_window=4,
        vision_block_ids=vision,
        document_ids=documents,
        allow_vision_bidirectional=True,
    )
    without_documents = reference_attention_varlen(
        q,
        k,
        v,
        cumulative,
        cumulative,
        sliding_window=4,
        vision_block_ids=vision,
        allow_vision_bidirectional=True,
    )

    assert torch.count_nonzero(with_documents[1]) == 0
    assert torch.count_nonzero(without_documents[1]) > 0


def test_packed_varlen_reference_isolates_repeated_ids_across_segments():
    q = torch.zeros(4, 2, 4, dtype=torch.float64)
    k = torch.zeros(4, 1, 4, dtype=torch.float64)
    v = torch.zeros(4, 1, 4, dtype=torch.float64)
    cumulative = torch.tensor([0, 2, 4], dtype=torch.int32)
    vision = torch.zeros(4, dtype=torch.int32)
    documents = torch.zeros(4, dtype=torch.int32)

    baseline = reference_attention_varlen(
        q,
        k,
        v,
        cumulative,
        cumulative,
        sliding_window=4,
        vision_block_ids=vision,
        document_ids=documents,
        allow_vision_bidirectional=True,
    )
    v_mutated = v.clone()
    v_mutated[2:] = 1024
    mutated = reference_attention_varlen(
        q,
        k,
        v_mutated,
        cumulative,
        cumulative,
        sliding_window=4,
        vision_block_ids=vision,
        document_ids=documents,
        allow_vision_bidirectional=True,
    )

    torch.testing.assert_close(mutated[:2], baseline[:2], atol=0, rtol=0)
    assert torch.count_nonzero(mutated[2:]) > 0


def test_packed_varlen_reference_rejects_metadata_not_matching_k_total():
    q = torch.zeros(2, 2, 4)
    k = torch.zeros(3, 1, 4)
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 3], dtype=torch.int32)
    vision = torch.tensor([0, 0], dtype=torch.int32)

    with pytest.raises(ValueError, match="match Tk"):
        reference_attention_varlen(
            q,
            k,
            v,
            cu_q,
            cu_k,
            vision_block_ids=vision,
        )


def test_packed_varlen_reference_lower_right_vision_truth_table():
    q = torch.zeros(3, 1, 1, dtype=torch.float64)
    k = torch.zeros(5, 1, 1, dtype=torch.float64)
    v = torch.arange(5, dtype=torch.float64).reshape(5, 1, 1)
    cu_q = torch.tensor([0, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 5], dtype=torch.int32)
    vision = torch.tensor([-1, -1, 7, 7, 8], dtype=torch.int32)

    out = reference_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sliding_window=1024,
        vision_block_ids=vision,
        allow_vision_bidirectional=True,
    )

    # q-local 0 maps to absolute K position 2: K0..2 are causal, K3 is the
    # same future vision block, and K4 is a different future block.
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(1.5, dtype=torch.float64))


def test_packed_varlen_reference_strict_w1024_lower_edge():
    q = torch.zeros(1, 1, 1, dtype=torch.float64)
    k = torch.zeros(1025, 1, 1, dtype=torch.float64)
    v = torch.zeros_like(k)
    v[0] = 100
    v[1] = 1
    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_k = torch.tensor([0, 1025], dtype=torch.int32)
    vision = torch.zeros(1025, dtype=torch.int32)

    out = reference_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sliding_window=1024,
        vision_block_ids=vision,
        allow_vision_bidirectional=True,
    )

    # q_abs=1024: K0 fails the strict k>0 bound while K1 is included.
    torch.testing.assert_close(
        out[0, 0, 0],
        torch.tensor(1.0 / 1024.0, dtype=torch.float64),
    )


def test_separate_q_k_v_gradients_exist():
    spec = tiny_spec()
    q, k, v = make_qkv(spec, batch=1, seqlen=4, requires_grad=True)
    out = reference_attention(q, k, v, sliding_window=4)
    out.square().sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert k.grad.data_ptr() != v.grad.data_ptr()


def _explicit_global_kv(source, weight, cos, sin):
    inverse_rms = torch.rsqrt(source.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized = source * inverse_rms
    k_before_rope = normalized * weight.float()
    k_rotary, k_pass = k_before_rope[..., :128], k_before_rope[..., 128:]
    first, second = k_rotary.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    expected_k = torch.cat((k_rotary * cos[:, None] + rotated_half * sin[:, None], k_pass), dim=-1)
    return expected_k.to(source.dtype), normalized.to(source.dtype)


def test_shared_projection_source_uses_exact_global_k_and_v_preparation():
    source = torch.randn(1, 2, 3, 512, dtype=torch.float64, requires_grad=True)
    weight = torch.linspace(0.75, 1.25, 512, dtype=torch.float64, requires_grad=True)
    angles = torch.linspace(0.01, 0.91, 3 * 128, dtype=torch.float64).reshape(1, 3, 128)
    cos, sin = torch.cos(angles), torch.sin(angles)
    k, v = prepare_global_kv_from_projection_source(
        source,
        k_norm_weight=weight,
        cos=cos,
        sin=sin,
        partial_rotary_factor=0.25,
    )
    expected_k, expected_v = _explicit_global_kv(source, weight, cos, sin)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    assert k.data_ptr() != v.data_ptr()

    grad_k = torch.linspace(-1.0, 1.0, k.numel(), dtype=k.dtype).reshape_as(k)
    grad_v = torch.linspace(0.5, 1.5, v.numel(), dtype=v.dtype).reshape_as(v)
    torch.autograd.backward((k, v), (grad_k, grad_v))

    source_ref = source.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    explicit_k, explicit_v = _explicit_global_kv(source_ref, weight_ref, cos, sin)
    torch.autograd.backward((explicit_k, explicit_v), (grad_k, grad_v))
    torch.testing.assert_close(source.grad, source_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


def test_flop_modes_are_consistent():
    spec = tiny_spec()
    fwd = attention_flops(spec, batch=1, seqlen=16, mode="fwd")
    assert attention_flops(spec, batch=1, seqlen=16, mode="bwd") == 2.5 * fwd
    assert attention_flops(spec, batch=1, seqlen=16, mode="fwd_bwd") == 3.5 * fwd


def test_all_masked_query_row_returns_zero_and_negative_infinite_lse():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    v = torch.randn(1, 1, 1, 8)
    out, lse = reference_attention(
        q,
        k,
        v,
        key_padding_mask=torch.tensor([[False]]),
        return_lse=True,
    )
    torch.testing.assert_close(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()
