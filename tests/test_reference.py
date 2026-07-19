import math

import torch

from gemma4_fa4.model_spec import AttentionLayerSpec
from gemma4_fa4.reference import (
    attention_flops,
    make_qkv,
    prepare_global_kv_from_projection_source,
    reference_attention,
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


def test_gqa_matches_materialized_mha():
    q = torch.randn(1, 4, 5, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 5, 8, dtype=torch.float64)
    a = reference_attention(q, k, v)
    b = reference_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
    )
    torch.testing.assert_close(a, b)


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


def test_separate_q_k_v_gradients_exist():
    spec = tiny_spec()
    q, k, v = make_qkv(spec, batch=1, seqlen=4, requires_grad=True)
    out = reference_attention(q, k, v, sliding_window=4)
    out.square().sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert k.grad.data_ptr() != v.grad.data_ptr()


def test_shared_projection_source_produces_distinct_k_and_v():
    source = torch.randn(1, 2, 3, 8)
    weight = torch.linspace(0.8, 1.2, 8)
    cos = torch.ones(1, 3, 2)
    sin = torch.zeros(1, 3, 2)
    k, v = prepare_global_kv_from_projection_source(
        source,
        k_norm_weight=weight,
        cos=cos,
        sin=sin,
        partial_rotary_factor=0.25,
    )
    assert k.data_ptr() != v.data_ptr()
    assert not torch.allclose(k, v)


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


def test_shared_source_preparation_backward_combines_only_through_source():
    source = torch.randn(1, 2, 3, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.linspace(0.8, 1.2, 8, dtype=torch.float64, requires_grad=True)
    angle = torch.linspace(0.0, 0.7, 3, dtype=torch.float64)
    cos = torch.cos(angle)[None, :, None].expand(1, 3, 2)
    sin = torch.sin(angle)[None, :, None].expand(1, 3, 2)
    k, v = prepare_global_kv_from_projection_source(
        source,
        k_norm_weight=weight,
        cos=cos,
        sin=sin,
        partial_rotary_factor=0.25,
    )
    loss = (2.0 * k).sum() + (3.0 * v).sum()
    loss.backward()
    assert source.grad is not None
    assert weight.grad is not None
    assert k.grad_fn is not v.grad_fn
