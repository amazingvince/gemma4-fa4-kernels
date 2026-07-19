import math

import pytest
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
