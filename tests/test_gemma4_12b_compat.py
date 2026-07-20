from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import gemma4_fa4.gemma4_12b_compat as compat
from gemma4_fa4.model_spec import GEMMA4_31B
from gemma4_fa4.reference import reference_attention


def _module(layer_idx: int) -> SimpleNamespace:
    is_global = (layer_idx + 1) % 6 == 0
    return SimpleNamespace(
        layer_idx=layer_idx,
        layer_type="full_attention" if is_global else "sliding_attention",
        head_dim=512 if is_global else 256,
        num_key_value_groups=16 if is_global else 2,
        scaling=1.0,
        sliding_window=None if is_global else 1024,
    )


def _qkv(layer_idx: int, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(3600 + layer_idx)
    is_global = (layer_idx + 1) % 6 == 0
    dim = 512 if is_global else 256
    kv_heads = 1 if is_global else 8
    tensors = (
        torch.randn(1, 16, 3, dim, generator=generator, dtype=torch.float64),
        torch.randn(1, kv_heads, 3, dim, generator=generator, dtype=torch.float64),
        torch.randn(1, kv_heads, 3, dim, generator=generator, dtype=torch.float64),
    )
    if requires_grad:
        tensors = tuple(tensor.requires_grad_() for tensor in tensors)
    return tensors


def test_locked_12b_geometry_and_pattern():
    spec = compat.GEMMA4_12B_HARNESS
    assert spec.model_id == "google/gemma-4-12B-it"
    assert spec.num_hidden_layers == 48
    assert spec.sliding.num_q_heads == 16
    assert spec.sliding.num_kv_heads == 8
    assert spec.sliding.head_dim_qk == 256
    assert spec.full.num_q_heads == 16
    assert spec.full.num_kv_heads == 1
    assert spec.full.head_dim_qk == 512
    assert spec.full.qhead_per_kvhead == 16
    assert spec.layer_types().count("sliding_attention") == 40
    assert spec.layer_types().count("full_attention") == 8


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_head_adapter_matches_31b_kernel_geometry(layer_idx):
    q, k, v = _qkv(layer_idx)
    adapted = compat.adapt_12b_prepared(_module(layer_idx), q, k, v)
    target = GEMMA4_31B.spec_for_layer(layer_idx)

    assert adapted.query.shape[1] == target.num_q_heads
    assert adapted.key.shape[1] == adapted.value.shape[1] == target.num_kv_heads
    assert adapted.output_heads == 16
    torch.testing.assert_close(adapted.query[:, :16], q)
    assert torch.count_nonzero(adapted.query[:, 16:]).item() == 0
    if layer_idx == 0:
        torch.testing.assert_close(adapted.key[:, :8], k)
        torch.testing.assert_close(adapted.value[:, :8], v)
        assert torch.count_nonzero(adapted.key[:, 8:]).item() == 0
        assert torch.count_nonzero(adapted.value[:, 8:]).item() == 0
    else:
        torch.testing.assert_close(adapted.key[:, 0], k[:, 0])
        torch.testing.assert_close(adapted.key[:, 1], k[:, 0])
        torch.testing.assert_close(adapted.value[:, 0], v[:, 0])
        torch.testing.assert_close(adapted.value[:, 1], v[:, 0])
        assert torch.count_nonzero(adapted.key[:, 2:]).item() == 0
        assert torch.count_nonzero(adapted.value[:, 2:]).item() == 0


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_head_adapter_is_forward_equivalent(layer_idx):
    q, k, v = _qkv(layer_idx)
    spec = compat.GEMMA4_12B_HARNESS.spec_for_layer(layer_idx)
    expected = reference_attention(
        q,
        k,
        v,
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        allow_vision_bidirectional=spec.vision_bidirectional_within_block,
    )
    adapted = compat.adapt_12b_prepared(_module(layer_idx), q, k, v)
    actual = reference_attention(
        adapted.query,
        adapted.key,
        adapted.value,
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        allow_vision_bidirectional=spec.vision_bidirectional_within_block,
    )[:, :16]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_head_adapter_is_backward_equivalent(layer_idx):
    q, k, v = _qkv(layer_idx, requires_grad=True)
    spec = compat.GEMMA4_12B_HARNESS.spec_for_layer(layer_idx)
    expected = reference_attention(
        q,
        k,
        v,
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        allow_vision_bidirectional=spec.vision_bidirectional_within_block,
    )
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(99), dtype=q.dtype)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)

    q2, k2, v2 = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    adapted = compat.adapt_12b_prepared(_module(layer_idx), q2, k2, v2)
    actual = reference_attention(
        adapted.query,
        adapted.key,
        adapted.value,
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        allow_vision_bidirectional=spec.vision_bidirectional_within_block,
    )[:, :16]
    actual_grads = torch.autograd.grad(actual, (q2, k2, v2), grad)

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)


def test_attention_entrypoint_slices_bshd_output_and_records_route(monkeypatch):
    q = torch.zeros(1, 16, 2, 256, dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 2, 256, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    module = _module(0)
    captured = {}

    def fake_prepared(surrogate, q_arg, k_arg, v_arg, mask, **kwargs):
        captured.update(
            surrogate=surrogate,
            shapes=(q_arg.shape, k_arg.shape, v_arg.shape),
            mask=mask,
            kwargs=kwargs,
        )
        output = torch.arange(1 * 2 * 32 * 256, dtype=torch.float32).reshape(1, 2, 32, 256)
        return SimpleNamespace(output=output, path="fa4_local_fixed")

    monkeypatch.setattr(compat, "gemma4_fa4_prepared", fake_prepared)
    output, weights = compat.gemma4_12b_compat_attention_forward(
        module,
        q,
        k,
        v,
        None,
        dropout=0.0,
        scaling=1.0,
        sliding_window=1024,
    )

    assert output.shape == (1, 2, 16, 256)
    assert weights is None
    assert captured["shapes"] == ((1, 32, 2, 256), (1, 16, 2, 256), (1, 16, 2, 256))
    assert captured["kwargs"]["allow_flex_fallback"] is False
    assert module._gemma4_fa4_last_path == "fa4_12b_compat/fa4_local_fixed"
    assert module._gemma4_fa4_route_counts == {"fa4_12b_compat/fa4_local_fixed": 1}


@pytest.mark.parametrize(
    ("layer_idx", "shape_index", "replacement", "message"),
    [
        (0, 0, (1, 15, 3, 256), "16 query heads"),
        (0, 1, (1, 7, 3, 256), "8 KV heads"),
        (5, 1, (1, 2, 3, 512), "1 KV head"),
    ],
)
def test_invalid_12b_geometry_fails_closed(layer_idx, shape_index, replacement, message):
    tensors = list(_qkv(layer_idx))
    tensors[shape_index] = torch.zeros(replacement, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        compat.adapt_12b_prepared(_module(layer_idx), *tensors)


def test_12b_registration_is_idempotent_and_reuses_base_mask(monkeypatch):
    class Registry(dict):
        def register(self, name, value):
            self[name] = value

    attentions = Registry()
    masks = Registry()
    base_mask = object()

    def register_base():
        masks[compat.BACKEND_NAME] = base_mask
        return compat.BACKEND_NAME

    monkeypatch.setattr(compat, "register_gemma4_fa4_h100", register_base)
    modeling = ModuleType("transformers.modeling_utils")
    modeling.ALL_ATTENTION_FUNCTIONS = attentions
    masking = ModuleType("transformers.masking_utils")
    masking.ALL_MASK_ATTENTION_FUNCTIONS = masks
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking)

    assert compat.register_gemma4_fa4_h100_12b_compat() == compat.BACKEND_NAME_12B_COMPAT
    assert compat.register_gemma4_fa4_h100_12b_compat() == compat.BACKEND_NAME_12B_COMPAT
    assert attentions[compat.BACKEND_NAME_12B_COMPAT] is compat.gemma4_12b_compat_attention_forward
    assert masks[compat.BACKEND_NAME_12B_COMPAT] is base_mask

    attentions[compat.BACKEND_NAME_12B_COMPAT] = object()
    with pytest.raises(RuntimeError, match="already registered"):
        compat.register_gemma4_fa4_h100_12b_compat()
