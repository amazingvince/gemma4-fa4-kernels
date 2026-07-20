from __future__ import annotations

import sys
from dataclasses import fields
from types import ModuleType, SimpleNamespace

import pytest
import torch

import gemma4_fa4.transformers_integration as integration
from gemma4_fa4.h100 import UnsupportedH100Path
from gemma4_fa4.model_spec import GEMMA4_31B


def _module(layer_idx: int):
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    return SimpleNamespace(
        layer_idx=layer_idx,
        layer_type=spec.kind,
        is_sliding=spec.kind == "sliding_attention",
        head_dim=spec.head_dim_qk,
        num_key_value_groups=spec.qhead_per_kvhead,
        scaling=spec.softmax_scale,
        sliding_window=spec.sliding_window,
    )


def _qkv_bhsd(
    layer_idx: int = 0,
    *,
    batch: int = 1,
    q_length: int = 3,
    kv_length: int | None = None,
):
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    kv_length = q_length if kv_length is None else kv_length
    q = torch.zeros(
        batch,
        spec.num_q_heads,
        q_length,
        spec.head_dim_qk,
        dtype=torch.bfloat16,
    )
    k = torch.ones(
        batch,
        spec.num_kv_heads,
        kv_length,
        spec.head_dim_qk,
        dtype=torch.bfloat16,
    )
    v = torch.full(
        (batch, spec.num_kv_heads, kv_length, spec.head_dim_v),
        2,
        dtype=torch.bfloat16,
    )
    return q, k, v


def _prepared(layer_idx: int, q, k, v, attention_mask=None, **kwargs):
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    return integration.gemma4_fa4_prepared(
        _module(layer_idx),
        q,
        k,
        v,
        attention_mask,
        dropout=kwargs.pop("dropout", 0.0),
        scaling=kwargs.pop("scaling", 1.0),
        sliding_window=kwargs.pop("sliding_window", spec.sliding_window),
        **kwargs,
    )


@pytest.fixture
def fast_paths(monkeypatch):
    calls = {"local": [], "varlen": [], "global": []}

    def local(q, k, v, **kwargs):
        calls["local"].append((q, k, v, kwargs))
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q.clone(), lse

    def varlen(q, k, v, cu_q, cu_k, **kwargs):
        calls["varlen"].append((q, k, v, cu_q, cu_k, kwargs))
        lse = torch.zeros(q.shape[1], q.shape[0], dtype=torch.float32)
        return q.clone(), lse

    def global_fixed(q, k, v, **kwargs):
        calls["global"].append((q, k, v, kwargs))
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q.clone(), lse

    monkeypatch.setattr(integration, "fa4_local_forward", local)
    monkeypatch.setattr(integration, "fa4_local_varlen_forward", varlen)
    monkeypatch.setattr(integration, "fa4_global_text_forward", global_fixed)
    return calls


def test_public_contract_types_are_stable():
    assert integration.BACKEND_NAME == "gemma4_fa4_h100"
    assert tuple(field.name for field in fields(integration.Gemma4MaskPlan)) == (
        "batch_size",
        "q_length",
        "kv_length",
        "q_offset",
        "kv_offset",
        "mask_function",
        "attention_mask",
    )
    assert tuple(field.name for field in fields(integration.Gemma4DispatchResult)) == (
        "output",
        "lse",
        "path",
    )


def test_mask_adapter_preserves_callable_padding_and_offsets():
    def mask_function(*_args):
        return True

    padding = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.int64)

    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=3,
        kv_length=5,
        q_offset=7,
        kv_offset=2,
        mask_function=mask_function,
        attention_mask=padding,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert isinstance(plan, integration.Gemma4MaskPlan)
    assert (plan.batch_size, plan.q_length, plan.kv_length) == (2, 3, 5)
    assert (plan.q_offset, plan.kv_offset) == (7, 2)
    assert plan.mask_function is mask_function
    assert plan.attention_mask is padding


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
@pytest.mark.parametrize("mask_kind", ["dense-4d", "stricter-callable"])
def test_arbitrary_exact_masks_never_enter_a_native_fast_path(
    monkeypatch,
    fast_paths,
    layer_idx,
    mask_kind,
):
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=3)
    expected = torch.full(
        (1, 3, 32, q.shape[-1]),
        29,
        dtype=torch.bfloat16,
    )
    if mask_kind == "dense-4d":
        allowed = torch.tril(torch.ones(1, 1, 3, 3, dtype=torch.bool))
        allowed[..., 0] = False
        dense_mask = torch.where(allowed, 0.0, -torch.inf)
        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=3,
            kv_length=3,
            attention_mask=dense_mask,
        )
    else:

        def stricter_mask(_batch, _head, query, key):
            return (key <= query) & (key != 0)

        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=3,
            kv_length=3,
            mask_function=stricter_mask,
        )

    captured = {}

    def fallback(module, q_arg, k_arg, v_arg, received_plan, *, scaling):
        captured.update(module=module, plan=received_plan, scaling=scaling)
        return expected

    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    result = _prepared(layer_idx, q, k, v, plan)

    assert result.output is expected
    assert result.path == "flex_attention"
    assert captured["plan"] is plan
    assert captured["module"].layer_idx == layer_idx
    assert captured["scaling"] == 1.0
    assert not any(fast_paths.values())


def test_forged_pinned_callable_metadata_cannot_enter_native_path(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=3)

    def stricter_mask(_batch, _head, query, key):
        return (key <= query) & (key != 0)

    stricter_mask.__module__ = "transformers.masking_utils"
    stricter_mask.__qualname__ = "causal_mask_function"
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=3,
        kv_length=3,
        mask_function=stricter_mask,
    )
    expected = torch.full((1, 3, 32, 512), 41, dtype=torch.bfloat16)
    monkeypatch.setattr(
        integration,
        "_run_flex_fallback",
        lambda *_args, **_kwargs: expected,
    )

    result = _prepared(5, q, k, v, plan)

    assert result.path == "flex_attention"
    assert result.output is expected
    assert not any(fast_paths.values())


def test_all_negative_pinned_vision_overlay_is_proven_text_only(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=0, q_length=5)
    blocks = torch.full((1, 5), -1, dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=5,
        kv_length=5,
        mask_function=lambda *_args: True,
    )
    expression = (
        "or",
        integration._and_expression(("causal",), ("sliding", 1024)),
        ("vision",),
    )
    monkeypatch.setattr(
        integration,
        "_parse_pinned_mask_function",
        lambda _function: (expression, {"vision": [blocks], "packed": []}),
    )
    monkeypatch.setattr(
        integration,
        "_mask_plan_has_future",
        lambda *_args, **_kwargs: pytest.fail(
            "a proven all-negative vision overlay was re-evaluated"
        ),
    )

    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        position_ids=torch.arange(5).unsqueeze(0),
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_local_fixed"
    assert len(fast_paths["local"]) == 1


def test_all_negative_pinned_global_vision_overlay_reduces_to_causal(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=5)
    blocks = torch.full((1, 5), -1, dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=5,
        kv_length=5,
        mask_function=lambda *_args: True,
    )
    expression = ("or", ("causal",), ("vision",))
    monkeypatch.setattr(
        integration,
        "_parse_pinned_mask_function",
        lambda _function: (expression, {"vision": [blocks], "packed": []}),
    )

    result = _prepared(
        5,
        q,
        k,
        v,
        plan,
        position_ids=torch.arange(5).unsqueeze(0),
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_fixed"
    assert len(fast_paths["global"]) == 1


def test_active_pinned_vision_overlay_without_metadata_stays_fail_closed(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=0, q_length=5)
    blocks = torch.tensor([[-1, 7, 7, -1, -1]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=5,
        kv_length=5,
        mask_function=lambda *_args: True,
    )
    expression = integration._and_expression(
        ("or", ("causal",), ("vision",)),
        ("sliding", 1024),
    )
    monkeypatch.setattr(
        integration,
        "_parse_pinned_mask_function",
        lambda _function: (expression, {"vision": [blocks], "packed": []}),
    )

    with pytest.raises(UnsupportedH100Path, match="not the proven"):
        _prepared(
            0,
            q,
            k,
            v,
            plan,
            position_ids=torch.arange(5).unsqueeze(0),
            allow_flex_fallback=False,
        )

    assert not any(fast_paths.values())


def test_active_pinned_global_vision_overlay_stays_fail_closed(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=5)
    blocks = torch.tensor([[-1, 7, 7, -1, -1]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=5,
        kv_length=5,
        mask_function=lambda *_args: True,
    )
    monkeypatch.setattr(
        integration,
        "_parse_pinned_mask_function",
        lambda _function: (
            ("or", ("causal",), ("vision",)),
            {"vision": [blocks], "packed": []},
        ),
    )

    with pytest.raises(UnsupportedH100Path, match="not the proven"):
        _prepared(
            5,
            q,
            k,
            v,
            plan,
            position_ids=torch.arange(5).unsqueeze(0),
            allow_flex_fallback=False,
        )

    assert not any(fast_paths.values())


def test_active_pinned_vision_overlay_with_negative_kv_offset_stays_fail_closed(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=0, q_length=5)
    blocks = torch.tensor([[-1, 7, 7, -1, -1]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=5,
        kv_length=5,
        kv_offset=-5,
        mask_function=lambda *_args: True,
    )
    expression = (
        "or",
        integration._and_expression(("causal",), ("sliding", 1024)),
        ("vision",),
    )
    monkeypatch.setattr(
        integration,
        "_parse_pinned_mask_function",
        lambda _function: (expression, {"vision": [blocks], "packed": []}),
    )

    with pytest.raises(UnsupportedH100Path, match="not the proven"):
        _prepared(
            0,
            q,
            k,
            v,
            plan,
            position_ids=torch.arange(5).unsqueeze(0),
            allow_flex_fallback=False,
        )

    assert not any(fast_paths.values())


def test_boolean_4d_mask_fails_closed_instead_of_becoming_additive_scores(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=3)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=3,
        kv_length=3,
        attention_mask=torch.tril(torch.ones(1, 1, 3, 3, dtype=torch.bool)),
    )
    flex_module = ModuleType("transformers.integrations.flex_attention")
    flex_module.flex_attention_forward = lambda *_args, **_kwargs: pytest.fail(
        "boolean 4D mask reached the additive FlexAttention interface"
    )
    masking_module = ModuleType("transformers.masking_utils")
    masking_module.causal_mask_function = lambda _b, _h, q_idx, kv_idx: kv_idx <= q_idx
    masking_module.flex_attention_mask = lambda **_kwargs: pytest.fail(
        "a 4D mask must not be rebuilt as a block mask"
    )
    monkeypatch.setitem(sys.modules, "transformers.integrations.flex_attention", flex_module)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_module)

    with pytest.raises(UnsupportedH100Path, match="ambiguous semantics"):
        integration._run_flex_fallback(
            _module(5),
            q,
            k,
            v,
            plan,
            scaling=1.0,
        )


def test_locked_sixty_layer_pattern_routes_fifty_local_and_ten_global(fast_paths):
    observed_paths = []
    for layer_idx, kind in enumerate(GEMMA4_31B.layer_types()):
        q, k, v = _qkv_bhsd(layer_idx, q_length=1)
        result = _prepared(layer_idx, q, k, v, allow_flex_fallback=False)
        observed_paths.append(result.path)
        assert result.output.shape == (1, 1, 32, q.shape[-1])
        expected = "fa4_global_fixed" if kind == "full_attention" else "fa4_local_fixed"
        assert result.path == expected

    assert observed_paths.count("fa4_local_fixed") == 50
    assert observed_paths.count("fa4_global_fixed") == 10
    assert len(fast_paths["local"]) == 50
    assert len(fast_paths["global"]) == 10


def test_fixed_path_uses_storage_sharing_bhsd_to_bshd_views(fast_paths):
    q, k, v = _qkv_bhsd(q_length=3)

    result = _prepared(0, q, k, v, allow_flex_fallback=False)
    q_arg, k_arg, v_arg, _ = fast_paths["local"][0]

    for original, received in zip((q, k, v), (q_arg, k_arg, v_arg), strict=True):
        expected = original.transpose(1, 2)
        assert received.shape == expected.shape
        assert received.stride() == expected.stride()
        assert received.storage_offset() == original.storage_offset()
        assert received.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        assert not received.is_contiguous()
    assert result.path == "fa4_local_fixed"


def test_attention_interface_returns_bshd_output_and_no_weights(fast_paths):
    q, k, v = _qkv_bhsd(q_length=2)

    output, weights = integration.gemma4_fa4_attention_forward(
        _module(0),
        q,
        k,
        v,
        None,
        dropout=0.0,
        scaling=1.0,
        sliding_window=1024,
        allow_flex_fallback=False,
    )

    assert output.shape == (1, 2, 32, 256)
    assert weights is None
    assert len(fast_paths["local"]) == 1


def test_local_fixed_path_forwards_explicit_vision_metadata(fast_paths):
    q, k, v = _qkv_bhsd(q_length=5)
    vision = torch.tensor([[-1, 7, 7, 7, -1]], dtype=torch.int32)

    result = _prepared(
        0,
        q,
        k,
        v,
        vision_block_ids=vision,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_local_fixed"
    forwarded = fast_paths["local"][0][3]["vision_block_ids"]
    torch.testing.assert_close(forwarded, vision)


def test_explicit_packed_cumulative_arrays_preserve_token_order(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(q_length=5)
    for token in range(5):
        q[:, :, token].fill_(token)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(batch_size=1, q_length=5, kv_length=5)

    def packed(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        fast_paths["varlen"].append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg + 10, torch.zeros(32, 5, dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_local_varlen_forward", packed)
    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu.clone(),
        max_length_q=3,
        max_length_k=3,
        allow_flex_fallback=False,
    )

    q_arg, _, _, cu_q, cu_k, kwargs = fast_paths["varlen"][0]
    assert q_arg.shape == (5, 32, 256)
    torch.testing.assert_close(q_arg[:, 0, 0], torch.arange(5, dtype=torch.bfloat16))
    torch.testing.assert_close(cu_q, cu)
    torch.testing.assert_close(cu_k, cu)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert result.path == "fa4_local_varlen"
    torch.testing.assert_close(result.output[0, :, 0, 0], torch.arange(10, 15).bfloat16())


def test_broadcast_positions_work_and_explicit_cu_cannot_cross_batch_rows(fast_paths):
    q, k, v = _qkv_bhsd(batch=2, q_length=4)
    padding = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
    )

    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        position_ids=torch.tensor([[0, 1, 0, 99]]),
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_local_varlen"
    _, _, _, cu_q, cu_k, kwargs = fast_paths["varlen"][0]
    expected_cu = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int32)
    torch.testing.assert_close(cu_q, expected_cu)
    torch.testing.assert_close(cu_k, expected_cu)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 2

    unpadded_plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
    )
    crossing_rows = torch.tensor([0, 6, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="batch-row boundary|batch row boundary"):
        _prepared(
            0,
            q,
            k,
            v,
            unpadded_plan,
            cu_seq_lens_q=crossing_rows,
            cu_seq_lens_k=crossing_rows.clone(),
            allow_flex_fallback=False,
        )
    assert len(fast_paths["varlen"]) == 1


def test_lower_right_local_context_routes_native_varlen(fast_paths):
    q, k, v = _qkv_bhsd(q_length=1, kv_length=5)
    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_k = torch.tensor([0, 5], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=5,
        q_offset=4,
    )

    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        position_ids=torch.tensor([[4]]),
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=1,
        max_length_k=5,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_local_varlen"
    q_arg, k_arg, _, got_cu_q, got_cu_k, kwargs = fast_paths["varlen"][0]
    assert q_arg.shape == (1, 32, 256)
    assert k_arg.shape == (5, 16, 256)
    torch.testing.assert_close(got_cu_q, cu_q)
    torch.testing.assert_close(got_cu_k, cu_k)
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 5


def test_padding_mask_derives_cumulative_arrays_and_scatters_output(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(batch=2, q_length=4)
    for batch in range(2):
        for token in range(4):
            q[batch, :, token].fill_(10 * batch + token)
    padding = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
    )

    def packed(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        fast_paths["varlen"].append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg + 100, torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_local_varlen_forward", packed)
    result = _prepared(0, q, k, v, plan, allow_flex_fallback=False)

    q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = fast_paths["varlen"][0]
    assert q_arg.shape == (5, 32, 256)
    assert k_arg.shape == v_arg.shape == (5, 16, 256)
    torch.testing.assert_close(cu_q, torch.tensor([0, 3, 5], dtype=torch.int32))
    torch.testing.assert_close(cu_k, torch.tensor([0, 3, 5], dtype=torch.int32))
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert result.path == "fa4_local_varlen"
    torch.testing.assert_close(
        result.output[:, :, 0, 0],
        torch.tensor([[100, 101, 102, 0], [110, 111, 0, 0]], dtype=torch.bfloat16),
    )
    assert result.lse.shape == (2, 32, 4)
    assert torch.isneginf(result.lse[0, :, 3]).all()
    assert torch.isneginf(result.lse[1, :, 2:]).all()


@pytest.mark.parametrize(
    ("layer_idx", "expected_path", "backend_name"),
    [
        (0, "fa4_local_varlen", "fa4_local_varlen_forward"),
        (5, "fa4_global_varlen_native", "fa4_global_varlen_forward"),
    ],
    ids=["local", "global"],
)
def test_fully_padded_row_packs_as_empty_segment_and_restores_sentinels(
    monkeypatch,
    layer_idx,
    expected_path,
    backend_name,
):
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, batch=2, q_length=4)
    q[1, :, 0].fill_(11)
    q[1, :, 1].fill_(12)
    q[1, :, 2].fill_(13)
    q.requires_grad_()
    padding = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
    )
    calls = []

    def packed(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        calls.append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        lse = torch.zeros(q_arg.shape[1], q_arg.shape[0], dtype=torch.float32)
        return q_arg + 100, lse

    monkeypatch.setattr(integration, backend_name, packed)
    result = _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)

    assert result.path == expected_path
    assert len(calls) == 1
    q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = calls[0]
    assert q_arg.shape[0] == k_arg.shape[0] == v_arg.shape[0] == 3
    expected_cu = torch.tensor([0, 0, 3], dtype=torch.int32)
    torch.testing.assert_close(cu_q, expected_cu)
    torch.testing.assert_close(cu_k, expected_cu)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3
    assert torch.count_nonzero(result.output[0]).item() == 0
    assert torch.isneginf(result.lse[0]).all()
    torch.testing.assert_close(
        result.output[1, :, 0, 0],
        torch.tensor([111, 112, 113, 0], dtype=torch.bfloat16),
    )
    assert torch.isneginf(result.lse[1, :, 3]).all()


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_all_empty_padded_workload_rejects_before_backend(monkeypatch, layer_idx):
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, batch=2, q_length=4)
    q.requires_grad_()
    padding = torch.zeros((2, 4), dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
    )
    backend_calls = []

    def forbidden_backend(*_args, **_kwargs):
        backend_calls.append(True)
        raise AssertionError("all-empty workload reached an attention backend")

    monkeypatch.setattr(integration, "fa4_local_varlen_forward", forbidden_backend)
    monkeypatch.setattr(integration, "fa4_global_varlen_forward", forbidden_backend)

    with pytest.raises(UnsupportedH100Path, match="positive"):
        _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)
    assert not backend_calls


def test_padding_pack_gathers_vision_and_document_ids(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(batch=2, q_length=4)
    padding = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)
    vision = torch.tensor([[-1, 7, 7, 99], [-1, 8, 99, 99]], dtype=torch.int32)
    documents = torch.tensor([[0, 0, 1, 99], [4, 4, 99, 99]], dtype=torch.int32)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
    )

    def packed(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        fast_paths["varlen"].append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_local_varlen_forward", packed)
    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        vision_block_ids=vision,
        document_ids=documents,
        allow_flex_fallback=False,
    )

    kwargs = fast_paths["varlen"][0][5]
    torch.testing.assert_close(
        kwargs["vision_block_ids"],
        torch.tensor([-1, 7, 7, -1, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        kwargs["document_ids"],
        torch.tensor([0, 0, 1, 4, 4], dtype=torch.int32),
    )
    assert result.path == "fa4_local_varlen"


def test_global_fixed_envelope_uses_composed_h100_path(fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=4)

    result = _prepared(5, q, k, v, allow_flex_fallback=False)

    assert result.path == "fa4_global_fixed"
    assert result.lse.shape == (1, 32, 4)
    assert len(fast_paths["global"]) == 1
    assert not fast_paths["local"] and not fast_paths["varlen"]


def test_packed_cumulative_total_rejects_signed_int32_overflow():
    assert integration._cumulative([1, 262_144], torch.device("cpu")).tolist() == [0, 1, 262145]
    with pytest.raises(UnsupportedH100Path, match="signed INT32"):
        integration._cumulative([torch.iinfo(torch.int32).max, 1], torch.device("cpu"))


@pytest.mark.parametrize("scenario", ["batch", "padding-packing", "lower-right"])
def test_global_varlen_composes_exact_per_segment_calls(monkeypatch, fast_paths, scenario):
    if scenario == "batch":
        q, k, v = _qkv_bhsd(layer_idx=5, batch=2, q_length=3)
        plan = integration.gemma4_fa4_mask(batch_size=2, q_length=3, kv_length=3)
        call_kwargs = {}
    elif scenario == "padding-packing":
        q, k, v = _qkv_bhsd(layer_idx=5, batch=2, q_length=4)
        padding = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.int32)
        plan = integration.gemma4_fa4_mask(
            batch_size=2,
            q_length=4,
            kv_length=4,
            attention_mask=padding,
        )
        call_kwargs = {
            "position_ids": torch.tensor([[0, 1, 2, 99], [0, 1, 0, 1]]),
        }
    else:
        q, k, v = _qkv_bhsd(layer_idx=5, q_length=2, kv_length=5)
        q.fill_(4)
        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=2,
            kv_length=5,
            q_offset=3,
        )
        call_kwargs = {"position_ids": torch.tensor([[3, 4]])}

    def global_segment(q_arg, k_arg, v_arg, **kwargs):
        fast_paths["global"].append((q_arg, k_arg, v_arg, kwargs))
        seqlen = q_arg.shape[1]
        lse = torch.arange(seqlen, dtype=torch.float32).view(1, 1, seqlen)
        lse = lse.expand(1, 32, seqlen).clone()
        return q_arg + 10, lse

    monkeypatch.setattr(integration, "fa4_global_text_forward", global_segment)
    result = _prepared(
        5,
        q,
        k,
        v,
        plan,
        allow_flex_fallback=False,
        **call_kwargs,
    )

    assert result.path == "fa4_global_varlen"
    if scenario == "batch":
        assert len(fast_paths["global"]) == 2
        assert [call[0].shape[1] for call in fast_paths["global"]] == [3, 3]
        torch.testing.assert_close(result.output, torch.full_like(result.output, 10))
        expected_lse = torch.arange(3, dtype=torch.float32).view(1, 1, 3)
        torch.testing.assert_close(result.lse, expected_lse.expand(2, 32, 3))
    elif scenario == "padding-packing":
        assert [call[0].shape[1] for call in fast_paths["global"]] == [3, 2, 2]
        for q_arg, k_arg, v_arg, _ in fast_paths["global"]:
            assert q_arg.is_contiguous() and k_arg.is_contiguous() and v_arg.is_contiguous()
        assert torch.all(result.output[0, :3] == 10)
        assert torch.count_nonzero(result.output[0, 3]) == 0
        assert torch.all(result.output[1] == 10)
        torch.testing.assert_close(result.lse[0, 0, :3], torch.tensor([0.0, 1.0, 2.0]))
        assert torch.isneginf(result.lse[0, :, 3]).all()
        torch.testing.assert_close(result.lse[1, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    else:
        assert len(fast_paths["global"]) == 1
        padded_q, padded_k, padded_v, _ = fast_paths["global"][0]
        assert padded_q.shape == (1, 5, 32, 512)
        assert padded_k.shape == padded_v.shape == (1, 5, 4, 512)
        assert torch.count_nonzero(padded_q[:, :3]) == 0
        assert torch.all(padded_q[:, 3:] == 4)
        assert torch.all(result.output == 14)
        torch.testing.assert_close(result.lse[0, 0], torch.tensor([3.0, 4.0]))


def test_long_global_no_grad_routes_fixed_and_packed_forward_only(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=1025)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=1025,
        q_offset=1024,
    )
    calls = []

    def fixed(q_arg, k_arg, v_arg, **_kwargs):
        calls.append(("fixed", q_arg, k_arg, v_arg))
        return q_arg.clone(), torch.zeros(1, 32, 1, dtype=torch.float32)

    def varlen(q_arg, k_arg, v_arg, cu_q, cu_k, **_kwargs):
        calls.append(("varlen", q_arg, k_arg, v_arg, cu_q, cu_k))
        return q_arg.clone(), torch.zeros(32, 1, dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_global_forward_only", fixed)
    monkeypatch.setattr(integration, "fa4_global_varlen_forward_only", varlen)

    direct = _prepared(5, q, k, v, plan, allow_flex_fallback=False)
    assert direct.path == "fa4_global_forward_only"
    assert calls[0][0] == "fixed"

    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_k = torch.tensor([0, 1025], dtype=torch.int32)
    packed = _prepared(
        5,
        q,
        k,
        v,
        plan,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=1,
        max_length_k=1025,
        allow_flex_fallback=False,
    )
    assert packed.path == "fa4_global_varlen_forward_only"
    assert calls[1][0] == "varlen"
    torch.testing.assert_close(calls[1][4], cu_q)
    torch.testing.assert_close(calls[1][5], cu_k)
    assert not any(fast_paths.values())


def test_long_global_grad_uses_native_varlen_without_zero_prefix(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=2049)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=2049,
        q_offset=2048,
    )

    native_calls = []

    def global_native(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        native_calls.append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_global_varlen_forward", global_native)
    result = _prepared(5, q, k, v, plan, allow_flex_fallback=False)

    assert result.path == "fa4_global_varlen_native"
    assert len(native_calls) == 1
    packed_q, packed_k, packed_v, cu_q, cu_k, kwargs = native_calls[0]
    assert packed_q.shape == (1, 32, 512)
    assert packed_k.shape == packed_v.shape == (2049, 4, 512)
    torch.testing.assert_close(cu_q, torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(cu_k, torch.tensor([0, 2049], dtype=torch.int32))
    assert kwargs["max_seqlen_q"] == 1 and kwargs["max_seqlen_k"] == 2049
    assert not fast_paths["global"]
    assert result.output.shape == (1, 1, 32, 512)


@pytest.mark.parametrize(
    ("q_lengths", "k_lengths"),
    [((33,), (4097,)), ((33, 65), (2049, 4097))],
)
def test_exp0014_long_explicit_packed_training_routes_native_thd(
    monkeypatch,
    fast_paths,
    q_lengths,
    k_lengths,
):
    q_total, k_total = sum(q_lengths), sum(k_lengths)
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=q_total, kv_length=k_total)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=q_total,
        kv_length=k_total,
        q_offset=k_total - q_total,
    )
    native_calls = []

    def global_native(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        native_calls.append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    def cumulative(lengths):
        values = [0]
        for length in lengths:
            values.append(values[-1] + length)
        return torch.tensor(values, dtype=torch.int32)

    cu_q, cu_k = cumulative(q_lengths), cumulative(k_lengths)
    monkeypatch.setattr(integration, "fa4_global_varlen_forward", global_native)
    result = _prepared(
        5,
        q,
        k,
        v,
        plan,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=max(q_lengths),
        max_length_k=max(k_lengths),
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_varlen_native"
    assert len(native_calls) == 1
    q_arg, k_arg, v_arg, observed_cu_q, observed_cu_k, kwargs = native_calls[0]
    assert q_arg.shape == (q_total, 32, 512)
    assert k_arg.shape == v_arg.shape == (k_total, 4, 512)
    torch.testing.assert_close(observed_cu_q, cu_q)
    torch.testing.assert_close(observed_cu_k, cu_k)
    assert kwargs["max_seqlen_q"] == max(q_lengths)
    assert kwargs["max_seqlen_k"] == max(k_lengths)
    assert not any(fast_paths.values())


def test_explicit_global_plateaus_reach_native_backend_unchanged(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=2, kv_length=5)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=5,
        q_offset=3,
    )
    cu_q = torch.tensor([0, 0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    calls = []

    def global_native(q_arg, k_arg, v_arg, got_cu_q, got_cu_k, **kwargs):
        calls.append((q_arg, k_arg, v_arg, got_cu_q, got_cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_global_varlen_forward", global_native)
    result = _prepared(
        5,
        q,
        k,
        v,
        plan,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=2,
        max_length_k=3,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_varlen_native"
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][3], cu_q)
    torch.testing.assert_close(calls[0][4], cu_k)
    assert calls[0][5]["max_seqlen_q"] == 2
    assert calls[0][5]["max_seqlen_k"] == 3


def test_global_exact_composer_skips_query_empty_key_nonempty_segment(monkeypatch):
    spec = GEMMA4_31B.spec_for_layer(5)
    q = torch.randn(2, 32, 512, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(5, 4, 512, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(5, 4, 512, dtype=torch.bfloat16, requires_grad=True)
    packed = integration._PackedLocalInputs(
        q=q,
        k=k,
        v=v,
        cu_q=torch.tensor([0, 0, 2], dtype=torch.int32),
        cu_k=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_q=2,
        max_k=3,
        q_flat_indices=None,
        original_q_shape=(1, 2, 32, 512),
        vision_block_ids=None,
        document_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
    )
    calls = []
    monkeypatch.setattr(integration, "_GLOBAL_COMPOSED_BACKWARD_MAX_SEQLEN", 2)

    def global_segment(q_arg, k_arg, v_arg, **_kwargs):
        calls.append((q_arg, k_arg, v_arg))
        expanded_k = torch.repeat_interleave(k_arg, spec.qhead_per_kvhead, dim=2)
        expanded_v = torch.repeat_interleave(v_arg, spec.qhead_per_kvhead, dim=2)
        output = q_arg + expanded_k + expanded_v
        lse = (q_arg[..., 0].float() + expanded_k[..., 0].float()).permute(0, 2, 1)
        return output, lse

    monkeypatch.setattr(integration, "fa4_global_text_forward", global_segment)
    output, lse = integration._run_global_composed(packed, spec)
    gradients = torch.autograd.grad(output.float().sum() + lse.sum(), (q, k, v))

    assert len(calls) == 1
    assert calls[0][0].shape[1] == calls[0][1].shape[1] == 2
    assert output.shape == (1, 2, 32, 512)
    assert lse.shape == (1, 32, 2)
    assert torch.count_nonzero(gradients[1][:3]).item() == 0
    assert torch.count_nonzero(gradients[2][:3]).item() == 0
    assert torch.count_nonzero(gradients[1][3:]).item() > 0
    assert torch.count_nonzero(gradients[2][3:]).item() > 0


def test_global_budget_fallback_ignores_over_cap_k_for_query_empty_segment(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=2, kv_length=5)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=5,
        q_offset=3,
    )
    cu_q = torch.tensor([0, 0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    composed_calls = []

    def native_budget(*_args, **_kwargs):
        raise integration.GlobalBackwardBudgetExceeded("synthetic mixed-empty budget")

    def global_segment(q_arg, k_arg, v_arg, **_kwargs):
        composed_calls.append((q_arg, k_arg, v_arg))
        return q_arg.clone(), torch.zeros(1, 32, q_arg.shape[1], dtype=torch.float32)

    monkeypatch.setattr(integration, "_GLOBAL_COMPOSED_BACKWARD_MAX_SEQLEN", 2)
    monkeypatch.setattr(integration, "fa4_global_varlen_forward", native_budget)
    monkeypatch.setattr(integration, "fa4_global_text_forward", global_segment)
    result = _prepared(
        5,
        q,
        k,
        v,
        plan,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=2,
        max_length_k=3,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_varlen_composed_budget_fallback"
    assert len(composed_calls) == 1
    assert composed_calls[0][0].shape[1] == composed_calls[0][1].shape[1] == 2


def test_native_varlen_budget_only_falls_back_to_composition(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=1025)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=1025,
        q_offset=1024,
    )
    composed_calls = []

    def native_budget(*_args, **_kwargs):
        raise integration.GlobalBackwardBudgetExceeded("synthetic native budget rejection")

    def composed(packed, _spec):
        composed_calls.append(packed)
        output = torch.zeros(1, 1, 32, 512, dtype=torch.bfloat16)
        lse = torch.zeros(1, 32, 1, dtype=torch.float32)
        return output, lse

    monkeypatch.setattr(integration, "_run_global_native_varlen", native_budget)
    monkeypatch.setattr(integration, "_run_global_composed", composed)
    result = _prepared(5, q, k, v, plan, allow_flex_fallback=False)

    assert result.path == "fa4_global_varlen_composed_budget_fallback"
    assert len(composed_calls) == 1


def test_long_native_varlen_budget_rejection_never_enters_composer_or_flex(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=2049)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=2049,
        q_offset=2048,
    )
    composed_calls = []
    flex_calls = []

    def native_budget(*_args, **_kwargs):
        raise integration.GlobalBackwardBudgetExceeded("synthetic long native budget")

    monkeypatch.setattr(integration, "_run_global_native_varlen", native_budget)
    monkeypatch.setattr(
        integration,
        "_run_global_composed",
        lambda *_args, **_kwargs: composed_calls.append(True),
    )
    monkeypatch.setattr(
        integration,
        "_run_flex_fallback",
        lambda *_args, **_kwargs: flex_calls.append(True),
    )

    with pytest.raises(
        integration.GlobalBackwardBudgetExceeded,
        match="synthetic long native budget",
    ):
        _prepared(5, q, k, v, plan)
    assert not composed_calls
    assert not flex_calls


def test_native_varlen_runtime_failure_does_not_fall_back(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=1025)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=1025,
        q_offset=1024,
    )
    composed_calls = []

    def native_failure(*_args, **_kwargs):
        raise RuntimeError("synthetic native kernel failure")

    monkeypatch.setattr(integration, "_run_global_native_varlen", native_failure)
    monkeypatch.setattr(
        integration,
        "_run_global_composed",
        lambda *_args, **_kwargs: composed_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="synthetic native kernel failure"):
        _prepared(5, q, k, v, plan, allow_flex_fallback=False)
    assert not composed_calls


def test_native_varlen_rebuilds_cumulative_arrays_after_document_split(monkeypatch):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=5)
    q.requires_grad_()
    documents = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
    native_calls = []

    def global_native(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        native_calls.append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_global_varlen_forward", global_native)
    result = _prepared(
        5,
        q,
        k,
        v,
        document_ids=documents,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_varlen_native"
    assert len(native_calls) == 1
    q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = native_calls[0]
    assert q_arg.shape == (5, 32, 512)
    assert k_arg.shape == v_arg.shape == (5, 4, 512)
    expected = torch.tensor([0, 2, 5], dtype=torch.int32)
    torch.testing.assert_close(cu_q, expected)
    torch.testing.assert_close(cu_k, expected)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 3


def test_equal_s2049_global_grad_routes_native_thd(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=2049)
    q.requires_grad_()
    native_calls = []

    def global_native(q_arg, k_arg, v_arg, cu_q, cu_k, **kwargs):
        native_calls.append((q_arg, k_arg, v_arg, cu_q, cu_k, kwargs))
        return q_arg.clone(), torch.zeros(32, q_arg.shape[0], dtype=torch.float32)

    monkeypatch.setattr(integration, "fa4_global_varlen_forward", global_native)
    result = _prepared(5, q, k, v, allow_flex_fallback=False)

    assert result.path == "fa4_global_varlen_native"
    assert len(native_calls) == 1
    q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = native_calls[0]
    assert q_arg.shape == (2049, 32, 512)
    assert k_arg.shape == v_arg.shape == (2049, 4, 512)
    expected = torch.tensor([0, 2049], dtype=torch.int32)
    torch.testing.assert_close(cu_q, expected)
    torch.testing.assert_close(cu_k, expected)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 2049
    assert not any(fast_paths.values())


def test_global_noncontiguous_repeated_documents_fail_closed_to_exact_fallback(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=3)
    v[:, :, 0] = 1
    v[:, :, 1] = 100
    v[:, :, 2] = 3
    documents = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    expected = torch.empty((1, 3, 32, 512), dtype=torch.bfloat16)
    expected[:, 0] = 1
    expected[:, 1] = 100
    expected[:, 2] = 2
    captured = {}

    def fallback(_module, _q, _k, _v, plan, *, scaling):
        scalar = torch.tensor(0)
        query_two = torch.tensor(2)
        captured["q2_k0"] = bool(plan.mask_function(scalar, scalar, query_two, scalar))
        captured["q2_k1"] = bool(plan.mask_function(scalar, scalar, query_two, torch.tensor(1)))
        captured["scaling"] = scaling
        return expected

    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    result = _prepared(5, q, k, v, document_ids=documents)

    assert result.path == "flex_attention"
    assert result.output is expected
    assert captured == {"q2_k0": True, "q2_k1": False, "scaling": 1.0}
    assert not any(fast_paths.values())

    with pytest.raises(UnsupportedH100Path, match="noncontiguous|backward"):
        _prepared(5, q.requires_grad_(), k, v, document_ids=documents)


@pytest.mark.parametrize("value", [torch.tensor(True), torch.tensor(3.75)])
def test_python_int_rejects_bool_and_float_scalar_tensors(value):
    with pytest.raises(TypeError, match="integer, non-bool"):
        integration._python_int(value, name="offset")


def test_cache_or_non_lower_right_offset_uses_exact_fallback(monkeypatch, fast_paths):
    q, k, v = _qkv_bhsd(q_length=1, kv_length=5)

    def mask_function(*_args):
        return True

    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=5,
        q_offset=2,
        mask_function=mask_function,
    )
    expected = torch.full((1, 1, 32, 256), 23, dtype=torch.bfloat16)
    captured = {}

    def fallback(module, q_arg, k_arg, v_arg, received_plan, *, scaling):
        captured.update(plan=received_plan, q=q_arg, k=k_arg, v=v_arg, scaling=scaling)
        return expected

    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    result = _prepared(0, q, k, v, plan, past_key_values=object())

    assert result.output is expected
    assert result.path == "flex_attention"
    assert captured["plan"] is plan
    assert captured["plan"].mask_function is mask_function
    assert (captured["plan"].q_offset, captured["plan"].kv_offset) == (2, 0)
    assert not fast_paths["local"] and not fast_paths["varlen"]


def test_mask_adapter_snapshots_mutable_real_cache_offsets():
    q_offset = torch.tensor(3, dtype=torch.int64)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=q_offset,
    )

    assert isinstance(plan.q_offset, torch.Tensor)
    assert plan.q_offset is not q_offset
    assert plan.q_offset.dtype == q_offset.dtype
    assert plan.q_offset.device == q_offset.device
    assert plan.q_offset.untyped_storage().data_ptr() != q_offset.untyped_storage().data_ptr()
    q_offset.add_(11)
    assert q_offset.item() == 14
    assert plan.q_offset.item() == 3


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
@pytest.mark.parametrize(
    "mask_style",
    ["short", "physical-zero-tail", "physical-all-ones"],
)
def test_static_cache_right_unwritten_prefix_routes_trimmed_storage_views(
    layer_idx,
    mask_style,
    fast_paths,
):
    physical_k = 8
    active_k = 5
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=2, kv_length=physical_k)
    k[:, :, :active_k].fill_(1)
    v[:, :, :active_k].fill_(2)
    k[:, :, active_k:].fill_(127)
    v[:, :, active_k:].fill_(-91)
    attention_mask = {
        "short": torch.ones((1, active_k), dtype=torch.int32),
        "physical-zero-tail": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32),
        "physical-all-ones": torch.ones((1, physical_k), dtype=torch.int32),
    }[mask_style]
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=physical_k,
        q_offset=torch.tensor(3) if layer_idx == 5 else 3,
        attention_mask=attention_mask,
    )

    result = _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)

    if layer_idx == 0:
        assert result.path == "fa4_local_varlen"
        assert len(fast_paths["varlen"]) == 1
        q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = fast_paths["varlen"][0]
        expected_q = q.transpose(1, 2)[0]
        expected_k = k[:, :, :active_k].transpose(1, 2)[0]
        expected_v = v[:, :, :active_k].transpose(1, 2)[0]
        torch.testing.assert_close(cu_q, torch.tensor([0, 2], dtype=torch.int32))
        torch.testing.assert_close(cu_k, torch.tensor([0, active_k], dtype=torch.int32))
        assert kwargs["max_seqlen_q"] == 2
        assert kwargs["max_seqlen_k"] == active_k
        assert not fast_paths["local"] and not fast_paths["global"]
    else:
        assert result.path == "fa4_global_varlen"
        assert len(fast_paths["global"]) == 1
        q_arg, k_arg, v_arg, _kwargs = fast_paths["global"][0]
        assert q_arg.shape == (1, active_k, 32, 512)
        assert torch.count_nonzero(q_arg[:, : active_k - 2]).item() == 0
        expected_q = None
        expected_k = k[:, :, :active_k].transpose(1, 2)[0].unsqueeze(0)
        expected_v = v[:, :, :active_k].transpose(1, 2)[0].unsqueeze(0)
        assert not fast_paths["local"] and not fast_paths["varlen"]

    for original, expected, received in (
        (k, expected_k, k_arg),
        (v, expected_v, v_arg),
    ):
        assert received.shape == expected.shape
        assert received.stride() == expected.stride()
        assert received.storage_offset() == expected.storage_offset()
        assert received.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        torch.testing.assert_close(received, expected)
    if expected_q is not None:
        assert q_arg.stride() == expected_q.stride()
        assert q_arg.storage_offset() == expected_q.storage_offset()
        assert q_arg.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    assert torch.all(k_arg == 1)
    assert torch.all(v_arg == 2)
    assert result.output.shape == (1, 2, 32, q.shape[-1])
    assert result.lse.shape == (1, 32, 2)


@pytest.mark.parametrize(
    ("layer_idx", "expected_path", "backend"),
    [
        (0, "fa4_local_fixed", "local"),
        (5, "fa4_global_fixed", "global"),
    ],
    ids=["local", "global"],
)
def test_static_cache_trim_does_not_reinterpret_position_resets_as_packing(
    layer_idx,
    expected_path,
    backend,
    fast_paths,
):
    active_k = 5
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=active_k, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=active_k,
        kv_length=8,
        q_offset=0,
        attention_mask=torch.ones((1, 8), dtype=torch.int32),
    )

    result = _prepared(
        layer_idx,
        q,
        k,
        v,
        plan,
        position_ids=torch.tensor([[0, 1, 0, 1, 2]], dtype=torch.int64),
        allow_flex_fallback=False,
    )

    assert result.path == expected_path
    assert len(fast_paths[backend]) == 1
    assert not fast_paths["varlen"]
    received_k = fast_paths[backend][0][1]
    assert received_k.shape[1] == active_k
    assert received_k.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_static_cache_hostile_unwritten_tail_is_absent_at_backend_dispatch(
    layer_idx,
    fast_paths,
):
    active_k = 5
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=2, kv_length=8)
    k[:, :, :active_k].fill_(1)
    v[:, :, :active_k].fill_(2)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=torch.tensor(3) if layer_idx == 5 else 3,
        attention_mask=torch.ones((1, active_k), dtype=torch.int32),
    )

    k[:, :, active_k:].fill_(127)
    v[:, :, active_k:].fill_(-91)
    first = _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)
    first_call = fast_paths["varlen"][0] if layer_idx == 0 else fast_paths["global"][0]
    first_k = first_call[1].clone()
    first_v = first_call[2].clone()

    k[:, :, active_k:].fill_(-63)
    v[:, :, active_k:].fill_(119)
    second = _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)
    second_call = fast_paths["varlen"][1] if layer_idx == 0 else fast_paths["global"][1]
    second_k = second_call[1].clone()
    second_v = second_call[2].clone()

    torch.testing.assert_close(first_k, second_k, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first_v, second_v, atol=0.0, rtol=0.0)
    assert torch.all(first_k == 1) and torch.all(first_v == 2)
    torch.testing.assert_close(first.output, second.output, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.lse, second.lse, atol=0.0, rtol=0.0)


def test_static_cache_all_negative_vision_metadata_is_text_only(fast_paths):
    q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=3,
        attention_mask=torch.ones((1, 5), dtype=torch.int32),
    )

    result = _prepared(
        0,
        q,
        k,
        v,
        plan,
        vision_block_ids=torch.full((1, 8), -1, dtype=torch.int32),
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_local_varlen"
    assert fast_paths["varlen"][0][5]["vision_block_ids"] is None


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        (
            {"vision_block_ids": torch.tensor([[0, -1, -1, -1, -1, -1, -1, -1]])},
            "text-only vision metadata",
        ),
        (
            {"document_ids": torch.zeros((1, 8), dtype=torch.int32)},
            "document metadata",
        ),
    ],
    ids=["active-vision", "documents"],
)
def test_static_cache_underfilled_prefix_rejects_deferred_metadata(
    metadata,
    error,
    fast_paths,
):
    q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=3,
        attention_mask=torch.ones((1, 8), dtype=torch.int32),
    )

    with pytest.raises(UnsupportedH100Path, match=error):
        _prepared(0, q, k, v, plan, allow_flex_fallback=False, **metadata)

    assert not any(fast_paths.values())


def test_static_cache_underfilled_prefix_rejects_nonzero_kv_offset(fast_paths):
    q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=4,
        kv_offset=1,
        attention_mask=torch.ones((1, 9), dtype=torch.int32),
    )

    with pytest.raises(UnsupportedH100Path, match="physical K offset zero|fallback"):
        _prepared(0, q, k, v, plan, allow_flex_fallback=False)

    assert not any(fast_paths.values())


def test_static_cache_native_failure_falls_back_with_original_physical_inputs(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=3,
        attention_mask=torch.ones((1, 5), dtype=torch.int32),
    )
    expected = torch.full((1, 2, 32, 256), 43, dtype=torch.bfloat16)
    fallback_calls = []

    def native_failure(*_args, **_kwargs):
        raise UnsupportedH100Path("forced static-prefix backend rejection")

    def fallback(module, q_arg, k_arg, v_arg, received_plan, *, scaling):
        fallback_calls.append((module, q_arg, k_arg, v_arg, received_plan, scaling))
        return expected

    monkeypatch.setattr(integration, "fa4_local_varlen_forward", native_failure)
    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    with pytest.warns(RuntimeWarning, match="forced static-prefix"):
        result = _prepared(0, q, k, v, plan)

    assert result.path == "flex_attention" and result.output is expected
    assert len(fallback_calls) == 1
    _, received_q, received_k, received_v, received_plan, scaling = fallback_calls[0]
    assert received_q is q and received_k is k and received_v is v
    assert received_plan is plan and received_plan.kv_length == 8
    assert scaling == 1.0
    assert not any(fast_paths.values())


def test_static_cache_rolled_local_window_keeps_the_full_physical_view(fast_paths):
    q, k, v = _qkv_bhsd(q_length=1, kv_length=1024)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=1024,
        q_offset=1024,
        kv_offset=1,
        attention_mask=torch.ones((1, 1025), dtype=torch.int32),
    )

    result = _prepared(0, q, k, v, plan, allow_flex_fallback=False)

    assert result.path == "fa4_local_varlen"
    assert len(fast_paths["varlen"]) == 1
    q_arg, k_arg, v_arg, cu_q, cu_k, kwargs = fast_paths["varlen"][0]
    torch.testing.assert_close(cu_q, torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(cu_k, torch.tensor([0, 1024], dtype=torch.int32))
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 1024
    for original, received in ((k, k_arg), (v, v_arg)):
        expected = original.transpose(1, 2)[0]
        assert received.shape == expected.shape
        assert received.stride() == expected.stride()
        assert received.storage_offset() == expected.storage_offset()
        assert received.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert q_arg.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()


def test_static_cache_underfilled_batched_capacity_remains_fallback_only(
    monkeypatch,
    fast_paths,
):
    q, k, v = _qkv_bhsd(batch=2, q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=2,
        q_length=2,
        kv_length=8,
        q_offset=3,
        attention_mask=torch.ones((2, 5), dtype=torch.int32),
    )
    expected = torch.zeros((2, 2, 32, 256), dtype=torch.bfloat16)
    fallback_calls = []

    def fallback(module, q_arg, k_arg, v_arg, received_plan, *, scaling):
        fallback_calls.append((q_arg, k_arg, v_arg, received_plan, scaling))
        return expected

    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    with pytest.warns(RuntimeWarning, match="B1"):
        result = _prepared(0, q, k, v, plan)

    assert result.path == "flex_attention" and result.output is expected
    assert len(fallback_calls) == 1
    received_q, received_k, received_v, received_plan, scaling = fallback_calls[0]
    assert received_q is q and received_k is k and received_v is v
    assert received_plan is plan and scaling == 1.0
    assert not any(fast_paths.values())


@pytest.mark.parametrize(
    "packed_kwargs",
    [
        {"max_length_k": 5},
        {
            "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 8], dtype=torch.int32),
        },
    ],
    ids=["explicit-max", "explicit-cu"],
)
def test_static_cache_underfilled_capacity_rejects_explicit_packed_metadata(
    packed_kwargs,
    fast_paths,
):
    q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=3,
        attention_mask=torch.ones((1, 5), dtype=torch.int32),
    )

    with pytest.raises(UnsupportedH100Path, match="explicit packed lengths|fallback"):
        _prepared(0, q, k, v, plan, allow_flex_fallback=False, **packed_kwargs)

    assert not any(fast_paths.values())


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_all_valid_dynamic_lower_right_inputs_remain_untrimmed(layer_idx, fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=6,
        attention_mask=torch.ones((1, 8), dtype=torch.int32),
    )

    result = _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)

    if layer_idx == 0:
        assert result.path == "fa4_local_varlen"
        k_arg, v_arg = fast_paths["varlen"][0][1:3]
        assert not fast_paths["global"]
    else:
        assert result.path == "fa4_global_varlen"
        k_arg, v_arg = fast_paths["global"][0][1:3]
        assert not fast_paths["varlen"]
    for original, received in ((k, k_arg), (v, v_arg)):
        assert received.shape[-3 if layer_idx == 5 else 0] == 8
        assert received.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
@pytest.mark.parametrize(
    ("case", "mask_values", "q_offset"),
    [
        ("left", [0, 1, 1, 1, 1, 0, 0, 0], 3),
        ("holey", [1, 1, 0, 1, 1, 0, 0, 0], 3),
        ("all-masked", [0, 0, 0, 0, 0, 0, 0, 0], 3),
        ("mask-offset-disagreement", [1, 1, 1, 1, 0, 0, 0, 0], 3),
        ("out-of-range", [1, 1, 1, 1, 1, 1, 1, 1], 7),
    ],
    ids=["left", "holey", "all-masked", "mask-offset-disagreement", "out-of-range"],
)
def test_invalid_static_cache_prefix_falls_back_and_disabled_fallback_fails_closed(
    monkeypatch,
    fast_paths,
    layer_idx,
    case,
    mask_values,
    q_offset,
):
    del case
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=2, kv_length=8)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=torch.tensor(q_offset) if layer_idx == 5 else q_offset,
        attention_mask=torch.tensor([mask_values], dtype=torch.int32),
    )
    expected = torch.full((1, 2, 32, q.shape[-1]), 31, dtype=torch.bfloat16)
    fallback_calls = []

    def fallback(module, q_arg, k_arg, v_arg, received_plan, *, scaling):
        fallback_calls.append((module, q_arg, k_arg, v_arg, received_plan, scaling))
        return expected

    monkeypatch.setattr(integration, "_run_flex_fallback", fallback)
    with pytest.warns(RuntimeWarning, match="StaticCache"):
        result = _prepared(layer_idx, q, k, v, plan)

    assert result.path == "flex_attention"
    assert result.output is expected
    assert len(fallback_calls) == 1
    assert fallback_calls[0][4] is plan
    with pytest.raises(UnsupportedH100Path, match="fallback is disabled"):
        _prepared(layer_idx, q, k, v, plan, allow_flex_fallback=False)
    assert len(fallback_calls) == 1
    assert not any(fast_paths.values())


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["local", "global"])
def test_grad_enabled_static_cache_prefix_rejects_before_native_or_flex_dispatch(
    monkeypatch,
    fast_paths,
    layer_idx,
):
    q, k, v = _qkv_bhsd(layer_idx=layer_idx, q_length=2, kv_length=8)
    q.requires_grad_()
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=8,
        q_offset=torch.tensor(3) if layer_idx == 5 else 3,
        attention_mask=torch.ones((1, 5), dtype=torch.int32),
    )
    fallback_calls = []

    monkeypatch.setattr(
        integration,
        "_run_flex_fallback",
        lambda *_args, **_kwargs: fallback_calls.append(True),
    )
    with pytest.raises(UnsupportedH100Path, match="StaticCache|inference|grad|autograd|backward"):
        _prepared(layer_idx, q, k, v, plan)
    assert not fallback_calls
    assert not any(fast_paths.values())


def test_static_prefix_framework_fake_tensor_still_fails_closed(fast_paths):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q, k, v = _qkv_bhsd(q_length=2, kv_length=8)
        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=2,
            kv_length=8,
            q_offset=torch.tensor(3),
            attention_mask=torch.ones((1, 5), dtype=torch.int32),
        )
        with pytest.raises(UnsupportedH100Path, match="FakeTensor|compile|tracing"):
            _prepared(0, q, k, v, plan)

    assert not any(fast_paths.values())


def test_fallback_can_be_disabled_for_unsupported_global_shape(fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=1, kv_length=1025)
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=1,
        kv_length=1025,
        q_offset=1024,
    )

    with pytest.raises(UnsupportedH100Path, match="fallback|global|1024"):
        _prepared(5, q, k, v, plan, allow_flex_fallback=False)

    assert not fast_paths["global"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"scaling": 0.0625}, "scale|scaling|1.0"),
        ({"dropout": 0.1}, "dropout|zero"),
        ({"output_attentions": True}, "output_attentions|attention weights"),
    ],
)
def test_invalid_attention_semantics_reject_before_dispatch(kwargs, error, fast_paths):
    q, k, v = _qkv_bhsd(q_length=2)

    with pytest.raises((TypeError, ValueError, UnsupportedH100Path), match=error):
        _prepared(0, q, k, v, **kwargs)

    assert not any(fast_paths.values())


def test_prepared_k_and_v_must_not_alias(fast_paths):
    q, k, _ = _qkv_bhsd(q_length=2)

    with pytest.raises(ValueError, match="distinct|alias"):
        _prepared(0, q, k, k)

    assert not any(fast_paths.values())


def test_nonunit_last_dimension_stride_rejects_instead_of_copying(fast_paths):
    q, k, v = _qkv_bhsd(q_length=2)
    q_storage = torch.empty(*q.shape[:-1], q.shape[-1] * 2, dtype=torch.bfloat16)
    q_strided = q_storage[..., ::2]
    assert q_strided.shape == q.shape and q_strided.stride(-1) == 2

    with pytest.raises(ValueError, match="last.*stride|layout|contiguous"):
        _prepared(0, q_strided, k, v)

    assert not any(fast_paths.values())


def test_framework_fake_tensor_tracing_fails_closed(fast_paths):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q, k, v = _qkv_bhsd(q_length=2)
        with pytest.raises(UnsupportedH100Path, match="FakeTensor|compile|tracing"):
            _prepared(0, q, k, v)

    assert not any(fast_paths.values())


def test_locked_layer_index_wins_over_mutable_layer_label(fast_paths):
    q, k, v = _qkv_bhsd(layer_idx=5, q_length=2)
    module = _module(5)
    module.layer_type = "sliding_attention"
    module.is_sliding = True

    result = integration.gemma4_fa4_prepared(
        module,
        q,
        k,
        v,
        None,
        dropout=0.0,
        scaling=1.0,
        sliding_window=None,
        allow_flex_fallback=False,
    )

    assert result.path == "fa4_global_fixed"
    assert len(fast_paths["global"]) == 1
    assert not fast_paths["local"]


@pytest.mark.parametrize("layer_idx", [-1, 60])
def test_layer_index_outside_locked_model_rejects(layer_idx):
    q, k, v = _qkv_bhsd(q_length=1)
    module = SimpleNamespace(layer_idx=layer_idx)

    with pytest.raises((IndexError, ValueError), match=str(abs(layer_idx))):
        integration.gemma4_fa4_prepared(module, q, k, v, None)


class _Registry(dict):
    def __init__(self):
        super().__init__()
        self.register_calls = []

    def register(self, key, value):
        self.register_calls.append((key, value))
        self[key] = value


def _install_fake_transformers_registries(monkeypatch):
    attention = _Registry()
    masks = _Registry()
    package = ModuleType("transformers")
    package.__path__ = []
    modeling_utils = ModuleType("transformers.modeling_utils")
    masking_utils = ModuleType("transformers.masking_utils")
    modeling_utils.ALL_ATTENTION_FUNCTIONS = attention
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS = masks
    package.modeling_utils = modeling_utils
    package.masking_utils = masking_utils
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_utils)
    return attention, masks


@pytest.mark.parametrize("collision", ["attention", "mask"])
def test_registration_rejects_an_existing_different_backend(monkeypatch, collision):
    attention, masks = _install_fake_transformers_registries(monkeypatch)
    target = attention if collision == "attention" else masks
    target[integration.BACKEND_NAME] = lambda: None

    with pytest.raises(RuntimeError, match="already|collision|registered"):
        integration.register_gemma4_fa4_h100()


def test_registration_is_idempotent_for_the_same_callables(monkeypatch):
    attention, masks = _install_fake_transformers_registries(monkeypatch)

    integration.register_gemma4_fa4_h100()
    integration.register_gemma4_fa4_h100()

    assert attention[integration.BACKEND_NAME] is integration.gemma4_fa4_attention_forward
    assert masks[integration.BACKEND_NAME] is integration._registered_gemma4_fa4_mask
    assert attention.register_calls == [
        (integration.BACKEND_NAME, integration.gemma4_fa4_attention_forward)
    ]
    assert masks.register_calls == [
        (integration.BACKEND_NAME, integration._registered_gemma4_fa4_mask)
    ]
