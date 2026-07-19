from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import gemma4_fa4.h100 as h100
from gemma4_fa4.model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention, reference_attention_varlen
from gemma4_fa4.sparse_schedule import SparseTileRows

OUT_ATOL = 0.03125
OUT_RTOL = 0.02
LSE_ATOL = 0.125
GLOBAL_OUT_ATOL = 0.0625
GLOBAL_OUT_RTOL = 0.03
GLOBAL_LSE_ATOL = 0.25


def _cpu_qkv(spec=SLIDING_ATTENTION, seqlen=2):
    q = torch.zeros(1, seqlen, spec.num_q_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    k = torch.ones(1, seqlen, spec.num_kv_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    v = torch.full(
        (1, seqlen, spec.num_kv_heads, spec.head_dim_v),
        2,
        dtype=torch.bfloat16,
    )
    return q, k, v


def _cpu_varlen_qkv(q_lengths=(2, 1), k_lengths=(3, 2)):
    spec = SLIDING_ATTENTION
    q = torch.zeros(sum(q_lengths), spec.num_q_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    k = torch.ones(sum(k_lengths), spec.num_kv_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    v = torch.full_like(k, 2)
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def test_local_adapter_fixes_every_semantic_keyword(monkeypatch):
    q, k, v = _cpu_qkv()
    captured = {}

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v
        captured.update(kwargs)
        return q.clone(), torch.zeros(1, 32, 2, dtype=torch.float32)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    out, lse = h100.fa4_local_text_forward(q, k, v)

    assert out.shape == q.shape
    assert lse.shape == (1, 32, 2)
    assert captured == {
        "causal": True,
        "window_size": (1023, 0),
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "return_lse": True,
    }


def test_local_multimodal_adapter_uses_one_complete_custom_mask(monkeypatch):
    q, k, v = _cpu_qkv(seqlen=4)
    vision_ids = torch.tensor([[-1, 0, 0, -1]], dtype=torch.int64)
    captured = {}
    mask_sentinel = object()

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v
        captured.update(kwargs)
        return q.clone(), torch.zeros(1, 32, 4, dtype=torch.float32)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    monkeypatch.setattr(h100, "_load_local_vision_mask", lambda: mask_sentinel)

    out, lse = h100.fa4_local_forward(q, k, v, vision_block_ids=vision_ids)

    assert out.shape == q.shape
    assert lse.shape == (1, 32, 4)
    aux_vision_ids = captured.pop("aux_tensors")[0]
    assert aux_vision_ids.dtype == torch.int32
    assert aux_vision_ids.is_contiguous()
    torch.testing.assert_close(aux_vision_ids, vision_ids.to(torch.int32).view(-1))
    assert captured == {
        "causal": False,
        "window_size": (None, None),
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "mask_mod": mask_sentinel,
        "return_lse": True,
    }


def test_local_multimodal_adapter_rejects_invalid_vision_ids(monkeypatch):
    q, k, v = _cpu_qkv(seqlen=4)
    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)

    with pytest.raises(ValueError, match="shape"):
        h100.fa4_local_forward(
            q,
            k,
            v,
            vision_block_ids=torch.zeros(1, 3, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="INT32 or INT64"):
        h100.fa4_local_forward(
            q,
            k,
            v,
            vision_block_ids=torch.zeros(1, 4, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="same device"):
        h100.fa4_local_forward(
            q,
            k,
            v,
            vision_block_ids=torch.empty(1, 4, dtype=torch.int32, device="meta"),
        )
    for value in (torch.iinfo(torch.int32).min - 1, torch.iinfo(torch.int32).max + 1):
        with pytest.raises(ValueError, match="fit exactly in INT32"):
            h100.fa4_local_forward(
                q,
                k,
                v,
                vision_block_ids=torch.full((1, 4), value, dtype=torch.int64),
            )


def test_local_varlen_text_adapter_fixes_native_semantics(monkeypatch):
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv()
    captured = {}

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v
        captured.update(kwargs)
        return q.clone(), torch.zeros(32, q.shape[0], dtype=torch.float32)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_varlen_func", lambda: fake_backend)
    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=2,
        max_seqlen_k=3,
    )

    assert out.shape == q.shape
    assert lse.shape == (32, 3)
    assert captured == {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": 2,
        "max_seqlen_k": 3,
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "deterministic": False,
        "return_lse": True,
        "causal": True,
        "window_size": (1023, 0),
    }


def test_local_varlen_custom_adapter_owns_complete_predicate(monkeypatch):
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv()
    vision_ids = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.int64)
    captured = {}
    mask_sentinel = object()

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k and v_arg is v
        captured.update(kwargs)
        return q.clone(), torch.zeros(32, q.shape[0], dtype=torch.float32)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_varlen_func", lambda: fake_backend)
    monkeypatch.setattr(h100, "_load_local_varlen_mask", lambda: mask_sentinel)
    h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=2,
        max_seqlen_k=3,
        vision_block_ids=vision_ids,
    )

    aux_vision, aux_documents = captured.pop("aux_tensors")
    torch.testing.assert_close(aux_vision, vision_ids.to(torch.int32))
    assert aux_vision.dtype == torch.int32 and aux_vision.is_contiguous()
    assert torch.count_nonzero(aux_documents) == 0
    assert captured == {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": 2,
        "max_seqlen_k": 3,
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "deterministic": False,
        "return_lse": True,
        "causal": False,
        "window_size": (None, None),
        "mask_mod": mask_sentinel,
    }


def test_local_varlen_native_validation_admits_locked_model_maximum():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = SLIDING_ATTENTION
    with FakeTensorMode():
        q = torch.empty(1, spec.num_q_heads, spec.head_dim_qk, dtype=torch.bfloat16)
        k = torch.empty(
            262_144,
            spec.num_kv_heads,
            spec.head_dim_qk,
            dtype=torch.bfloat16,
        )
        v = torch.empty_like(k)
        cu_q = torch.empty(2, dtype=torch.int32)
        cu_k = torch.empty(2, dtype=torch.int32)
        assert h100._validate_local_varlen(
            q,
            k,
            v,
            cu_q,
            cu_k,
            1,
            262_144,
            spec,
            require_sm90=False,
        ) == (None, None)


def test_local_varlen_rejects_lengths_above_locked_model_maximum():
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = SLIDING_ATTENTION
    with FakeTensorMode():
        q = torch.empty(1, spec.num_q_heads, spec.head_dim_qk, dtype=torch.bfloat16)
        k = torch.empty(
            262_145,
            spec.num_kv_heads,
            spec.head_dim_qk,
            dtype=torch.bfloat16,
        )
        v = torch.empty_like(k)
        cu_q = torch.empty(2, dtype=torch.int32)
        cu_k = torch.empty(2, dtype=torch.int32)
        with pytest.raises(h100.UnsupportedH100Path, match="262144"):
            h100._validate_local_varlen(
                q,
                k,
                v,
                cu_q,
                cu_k,
                1,
                262_145,
                spec,
                require_sm90=False,
            )


@pytest.mark.parametrize("metadata_name", ["vision_block_ids", "document_ids"])
def test_local_varlen_long_metadata_uses_sparse_per_sequence(monkeypatch, metadata_name):
    q_lengths, k_lengths = (1, 2), (1026, 3)
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv(q_lengths, k_lengths)
    metadata = torch.full((sum(k_lengths),), -1, dtype=torch.int32)
    if metadata_name == "document_ids":
        metadata.zero_()
    calls = []
    mask_sentinel = object()

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        calls.append((q_arg, k_arg, v_arg, kwargs))
        return q_arg.clone(), torch.zeros(1, 32, q_arg.shape[1], dtype=torch.float32)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    monkeypatch.setattr(h100, "_load_local_segment_mask", lambda: mask_sentinel)
    monkeypatch.setattr(
        h100,
        "_load_block_sparse_tensors_type",
        lambda: lambda **kwargs: SimpleNamespace(**kwargs),
    )

    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        **{metadata_name: metadata},
    )

    assert out.shape == q.shape
    assert lse.shape == (32, sum(q_lengths))
    assert len(calls) == len(q_lengths)
    for index, (q_arg, k_arg, v_arg, kwargs) in enumerate(calls):
        assert q_arg.shape == (1, q_lengths[index], 32, 256)
        assert k_arg.shape == v_arg.shape == (1, k_lengths[index], 16, 256)
        sparse_fwd = kwargs.pop("block_sparse_tensors")
        sparse_bwd = kwargs.pop("block_sparse_tensors_bwd")
        aux_vision, aux_documents = kwargs.pop("aux_tensors")
        assert aux_vision.shape == aux_documents.shape == (k_lengths[index],)
        assert aux_vision.dtype == aux_documents.dtype == torch.int32
        assert sparse_fwd.block_size == (128, 80)
        assert sparse_bwd.block_size == (64, 64)
        assert sparse_fwd.mask_block_cnt.ndim == sparse_bwd.mask_block_cnt.ndim == 3
        assert sparse_fwd.mask_block_idx.ndim == sparse_bwd.mask_block_idx.ndim == 4
        assert torch.all(sparse_fwd.mask_block_cnt <= sparse_fwd.mask_block_idx.shape[-1])
        assert torch.all(sparse_bwd.mask_block_cnt <= sparse_bwd.mask_block_idx.shape[-1])
        for sparse in (sparse_fwd, sparse_bwd):
            assert sparse.full_block_cnt.shape == sparse.mask_block_cnt.shape
            assert sparse.full_block_idx.shape == (*sparse.mask_block_idx.shape[:-1], 1)
            assert torch.count_nonzero(sparse.full_block_cnt).item() == 0
            assert torch.count_nonzero(sparse.full_block_idx).item() == 0
        assert kwargs == {
            "causal": False,
            "window_size": (None, None),
            "softmax_scale": 1.0,
            "num_splits": 1,
            "pack_gqa": False,
            "deterministic": False,
            "mask_mod": mask_sentinel,
            "return_lse": True,
        }


def test_local_sparse_metadata_rejects_above_absolute_limit():
    with pytest.raises(h100.UnsupportedH100Path, match="above the 2 GiB safety limit"):
        h100._check_local_sparse_metadata_budget(
            2 * 1024**3 + 1,
            torch.device("cpu"),
            label="test sparse metadata",
        )


def test_local_sparse_metadata_rejects_above_free_hbm_fraction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1000, 2000))
    with pytest.raises(h100.UnsupportedH100Path, match="10% free-HBM"):
        h100._check_local_sparse_metadata_budget(
            101,
            torch.device("cuda"),
            label="test sparse metadata",
        )


def test_local_varlen_sparse_work_limit_is_public_error(monkeypatch):
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv((1,), (1026,))
    metadata = torch.full((1026,), -1, dtype=torch.int32)

    def exhaust_work(*_args, **_kwargs):
        raise h100.SparseScheduleWorkLimitExceeded("test construction budget")

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "build_gemma4_local_sparse_schedule", exhaust_work)

    with pytest.raises(h100.UnsupportedH100Path, match="padded-score work safety limit"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=1026,
            vision_block_ids=metadata,
        )


@pytest.mark.parametrize(
    ("cu_q", "cu_k", "error"),
    [
        (torch.tensor([1, 2, 3], dtype=torch.int32), None, "start at zero"),
        (torch.tensor([0, 0, 3], dtype=torch.int32), None, "empty packed segments"),
        (torch.tensor([0, 2, 4], dtype=torch.int32), None, "packed Q/K totals"),
        (torch.tensor([0, 2, 3], dtype=torch.int64), None, "must use INT32"),
        (torch.tensor([0, 3], dtype=torch.int32), None, "same batch count"),
        (
            torch.tensor([0, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 5], dtype=torch.int32),
            "each packed sequence requires Sq <= Sk",
        ),
    ],
)
def test_local_varlen_rejects_malformed_cumulative_arrays(monkeypatch, cu_q, cu_k, error):
    q, k, v, valid_q, valid_k = _cpu_varlen_qkv()
    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    with pytest.raises((ValueError, h100.UnsupportedH100Path), match=error):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            valid_q if cu_q is None else cu_q,
            valid_k if cu_k is None else cu_k,
            max_seqlen_q=2,
            max_seqlen_k=4 if error.startswith("each") else 3,
        )


def test_local_varlen_rejects_wrong_maxima_and_metadata(monkeypatch):
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv()
    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    with pytest.raises(ValueError, match="must equal the cumulative maxima"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=1,
            max_seqlen_k=3,
        )
    with pytest.raises(ValueError, match="shape"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=2,
            max_seqlen_k=3,
            vision_block_ids=torch.zeros(k.shape[0] - 1, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="INT32 or INT64"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=2,
            max_seqlen_k=3,
            document_ids=torch.zeros(k.shape[0], dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="fit exactly in INT32"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=2,
            max_seqlen_k=3,
            vision_block_ids=torch.full(
                (k.shape[0],),
                torch.iinfo(torch.int32).max + 1,
                dtype=torch.int64,
            ),
        )


def test_local_varlen_rejects_mixed_real_and_fake_metadata(monkeypatch):
    q, k, v, cu_q, cu_k = _cpu_varlen_qkv()
    vision_ids = torch.zeros(k.shape[0], dtype=torch.int32)
    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_is_fake_tensor", lambda tensor: tensor is vision_ids)

    with pytest.raises(ValueError, match="share real/fake tensor mode"):
        h100.fa4_local_varlen_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=2,
            max_seqlen_k=3,
            vision_block_ids=vision_ids,
        )


def test_local_adapter_rejects_kv_aliasing_and_wrong_contracts():
    q, k, v = _cpu_qkv()
    with pytest.raises(ValueError, match="distinct, non-aliasing"):
        h100._validate_local_bshd(q, k, k, SLIDING_ATTENTION, require_sm90=False)
    with pytest.raises(ValueError, match="BF16"):
        h100._validate_local_bshd(
            q.float(), k.float(), v.float(), SLIDING_ATTENTION, require_sm90=False
        )
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    q_view = q_bhsd.transpose(1, 2)
    k_view = k_bhsd.transpose(1, 2)
    v_view = v_bhsd.transpose(1, 2)
    assert not q_view.is_contiguous()
    h100._validate_local_bshd(
        q_view,
        k_view,
        v_view,
        SLIDING_ATTENTION,
        require_sm90=False,
    )
    q_strided_storage = torch.empty(*q.shape[:-1], q.shape[-1] * 2, dtype=q.dtype)
    q_strided = q_strided_storage[..., ::2]
    with pytest.raises(ValueError, match="last-dimension stride"):
        h100._validate_local_bshd(q_strided, k, v, SLIDING_ATTENTION, require_sm90=False)
    q_storage = torch.empty(q.numel() + 1, dtype=q.dtype)
    q_misaligned = q_storage[1:].view_as(q)
    assert q_misaligned.is_contiguous() and q_misaligned.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte aligned"):
        h100._validate_local_bshd(q_misaligned, k, v, SLIDING_ATTENTION, require_sm90=False)
    with pytest.raises(h100.UnsupportedH100Path, match="local d256"):
        h100._validate_local_bshd(
            q,
            k,
            v,
            replace(SLIDING_ATTENTION, softmax_scale=0.5),
            require_sm90=False,
        )


def test_adapter_validation_supports_all_fake_inputs_without_pointer_checks():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q, k, v = _cpu_qkv()
        h100._validate_local_bshd(q, k, v, SLIDING_ATTENTION, require_sm90=False)


def test_local_adapter_rejects_shapes_outside_the_proven_envelope():
    q, k, v = _cpu_qkv(seqlen=2)
    with pytest.raises(h100.UnsupportedH100Path, match="B=1"):
        h100._validate_local_bshd(
            q.expand(2, -1, -1, -1).contiguous(),
            k.expand(2, -1, -1, -1).contiguous(),
            v.expand(2, -1, -1, -1).contiguous(),
            SLIDING_ATTENTION,
            require_sm90=False,
        )
    q, k, v = _cpu_qkv(SLIDING_ATTENTION, seqlen=1026)
    with pytest.raises(h100.UnsupportedH100Path, match="S <= 1025"):
        h100._validate_local_bshd(q, k, v, SLIDING_ATTENTION, require_sm90=False)
    one_head_spec = replace(SLIDING_ATTENTION, num_q_heads=1, num_kv_heads=1)
    q, k, v = _cpu_qkv(one_head_spec)
    with pytest.raises(h100.UnsupportedH100Path, match="local d256"):
        h100._validate_local_bshd(q, k, v, one_head_spec, require_sm90=False)


def test_global_adapter_uses_two_exact_v_slabs(monkeypatch):
    q, k, v = _cpu_qkv(GLOBAL_ATTENTION)
    v[..., 256:] = 3
    calls = []

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        assert q_arg is q and k_arg is k
        calls.append((v_arg.clone(), kwargs))
        out_shape = (*q_arg.shape[:-1], v_arg.shape[-1])
        out = torch.full(out_shape, len(calls), dtype=torch.bfloat16)
        lse = torch.zeros(1, 32, q_arg.shape[1], dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    out, lse = h100.fa4_global_text_forward(q, k, v)

    assert out.shape == q.shape
    assert lse.shape == (1, 32, 2)
    assert len(calls) == 2
    assert torch.all(calls[0][0] == 2)
    assert torch.all(calls[1][0] == 3)
    assert torch.all(out[..., :256] == 1)
    assert torch.all(out[..., 256:] == 2)
    expected_kwargs = {
        "causal": True,
        "window_size": (None, None),
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "return_lse": True,
    }
    assert calls[0][1] == expected_kwargs
    assert calls[1][1] == expected_kwargs


def test_global_adapter_coordinates_one_full_precision_backward(monkeypatch):
    q, k, v = [tensor.requires_grad_() for tensor in _cpu_qkv(GLOBAL_ATTENTION)]
    backward_calls = []

    def fake_backend(q_arg, _k_arg, v_arg, **_kwargs):
        out = torch.zeros((*q_arg.shape[:-1], v_arg.shape[-1]), dtype=torch.bfloat16)
        lse = torch.zeros(1, 32, q_arg.shape[1], dtype=torch.float32)
        return out, lse

    def fake_backward(q_arg, k_arg, v_arg, out_arg, dout_arg, lse_arg, dlse_arg):
        backward_calls.append((out_arg.shape, dout_arg.shape, lse_arg.shape, dlse_arg))
        return torch.ones_like(q_arg), torch.full_like(k_arg, 2), torch.full_like(v_arg, 3)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    monkeypatch.setattr(h100, "_load_global_backward_func", lambda: fake_backward)

    out, _ = h100.fa4_global_text_forward(q, k, v)
    out.sum().backward()

    assert backward_calls == [((1, 2, 32, 512), (1, 2, 32, 512), (1, 32, 2), None)]
    assert torch.all(q.grad == 1)
    assert torch.all(k.grad == 2)
    assert torch.all(v.grad == 3)


def test_global_backward_preflight_runs_before_forward_backend(monkeypatch):
    q, k, v = [tensor.requires_grad_() for tensor in _cpu_qkv(GLOBAL_ATTENTION)]
    backend_called = False

    def fake_backend(*_args, **_kwargs):
        nonlocal backend_called
        backend_called = True
        raise AssertionError("backend must not run after a failed preflight")

    def reject_preflight(_q):
        raise h100.UnsupportedH100Path("synthetic global backward budget rejection")

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_preflight_global_backward", reject_preflight)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)

    with pytest.raises(h100.UnsupportedH100Path, match="budget rejection"):
        h100.fa4_global_text_forward(q, k, v)
    assert not backend_called


def test_global_adapter_forwards_lse_gradient_and_materializes_zero_dout(monkeypatch):
    q, k, v = [tensor.requires_grad_() for tensor in _cpu_qkv(GLOBAL_ATTENTION)]
    backward_calls = []

    def fake_backend(q_arg, _k_arg, v_arg, **_kwargs):
        out = torch.zeros((*q_arg.shape[:-1], v_arg.shape[-1]), dtype=torch.bfloat16)
        lse = torch.zeros(1, 32, q_arg.shape[1], dtype=torch.float32)
        return out, lse

    def fake_backward(q_arg, k_arg, v_arg, _out, dout, _lse, dlse):
        backward_calls.append((dout.clone(), dlse.clone()))
        return torch.ones_like(q_arg), torch.ones_like(k_arg), torch.ones_like(v_arg)

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    monkeypatch.setattr(h100, "_load_global_backward_func", lambda: fake_backward)

    out, lse = h100.fa4_global_text_forward(q, k, v)
    assert out.grad_fn is not None
    lse.sum().backward()

    assert len(backward_calls) == 1
    dout, dlse = backward_calls[0]
    assert torch.count_nonzero(dout) == 0
    assert torch.all(dlse == 1)


def test_global_adapter_rejects_alias_and_wrong_contract():
    q, k, v = _cpu_qkv(GLOBAL_ATTENTION)
    with pytest.raises(ValueError, match="distinct, non-aliasing"):
        h100._validate_global_bshd(q, k, k, GLOBAL_ATTENTION, require_sm90=False)
    with pytest.raises(h100.UnsupportedH100Path, match="global d512"):
        h100._validate_global_bshd(
            q,
            k,
            v,
            replace(GLOBAL_ATTENTION, softmax_scale=0.5),
            require_sm90=False,
        )


def test_global_backward_workspace_budget_is_bounded_and_fail_closed():
    seqlen = 1025
    expected_workspace = 147456 * seqlen + 66048 * 1088 + 16384 * 1056
    assert h100._global_backward_workspace_bytes(seqlen) == expected_workspace

    q = torch.empty((1, seqlen, 32, 512), dtype=torch.bfloat16, device="meta")
    output_bytes = q.numel() * q.element_size()
    lse_bytes = q.numel() // q.shape[-1] * torch.float32.itemsize
    assert h100._global_backward_additional_bytes(q) == (
        expected_workspace + 2 * (output_bytes + lse_bytes)
    )

    gib = 1024**3
    assert h100._global_backward_budget(10 * gib) == 8 * gib
    assert h100._global_backward_budget(3 * gib) == 1 * gib
    h100._check_global_backward_budget(8 * gib, 10 * gib)
    with pytest.raises(h100.UnsupportedH100Path, match="additional bytes|guarded budget"):
        h100._check_global_backward_budget(8 * gib + 1, 10 * gib)


def test_global_backward_validation_stops_after_exp0012_envelope(monkeypatch):
    monkeypatch.setattr(h100, "_is_fake_tensor", lambda _tensor: True)
    q = torch.empty((1, 2049, 32, 512), dtype=torch.bfloat16, device="meta")
    k = torch.empty((1, 2049, 4, 512), dtype=torch.bfloat16, device="meta")
    v = torch.empty_like(k)

    with pytest.raises(h100.UnsupportedH100Path, match="1 <= S <= 2048"):
        h100._validate_global_bshd(q, k, v, GLOBAL_ATTENTION, require_sm90=False)
    with pytest.raises(h100.UnsupportedH100Path, match="B=1"):
        h100._validate_global_bshd(
            q.expand(2, -1, -1, -1).contiguous(),
            k.expand(2, -1, -1, -1).contiguous(),
            v.expand(2, -1, -1, -1).contiguous(),
            GLOBAL_ATTENTION,
            require_sm90=False,
        )


def test_global_forward_only_supports_batched_lower_right_and_rejects_autograd(monkeypatch):
    q = torch.zeros(2, 3, 32, 512, dtype=torch.bfloat16)
    k = torch.ones(2, 5, 4, 512, dtype=torch.bfloat16)
    v = torch.full((2, 5, 4, 512), 2, dtype=torch.bfloat16)
    calls = []

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        calls.append((q_arg, k_arg, v_arg.clone(), kwargs))
        output = torch.full((*q_arg.shape[:-1], 256), len(calls), dtype=torch.bfloat16)
        lse = torch.zeros(2, 32, 3, dtype=torch.float32)
        return output, lse

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_preflight_global_forward_outputs", lambda _q: None)
    monkeypatch.setattr(h100, "_load_flash_attn_func", lambda: fake_backend)
    with torch.no_grad():
        output, lse = h100.fa4_global_forward_only(q, k, v)

    assert output.shape == q.shape
    assert lse.shape == (2, 32, 3)
    assert len(calls) == 2
    assert all(call[0] is q and call[1] is k for call in calls)
    assert all(call[3]["causal"] and call[3]["softmax_scale"] == 1.0 for call in calls)

    with pytest.raises(h100.UnsupportedH100Path, match="autograd"):
        h100.fa4_global_forward_only(q.requires_grad_(), k, v)


def test_global_varlen_forward_only_uses_two_exact_slabs(monkeypatch):
    q = torch.zeros(4, 32, 512, dtype=torch.bfloat16)
    k = torch.ones(8, 4, 512, dtype=torch.bfloat16)
    v = torch.full((8, 4, 512), 2, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 3, 4], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 8], dtype=torch.int32)
    calls = []

    def fake_backend(q_arg, k_arg, v_arg, **kwargs):
        calls.append((q_arg, k_arg, v_arg.clone(), kwargs))
        output = torch.full((*q_arg.shape[:-1], 256), len(calls), dtype=torch.bfloat16)
        lse = torch.zeros(32, 4, dtype=torch.float32)
        return output, lse

    monkeypatch.setattr(h100, "_require_sm90", lambda _device: None)
    monkeypatch.setattr(h100, "_preflight_global_forward_outputs", lambda _q: None)
    monkeypatch.setattr(h100, "_load_flash_attn_varlen_func", lambda: fake_backend)
    with torch.no_grad():
        output, lse = h100.fa4_global_varlen_forward_only(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=3,
            max_seqlen_k=5,
        )

    assert output.shape == q.shape
    assert lse.shape == (32, 4)
    assert len(calls) == 2
    assert all(call[3]["causal"] and call[3]["max_seqlen_k"] == 5 for call in calls)


def _has_h100_fa4() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        from flash_attn.cute import flash_attn_func as _  # noqa: F401
    except Exception:
        return False
    return True


def _fake_tensor_mode_if_requested(fn):
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1":
        return fn
    from flash_attn.cute.testing import maybe_fake_tensor_mode

    return maybe_fake_tensor_mode(True)(fn)


H100_FA4 = pytest.mark.skipif(
    not _has_h100_fa4() or os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") == "1",
    reason="requires real pinned FA4 execution on H100",
)
H100_FA4_FAKE = pytest.mark.skipif(
    not _has_h100_fa4() or os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1",
    reason="requires FLASH_ATTENTION_FAKE_TENSOR=1 on H100",
)


def _gpu_qkv(spec, seqlen, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        1,
        seqlen,
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn(
        1,
        seqlen,
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v = torch.randn(
        1,
        seqlen,
        spec.num_kv_heads,
        spec.head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return q, k, v


def _gpu_varlen_qkv(q_lengths, k_lengths, seed, *, requires_grad=False):
    spec = SLIDING_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(q_lengths),
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    q_cumulative = [0]
    k_cumulative = [0]
    for q_len in q_lengths:
        q_cumulative.append(q_cumulative[-1] + q_len)
    for k_len in k_lengths:
        k_cumulative.append(k_cumulative[-1] + k_len)
    cu_q = torch.tensor(q_cumulative, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_cumulative, device="cuda", dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _packed_k_ids(k_lengths, pattern):
    ids = torch.full((sum(k_lengths),), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for seqlen in k_lengths:
        if pattern == "all":
            ids[offset : offset + seqlen] = 0
        elif pattern in {"mixed", "adjacent"}:
            middle = seqlen // 2
            ids[offset + max(0, middle - 3) : offset + min(seqlen, middle + 2)] = 0
            if pattern == "adjacent":
                ids[offset + min(seqlen, middle + 2) : offset + min(seqlen, middle + 6)] = 1
        offset += seqlen
    return ids


def _assert_varlen_forward_matches_reference(
    q_lengths,
    k_lengths,
    *,
    seed,
    vision_pattern=None,
    documents=None,
):
    q, k, v, cu_q, cu_k = _gpu_varlen_qkv(q_lengths, k_lengths, seed)
    vision_ids = _packed_k_ids(k_lengths, vision_pattern) if vision_pattern is not None else None
    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        vision_block_ids=vision_ids,
        document_ids=documents,
    )
    out_ref, lse_ref = reference_attention_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        softmax_scale=1.0,
        sliding_window=1024,
        vision_block_ids=vision_ids,
        document_ids=documents,
        allow_vision_bidirectional=vision_ids is not None,
        upcast=torch.float32,
        return_lse=True,
    )
    torch.testing.assert_close(out, out_ref, atol=OUT_ATOL, rtol=OUT_RTOL)
    torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=0.0)


def _assert_forward_matches_reference(
    spec,
    seqlen,
    seed,
    *,
    forward=h100.fa4_local_text_forward,
    out_atol=OUT_ATOL,
    out_rtol=OUT_RTOL,
    lse_atol=LSE_ATOL,
    vision_block_ids=None,
):
    q, k, v = _gpu_qkv(spec, seqlen, seed)
    candidate_kwargs = {"spec": spec}
    if vision_block_ids is not None:
        candidate_kwargs["vision_block_ids"] = vision_block_ids
    out, lse = forward(q, k, v, **candidate_kwargs)
    out_ref, lse_ref = reference_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
        vision_block_ids=vision_block_ids,
        allow_vision_bidirectional=vision_block_ids is not None,
        upcast=torch.float32,
        return_lse=True,
    )
    out_ref = out_ref.transpose(1, 2)
    torch.testing.assert_close(out, out_ref, atol=out_atol, rtol=out_rtol)
    torch.testing.assert_close(lse, lse_ref, atol=lse_atol, rtol=0.0)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_d256_forward_fake_compile():
    q, k, v = _gpu_qkv(SLIDING_ATTENTION, seqlen=128, seed=1)
    out, lse = h100.fa4_local_text_forward(q, k, v)
    assert out.shape == q.shape
    assert lse.shape == (1, 32, 128)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_d256_multimodal_forward_fake_compile():
    q, k, v = _gpu_qkv(SLIDING_ATTENTION, seqlen=128, seed=11)
    vision_ids = torch.full((1, 128), -1, device="cuda", dtype=torch.int32)
    vision_ids[:, 63:66] = 0
    out, lse = h100.fa4_local_forward(q, k, v, vision_block_ids=vision_ids)
    assert out.shape == q.shape
    assert lse.shape == (1, 32, 128)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_varlen_text_fake_compile():
    q, k, v, cu_q, cu_k = _gpu_varlen_qkv([63, 65], [64, 65], seed=8001)
    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=65,
        max_seqlen_k=65,
    )
    assert out.shape == q.shape
    assert lse.shape == (32, q.shape[0])


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_varlen_max_context_text_fake_compile():
    q, k, v, cu_q, cu_k = _gpu_varlen_qkv(
        [1],
        [262_144],
        seed=8004,
        requires_grad=True,
    )
    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=262_144,
    )
    grads = torch.autograd.grad(
        (out, lse),
        (q, k, v),
        (torch.ones_like(out), torch.ones_like(lse)),
    )
    assert out.shape == q.shape
    assert lse.shape == (32, 1)
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_varlen_custom_backward_fake_compile():
    q, k, v, cu_q, cu_k = _gpu_varlen_qkv(
        [63, 65],
        [64, 65],
        seed=8002,
        requires_grad=True,
    )
    vision_ids = _packed_k_ids([64, 65], "mixed")
    out, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=65,
        max_seqlen_k=65,
        vision_block_ids=vision_ids,
    )
    grads = torch.autograd.grad(
        (out, lse),
        (q, k, v),
        (torch.ones_like(out), torch.ones_like(lse)),
    )
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_varlen_lse_only_fake_compile():
    q, k, v, cu_q, cu_k = _gpu_varlen_qkv(
        [33, 64],
        [64, 64],
        seed=8003,
        requires_grad=True,
    )
    vision_ids = _packed_k_ids([64, 64], "adjacent")
    _, lse = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=64,
        max_seqlen_k=64,
        vision_block_ids=vision_ids,
    )
    grads = torch.autograd.grad(lse, (q, k, v), torch.ones_like(lse))
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_local_sparse_empty_full_sentinel_fake_compile():
    q, k, v = [
        tensor.requires_grad_() for tensor in _gpu_qkv(SLIDING_ATTENTION, seqlen=160, seed=8005)
    ]
    sparse_fwd = h100._sparse_rows_to_tensors(
        SparseTileRows(((0, 1), ()), 128, 80, 2),
        block_size=(128, 80),
        device=q.device,
    )
    sparse_bwd = h100._sparse_rows_to_tensors(
        SparseTileRows(((0,), (), (2,)), 64, 64, 3),
        block_size=(64, 64),
        device=q.device,
    )
    vision_ids = torch.full((160,), -1, device="cuda", dtype=torch.int32)
    document_ids = torch.zeros(160, device="cuda", dtype=torch.int32)
    out, lse = h100._load_flash_attn_func()(
        q,
        k,
        v,
        causal=False,
        window_size=(None, None),
        softmax_scale=1.0,
        num_splits=1,
        pack_gqa=False,
        deterministic=False,
        mask_mod=h100._load_local_segment_mask(),
        aux_tensors=[vision_ids, document_ids],
        block_sparse_tensors=sparse_fwd,
        block_sparse_tensors_bwd=sparse_bwd,
        return_lse=True,
    )
    grads = torch.autograd.grad(
        (out, lse),
        (q, k, v),
        (torch.ones_like(out), torch.ones_like(lse)),
    )
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


@H100_FA4
@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [31, 32, 33],
        [63, 64, 65, 127, 128, 129],
        [1023, 1024, 1025],
        [129, 1, 65, 33],
        [33, 65, 1, 129],
    ],
)
def test_h100_local_varlen_text_forward(lengths):
    _assert_varlen_forward_matches_reference(lengths, lengths, seed=8100 + max(lengths))


@H100_FA4
@pytest.mark.parametrize(
    ("q_lengths", "k_lengths"),
    [
        ([1, 31, 64, 129], [33, 64, 128, 1025]),
        ([1, 63, 128], [17, 64, 129]),
        ([1, 64, 129], [1025, 1024, 1025]),
    ],
)
def test_h100_local_varlen_lower_right_text_forward(q_lengths, k_lengths):
    _assert_varlen_forward_matches_reference(q_lengths, k_lengths, seed=8200)


@H100_FA4
@pytest.mark.parametrize(
    ("q_lengths", "k_lengths"),
    [([2048], [2048]), ([64, 65], [2048, 4097])],
)
def test_h100_local_varlen_long_text_forward(q_lengths, k_lengths):
    _assert_varlen_forward_matches_reference(q_lengths, k_lengths, seed=8250)


@H100_FA4
@pytest.mark.parametrize("with_metadata", [False, True], ids=["native", "sparse-metadata"])
def test_h100_local_varlen_max_context_far_offset_forward_and_backward(with_metadata):
    spec = SLIDING_ATTENTION
    max_context = 262_144
    q_absolute = max_context - 1
    excluded_key = q_absolute - spec.sliding_window
    first_allowed_key = excluded_key + 1
    q = torch.zeros(
        1,
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        max_context,
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, max_context], device="cuda", dtype=torch.int32)
    metadata_kwargs = {}
    if with_metadata:
        metadata_kwargs = {
            "vision_block_ids": torch.full((max_context,), -1, device="cuda", dtype=torch.int32),
            "document_ids": torch.zeros(max_context, device="cuda", dtype=torch.int32),
        }

    v[excluded_key, 0] = 2048
    out_excluded, lse_excluded = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max_context,
        **metadata_kwargs,
    )
    assert torch.count_nonzero(out_excluded) == 0

    v.zero_()
    v[first_allowed_key, 0] = 2048
    out_included, lse_included = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max_context,
        **metadata_kwargs,
    )
    torch.testing.assert_close(
        out_included[0, :2],
        torch.full_like(out_included[0, :2], 2.0),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(out_included[0, 2:]) == 0
    torch.testing.assert_close(lse_included, lse_excluded, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        lse_included,
        torch.full_like(lse_included, torch.log(torch.tensor(1024.0)).item()),
        atol=1e-5,
        rtol=0.0,
    )

    v.zero_()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out, _ = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=max_context,
        **metadata_kwargs,
    )
    dout = torch.zeros_like(out)
    dout[0, 0] = 1
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()
    assert len({grad.untyped_storage().data_ptr() for grad in (dq, dk, dv)}) == 3
    assert torch.count_nonzero(dv[excluded_key, 0]) == 0
    assert torch.count_nonzero(dv[first_allowed_key, 0]) > 0
    assert torch.count_nonzero(dv[:, 1:]) == 0


@H100_FA4
@pytest.mark.parametrize(
    ("lengths", "pattern"),
    [([33, 65], "mixed"), ([127, 128, 129], "adjacent"), ([1, 17, 33, 63, 64, 65, 129], "all")],
)
def test_h100_local_varlen_vision_forward(lengths, pattern):
    _assert_varlen_forward_matches_reference(
        lengths,
        lengths,
        seed=8300 + max(lengths),
        vision_pattern=pattern,
    )


@H100_FA4
def test_h100_local_varlen_lower_right_vision_and_document_forward():
    q_lengths, k_lengths = [3, 64], [5, 129]
    documents = []
    for seqlen in k_lengths:
        doc = torch.zeros(seqlen, dtype=torch.int32)
        doc[seqlen // 2 :] = 1
        documents.append(doc)
    _assert_varlen_forward_matches_reference(
        q_lengths,
        k_lengths,
        seed=8400,
        vision_pattern="all",
        documents=torch.cat(documents).to(device="cuda"),
    )


@H100_FA4
def test_h100_local_varlen_long_vision_and_document_forward():
    q_lengths, k_lengths = [33, 65], [2049, 4097]
    documents = []
    for seqlen in k_lengths:
        doc = torch.zeros(seqlen, dtype=torch.int32)
        doc[seqlen // 3 : 2 * seqlen // 3] = 1
        doc[2 * seqlen // 3 :] = 2
        documents.append(doc)
    _assert_varlen_forward_matches_reference(
        q_lengths,
        k_lengths,
        seed=8450,
        vision_pattern="adjacent",
        documents=torch.cat(documents).to(device="cuda"),
    )


@H100_FA4
def test_h100_local_varlen_max_context_future_vision_and_document_ownership():
    spec = SLIDING_ATTENTION
    q_length, k_length = 2049, 262_144
    q_absolute = k_length - q_length
    future_key = k_length - 1
    mismatched_future_key = k_length - 161
    far_past_key = 0

    q = torch.zeros(
        q_length,
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        k_length,
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, q_length], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, k_length], device="cuda", dtype=torch.int32)
    vision_ids = torch.full((k_length,), -1, device="cuda", dtype=torch.int32)
    document_ids = torch.zeros(k_length, device="cuda", dtype=torch.int32)
    vision_ids[q_absolute] = 7
    vision_ids[future_key] = 7
    vision_ids[mismatched_future_key] = 7
    vision_ids[far_past_key] = 7
    document_ids[mismatched_future_key] = 1
    kwargs = {
        "max_seqlen_q": q_length,
        "max_seqlen_k": k_length,
        "vision_block_ids": vision_ids,
        "document_ids": document_ids,
    }

    v[future_key, 0] = 1024
    out_future, lse_future = h100.fa4_local_varlen_forward(q, k, v, cu_q, cu_k, **kwargs)
    torch.testing.assert_close(
        out_future[0, :2],
        torch.ones_like(out_future[0, :2]),
        atol=0.0078125,
        rtol=0.0,
    )
    assert torch.count_nonzero(out_future[0, 2:]) == 0
    torch.testing.assert_close(
        lse_future[:, 0],
        torch.full_like(lse_future[:, 0], torch.log(torch.tensor(1025.0)).item()),
        atol=LSE_ATOL,
        rtol=0.0,
    )

    v.zero_()
    v[mismatched_future_key, 0] = 2048
    out_mismatch, _ = h100.fa4_local_varlen_forward(q, k, v, cu_q, cu_k, **kwargs)
    assert torch.count_nonzero(out_mismatch[0]) == 0

    v.zero_()
    v[far_past_key, 0] = 2048
    out_far_past, _ = h100.fa4_local_varlen_forward(q, k, v, cu_q, cu_k, **kwargs)
    assert torch.count_nonzero(out_far_past[0]) == 0

    v.zero_()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out, _ = h100.fa4_local_varlen_forward(q, k, v, cu_q, cu_k, **kwargs)
    dout = torch.zeros_like(out)
    dout[0, 0] = 1
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()
    assert torch.count_nonzero(dv[future_key, 0]) > 0
    assert torch.count_nonzero(dv[mismatched_future_key, 0]) == 0
    assert torch.count_nonzero(dv[far_past_key, 0]) == 0
    assert torch.count_nonzero(dv[:, 1:]) == 0


@H100_FA4
def test_h100_local_varlen_strict_w1024_adversarial_sentinel():
    spec = SLIDING_ATTENTION
    q = torch.zeros(1, spec.num_q_heads, spec.head_dim_qk, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1025, spec.num_kv_heads, spec.head_dim_qk, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 1025], device="cuda", dtype=torch.int32)
    vision_ids = torch.zeros(1025, device="cuda", dtype=torch.int32)

    v_excluded = v.clone()
    v_excluded[0, 0] = 2048
    out_excluded, lse_excluded = h100.fa4_local_varlen_forward(
        q,
        k,
        v_excluded,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=1025,
        vision_block_ids=vision_ids,
    )
    assert torch.count_nonzero(out_excluded[0, 0]) == 0

    v_included = v.clone()
    v_included[1, 0] = 2048
    out_included, lse_included = h100.fa4_local_varlen_forward(
        q,
        k,
        v_included,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=1025,
        vision_block_ids=vision_ids,
    )
    torch.testing.assert_close(
        out_included[0, 0],
        torch.full_like(out_included[0, 0], 2.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(lse_included, lse_excluded, atol=0.0, rtol=0.0)


@H100_FA4
def test_h100_local_varlen_backward_strict_w1024_ownership_sentinel():
    spec = SLIDING_ATTENTION
    q = torch.zeros(
        1,
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.zeros(
        1025,
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.zeros_like(k, requires_grad=True)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 1025], device="cuda", dtype=torch.int32)
    vision_ids = torch.zeros(1025, device="cuda", dtype=torch.int32)
    out, _ = h100.fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=1025,
        vision_block_ids=vision_ids,
    )
    dout = torch.zeros_like(out)
    dout[0, 0] = 1
    _, _, dv = torch.autograd.grad(out, (q, k, v), dout)

    assert torch.count_nonzero(dv[0, 0]) == 0
    assert torch.count_nonzero(dv[1, 0]) > 0
    assert torch.count_nonzero(dv[:, 1:]) == 0


@H100_FA4
@pytest.mark.parametrize("seqlen", [1, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025])
def test_h100_local_d256_forward_exact_geometry_and_boundaries(seqlen):
    _assert_forward_matches_reference(SLIDING_ATTENTION, seqlen, seed=seqlen)


@H100_FA4
@pytest.mark.parametrize("gqa_ratio", [1, 2, 4, 8])
def test_h100_local_d256_forward_gqa_ratios(gqa_ratio):
    spec = replace(SLIDING_ATTENTION, num_kv_heads=32 // gqa_ratio)
    _assert_forward_matches_reference(spec, seqlen=33, seed=100 + gqa_ratio)


@H100_FA4
@pytest.mark.parametrize(
    ("seqlen", "pattern"),
    [
        (1, "text"),
        (31, "all"),
        (32, "mixed"),
        (33, "text"),
        (63, "adjacent"),
        (64, "all"),
        (65, "adjacent"),
        (127, "mixed"),
        (128, "text"),
        (129, "adjacent"),
        (1023, "mixed"),
        (1024, "text"),
        (1025, "all"),
    ],
)
def test_h100_local_d256_multimodal_forward(seqlen, pattern):
    vision_ids = torch.full((1, seqlen), -1, device="cuda", dtype=torch.int32)
    if pattern == "all":
        vision_ids.zero_()
    elif pattern == "mixed":
        first_start = max(1, seqlen // 5)
        first_end = min(seqlen, first_start + max(2, min(12, seqlen // 4)))
        second_start = min(seqlen, first_end + max(1, seqlen // 8))
        second_end = min(seqlen, second_start + max(2, min(9, seqlen // 5)))
        vision_ids[:, first_start:first_end] = 0
        vision_ids[:, second_start:second_end] = 1
    elif pattern == "adjacent":
        boundary = 64 if seqlen >= 65 else max(1, seqlen // 2)
        vision_ids[:, max(0, boundary - 4) : min(seqlen, boundary + 1)] = 0
        vision_ids[:, min(seqlen, boundary + 1) : min(seqlen, boundary + 8)] = 1
        if seqlen >= 129:
            vision_ids[:, 124:129] = 2
    _assert_forward_matches_reference(
        SLIDING_ATTENTION,
        seqlen,
        seed=4000 + seqlen,
        forward=h100.fa4_local_forward,
        vision_block_ids=vision_ids,
    )


@H100_FA4
def test_h100_local_multimodal_masked_kv_sentinels_do_not_leak():
    seqlen, query, allowed_future, masked_future = 65, 31, 32, 33
    q = torch.zeros(
        1,
        seqlen,
        SLIDING_ATTENTION.num_q_heads,
        SLIDING_ATTENTION.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        1,
        seqlen,
        SLIDING_ATTENTION.num_kv_heads,
        SLIDING_ATTENTION.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    vision_ids = torch.full((1, seqlen), -1, device="cuda", dtype=torch.int32)
    vision_ids[:, query : allowed_future + 1] = 0
    vision_ids[:, masked_future] = 1

    q[:, query, 0, 0] = 1
    out_base, lse_base = h100.fa4_local_forward(q, k, v, vision_block_ids=vision_ids)

    k_masked, v_masked = k.clone(), v.clone()
    k_masked[:, masked_future, 0, 0] = 128
    v_masked[:, masked_future, 0] = 128
    out_masked, lse_masked = h100.fa4_local_forward(
        q,
        k_masked,
        v_masked,
        vision_block_ids=vision_ids,
    )
    torch.testing.assert_close(out_masked[:, query, 0], out_base[:, query, 0], atol=0, rtol=0)
    torch.testing.assert_close(lse_masked[:, 0, query], lse_base[:, 0, query], atol=0, rtol=0)

    v_allowed = v.clone()
    v_allowed[:, allowed_future, 0] = 128
    out_allowed, _ = h100.fa4_local_forward(q, k, v_allowed, vision_block_ids=vision_ids)
    assert not torch.equal(out_allowed[:, query, 0], out_base[:, query, 0])


@H100_FA4
def test_h100_local_multimodal_s1025_exact_window_and_future_edges():
    seqlen = 1025
    q = torch.zeros(
        1,
        seqlen,
        SLIDING_ATTENTION.num_q_heads,
        SLIDING_ATTENTION.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros(
        1,
        seqlen,
        SLIDING_ATTENTION.num_kv_heads,
        SLIDING_ATTENTION.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    vision_ids = torch.zeros((1, seqlen), device="cuda", dtype=torch.int32)

    v_k0 = v.clone()
    v_k0[:, 0, 0] = 128
    out_k0, _ = h100.fa4_local_forward(q, k, v_k0, vision_block_ids=vision_ids)
    assert torch.count_nonzero(out_k0[:, 1024, 0]) == 0

    v_k1 = v.clone()
    v_k1[:, 1, 0] = 128
    out_k1, _ = h100.fa4_local_forward(q, k, v_k1, vision_block_ids=vision_ids)
    assert torch.count_nonzero(out_k1[:, 1024, 0]) > 0

    v_k1024 = v.clone()
    v_k1024[:, 1024, 0] = 128
    out_k1024, _ = h100.fa4_local_forward(q, k, v_k1024, vision_block_ids=vision_ids)
    assert torch.count_nonzero(out_k1024[:, 0, 0]) > 0


@H100_FA4
def test_h100_local_d256_forward_repeats_on_nondefault_stream():
    q, k, v = _gpu_qkv(SLIDING_ATTENTION, seqlen=129, seed=7)
    out0, lse0 = h100.fa4_local_text_forward(q, k, v)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out1, lse1 = h100.fa4_local_text_forward(q, k, v)
    stream.synchronize()
    torch.testing.assert_close(out1, out0, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse1, lse0, atol=0.0, rtol=0.0)


@H100_FA4_FAKE
@_fake_tensor_mode_if_requested
def test_h100_global_d512_forward_fake_compile():
    q, k, v = _gpu_qkv(GLOBAL_ATTENTION, seqlen=128, seed=2)
    out, lse = h100.fa4_global_text_forward(q, k, v)
    assert out.shape == q.shape
    assert lse.shape == (1, 32, 128)


@H100_FA4
@pytest.mark.parametrize(
    "seqlen",
    [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1024],
)
def test_h100_global_d512_forward_exact_geometry_and_boundaries(seqlen):
    _assert_forward_matches_reference(
        GLOBAL_ATTENTION,
        seqlen,
        seed=2000 + seqlen,
        forward=h100.fa4_global_text_forward,
        out_atol=GLOBAL_OUT_ATOL,
        out_rtol=GLOBAL_OUT_RTOL,
        lse_atol=GLOBAL_LSE_ATOL,
    )


@H100_FA4
def test_h100_global_d512_forward_repeats_on_nondefault_stream():
    q, k, v = _gpu_qkv(GLOBAL_ATTENTION, seqlen=129, seed=9)
    out0, lse0 = h100.fa4_global_text_forward(q, k, v)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out1, lse1 = h100.fa4_global_text_forward(q, k, v)
    stream.synchronize()
    torch.testing.assert_close(out1, out0, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse1, lse0, atol=0.0, rtol=0.0)
