from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

import gemma4_fa4.h100 as h100
from gemma4_fa4.model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION
from gemma4_fa4.reference import reference_attention

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


def test_local_adapter_rejects_kv_aliasing_and_wrong_contracts():
    q, k, v = _cpu_qkv()
    with pytest.raises(ValueError, match="distinct, non-aliasing"):
        h100._validate_local_bshd(q, k, k, SLIDING_ATTENTION, require_sm90=False)
    with pytest.raises(ValueError, match="BF16"):
        h100._validate_local_bshd(
            q.float(), k.float(), v.float(), SLIDING_ATTENTION, require_sm90=False
        )
    with pytest.raises(ValueError, match="contiguous BSHD"):
        h100._validate_local_bshd(q.transpose(1, 2), k, v, SLIDING_ATTENTION, require_sm90=False)
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
    with pytest.raises(h100.UnsupportedH100Path, match="B=1"):
        h100._validate_global_bshd(
            q.expand(2, -1, -1, -1).contiguous(),
            k.expand(2, -1, -1, -1).contiguous(),
            v.expand(2, -1, -1, -1).contiguous(),
            GLOBAL_ATTENTION,
            require_sm90=False,
        )


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


def _assert_forward_matches_reference(
    spec,
    seqlen,
    seed,
    *,
    forward=h100.fa4_local_text_forward,
    out_atol=OUT_ATOL,
    out_rtol=OUT_RTOL,
    lse_atol=LSE_ATOL,
):
    q, k, v = _gpu_qkv(spec, seqlen, seed)
    out, lse = forward(q, k, v, spec=spec)
    out_ref, lse_ref = reference_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=1.0,
        sliding_window=spec.sliding_window,
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
