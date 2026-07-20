from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch

from gemma4_fa4 import h100_torch_ops


def _has_h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


H100_CUSTOM_OPS = pytest.mark.skipif(
    not (_has_h100() and h100_torch_ops.CUSTOM_OPS_AVAILABLE),
    reason="requires torch.library custom ops and real H100 execution",
)


def _inputs(
    *,
    family: str,
    seqlen: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_heads, head_dim = (16, 256) if family == "local" else (4, 512)
    q = torch.randn(1, 32, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, kv_heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(seqlen, dtype=torch.int64, device=device).unsqueeze(0)
    packed_sequence_ids = torch.zeros(1, seqlen, dtype=torch.int64, device=device)
    return q, k, v, position_ids, packed_sequence_ids


def _op(family: str) -> Callable:
    return h100_torch_ops.h100_local_fwd if family == "local" else h100_torch_ops.h100_global_fwd


def _layer_inputs(
    *,
    family: str,
    seqlen: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, ...]:
    head_dim = 256 if family == "local" else 512
    kv_heads = 16 if family == "local" else 4
    hidden = torch.randn(1, seqlen, 5376, dtype=torch.bfloat16, device=device)
    cos = torch.randn(1, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(1, seqlen, head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seqlen, dtype=torch.int64, device=device).unsqueeze(0)
    packed = torch.zeros(1, seqlen, dtype=torch.int64, device=device)
    weights = [
        torch.randn(32 * head_dim, 5376, dtype=torch.bfloat16, device=device),
        torch.randn(kv_heads * head_dim, 5376, dtype=torch.bfloat16, device=device),
    ]
    if family == "local":
        weights.append(torch.randn(kv_heads * head_dim, 5376, dtype=torch.bfloat16, device=device))
    weights.extend(
        [
            torch.randn(5376, 32 * head_dim, dtype=torch.bfloat16, device=device),
            torch.randn(head_dim, dtype=torch.bfloat16, device=device),
            torch.randn(head_dim, dtype=torch.bfloat16, device=device),
        ]
    )
    return hidden, cos, sin, positions, packed, *weights


def _layer_op(family: str) -> Callable:
    return (
        h100_torch_ops.h100_local_layer_fwd
        if family == "local"
        else h100_torch_ops.h100_global_layer_fwd
    )


def _layer_opcheck_target(family: str) -> Callable:
    return (
        h100_torch_ops.h100_local_layer_op
        if family == "local"
        else h100_torch_ops.h100_global_layer_op
    )


def test_custom_op_capability_matches_installed_torch() -> None:
    torch_library = getattr(torch, "library", None)
    expected = callable(getattr(torch_library, "custom_op", None)) and callable(
        getattr(torch_library, "register_fake", None)
    )
    assert h100_torch_ops.CUSTOM_OPS_AVAILABLE is expected

    if expected:
        assert str(h100_torch_ops.h100_local_fwd).endswith("gemma4_fa4::h100_local_fwd)>")
        assert str(h100_torch_ops.h100_global_fwd).endswith("gemma4_fa4::h100_global_fwd)>")
        assert str(h100_torch_ops.h100_local_layer_op).endswith(
            "gemma4_fa4::h100_local_layer_fwd)>"
        )
        assert str(h100_torch_ops.h100_global_layer_op).endswith(
            "gemma4_fa4::h100_global_layer_fwd)>"
        )
        assert str(h100_torch_ops.h100_global_static_cache_decode_op).endswith(
            "gemma4_fa4::h100_global_static_cache_decode_fwd)>"
        )
        assert str(h100_torch_ops.h100_local_static_cache_decode_op).endswith(
            "gemma4_fa4::h100_local_static_cache_decode_fwd)>"
        )
        global_schema = str(h100_torch_ops.h100_global_static_cache_decode_op._schema)
        assert "Tensor(a4!) cache_k" in global_schema
        assert "Tensor(a5!) cache_v" in global_schema
        assert "Tensor(a6!) cache_length" in global_schema
        local_schema = str(h100_torch_ops.h100_local_static_cache_decode_op._schema)
        assert "Tensor(a4!) cache_k" in local_schema
        assert "Tensor(a5!) cache_v" in local_schema
        assert "Tensor cache_length" in local_schema
        assert "Tensor(a6!) cache_length" not in local_schema
    else:
        q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cpu")
        with pytest.raises(RuntimeError, match="custom_op.*register_fake"):
            h100_torch_ops.h100_local_fwd(q, k, v, position_ids, packed_sequence_ids)
        with pytest.raises(RuntimeError, match="custom_op.*register_fake"):
            h100_torch_ops.h100_local_layer_fwd(q, k, v, position_ids, packed_sequence_ids)


def test_whole_layer_real_abis_are_tensor_explicit() -> None:
    expected_local = (
        "hidden_states",
        "cos",
        "sin",
        "position_ids",
        "packed_sequence_ids",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
    )
    expected_global = (
        "hidden_states",
        "cos",
        "sin",
        "position_ids",
        "packed_sequence_ids",
        "q_proj_weight",
        "k_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
    )
    assert (
        tuple(inspect.signature(h100_torch_ops._h100_local_layer_impl).parameters) == expected_local
    )
    assert (
        tuple(inspect.signature(h100_torch_ops._h100_global_layer_impl).parameters)
        == expected_global
    )
    assert tuple(
        inspect.signature(h100_torch_ops._h100_global_static_cache_decode_impl).parameters
    ) == (
        "hidden_states",
        "cos",
        "sin",
        "position_ids",
        "cache_k",
        "cache_v",
        "cache_length",
        "q_proj_weight",
        "k_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
    )
    assert tuple(
        inspect.signature(h100_torch_ops._h100_local_static_cache_decode_impl).parameters
    ) == (
        "hidden_states",
        "cos",
        "sin",
        "position_ids",
        "cache_k",
        "cache_v",
        "cache_length",
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "o_proj_weight",
        "q_norm_weight",
        "k_norm_weight",
    )


@pytest.mark.skipif(
    not h100_torch_ops.CUSTOM_OPS_AVAILABLE,
    reason="installed PyTorch does not provide torch.library custom ops",
)
@pytest.mark.parametrize(
    ("family", "kv_heads", "head_dim"),
    [("local", 16, 256), ("global", 4, 512)],
)
def test_fake_registration_returns_fresh_symbolic_contract(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    kv_heads: int,
    head_dim: int,
) -> None:
    def unexpected_real_call(*_args, **_kwargs):
        raise AssertionError("the fake implementation entered a real FA4 route")

    monkeypatch.setattr(h100_torch_ops, f"fa4_{family}_text_forward", unexpected_real_call)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 32, 33, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.empty(1, kv_heads, 33, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.empty(1, kv_heads, 33, head_dim, dtype=torch.bfloat16, device="cuda")
        position_ids = torch.empty(1, 33, dtype=torch.int64, device="cuda")
        packed_sequence_ids = torch.empty(1, 33, dtype=torch.int64, device="cuda")
        out, lse = _op(family)(q, k, v, position_ids, packed_sequence_ids)

    assert out.shape == (1, 33, 32, head_dim)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 33)
    assert lse.dtype == torch.float32
    assert out is not q
    assert lse is not q


@pytest.mark.skipif(
    not h100_torch_ops.CUSTOM_OPS_AVAILABLE,
    reason="installed PyTorch does not provide torch.library custom ops",
)
@pytest.mark.parametrize("family", ["local", "global"])
def test_whole_layer_fake_registration_is_shape_only_and_fresh(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    def unexpected_real_call(*_args, **_kwargs):
        raise AssertionError("the whole-layer fake implementation entered a real body")

    monkeypatch.setattr(h100_torch_ops, f"_h100_{family}_layer_impl", unexpected_real_call)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.inference_mode():
        inputs = _layer_inputs(family=family, seqlen=33, device="cuda")
        out, lse = _layer_op(family)(*inputs)

    assert out.shape == (1, 33, 5376)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 33)
    assert lse.dtype == torch.float32
    assert all(out is not tensor and lse is not tensor for tensor in inputs)


@pytest.mark.skipif(
    not h100_torch_ops.CUSTOM_OPS_AVAILABLE,
    reason="installed PyTorch does not provide torch.library custom ops",
)
def test_static_cache_decode_fake_registration_is_shape_only_and_fresh() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.inference_mode():
        device = "cuda"
        inputs = (
            torch.empty((1, 1, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1, 512), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1, 512), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1), dtype=torch.int64, device=device),
            torch.empty((1, 4, 65, 512), dtype=torch.bfloat16, device=device),
            torch.empty((1, 4, 65, 512), dtype=torch.bfloat16, device=device),
            torch.empty((), dtype=torch.int64, device=device),
            torch.empty((32 * 512, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((4 * 512, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((5376, 32 * 512), dtype=torch.bfloat16, device=device),
            torch.empty((512,), dtype=torch.bfloat16, device=device),
            torch.empty((512,), dtype=torch.bfloat16, device=device),
        )
        out, lse = h100_torch_ops.h100_global_static_cache_decode_fwd(*inputs)

    assert out.shape == (1, 1, 5376)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 1)
    assert lse.dtype == torch.float32
    assert all(out is not tensor and lse is not tensor for tensor in inputs)


@pytest.mark.skipif(
    not h100_torch_ops.CUSTOM_OPS_AVAILABLE,
    reason="installed PyTorch does not provide torch.library custom ops",
)
def test_local_static_cache_decode_fake_registration_is_shape_only_and_fresh() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(), torch.inference_mode():
        device = "cuda"
        inputs = (
            torch.empty((1, 1, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1, 256), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1, 256), dtype=torch.bfloat16, device=device),
            torch.empty((1, 1), dtype=torch.int64, device=device),
            torch.empty((1, 16, 1024, 256), dtype=torch.bfloat16, device=device),
            torch.empty((1, 16, 1024, 256), dtype=torch.bfloat16, device=device),
            torch.empty((), dtype=torch.int64, device=device),
            torch.empty((32 * 256, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((16 * 256, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((16 * 256, 5376), dtype=torch.bfloat16, device=device),
            torch.empty((5376, 32 * 256), dtype=torch.bfloat16, device=device),
            torch.empty((256,), dtype=torch.bfloat16, device=device),
            torch.empty((256,), dtype=torch.bfloat16, device=device),
        )
        out, lse = h100_torch_ops.h100_local_static_cache_decode_fwd(*inputs)

    assert out.shape == (1, 1, 5376)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 1)
    assert lse.dtype == torch.float32
    assert all(out is not tensor and lse is not tensor for tensor in inputs)


@pytest.mark.skipif(
    not h100_torch_ops.CUSTOM_OPS_AVAILABLE,
    reason="installed PyTorch does not provide torch.library custom ops",
)
def test_real_cpu_execution_fails_explicitly() -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cpu")
    with pytest.raises(h100_torch_ops.UnsupportedH100Path, match="CUDA tensors"):
        h100_torch_ops.h100_local_fwd(q, k, v, position_ids, packed_sequence_ids)


@H100_CUSTOM_OPS
@pytest.mark.parametrize("family", ["local", "global"])
def test_real_h100_outputs_are_fresh_and_exactly_typed(family: str) -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family=family, seqlen=1, device="cuda")
    with torch.inference_mode():
        out, lse = _op(family)(q, k, v, position_ids, packed_sequence_ids)

    assert out.shape == (1, 1, 32, q.shape[-1])
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 1)
    assert lse.dtype == torch.float32
    pointers = {tensor.untyped_storage().data_ptr() for tensor in (q, k, v, out, lse)}
    assert len(pointers) == 5


@H100_CUSTOM_OPS
@pytest.mark.parametrize("family", ["local", "global"])
def test_real_h100_whole_layer_outputs_are_fresh_and_exactly_typed(family: str) -> None:
    inputs = _layer_inputs(family=family, seqlen=1, device="cuda")
    with torch.inference_mode():
        out, lse = _layer_op(family)(*inputs)

    assert out.shape == (1, 1, 5376)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (1, 32, 1)
    assert lse.dtype == torch.float32
    pointers = {tensor.untyped_storage().data_ptr() for tensor in (*inputs, out, lse)}
    assert len(pointers) == len(inputs) + 2


@H100_CUSTOM_OPS
@pytest.mark.parametrize("family", ["local", "global"])
def test_opcheck_passes_complete_custom_op_contract(family: str) -> None:
    inputs = _inputs(family=family, seqlen=1, device="cuda")
    result = torch.library.opcheck(_op(family), inputs)
    assert set(result.values()) == {"SUCCESS"}


@H100_CUSTOM_OPS
@pytest.mark.parametrize("family", ["local", "global"])
def test_whole_layer_opcheck_passes_complete_tensor_explicit_contract(family: str) -> None:
    inputs = _layer_inputs(family=family, seqlen=1, device="cuda")
    with torch.inference_mode():
        result = torch.library.opcheck(_layer_opcheck_target(family), inputs)
    assert set(result.values()) == {"SUCCESS"}


@H100_CUSTOM_OPS
def test_real_h100_rejects_requires_grad_even_under_no_grad() -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cuda")
    q.requires_grad_(True)
    with (
        torch.no_grad(),
        pytest.raises(
            h100_torch_ops.UnsupportedH100Path,
            match="requires_grad",
        ),
    ):
        h100_torch_ops.h100_local_fwd(q, k, v, position_ids, packed_sequence_ids)


@H100_CUSTOM_OPS
@pytest.mark.parametrize("family", ["local", "global"])
def test_whole_layer_accepts_dormant_weight_flags_only_without_grad(family: str) -> None:
    inputs = list(_layer_inputs(family=family, seqlen=1, device="cuda"))
    for weight in inputs[5:]:
        weight.requires_grad_(True)
    with torch.inference_mode():
        out, lse = _layer_op(family)(*inputs)
    assert not out.requires_grad and not lse.requires_grad

    with pytest.raises(
        (h100_torch_ops.UnsupportedH100Path, RuntimeError),
        match="inference|no-grad|autograd",
    ):
        _layer_op(family)(*inputs)


@H100_CUSTOM_OPS
def test_real_h100_rejects_reset_positions_and_aliasing() -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=2, device="cuda")
    reset_positions = torch.tensor([[0, 0]], dtype=torch.int64, device="cuda")
    with (
        torch.inference_mode(),
        pytest.raises(
            h100_torch_ops.UnsupportedH100Path,
            match="arange",
        ),
    ):
        h100_torch_ops.h100_local_fwd(q, k, v, reset_positions, packed_sequence_ids)

    with torch.inference_mode(), pytest.raises(ValueError, match="distinct storage"):
        h100_torch_ops.h100_local_fwd(q, k, k, position_ids, packed_sequence_ids)

    with torch.inference_mode(), pytest.raises(ValueError, match="distinct storage"):
        h100_torch_ops.h100_local_fwd(
            q,
            q[:, :16],
            v,
            position_ids,
            packed_sequence_ids,
        )

    forged_packed = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    with (
        torch.inference_mode(),
        pytest.raises(
            h100_torch_ops.UnsupportedH100Path,
            match="derived exactly",
        ),
    ):
        h100_torch_ops.h100_local_fwd(q, k, v, position_ids, forged_packed)


@H100_CUSTOM_OPS
@pytest.mark.parametrize(
    "position_ids",
    [
        torch.empty(1, dtype=torch.int64),
        torch.empty((1, 1), dtype=torch.float32),
    ],
)
def test_real_h100_rejects_position_metadata_contract(position_ids: torch.Tensor) -> None:
    q, k, v, _, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cuda")
    with torch.inference_mode(), pytest.raises(ValueError, match="position_ids"):
        h100_torch_ops.h100_local_fwd(
            q,
            k,
            v,
            position_ids.to(device="cuda"),
            packed_sequence_ids,
        )


@H100_CUSTOM_OPS
def test_real_h100_rejects_rank_dtype_geometry_and_length_boundaries() -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cuda")
    invalid_cases = [
        ((q[0], k, v, position_ids, packed_sequence_ids), ValueError, "rank-4"),
        ((q.float(), k, v, position_ids, packed_sequence_ids), ValueError, "BF16"),
        ((q[:, :31], k, v, position_ids, packed_sequence_ids), ValueError, "geometry"),
    ]
    for inputs, error_type, match in invalid_cases:
        with torch.inference_mode(), pytest.raises(error_type, match=match):
            h100_torch_ops.h100_local_fwd(*inputs)

    q0 = torch.empty((1, 32, 0, 256), dtype=torch.bfloat16, device="cuda")
    k0 = torch.empty((1, 16, 0, 256), dtype=torch.bfloat16, device="cuda")
    v0 = torch.empty_like(k0)
    positions0 = torch.empty((1, 0), dtype=torch.int64, device="cuda")
    packed0 = torch.empty_like(positions0)
    with (
        torch.inference_mode(),
        pytest.raises(
            h100_torch_ops.UnsupportedH100Path,
            match="1 <= S <= 1024",
        ),
    ):
        h100_torch_ops.h100_local_fwd(q0, k0, v0, positions0, packed0)

    q_long = torch.empty((1, 32, 1025, 256), dtype=torch.bfloat16, device="cuda")
    k_long = torch.empty((1, 16, 1025, 256), dtype=torch.bfloat16, device="cuda")
    v_long = torch.empty_like(k_long)
    positions_long = torch.arange(1025, dtype=torch.int64, device="cuda").unsqueeze(0)
    packed_long = torch.zeros_like(positions_long)
    with (
        torch.inference_mode(),
        pytest.raises(
            h100_torch_ops.UnsupportedH100Path,
            match="1 <= S <= 1024",
        ),
    ):
        h100_torch_ops.h100_local_fwd(
            q_long,
            k_long,
            v_long,
            positions_long,
            packed_long,
        )


@H100_CUSTOM_OPS
def test_real_h100_rejects_invalid_packed_metadata_and_layout() -> None:
    q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cuda")
    for invalid_packed in (
        packed_sequence_ids[0],
        packed_sequence_ids.to(torch.float32),
    ):
        with torch.inference_mode(), pytest.raises(ValueError, match="packed_sequence_ids"):
            h100_torch_ops.h100_local_fwd(q, k, v, position_ids, invalid_packed)

    misaligned_q = torch.empty(
        (1, 32, 1, 257),
        dtype=torch.bfloat16,
        device="cuda",
    )[..., 1:]
    with torch.inference_mode(), pytest.raises(ValueError, match="16-byte aligned"):
        h100_torch_ops.h100_local_fwd(
            misaligned_q,
            k,
            v,
            position_ids,
            packed_sequence_ids,
        )
