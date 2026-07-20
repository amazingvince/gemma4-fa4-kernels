from __future__ import annotations

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


def test_custom_op_capability_matches_installed_torch() -> None:
    torch_library = getattr(torch, "library", None)
    expected = callable(getattr(torch_library, "custom_op", None)) and callable(
        getattr(torch_library, "register_fake", None)
    )
    assert h100_torch_ops.CUSTOM_OPS_AVAILABLE is expected

    if expected:
        assert str(h100_torch_ops.h100_local_fwd).endswith("gemma4_fa4::h100_local_fwd)>")
        assert str(h100_torch_ops.h100_global_fwd).endswith("gemma4_fa4::h100_global_fwd)>")
    else:
        q, k, v, position_ids, packed_sequence_ids = _inputs(family="local", seqlen=1, device="cpu")
        with pytest.raises(RuntimeError, match="custom_op.*register_fake"):
            h100_torch_ops.h100_local_fwd(q, k, v, position_ids, packed_sequence_ids)


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
def test_opcheck_passes_complete_custom_op_contract(family: str) -> None:
    inputs = _inputs(family=family, seqlen=1, device="cuda")
    result = torch.library.opcheck(_op(family), inputs)
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
