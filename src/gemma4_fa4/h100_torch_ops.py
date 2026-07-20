"""Opaque PyTorch operators for the scoped H100 compiler boundary.

The operator bodies consume prepared Transformers ``(B, H, S, D)`` tensors
and reuse the retained fixed FA4 routes.  Fake implementations describe only
fresh output metadata; they deliberately do not validate storage or execute a
CuTe kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .h100 import UnsupportedH100Path, fa4_global_text_forward, fa4_local_text_forward

LOCAL_OP_NAME = "gemma4_fa4::h100_local_fwd"
GLOBAL_OP_NAME = "gemma4_fa4::h100_global_fwd"

_LOCAL_GEOMETRY = (32, 16, 256)
_GLOBAL_GEOMETRY = (32, 4, 512)
_MAX_SEQLEN = 1024


def _validate_real_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
    *,
    geometry: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the real, inference-only BHSD ABI and return BSHD views."""

    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        raise ValueError("compiled H100 q, k, and v must be rank-4 BHSD tensors")
    if any(tensor.requires_grad for tensor in (q, k, v)):
        raise UnsupportedH100Path("compiled H100 forward ops reject every requires_grad input")
    if q.device.type != "cuda" or not (q.device == k.device == v.device):
        raise UnsupportedH100Path("compiled H100 q, k, and v must be CUDA tensors on one device")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (9, 0):
        raise UnsupportedH100Path(
            f"compiled H100 forward ops require compute capability 9.0, found {capability}"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise ValueError("compiled H100 q, k, and v must use BF16")

    q_heads, kv_heads, head_dim = geometry
    batch, actual_q_heads, seqlen, actual_q_dim = q.shape
    if batch != 1 or not 1 <= seqlen <= _MAX_SEQLEN:
        raise UnsupportedH100Path("compiled H100 forward ops require B=1 and 1 <= S <= 1024")
    if (actual_q_heads, actual_q_dim) != (q_heads, head_dim):
        raise ValueError("compiled H100 q does not match the locked layer geometry")
    expected_kv = (1, kv_heads, seqlen, head_dim)
    if k.shape != expected_kv or v.shape != expected_kv:
        raise ValueError("compiled H100 k/v must match q length and the locked layer geometry")

    if position_ids.ndim != 2 or position_ids.shape != (1, seqlen):
        raise ValueError("compiled H100 position_ids must have shape (1, S)")
    if position_ids.device != q.device:
        raise ValueError("compiled H100 position_ids must share the q/k/v CUDA device")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compiled H100 position_ids must use INT32 or INT64")
    expected_positions = torch.arange(
        seqlen,
        device=position_ids.device,
        dtype=position_ids.dtype,
    ).unsqueeze(0)
    if not torch.equal(position_ids, expected_positions):
        raise UnsupportedH100Path("compiled H100 position_ids must equal zero-based arange(S)")

    if packed_sequence_ids.ndim != 2 or packed_sequence_ids.shape != (1, seqlen):
        raise ValueError("compiled H100 packed_sequence_ids must have shape (1, S)")
    if packed_sequence_ids.device != q.device:
        raise ValueError("compiled H100 packed_sequence_ids must share the q/k/v CUDA device")
    if packed_sequence_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compiled H100 packed_sequence_ids must use INT32 or INT64")
    packed_boundaries = torch.nn.functional.pad(
        torch.diff(position_ids, dim=-1) != 1,
        (1, 0),
        value=False,
    )
    expected_packed = packed_boundaries.to(torch.int64).cumsum(-1)
    if not torch.equal(packed_sequence_ids.to(torch.int64), expected_packed):
        raise UnsupportedH100Path(
            "compiled H100 packed_sequence_ids must be derived exactly from position_ids"
        )

    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in (q, k, v)}
    if len(storage_pointers) != 3:
        raise ValueError("compiled H100 q, k, and v must use distinct storage")

    # The retained fixed-route validators prove unit-D, positive aligned outer
    # strides, non-overlap, distinct prepared K/V, and the exact model family.
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _validate_real_outputs(
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_out = (q.shape[0], q.shape[2], q.shape[1], v.shape[3])
    expected_lse = (q.shape[0], q.shape[1], q.shape[2])
    if out.shape != expected_out or out.dtype != torch.bfloat16 or out.device != q.device:
        raise RuntimeError("compiled H100 FA4 returned an invalid BSHD output contract")
    if lse.shape != expected_lse or lse.dtype != torch.float32 or lse.device != q.device:
        raise RuntimeError("compiled H100 FA4 returned an invalid FP32 LSE contract")

    input_pointers = {tensor.untyped_storage().data_ptr() for tensor in (q, k, v)}
    out_pointer = out.untyped_storage().data_ptr()
    lse_pointer = lse.untyped_storage().data_ptr()
    if out_pointer in input_pointers or lse_pointer in input_pointers or out_pointer == lse_pointer:
        raise RuntimeError("compiled H100 FA4 outputs must use fresh, non-aliasing storage")
    return out, lse


def _h100_local_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_bshd, k_bshd, v_bshd = _validate_real_inputs(
        q,
        k,
        v,
        position_ids,
        packed_sequence_ids,
        geometry=_LOCAL_GEOMETRY,
    )
    out, lse = fa4_local_text_forward(q_bshd, k_bshd, v_bshd)
    return _validate_real_outputs(out, lse, q, k, v)


def _h100_global_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_bshd, k_bshd, v_bshd = _validate_real_inputs(
        q,
        k,
        v,
        position_ids,
        packed_sequence_ids,
        geometry=_GLOBAL_GEOMETRY,
    )
    out, lse = fa4_global_text_forward(q_bshd, k_bshd, v_bshd)
    return _validate_real_outputs(out, lse, q, k, v)


def _fake_forward(
    q: torch.Tensor,
    _k: torch.Tensor,
    v: torch.Tensor,
    _position_ids: torch.Tensor,
    _packed_sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only fresh symbolic BSHD output and FP32-LSE metadata."""

    out = torch.empty(
        (q.shape[0], q.shape[2], q.shape[1], v.shape[3]),
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        dtype=torch.float32,
        device=q.device,
    )
    return out, lse


_torch_library = getattr(torch, "library", None)
_custom_op: Callable | None = getattr(_torch_library, "custom_op", None)
_register_fake: Callable | None = getattr(_torch_library, "register_fake", None)
CUSTOM_OPS_AVAILABLE = callable(_custom_op) and callable(_register_fake)


if CUSTOM_OPS_AVAILABLE:
    h100_local_fwd = _custom_op(LOCAL_OP_NAME, mutates_args=())(_h100_local_impl)
    _register_fake(h100_local_fwd)(_fake_forward)

    h100_global_fwd = _custom_op(GLOBAL_OP_NAME, mutates_args=())(_h100_global_impl)
    _register_fake(h100_global_fwd)(_fake_forward)
else:

    def _unavailable(
        _q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        _position_ids: torch.Tensor,
        _packed_sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "H100 opaque forward ops require PyTorch with "
            "torch.library.custom_op and torch.library.register_fake"
        )

    h100_local_fwd = _unavailable
    h100_global_fwd = _unavailable


__all__ = [
    "CUSTOM_OPS_AVAILABLE",
    "GLOBAL_OP_NAME",
    "LOCAL_OP_NAME",
    "h100_global_fwd",
    "h100_local_fwd",
]
