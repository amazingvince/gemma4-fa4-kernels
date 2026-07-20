"""Opaque PyTorch operators for the scoped H100 compiler boundary.

The prepared operators consume Transformers ``(B, H, S, D)`` tensors.  The
whole-layer operators additionally reproduce the exact pinned projection,
normalization, rotary, reshape, and output-projection order around those
retained FA4 routes.  Every weight is an explicit tensor argument; no module
object or mutable registry participates in the operator ABI.  Fake
implementations describe only fresh output metadata and deliberately execute
neither tensor validation nor a CuTe kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .h100 import (
    UnsupportedH100Path,
    fa4_global_forward_only,
    fa4_global_text_forward,
    fa4_local_text_forward,
    fa4_local_varlen_forward,
)

LOCAL_OP_NAME = "gemma4_fa4::h100_local_fwd"
GLOBAL_OP_NAME = "gemma4_fa4::h100_global_fwd"
LOCAL_LAYER_OP_NAME = "gemma4_fa4::h100_local_layer_fwd"
GLOBAL_LAYER_OP_NAME = "gemma4_fa4::h100_global_layer_fwd"
GLOBAL_STATIC_CACHE_DECODE_OP_NAME = (
    "gemma4_fa4::h100_global_static_cache_decode_fwd"
)
LOCAL_STATIC_CACHE_DECODE_OP_NAME = (
    "gemma4_fa4::h100_local_static_cache_decode_fwd"
)

_LOCAL_GEOMETRY = (32, 16, 256)
_GLOBAL_GEOMETRY = (32, 4, 512)
_HIDDEN_SIZE = 5376
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


def _validate_layer_real_inputs(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    *,
    geometry: tuple[int, int, int],
    expected_weight_shapes: tuple[tuple[int, ...], ...],
) -> int:
    """Validate EXP-0020's inference-only tensor-explicit whole-layer ABI."""

    if hidden_states.ndim != 3:
        raise ValueError("compiled H100 whole-layer hidden_states must be rank-3 BSH")
    batch, seqlen, hidden_size = hidden_states.shape
    if batch != 1 or not 1 <= seqlen <= _MAX_SEQLEN:
        raise UnsupportedH100Path("compiled H100 whole-layer ops require B=1 and 1 <= S <= 1024")
    if hidden_size != _HIDDEN_SIZE:
        raise ValueError("compiled H100 whole-layer hidden size must equal 5376")

    _q_heads, _kv_heads, head_dim = geometry
    explicit_tensors = (
        hidden_states,
        cos,
        sin,
        position_ids,
        packed_sequence_ids,
        *weights,
    )
    if any(tensor.device != hidden_states.device for tensor in explicit_tensors):
        raise ValueError("compiled H100 whole-layer tensors must share one device")
    if hidden_states.device.type != "cuda":
        raise UnsupportedH100Path("compiled H100 whole-layer tensors must be CUDA tensors")
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability != (9, 0):
        raise UnsupportedH100Path(
            f"compiled H100 whole-layer ops require compute capability 9.0, found {capability}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("compiled H100 whole-layer hidden_states must use BF16")
    if cos.shape != (1, seqlen, head_dim) or sin.shape != (1, seqlen, head_dim):
        raise ValueError("compiled H100 whole-layer rotary tensors have the wrong shape")
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise ValueError("compiled H100 whole-layer rotary tensors must use BF16")
    if position_ids.shape != (1, seqlen) or position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compiled H100 whole-layer position_ids must be INT32/INT64 (1, S)")
    if packed_sequence_ids.shape != (1, seqlen) or packed_sequence_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("compiled H100 whole-layer packed_sequence_ids must be INT32/INT64 (1, S)")
    expected_positions = torch.arange(
        seqlen,
        device=position_ids.device,
        dtype=position_ids.dtype,
    ).unsqueeze(0)
    if not torch.equal(position_ids, expected_positions):
        raise UnsupportedH100Path(
            "compiled H100 whole-layer position_ids must equal zero-based arange(S)"
        )
    packed_boundaries = torch.nn.functional.pad(
        torch.diff(position_ids, dim=-1) != 1,
        (1, 0),
        value=False,
    )
    expected_packed = packed_boundaries.to(torch.int64).cumsum(-1)
    if not torch.equal(packed_sequence_ids.to(torch.int64), expected_packed):
        raise UnsupportedH100Path(
            "compiled H100 whole-layer packed IDs must be derived exactly from position_ids"
        )

    if len(weights) != len(expected_weight_shapes):
        raise AssertionError("whole-layer weight schema and validation table disagree")
    for weight, expected_shape in zip(weights, expected_weight_shapes, strict=True):
        if weight.shape != expected_shape:
            raise ValueError(
                "compiled H100 whole-layer weight shape conflicts with the locked layer"
            )
        if weight.dtype != torch.bfloat16:
            raise ValueError("compiled H100 whole-layer weights must use BF16")
        if not weight.is_contiguous():
            raise ValueError("compiled H100 whole-layer weights must be contiguous")
    requires_grad_names = [
        name
        for name, tensor in zip(
            ("hidden_states", "cos", "sin"),
            (hidden_states, cos, sin),
            strict=True,
        )
        if tensor.requires_grad
    ]
    if requires_grad_names:
        raise UnsupportedH100Path(
            "compiled H100 whole-layer ops reject requires_grad activations; found "
            + ", ".join(requires_grad_names)
        )
    if not hidden_states.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("compiled H100 whole-layer activations must be contiguous")

    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise ValueError("compiled H100 whole-layer arguments must use distinct storage")
    return seqlen


def _rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """Reproduce pinned Gemma4RMSNorm's exact FP32 operation order."""

    float_states = hidden_states.float()
    mean_squared = float_states.pow(2).mean(-1, keepdim=True) + 1e-6
    normed = float_states * torch.pow(mean_squared, -0.5)
    if weight is not None:
        normed = normed * weight.float()
    return normed.type_as(hidden_states)


def _apply_rotary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reproduce pinned apply_rotary_pos_emb(..., unsqueeze_dim=2)."""

    first = hidden_states[..., : hidden_states.shape[-1] // 2]
    second = hidden_states[..., hidden_states.shape[-1] // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return (hidden_states * cos.unsqueeze(2)) + (rotated * sin.unsqueeze(2))


def _validate_layer_outputs(
    out: torch.Tensor,
    lse: torch.Tensor,
    explicit_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = explicit_inputs[0]
    expected_out = (1, hidden_states.shape[1], _HIDDEN_SIZE)
    expected_lse = (1, _LOCAL_GEOMETRY[0], hidden_states.shape[1])
    if (
        out.shape != expected_out
        or out.dtype != torch.bfloat16
        or out.device != hidden_states.device
    ):
        raise RuntimeError("compiled H100 whole-layer op returned an invalid BF16 output")
    if (
        lse.shape != expected_lse
        or lse.dtype != torch.float32
        or lse.device != hidden_states.device
    ):
        raise RuntimeError("compiled H100 whole-layer op returned an invalid FP32 LSE")
    if out.requires_grad or lse.requires_grad:
        raise RuntimeError("compiled H100 whole-layer outputs must not require grad")
    input_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_inputs}
    out_pointer = out.untyped_storage().data_ptr()
    lse_pointer = lse.untyped_storage().data_ptr()
    if out_pointer in input_pointers or lse_pointer in input_pointers or out_pointer == lse_pointer:
        raise RuntimeError("compiled H100 whole-layer outputs must use fresh storage")
    return out, lse


def _h100_local_layer_impl(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = (
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    seqlen = _validate_layer_real_inputs(
        hidden_states,
        cos,
        sin,
        position_ids,
        packed_sequence_ids,
        weights,
        geometry=_LOCAL_GEOMETRY,
        expected_weight_shapes=(
            (32 * 256, _HIDDEN_SIZE),
            (16 * 256, _HIDDEN_SIZE),
            (16 * 256, _HIDDEN_SIZE),
            (_HIDDEN_SIZE, 32 * 256),
            (256,),
            (256,),
        ),
    )
    hidden_shape = (1, seqlen, -1, 256)
    query = torch.nn.functional.linear(hidden_states, q_proj_weight).view(hidden_shape)
    query = _apply_rotary(_rms_norm(query, q_norm_weight), cos, sin).transpose(1, 2)
    key = torch.nn.functional.linear(hidden_states, k_proj_weight).view(hidden_shape)
    value = torch.nn.functional.linear(hidden_states, v_proj_weight).view(hidden_shape)
    key = _apply_rotary(_rms_norm(key, k_norm_weight), cos, sin).transpose(1, 2)
    value = _rms_norm(value, None).transpose(1, 2)
    attention, lse = _h100_local_impl(query, key, value, position_ids, packed_sequence_ids)
    flattened = attention.reshape(1, seqlen, -1).contiguous()
    out = torch.nn.functional.linear(flattened, o_proj_weight)
    return _validate_layer_outputs(
        out,
        lse,
        (
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            *weights,
        ),
    )


def _h100_global_layer_impl(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    packed_sequence_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = (
        q_proj_weight,
        k_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    seqlen = _validate_layer_real_inputs(
        hidden_states,
        cos,
        sin,
        position_ids,
        packed_sequence_ids,
        weights,
        geometry=_GLOBAL_GEOMETRY,
        expected_weight_shapes=(
            (32 * 512, _HIDDEN_SIZE),
            (4 * 512, _HIDDEN_SIZE),
            (_HIDDEN_SIZE, 32 * 512),
            (512,),
            (512,),
        ),
    )
    hidden_shape = (1, seqlen, -1, 512)
    query = torch.nn.functional.linear(hidden_states, q_proj_weight).view(hidden_shape)
    query = _apply_rotary(_rms_norm(query, q_norm_weight), cos, sin).transpose(1, 2)
    projected_key = torch.nn.functional.linear(hidden_states, k_proj_weight).view(hidden_shape)
    key = _apply_rotary(_rms_norm(projected_key, k_norm_weight), cos, sin).transpose(1, 2)
    value = _rms_norm(projected_key, None).transpose(1, 2)
    attention, lse = _h100_global_impl(query, key, value, position_ids, packed_sequence_ids)
    flattened = attention.reshape(1, seqlen, -1).contiguous()
    out = torch.nn.functional.linear(flattened, o_proj_weight)
    return _validate_layer_outputs(
        out,
        lse,
        (
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            *weights,
        ),
    )


def _validate_global_static_cache_decode_real_inputs(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
) -> int:
    """Validate EXP-0024's opaque global Q1 StaticCache mutation ABI."""

    explicit_tensors = (
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        *weights,
    )
    if hidden_states.shape != (1, 1, _HIDDEN_SIZE):
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode requires B1, Q1, and hidden size 5376"
        )
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in explicit_tensors
    ):
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode tensors must share one CUDA device"
        )
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability != (9, 0):
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode requires compute capability 9.0, "
            f"found {capability}"
        )
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (hidden_states, cos, sin, cache_k, cache_v, *weights)
    ):
        raise ValueError(
            "compiled H100 global StaticCache decode activations, cache, and weights must use BF16"
        )
    if cos.shape != (1, 1, 512) or sin.shape != (1, 1, 512):
        raise ValueError("compiled H100 global StaticCache decode rotary tensors are invalid")
    if position_ids.shape != (1, 1) or position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "compiled H100 global StaticCache decode position_ids must be INT32/INT64 (1, 1)"
        )
    if cache_length.ndim != 0 or cache_length.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "compiled H100 global StaticCache decode length must be a scalar integer tensor"
        )
    if cache_k.ndim != 4 or cache_k.shape[:2] != (1, 4) or cache_k.shape[-1] != 512:
        raise ValueError("compiled H100 global StaticCache K backing has the wrong geometry")
    if cache_v.shape != cache_k.shape:
        raise ValueError("compiled H100 global StaticCache K/V backings must have equal geometry")
    capacity = cache_k.shape[2]
    logical_length = int(cache_length.detach().item())
    if not (1 <= logical_length < _MAX_SEQLEN * 256):
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode requires a nonempty in-range prefix"
        )
    if capacity > _MAX_SEQLEN * 256 or logical_length + 1 >= capacity:
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode requires spare unwritten capacity"
        )
    if int(position_ids.detach().item()) != logical_length:
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode position must equal the active prefix length"
        )
    expected_weight_shapes = (
        (32 * 512, _HIDDEN_SIZE),
        (4 * 512, _HIDDEN_SIZE),
        (_HIDDEN_SIZE, 32 * 512),
        (512,),
        (512,),
    )
    if len(weights) != len(expected_weight_shapes):
        raise AssertionError("global StaticCache decode weight schema is inconsistent")
    for weight, expected_shape in zip(weights, expected_weight_shapes, strict=True):
        if weight.shape != expected_shape or not weight.is_contiguous():
            raise ValueError(
                "compiled H100 global StaticCache decode weight conflicts with the locked layer"
            )
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin, cache_k, cache_v)):
        raise UnsupportedH100Path(
            "compiled H100 global StaticCache decode rejects requires_grad activations/cache"
        )
    if any(
        not tensor.is_contiguous()
        for tensor in (hidden_states, cos, sin, position_ids, cache_k, cache_v, cache_length)
    ):
        raise ValueError("compiled H100 global StaticCache decode inputs must be contiguous")
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise ValueError(
            "compiled H100 global StaticCache decode arguments must use distinct storage"
        )
    return logical_length


def _validate_local_static_cache_decode_real_inputs(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
) -> int:
    """Validate EXP-0027's opaque local Q1 sliding-cache mutation ABI."""

    explicit_tensors = (
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        *weights,
    )
    if hidden_states.shape != (1, 1, _HIDDEN_SIZE):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow decode requires B1, Q1, "
            "and hidden size 5376"
        )
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in explicit_tensors
    ):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow decode tensors must share one CUDA device"
        )
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability != (9, 0):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow decode requires compute capability 9.0, "
            f"found {capability}"
        )
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (hidden_states, cos, sin, cache_k, cache_v, *weights)
    ):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode activations, cache, and "
            "weights must use BF16"
        )
    if cos.shape != (1, 1, 256) or sin.shape != (1, 1, 256):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode rotary tensors are invalid"
        )
    if position_ids.shape != (1, 1) or position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode position_ids must be "
            "INT32/INT64 (1, 1)"
        )
    if cache_length.ndim != 0 or cache_length.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode length must be a scalar "
            "integer tensor"
        )
    if cache_k.shape != (1, 16, _MAX_SEQLEN, 256) or cache_v.shape != cache_k.shape:
        raise ValueError(
            "compiled H100 local StaticSlidingWindow K/V backings have the wrong geometry"
        )
    absolute_position = int(position_ids.detach().item())
    tensor_length = int(cache_length.detach().item())
    if not (1 <= absolute_position < _MAX_SEQLEN * 256):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow decode requires a nonempty "
            "in-range prefix"
        )
    if tensor_length != min(absolute_position, _MAX_SEQLEN):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow tensor length conflicts with "
            "the absolute position"
        )
    expected_weight_shapes = (
        (32 * 256, _HIDDEN_SIZE),
        (16 * 256, _HIDDEN_SIZE),
        (16 * 256, _HIDDEN_SIZE),
        (_HIDDEN_SIZE, 32 * 256),
        (256,),
        (256,),
    )
    if len(weights) != len(expected_weight_shapes):
        raise AssertionError("local StaticSlidingWindow decode weight schema is inconsistent")
    for weight, expected_shape in zip(weights, expected_weight_shapes, strict=True):
        if weight.shape != expected_shape or not weight.is_contiguous():
            raise ValueError(
                "compiled H100 local StaticSlidingWindow decode weight conflicts with "
                "the locked layer"
            )
    if any(tensor.requires_grad for tensor in (hidden_states, cos, sin, cache_k, cache_v)):
        raise UnsupportedH100Path(
            "compiled H100 local StaticSlidingWindow decode rejects requires_grad "
            "activations/cache"
        )
    if any(not tensor.is_contiguous() for tensor in explicit_tensors):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode inputs must be contiguous"
        )
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in explicit_tensors}
    if len(storage_pointers) != len(explicit_tensors):
        raise ValueError(
            "compiled H100 local StaticSlidingWindow decode arguments must use distinct storage"
        )
    return absolute_position


def _h100_local_static_cache_decode_impl(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror pinned Q1 sliding-cache mutation, then run local packed FA4."""

    weights = (
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    absolute_position = _validate_local_static_cache_decode_real_inputs(
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        weights,
    )
    hidden_shape = (1, 1, -1, 256)
    query = torch.nn.functional.linear(hidden_states, q_proj_weight).view(hidden_shape)
    query = _apply_rotary(_rms_norm(query, q_norm_weight), cos, sin)
    key = torch.nn.functional.linear(hidden_states, k_proj_weight).view(hidden_shape)
    value = torch.nn.functional.linear(hidden_states, v_proj_weight).view(hidden_shape)
    key = _apply_rotary(_rms_norm(key, k_norm_weight), cos, sin).transpose(1, 2)
    value = _rms_norm(value, None).transpose(1, 2)

    if absolute_position < _MAX_SEQLEN:
        cache_position = (
            torch.arange(1, dtype=cache_length.dtype, device=cache_k.device) + cache_length
        )
        cache_k.index_copy_(2, cache_position, key)
        cache_v.index_copy_(2, cache_position, value)
        cache_length.add_(1)
        active_length = absolute_position + 1
    else:
        new_keys = cache_k.roll(-1, dims=2)
        new_values = cache_v.roll(-1, dims=2)
        last_index = torch.tensor([-1], dtype=torch.int64, device=cache_k.device)
        new_keys[:, :, last_index] = key
        new_values[:, :, last_index] = value
        cache_k.copy_(new_keys)
        cache_v.copy_(new_values)
        active_length = _MAX_SEQLEN

    packed_query = query.reshape(1, _LOCAL_GEOMETRY[0], _LOCAL_GEOMETRY[2])
    packed_key = cache_k[:, :, :active_length, :].transpose(1, 2).reshape(
        active_length,
        _LOCAL_GEOMETRY[1],
        _LOCAL_GEOMETRY[2],
    )
    packed_value = cache_v[:, :, :active_length, :].transpose(1, 2).reshape(
        active_length,
        _LOCAL_GEOMETRY[1],
        _LOCAL_GEOMETRY[2],
    )
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=cache_k.device)
    cu_seqlens_k = torch.tensor(
        [0, active_length],
        dtype=torch.int32,
        device=cache_k.device,
    )
    attention, packed_lse = fa4_local_varlen_forward(
        packed_query,
        packed_key,
        packed_value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=active_length,
    )
    flattened = attention.reshape(1, 1, -1).contiguous()
    out = torch.nn.functional.linear(flattened, o_proj_weight)
    lse = packed_lse.unsqueeze(0)
    return _validate_layer_outputs(
        out,
        lse,
        (
            hidden_states,
            cos,
            sin,
            position_ids,
            cache_k,
            cache_v,
            cache_length,
            *weights,
        ),
    )


def _h100_global_static_cache_decode_impl(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_length: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Q/K/V, mutate one global cache slot, and run exact lower-right FA4."""

    weights = (
        q_proj_weight,
        k_proj_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
    )
    logical_length = _validate_global_static_cache_decode_real_inputs(
        hidden_states,
        cos,
        sin,
        position_ids,
        cache_k,
        cache_v,
        cache_length,
        weights,
    )
    hidden_shape = (1, 1, -1, 512)
    query = torch.nn.functional.linear(hidden_states, q_proj_weight).view(hidden_shape)
    query = _apply_rotary(_rms_norm(query, q_norm_weight), cos, sin)
    projected_key = torch.nn.functional.linear(hidden_states, k_proj_weight).view(hidden_shape)
    key = _apply_rotary(_rms_norm(projected_key, k_norm_weight), cos, sin)
    value = _rms_norm(projected_key, None)

    cache_position = torch.arange(1, device=cache_k.device) + cache_length
    cache_length.add_(1)
    cache_k.index_copy_(2, cache_position, key.transpose(1, 2))
    cache_v.index_copy_(2, cache_position, value.transpose(1, 2))

    active_length = logical_length + 1
    active_k = cache_k[:, :, :active_length, :].transpose(1, 2)
    active_v = cache_v[:, :, :active_length, :].transpose(1, 2)
    if active_length <= _MAX_SEQLEN:
        prefix = torch.zeros(
            (1, active_length - 1, 32, 512),
            dtype=query.dtype,
            device=query.device,
        )
        padded_query = torch.cat((prefix, query), dim=1)
        attention, full_lse = fa4_global_text_forward(padded_query, active_k, active_v)
        attention = attention[:, -1:, :, :]
        lse = full_lse[:, :, -1:].contiguous()
    else:
        attention, lse = fa4_global_forward_only(query, active_k, active_v)
    flattened = attention.reshape(1, 1, -1).contiguous()
    out = torch.nn.functional.linear(flattened, o_proj_weight)
    return _validate_layer_outputs(
        out,
        lse,
        (
            hidden_states,
            cos,
            sin,
            position_ids,
            cache_k,
            cache_v,
            cache_length,
            *weights,
        ),
    )


def _fake_global_static_cache_decode(
    hidden_states: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _position_ids: torch.Tensor,
    _cache_k: torch.Tensor,
    _cache_v: torch.Tensor,
    _cache_length: torch.Tensor,
    _q_proj_weight: torch.Tensor,
    _k_proj_weight: torch.Tensor,
    _o_proj_weight: torch.Tensor,
    _q_norm_weight: torch.Tensor,
    _k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (hidden_states.shape[0], hidden_states.shape[1], _HIDDEN_SIZE),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    lse = torch.empty(
        (hidden_states.shape[0], _GLOBAL_GEOMETRY[0], hidden_states.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return out, lse


def _fake_local_static_cache_decode(
    hidden_states: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _position_ids: torch.Tensor,
    _cache_k: torch.Tensor,
    _cache_v: torch.Tensor,
    _cache_length: torch.Tensor,
    _q_proj_weight: torch.Tensor,
    _k_proj_weight: torch.Tensor,
    _v_proj_weight: torch.Tensor,
    _o_proj_weight: torch.Tensor,
    _q_norm_weight: torch.Tensor,
    _k_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (hidden_states.shape[0], hidden_states.shape[1], _HIDDEN_SIZE),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    lse = torch.empty(
        (hidden_states.shape[0], _LOCAL_GEOMETRY[0], hidden_states.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return out, lse


def _fake_layer_forward(
    hidden_states: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _position_ids: torch.Tensor,
    _packed_sequence_ids: torch.Tensor,
    *_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (hidden_states.shape[0], hidden_states.shape[1], _HIDDEN_SIZE),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    lse = torch.empty(
        (hidden_states.shape[0], _LOCAL_GEOMETRY[0], hidden_states.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return out, lse


def _validate_layer_call_boundary(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Reject autograd before ``custom_op`` disables grad mode for its body."""

    if torch.is_grad_enabled():
        raise UnsupportedH100Path(
            "compiled H100 whole-layer ops require global inference/no-grad mode"
        )
    requires_grad_names = [
        name
        for name, tensor in zip(
            ("hidden_states", "cos", "sin"),
            (hidden_states, cos, sin),
            strict=True,
        )
        if tensor.requires_grad
    ]
    if requires_grad_names:
        raise UnsupportedH100Path(
            "compiled H100 whole-layer ops reject requires_grad activations; found "
            + ", ".join(requires_grad_names)
        )


_torch_library = getattr(torch, "library", None)
_custom_op: Callable | None = getattr(_torch_library, "custom_op", None)
_register_fake: Callable | None = getattr(_torch_library, "register_fake", None)
CUSTOM_OPS_AVAILABLE = callable(_custom_op) and callable(_register_fake)


if CUSTOM_OPS_AVAILABLE:
    h100_local_fwd = _custom_op(LOCAL_OP_NAME, mutates_args=())(_h100_local_impl)
    _register_fake(h100_local_fwd)(_fake_forward)

    h100_global_fwd = _custom_op(GLOBAL_OP_NAME, mutates_args=())(_h100_global_impl)
    _register_fake(h100_global_fwd)(_fake_forward)

    h100_local_layer_op = _custom_op(LOCAL_LAYER_OP_NAME, mutates_args=())(_h100_local_layer_impl)
    _register_fake(h100_local_layer_op)(_fake_layer_forward)

    h100_global_layer_op = _custom_op(GLOBAL_LAYER_OP_NAME, mutates_args=())(
        _h100_global_layer_impl
    )
    _register_fake(h100_global_layer_op)(_fake_layer_forward)

    h100_global_static_cache_decode_op = _custom_op(
        GLOBAL_STATIC_CACHE_DECODE_OP_NAME,
        mutates_args={"cache_k", "cache_v", "cache_length"},
    )(_h100_global_static_cache_decode_impl)
    _register_fake(h100_global_static_cache_decode_op)(
        _fake_global_static_cache_decode
    )

    h100_local_static_cache_decode_op = _custom_op(
        LOCAL_STATIC_CACHE_DECODE_OP_NAME,
        mutates_args={"cache_k", "cache_v", "cache_length"},
    )(_h100_local_static_cache_decode_impl)
    _register_fake(h100_local_static_cache_decode_op)(
        _fake_local_static_cache_decode
    )

    def h100_local_layer_fwd(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        packed_sequence_ids: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_layer_call_boundary(hidden_states, cos, sin)
        return h100_local_layer_op(
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        )

    def h100_global_layer_fwd(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        packed_sequence_ids: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_layer_call_boundary(hidden_states, cos, sin)
        return h100_global_layer_op(
            hidden_states,
            cos,
            sin,
            position_ids,
            packed_sequence_ids,
            q_proj_weight,
            k_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        )

    def h100_global_static_cache_decode_fwd(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_length: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_layer_call_boundary(hidden_states, cos, sin)
        return h100_global_static_cache_decode_op(
            hidden_states,
            cos,
            sin,
            position_ids,
            cache_k,
            cache_v,
            cache_length,
            q_proj_weight,
            k_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        )

    def h100_local_static_cache_decode_fwd(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_length: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_layer_call_boundary(hidden_states, cos, sin)
        return h100_local_static_cache_decode_op(
            hidden_states,
            cos,
            sin,
            position_ids,
            cache_k,
            cache_v,
            cache_length,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        )

else:

    def _unavailable(*_args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "H100 opaque forward ops require PyTorch with "
            "torch.library.custom_op and torch.library.register_fake"
        )

    h100_local_fwd = _unavailable
    h100_global_fwd = _unavailable
    h100_local_layer_op = _unavailable
    h100_global_layer_op = _unavailable
    h100_global_static_cache_decode_op = _unavailable
    h100_local_static_cache_decode_op = _unavailable
    h100_local_layer_fwd = _unavailable
    h100_global_layer_fwd = _unavailable
    h100_global_static_cache_decode_fwd = _unavailable
    h100_local_static_cache_decode_fwd = _unavailable


__all__ = [
    "CUSTOM_OPS_AVAILABLE",
    "GLOBAL_OP_NAME",
    "GLOBAL_LAYER_OP_NAME",
    "GLOBAL_STATIC_CACHE_DECODE_OP_NAME",
    "LOCAL_OP_NAME",
    "LOCAL_LAYER_OP_NAME",
    "LOCAL_STATIC_CACHE_DECODE_OP_NAME",
    "h100_global_fwd",
    "h100_global_layer_fwd",
    "h100_global_layer_op",
    "h100_global_static_cache_decode_fwd",
    "h100_global_static_cache_decode_op",
    "h100_local_fwd",
    "h100_local_layer_fwd",
    "h100_local_layer_op",
    "h100_local_static_cache_decode_fwd",
    "h100_local_static_cache_decode_op",
]
