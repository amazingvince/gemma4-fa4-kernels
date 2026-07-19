"""Contract adapters for the pinned FA4 CuTe SM90 path.

M1 covers fixed-length local-d256 text forward and autograd backward plus the
global-d512 correctness path. Global forward is an exact two-launch
composition over V/O slabs; its custom backward preserves FP32 accumulation
across both slabs before the final BF16 conversion.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from .model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION, AttentionLayerSpec


class UnsupportedH100Path(RuntimeError):
    """Raised when an input would leave the validated SM90 contract."""


def _is_fake_tensor(tensor: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except ImportError:  # pragma: no cover - pinned PyTorch provides it
        return False
    return isinstance(tensor, FakeTensor)


def _load_flash_attn_func() -> Callable:
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "pinned flash-attn-4 is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return flash_attn_func


def _load_flash_attn_varlen_func() -> Callable:
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "pinned flash-attn-4 varlen is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return flash_attn_varlen_func


def _load_local_vision_mask() -> Callable:
    try:
        from .h100_masks import gemma4_local_vision_mask
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "the pinned CuTe DSL local-vision mask is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return gemma4_local_vision_mask


def _load_local_varlen_mask() -> Callable:
    try:
        from .h100_masks import gemma4_local_varlen_mask
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "the pinned CuTe DSL local-varlen mask is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return gemma4_local_varlen_mask


def _load_global_backward_func() -> Callable:
    try:
        from flash_attn.cute.interface import _flash_attn_bwd_gemma4_global_d512
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "the pinned H100 global-backward patch is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return _flash_attn_bwd_gemma4_global_d512


def _require_sm90(device: torch.device) -> None:
    if device.type != "cuda":
        raise UnsupportedH100Path("H100 FA4 inputs must be CUDA tensors")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise UnsupportedH100Path(f"H100 FA4 requires compute capability 9.0, found {capability}")


def _validate_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
    *,
    require_sm90: bool = True,
) -> None:
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q, k, and v must be rank-4 B/S/H/D tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on one device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("the H100 M1 path accepts BF16 q, k, and v only")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("q, k, and v must use contiguous BSHD storage")
    fake_inputs = [_is_fake_tensor(t) for t in (q, k, v)]
    if any(fake_inputs) and not all(fake_inputs):
        raise ValueError("q, k, and v must all be real tensors or all be fake tensors")
    if not all(fake_inputs):
        if any(t.data_ptr() % 16 != 0 for t in (q, k, v)):
            raise ValueError("q, k, and v base pointers must be 16-byte aligned")

    batch, seqlen, q_heads, q_dim = q.shape
    if batch <= 0 or seqlen <= 0:
        raise ValueError("batch and sequence length must be positive")
    expected_k = (batch, seqlen, spec.num_kv_heads, spec.head_dim_qk)
    expected_v = (batch, seqlen, spec.num_kv_heads, spec.head_dim_v)
    if (q_heads, q_dim) != (spec.num_q_heads, spec.head_dim_qk):
        raise ValueError("q shape does not match the attention spec")
    if k.shape != expected_k or v.shape != expected_v:
        raise ValueError("k/v shapes do not match the attention spec")
    if spec.qhead_per_kvhead not in {1, 2, 4, 8}:
        raise ValueError("the validated GQA ratios are 1, 2, 4, and 8")
    if not all(fake_inputs) and k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        raise ValueError("K and V must be distinct, non-aliasing prepared operands")
    if require_sm90:
        _require_sm90(q.device)


def _validate_local_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
    *,
    require_sm90: bool = True,
) -> None:
    _validate_bshd(q, k, v, spec, require_sm90=require_sm90)
    if q.shape[0] != 1 or q.shape[1] > 1025:
        raise UnsupportedH100Path("local M1 evidence covers only B=1 and 1 <= S <= 1025")
    if (
        spec.kind != "sliding_attention"
        or spec.head_dim_qk != 256
        or spec.head_dim_v != 256
        or spec.num_q_heads != 32
        or spec.softmax_scale != 1.0
        or spec.sliding_window != 1024
        or not spec.is_causal
    ):
        raise UnsupportedH100Path("only the locked local d256 attention contract is enabled")


def _prepare_vision_block_ids(
    vision_block_ids: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    expected_shape = q.shape[:2]
    if vision_block_ids.shape != expected_shape:
        raise ValueError(
            f"vision_block_ids must have shape {expected_shape}, "
            f"got {tuple(vision_block_ids.shape)}"
        )
    if vision_block_ids.device != q.device:
        raise ValueError("vision_block_ids must be on the same device as q, k, and v")
    normalized = _normalize_int32_metadata(vision_block_ids, name="vision_block_ids")
    # EXP-0007 is B=1. A private rank-1 auxiliary avoids CuTe's ambiguous
    # leading-dimension inference for the public (1, 1) S=1 input.
    return normalized.view(-1)


def _normalize_int32_metadata(
    values: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if values.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use INT32 or INT64 values")
    if values.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if values.dtype == torch.int64 and not _is_fake_tensor(values):
        minimum, maximum = torch.aminmax(values)
        int32 = torch.iinfo(torch.int32)
        if minimum.item() < int32.min or maximum.item() > int32.max:
            raise ValueError(f"INT64 {name} values must fit exactly in INT32")
    return values.to(dtype=torch.int32).contiguous()


def _validate_local_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    spec: AttentionLayerSpec,
    *,
    require_sm90: bool = True,
) -> tuple[list[int] | None, list[int] | None]:
    if any(t.ndim != 3 for t in (q, k, v)):
        raise ValueError("packed q, k, and v must be rank-3 T/H/D tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("packed q, k, and v must be on one device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("the H100 M1 varlen path accepts BF16 q, k, and v only")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("packed q, k, and v must use contiguous THD storage")
    fake_inputs = [_is_fake_tensor(t) for t in (q, k, v)]
    if any(fake_inputs) and not all(fake_inputs):
        raise ValueError("packed q, k, and v must all be real tensors or all be fake tensors")
    if not all(fake_inputs):
        if any(t.data_ptr() % 16 != 0 for t in (q, k, v)):
            raise ValueError("packed q, k, and v base pointers must be 16-byte aligned")
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise ValueError("K and V must be distinct, non-aliasing prepared operands")

    if q.shape[0] <= 0 or k.shape[0] <= 0:
        raise ValueError("packed Q and K totals must be positive")
    if q.shape[1:] != (spec.num_q_heads, spec.head_dim_qk):
        raise ValueError("packed q shape does not match the attention spec")
    if k.shape[1:] != (spec.num_kv_heads, spec.head_dim_qk):
        raise ValueError("packed k shape does not match the attention spec")
    if v.shape != (k.shape[0], spec.num_kv_heads, spec.head_dim_v):
        raise ValueError("packed v shape does not match the attention spec")
    if (
        spec.kind != "sliding_attention"
        or spec.head_dim_qk != 256
        or spec.head_dim_v != 256
        or spec.num_q_heads != 32
        or spec.num_kv_heads != 16
        or spec.softmax_scale != 1.0
        or spec.sliding_window != 1024
        or not spec.is_causal
    ):
        raise UnsupportedH100Path("only the locked local d256 GQA-2 varlen contract is enabled")

    if (
        isinstance(max_seqlen_q, bool)
        or not isinstance(max_seqlen_q, int)
        or isinstance(max_seqlen_k, bool)
        or not isinstance(max_seqlen_k, int)
    ):
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0 or max_seqlen_k > 1025:
        raise UnsupportedH100Path("varlen M1 requires 1 <= max Sq <= max Sk <= 1025")
    if max_seqlen_q > max_seqlen_k:
        raise UnsupportedH100Path("varlen M1 requires max Sq <= max Sk")

    cumulative = (cu_seqlens_q, cu_seqlens_k)
    if any(t.ndim != 1 or t.numel() < 2 for t in cumulative):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be rank-1 B+1 tensors")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same batch count")
    if any(t.device != q.device for t in cumulative):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must share the q/k/v device")
    if any(t.dtype != torch.int32 for t in cumulative):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must use INT32")
    if any(not t.is_contiguous() for t in cumulative):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be contiguous")
    fake_cumulative = [_is_fake_tensor(t) for t in cumulative]
    if any(fake_cumulative) and not all(fake_cumulative):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must share real/fake tensor mode")
    if all(fake_cumulative) != all(fake_inputs):
        raise ValueError("packed q/k/v and cumulative arrays must share real/fake tensor mode")

    if all(fake_inputs):
        if require_sm90:
            _require_sm90(q.device)
        return None, None

    q_values = [int(value) for value in cu_seqlens_q.detach().cpu().tolist()]
    k_values = [int(value) for value in cu_seqlens_k.detach().cpu().tolist()]
    if q_values[0] != 0 or k_values[0] != 0:
        raise ValueError("cumulative arrays must start at zero")
    if q_values[-1] != q.shape[0] or k_values[-1] != k.shape[0]:
        raise ValueError("cumulative arrays must end at the packed Q/K totals")
    q_lengths = [end - start for start, end in zip(q_values[:-1], q_values[1:], strict=True)]
    k_lengths = [end - start for start, end in zip(k_values[:-1], k_values[1:], strict=True)]
    if any(length <= 0 for length in (*q_lengths, *k_lengths)):
        raise ValueError("empty packed segments are unsupported")
    if any(q_len > k_len for q_len, k_len in zip(q_lengths, k_lengths, strict=True)):
        raise UnsupportedH100Path("each packed sequence requires Sq <= Sk")
    if max(q_lengths) != max_seqlen_q or max(k_lengths) != max_seqlen_k:
        raise ValueError("max_seqlen_q/max_seqlen_k must equal the cumulative maxima")
    if require_sm90:
        _require_sm90(q.device)
    return q_lengths, k_lengths


def _prepare_packed_k_metadata(
    values: torch.Tensor | None,
    k: torch.Tensor,
    *,
    name: str,
    default: int,
) -> torch.Tensor:
    if values is None:
        return torch.full((k.shape[0],), default, dtype=torch.int32, device=k.device)
    if values.shape != (k.shape[0],):
        raise ValueError(f"{name} must have shape ({k.shape[0]},)")
    if values.device != k.device:
        raise ValueError(f"{name} must share the q/k/v device")
    if _is_fake_tensor(values) != _is_fake_tensor(k):
        raise ValueError(f"{name} and packed q/k/v must share real/fake tensor mode")
    return _normalize_int32_metadata(values, name=name)


def _validate_local_varlen_result(
    result,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("pinned flash_attn_varlen_func must return (out, lse)")
    out, lse = result
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise RuntimeError("FA4 returned an invalid local varlen output contract")
    expected_lse = (q.shape[1], q.shape[0])
    if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
        raise RuntimeError("FA4 returned an invalid local varlen FP32 LSE contract")
    return out, lse


def _validate_local_result(
    result,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("pinned flash_attn_func must return (out, lse)")
    out, lse = result
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise RuntimeError("FA4 returned an invalid local output contract")
    expected_lse = (q.shape[0], q.shape[2], q.shape[1])
    if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
        raise RuntimeError("FA4 returned an invalid FP32 LSE contract")
    return out, lse


def _validate_global_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
    *,
    require_sm90: bool = True,
) -> None:
    _validate_bshd(q, k, v, spec, require_sm90=require_sm90)
    if q.shape[0] != 1 or q.shape[1] > 1024:
        raise UnsupportedH100Path("global M1 evidence covers only B=1 and 1 <= S <= 1024")
    if (
        spec.kind != "full_attention"
        or spec.head_dim_qk != 512
        or spec.head_dim_v != 512
        or spec.num_q_heads != 32
        or spec.num_kv_heads != 4
        or spec.softmax_scale != 1.0
        or spec.sliding_window is not None
        or not spec.is_causal
    ):
        raise UnsupportedH100Path("only the locked global d512 text-forward contract is enabled")


def fa4_local_text_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionLayerSpec = SLIDING_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length Gemma local text forward on the pinned SM90 FA4 path.

    Inputs and output use 16-byte-aligned contiguous ``(B, S, H, D)`` at B=1
    and ``1 <= S <= 1025``. LSE is FP32 with shape ``(B, Hq, S)``.
    Vision-block masking is a later gated path and is not silently
    approximated here.
    """

    _validate_local_bshd(q, k, v, spec)
    result = _load_flash_attn_func()(
        q,
        k,
        v,
        causal=True,
        window_size=(spec.fa_window_size_left, 0),
        softmax_scale=1.0,
        num_splits=1,
        pack_gqa=False,
        return_lse=True,
    )
    return _validate_local_result(result, q)


def fa4_local_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    vision_block_ids: torch.Tensor | None = None,
    spec: AttentionLayerSpec = SLIDING_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length Gemma local attention with the exact vision overlay.

    ``vision_block_ids=None`` preserves the accepted native text path. With
    IDs, the custom CuTe predicate owns the complete local mask, so native
    causal/window flags are deliberately disabled for both forward and
    autograd backward.
    """

    if vision_block_ids is None:
        return fa4_local_text_forward(q, k, v, spec=spec)

    _validate_local_bshd(q, k, v, spec)
    normalized_ids = _prepare_vision_block_ids(vision_block_ids, q)
    result = _load_flash_attn_func()(
        q,
        k,
        v,
        causal=False,
        window_size=(None, None),
        softmax_scale=1.0,
        num_splits=1,
        pack_gqa=False,
        mask_mod=_load_local_vision_mask(),
        aux_tensors=[normalized_ids],
        return_lse=True,
    )
    return _validate_local_result(result, q)


def fa4_local_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    spec: AttentionLayerSpec = SLIDING_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed Gemma local self-attention on the pinned SM90 FA4 path.

    Each packed Q segment represents the lower-right suffix of its K segment.
    Vision/document IDs follow the packed K stream. With no metadata the
    accepted native causal/local varlen path is used; otherwise one custom
    callable owns the complete document/window/vision predicate.
    """

    _validate_local_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        spec,
    )
    common_kwargs = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "deterministic": False,
        "return_lse": True,
    }
    if vision_block_ids is None and document_ids is None:
        result = _load_flash_attn_varlen_func()(
            q,
            k,
            v,
            causal=True,
            window_size=(spec.fa_window_size_left, 0),
            **common_kwargs,
        )
    else:
        vision_ids = _prepare_packed_k_metadata(
            vision_block_ids,
            k,
            name="vision_block_ids",
            default=-1,
        )
        documents = _prepare_packed_k_metadata(
            document_ids,
            k,
            name="document_ids",
            default=0,
        )
        result = _load_flash_attn_varlen_func()(
            q,
            k,
            v,
            causal=False,
            window_size=(None, None),
            mask_mod=_load_local_varlen_mask(),
            aux_tensors=[vision_ids, documents],
            **common_kwargs,
        )
    return _validate_local_varlen_result(result, q)


def fa4_global_text_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionLayerSpec = GLOBAL_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length Gemma global text forward as two exact V256 slabs.

    Both launches compute the same d512 QK scores and causal softmax. Their
    d256 outputs are concatenated because ``P @ concat(V0, V1)`` equals
    ``concat(P @ V0, P @ V1)``. Inputs are restricted to the proven B=1,
    ``1 <= S <= 1024`` envelope. The H100 patch stack enables the exact
    asymmetric d512-QK/d256-V SM90 dimension/tile specialization used here;
    this adapter supplies its global-causal semantic guard.
    """

    _validate_global_bshd(q, k, v, spec)
    return _FA4GlobalTextFunction.apply(q, k, v)


class _FA4GlobalTextFunction(torch.autograd.Function):
    """Coordinate both global V slabs across one exact backward contract."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        backend = _load_flash_attn_func()
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        for v_slab in v.split(256, dim=-1):
            result = backend(
                q,
                k,
                v_slab.contiguous(),
                causal=True,
                window_size=(None, None),
                softmax_scale=1.0,
                num_splits=1,
                pack_gqa=False,
                return_lse=True,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("pinned flash_attn_func must return (out, lse)")
            out_slab, lse = result
            expected_out = (*q.shape[:-1], 256)
            expected_lse = (q.shape[0], q.shape[2], q.shape[1])
            if out_slab.shape != expected_out or out_slab.dtype != torch.bfloat16:
                raise RuntimeError("FA4 returned an invalid global output-slab contract")
            if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
                raise RuntimeError("FA4 returned an invalid global FP32 LSE contract")
            outputs.append(out_slab)
            lses.append(lse)
        if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1" and not torch.equal(
            lses[0], lses[1]
        ):
            raise RuntimeError("global V-slab launches returned different LSE values")
        out = torch.cat(outputs, dim=-1)
        ctx.save_for_backward(q, k, v, out, lses[0])
        ctx.set_materialize_grads(False)
        return out, lses[0]

    @staticmethod
    def backward(ctx, dout: torch.Tensor | None, dlse: torch.Tensor | None):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _load_global_backward_func()(q, k, v, out, dout, lse, dlse)
        return dq, dk, dv
