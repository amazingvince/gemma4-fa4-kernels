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

from .model_spec import GEMMA4_31B, GLOBAL_ATTENTION, SLIDING_ATTENTION, AttentionLayerSpec
from .sparse_schedule import (
    Gemma4LocalSparseSchedule,
    SparseScheduleWorkLimitExceeded,
    SparseTileRows,
    build_gemma4_local_sparse_schedule,
    gemma4_local_sparse_storage_upper_bound,
)

_LOCAL_CUSTOM_DENSE_MAX_SEQLEN = 1025
_LOCAL_MODEL_MAX_SEQLEN = GEMMA4_31B.max_position_embeddings
_LOCAL_SPARSE_FWD_BLOCK_SIZE = (128, 80)
_LOCAL_SPARSE_BWD_BLOCK_SIZE = (64, 64)
_LOCAL_SPARSE_METADATA_MAX_BYTES = 2 * 1024**3
_LOCAL_SPARSE_WORK_MAX_SCORE_SLOTS = 1 << 40
_GLOBAL_BACKWARD_MAX_SEQLEN = 2048
_GLOBAL_BACKWARD_RESERVE_BYTES = 2 * 1024**3


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


def _load_local_segment_mask() -> Callable:
    try:
        from .h100_masks import gemma4_local_segment_mask
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "the pinned CuTe DSL local-segment mask is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return gemma4_local_segment_mask


def _load_block_sparse_tensors_type():
    try:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "pinned FA4 block sparsity is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return BlockSparseTensorsTorch


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


def _validate_fa4_layout(tensor: torch.Tensor, *, name: str) -> None:
    """Validate the pinned CuTe DLPack/128-bit dynamic-stride contract."""

    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have unit last-dimension stride")
    if any(stride <= 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must use positive, non-broadcast strides")
    if any(stride % 8 for stride in tensor.stride()[:-1]):
        raise ValueError(f"{name} outer strides must be 16-byte aligned for BF16")
    if torch._debug_has_internal_overlap(tensor) != 0:
        raise ValueError(f"{name} must use a non-overlapping layout")


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
    for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True):
        _validate_fa4_layout(tensor, name=name)
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
    for name, tensor in zip(("packed q", "packed k", "packed v"), (q, k, v), strict=True):
        _validate_fa4_layout(tensor, name=name)
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
    if max_seqlen_q <= 0 or max_seqlen_k <= 0 or max_seqlen_k > _LOCAL_MODEL_MAX_SEQLEN:
        raise UnsupportedH100Path("native varlen text requires 1 <= max Sq <= max Sk <= 262144")
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


def _sparse_rows_to_tensors(
    rows: SparseTileRows,
    *,
    block_size: tuple[int, int],
    device: torch.device,
):
    if block_size != (rows.row_block_size, rows.column_block_size):
        raise ValueError("sparse tensor block size must match its tile rows")
    sparse_type = _load_block_sparse_tensors_type()
    counts = torch.tensor(
        [[[len(row) for row in rows.rows]]],
        dtype=torch.int32,
        device=device,
    )
    indices = torch.tensor(
        [[rows.padded_indices()]],
        dtype=torch.int32,
        device=device,
    )
    # SM90 traces both dynamic empty/nonempty list branches. Keep an explicit
    # one-slot full-list sentinel so the zero-count branch never presents a
    # compile-time None tensor to the pinned sparse loader.
    full_counts = torch.zeros_like(counts)
    full_indices = torch.zeros(
        (*indices.shape[:-1], 1),
        dtype=torch.int32,
        device=device,
    )
    return sparse_type(
        mask_block_cnt=counts,
        mask_block_idx=indices,
        full_block_cnt=full_counts,
        full_block_idx=full_indices,
        block_size=block_size,
    )


def _check_local_sparse_metadata_budget(
    required: int,
    device: torch.device,
    *,
    label: str,
) -> None:
    if required > _LOCAL_SPARSE_METADATA_MAX_BYTES:
        raise UnsupportedH100Path(
            f"{label} requires {required} bytes, above the 2 GiB safety limit"
        )
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = free // 10
        if required > budget:
            raise UnsupportedH100Path(
                f"{label} requires {required} bytes, exceeding the 10% free-HBM "
                f"budget of {budget} bytes ({free} bytes free)"
            )


def _preflight_local_sparse_rectangular_upper_bound(
    q_lengths: list[int],
    k_lengths: list[int],
    device: torch.device,
) -> None:
    required = sum(
        gemma4_local_sparse_storage_upper_bound(
            q_length,
            k_length,
            forward_block_size=_LOCAL_SPARSE_FWD_BLOCK_SIZE,
            backward_block_size=_LOCAL_SPARSE_BWD_BLOCK_SIZE,
        )
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )
    _check_local_sparse_metadata_budget(
        required,
        device,
        label="exact sparse metadata rectangular upper bound",
    )


def _preflight_local_sparse_metadata(
    schedules: list[Gemma4LocalSparseSchedule],
    device: torch.device,
    *,
    num_q_heads: int,
) -> None:
    required = sum(schedule.storage_bytes for schedule in schedules)
    _check_local_sparse_metadata_budget(
        required,
        device,
        label="exact sparse metadata",
    )
    scheduled_score_slots = sum(
        schedule.scheduled_score_slots(num_q_heads) for schedule in schedules
    )
    if scheduled_score_slots > _LOCAL_SPARSE_WORK_MAX_SCORE_SLOTS:
        raise UnsupportedH100Path(
            "exact sparse metadata schedules "
            f"{scheduled_score_slots} padded score slots, above the "
            f"{_LOCAL_SPARSE_WORK_MAX_SCORE_SLOTS} safety limit"
        )


def _fa4_local_varlen_sparse_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: list[int],
    k_lengths: list[int],
    vision_ids: torch.Tensor,
    documents: torch.Tensor,
    spec: AttentionLayerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    _preflight_local_sparse_rectangular_upper_bound(q_lengths, k_lengths, q.device)
    vision_values = vision_ids.detach().cpu()
    document_values = documents.detach().cpu()
    schedules: list[Gemma4LocalSparseSchedule] = []
    k_start = 0
    remaining_work = _LOCAL_SPARSE_WORK_MAX_SCORE_SLOTS
    for q_length, k_length in zip(q_lengths, k_lengths, strict=True):
        k_end = k_start + k_length
        if remaining_work <= 0:
            raise UnsupportedH100Path(
                "exact sparse metadata exhausts the padded-score work safety limit; "
                "split the request or shorten the vision span"
            )
        try:
            schedule = build_gemma4_local_sparse_schedule(
                vision_values[k_start:k_end].tolist(),
                document_values[k_start:k_end].tolist(),
                q_length=q_length,
                k_length=k_length,
                sliding_window=spec.sliding_window,
                forward_block_size=_LOCAL_SPARSE_FWD_BLOCK_SIZE,
                backward_block_size=_LOCAL_SPARSE_BWD_BLOCK_SIZE,
                num_q_heads=spec.num_q_heads,
                max_scheduled_score_slots=remaining_work,
            )
        except SparseScheduleWorkLimitExceeded as exc:
            raise UnsupportedH100Path(
                "exact sparse metadata exceeds the padded-score work safety limit; "
                "split the request or shorten the vision span"
            ) from exc
        schedules.append(schedule)
        remaining_work -= schedule.scheduled_score_slots(spec.num_q_heads)
        k_start = k_end
    _preflight_local_sparse_metadata(
        schedules,
        q.device,
        num_q_heads=spec.num_q_heads,
    )

    backend = _load_flash_attn_func()
    mask_mod = _load_local_segment_mask()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_segments = torch.split(q, q_lengths, dim=0)
    k_segments = torch.split(k, k_lengths, dim=0)
    v_segments = torch.split(v, k_lengths, dim=0)
    vision_segments = torch.split(vision_ids, k_lengths, dim=0)
    document_segments = torch.split(documents, k_lengths, dim=0)
    for q_segment, k_segment, v_segment, vision_segment, document_segment, schedule in zip(
        q_segments,
        k_segments,
        v_segments,
        vision_segments,
        document_segments,
        schedules,
        strict=True,
    ):
        q_segment = q_segment.unsqueeze(0)
        k_segment = k_segment.unsqueeze(0)
        v_segment = v_segment.unsqueeze(0)
        sparse_fwd = _sparse_rows_to_tensors(
            schedule.forward,
            block_size=_LOCAL_SPARSE_FWD_BLOCK_SIZE,
            device=q.device,
        )
        sparse_bwd = _sparse_rows_to_tensors(
            schedule.backward,
            block_size=_LOCAL_SPARSE_BWD_BLOCK_SIZE,
            device=q.device,
        )
        result = backend(
            q_segment,
            k_segment,
            v_segment,
            causal=False,
            window_size=(None, None),
            softmax_scale=1.0,
            num_splits=1,
            pack_gqa=False,
            deterministic=False,
            mask_mod=mask_mod,
            aux_tensors=[vision_segment, document_segment],
            block_sparse_tensors=sparse_fwd,
            block_sparse_tensors_bwd=sparse_bwd,
            return_lse=True,
        )
        out_segment, lse_segment = _validate_local_result(result, q_segment)
        outputs.append(out_segment.squeeze(0))
        lses.append(lse_segment.squeeze(0))

    return _validate_local_varlen_result(
        (torch.cat(outputs, dim=0), torch.cat(lses, dim=1)),
        q,
    )


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
    if q.shape[0] != 1 or q.shape[1] > _GLOBAL_BACKWARD_MAX_SEQLEN:
        raise UnsupportedH100Path("EXP-0012 global backward requires B=1 and 1 <= S <= 2048")
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


def _global_backward_workspace_bytes(seqlen: int) -> int:
    """Conservative peak increment for the current two-slab split backward."""

    seqlen_q_rounded = (seqlen + 63) // 64 * 64
    seqlen_k_rounded = (seqlen + 31) // 32 * 32
    return 147456 * seqlen + 66048 * seqlen_q_rounded + 16384 * seqlen_k_rounded


def _global_backward_additional_bytes(q: torch.Tensor) -> int:
    output_bytes = q.numel() * q.element_size()
    lse_bytes = q.numel() // q.shape[-1] * torch.float32.itemsize
    return _global_backward_workspace_bytes(q.shape[1]) + 2 * (output_bytes + lse_bytes)


def _global_backward_budget(free_bytes: int) -> int:
    return min(
        free_bytes * 4 // 5,
        max(0, free_bytes - _GLOBAL_BACKWARD_RESERVE_BYTES),
    )


def _check_global_backward_budget(required: int, free: int) -> None:
    budget = _global_backward_budget(free)
    if required > budget:
        raise UnsupportedH100Path(
            f"global backward needs about {required} additional bytes, above the guarded "
            f"budget of {budget} bytes ({free} bytes currently free)"
        )


def _preflight_global_backward(q: torch.Tensor) -> None:
    if _is_fake_tensor(q) or q.device.type != "cuda":
        return
    required = _global_backward_additional_bytes(q)
    free, _total = torch.cuda.mem_get_info(q.device)
    _check_global_backward_budget(required, free)


def fa4_local_text_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionLayerSpec = SLIDING_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length Gemma local text forward on the pinned SM90 FA4 path.

    Inputs and output use 16-byte-aligned, non-overlapping ``(B, S, H, D)``
    layouts with unit D stride at B=1 and ``1 <= S <= 1025``. LSE is FP32
    with shape ``(B, Hq, S)``.
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
    callable owns the complete document/window/vision predicate. Native text
    admits the locked model maximum. Metadata calls through the EXP-0008
    boundary use packed varlen directly; longer metadata calls compose exact
    per-sequence fixed block-sparse launches.
    """

    q_lengths, k_lengths = _validate_local_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        spec,
    )
    has_custom_metadata = vision_block_ids is not None or document_ids is not None
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
    if not has_custom_metadata:
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
        if max_seqlen_k > _LOCAL_CUSTOM_DENSE_MAX_SEQLEN:
            if q_lengths is None or k_lengths is None:
                raise UnsupportedH100Path(
                    "long metadata sparse scheduling requires real cumulative values"
                )
            return _fa4_local_varlen_sparse_metadata(
                q,
                k,
                v,
                q_lengths,
                k_lengths,
                vision_ids,
                documents,
                spec,
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


def _validate_global_forward_only_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionLayerSpec,
) -> None:
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("global forward-only q, k, and v must be rank-4 B/S/H/D tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("global forward-only q, k, and v must be on one device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("global forward-only q, k, and v must use BF16")
    for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True):
        _validate_fa4_layout(tensor, name=name)
    fake_inputs = [_is_fake_tensor(tensor) for tensor in (q, k, v)]
    if any(fake_inputs) and not all(fake_inputs):
        raise ValueError("global forward-only q, k, and v must all be real or all fake")
    if not all(fake_inputs):
        if any(tensor.data_ptr() % 16 for tensor in (q, k, v)):
            raise ValueError("global forward-only q, k, and v must be 16-byte aligned")
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise ValueError("global forward-only K and V must be distinct operands")

    batch, q_length, q_heads, q_dim = q.shape
    kv_length = k.shape[1]
    if batch <= 0 or not (1 <= q_length <= kv_length <= _LOCAL_MODEL_MAX_SEQLEN):
        raise UnsupportedH100Path("global forward-only requires B>=1 and 1 <= Sq <= Sk <= 262144")
    if (q_heads, q_dim) != (spec.num_q_heads, spec.head_dim_qk):
        raise ValueError("global forward-only q shape conflicts with the locked geometry")
    if k.shape != (batch, kv_length, spec.num_kv_heads, spec.head_dim_qk):
        raise ValueError("global forward-only k shape conflicts with the locked geometry")
    if v.shape != (batch, kv_length, spec.num_kv_heads, spec.head_dim_v):
        raise ValueError("global forward-only v shape conflicts with the locked geometry")
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
        raise UnsupportedH100Path("only the locked global d512 contract is enabled")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise UnsupportedH100Path(
            "global forward-only FA4 cannot participate in autograd; use composed S<=2048"
        )
    _require_sm90(q.device)


def _preflight_global_forward_outputs(q: torch.Tensor) -> None:
    if _is_fake_tensor(q) or q.device.type != "cuda":
        return
    output_bytes = q.numel() * q.element_size()
    lse_bytes = q.numel() // q.shape[-1] * torch.float32.itemsize
    required = 2 * (output_bytes + lse_bytes)
    free, _total = torch.cuda.mem_get_info(q.device)
    if required > free * 8 // 10:
        raise UnsupportedH100Path(
            f"global forward-only outputs require about {required} bytes, above 80% "
            f"of the {free} currently free HBM bytes"
        )


def _validate_global_forward_only_result(
    result,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("pinned global forward-only FA4 must return (out, lse)")
    output, lse = result
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        raise RuntimeError("global forward-only FA4 returned an invalid output contract")
    expected_lse = (q.shape[0], q.shape[2], q.shape[1])
    if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
        raise RuntimeError("global forward-only FA4 returned an invalid FP32 LSE contract")
    return output, lse


def fa4_global_forward_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionLayerSpec = GLOBAL_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact fixed/rectangular global d512 forward without autograd.

    Causal alignment is lower-right when ``Sq < Sk``. The V512 result is the
    exact concatenation of two pinned d512-QK/d256-V FA4 launches.
    """

    _validate_global_forward_only_bshd(q, k, v, spec)
    _preflight_global_forward_outputs(q)
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
        output_slab, lse = result
        expected_output = (*q.shape[:-1], 256)
        expected_lse = (q.shape[0], q.shape[2], q.shape[1])
        if output_slab.shape != expected_output or output_slab.dtype != torch.bfloat16:
            raise RuntimeError("global forward-only FA4 returned an invalid output slab")
        if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
            raise RuntimeError("global forward-only FA4 returned an invalid FP32 LSE")
        outputs.append(output_slab)
        lses.append(lse)
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1" and not torch.equal(lses[0], lses[1]):
        raise RuntimeError("global forward-only V slabs returned different LSE values")
    return _validate_global_forward_only_result((torch.cat(outputs, dim=-1), lses[0]), q)


def _validate_global_varlen_forward_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    spec: AttentionLayerSpec,
) -> None:
    if any(tensor.ndim != 3 for tensor in (q, k, v)):
        raise ValueError("global packed forward-only q, k, and v must be rank-3 T/H/D")
    if not (q.device == k.device == v.device):
        raise ValueError("global packed forward-only q, k, and v must share one device")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise ValueError("global packed forward-only q, k, and v must use BF16")
    for name, tensor in zip(("packed q", "packed k", "packed v"), (q, k, v), strict=True):
        _validate_fa4_layout(tensor, name=name)
    fake_inputs = [_is_fake_tensor(tensor) for tensor in (q, k, v)]
    if any(fake_inputs) and not all(fake_inputs):
        raise ValueError("global packed forward-only operands must all be real or all fake")
    if not all(fake_inputs):
        if any(tensor.data_ptr() % 16 for tensor in (q, k, v)):
            raise ValueError("global packed forward-only operands must be 16-byte aligned")
        if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
            raise ValueError("global packed forward-only K and V must be distinct")
    if q.shape[0] <= 0 or k.shape[0] <= 0:
        raise ValueError("global packed forward-only totals must be positive")
    if q.shape[1:] != (spec.num_q_heads, spec.head_dim_qk):
        raise ValueError("global packed forward-only q shape conflicts with the lock")
    if k.shape[1:] != (spec.num_kv_heads, spec.head_dim_qk):
        raise ValueError("global packed forward-only k shape conflicts with the lock")
    if v.shape != (k.shape[0], spec.num_kv_heads, spec.head_dim_v):
        raise ValueError("global packed forward-only v shape conflicts with the lock")
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
        raise UnsupportedH100Path("only the locked global d512 contract is enabled")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (max_seqlen_q, max_seqlen_k)
    ):
        raise TypeError("global packed maxima must be Python integers")
    if not (1 <= max_seqlen_q <= max_seqlen_k <= _LOCAL_MODEL_MAX_SEQLEN):
        raise UnsupportedH100Path(
            "global packed forward-only maxima must satisfy 1 <= Sq <= Sk <= 262144"
        )
    cumulative = (cu_seqlens_q, cu_seqlens_k)
    if any(tensor.ndim != 1 or tensor.numel() < 2 for tensor in cumulative):
        raise ValueError("global packed cumulative arrays must be rank-1 B+1 tensors")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("global packed cumulative arrays must have the same batch count")
    if any(tensor.device != q.device for tensor in cumulative):
        raise ValueError("global packed cumulative arrays must share the operand device")
    if any(tensor.dtype != torch.int32 or not tensor.is_contiguous() for tensor in cumulative):
        raise ValueError("global packed cumulative arrays must be contiguous INT32")
    fake_cumulative = [_is_fake_tensor(tensor) for tensor in cumulative]
    if fake_cumulative != fake_inputs[:2] or fake_cumulative[0] != fake_cumulative[1]:
        raise ValueError("global packed operands and cumulative arrays must share real/fake mode")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise UnsupportedH100Path(
            "global packed forward-only FA4 cannot participate in autograd; use composed S<=2048"
        )
    if not all(fake_inputs):
        q_values = [int(value) for value in cu_seqlens_q.detach().cpu().tolist()]
        k_values = [int(value) for value in cu_seqlens_k.detach().cpu().tolist()]
        if q_values[0] != 0 or k_values[0] != 0:
            raise ValueError("global packed cumulative arrays must start at zero")
        if q_values[-1] != q.shape[0] or k_values[-1] != k.shape[0]:
            raise ValueError("global packed cumulative arrays must end at packed totals")
        q_lengths = [end - start for start, end in zip(q_values[:-1], q_values[1:], strict=True)]
        k_lengths = [end - start for start, end in zip(k_values[:-1], k_values[1:], strict=True)]
        if any(
            q_length <= 0 or q_length > k_length
            for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
        ):
            raise UnsupportedH100Path(
                "global packed forward-only requires 1 <= Sq <= Sk per segment"
            )
        if max(q_lengths) != max_seqlen_q or max(k_lengths) != max_seqlen_k:
            raise ValueError("global packed maxima must equal the cumulative segment maxima")
    _require_sm90(q.device)


def fa4_global_varlen_forward_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    spec: AttentionLayerSpec = GLOBAL_ATTENTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed lower-right global d512 forward without autograd."""

    _validate_global_varlen_forward_only(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        spec,
    )
    _preflight_global_forward_outputs(q)
    common_kwargs = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "causal": True,
        "window_size": (None, None),
        "softmax_scale": 1.0,
        "num_splits": 1,
        "pack_gqa": False,
        "deterministic": False,
        "return_lse": True,
    }
    backend = _load_flash_attn_varlen_func()
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for v_slab in v.split(256, dim=-1):
        result = backend(q, k, v_slab.contiguous(), **common_kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("pinned global packed forward-only FA4 must return (out, lse)")
        output_slab, lse = result
        if output_slab.shape != (*q.shape[:-1], 256) or output_slab.dtype != torch.bfloat16:
            raise RuntimeError("global packed forward-only FA4 returned an invalid output slab")
        if lse is None or lse.shape != (q.shape[1], q.shape[0]) or lse.dtype != torch.float32:
            raise RuntimeError("global packed forward-only FA4 returned an invalid FP32 LSE")
        outputs.append(output_slab)
        lses.append(lse)
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1" and not torch.equal(lses[0], lses[1]):
        raise RuntimeError("global packed forward-only V slabs returned different LSE values")
    output = torch.cat(outputs, dim=-1)
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        raise RuntimeError("global packed forward-only FA4 returned an invalid output contract")
    return output, lses[0]


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
    ``1 <= S <= 2048`` EXP-0012 envelope subject to HBM preflight. The H100 patch stack enables the exact
    asymmetric d512-QK/d256-V SM90 dimension/tile specialization used here;
    this adapter supplies its global-causal semantic guard.
    """

    _validate_global_bshd(q, k, v, spec)
    requires_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v)
    )
    if requires_backward:
        _preflight_global_backward(q)
    else:
        _preflight_global_forward_outputs(q)
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
