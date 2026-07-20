"""Small-shape, high-precision reference attention for Gemma 4 kernel tests."""

from __future__ import annotations

from typing import Literal

import torch

from .masks import gemma4_attention_mask
from .model_spec import AttentionLayerSpec

TimingMode = Literal["fwd", "bwd", "fwd_bwd"]


def expand_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, Hkv, S, D)`` to query heads without materializing first."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    allow_vision_bidirectional: bool = False,
    q_start: int | None = None,
    upcast: torch.dtype = torch.float64,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute prepared-Q/K/V attention with Gemma 4 mask semantics.

    Inputs use ``(B, H, S, D)``. K and V are always distinct operands at this
    boundary, even when they originated from one projection source upstream.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 B/H/S/D tensors")
    b, hq, q_len, d = q.shape
    bk, hkv, kv_len, dk = k.shape
    bv, hv, vv_len, dv = v.shape
    if (bk, bv) != (b, b) or hkv != hv or kv_len != vv_len or d != dk:
        raise ValueError("incompatible q/k/v shapes")
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")

    out_dtype = q.dtype
    qf, kf, vf = (x.to(upcast) for x in (q, k, v))
    kf = expand_kv(kf, hq // hkv)
    vf = expand_kv(vf, hq // hkv)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * float(softmax_scale)
    mask = gemma4_attention_mask(
        batch_size=b,
        q_len=q_len,
        kv_len=kv_len,
        device=q.device,
        sliding_window=sliding_window,
        vision_block_ids=vision_block_ids,
        document_ids=document_ids,
        key_padding_mask=key_padding_mask,
        q_start=q_start,
        allow_vision_bidirectional=allow_vision_bidirectional,
    )
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None])
    probs = torch.where(torch.isfinite(lse)[..., None], probs, torch.zeros_like(probs))
    out = torch.einsum("bhqk,bhkd->bhqd", probs, vf).to(out_dtype)
    return (out, lse.to(torch.float32)) if return_lse else out


def _packed_cumulative_values(
    cumulative: torch.Tensor,
    *,
    total: int,
    name: str,
) -> list[int]:
    if cumulative.ndim != 1 or cumulative.numel() < 2:
        raise ValueError(f"{name} must be rank-1 with at least two entries")
    if cumulative.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use an integer dtype")
    values = [int(value) for value in cumulative.detach().cpu().tolist()]
    if values[0] != 0 or values[-1] != total:
        raise ValueError(f"{name} must start at zero and end at the packed total")
    if any(end < start for start, end in zip(values[:-1], values[1:], strict=True)):
        raise ValueError(f"{name} must be nondecreasing")
    return values


def reference_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    allow_vision_bidirectional: bool = False,
    upcast: torch.dtype = torch.float64,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference packed self-attention with lower-right Q/K alignment.

    Q uses ``(Tq,Hq,D)`` and K/V use ``(Tk,Hkv,D)``. Each query segment
    corresponds to the final ``Sq`` positions of its key segment. Vision and
    document metadata therefore follow the packed K stream; each query reads
    the metadata at its lower-right absolute K position.
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed q, k, and v must be rank-3 T/H/D tensors")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible packed q/k/v shapes")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if not (q.device == k.device == v.device):
        raise ValueError("packed q, k, and v must share one device")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("packed cumulative arrays must share the q/k/v device")
    if q.shape[0] <= 0 or k.shape[0] <= 0:
        raise ValueError("packed Q and K totals must be positive")

    q_cumulative = _packed_cumulative_values(
        cu_seqlens_q,
        total=q.shape[0],
        name="cu_seqlens_q",
    )
    k_cumulative = _packed_cumulative_values(
        cu_seqlens_k,
        total=k.shape[0],
        name="cu_seqlens_k",
    )
    if len(q_cumulative) != len(k_cumulative):
        raise ValueError("packed Q and K must have the same batch count")

    for name, metadata in (
        ("vision_block_ids", vision_block_ids),
        ("document_ids", document_ids),
    ):
        if metadata is None:
            continue
        if metadata.shape != (k.shape[0],):
            raise ValueError(f"packed {name} must match Tk")
        if metadata.device != q.device:
            raise ValueError(f"packed {name} must share the q/k/v device")
        if metadata.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"packed {name} must use an integer dtype")

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx, (q_start, q_end, k_start, k_end) in enumerate(
        zip(
            q_cumulative[:-1],
            q_cumulative[1:],
            k_cumulative[:-1],
            k_cumulative[1:],
            strict=True,
        )
    ):
        q_len, k_len = q_end - q_start, k_end - k_start
        if q_len > k_len:
            raise ValueError(f"packed sequence {batch_idx} has Sq greater than Sk")
        if q_len == 0:
            continue
        vision_ids = (
            vision_block_ids[k_start:k_end].unsqueeze(0) if vision_block_ids is not None else None
        )
        document_ids_i = (
            document_ids[k_start:k_end].unsqueeze(0) if document_ids is not None else None
        )
        out, lse = reference_attention(
            q[q_start:q_end].transpose(0, 1).unsqueeze(0),
            k[k_start:k_end].transpose(0, 1).unsqueeze(0),
            v[k_start:k_end].transpose(0, 1).unsqueeze(0),
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            vision_block_ids=vision_ids,
            document_ids=document_ids_i,
            allow_vision_bidirectional=allow_vision_bidirectional,
            q_start=k_len - q_len,
            upcast=upcast,
            return_lse=True,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
        lses.append(lse.squeeze(0))

    packed_out = torch.cat(outputs, dim=0)
    packed_lse = torch.cat(lses, dim=1)
    return (packed_out, packed_lse) if return_lse else packed_out


def reference_layer(
    spec: AttentionLayerSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if q.shape[1] != spec.num_q_heads or k.shape[1] != spec.num_kv_heads:
        raise ValueError("head count does not match layer spec")
    if q.shape[-1] != spec.head_dim_qk or k.shape[-1] != spec.head_dim_qk:
        raise ValueError("Q/K dimension does not match layer spec")
    if v.shape[-1] != spec.head_dim_v:
        raise ValueError("V dimension does not match layer spec")
    return reference_attention(
        q,
        k,
        v,
        softmax_scale=spec.softmax_scale,
        sliding_window=spec.sliding_window,
        vision_block_ids=vision_block_ids,
        document_ids=document_ids,
        key_padding_mask=key_padding_mask,
        allow_vision_bidirectional=spec.vision_bidirectional_within_block,
        return_lse=return_lse,
    )


def make_qkv(
    spec: AttentionLayerSpec,
    *,
    batch: int,
    seqlen: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 0,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch,
        spec.num_q_heads,
        seqlen,
        spec.head_dim_qk,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        batch,
        spec.num_kv_heads,
        seqlen,
        spec.head_dim_qk,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        batch,
        spec.num_kv_heads,
        seqlen,
        spec.head_dim_v,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=requires_grad,
    )
    return q, k, v


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    mean_squared = x.float().pow(2).mean(dim=-1, keepdim=True) + eps
    y = x * torch.pow(mean_squared, -0.5)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a leading subspace and leave the remainder untouched."""
    if rotary_dim == 0:
        return x
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    while cos.ndim < x_rot.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat((rotated, x_pass), dim=-1)


def prepare_global_kv_from_projection_source(
    source: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 0.25,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model the global shared-projection-source preparation contract.

    This helper exists to prevent the incorrect ``K is V`` FMHA assumption:
    K receives learned RMSNorm and partial RoPE, while V receives unscaled
    RMSNorm and no RoPE.  The returned tensors are therefore distinct.
    """
    rotary_width = source.shape[-1] * partial_rotary_factor
    if int(rotary_width) != rotary_width or int(rotary_width) % 2:
        raise ValueError("partial rotary width must be an even integer")
    rotary_dim = int(rotary_width)
    k = rms_norm(source, k_norm_weight, eps=eps)
    k = apply_partial_rotary(k, cos, sin, rotary_dim)
    v = rms_norm(source, None, eps=eps)
    return k, v


def valid_pair_count(spec: AttentionLayerSpec, seqlen: int) -> int:
    if spec.sliding_window is None:
        return seqlen * (seqlen + 1) // 2
    w = min(spec.sliding_window, seqlen)
    return seqlen * w - w * (w - 1) // 2


def attention_flops(
    spec: AttentionLayerSpec,
    *,
    batch: int,
    seqlen: int,
    mode: TimingMode = "fwd",
) -> float:
    """Matmul FLOPs using valid text-token pairs (no multimodal expansion)."""
    pairs = valid_pair_count(spec, seqlen)
    fwd = 2.0 * batch * spec.num_q_heads * pairs * (spec.head_dim_qk + spec.head_dim_v)
    # Backward includes score recomputation, dV, dP, dQ, and dK.
    bwd = 2.0 * batch * spec.num_q_heads * pairs * (3 * spec.head_dim_qk + 2 * spec.head_dim_v)
    if mode == "fwd":
        return fwd
    if mode == "bwd":
        return bwd
    if mode == "fwd_bwd":
        return fwd + bwd
    raise ValueError(mode)


def qkv_bytes(spec: AttentionLayerSpec, *, batch: int, seqlen: int, dtype_bytes: int = 2) -> int:
    """Materialized FMHA-boundary Q/K/V bytes; K and V are separate tensors."""
    return (
        batch
        * seqlen
        * dtype_bytes
        * (
            spec.num_q_heads * spec.head_dim_qk
            + spec.num_kv_heads * spec.head_dim_qk
            + spec.num_kv_heads * spec.head_dim_v
        )
    )
