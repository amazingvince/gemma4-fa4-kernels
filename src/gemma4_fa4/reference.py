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
