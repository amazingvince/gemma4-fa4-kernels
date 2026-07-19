"""Contract adapters for the pinned FA4 CuTe SM90 path.

M1 covers fixed-length text forward only. The global d512 path is an exact
two-launch composition over V/O slabs; it is a correctness path, not a
performance claim.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from .model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION, AttentionLayerSpec


class UnsupportedH100Path(RuntimeError):
    """Raised when an input would leave the validated SM90 contract."""


def _load_flash_attn_func() -> Callable:
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:  # pragma: no cover - depends on the GPU environment
        raise UnsupportedH100Path(
            "pinned flash-attn-4 is not importable; run scripts/setup_env.sh h100"
        ) from exc
    return flash_attn_func


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
    if k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
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
        raise UnsupportedH100Path("only the locked local d256 text-forward contract is enabled")


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
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("pinned flash_attn_func must return (out, lse)")
    out, lse = result
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise RuntimeError("FA4 returned an invalid local output contract")
    expected_lse = (q.shape[0], q.shape[2], q.shape[1])
    if lse is None or lse.shape != expected_lse or lse.dtype != torch.float32:
        raise RuntimeError("FA4 returned an invalid FP32 LSE contract")
    return out, lse


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
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR") != "1" and not torch.equal(lses[0], lses[1]):
        raise RuntimeError("global V-slab launches returned different LSE values")
    return torch.cat(outputs, dim=-1), lses[0]
