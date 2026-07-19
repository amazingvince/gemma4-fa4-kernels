"""Exact index-level masks used by the Gemma 4 text attention contract."""

from __future__ import annotations

import torch


def gemma4_attention_mask(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    device: torch.device | str,
    sliding_window: int | None,
    vision_block_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    q_start: int | None = None,
    allow_vision_bidirectional: bool = False,
) -> torch.Tensor:
    """Return a bool mask shaped ``(B, 1, Q, K)`` where True means allowed.

    Current Transformers Gemma 4 composes the masks as follows:

    * full layers: causal only (plus padding/document constraints);
    * sliding layers: ``sliding_window AND (causal OR same vision block)``.

    ``sliding_window=1024`` uses the strict lower bound ``k > q - 1024``;
    in FlashAttention terms that is ``window_size_left=1023``.

    Query positions default to the final ``q_len`` positions of the KV stream,
    matching chunked-prefill/lower-right causal alignment.
    """
    if batch_size < 1 or q_len < 0 or kv_len < 0:
        raise ValueError("invalid attention shape")
    if q_len > kv_len and q_start is None:
        raise ValueError("q_len cannot exceed kv_len when q_start is omitted")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")

    q_start = kv_len - q_len if q_start is None else q_start
    if q_start < 0 or q_start + q_len > kv_len:
        raise ValueError("query positions must lie inside the KV stream")
    q_idx = torch.arange(q_start, q_start + q_len, device=device, dtype=torch.long)
    kv_idx = torch.arange(kv_len, device=device, dtype=torch.long)
    causal = kv_idx[None, :] <= q_idx[:, None]
    allowed = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()

    if vision_block_ids is not None:
        if vision_block_ids.shape != (batch_size, kv_len):
            raise ValueError(
                f"vision_block_ids must be {(batch_size, kv_len)}, got {tuple(vision_block_ids.shape)}"
            )
        if allow_vision_bidirectional:
            q_blocks = vision_block_ids[:, q_idx]
            same_nonnegative_block = (q_blocks[:, :, None] == vision_block_ids[:, None, :]) & (
                q_blocks[:, :, None] >= 0
            )
            allowed |= same_nonnegative_block

    if sliding_window is not None:
        lower_band = kv_idx[None, :] > q_idx[:, None] - sliding_window
        allowed &= lower_band.unsqueeze(0)

    if document_ids is not None:
        if document_ids.shape != (batch_size, kv_len):
            raise ValueError(
                f"document_ids must be {(batch_size, kv_len)}, got {tuple(document_ids.shape)}"
            )
        q_docs = document_ids[:, q_idx]
        allowed &= q_docs[:, :, None] == document_ids[:, None, :]

    if key_padding_mask is not None:
        if key_padding_mask.shape != (batch_size, kv_len):
            raise ValueError(
                f"key_padding_mask must be {(batch_size, kv_len)}, got {tuple(key_padding_mask.shape)}"
            )
        allowed &= key_padding_mask[:, None, :].to(torch.bool)

    return allowed[:, None, :, :]


def additive_mask(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an allowed-position bool mask to a 0/-inf additive mask."""
    out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return out.masked_fill(~mask, float("-inf"))
