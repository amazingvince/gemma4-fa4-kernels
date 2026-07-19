"""Revision-pinned Gemma 4 31B attention contract.

The attention-kernel boundary receives prepared Q, K, and V tensors.  The
model's global path reuses the K projection output as the *source* for V, but
K and V are no longer equal after K normalization + RoPE versus V
normalization.  Keep that distinction explicit in every kernel interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LayerKind = Literal["sliding_attention", "full_attention"]


@dataclass(frozen=True)
class AttentionLayerSpec:
    kind: LayerKind
    head_dim_qk: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    softmax_scale: float
    is_causal: bool
    vision_bidirectional_within_block: bool
    sliding_window: int | None
    rope_type: str
    rope_theta: float
    partial_rotary_factor: float
    projection_source_shared_between_k_and_v: bool

    @property
    def qhead_per_kvhead(self) -> int:
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        return self.num_q_heads // self.num_kv_heads

    @property
    def rotary_dim(self) -> int:
        value = self.head_dim_qk * self.partial_rotary_factor
        if int(value) != value:
            raise ValueError("rotary dimension must be integral")
        return int(value)

    @property
    def fa_window_size_left(self) -> int | None:
        """FlashAttention's inclusive left-window argument for this layer."""
        return None if self.sliding_window is None else self.sliding_window - 1


@dataclass(frozen=True)
class Gemma4ModelSpec:
    model_id: str
    revision: str
    hidden_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_kv_shared_layers: int
    use_bidirectional_attention: str
    sliding: AttentionLayerSpec
    full: AttentionLayerSpec

    def layer_types(self) -> tuple[LayerKind, ...]:
        return tuple(
            "full_attention" if (idx + 1) % 6 == 0 else "sliding_attention"
            for idx in range(self.num_hidden_layers)
        )

    def spec_for_layer(self, layer_idx: int) -> AttentionLayerSpec:
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(layer_idx)
        return self.full if self.layer_types()[layer_idx] == "full_attention" else self.sliding

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers


SLIDING_ATTENTION = AttentionLayerSpec(
    kind="sliding_attention",
    head_dim_qk=256,
    head_dim_v=256,
    num_q_heads=32,
    num_kv_heads=16,
    softmax_scale=1.0,
    is_causal=True,
    vision_bidirectional_within_block=True,
    sliding_window=1024,
    rope_type="default",
    rope_theta=10_000.0,
    partial_rotary_factor=1.0,
    projection_source_shared_between_k_and_v=False,
)

GLOBAL_ATTENTION = AttentionLayerSpec(
    kind="full_attention",
    head_dim_qk=512,
    head_dim_v=512,
    num_q_heads=32,
    num_kv_heads=4,
    softmax_scale=1.0,
    is_causal=True,
    vision_bidirectional_within_block=False,
    sliding_window=None,
    rope_type="proportional",
    rope_theta=1_000_000.0,
    partial_rotary_factor=0.25,
    projection_source_shared_between_k_and_v=True,
)

GEMMA4_31B = Gemma4ModelSpec(
    model_id="google/gemma-4-31B",
    revision="2d418d1b7ed8c04d732c3359e19a11fbc85b6842",
    hidden_size=5376,
    num_hidden_layers=60,
    max_position_embeddings=262_144,
    num_kv_shared_layers=0,
    use_bidirectional_attention="vision",
    sliding=SLIDING_ATTENTION,
    full=GLOBAL_ATTENTION,
)
