"""Optional parity checks against the pinned Transformers implementation."""

import json
from pathlib import Path

import pytest
import torch

from gemma4_fa4.masks import gemma4_attention_mask

pytest.importorskip("transformers")
masking_utils = pytest.importorskip("transformers.masking_utils")
configuration_gemma4 = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

blockwise_overlay = masking_utils.blockwise_overlay
sliding_window_overlay = masking_utils.sliding_window_overlay
Gemma4TextConfig = configuration_gemma4.Gemma4TextConfig
Gemma4TextAttention = modeling_gemma4.Gemma4TextAttention
create_masks_for_vision_model = modeling_gemma4.create_masks_for_vision_model

ROOT = Path(__file__).resolve().parents[1]


def _config():
    lock = json.loads((ROOT / "configs/model/gemma4-31b.lock.json").read_text())
    text = dict(lock["text_config"])
    # Preserve exact head geometry while avoiding hundreds of MB of test-only projection weights.
    text["hidden_size"] = 64
    text["intermediate_size"] = 128
    return Gemma4TextConfig(**text)


def test_hf_attention_module_contract():
    cfg = _config()
    local = Gemma4TextAttention(cfg, layer_idx=0)
    global_attn = Gemma4TextAttention(cfg, layer_idx=5)
    assert local.scaling == global_attn.scaling == 1.0
    assert local.head_dim == 256
    assert global_attn.head_dim == 512
    assert local.num_key_value_groups == 2
    assert global_attn.num_key_value_groups == 8
    assert global_attn.v_proj is None  # one projection source
    assert local.q_norm.with_scale and local.k_norm.with_scale
    assert not local.v_norm.with_scale
    assert global_attn.q_norm.with_scale and global_attn.k_norm.with_scale
    assert not global_attn.v_norm.with_scale
    assert global_attn.k_norm is not global_attn.v_norm  # distinct preparation paths


def test_hf_mask_primitives_match_local_reference():
    blocks = torch.tensor([[-1, 7, 7, 7, -1]], dtype=torch.int32)
    ours = gemma4_attention_mask(
        batch_size=1,
        q_len=5,
        kv_len=5,
        device="cpu",
        sliding_window=5,
        vision_block_ids=blocks,
        allow_vision_bidirectional=True,
    )[0, 0]
    window_fn = sliding_window_overlay(5)
    block_fn = blockwise_overlay(blocks)
    expected = torch.empty_like(ours)
    for q in range(5):
        for k in range(5):
            b = torch.tensor(0)
            h = torch.tensor(0)
            qi = torch.tensor(q)
            ki = torch.tensor(k)
            expected[q, k] = bool(window_fn(b, h, qi, ki) and ((k <= q) or block_fn(b, h, qi, ki)))
    torch.testing.assert_close(ours, expected)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pinned Transformers maps flash_attention_4 to the generic 2D flash mask "
        "adapter, which cannot encode the blockwise future-token exception."
    ),
)
def test_hf_fa4_mask_adapter_preserves_vision_block_semantics():
    cfg = _config()
    cfg._attn_implementation = "flash_attention_4"
    inputs = torch.zeros(1, 5, 64)
    blocks = torch.tensor([[-1, 7, 7, 7, -1]], dtype=torch.int32)
    masks = create_masks_for_vision_model(
        cfg,
        inputs_embeds=inputs,
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(5).unsqueeze(0),
        block_sequence_ids=blocks,
    )
    local = masks["sliding_attention"]
    assert local is not None and local.ndim == 4
    # Query token 1 and future key token 2 share the same vision block.
    assert bool(local[0, 0, 1, 2])
