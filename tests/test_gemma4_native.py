from __future__ import annotations

import pytest
import torch

from gemma4_fa4.gemma4_native import (
    _deterministic_requested,
    _fa4_window_size,
    _positions_are_packed,
)
from gemma4_fa4.model_spec import GLOBAL_ATTENTION, SLIDING_ATTENTION
from gemma4_fa4.transformers_integration import (
    Gemma4MaskPlan,
    _padding_mask,
)


def test_deterministic_policy_is_opt_in(monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    assert _deterministic_requested() is False
    assert _deterministic_requested(True) is True
    assert _deterministic_requested(False) is False

    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    assert _deterministic_requested() is True
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _deterministic_requested()


def test_fa4_uses_full_causal_path_when_local_window_cannot_exclude_keys():
    assert _fa4_window_size(SLIDING_ATTENTION, kv_length=1024) == (None, None)
    assert _fa4_window_size(SLIDING_ATTENTION, kv_length=1025) == (1023, 0)
    assert _fa4_window_size(GLOBAL_ATTENTION, kv_length=262144) == (None, None)


def test_position_packedness_cache_tracks_tensor_version():
    position_ids = torch.arange(8).unsqueeze(0)
    assert not _positions_are_packed(
        position_ids,
        padding=None,
        q_length=8,
        kv_length=8,
    )
    cached_fingerprint, cached_groups = position_ids._gemma4_fa4_packed_groups
    assert cached_groups is None

    position_ids[0, 4:] -= 4
    assert _positions_are_packed(
        position_ids,
        padding=None,
        q_length=8,
        kv_length=8,
    )
    assert position_ids._gemma4_fa4_packed_groups[0] != cached_fingerprint


def test_position_packedness_respects_padding():
    position_ids = torch.tensor([[0, 1, 2, 0, 0]])
    padding = torch.tensor([[True, True, True, False, False]])
    assert not _positions_are_packed(
        position_ids,
        padding=padding,
        q_length=5,
        kv_length=5,
    )


def test_all_valid_padding_mask_is_normalized_to_none():
    attention_mask = torch.ones((1, 8), dtype=torch.int64)
    plan = Gemma4MaskPlan(1, 8, 8, 0, 0, None, attention_mask)
    assert _padding_mask(plan, torch.device("cpu")) is None
    assert attention_mask._gemma4_fa4_padding_properties[1:] == (True, True)


def test_padding_mask_rejects_nonbinary_values():
    attention_mask = torch.tensor([[1, 1, 2, 0]])
    plan = Gemma4MaskPlan(1, 4, 4, 0, 0, None, attention_mask)
    with pytest.raises(ValueError, match="must use 0/1"):
        _padding_mask(plan, torch.device("cpu"))
