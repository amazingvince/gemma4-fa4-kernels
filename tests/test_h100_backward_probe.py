from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_local_backward",
    ROOT / "scripts/probe_h100_local_backward.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _triplet(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.tensor([value], dtype=torch.float32) for _ in range(3))


def test_quantization_atol_matches_upstream_roundtrip_rule():
    reference = torch.tensor([1.0, 16.0], dtype=torch.bfloat16)
    expected = 2.0 * (((reference + 0.3 - 0.3) - reference).abs().max().item())

    assert PROBE._quantization_atol(reference) == expected


def test_upstream_relative_policy_uses_independent_baseline_limit():
    failures = PROBE._check_gradients(
        _triplet(2.0),
        _triplet(0.0),
        policy="upstream-relative",
        bf16_refs=_triplet(1.0),
        run_label="test",
    )

    assert failures == []


def test_upstream_relative_policy_rejects_error_above_limit():
    failures = PROBE._check_gradients(
        _triplet(2.25),
        _triplet(0.0),
        policy="upstream-relative",
        bf16_refs=_triplet(1.0),
        run_label="test",
    )

    assert len(failures) == 3
    assert all("upstream-relative limit" in failure for failure in failures)


def test_bf16_baseline_uses_independent_softmax_path(monkeypatch):
    calls = 0
    original_softmax = torch.softmax

    def tracked_softmax(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_softmax(*args, **kwargs)

    monkeypatch.setattr(PROBE.torch, "softmax", tracked_softmax)
    q = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn_like(q)

    grads = PROBE._run_upstream_style_bf16_baseline(q, k, v, do)

    assert calls == 1
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


def test_vision_pattern_builder_preserves_zero_and_adjacent_ids():
    assert PROBE._make_vision_block_ids(8, "none", device="cpu") is None
    text = PROBE._make_vision_block_ids(8, "text", device="cpu")
    assert torch.all(text == -1)
    all_vision = PROBE._make_vision_block_ids(8, "all", device="cpu")
    assert all_vision.dtype == torch.int64
    assert torch.count_nonzero(all_vision) == 0

    adjacent = PROBE._make_vision_block_ids(129, "adjacent", device="cpu")
    assert adjacent[0, 64].item() == 0
    assert adjacent[0, 65].item() == 1
    assert adjacent[0, 59].item() == -1


def test_bf16_baseline_applies_future_vision_exception():
    q = torch.zeros(1, 4, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    do = torch.zeros_like(q)
    do[:, 1, 0] = 1
    vision_ids = torch.tensor([[-1, 0, 0, -1]], dtype=torch.int64)

    _, _, dv = PROBE._run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        vision_block_ids=vision_ids,
    )

    assert torch.count_nonzero(dv[:, 2]) > 0
    assert torch.count_nonzero(dv[:, 3]) == 0


@pytest.mark.parametrize("gradient_source", ["lse", "out_lse"])
def test_bf16_baseline_supports_differentiable_lse(gradient_source):
    q = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 3, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 3, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn_like(q)
    dlse = torch.randn(1, 2, 3, dtype=torch.float32)

    grads = PROBE._run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        dlse=dlse,
        gradient_source=gradient_source,
    )

    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_structured_ownership_checker_enforces_query_and_gqa_mapping():
    dq = torch.zeros(1, 4, 4, 1)
    dk = torch.zeros(1, 4, 2, 1)
    dv = torch.zeros(1, 4, 2, 1)
    dq[:, 1, 3] = 1
    dk[:, 2, 1] = 1
    dv[:, 2, 1] = 1

    PROBE._assert_structured_ownership(
        (dq, dk, dv),
        q_index=1,
        q_head=3,
        same_block_future=2,
        different_block_future=3,
        gqa_ratio=2,
    )

    dk[:, 3, 1] = 1
    with pytest.raises(AssertionError, match="different-block future key"):
        PROBE._assert_structured_ownership(
            (dq, dk, dv),
            q_index=1,
            q_head=3,
            same_block_future=2,
            different_block_future=3,
            gqa_ratio=2,
        )
