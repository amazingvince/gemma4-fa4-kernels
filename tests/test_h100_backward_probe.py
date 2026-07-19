from __future__ import annotations

import importlib.util
from pathlib import Path

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
