from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_global_backward",
    ROOT / "scripts/probe_h100_global_backward.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _exact_global_inputs(seqlen: int = 2):
    q = torch.zeros(1, seqlen, 32, 512, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, seqlen, 4, 512, dtype=torch.bfloat16, requires_grad=True)
    v = torch.empty(1, seqlen, 4, 512, dtype=torch.bfloat16)
    v[:, 0].fill_(1.0)
    if seqlen > 1:
        v[:, 1:].fill_(3.0)
    v.requires_grad_()
    return q, k, v, torch.ones_like(q)


def _triplet(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.tensor([value], dtype=torch.float32) for _ in range(3))


def test_independent_references_lock_causal_gqa8_scale_one_and_separate_gradients():
    q, k, v, do = _exact_global_inputs()

    out, lse, grads = PROBE._run_fp32_reference(q, k, v, do)
    out_bf16, grads_bf16 = PROBE._run_bf16_reference(q, k, v, do)

    assert out.shape == out_bf16.shape == q.shape
    assert lse.shape == (1, 32, 2)
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)
    assert tuple(grad.shape for grad in grads_bf16) == (q.shape, k.shape, v.shape)
    assert out[0, 0, 0, 0].item() == out_bf16[0, 0, 0, 0].item() == 1.0
    assert out[0, 1, 0, 0].item() == out_bf16[0, 1, 0, 0].item() == 2.0
    torch.testing.assert_close(lse[0, 0], torch.tensor([0.0, torch.log(torch.tensor(2.0))]))


def test_upstream_relative_gradient_policy_accepts_and_rejects_independently():
    assert (
        PROBE._check_gradient_policy(
            _triplet(2.0),
            _triplet(0.0),
            _triplet(1.0),
            run_label="test",
        )
        == []
    )

    failures = PROBE._check_gradient_policy(
        _triplet(2.25),
        _triplet(0.0),
        _triplet(1.0),
        run_label="test",
    )

    assert len(failures) == 3
    assert all("upstream-relative limit" in failure for failure in failures)


def test_default_matrix_covers_small_n32_and_m64_boundaries():
    assert PROBE.DEFAULT_SEQLENS == (1, 31, 32, 33, 63, 64, 65, 127, 128, 129)
