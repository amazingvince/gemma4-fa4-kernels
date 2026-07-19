from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_local_varlen_backward",
    ROOT / "scripts/probe_h100_local_varlen_backward.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _triplet(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.tensor([value], dtype=torch.float32) for _ in range(3))


def test_parse_and_validate_packed_lengths():
    assert PROBE._parse_lengths("1, 64,129") == (1, 64, 129)
    PROBE._validate_lengths((1, 64, 129), (33, 64, 1025))
    PROBE._validate_lengths((1,), (262_144,))

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        PROBE._parse_lengths("1,0")
    with pytest.raises(ValueError, match="batch count"):
        PROBE._validate_lengths((1,), (1, 2))
    with pytest.raises(ValueError, match="Sq <= Sk"):
        PROBE._validate_lengths((3,), (2,))
    with pytest.raises(ValueError, match="262144"):
        PROBE._validate_lengths((1,), (262_145,))


def test_long_probe_memory_preflight_is_conservative(monkeypatch, capsys):
    sparse_bytes = 4 * 4096 * (3 + 1) + 4 * 1 * (3 + 3277)
    expected = (
        7 * (1 * 32 * 256 * 2)
        + 8 * (262_144 * 16 * 256 * 2)
        + 5 * (32 * 1 * 4)
        + sparse_bytes
        + 2 * 1024**3
    )
    assert PROBE._estimate_probe_live_bytes((1,), (262_144,)) == expected
    retained_repeat = 2 * (1 * 32 * 256 * 2) + 2 * (262_144 * 16 * 256 * 2) + 32 * 1 * 4
    assert PROBE._estimate_probe_live_bytes((1,), (262_144,), repeats=2) == (
        expected + retained_repeat
    )
    assert PROBE._estimate_probe_live_bytes((262_144,), (262_144,), reference=True) > (80 * 1024**3)
    with pytest.raises(ValueError, match="positive"):
        PROBE._estimate_probe_live_bytes((1,), (1,), repeats=0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * 1024**3, 80 * 1024**3))
    PROBE._preflight_cuda_memory((1,), (262_144,))
    assert "memory_preflight" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="memory preflight"):
        PROBE._preflight_cuda_memory((262_144,), (262_144,), reference=True)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1024**3, 80 * 1024**3))
    with pytest.raises(RuntimeError, match="memory preflight"):
        PROBE._preflight_cuda_memory((262_144,), (262_144,))


def test_metadata_builders_repeat_ids_across_packed_boundaries():
    vision = PROBE._make_vision_block_ids((2, 3), (2, 3), "all", device="cpu")
    documents = PROBE._make_document_ids((2, 3), "single", device="cpu")

    assert vision.dtype == documents.dtype == torch.int64
    assert vision.tolist() == [0, 0, 0, 0, 0]
    assert documents.tolist() == [0, 0, 0, 0, 0]


def test_bf16_baseline_composes_lower_right_vision_and_document_masks():
    q = torch.zeros(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(5, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.arange(5, dtype=torch.bfloat16).view(5, 1, 1).expand(-1, 1, 4).clone()
    v.requires_grad_()
    do = torch.zeros_like(q)
    do[0, 0] = 1
    cu_q = torch.tensor([0, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 5], dtype=torch.int32)
    vision = torch.tensor([-1, -1, 7, 7, 8], dtype=torch.int64)
    documents = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)

    _, _, dv = PROBE._run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        vision_block_ids=vision,
        document_ids=documents,
    )

    assert torch.count_nonzero(dv[2]).item() > 0
    assert torch.count_nonzero(dv[3]).item() > 0
    assert torch.count_nonzero(dv[[0, 1, 4]]).item() == 0


@pytest.mark.parametrize("gradient_source", ["lse", "out_lse"])
def test_bf16_baseline_supports_packed_lse_gradients(gradient_source):
    q = torch.randn(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(5, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(5, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn_like(q)
    dlse = torch.randn(2, 3, dtype=torch.float32)
    cu_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 5], dtype=torch.int32)

    grads = PROBE._run_upstream_style_bf16_baseline(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        dlse=dlse,
        gradient_source=gradient_source,
    )

    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)
    assert all(torch.isfinite(grad).all() for grad in grads)
    if gradient_source == "lse":
        assert torch.count_nonzero(grads[2]).item() == 0


def test_upstream_relative_policy_uses_independent_baseline_limit():
    assert (
        PROBE._check_gradients(
            _triplet(2.0),
            _triplet(0.0),
            policy="upstream-relative",
            bf16_refs=_triplet(1.0),
            run_label="test",
        )
        == []
    )

    failures = PROBE._check_gradients(
        _triplet(2.25),
        _triplet(0.0),
        policy="upstream-relative",
        bf16_refs=_triplet(1.0),
        run_label="test",
    )
    assert len(failures) == 3
    assert all("upstream-relative limit" in failure for failure in failures)


def test_structured_checker_enforces_packed_query_gqa_and_key_ownership():
    dq = torch.zeros(5, 4, 1)
    dk = torch.zeros(7, 2, 1)
    dv = torch.zeros(7, 2, 1)
    dq[2, 3] = 1
    dk[[4, 5], 1] = 1
    dv[[4, 5], 1] = 1

    PROBE._assert_structured_ownership(
        (dq, dk, dv),
        q_index=2,
        q_head=3,
        allowed_key_indices=(4, 5),
        same_block_future=5,
        gqa_ratio=2,
    )

    dk[6, 1] = 1
    with pytest.raises(AssertionError, match="masked or cross-sequence"):
        PROBE._assert_structured_ownership(
            (dq, dk, dv),
            q_index=2,
            q_head=3,
            allowed_key_indices=(4, 5),
            same_block_future=5,
            gqa_ratio=2,
        )


def test_isolation_checker_covers_outputs_lse_and_all_gradient_slices():
    out = torch.zeros(3, 2, 1)
    lse = torch.zeros(2, 3)
    grads = (torch.zeros_like(out), torch.zeros(5, 1, 1), torch.zeros(5, 1, 1))
    base = (out, lse, grads)
    mutated = tuple(
        item.clone() if isinstance(item, torch.Tensor) else tuple(tensor.clone() for tensor in item)
        for item in base
    )

    PROBE._assert_isolated_prefix_equal(base, mutated, q_prefix=1, k_prefix=2)
    mutated[2][1][0] = 1
    with pytest.raises(AssertionError, match="dK"):
        PROBE._assert_isolated_prefix_equal(base, mutated, q_prefix=1, k_prefix=2)


def test_isolation_checker_can_defer_dq_to_independent_reference_policy():
    out = torch.zeros(1, 1, 1)
    lse = torch.zeros(1, 1)
    grads = (torch.zeros_like(out), torch.zeros(2, 1, 1), torch.zeros(2, 1, 1))
    mutated_grads = tuple(grad.clone() for grad in grads)
    mutated_grads[0][0] = 0.0625

    with pytest.raises(AssertionError, match="dQ"):
        PROBE._assert_isolated_prefix_equal(
            (out, lse, grads),
            (out.clone(), lse.clone(), mutated_grads),
            q_prefix=1,
            k_prefix=2,
        )
    PROBE._assert_isolated_prefix_equal(
        (out, lse, grads),
        (out.clone(), lse.clone(), mutated_grads),
        q_prefix=1,
        k_prefix=2,
        check_dq=False,
    )
