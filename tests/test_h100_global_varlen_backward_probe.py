from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_global_varlen_backward",
    ROOT / "scripts/probe_h100_global_varlen_backward.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _triplet(value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.tensor([value], dtype=torch.float32) for _ in range(3))


def test_parse_resolve_and_validate_global_varlen_lengths():
    assert PROBE._parse_lengths("1, 33,65") == (1, 33, 65)
    assert PROBE._resolve_lengths(None, None, None) == (
        PROBE.DEFAULT_Q_LENGTHS,
        PROBE.DEFAULT_K_LENGTHS,
    )
    assert PROBE._resolve_lengths(None, (33, 65), None) == ((33, 65), (33, 65))
    assert PROBE._resolve_lengths("mixed", None, None) == PROBE.PRESET_LENGTHS["mixed"]
    PROBE._validate_lengths((1, 33, 65), (33, 65, 2048))

    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        PROBE._parse_lengths("1,0")
    with pytest.raises(ValueError, match="batch count"):
        PROBE._validate_lengths((1,), (1, 2))
    with pytest.raises(ValueError, match="1 <= Sq <= Sk <= 262144"):
        PROBE._validate_lengths((3,), (2,))
    PROBE._validate_lengths((1,), (2049,))
    PROBE._validate_lengths((1,), (262_144,))
    with pytest.raises(ValueError, match="1 <= Sq <= Sk <= 262144"):
        PROBE._validate_lengths((1,), (262_145,))
    with pytest.raises(ValueError, match="cannot be combined"):
        PROBE._resolve_lengths("mixed", (1,), None)
    with pytest.raises(ValueError, match="requires --q-lengths"):
        PROBE._resolve_lengths(None, None, (1,))


def test_named_cases_cover_b33_tiny_mixed_and_reversed_schedulers():
    tiny_q, tiny_k = PROBE.PRESET_LENGTHS["b33-tiny"]
    assert len(tiny_q) == len(tiny_k) == 33
    assert tiny_q == tiny_k == (1,) * 33

    mixed_q, mixed_k = PROBE.PRESET_LENGTHS["mixed"]
    reversed_q, reversed_k = PROBE.PRESET_LENGTHS["reversed"]
    assert reversed_q == tuple(reversed(mixed_q))
    assert reversed_k == tuple(reversed(mixed_k))
    PROBE._validate_lengths(mixed_q, mixed_k)
    PROBE._validate_lengths(reversed_q, reversed_k)


def test_dense_reference_envelope_rejects_quadratic_or_gqa_expansion_ooms():
    assert PROBE._dense_reference_is_safe((2049,), (2049,))
    assert PROBE._dense_reference_is_safe((33,), (4097,))
    assert not PROBE._dense_reference_is_safe((2050,), (2050,))
    assert not PROBE._dense_reference_is_safe((1,), (262_144,))


def test_preflight_only_uses_meta_shapes_and_requires_expected_rejection(monkeypatch, capsys):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * gib, 80 * gib))
    PROBE._run_preflight_only((1,), (262_144,), expect_budget_rejection=False)
    assert "admitted=True" in capsys.readouterr().out

    PROBE._run_preflight_only((262_144,), (262_144,), expect_budget_rejection=True)
    output = capsys.readouterr().out
    assert "admitted=False" in output
    assert "expected_rejection=True" in output


def test_segment_bounds_and_cumulative_tensor_are_exact():
    lengths = (1, 33, 65)
    assert PROBE._segment_bounds(lengths, 0) == (0, 1)
    assert PROBE._segment_bounds(lengths, 1) == (1, 34)
    assert PROBE._segment_bounds(lengths, 2) == (34, 99)
    assert PROBE._cumulative_tensor(lengths, device="cpu").tolist() == [0, 1, 34, 99]
    with pytest.raises(IndexError, match="out of range"):
        PROBE._segment_bounds(lengths, 3)


def test_upstream_relative_gradient_policy_accepts_and_rejects_each_gradient():
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


def test_fp32_reference_locks_lower_right_causal_alignment_and_scale_one():
    q = torch.zeros(1, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(3, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    v = torch.arange(1, 4, dtype=torch.bfloat16).view(3, 1, 1).expand(-1, 1, 4).clone()
    v.requires_grad_()
    do = torch.ones_like(q)
    dlse = torch.zeros(2, 1, dtype=torch.float32)
    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_k = torch.tensor([0, 3], dtype=torch.int32)

    out, lse, grads = PROBE._run_fp32_reference(
        q,
        k,
        v,
        do,
        dlse,
        cu_q,
        cu_k,
        "out",
    )

    assert torch.equal(out, torch.full_like(out, 2.0))
    torch.testing.assert_close(lse, torch.full_like(lse, torch.log(torch.tensor(3.0))))
    assert tuple(grad.shape for grad in grads) == (q.shape, k.shape, v.shape)


def test_contract_enforces_exact_gemma_geometry_and_lse_only_zero_dv():
    q = torch.zeros(1, 32, 512, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
    v = torch.zeros(1, 4, 512, dtype=torch.bfloat16)
    out = torch.zeros_like(q)
    lse = torch.zeros(32, 1, dtype=torch.float32)
    grads = (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))

    PROBE._check_contract(q, k, v, out, lse, grads, gradient_source="lse")

    with pytest.raises(AssertionError, match="Q geometry"):
        PROBE._check_contract(q[:, :31], k, v, out[:, :31], lse[:31], grads, gradient_source="out")

    bad_grads = tuple(grad.clone() for grad in grads)
    bad_grads[2][0, 0, 0] = 1
    with pytest.raises(AssertionError, match="exact-zero dV"):
        PROBE._check_contract(q, k, v, out, lse, bad_grads, gradient_source="lse")


def test_zero_score_analytic_checks_lower_right_counts_and_separate_gradients():
    q_lengths, k_lengths = (1, 2), (3, 4)
    out = torch.full((3, 32, 512), PROBE.ANALYTIC_VALUE, dtype=torch.bfloat16)
    counts = torch.tensor([3.0, 3.0, 4.0]).log()
    lse = counts.unsqueeze(0).expand(32, -1).clone()
    dq = torch.ones_like(out)
    dk = torch.zeros(7, 4, 512, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    dv[:3].fill_(8.0 / 3.0)
    dv[3:].fill_(8.0 / 4.0)

    PROBE._check_zero_score_analytic(
        q_lengths,
        k_lengths,
        (out, lse, (dq, dk, dv)),
        gradient_source="out_lse",
        pattern="final",
        score_case="zero",
        run_label="unit",
    )

    bad_dq = dq.clone()
    bad_dq[0, 0, 0] = 0
    with pytest.raises(AssertionError, match="dQ staged-profile"):
        PROBE._check_zero_score_analytic(
            q_lengths,
            k_lengths,
            (out, lse, (bad_dq, dk, dv)),
            gradient_source="out_lse",
            pattern="final",
            score_case="zero",
            run_label="unit",
        )


def test_zero_score_causal_boundary_pattern_checks_long_q_regions():
    seqlen = 65
    out = torch.full((seqlen, 32, 512), PROBE.ANALYTIC_VALUE, dtype=torch.bfloat16)
    lse = torch.arange(1, seqlen + 1, dtype=torch.float32).log()
    lse = lse.unsqueeze(0).expand(32, -1).clone()
    dq_rows = PROBE._staged_uniform_dq_profile(
        (seqlen,),
        (seqlen,),
        score_case="zero",
        lse=lse,
    )
    dq = dq_rows.unsqueeze(-1).expand_as(out).clone()
    dk = torch.zeros(seqlen, 4, 512, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    dv[:1].fill_(32.0)
    dv[1:32].fill_(24.0)
    dv[32:64].fill_(16.0)
    dv[64:].fill_(8.0)

    PROBE._check_zero_score_analytic(
        (seqlen,),
        (seqlen,),
        (out, lse, (dq, dk, dv)),
        gradient_source="out_lse",
        pattern="causal-boundaries",
        score_case="zero",
        run_label="unit-boundaries",
    )

    predecessor = torch.nextafter(
        torch.tensor(1.0, dtype=torch.bfloat16),
        torch.tensor(0.0, dtype=torch.bfloat16),
    )
    assert torch.all(dq_rows[0] == 1.0)
    assert torch.all(dq_rows[60] == predecessor)

    bad_dq = dq.clone()
    bad_dq[0].fill_(predecessor.item())
    with pytest.raises(AssertionError, match="staged-profile"):
        PROBE._check_zero_score_analytic(
            (seqlen,),
            (seqlen,),
            (out, lse, (bad_dq, dk, dv)),
            gradient_source="out_lse",
            pattern="causal-boundaries",
            score_case="zero",
            run_label="unit-boundaries-wrong-row",
        )


def test_finite_large_uniform_score_checks_nonzero_long_dk_contract():
    k_length = 4
    out = torch.full((1, 32, 512), PROBE.ANALYTIC_VALUE, dtype=torch.bfloat16)
    lse = torch.full((32, 1), 32.0 + torch.log(torch.tensor(4.0)), dtype=torch.float32)
    dq = torch.full_like(out, 0.25)
    dk = torch.full((k_length, 4, 512), 2.0 / k_length, dtype=torch.bfloat16)
    dv = torch.full_like(dk, 8.0 / k_length)

    PROBE._check_zero_score_analytic(
        (1,),
        (k_length,),
        (out, lse, (dq, dk, dv)),
        gradient_source="out_lse",
        pattern="final",
        score_case="finite-large",
        run_label="unit-finite-large",
    )


def test_inactive_gradient_checker_covers_separate_dq_dk_and_dv():
    dq = torch.zeros(3, 2, 1)
    dk = torch.zeros(5, 1, 1)
    dv = torch.zeros(5, 1, 1)
    dq[0] = 1
    dk[:2] = 1
    dv[:2] = 1
    PROBE._assert_inactive_gradients_zero(
        (dq, dk, dv),
        q_bounds=(0, 1),
        k_bounds=(0, 2),
    )

    for index, name in enumerate(PROBE.GRAD_NAMES):
        bad = tuple(grad.clone() for grad in (dq, dk, dv))
        bad[index][-1] = 1
        with pytest.raises(AssertionError, match=name):
            PROBE._assert_inactive_gradients_zero(
                bad,
                q_bounds=(0, 1),
                k_bounds=(0, 2),
            )


def test_repeat_drift_helper_reports_nonbitwise_values_without_a_tolerance():
    first = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    same = first.clone()
    drifted = torch.tensor([0.0, 1.125], dtype=torch.bfloat16)
    assert PROBE._max_pairwise_abs(first, [same]) == 0.0
    assert PROBE._max_pairwise_abs(first, [same, drifted]) == 0.125
    assert PROBE._max_pairwise_abs(first, []) == 0.0


def test_fixed_parity_checks_outputs_lse_and_each_gradient():
    native = (
        torch.tensor([0.0], dtype=torch.bfloat16),
        torch.tensor([0.0], dtype=torch.float32),
        _triplet(0.0),
    )
    fixed = (
        torch.tensor([0.0], dtype=torch.bfloat16),
        torch.tensor([0.0], dtype=torch.float32),
        _triplet(0.0),
    )
    refs = (native[0].float(), native[1], _triplet(0.0))
    bf16_refs = (native[0], _triplet(0.0))
    assert PROBE._check_fixed_parity(native, fixed, refs, bf16_refs) == []

    bad_fixed = (fixed[0], fixed[1], _triplet(1.0))
    failures = PROBE._check_fixed_parity(native, bad_fixed, refs, bf16_refs)
    assert len(failures) == 3
    assert all("native/fixed" in failure for failure in failures)


def test_isolation_never_labels_active_gradient_drift_as_pass(monkeypatch):
    q = torch.zeros(2, 1, 1, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    do = torch.zeros_like(q)
    dlse = torch.zeros(1, 2, dtype=torch.float32)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    calls = 0

    def candidate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        grads = tuple(torch.zeros_like(tensor) for tensor in (q, k, v))
        if calls == 2:
            grads[0][0] = 16
        return torch.zeros_like(q), torch.zeros_like(dlse), grads

    def fp32_reference(*_args, **_kwargs):
        return (
            torch.zeros_like(q),
            torch.zeros_like(dlse),
            tuple(torch.zeros_like(tensor) for tensor in (q, k, v)),
        )

    def bf16_reference(*_args, **_kwargs):
        return torch.zeros_like(q), tuple(torch.zeros_like(tensor) for tensor in (q, k, v))

    monkeypatch.setattr(PROBE, "_run_candidate", candidate)
    monkeypatch.setattr(PROBE, "_run_fp32_reference", fp32_reference)
    monkeypatch.setattr(PROBE, "_run_bf16_reference", bf16_reference)

    with pytest.raises(AssertionError, match="packed-segment isolation reference failure"):
        PROBE._run_isolation(
            q,
            k,
            v,
            do,
            dlse,
            cu,
            cu,
            (1, 1),
            (1, 1),
            "out",
        )
