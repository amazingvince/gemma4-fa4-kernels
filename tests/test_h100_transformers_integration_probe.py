from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_transformers_integration",
    ROOT / "scripts/probe_h100_transformers_integration.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _document_lengths(document_ids: torch.Tensor) -> tuple[int, ...]:
    raw = [int(value) for value in document_ids.tolist()]
    boundaries = [index for index in range(1, len(raw)) if raw[index] != raw[index - 1]]
    points = [0, *boundaries, len(raw)]
    return tuple(end - start for start, end in zip(points[:-1], points[1:], strict=True))


def test_exp0013_edge_cases_are_registered():
    expected = {
        "global-document-split-backward",
        "global-native-varlen-noncontiguous-gradients",
    }

    assert expected <= set(PROBE.CASES)
    assert expected <= set(PROBE.RUNNERS)


def test_exp0015_empty_row_case_is_registered():
    assert "global-packed-empty-row" in PROBE.CASES
    assert "global-packed-empty-row" in PROBE.RUNNERS


def test_exp0016_static_cache_cases_are_registered():
    expected = {
        "static-cache-local-small",
        "static-cache-local-boundary",
        "static-cache-local-first-roll",
        "static-cache-global-small",
        "static-cache-global-k1025",
    }

    assert set(PROBE.STATIC_CACHE_CASES) == expected
    assert expected <= set(PROBE.CASES)
    assert expected <= set(PROBE.RUNNERS)


@pytest.mark.parametrize(
    ("case", "spec", "layer_idx", "capacity", "prompt_length", "step_names", "paths"),
    (
        (
            "static-cache-local-small",
            PROBE.SLIDING_ATTENTION,
            0,
            1024,
            32,
            ("prefill", "decode"),
            ("fa4_local_fixed", "fa4_local_varlen"),
        ),
        (
            "static-cache-local-boundary",
            PROBE.SLIDING_ATTENTION,
            0,
            1024,
            1023,
            ("prefill", "boundary_decode"),
            ("fa4_local_fixed", "fa4_local_varlen"),
        ),
        (
            "static-cache-local-first-roll",
            PROBE.SLIDING_ATTENTION,
            0,
            1024,
            1023,
            ("prefill", "boundary_decode", "first_roll"),
            ("fa4_local_fixed", "fa4_local_varlen", "fa4_local_varlen"),
        ),
        (
            "static-cache-global-small",
            PROBE.GLOBAL_ATTENTION,
            5,
            65,
            32,
            ("prefill", "decode"),
            ("fa4_global_fixed", "fa4_global_varlen"),
        ),
        (
            "static-cache-global-k1025",
            PROBE.GLOBAL_ATTENTION,
            5,
            1026,
            1024,
            ("prefill", "decode"),
            ("fa4_global_fixed", "fa4_global_forward_only"),
        ),
    ),
)
def test_exp0016_static_cache_runner_dispatch(
    monkeypatch,
    case,
    spec,
    layer_idx,
    capacity,
    prompt_length,
    step_names,
    paths,
):
    captured = None

    def run_static_cache_sequence(**kwargs):
        nonlocal captured
        captured = kwargs
        return {"case": kwargs["case"], "paths": list(kwargs["expected_paths"])}

    monkeypatch.setattr(PROBE, "_run_static_cache_sequence", run_static_cache_sequence)

    result = PROBE.RUNNERS[case](seed=73)

    assert captured == {
        "case": case,
        "spec": spec,
        "layer_idx": layer_idx,
        "capacity": capacity,
        "prompt_length": prompt_length,
        "step_names": step_names,
        "expected_paths": paths,
        "seed": 73,
    }
    assert result == {"case": case, "paths": list(paths)}


@pytest.mark.parametrize("case", PROBE.STATIC_CACHE_CASES)
def test_exp0016_cli_selects_each_static_cache_case(monkeypatch, capsys, case):
    calls = []

    def runner(seed):
        calls.append(seed)
        return {"case": case, "path": "sentinel"}

    monkeypatch.setitem(PROBE.RUNNERS, case, runner)
    monkeypatch.setattr(PROBE.sys, "argv", ["probe", "--case", case, "--seed", "91"])
    monkeypatch.setattr(PROBE.torch.cuda, "is_available", lambda: False)

    assert PROBE.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == [91]
    assert payload == {
        "device": None,
        "requested_case": case,
        "results": [{"case": case, "path": "sentinel"}],
        "status": "passed",
    }


def test_packed_thd_reference_resets_causality_at_document_boundaries():
    spec = PROBE.GLOBAL_ATTENTION
    lengths = (2, 3)
    q = torch.zeros(5, spec.num_q_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    k = torch.zeros(5, spec.num_kv_heads, spec.head_dim_qk, dtype=torch.bfloat16)
    values = torch.tensor((1, 3, 10, 20, 30), dtype=torch.bfloat16)
    v = values[:, None, None].expand(-1, spec.num_kv_heads, spec.head_dim_v).clone()
    builder = PROBE._packed_thd_builder(
        lengths,
        lengths,
        spec,
        upcast=torch.float32,
    )

    output, lse = builder(q, k, v)

    expected = torch.tensor((1, 2, 10, 15, 20), dtype=torch.bfloat16)
    torch.testing.assert_close(output[:, 0, 0], expected)
    expected_lse = torch.log(torch.tensor((1, 2, 1, 2, 3), dtype=torch.float32))
    torch.testing.assert_close(lse[0], expected_lse)
    assert output.shape == q.shape
    assert lse.shape == (spec.num_q_heads, q.shape[0])


def test_odd_padded_gradient_outputs_are_noncontiguous_unit_inner_stride_views():
    output = torch.empty((7, 32, 512), dtype=torch.bfloat16)
    lse = torch.empty((32, 7), dtype=torch.float32)

    dout, dlse = PROBE._odd_padded_gradient_outputs(output, lse, seed=13)
    repeated_dout, repeated_dlse = PROBE._odd_padded_gradient_outputs(
        output,
        lse,
        seed=13,
    )

    assert dout.shape == output.shape
    assert dlse.shape == lse.shape
    assert dout.dtype == torch.bfloat16
    assert dlse.dtype == torch.float32
    assert dout.stride() == (32 * 513, 513, 1)
    assert dlse.stride() == (8, 1)
    assert not dout.is_contiguous()
    assert not dlse.is_contiguous()
    assert torch.equal(dout, repeated_dout)
    assert torch.equal(dlse, repeated_dlse)


def test_document_split_runner_checks_hostile_forward_and_gradient_isolation(monkeypatch):
    def make_inputs(spec, *, batch, q_length, kv_length=None, seed):
        kv_length = q_length if kv_length is None else kv_length
        generator = torch.Generator().manual_seed(seed)
        shapes = (
            (batch, spec.num_q_heads, q_length, spec.head_dim_qk),
            (batch, spec.num_kv_heads, kv_length, spec.head_dim_qk),
            (batch, spec.num_kv_heads, kv_length, spec.head_dim_v),
        )
        return tuple(
            torch.randn(shape, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
            for shape in shapes
        )

    def module_call(_module, spec, q, k, v, _attention_mask=None, **kwargs):
        document_ids = kwargs["document_ids"]
        lengths = _document_lengths(document_ids[0])
        builder = PROBE._packed_rectangular_builder(
            lengths,
            lengths,
            spec,
            upcast=torch.float32,
        )
        output, lse = builder(q, k, v)
        return PROBE.Gemma4DispatchResult(
            output=output,
            lse=lse,
            path="fa4_global_varlen_native",
        )

    def validate_kernel_result(**kwargs):
        assert kwargs["expected_path"] == "fa4_global_varlen_native"
        assert kwargs["result"].path == kwargs["expected_path"]
        return {"case": kwargs["case"], "path": kwargs["result"].path}

    monkeypatch.setattr(PROBE, "_require_h100", lambda: None)
    monkeypatch.setattr(PROBE, "_make_inputs", make_inputs)
    monkeypatch.setattr(PROBE, "_module_call", module_call)
    monkeypatch.setattr(PROBE, "_validate_kernel_result", validate_kernel_result)

    record = PROBE._run_global_document_split_backward(seed=17)

    assert record["path"] == "fa4_global_varlen_native"
    assert record["document_lengths"] == [31, 33, 65]
    assert record["rebuilt_cu_seqlens"] == [0, 31, 64, 129]
    assert record["hostile_forward_isolation"] is True
    assert record["structured_gradient_isolation"] is True
    assert set(record["isolation_gradient_errors"]) == {"dQ", "dK", "dV"}


def test_empty_row_runner_checks_sentinels_and_neighbor_isolation(monkeypatch):
    def make_inputs(spec, *, batch, q_length, kv_length=None, seed):
        kv_length = q_length if kv_length is None else kv_length
        generator = torch.Generator().manual_seed(seed)
        shapes = (
            (batch, spec.num_q_heads, q_length, spec.head_dim_qk),
            (batch, spec.num_kv_heads, kv_length, spec.head_dim_qk),
            (batch, spec.num_kv_heads, kv_length, spec.head_dim_v),
        )
        return tuple(
            torch.randn(shape, dtype=torch.bfloat16, generator=generator).requires_grad_(True)
            for shape in shapes
        )

    def module_call(_module, spec, q, k, v, attention_mask=None, **_kwargs):
        assert attention_mask is not None
        active = attention_mask.attention_mask.to(dtype=q.dtype)[:, None, :, None]
        expanded_k = torch.repeat_interleave(k, spec.qhead_per_kvhead, dim=1)
        expanded_v = torch.repeat_interleave(v, spec.qhead_per_kvhead, dim=1)
        output = ((q + expanded_k + expanded_v) * active).transpose(1, 2)
        finite_lse = q[..., 0].float() + expanded_k[..., 0].float()
        lse = torch.where(
            active[..., 0].bool(),
            finite_lse,
            torch.full_like(finite_lse, -torch.inf),
        )
        return PROBE.Gemma4DispatchResult(
            output=output,
            lse=lse,
            path="fa4_global_varlen_native",
        )

    def validate_kernel_result(**kwargs):
        assert kwargs["expected_path"] == "fa4_global_varlen_native"
        assert kwargs["result"].path == kwargs["expected_path"]
        return {"case": kwargs["case"], "path": kwargs["result"].path}

    monkeypatch.setattr(PROBE, "_require_h100", lambda: None)
    monkeypatch.setattr(PROBE, "_make_inputs", make_inputs)
    monkeypatch.setattr(PROBE, "_module_call", module_call)
    monkeypatch.setattr(PROBE, "_validate_kernel_result", validate_kernel_result)
    monkeypatch.setattr(PROBE.torch.cuda, "is_available", lambda: False)

    record = PROBE._run_global_packed_empty_row(seed=19)

    assert record["path"] == "fa4_global_varlen_native"
    assert record["packed_lengths"] == [0, 65]
    assert record["cu_seqlens"] == [0, 0, 65]
    assert record["padding_sentinels"] == "zero_O/-inf_LSE"
    assert record["hostile_empty_row_isolation"] is True
    assert record["empty_row_gradients_exact_zero"] is True
