from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_local_varlen_cache",
    ROOT / "scripts/probe_h100_local_varlen_cache.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_parser_preserves_default_mode_and_adds_explicit_static_prefix_mode():
    default = PROBE._build_parser().parse_args([])
    assert not default.static_prefix_replay
    assert not default.custom
    assert not default.backward
    assert not default.long_text
    assert not default.empty_replay

    static = PROBE._build_parser().parse_args(["--static-prefix-replay"])
    assert static.static_prefix_replay


def test_static_prefix_cases_change_capacity_and_outer_stride_layout():
    PROBE._validate_static_prefix_cases(PROBE._STATIC_PREFIX_CASES)
    assert PROBE._STATIC_PREFIX_Q_LENGTH == 1
    assert PROBE._STATIC_PREFIX_K_LENGTH == 33
    assert {case.physical_capacity for case in PROBE._STATIC_PREFIX_CASES} == {65, 129}
    assert {case.layout for case in PROBE._STATIC_PREFIX_CASES} == {
        "bhsd-contiguous",
        "bshd-backed",
    }

    with pytest.raises(ValueError, match="at least two"):
        PROBE._validate_static_prefix_cases(PROBE._STATIC_PREFIX_CASES[:1])
    with pytest.raises(ValueError, match="distinct physical capacities"):
        PROBE._validate_static_prefix_cases(
            (
                PROBE._StaticPrefixCase(65, "bhsd-contiguous"),
                PROBE._StaticPrefixCase(65, "bshd-backed"),
            )
        )


def test_exact_result_requires_bitwise_output_and_lse_equivalence():
    torch = PROBE.torch
    reference = PROBE.Gemma4DispatchResult(
        output=torch.tensor([1.0], dtype=torch.bfloat16),
        lse=torch.tensor([2.0], dtype=torch.float32),
        path="fa4_local_varlen",
    )
    matching = PROBE.Gemma4DispatchResult(
        output=reference.output.clone(),
        lse=reference.lse.clone(),
        path=reference.path,
    )
    PROBE._require_exact_result(reference, matching, label="matching")

    changed = PROBE.Gemma4DispatchResult(
        output=torch.tensor([3.0], dtype=torch.bfloat16),
        lse=reference.lse.clone(),
        path=reference.path,
    )
    with pytest.raises(AssertionError, match="changed the logical-prefix output"):
        PROBE._require_exact_result(reference, changed, label="changed")


def test_cache_identity_rejects_object_or_forward_application_growth():
    expected_objects = {"fwd/a.o": "digest-a"}
    expected_applications = ("key-a",)
    PROBE._require_cache_identity(
        expected_objects,
        dict(expected_objects),
        expected_applications,
        tuple(expected_applications),
        label="same",
    )

    with pytest.raises(AssertionError, match="changed persistent cache objects"):
        PROBE._require_cache_identity(
            expected_objects,
            {**expected_objects, "fwd/b.o": "digest-b"},
            expected_applications,
            expected_applications,
            label="object-growth",
        )
    with pytest.raises(AssertionError, match="changed _flash_attn_fwd application keys"):
        PROBE._require_cache_identity(
            expected_objects,
            expected_objects,
            expected_applications,
            (*expected_applications, "key-b"),
            label="application-growth",
        )


def test_application_key_digest_snapshot_is_order_independent():
    first = ("bf16", 256, True)
    second = ("bf16", 256, False)
    assert PROBE._application_key_digests((first, second)) == PROBE._application_key_digests(
        (second, first)
    )
