from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_transformers_cache",
    ROOT / "scripts/probe_h100_transformers_cache.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _application_key(
    abi: str,
    scheduler_class: str,
    variant: str,
) -> tuple[object, ...]:
    flags = {
        "single_single": (True, True),
        "single_multi": (True, False),
        "multi_multi": (False, False),
    }[scheduler_class]
    core = tuple(range(23))
    if abi == "fixed_bshd":
        return (*core, *flags, variant)
    if abi == "native_thd":
        return (*core, *flags, False, False, variant)
    raise ValueError(abi)


def _cumulative_values(lengths: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return tuple(values)


def test_scheduler_contract_has_exactly_three_reachable_classes():
    assert PROBE._scheduler_class(1, 1) == "single_single"
    assert PROBE._scheduler_class(32, 32) == "single_single"
    assert PROBE._scheduler_class(33, 33) == "single_multi"
    assert PROBE._scheduler_class(64, 2048) == "single_multi"
    assert PROBE._scheduler_class(65, 65) == "multi_multi"
    assert PROBE._scheduler_class(129, 2048) == "multi_multi"
    assert PROBE._scheduler_class(1, 262_144) == "single_multi"
    assert PROBE._scheduler_class(32768, 32768) == "multi_multi"

    with pytest.raises(ValueError, match="1 <= Sq <= Sk <= 262144"):
        PROBE._scheduler_class(33, 32)
    with pytest.raises(ValueError, match="1 <= Sq <= Sk <= 262144"):
        PROBE._scheduler_class(1, 262_145)
    with pytest.raises(ValueError, match="impossible"):
        PROBE._scheduler_class_from_flags(False, True)


def test_native_matrix_changes_runtime_values_without_changing_class():
    assert tuple(case[0] for case in PROBE._NATIVE_BACKWARD_CASES) == (
        "single_single",
        "single_multi",
        "multi_multi",
    )

    replay_k_lengths: set[int] = set()
    for scheduler_class, warm, replay in PROBE._NATIVE_BACKWARD_CASES:
        assert PROBE._validate_packed_lengths(warm.q_lengths, warm.k_lengths) == scheduler_class
        assert PROBE._validate_packed_lengths(replay.q_lengths, replay.k_lengths) == scheduler_class
        assert len(warm.q_lengths) != len(replay.q_lengths)
        assert sum(warm.q_lengths) != sum(replay.q_lengths)
        assert sum(warm.k_lengths) != sum(replay.k_lengths)
        assert _cumulative_values(warm.q_lengths) != _cumulative_values(replay.q_lengths)
        assert _cumulative_values(warm.k_lengths) != _cumulative_values(replay.k_lengths)
        assert warm.q_lengths[0] in replay.q_lengths[1:]
        assert warm.layout != replay.layout
        replay_k_lengths.update(replay.k_lengths)

    assert {1025, 2048} <= replay_k_lengths


def test_packed_length_validation_rejects_unrepresentable_runtime_cases():
    with pytest.raises(ValueError, match="nonzero batch count"):
        PROBE._validate_packed_lengths((), ())
    with pytest.raises(ValueError, match="same nonzero batch count"):
        PROBE._validate_packed_lengths((1,), (1, 2))
    with pytest.raises(ValueError, match="0 <= Sq <= Sk <= 262144"):
        PROBE._validate_packed_lengths((2,), (1,))
    assert PROBE._validate_packed_lengths((0, 1, 0), (1, 1, 0)) == "single_single"
    with pytest.raises(ValueError, match="positive Q and K totals"):
        PROBE._validate_packed_lengths((0, 0), (0, 1))
    assert PROBE._validate_packed_lengths((1,), (2049,)) == "single_multi"
    assert PROBE._validate_packed_lengths((1,), (262_144,)) == "single_multi"
    with pytest.raises(ValueError, match="0 <= Sq <= Sk <= 262144"):
        PROBE._validate_packed_lengths((1,), (262_145,))
    with pytest.raises(ValueError, match="0 <= Sq <= Sk <= 262144"):
        PROBE._validate_packed_lengths((True,), (1,))


def test_long_native_replays_cover_each_required_runtime_without_new_class():
    cases = dict(PROBE._NATIVE_LONG_REUSE_CASES)
    assert set(cases) == {"mixed_k2049_k4097", "square_s32768", "max_k262144"}
    assert cases["mixed_k2049_k4097"].k_lengths == (2049, 4097)
    assert cases["square_s32768"].q_lengths == cases["square_s32768"].k_lengths == (32768,)
    assert cases["max_k262144"].q_lengths == (1,)
    assert cases["max_k262144"].k_lengths == (262144,)
    assert {
        PROBE._validate_packed_lengths(case.q_lengths, case.k_lengths) for case in cases.values()
    } == {"single_multi", "multi_multi"}


def test_empty_native_replays_cover_plateaus_without_new_scheduler_class():
    cases = dict(PROBE._NATIVE_EMPTY_REUSE_CASES)
    assert set(cases) == {
        "single_single_middle_paired_empty",
        "single_multi_leading_q_empty",
        "multi_multi_middle_q_empty",
    }
    assert {
        PROBE._validate_packed_lengths(case.q_lengths, case.k_lengths) for case in cases.values()
    } == set(PROBE._SCHEDULER_CLASSES)
    for case in cases.values():
        assert 0 in case.q_lengths
        assert sum(case.q_lengths) > 0
        assert sum(case.k_lengths) > 0
        assert any(
            q_length == 0 and k_length > 0
            for q_length, k_length in zip(case.q_lengths, case.k_lengths, strict=True)
        ) or any(
            q_length == k_length == 0
            for q_length, k_length in zip(case.q_lengths, case.k_lengths, strict=True)
        )


def test_cumulative_builder_preserves_zero_length_plateaus(monkeypatch):
    tensor = PROBE.torch.tensor
    monkeypatch.setattr(
        PROBE.torch,
        "tensor",
        lambda values, *, device, dtype: tensor(values, dtype=dtype),
    )
    assert PROBE._cumulative((0, 3, 0, 1)).tolist() == [0, 0, 3, 3, 4]
    with pytest.raises(ValueError, match="nonnegative"):
        PROBE._cumulative((1, -1))


def test_application_key_inventory_separates_fixed_bshd_and_native_thd():
    keys = [
        _application_key(abi, scheduler_class, variant)
        for abi in ("fixed_bshd", "native_thd")
        for scheduler_class in PROBE._SCHEDULER_CLASSES
        for variant in PROBE._BACKWARD_VARIANTS
    ]
    snapshot = PROBE._application_snapshot_from_keys(keys)
    expected = PROBE._application_coordinates(
        "fixed_bshd", PROBE._SCHEDULER_CLASSES
    ) | PROBE._application_coordinates("native_thd", PROBE._SCHEDULER_CLASSES)

    assert set(snapshot) == expected
    assert len(snapshot) == 18
    fixed_hashes = {
        digest for coordinate, digest in snapshot.items() if coordinate[0] == "fixed_bshd"
    }
    native_hashes = {
        digest for coordinate, digest in snapshot.items() if coordinate[0] == "native_thd"
    }
    assert fixed_hashes.isdisjoint(native_hashes)


def test_application_key_decoder_fails_closed_on_abi_or_class_drift():
    native = list(_application_key("native_thd", "single_multi", "dkv"))
    native[-3] = True
    with pytest.raises(AssertionError, match="ABI marker"):
        PROBE._decode_global_backward_application_key(tuple(native))

    impossible = list(_application_key("fixed_bshd", "single_multi", "dq_lo"))
    impossible[-3:-1] = (False, True)
    with pytest.raises(ValueError, match="impossible"):
        PROBE._decode_global_backward_application_key(tuple(impossible))

    with pytest.raises(AssertionError, match="key length changed"):
        PROBE._decode_global_backward_application_key((1, 2, "dq_hi"))
    with pytest.raises(ValueError, match="unknown global backward ABI"):
        PROBE._application_coordinates("other", PROBE._SCHEDULER_CLASSES)

    first = _application_key("fixed_bshd", "single_single", "dkv")
    duplicate = (999, *first[1:])
    with pytest.raises(AssertionError, match="duplicate global backward"):
        PROBE._application_snapshot_from_keys((first, duplicate))


def test_application_delta_requires_exact_expected_class(capsys):
    fixed_key = _application_key("fixed_bshd", "single_single", "dkv")
    native_keys = [
        _application_key("native_thd", "single_multi", variant)
        for variant in PROBE._BACKWARD_VARIANTS
    ]
    before = PROBE._application_snapshot_from_keys((fixed_key,))
    after = PROBE._application_snapshot_from_keys((fixed_key, *native_keys))
    expected = PROBE._application_coordinates("native_thd", ("single_multi",))

    PROBE._expect_application_additions(before, after, label="test", expected=expected)
    assert "application_transition label=test" in capsys.readouterr().out

    with pytest.raises(AssertionError, match="application-key delta mismatch"):
        PROBE._expect_application_additions(before, after, label="test", expected=set())
