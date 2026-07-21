from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_production_soak",
    ROOT / "scripts/probe_h100_production_soak.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_production_soak_command_matrix_is_exact(tmp_path: Path) -> None:
    commands = PROBE._commands(
        tmp_path,
        global_repeats=3,
        local_seeds=(11, 12, 13),
        benchmark_warmup=2,
        benchmark_reps=5,
    )
    assert PROBE.EXPERIMENT == "EXP-0044"
    assert [label for label, _command in commands] == [
        "global-s64k-fwd-bwd",
        "global-q1-k262144",
        "local-s262144-seed-11",
        "local-s262144-seed-12",
        "local-s262144-seed-13",
    ]
    flat = [argument for _label, command in commands for argument in command]
    assert "global_s64k" in flat
    assert "fwd_bwd" in flat
    assert "262144" in flat
    assert "out_lse" in flat
    assert "final" in flat
    assert flat.count("--nondefault-stream") == 4


def test_production_soak_cli_requires_output_and_valid_seeds(tmp_path: Path) -> None:
    args = PROBE._build_parser().parse_args(
        ["--output-dir", str(tmp_path), "--local-seeds", "1,2,3"]
    )
    assert args.output_dir == tmp_path
    assert args.local_seeds == (1, 2, 3)
    with pytest.raises(SystemExit):
        PROBE._build_parser().parse_args([])
    with pytest.raises(SystemExit):
        PROBE._build_parser().parse_args(["--output-dir", str(tmp_path), "--local-seeds", "1,-2"])
