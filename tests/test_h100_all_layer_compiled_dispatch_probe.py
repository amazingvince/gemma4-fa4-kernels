from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gemma4_fa4.model_spec import GEMMA4_31B

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_all_layer_compiled_dispatch",
    ROOT / "scripts/probe_h100_all_layer_compiled_dispatch.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC.loader.exec_module(PROBE)
finally:
    sys.path.pop(0)


def test_all_layer_probe_identity_and_locked_family_counts() -> None:
    assert PROBE.EXPERIMENT == "EXP-0042"
    assert PROBE.BACKENDS == ("eager", "inductor")
    families = [PROBE._family(index) for index in range(GEMMA4_31B.num_hidden_layers)]
    assert families.count("local") == 50
    assert families.count("global") == 10
    assert [index for index, family in enumerate(families) if family == "global"] == list(
        range(5, 60, 6)
    )


def test_all_layer_probe_cli_is_explicit_and_bounded() -> None:
    args = PROBE._build_parser().parse_args(
        ["--backend", "inductor", "--seqlen", "1024", "--seed", "42"]
    )
    assert args.backend == "inductor"
    assert args.seqlen == 1024
    assert args.seed == 42
    with pytest.raises(SystemExit):
        PROBE._build_parser().parse_args(["--backend", "raw"])
