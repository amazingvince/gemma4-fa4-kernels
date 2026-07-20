from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_h100_guarded_compile_facade",
    ROOT / "scripts/probe_h100_guarded_compile_facade.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC.loader.exec_module(PROBE)
finally:
    sys.path.pop(0)


def test_guarded_probe_identity_and_matrix_are_locked() -> None:
    assert PROBE.EXPERIMENT == "EXP-0023"
    assert PROBE.DEFAULT_LENGTHS == (1, 32, 33, 1023, 1024)
    assert PROBE.FAMILY_LAYERS == {"local": 0, "global": 5}
    assert PROBE.BACKENDS == ("eager", "inductor")


def test_sanitizer_case_cli_is_explicit_and_narrow() -> None:
    args = PROBE._build_parser().parse_args(
        [
            "--family",
            "global",
            "--backend",
            "inductor",
            "--lengths",
            "1024",
            "--sanitizer-case",
        ]
    )
    assert args.family == "global"
    assert args.backend == "inductor"
    assert args.lengths == (1024,)
    assert args.sanitizer_case is True


def test_scoped_size_oblivious_cli_is_explicit() -> None:
    args = PROBE._build_parser().parse_args(["--scoped-size-oblivious"])
    assert args.scoped_size_oblivious is True
    assert args.sanitizer_case is False


def test_forbidden_inner_source_inventory_distinguishes_shape_symbols() -> None:
    graphs = [
        {
            "nodes": [
                {"op": "placeholder", "target": "s0", "example_value_type": "SymInt"},
                {
                    "op": "placeholder",
                    "target": "q_proj_weight",
                    "example_value_type": "FakeTensor",
                },
                {
                    "op": "placeholder",
                    "target": "module_scaling",
                    "example_value_type": "SymFloat",
                },
                {"op": "call_method", "target": "item", "example_value_type": "float"},
            ]
        }
    ]

    inventory = PROBE._forbidden_inner_sources(graphs)
    assert [(item["node_index"], item["target"]) for item in inventory] == [
        (2, "module_scaling"),
        (3, "item"),
    ]


def test_scalar_field_table_covers_and_restores_all_eleven_sources() -> None:
    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {"rope_theta": 10_000.0},
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
        attention_dropout=0.0,
        final_logit_softcapping=30.0,
        rms_norm_eps=1e-6,
    )
    module = SimpleNamespace(
        scaling=1.0,
        attention_dropout=0.0,
        q_norm=SimpleNamespace(eps=1e-6),
        k_norm=SimpleNamespace(eps=1e-6),
        v_norm=SimpleNamespace(eps=1e-6),
    )
    runtime = SimpleNamespace(config=config, layer=module)
    fields = PROBE._scalar_fields(runtime)

    assert len(fields) == 11
    assert len({field.name for field in fields}) == 11
    for field in fields:
        original = field.getter()
        field.setter(field.mutated)
        assert field.getter() == field.mutated
        field.setter(original)
        assert field.getter() == original
