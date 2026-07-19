import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH_SPEC = importlib.util.spec_from_file_location(
    "bench_attention", ROOT / "benchmarks/bench_attention.py"
)
assert BENCH_SPEC is not None
assert BENCH_SPEC.loader is not None
BENCH = importlib.util.module_from_spec(BENCH_SPEC)
sys.modules[BENCH_SPEC.name] = BENCH
BENCH_SPEC.loader.exec_module(BENCH)


def test_benchmark_ladders_are_well_formed():
    for path in sorted((ROOT / "configs/benchmarks").glob("*.json")):
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        names = set()
        for item in data["entries"]:
            assert item["name"] not in names
            names.add(item["name"])
            assert item["layer"] in {"sliding_attention", "full_attention"}
            assert item["mode"] in {"fwd", "bwd", "fwd_bwd"}
            assert item["batch"] > 0 and item["seqlen"] > 0
            qh = item.get("q_heads", 32)
            kvh = item.get("kv_heads", 16 if item["layer"] == "sliding_attention" else 4)
            assert qh > 0 and kvh > 0 and qh % kvh == 0


def test_forward_memory_plan_omits_upstream_gradient_and_counts_cold_l2():
    case = BENCH.BenchCase("fwd", BENCH.SLIDING_ATTENTION, batch=1, seqlen=8, mode="fwd")
    working = BENCH._estimated_bytes(case, 2)
    total, thrash = BENCH._memory_plan(case, 2, "cold", 16 << 20)
    assert thrash == 64 << 20
    assert total == working + thrash
    assert BENCH._make_grad_out(case, torch.bfloat16, device="cpu") is None


def test_backward_memory_plan_allocates_an_upstream_gradient():
    case = BENCH.BenchCase("bwd", BENCH.SLIDING_ATTENTION, batch=1, seqlen=8, mode="bwd")
    grad_out = BENCH._make_grad_out(case, torch.bfloat16, device="cpu")
    assert grad_out is not None
    assert grad_out.shape == (1, case.spec.num_q_heads, 8, case.spec.head_dim_v)


def test_generated_benchmark_masks_are_explicitly_text_only():
    assert BENCH._mask_semantics(BENCH.SLIDING_ATTENTION) == "local_text_causal_window"
    assert BENCH._mask_semantics(BENCH.GLOBAL_ATTENTION) == "global_causal"


def test_fa4_adapter_unpacks_pinned_out_lse_api_and_requests_scale(monkeypatch):
    captured = {}

    def fake_flash_attn_func(q, k, v, **kwargs):
        captured.update(kwargs)
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q, lse

    package = types.ModuleType("flash_attn")
    cute = types.ModuleType("flash_attn.cute")
    cute.flash_attn_func = fake_flash_attn_func
    package.cute = cute
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)

    q = torch.zeros(1, 32, 8, 256, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 8, 256, dtype=torch.bfloat16)
    v = torch.ones(1, 16, 8, 256, dtype=torch.bfloat16)
    out = BENCH._fa4(q, k, v, BENCH.SLIDING_ATTENTION)

    assert out.shape == q.shape
    assert captured == {
        "causal": True,
        "softmax_scale": 1.0,
        "window_size": (1023, 0),
        "return_lse": True,
    }
