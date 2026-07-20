import importlib.util
import json
import sys
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


def test_fa4_adapter_routes_short_local_through_the_project_fixed_path(monkeypatch):
    captured = {}

    def fake_local(q, k, v, *, spec):
        captured.update(q_shape=q.shape, k_shape=k.shape, v_shape=v.shape, spec=spec)
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q, lse

    monkeypatch.setattr(BENCH, "fa4_local_text_forward", fake_local, raising=False)

    q = torch.zeros(1, 32, 8, 256, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 8, 256, dtype=torch.bfloat16)
    v = torch.ones(1, 16, 8, 256, dtype=torch.bfloat16)
    out = BENCH._fa4(q, k, v, BENCH.SLIDING_ATTENTION)

    assert out.shape == q.shape
    assert captured == {
        "q_shape": (1, 8, 32, 256),
        "k_shape": (1, 8, 16, 256),
        "v_shape": (1, 8, 16, 256),
        "spec": BENCH.SLIDING_ATTENTION,
    }


def test_fa4_adapter_routes_long_local_through_project_varlen(monkeypatch):
    captured = {}

    def fake_local_varlen(q, k, v, cu_q, cu_k, **kwargs):
        captured.update(q_shape=q.shape, cu_q=cu_q.clone(), cu_k=cu_k.clone(), kwargs=kwargs)
        lse = torch.zeros(q.shape[1], q.shape[0], dtype=torch.float32)
        return q, lse

    monkeypatch.setattr(BENCH, "fa4_local_varlen_forward", fake_local_varlen, raising=False)
    q = torch.zeros(1, 32, 1026, 256, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 1026, 256, dtype=torch.bfloat16)
    v = torch.ones(1, 16, 1026, 256, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1026], dtype=torch.int32)

    out = BENCH._fa4(q, k, v, BENCH.SLIDING_ATTENTION, cu_seqlens=cu)

    assert out.shape == q.shape
    assert captured["q_shape"] == (1026, 32, 256)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["kwargs"]["max_seqlen_q"] == 1026
    assert captured["kwargs"]["max_seqlen_k"] == 1026


def test_fa4_adapter_routes_global_forward_through_project_composition(monkeypatch):
    called = []

    def fake_global(q, k, v, *, spec):
        called.append((q.shape, k.shape, v.shape, spec))
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q, lse

    monkeypatch.setattr(BENCH, "fa4_global_forward_only", fake_global, raising=False)
    q = torch.zeros(1, 32, 8, 512, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 8, 512, dtype=torch.bfloat16)
    v = torch.ones(1, 4, 8, 512, dtype=torch.bfloat16)

    out = BENCH._fa4(q, k, v, BENCH.GLOBAL_ATTENTION)

    assert out.shape == q.shape
    assert called == [((1, 8, 32, 512), (1, 8, 4, 512), (1, 8, 4, 512), BENCH.GLOBAL_ATTENTION)]


def test_fa4_adapter_routes_long_global_backward_through_project_varlen(monkeypatch):
    captured = {}

    def fake_global_varlen(q, k, v, cu_q, cu_k, **kwargs):
        captured.update(q_shape=q.shape, cu_q=cu_q.clone(), cu_k=cu_k.clone(), kwargs=kwargs)
        lse = torch.zeros(q.shape[1], q.shape[0], dtype=torch.float32)
        return q, lse

    monkeypatch.setattr(BENCH, "fa4_global_varlen_forward", fake_global_varlen, raising=False)
    q = torch.zeros(1, 32, 2049, 512, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 4, 2049, 512, dtype=torch.bfloat16, requires_grad=True)
    v = torch.ones(1, 4, 2049, 512, dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 2049], dtype=torch.int32)

    out = BENCH._fa4(q, k, v, BENCH.GLOBAL_ATTENTION, cu_seqlens=cu)

    assert out.shape == q.shape
    assert captured["q_shape"] == (2049, 32, 512)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["kwargs"]["max_seqlen_q"] == 2049
    assert captured["kwargs"]["max_seqlen_k"] == 2049
