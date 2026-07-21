import importlib.util
import json
import sys
from pathlib import Path

import pytest
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


def test_expanded_sdpa_memory_plan_counts_expanded_values_and_backward_gradients():
    case = BENCH.BenchCase("bwd", BENCH.GLOBAL_ATTENTION, batch=1, seqlen=8, mode="bwd")
    base, _ = BENCH._memory_plan(case, 2, "hot", 16 << 20)
    expanded, _ = BENCH._memory_plan(case, 2, "hot", 16 << 20, impl="sdpa_expanded")
    one_expanded_kv_pair = 1 * 8 * 32 * (512 + 512) * 2
    assert expanded == base + 2 * one_expanded_kv_pair


def test_generated_benchmark_masks_are_explicitly_text_only():
    assert BENCH._mask_semantics(BENCH.SLIDING_ATTENTION) == "local_text_causal_window"
    assert BENCH._mask_semantics(BENCH.GLOBAL_ATTENTION) == "global_causal"


def test_benchmark_result_labels_owner_dkv_default_and_rollback(monkeypatch):
    case = BENCH.BenchCase("global", BENCH.GLOBAL_ATTENTION, batch=1, seqlen=8, mode="bwd")
    monkeypatch.delenv("FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV", raising=False)
    assert BENCH._owner_dkv_enabled_for_result(case, impl="fa4", deterministic=False)
    assert not BENCH._owner_dkv_enabled_for_result(case, impl="fa4", deterministic=True)

    monkeypatch.setenv("FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV", "0")
    assert not BENCH._owner_dkv_enabled_for_result(case, impl="fa4", deterministic=False)


def test_explicit_kv_expansion_preserves_global_gqa_output_and_gradients():
    generator = torch.Generator().manual_seed(34001)
    q = torch.randn(1, 32, 5, 512, generator=generator, requires_grad=True)
    k = torch.randn(1, 4, 5, 512, generator=generator, requires_grad=True)
    v = torch.randn(1, 4, 5, 512, generator=generator, requires_grad=True)
    grad_out = torch.randn(1, 32, 5, 512, generator=generator)

    expanded = BENCH._sdpa_expanded(q, k, v, BENCH.GLOBAL_ATTENTION)
    expanded_grads = torch.autograd.grad(expanded, (q, k, v), grad_out)
    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    reference = BENCH.reference_layer(BENCH.GLOBAL_ATTENTION, q_ref, k_ref, v_ref)
    reference_grads = torch.autograd.grad(reference, (q_ref, k_ref, v_ref), grad_out)

    torch.testing.assert_close(expanded, reference, atol=2e-4, rtol=2e-4)
    for expanded_grad, reference_grad in zip(expanded_grads, reference_grads, strict=True):
        torch.testing.assert_close(expanded_grad, reference_grad, atol=2e-4, rtol=2e-4)


def test_explicit_kv_expansion_rejects_local_mask_substitution():
    q = torch.zeros(1, 32, 5, 256)
    k = torch.zeros(1, 16, 5, 256)
    v = torch.zeros(1, 16, 5, 256)
    with pytest.raises(BENCH.UnsupportedSemanticBaseline, match="not a sliding-window baseline"):
        BENCH._sdpa_expanded(q, k, v, BENCH.SLIDING_ATTENTION)


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

    out = BENCH._fa4(
        q,
        k,
        v,
        BENCH.GLOBAL_ATTENTION,
        cu_seqlens=cu,
        deterministic=True,
    )

    assert out.shape == q.shape
    assert captured["q_shape"] == (2049, 32, 512)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["kwargs"]["max_seqlen_q"] == 2049
    assert captured["kwargs"]["max_seqlen_k"] == 2049
    assert captured["kwargs"]["deterministic"] is True


def test_fa4_adapter_routes_short_global_deterministic_backward(monkeypatch):
    captured = {}

    def fake_global(q, k, v, **kwargs):
        captured.update(kwargs)
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return q, lse

    monkeypatch.setattr(BENCH, "fa4_global_text_forward", fake_global, raising=False)
    q = torch.zeros(1, 32, 8, 512, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 4, 8, 512, dtype=torch.bfloat16, requires_grad=True)
    v = torch.ones(1, 4, 8, 512, dtype=torch.bfloat16, requires_grad=True)

    out = BENCH._fa4(q, k, v, BENCH.GLOBAL_ATTENTION, deterministic=True)

    assert out.shape == q.shape
    assert captured == {"spec": BENCH.GLOBAL_ATTENTION, "deterministic": True}


def test_deterministic_benchmark_rejects_local_and_non_fa4_routes():
    q = torch.zeros(1, 32, 8, 256)
    k = torch.zeros(1, 16, 8, 256)
    v = torch.zeros(1, 16, 8, 256)
    with pytest.raises(BENCH.UnsupportedSemanticBaseline, match="only for global FA4"):
        BENCH._fa4(q, k, v, BENCH.SLIDING_ATTENTION, deterministic=True)
    with pytest.raises(BENCH.UnsupportedSemanticBaseline, match="cannot label"):
        BENCH._run("sdpa", q, k, v, BENCH.SLIDING_ATTENTION, deterministic=True)
