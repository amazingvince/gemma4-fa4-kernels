from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from gemma4_fa4 import transformers_integration as integration
from gemma4_fa4.h100 import UnsupportedH100Path
from gemma4_fa4.model_spec import GEMMA4_31B


class _PinnedConfig:
    _attn_implementation = integration.BACKEND_NAME
    hidden_size = GEMMA4_31B.hidden_size
    intermediate_size = 21_504
    num_hidden_layers = GEMMA4_31B.num_hidden_layers
    num_attention_heads = GEMMA4_31B.sliding.num_q_heads
    num_key_value_heads = GEMMA4_31B.sliding.num_kv_heads
    num_global_key_value_heads = GEMMA4_31B.full.num_kv_heads
    head_dim = GEMMA4_31B.sliding.head_dim_qk
    global_head_dim = GEMMA4_31B.full.head_dim_qk
    sliding_window = GEMMA4_31B.sliding.sliding_window
    max_position_embeddings = GEMMA4_31B.max_position_embeddings
    attention_dropout = 0.0
    attention_bias = False
    is_causal = True
    attention_k_eq_v = True
    num_kv_shared_layers = GEMMA4_31B.num_kv_shared_layers
    use_bidirectional_attention = GEMMA4_31B.use_bidirectional_attention
    hidden_size_per_layer_input = 0
    final_logit_softcapping = 30.0
    rms_norm_eps = 1e-6
    rope_parameters = {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    }
    layer_types = list(GEMMA4_31B.layer_types())


class _PinnedAttention:
    def __init__(self, layer_idx: int, config: _PinnedConfig) -> None:
        spec = GEMMA4_31B.spec_for_layer(layer_idx)
        self.layer_idx = layer_idx
        self.config = config
        self.training = False
        self.scaling = 1.0
        self.head_dim = spec.head_dim_qk
        self.num_key_value_groups = spec.qhead_per_kvhead


class _WholeLayerPinnedAttention(torch.nn.Module):
    def __init__(self, layer_idx: int, config: _PinnedConfig) -> None:
        super().__init__()
        spec = GEMMA4_31B.spec_for_layer(layer_idx)
        hidden_size = GEMMA4_31B.hidden_size
        q_width = spec.num_q_heads * spec.head_dim_qk
        kv_width = spec.num_kv_heads * spec.head_dim_qk
        factory = {"device": "cuda", "dtype": torch.bfloat16}
        self.layer_idx = layer_idx
        self.config = config
        self.is_sliding = spec.kind == "sliding_attention"
        self.is_kv_shared_layer = False
        self.store_full_length_kv = False
        self.use_alternative_attention = spec.kind == "full_attention"
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.head_dim = spec.head_dim_qk
        self.num_key_value_groups = spec.qhead_per_kvhead
        self.q_proj = torch.nn.Linear(hidden_size, q_width, bias=False, **factory)
        self.k_proj = torch.nn.Linear(hidden_size, kv_width, bias=False, **factory)
        self.v_proj = (
            None
            if self.use_alternative_attention
            else torch.nn.Linear(hidden_size, kv_width, bias=False, **factory)
        )
        self.o_proj = torch.nn.Linear(q_width, hidden_size, bias=False, **factory)
        self.q_norm = SimpleNamespace(
            eps=1e-6,
            with_scale=True,
            weight=torch.nn.Parameter(torch.ones(spec.head_dim_qk, **factory)),
        )
        self.k_norm = SimpleNamespace(
            eps=1e-6,
            with_scale=True,
            weight=torch.nn.Parameter(torch.ones(spec.head_dim_qk, **factory)),
        )
        self.v_norm = SimpleNamespace(eps=1e-6, with_scale=False)
        self.eval()


def _pinned_masking_module() -> ModuleType:
    module = ModuleType("transformers.masking_utils")
    exec(
        """
def causal_mask_function(_batch, _head, q_idx, kv_idx):
    return kv_idx <= q_idx

def sliding_window_overlay(sliding_window):
    def inner_mask(_batch, _head, q_idx, kv_idx):
        return kv_idx > q_idx - sliding_window
    return inner_mask

def packed_sequence_mask_function(packed_sequence_ids):
    def inner_mask(batch, _head, q_idx, kv_idx):
        return packed_sequence_ids[batch, q_idx] == packed_sequence_ids[batch, kv_idx]
    return inner_mask

def and_masks(*mask_functions):
    def and_mask(batch, head, q_idx, kv_idx):
        result = mask_functions[0](batch, head, q_idx, kv_idx)
        for function in mask_functions[1:]:
            result = result & function(batch, head, q_idx, kv_idx)
        return result
    return and_mask
""",
        vars(module),
    )
    module._GEMMA4_FA4_PLAIN_CAUSAL_MASK_ORIGIN = object()
    module._GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN = object()
    return module


@pytest.fixture
def pinned_mask_environment(monkeypatch):
    masking = _pinned_masking_module()
    monkeypatch.setattr(integration, "_PINNED_MASKING_UTILS_MODULE", masking)
    monkeypatch.setattr(integration, "_PINNED_GEMMA4_TEXT_CONFIG_CLASS", _PinnedConfig)
    monkeypatch.setattr(
        integration,
        "_PINNED_GEMMA4_TEXT_ATTENTION_CLASS",
        _PinnedAttention,
    )
    return _PinnedConfig(), masking


def _module(layer_idx: int, config: _PinnedConfig):
    return _PinnedAttention(layer_idx, config)


def _plan(config, masking, family: str, packed_sequence_ids, **kwargs):
    causal = masking.causal_mask_function
    if family == "local":
        causal = masking.and_masks(masking.sliding_window_overlay(1024), causal)
        local_size = 1024
        plain_origin = masking._GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN
    else:
        local_size = None
        plain_origin = masking._GEMMA4_FA4_PLAIN_CAUSAL_MASK_ORIGIN
    mask_function = masking.and_masks(
        causal,
        masking.packed_sequence_mask_function(packed_sequence_ids),
    )
    return integration._registered_gemma4_fa4_mask(
        batch_size=kwargs.pop("batch_size", packed_sequence_ids.shape[0]),
        q_length=kwargs.pop("q_length", packed_sequence_ids.shape[1]),
        kv_length=kwargs.pop("kv_length", packed_sequence_ids.shape[1]),
        mask_function=mask_function,
        config=config,
        use_vmap=kwargs.pop("use_vmap", False),
        local_size=kwargs.pop("local_size", local_size),
        past_key_values=kwargs.pop("past_key_values", None),
        _gemma4_fa4_plain_mask_origin=kwargs.pop(
            "_gemma4_fa4_plain_mask_origin",
            plain_origin,
        ),
        _gemma4_fa4_mask_recipient=kwargs.pop(
            "_gemma4_fa4_mask_recipient",
            integration._registered_gemma4_fa4_mask,
        ),
        **kwargs,
    )


def _fake_qkv(layer_idx: int, seqlen: int, *, batch: int = 1, k_length: int | None = None):
    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    k_length = seqlen if k_length is None else k_length
    q = torch.empty(
        (batch, spec.num_q_heads, seqlen, spec.head_dim_qk),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (batch, spec.num_kv_heads, k_length, spec.head_dim_qk),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.empty(
        (batch, spec.num_kv_heads, k_length, spec.head_dim_v),
        device="cuda",
        dtype=torch.bfloat16,
    )
    return q, k, v


def _fake_op(calls, family):
    def run(q, k, v, position_ids, packed_sequence_ids):
        calls.append((family, q, k, v, position_ids, packed_sequence_ids))
        output = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1], v.shape[3]),
            device=q.device,
            dtype=q.dtype,
        )
        lse = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        return output, lse

    return run


def _fake_layer_op(calls, family):
    def run(hidden, cos, sin, positions, packed, *weights):
        calls.append((family, hidden, cos, sin, positions, packed, *weights))
        output = torch.empty(
            (hidden.shape[0], hidden.shape[1], GEMMA4_31B.hidden_size),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        lse = torch.empty(
            (hidden.shape[0], GEMMA4_31B.sliding.num_q_heads, hidden.shape[1]),
            device=hidden.device,
            dtype=torch.float32,
        )
        return output, lse

    return run


def test_registered_attention_exposes_only_the_project_compile_layer_hook() -> None:
    assert (
        integration.gemma4_fa4_attention_forward._gemma4_fa4_compile_layer
        is integration.gemma4_fa4_compile_layer
    )


@pytest.mark.parametrize(
    ("layer_idx", "family", "expected_tensor_count"),
    [(0, "local", 12), (5, "global", 11)],
)
def test_whole_layer_hook_passes_only_explicit_source_tensors(
    monkeypatch,
    pinned_mask_environment,
    layer_idx,
    family,
    expected_tensor_count,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: True)
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(
        integration, "_PINNED_GEMMA4_TEXT_ATTENTION_CLASS", _WholeLayerPinnedAttention
    )
    monkeypatch.setattr(integration, "h100_local_layer_fwd", _fake_layer_op(calls, "local"))
    monkeypatch.setattr(integration, "h100_global_layer_fwd", _fake_layer_op(calls, "global"))

    spec = GEMMA4_31B.spec_for_layer(layer_idx)
    with FakeTensorMode(), torch.inference_mode():
        module = _WholeLayerPinnedAttention(layer_idx, config)
        hidden = torch.empty(
            (1, 33, GEMMA4_31B.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        cos = torch.empty((1, 33, spec.head_dim_qk), device="cuda", dtype=torch.bfloat16)
        sin = torch.empty_like(cos)
        positions = torch.arange(33, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 33), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, family, packed)
        output, weights = integration.gemma4_fa4_compile_layer(
            module,
            hidden,
            (cos, sin),
            plan,
            {},
            position_ids=positions,
            allow_flex_fallback=False,
        )

    assert weights is None
    assert output.shape == (1, 33, GEMMA4_31B.hidden_size)
    assert len(calls) == 1 and calls[0][0] == family
    explicit_tensors = calls[0][1:]
    assert len(explicit_tensors) == expected_tensor_count
    assert all(isinstance(tensor, torch.Tensor) for tensor in explicit_tensors)
    assert all(not tensor.requires_grad for tensor in explicit_tensors[:5])
    assert all(tensor.requires_grad for tensor in explicit_tensors[5:-1])
    scalar_attestation = explicit_tensors[-1]
    assert scalar_attestation.device.type == "cpu"
    assert scalar_attestation.dtype == torch.float64
    assert scalar_attestation.shape == (11,)
    assert not scalar_attestation.requires_grad
    assert calls[0][4] is positions
    assert calls[0][5].shape == positions.shape


def test_whole_layer_scalar_attestation_reflects_every_source_float() -> None:
    config = _PinnedConfig()
    module = SimpleNamespace(
        config=config,
        scaling=1.0,
        attention_dropout=0.0,
        q_norm=SimpleNamespace(eps=1e-6),
        k_norm=SimpleNamespace(eps=1e-6),
        v_norm=SimpleNamespace(eps=1e-6),
    )
    expected = torch.tensor(
        integration.WHOLE_LAYER_SCALAR_ATTESTATION_VALUES,
        dtype=torch.float64,
    )

    assert torch.equal(integration._whole_layer_scalar_attestation(module), expected)

    config.rms_norm_eps = 1e-5
    mutated = integration._whole_layer_scalar_attestation(module)
    assert mutated[5].item() == 1e-5
    assert torch.equal(mutated[:5], expected[:5])
    assert torch.equal(mutated[6:], expected[6:])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("eager", "compile-only"),
        ("cache", "cache"),
        ("shared", "shared prepared KV"),
        ("metadata", "outside its declared scope"),
        ("fallback", "fallback"),
        ("training", "exact pinned eval-mode"),
        ("v-scale", "scale-free V RMSNorm"),
        ("weight-alias", "tensor-explicit ABI"),
    ],
)
def test_whole_layer_hook_fails_closed_before_custom_op(
    monkeypatch,
    pinned_mask_environment,
    mutation,
    match,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: mutation != "eager")
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(
        integration, "_PINNED_GEMMA4_TEXT_ATTENTION_CLASS", _WholeLayerPinnedAttention
    )
    monkeypatch.setattr(integration, "h100_local_layer_fwd", _fake_layer_op(calls, "local"))

    with FakeTensorMode(), torch.inference_mode():
        module = _WholeLayerPinnedAttention(0, config)
        hidden = torch.empty((1, 2, 5376), device="cuda", dtype=torch.bfloat16)
        cos = torch.empty((1, 2, 256), device="cuda", dtype=torch.bfloat16)
        sin = torch.empty_like(cos)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, "local", packed)
        shared = {}
        past = None
        kwargs = {"position_ids": positions, "allow_flex_fallback": False}
        if mutation == "cache":
            past = object()
        elif mutation == "shared":
            shared["sliding_attention"] = (hidden, hidden)
        elif mutation == "metadata":
            kwargs["vision_block_ids"] = None
        elif mutation == "fallback":
            kwargs["allow_flex_fallback"] = True
        elif mutation == "training":
            module.train()
        elif mutation == "v-scale":
            module.v_norm.with_scale = True
        elif mutation == "weight-alias":
            module.v_proj.weight = module.k_proj.weight

        with pytest.raises(UnsupportedH100Path, match=match):
            integration.gemma4_fa4_compile_layer(
                module,
                hidden,
                (cos, sin),
                plan,
                shared,
                past_key_values=past,
                **kwargs,
            )

    assert not calls


@pytest.mark.parametrize(
    ("layer_idx", "family", "expected_path"),
    [
        (0, "local", "fa4_local_compiled_op"),
        (5, "global", "fa4_global_compiled_op"),
    ],
)
def test_fake_compiler_route_uses_family_op_and_forwards_pinned_packed_ids(
    monkeypatch,
    pinned_mask_environment,
    layer_idx,
    family,
    expected_path,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: True)
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op(calls, "local"))
    monkeypatch.setattr(integration, "h100_global_fwd", _fake_op(calls, "global"))

    with FakeTensorMode():
        q, k, v = _fake_qkv(layer_idx, 33)
        position_ids = torch.arange(33, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed_sequence_ids = torch.zeros((1, 33), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, family, packed_sequence_ids)
        result = integration.gemma4_fa4_prepared(
            _module(layer_idx, config),
            q,
            k,
            v,
            plan,
            sliding_window=GEMMA4_31B.spec_for_layer(layer_idx).sliding_window,
            position_ids=position_ids,
            allow_flex_fallback=False,
        )

    assert result.path == expected_path
    assert result.output.shape == (1, 33, 32, q.shape[-1])
    assert result.lse.shape == (1, 32, 33)
    assert result.lse.dtype == torch.float32
    assert len(calls) == 1 and calls[0][0] == family
    assert calls[0][-2] is position_ids
    assert calls[0][-1].shape == packed_sequence_ids.shape
    assert calls[0][-1].dtype == packed_sequence_ids.dtype


def test_fake_attention_entry_does_not_mutate_module_diagnostics(
    monkeypatch,
    pinned_mask_environment,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op([], "local"))
    module = _module(0, config)

    with FakeTensorMode():
        q, k, v = _fake_qkv(0, 2)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, "local", packed)
        output, weights = integration.gemma4_fa4_attention_forward(
            module,
            q,
            k,
            v,
            plan,
            sliding_window=1024,
            position_ids=positions,
        )

    assert output.shape == (1, 2, 32, 256)
    assert weights is None
    assert not hasattr(module, "_gemma4_fa4_last_path")


def test_direct_or_wrong_family_mask_plans_fail_before_custom_op(
    monkeypatch,
    pinned_mask_environment,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op(calls, "local"))

    with FakeTensorMode():
        q, k, v = _fake_qkv(0, 2)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        trusted_global = _plan(config, masking, "global", packed)
        direct = integration.Gemma4MaskPlan(
            batch_size=1,
            q_length=2,
            kv_length=2,
            q_offset=0,
            kv_offset=0,
            mask_function=masking.causal_mask_function,
            attention_mask=None,
        )
        for plan in (trusted_global, direct):
            with pytest.raises(UnsupportedH100Path, match="pinned|origin|family"):
                integration.gemma4_fa4_prepared(
                    _module(0, config),
                    q,
                    k,
                    v,
                    plan,
                    sliding_window=1024,
                    position_ids=positions,
                )

    assert not calls


@pytest.mark.parametrize(
    "mutation",
    ["bad-config", "padding", "vmap", "wrong-window", "arbitrary-callable"],
)
def test_mask_callback_mints_no_compile_origin_outside_exact_pinned_text_scope(
    pinned_mask_environment,
    mutation,
):
    config, masking = pinned_mask_environment
    packed = torch.zeros((1, 2), dtype=torch.int64)
    kwargs = {}
    if mutation == "bad-config":
        config = object()
    elif mutation == "padding":
        kwargs["attention_mask"] = torch.ones((1, 2), dtype=torch.int64)
    elif mutation == "vmap":
        kwargs["use_vmap"] = True
    elif mutation == "wrong-window":
        kwargs["local_size"] = 512

    if mutation == "arbitrary-callable":
        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=2,
            kv_length=2,
            mask_function=lambda _b, _h, q_idx, kv_idx: kv_idx <= q_idx,
            config=config,
            use_vmap=False,
        )
    else:
        plan = _plan(config, masking, "local", packed, **kwargs)

    assert not hasattr(plan, "_gemma4_fa4_compile_origin")


def test_public_mask_callback_cannot_mint_a_compile_origin(pinned_mask_environment):
    config, masking = pinned_mask_environment
    packed = torch.zeros((1, 2), dtype=torch.int64)
    causal = masking.and_masks(
        masking.sliding_window_overlay(1024),
        masking.causal_mask_function,
        masking.packed_sequence_mask_function(packed),
    )
    plan = integration.gemma4_fa4_mask(
        batch_size=1,
        q_length=2,
        kv_length=2,
        mask_function=causal,
        config=config,
        use_vmap=False,
        local_size=1024,
    )

    assert not hasattr(plan, "_gemma4_fa4_compile_origin")


def test_compiler_mode_rejects_non_null_cache_before_plan_construction(
    monkeypatch,
    pinned_mask_environment,
):
    config, masking = pinned_mask_environment
    packed = torch.zeros((1, 2), dtype=torch.int64)
    cache = object()
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: True)

    def unexpected_plan_construction(*_args, **_kwargs):
        raise AssertionError("cache rejection reached public plan construction")

    monkeypatch.setattr(integration, "gemma4_fa4_mask", unexpected_plan_construction)
    with pytest.raises(UnsupportedH100Path, match="EXP-0018.*cache.*before Cache.update"):
        _plan(config, masking, "local", packed, past_key_values=cache)


@pytest.mark.parametrize(
    "provenance",
    ["public", "direct-registered", "wrong-family", "wrong-recipient"],
)
def test_compiler_mode_unproven_mask_origins_cannot_mint(
    monkeypatch,
    pinned_mask_environment,
    provenance,
):
    config, masking = pinned_mask_environment
    packed = torch.zeros((1, 2), dtype=torch.int64)
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: True)

    if provenance == "public":
        mask_function = masking.and_masks(
            masking.sliding_window_overlay(1024),
            masking.causal_mask_function,
            masking.packed_sequence_mask_function(packed),
        )
        plan = integration.gemma4_fa4_mask(
            batch_size=1,
            q_length=2,
            kv_length=2,
            mask_function=mask_function,
            config=config,
            use_vmap=False,
            local_size=1024,
            past_key_values=None,
            _upstream_plain_origin=masking._GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN,
            _upstream_mask_recipient=integration._registered_gemma4_fa4_mask,
        )
    elif provenance == "direct-registered":
        plan = _plan(
            config,
            masking,
            "local",
            packed,
            _gemma4_fa4_plain_mask_origin=None,
            _gemma4_fa4_mask_recipient=None,
        )
    elif provenance == "wrong-family":
        plan = _plan(
            config,
            masking,
            "local",
            packed,
            _gemma4_fa4_plain_mask_origin=masking._GEMMA4_FA4_PLAIN_CAUSAL_MASK_ORIGIN,
        )
    else:
        plan = _plan(
            config,
            masking,
            "local",
            packed,
            _gemma4_fa4_mask_recipient=lambda **_kwargs: None,
        )

    assert not hasattr(plan, "_gemma4_fa4_compile_origin")


def test_compiler_mode_registry_forwarder_cannot_mint_with_all_private_kwargs(
    monkeypatch,
    pinned_mask_environment,
):
    config, masking = pinned_mask_environment
    packed = torch.zeros((1, 2), dtype=torch.int64)
    mask_function = masking.and_masks(
        masking.sliding_window_overlay(1024),
        masking.causal_mask_function,
        masking.packed_sequence_mask_function(packed),
    )
    forwarded = []
    monkeypatch.setattr(integration, "_TORCH_IS_COMPILING", lambda: True)

    def forwarding_registry_callback(**kwargs):
        forwarded.append(kwargs.copy())
        return integration._registered_gemma4_fa4_mask(**kwargs)

    plan = forwarding_registry_callback(
        batch_size=1,
        q_length=2,
        kv_length=2,
        mask_function=mask_function,
        config=config,
        use_vmap=False,
        local_size=1024,
        past_key_values=None,
        _gemma4_fa4_plain_mask_origin=masking._GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN,
        _gemma4_fa4_mask_recipient=forwarding_registry_callback,
    )

    assert forwarded[0]["past_key_values"] is None
    assert (
        forwarded[0]["_gemma4_fa4_plain_mask_origin"]
        is masking._GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN
    )
    assert forwarded[0]["_gemma4_fa4_mask_recipient"] is forwarding_registry_callback
    assert not hasattr(plan, "_gemma4_fa4_compile_origin")


def test_compiler_route_requires_exact_layer_type_index_mode_and_mask_config(
    monkeypatch,
    pinned_mask_environment,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op(calls, "local"))

    with FakeTensorMode():
        q, k, v = _fake_qkv(0, 2)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, "local", packed)
        valid = _module(0, config)
        wrong_type = SimpleNamespace(**vars(valid))
        wrong_index = _module(1, config)
        wrong_config = _module(0, _PinnedConfig())
        training = _module(0, config)
        training.training = True

        for module in (wrong_type, wrong_index, wrong_config, training):
            with pytest.raises(UnsupportedH100Path, match="exact pinned eval-mode"):
                integration.gemma4_fa4_prepared(
                    module,
                    q,
                    k,
                    v,
                    plan,
                    sliding_window=1024,
                    position_ids=positions,
                )

        mutations = {
            "is_causal": False,
            "attention_bias": True,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"sliding_attention": {"rope_type": "default"}},
        }
        for name, value in mutations.items():
            setattr(config, name, value)
            try:
                with pytest.raises(UnsupportedH100Path, match="exact pinned eval-mode"):
                    integration.gemma4_fa4_prepared(
                        valid,
                        q,
                        k,
                        v,
                        plan,
                        sliding_window=1024,
                        position_ids=positions,
                    )
            finally:
                delattr(config, name)

    assert not calls


@pytest.mark.parametrize(
    "unsupported",
    ["batch", "unequal", "q-offset", "kv-offset", "empty", "too-long"],
)
def test_compiler_route_rejects_shape_length_and_offset_scope(
    monkeypatch,
    pinned_mask_environment,
    unsupported,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op(calls, "local"))

    with FakeTensorMode():
        batch = 2 if unsupported == "batch" else 1
        q_length = 0 if unsupported == "empty" else 1025 if unsupported == "too-long" else 2
        k_length = 3 if unsupported == "unequal" else q_length
        q, k, v = _fake_qkv(0, q_length, batch=batch, k_length=k_length)
        positions = torch.arange(q_length, device="cuda", dtype=torch.int64).unsqueeze(0)
        if batch == 2:
            positions = positions.expand(2, -1)
        packed = torch.zeros((batch, q_length), device="cuda", dtype=torch.int64)
        plan = _plan(
            config,
            masking,
            "local",
            packed,
            q_length=q_length,
            kv_length=k_length,
            q_offset=1 if unsupported == "q-offset" else 0,
            kv_offset=1 if unsupported == "kv-offset" else 0,
        )
        with pytest.raises(UnsupportedH100Path, match="EXP-0017|B1|length|offset"):
            integration.gemma4_fa4_prepared(
                _module(0, config),
                q,
                k,
                v,
                plan,
                sliding_window=1024,
                position_ids=positions,
            )

    assert not calls


@pytest.mark.parametrize(
    "unsupported",
    ["vision", "document", "cu", "max", "cache", "positions", "requires-grad"],
)
def test_fake_compiler_scope_rejects_unaccepted_metadata_and_gradients(
    monkeypatch,
    pinned_mask_environment,
    unsupported,
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    calls = []
    monkeypatch.setattr(integration, "CUSTOM_OPS_AVAILABLE", True)
    monkeypatch.setattr(integration, "h100_local_fwd", _fake_op(calls, "local"))

    with FakeTensorMode():
        q, k, v = _fake_qkv(0, 2)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, "local", packed)
        kwargs = {"position_ids": positions}
        if unsupported == "vision":
            kwargs["vision_block_ids"] = torch.full((1, 2), -1, device="cuda", dtype=torch.int64)
        elif unsupported == "document":
            kwargs["document_ids"] = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        elif unsupported == "cu":
            kwargs["cu_seq_lens_q"] = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        elif unsupported == "max":
            kwargs["max_length_k"] = 2
        elif unsupported == "cache":
            kwargs["cache_position"] = positions
        elif unsupported == "positions":
            kwargs["position_ids"] = None
        else:
            q.requires_grad_(True)

        with pytest.raises(UnsupportedH100Path, match="EXP-0017|compile|position|grad|cache"):
            integration.gemma4_fa4_prepared(
                _module(0, config),
                q,
                k,
                v,
                plan,
                sliding_window=1024,
                **kwargs,
            )

    assert not calls


def test_old_torch_valid_fake_scope_fails_closed_when_custom_ops_are_unavailable(
    pinned_mask_environment,
):
    if integration.CUSTOM_OPS_AVAILABLE:
        pytest.skip("installed PyTorch provides the EXP-0017 custom-op APIs")

    from torch._subclasses.fake_tensor import FakeTensorMode

    config, masking = pinned_mask_environment
    with FakeTensorMode():
        q, k, v = _fake_qkv(0, 2)
        positions = torch.arange(2, device="cuda", dtype=torch.int64).unsqueeze(0)
        packed = torch.zeros((1, 2), device="cuda", dtype=torch.int64)
        plan = _plan(config, masking, "local", packed)
        with pytest.raises(UnsupportedH100Path, match="custom_op|register_fake|PyTorch"):
            integration.gemma4_fa4_prepared(
                _module(0, config),
                q,
                k,
                v,
                plan,
                sliding_window=1024,
                position_ids=positions,
            )
