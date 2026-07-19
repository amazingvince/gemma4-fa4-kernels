#!/usr/bin/env python3
"""Validate the locked Gemma 4 contract offline, optionally against live sources."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_fa4.model_spec import (  # noqa: E402
    GEMMA4_31B,
    GLOBAL_ATTENTION,
    SLIDING_ATTENTION,
)


def _load() -> dict:
    return json.loads((ROOT / "configs/model/gemma4-31b.lock.json").read_text())


def _assert_fields(config: dict) -> None:
    text = config["text_config"]
    expected = {
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "max_position_embeddings": 262_144,
        "attention_dropout": 0.0,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "use_bidirectional_attention": "vision",
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise AssertionError(f"{key}: expected {value!r}, got {text.get(key)!r}")
    expected_layers = list(GEMMA4_31B.layer_types())
    if text["layer_types"] != expected_layers:
        raise AssertionError("layer_types do not match the exact five-local/one-full pattern")
    expected_rope = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    }
    if text["rope_parameters"] != expected_rope:
        raise AssertionError("RoPE parameters do not match the locked checkpoint")
    derived = config["derived_attention_contract"]
    expected_derived = {
        "softmax_scale": 1.0,
        "local_window_size_left_for_fa": 1023,
        "sliding_rotary_dim": 256,
        "global_rotary_dim": 128,
        "global_projection_source_shared": True,
        "global_fmha_k_and_v_distinct_after_preparation": True,
        "cross_layer_kv_sharing_enabled": False,
    }
    for key, value in expected_derived.items():
        if derived.get(key) != value:
            raise AssertionError(f"derived contract mismatch for {key}")


def _assert_code() -> None:
    assert GEMMA4_31B.num_hidden_layers == 60
    assert GEMMA4_31B.first_kv_shared_layer_idx == 60
    assert SLIDING_ATTENTION.num_q_heads == 32
    assert SLIDING_ATTENTION.num_kv_heads == 16
    assert SLIDING_ATTENTION.fa_window_size_left == 1023
    assert GLOBAL_ATTENTION.num_q_heads == 32
    assert GLOBAL_ATTENTION.num_kv_heads == 4
    assert GLOBAL_ATTENTION.qhead_per_kvhead == 8
    assert SLIDING_ATTENTION.softmax_scale == GLOBAL_ATTENTION.softmax_scale == 1.0
    assert SLIDING_ATTENTION.is_causal and GLOBAL_ATTENTION.is_causal
    assert SLIDING_ATTENTION.vision_bidirectional_within_block
    assert not GLOBAL_ATTENTION.vision_bidirectional_within_block
    assert SLIDING_ATTENTION.rotary_dim == 256
    assert GLOBAL_ATTENTION.rotary_dim == 128


def _online(lock: dict) -> None:
    url = f"https://huggingface.co/{lock['model_id']}/raw/{lock['model_revision']}/config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        remote = json.load(response)
    remote_text = remote["text_config"]
    locked_text = lock["text_config"]
    for key, expected in locked_text.items():
        if key not in remote_text:
            raise AssertionError(f"online config is missing locked key {key}")
        if remote_text[key] != expected:
            raise AssertionError(
                f"online config mismatch for {key}: expected {expected!r}, got {remote_text[key]!r}"
            )


def _transformers(lock: dict) -> None:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    text = dict(lock["text_config"])
    text["hidden_size"] = 64
    text["intermediate_size"] = 128
    cfg = Gemma4TextConfig(**text)
    local = Gemma4TextAttention(cfg, layer_idx=0)
    full = Gemma4TextAttention(cfg, layer_idx=5)
    assert local.scaling == full.scaling == 1.0
    assert local.head_dim == 256 and full.head_dim == 512
    assert local.num_key_value_groups == 2 and full.num_key_value_groups == 8
    assert local.v_proj is not None
    assert full.v_proj is None
    assert not local.is_kv_shared_layer and not full.is_kv_shared_layer
    assert local.q_norm.with_scale and local.k_norm.with_scale
    assert not local.v_norm.with_scale
    assert full.q_norm.with_scale and full.k_norm.with_scale
    assert not full.v_norm.with_scale
    assert full.k_norm is not full.v_norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--transformers", action="store_true")
    args = parser.parse_args()
    lock = _load()
    _assert_fields(lock)
    _assert_code()
    if args.online:
        _online(lock)
    if args.transformers:
        _transformers(lock)
    print(
        json.dumps(
            {
                "status": "ok",
                "model": lock["model_id"],
                "model_revision": lock["model_revision"],
                "transformers_revision": lock["transformers_revision"],
                "online": args.online,
                "transformers_oracle": args.transformers,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
