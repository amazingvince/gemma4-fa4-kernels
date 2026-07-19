import json
from pathlib import Path

from gemma4_fa4.model_spec import GEMMA4_31B, GLOBAL_ATTENTION, SLIDING_ATTENTION

ROOT = Path(__file__).resolve().parents[1]


def test_locked_model_shape():
    lock = json.loads((ROOT / "configs/model/gemma4-31b.lock.json").read_text())
    text = lock["text_config"]
    assert text["hidden_size"] == 5376
    assert text["intermediate_size"] == 21504
    assert text["num_hidden_layers"] == 60
    assert text["num_attention_heads"] == 32
    assert text["num_key_value_heads"] == 16
    assert text["num_global_key_value_heads"] == 4
    assert text["head_dim"] == 256
    assert text["global_head_dim"] == 512
    assert text["sliding_window"] == 1024
    assert text["max_position_embeddings"] == 262_144
    assert text["attention_k_eq_v"] is True
    assert text["num_kv_shared_layers"] == 0
    assert text["hidden_size_per_layer_input"] == 0
    assert text["rms_norm_eps"] == 1e-6
    assert text["final_logit_softcapping"] == 30.0


def test_layer_pattern_and_no_cross_layer_kv_sharing():
    layers = GEMMA4_31B.layer_types()
    assert len(layers) == 60
    assert layers.count("sliding_attention") == 50
    assert layers.count("full_attention") == 10
    assert layers[-1] == "full_attention"
    assert GEMMA4_31B.first_kv_shared_layer_idx == 60


def test_attention_shapes_and_scale():
    assert SLIDING_ATTENTION.qhead_per_kvhead == 2
    assert GLOBAL_ATTENTION.qhead_per_kvhead == 8
    assert SLIDING_ATTENTION.softmax_scale == 1.0
    assert GLOBAL_ATTENTION.softmax_scale == 1.0
    assert SLIDING_ATTENTION.is_causal and GLOBAL_ATTENTION.is_causal
    assert SLIDING_ATTENTION.vision_bidirectional_within_block
    assert not GLOBAL_ATTENTION.vision_bidirectional_within_block
    assert SLIDING_ATTENTION.fa_window_size_left == 1023
    assert SLIDING_ATTENTION.rotary_dim == 256
    assert GLOBAL_ATTENTION.rotary_dim == 128


def test_global_contract_is_shared_source_not_shared_operand():
    assert GLOBAL_ATTENTION.projection_source_shared_between_k_and_v
    assert GLOBAL_ATTENTION.head_dim_qk == GLOBAL_ATTENTION.head_dim_v == 512
