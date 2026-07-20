import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_lock_has_full_shas():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for section in ("flash_attention", "transformers", "gemma4_31b"):
        assert len(lock[section]["revision"]) == 40


def test_flash_attention_patch_stack_is_hash_locked():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    patches = lock["flash_attention"]["patches"]
    assert len(patches) == 1
    patch_path = ROOT / patches[0]["path"]
    assert patch_path.is_file()
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == patches[0]["sha256"]


def test_flash_attention_patch_contains_accepted_exp0037_route():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    patch_path = ROOT / lock["flash_attention"]["patches"][0]["path"]
    patch_text = patch_path.read_text()
    assert (
        '+        os.environ.get("FLASH_ATTENTION_GEMMA4_EXPERIMENT_DKV_D256_STREAM", "1")'
        in patch_text
    )
    assert '"dkv_d256_stream_v1"' in patch_text
    assert "stream_do_d256_dkv=stream_do_d256_dkv" in patch_text
    assert "mma_one_m_block_dkv_d256_stream" in patch_text


def test_transformers_patch_stack_is_hash_locked():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    patches = lock["transformers"]["patches"]
    assert len(patches) == 1
    patch_path = ROOT / patches[0]["path"]
    assert patch_path.is_file()
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == patches[0]["sha256"]
    patch_text = patch_path.read_text()
    assert '+        block_sequence_ids = kwargs.get("vision_block_ids")' in patch_text
    assert '+            kwargs["vision_block_ids"] = block_sequence_ids' in patch_text
    assert "diff --git a/src/transformers/masking_utils.py" in patch_text
    assert "+_GEMMA4_FA4_PLAIN_CAUSAL_MASK_ORIGIN = object()" in patch_text
    assert "+_GEMMA4_FA4_PLAIN_SLIDING_MASK_ORIGIN = object()" in patch_text
    assert '+            "past_key_values": past_key_values,' in patch_text
    assert '+            "_gemma4_fa4_mask_recipient": mask_interface,' in patch_text
    assert "+        if torch.compiler.is_compiling():" in patch_text
    assert 'getattr(compile_attention_interface, "_gemma4_fa4_compile_layer", None)' in patch_text
    assert "+                    past_key_values=past_key_values," in patch_text
    assert "@@ -2687,8 +2709,16" in patch_text


def test_environment_policies_hash_lock_transformers_patch():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    expected = lock["transformers"]["patches"][0]
    h100 = dict(
        line.split("=", 1)
        for line in (ROOT / "configs/env/h100-compatible.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert h100["TRANSFORMERS_PATCH_PATH"] == expected["path"]
    assert h100["TRANSFORMERS_PATCH_SHA256"] == expected["sha256"]

    b300 = (ROOT / "configs/env/latest-compatible.env").read_text()
    assert "TRANSFORMERS_PATCH_PATH" not in b300
    assert "TRANSFORMERS_PATCH_SHA256" not in b300


def test_h100_environment_policy_hash_locks_flash_attention_patch():
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    expected = lock["flash_attention"]["patches"][0]
    h100 = dict(
        line.split("=", 1)
        for line in (ROOT / "configs/env/h100-compatible.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert h100["FLASH_ATTN_PATCH_PATH"] == expected["path"]
    assert h100["FLASH_ATTN_PATCH_SHA256"] == expected["sha256"]


def test_latest_environment_policy_is_explicit():
    text = (ROOT / "configs/env/latest-compatible.env").read_text()
    assert "CUDA_TOOLKIT_VERSION=13.3" in text
    assert "PYTORCH_VERSION=2.13.0" in text
    assert "PYTORCH_CUDA_WHEEL=cu132" in text
    assert "PYTORCH_CUDA_RUNTIME=13.2" in text
    assert "CUTLASS_DSL_VERSION=4.6.0.dev0" in text
    assert "FA4_EXTRAS=dev,cu13" in text


def test_h100_environment_policy_is_cuda12_and_uses_plain_dev_extra():
    text = (ROOT / "configs/env/h100-compatible.env").read_text()
    assert "CUDA_TOOLKIT_VERSION=12.8" in text
    assert "PYTORCH_VERSION=2.8.0" in text
    assert "PYTORCH_CUDA_WHEEL=cu128" in text
    assert "PYTORCH_CUDA_RUNTIME=12.8" in text
    assert "CUTLASS_DSL_VERSION=4.6.0.dev0" in text
    assert "FA4_EXTRAS=dev\n" in text
