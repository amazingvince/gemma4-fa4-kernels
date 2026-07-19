import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_env", ROOT / "scripts/check_env.py")
assert SPEC and SPEC.loader
CHECK_ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_ENV)


def test_nvcc_release_parser():
    text = "Cuda compilation tools, release 13.3, V13.3.48"
    assert CHECK_ENV.nvcc_release(text) == "13.3"
    assert CHECK_ENV.nvcc_release(None) is None


def test_target_policies_select_hopper_and_blackwell_extras():
    h100 = CHECK_ENV.load_policy("h100")
    b300 = CHECK_ENV.load_policy("b300")
    assert h100["CUDA_TOOLKIT_VERSION"].startswith("12.")
    assert h100["PYTORCH_CUDA_WHEEL"] == "cu128"
    assert h100["FA4_EXTRAS"] == "dev"
    assert h100["QUACK_KERNELS_VERSION"] == "0.5.3"
    assert h100["CUTE_DSL_ARCH"] == "sm_90a"
    assert h100["FLASH_ATTENTION_ARCH"] == "sm_90"
    assert h100["FLASH_ATTN_PATCH_PATH"].endswith("sm90-d512-v256-forward.patch")
    assert len(h100["FLASH_ATTN_PATCH_SHA256"]) == 64
    assert b300["CUDA_TOOLKIT_VERSION"].startswith("13.")
    assert b300["FA4_EXTRAS"] == "dev,cu13"
    assert b300["QUACK_KERNELS_VERSION"] == "0.5.3"
    assert b300["CUTE_DSL_ARCH"] == "sm_103a"


def test_version_tuple_parser():
    assert CHECK_ENV.version_tuple("610.43.02") == (610, 43, 2)


def test_torch_version_parts_do_not_hide_prereleases():
    assert CHECK_ENV.torch_version_parts("2.8.0+cu128") == ("2.8.0", "cu128")
    assert CHECK_ENV.torch_version_parts("2.8.0a0+cu128") == ("2.8.0a0", "cu128")
    assert CHECK_ENV.torch_version_parts("2.8.0") == ("2.8.0", None)


def test_imported_module_path_must_resolve_inside_pinned_checkout():
    checkout = ROOT / ".upstream/flash-attention"
    module = types.SimpleNamespace(__file__=checkout / "flash_attn/cute/__init__.py")
    assert CHECK_ENV.path_is_within(CHECK_ENV.imported_module_path(module), checkout)
    outside = types.SimpleNamespace(__file__=ROOT / ".venv/lib/flash_attn/cute/__init__.py")
    assert not CHECK_ENV.path_is_within(CHECK_ENV.imported_module_path(outside), checkout)
    assert CHECK_ENV.imported_module_path(object()) is None


def test_git_dirty_detects_modified_upstream_checkout(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("pinned\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "pinned"], cwd=repo, check=True)
    assert CHECK_ENV.git_dirty(repo) is False
    tracked.write_text("modified\n")
    assert CHECK_ENV.git_dirty(repo) is True
    assert "-pinned" in CHECK_ENV.git_diff(repo)
    assert "+modified" in CHECK_ENV.git_diff(repo)


def configure_strict_environment(monkeypatch, args):
    profile = "h100" if "h100" in args else "b300"
    policy = {
        **CHECK_ENV.load_policy(profile),
        "PYTHON_VERSION": ".".join(sys.version.split(".")[:2]),
    }
    monkeypatch.setattr(CHECK_ENV, "load_policy", lambda _profile: policy)
    torch = types.ModuleType("torch")
    torch.__version__ = f"{policy['PYTORCH_VERSION']}+{policy['PYTORCH_CUDA_WHEEL']}"
    torch.version = types.SimpleNamespace(cuda=policy["PYTORCH_CUDA_RUNTIME"])
    capability = (9, 0) if profile == "h100" else (10, 3)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _: capability,
        get_device_name=lambda _: "test GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    def package_version(name):
        if name in {"nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base"}:
            return policy["CUTLASS_DSL_VERSION"]
        if name == "quack-kernels":
            return policy["QUACK_KERNELS_VERSION"]
        if name == "nvidia-cutlass-dsl-libs-cu13" and profile == "b300":
            return policy["CUTLASS_DSL_VERSION"]
        raise CHECK_ENV.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        CHECK_ENV.importlib.metadata,
        "version",
        package_version,
    )

    def import_module(name):
        if name == "flash_attn.cute":
            path = ROOT / ".upstream/flash-attention/flash_attn/cute/__init__.py"
        else:
            path = ROOT / ".upstream/transformers/src/transformers/models/gemma4/modeling_gemma4.py"
        return types.SimpleNamespace(__file__=str(path))

    monkeypatch.setattr(CHECK_ENV.importlib, "import_module", import_module)
    monkeypatch.setattr(CHECK_ENV.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        CHECK_ENV,
        "command",
        lambda command, cwd=None: (
            f"Cuda compilation tools, release {policy['CUDA_TOOLKIT_VERSION']}, V0"
            if command == ["nvcc", "--version"]
            else f"test GPU, {policy['CUDA_DRIVER_MIN_FULL']}, 9.0, 80 MiB"
        ),
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_head",
        lambda path: (
            policy["FLASH_ATTN_REV"]
            if path.name == "flash-attention"
            else policy["TRANSFORMERS_REV"]
        ),
    )
    monkeypatch.setattr(CHECK_ENV.sys, "argv", ["check_env.py", "--strict", *args])
    monkeypatch.setenv("CUTE_DSL_ARCH", policy["CUTE_DSL_ARCH"])
    monkeypatch.setenv("FLASH_ATTENTION_ARCH", policy["FLASH_ATTENTION_ARCH"])


def test_strict_h100_profile_accepts_cuda12_without_cu13_dsl(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, ["--profile", "h100", "--expect-arch", "sm_90"])
    patch_path = ROOT / CHECK_ENV.load_policy("h100")["FLASH_ATTN_PATCH_PATH"]
    monkeypatch.setattr(
        CHECK_ENV,
        "git_dirty",
        lambda path: path.name == "flash-attention",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_diff",
        lambda path: patch_path.read_text() if path.name == "flash-attention" else "",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_status",
        lambda path: "M flash_attn/cute/interface.py" if path.name == "flash-attention" else "",
    )

    assert CHECK_ENV.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["profile"] == "h100"
    assert report["nvidia_cutlass_dsl_libs_base"] == "4.6.0.dev0"
    assert report["nvidia_cutlass_dsl_libs_cu13"] is None
    assert report["flash_attention_patch"]["applied_exactly"] is True
    assert report["errors"] == []


def test_strict_h100_rejects_prerelease_torch_with_matching_prefix(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, ["--profile", "h100", "--expect-arch", "sm_90"])
    sys.modules["torch"].__version__ = "2.8.0a0+cu128"
    patch_path = ROOT / CHECK_ENV.load_policy("h100")["FLASH_ATTN_PATCH_PATH"]
    monkeypatch.setattr(
        CHECK_ENV,
        "git_dirty",
        lambda path: path.name == "flash-attention",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_diff",
        lambda path: patch_path.read_text() if path.name == "flash-attention" else "",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_status",
        lambda path: "M flash_attn/cute/interface.py" if path.name == "flash-attention" else "",
    )
    assert CHECK_ENV.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert "torch 2.8.0a0+cu128 != policy 2.8.0" in report["errors"]


def test_strict_h100_rejects_runtime_import_outside_pinned_checkout(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, ["--profile", "h100", "--expect-arch", "sm_90"])
    patch_path = ROOT / CHECK_ENV.load_policy("h100")["FLASH_ATTN_PATCH_PATH"]
    monkeypatch.setattr(
        CHECK_ENV,
        "git_dirty",
        lambda path: path.name == "flash-attention",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_diff",
        lambda path: patch_path.read_text() if path.name == "flash-attention" else "",
    )
    monkeypatch.setattr(
        CHECK_ENV,
        "git_status",
        lambda path: "M flash_attn/cute/interface.py" if path.name == "flash-attention" else "",
    )

    def wrong_flash_import(name):
        if name == "flash_attn.cute":
            return types.SimpleNamespace(__file__="/site-packages/flash_attn/cute/__init__.py")
        path = ROOT / ".upstream/transformers/src/transformers/models/gemma4/modeling_gemma4.py"
        return types.SimpleNamespace(__file__=str(path))

    monkeypatch.setattr(CHECK_ENV.importlib, "import_module", wrong_flash_import)
    assert CHECK_ENV.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert (
        "flash_attn.cute was not imported from the pinned FlashAttention checkout"
        in report["errors"]
    )


def test_strict_mode_rejects_unknown_flash_attention_cleanliness(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, [])
    monkeypatch.setattr(CHECK_ENV, "git_dirty", lambda _: None)

    assert CHECK_ENV.main() == 1

    report = json.loads(capsys.readouterr().out)
    assert report["upstream_dirty"]["flash_attention"] is None
    assert report["errors"] == ["FlashAttention checkout cleanliness could not be determined"]


def test_strict_mode_rejects_unknown_required_transformers_cleanliness(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, ["--require-transformers"])
    monkeypatch.setattr(
        CHECK_ENV,
        "git_dirty",
        lambda path: None if path.name == "transformers" else False,
    )

    assert CHECK_ENV.main() == 1

    report = json.loads(capsys.readouterr().out)
    assert report["upstream_dirty"]["transformers"] is None
    assert report["errors"] == ["Transformers checkout cleanliness could not be determined"]
