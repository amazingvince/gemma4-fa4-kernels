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


def test_version_tuple_parser():
    assert CHECK_ENV.version_tuple("610.43.02") == (610, 43, 2)


def test_git_dirty_detects_modified_upstream_checkout(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("pinned\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "pinned"], cwd=repo, check=True)
    assert CHECK_ENV.git_dirty(repo) is False
    tracked.write_text("modified\n")
    assert CHECK_ENV.git_dirty(repo) is True


def configure_strict_environment(monkeypatch, args):
    policy = {
        **CHECK_ENV.load_policy(),
        "PYTHON_VERSION": ".".join(sys.version.split(".")[:2]),
    }
    monkeypatch.setattr(CHECK_ENV, "load_policy", lambda: policy)
    torch = types.ModuleType("torch")
    torch.__version__ = policy["PYTORCH_VERSION"]
    torch.version = types.SimpleNamespace(cuda=policy["PYTORCH_CUDA_RUNTIME"])
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _: (9, 0),
        get_device_name=lambda _: "test GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(CHECK_ENV.importlib.metadata, "version", lambda _: policy["CUTLASS_DSL_VERSION"])
    monkeypatch.setattr(CHECK_ENV.importlib, "import_module", lambda _: object())
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


def test_strict_mode_rejects_unknown_flash_attention_cleanliness(monkeypatch, capsys):
    configure_strict_environment(monkeypatch, [])
    monkeypatch.setattr(CHECK_ENV, "git_dirty", lambda _: None)

    assert CHECK_ENV.main() == 1

    report = json.loads(capsys.readouterr().out)
    assert report["upstream_dirty"]["flash_attention"] is None
    assert report["errors"] == [
        "FlashAttention checkout cleanliness could not be determined"
    ]


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
    assert report["errors"] == [
        "Transformers checkout cleanliness could not be determined"
    ]
