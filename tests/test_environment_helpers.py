import importlib.util
import subprocess
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
