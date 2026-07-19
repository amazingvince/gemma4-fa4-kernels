import importlib.util
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
