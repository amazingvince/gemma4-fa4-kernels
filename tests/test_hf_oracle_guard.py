import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_optional_oracle_skips_incompatible_transformers(tmp_path):
    package = tmp_path / "transformers"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.0"\n', encoding="utf-8")
    sentinel = tmp_path / "test_sentinel.py"
    sentinel.write_text("def test_sentinel():\n    pass\n", encoding="utf-8")

    env = os.environ.copy()
    pythonpath = [str(tmp_path), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(ROOT / "tests" / "test_hf_oracle_optional.py"),
            str(sentinel),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "1 passed" in output
    assert "1 skipped" in output
