# Assembly verification report

**Assembly date:** 2026-07-19

## Completed in the assembly environment

```text
python scripts/verify_model_contract.py
  status: pass

pytest -q
  28 passed, 1 skipped
  skipped: optional pinned Transformers oracle; Transformers was not installed

ruff check .
  pass

ruff format --check src tests scripts benchmarks
  pass

bash -n scripts/*.sh scripts/remote/*.sh
  pass

writing-cute-dsl-kernels static validator
  pass: 20 Markdown files, approximately 22,014 words

sha256sum -c skills/writing-cute-dsl-kernels/SHA256SUMS.txt
  pass for every preserved skill file

JSON parse of every repository *.json file
  pass

mocked SSH/rsync remote transport smoke test
  pass: leading ~/ remained expandable on the remote shell and the run command
  preserved argument quoting
```

## Not completed in the assembly environment

- The exact Transformers checkout could not be installed because the assembly
  container had no DNS access to GitHub. Its optional executable oracle is
  therefore present but skipped here. The implementation was audited against
  the pinned source revision, and remote bootstrap installs/runs the oracle.
- `shellcheck` was not installed. All shell files passed `bash -n`; remote hosts
  should install ShellCheck or run it in CI if available.
- No H100 or B300 was available. No FA4 JIT compile, CUDA execution, sanitizer,
  PTX/SASS, resource, or performance evidence was produced.
- The CUDA 13.3 / PyTorch 2.13.0 cu132 / CuTe DSL 4.6.0.dev0 compatibility set
  remains a target-host validation item.

## Required first remote evidence

```bash
bash scripts/remote/probe.sh h100
bash scripts/remote/bootstrap.sh h100
bash scripts/remote/check.sh h100

bash scripts/remote/probe.sh b300
bash scripts/remote/bootstrap.sh b300
bash scripts/remote/check.sh b300
```

Then run the pinned online/modeling oracle and record the exact pass/xfail
matrix before a kernel edit.
