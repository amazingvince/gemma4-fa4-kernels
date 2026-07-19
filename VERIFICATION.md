# Repository verification report

**Assembly date:** 2026-07-19

## Completed in the current local environment

```text
python -m compileall -q src tests scripts benchmarks
  pass

python -m pytest -q -p no:cacheprovider
  64 passed, 33 skipped
  skipped: 32 H100 execution gates and the unavailable pinned Transformers oracle

python -m ruff check --no-cache .
  pass

python -m ruff format --check src tests scripts benchmarks
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

## Completed on the H100 target

```text
strict environment check, including exact FA4 patch stack and profilers
  pass: H100 80GB, CC 9.0, driver 580.126.09, CUDA 12.8,
        PyTorch 2.8.0+cu128, CuTe DSL 4.6.0.dev0
        quack-kernels 0.5.3; FA4/Transformers imports bound to pinned checkouts

pytest -q
  96 passed, 2 skipped, 1 xfailed

local d256 fixed-length text forward
  pass: O/LSE, W1024 boundaries, GQA 1/2/4/8, stream repeat

composed global d512 fixed-length text forward
  pass: O/LSE through S1024, exact slab LSE, stream repeat
  pass: filtered memcheck, synccheck, racecheck
  SASS: HGMMA BF16/F32 and TMA engaged; 168 registers, no local/stack spill

local d256 backward
  reject: fake compile and execution succeeded, but dQ/dK exceeded the
          frozen numerical envelope at the first S128 real comparison
```

See `docs/status.md` and EXP-0001 through EXP-0003 for exact commands,
tolerances, cache keys, artifact hashes, and the stop condition.

## Not completed in the local environment

- The local Python environment does not contain the pinned Transformers
  checkout, so its optional executable oracle is skipped locally. The H100
  bootstrap installs that exact checkout and the remote oracle runs there.
- `shellcheck` is not installed locally. ShellCheck 0.9.0 is installed on the
  H100 and the complete bundle shell check passes there.
- Nsight Compute launch metrics remain unavailable on the pod because the
  driver denies performance-counter access (`ERR_NVGPUCTRPERM`). Exact dynamic
  shared memory is therefore still open; static/model estimates are labeled.
- B300 CUDA 13.3 / PyTorch 2.13.0 cu132 remains a separate, entirely unrun
  target-host gate.
- Global backward, multimodal kernels, varlen, long production lengths, and
  every benchmark remain unrun after the ordered local-backward failure.

## Remaining remote evidence

```bash
bash scripts/remote/probe.sh b300
bash scripts/remote/bootstrap.sh b300
bash scripts/remote/check.sh b300
```

Do not start B300 in the H100-only scope. The next H100 kernel session must
open a new experiment around the local d256 backward accumulation/configuration
failure; it must not skip ahead to global backward or benchmarks.
