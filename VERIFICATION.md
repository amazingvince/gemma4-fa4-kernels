# Repository verification report

**Assembly date:** 2026-07-19

## Completed in the current local environment

```text
python -m compileall -q src tests scripts benchmarks
  pass

python -m pytest -q -p no:cacheprovider
  74 passed, 33 skipped
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
  106 passed, 2 skipped, 1 xfailed on the final EXP-0006 tree

local d256 fixed-length text forward
  pass: O/LSE, W1024 boundaries, GQA 1/2/4/8, stream repeat

composed global d512 fixed-length text forward
  pass: O/LSE through S1024, exact slab LSE, stream repeat
  pass: filtered memcheck, synccheck, racecheck
  SASS: HGMMA BF16/F32 and TMA engaged; 168 registers, no local/stack spill

local d256 backward
  pass: EXP-0004 upstream-relative BF16 gate at
        S=1,63,64,65,127,128,129,1023,1024,1025
  pass: separate finite BF16 dQ/dK/dV, exact same-input repeats,
        nondefault stream, and memcheck/synccheck/racecheck at S128/S129
  SASS: 44 HGMMA BF16/F32 and 24 UTMALDG.4D instructions; 168 registers,
        no local/stack spill
  preserved reject: EXP-0003 fixed elementwise dQ/dK envelope at S128

global d512 backward
  pass: EXP-0006 split dKV-only and D256 dQ-only main launches at
        S=1,31,32,33,63,64,65,127,128,129,511,512,513,1024
  pass: independent FP32 and upstream-style BF16 references, slab
        superposition, isolated head ownership, repeats, and nondefault stream
  pass: memcheck/synccheck/racecheck at S128/S129
  resources: dKV 222208 B and dQ 218112 B dynamic shared memory; 168
             registers, 1 KiB static shared, zero local/stack for each variant
  preserved reject: EXP-0005 unchanged direct asymmetric-GQA path

experiment ledger
  pass: EXP-0001/0002 accepted and EXP-0003 rejected against source
        5b9bfab072e8cc28a7e92c9e956608db591b246c
  pass: EXP-0004 accepted against source
        49fbcad2e2b761d9de50312f03335e27236a8a13
  pass: EXP-0005 rejected against source
        d7ac7273aaed5c57923301afa6f052333e91c5b7
  pass: EXP-0006 accepted against source
        185f11cbda15ae7bd4841968c3dd46f95b670282
```

See `docs/status.md` and EXP-0001 through EXP-0006 for exact commands,
tolerances, cache keys, artifact hashes, and scoped decisions.

## Not completed in the local environment

- The local Python environment does not contain the pinned Transformers
  checkout, so its optional executable oracle is skipped locally. The H100
  bootstrap installs that exact checkout and the remote oracle runs there.
- `shellcheck` is not installed locally. ShellCheck 0.9.0 is installed on the
  H100 and the complete bundle shell check passes there.
- Nsight Compute performance counters remain unavailable on the pod because
  the driver denies access (`ERR_NVGPUCTRPERM`). EXP-0006 dynamic shared-memory
  values come directly from retained generated MLIR, not Nsight metrics.
- B300 CUDA 13.3 / PyTorch 2.13.0 cu132 remains a separate, entirely unrun
  target-host gate.
- Multimodal kernels, varlen, lengths beyond the prepared S1024 global gate,
  and every benchmark remain unrun. Multimodal local masking is the next
  ordered H100 gate.

## Remaining remote evidence

```bash
bash scripts/remote/probe.sh b300
bash scripts/remote/bootstrap.sh b300
bash scripts/remote/check.sh b300
```

Do not start B300 in the H100-only scope. The next H100 kernel session must
open the multimodal local-mask correctness experiment; it must not skip ahead
to benchmarks.
