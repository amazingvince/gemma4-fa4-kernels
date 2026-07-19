# Repository verification report

**Assembly date:** 2026-07-19

## Completed in the current local environment

```text
python -m compileall -q src tests scripts benchmarks
  pass

python -m pytest -q -p no:cacheprovider
  128 passed, 75 skipped
  skipped: H100 execution/fake-compile gates and the unavailable pinned
           Transformers oracle

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
  196 passed, 8 skipped, 1 xfailed on the final EXP-0010 implementation tree

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

local d256 fixed multimodal forward/backward
  pass: EXP-0007 exact Gemma vision predicate through S1025
  pass: O/LSE, separate dQ/dK/dV, LSE-only/combined gradients,
        exact GQA ownership, repeats/nondefault stream
  pass: memcheck/synccheck/racecheck at S128/S129 and S1025 memcheck

local d256 packed native/custom forward/backward
  pass: EXP-0008 nonempty B>=1, 1<=Sq<=Sk<=1025, lower-right text and
        K-stream vision/document metadata
  pass: O/LSE, true dout=None LSE-only and combined dQ/dK/dV, exact
        W1024 forward/backward sentinels, document and packed-boundary isolation
  pass: exact O/LSE/dQ/dK/dV repeats on a nondefault stream and changed-total
        native/custom forward/backward cache reuse
  pass: single/multi-block memcheck/synccheck/racecheck and K1025 memcheck
  SASS: M128xN80 forward and M64xN64 backward; 168 registers and 1 KiB
        static shared; forward 104-byte stack, backward zero stack/local

local d256 packed native text at production lengths
  pass: EXP-0009 nonempty B>=1, 1<=Sq<=Sk<=262144 on H100 SM90
  pass: S2048 independent references, repeats and nondefault stream; O/LSE/dK/dV
        bitwise, dQ pairwise max 0.03125 and every gradient oracle valid
  pass: S32768 nondefault-stream smoke and true Q=K=262144 out_lse execution
        after a 45231374336-byte preflight with 84465025024 bytes free
  pass: exact Q1/K262144 boundary sentinel and hostile Q=[33,65],
        K=[2049,4097] packed-sequence isolation
  pass: memcheck/synccheck/racecheck at Q=[64,65], K=[2048,2049] and
        Q1/K262144 memcheck
  pass: long runtime lengths retained one forward/four backward-invocation
        cache objects and unchanged native PTX/cubin/SASS resources

local d256 packed vision/document metadata at production lengths
  pass: EXP-0010 nonempty B>=1, 1<=Sq<=Sk<=262144 inside the declared
        2^40 padded-score, 2 GiB metadata, and 10%-free-HBM envelope
  pass: exact Q128xK80 forward and independently generated/transposed
        Q64xK64 backward tile incidence, with the complete token predicate
  pass: tractable O/LSE and O-only/LSE-only/combined dQ/dK/dV references,
        hostile packed/document isolation, repeats, and nondefault stream
  pass: Q1/K262144 strict-window ownership and Q2049/K262144 future-vision,
        document-isolation, LSE, and dV-ownership sentinels
  pass: memcheck/synccheck/racecheck, bounded cache reuse, and explicit sparse
        forward/backward fake compilation
  SASS: HGMMA/TMA retained; main backward has zero stack/local traffic;
        forward has LOCAL=0 plus a recorded 144-byte stack and LDL/STL traffic

experiment ledger
  pass: EXP-0001/0002 accepted and EXP-0003 rejected against source
        5b9bfab072e8cc28a7e92c9e956608db591b246c
  pass: EXP-0004 accepted against source
        49fbcad2e2b761d9de50312f03335e27236a8a13
  pass: EXP-0005 rejected against source
        d7ac7273aaed5c57923301afa6f052333e91c5b7
  pass: EXP-0006 accepted against source
        185f11cbda15ae7bd4841968c3dd46f95b670282
  pass: EXP-0007 accepted against source
        d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4
  pass: EXP-0008 accepted against source
        de6450a9cf5040a7432ed7641b230bb29f835248
  pass: EXP-0009 accepted against source
        9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf
  pass: EXP-0010 accepted against source
        12cfe711ad29139c7c78dcb355645ee5b9a70bb0
```

See `docs/status.md` and EXP-0001 through EXP-0010 for exact commands,
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
- Over-budget sparse schedules, empty packed segments, lengths beyond the
  prepared S1024 global gate, deterministic dQ, generic framework
  dispatch/context offsets, and every benchmark remain unrun. Framework and
  context-offset integration are the next ordered H100 compatibility gate.

## Remaining remote evidence

```bash
bash scripts/remote/probe.sh b300
bash scripts/remote/bootstrap.sh b300
bash scripts/remote/check.sh b300
```

Do not start B300 in the H100-only scope. The next H100 session must preserve
the accepted EXP-0010 sparse envelope while proving per-layer framework
dispatch and context-offset integration; it must not skip ahead to benchmarks.
