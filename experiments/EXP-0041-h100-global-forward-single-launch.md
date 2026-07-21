# EXP-0041: H100 global D512 single-launch forward

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-fwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Environment: `configs/env/h100-compatible.env`
- Design brief: `docs/h100-global-forward-d512-single-launch-design.md`

## Invariant changed

Replace two independent D256 forward launches, each recomputing QK and
softmax, with one launch that owns the complete O512 result and one softmax/LSE
state. The accepted composition remains the rollback.

## Hypothesis

An M64 x N32 specialization with disjoint O256 warpgroup owners and one shared
QK/online-softmax producer will compile without local-memory spills, preserve
the exact FP32 LSE, and reduce hot-L2 S8K and S64K forward median latency
relative to the two-launch V256 composition.

## Single change

Admit SM90 `(512,512)` and add one cooperative M64 x N32 path. WG0 owns QK,
online softmax, and O-low; WG1 owns O-high. WG0 shares BF16 P and FP32 row
rescale factors through named-barrier-protected shared memory. The first direct
M128 candidate was rejected separately because WGMMA PV N cannot exceed 256.

## Correctness evidence

- [x] direct full-D compile discriminator rejected at WGMMA N=512
- [x] fixed S={1,31,32,33,63,64,65,127,128,129,511,512,513,1024}
- [x] native packed mixed, empty, isolation, and fixed/native parity matrices
- [x] exact rollback parity for fixed S128 and packed mixed-empty
- [x] independent FP32 references for O/LSE and accepted backward gradients
- [x] repeated-run, GQA ownership, distinct-K/V, and nondefault-stream checks

## Synchronization and generated code

- [x] fixed and packed memcheck
- [x] fixed and packed synccheck
- [x] fixed and packed racecheck
- [x] Nsight launch attribution and SASS instruction inventory
- [x] 168 registers, zero stack/local, 1 KiB static, 205,824 B dynamic shared

The fixed and native packed objects contain 102 HGMMA instructions for the
cooperative route. Nsight Systems records exactly one project forward kernel,
grid 4096 x block 384 at S8K. The fixed/packed sanitizer runs use
`--report-api-errors no --error-exitcode 99`; memcheck and synccheck report
zero errors and racecheck reports zero hazards, errors, or warnings.

## Regression and provenance

- Local: compileall, Ruff, and the locked model contract pass; pytest reports
  `443 passed, 106 skipped`.
- H100: the strict profiler-required environment check reports the pinned
  patch SHA256 and `applied_exactly: true`; the full suite reports
  `536 passed, 17 skipped, 1 xfailed`.
- Pinned Transformers: model-contract oracle passes; the focused oracle suite
  reports `5 passed, 1 xfailed` (the declared generic FA4 mask-adapter gap).
- Fresh base revision plus patch reproduces the four-file upstream diff; the
  active checkout matches that reconstruction byte-for-byte.

## Measurement

- Clock/power state: unlocked RunPod H100 clocks; same device and process policy
- Hot/cold L2: hot
- Warmup/repetitions/statistic: S8K 10/30; S64K 10/30; median and inclusive IQR
- Semantically equivalent baseline: accepted exact two-V256-launch composition

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K forward | 6.1936 / 0.3333 ms | 5.5986 / 0.0226 ms | **-9.61%** |
| global S64K forward | 364.7140 / 3.0814 ms | 356.3823 / 2.5474 ms | **-2.28%** |

## Decision

**ACCEPT.** Make the cooperative route the H100 global-forward default and
retain `FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH=0` as the
tested exact rollback. This removes the duplicated QK/softmax launch; it does
not combine forward with backward or change the accepted backward ownership.
