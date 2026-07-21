# EXP-0044: H100 production long-context soak

- Date / author: 2026-07-20 / Codex
- Kernel family: production-validation
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Environment: `configs/env/h100-compatible.env`
- Design brief: `docs/h100-production-soak-design.md`

## Invariant changed

None. Add a reproducible repeated-execution harness around accepted production
routes without changing runtime behavior.

## Hypothesis

Fresh-process long-context repetitions preserve every accepted output,
gradient, stream, memory-admission, compile-key, and allocator-release contract
at global S65536, global Q1/K262144, and local Q=K=262144.

## Single change

Add and run `scripts/probe_h100_production_soak.py`. Do not tune a kernel,
change a route, widen semantics, or reinterpret diagnostic timing as a speedup.

## Acceptance gates

- [x] global S65536 default forward+backward completes 2 warmups + 5 samples
- [x] global Q1/K262144 analytic out+LSE backward passes 3 stream repetitions
- [x] local Q=K=262144 forward+backward passes 3 fresh seeds/streams
- [x] every child exits zero and leaves no compute process or allocation
- [x] exact commands, complete logs, timings, and cache inventory are recorded
- [x] strict environment, full suites, oracle, schema, and bundle remain green

The final replay completes all five child processes in 76.8 seconds. Global
S65536 forward+backward reports a 2903.654 ms hot-L2 median and 2.035 ms IQR
with `forward_single_launch=true` and `owner_computes_dkv=true`; this is a
diagnostic, not a new comparison. Global Q1/K262144 is bitwise across three
nondefault-stream repetitions with exact O/LSE/dQ/dK/dV versus the analytic
finite-score oracle. Three fresh local S262144 seeds pass the BF16 O/LSE and
separate-gradient contracts after a 49,653,923,840-byte memory preflight.

Every child exits zero and leaves an empty compute-process list. The FA4 disk
cache grows from 10 files after global S64K, to 16 after asymmetric global, to
22 after the first local maximum-context process, then remains exactly 22 files
and 1,199,032 bytes through the other two local seeds.

The local suite reports `451 passed, 106 skipped`; the fresh-cache H100 suite
reports `544 passed, 17 skipped, 1 xfailed`. Compileall, Ruff check/format, the
locked model contract, strict exact-patch environment, and pinned Transformers
oracle pass; the focused oracle reports `5 passed, 1 xfailed` for the declared
generic FA4 mask-adapter gap. The schema result is pinned to implementation
`d7fe7f4`; final bundle verification passes with that record and the refreshed
manifest.

## Decision

**ACCEPT.** Keep the soak harness as a production regression. It validates the
accepted routes and resource lifecycle only; it does not change dispatch or
establish a new comparative performance result.
