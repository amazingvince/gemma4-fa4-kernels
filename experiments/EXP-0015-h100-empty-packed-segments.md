# EXP-0015: H100 mixed empty packed segments

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256 and global-d512 packed-varlen forward/backward
- Architecture: sm_90
- Starting revision: `f4a68d5d6a54375e85776a5508cdbcbfb61e7fe6`
- Implementation revision: pending
- Result record: pending
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `97dd1dd7c9c8efb5f2b2fd06f601a1bcc8abbe9ebcf43e9ea86365769c2e2749`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `91827fafa60d1710abb22034581f4db6c95d654c7af9b1e7ba5a07a51b4d6e35`

## Invariant changed

No model, mask, ownership, accumulation, or tile invariant changes. Retain
exact BF16 prepared Q/K/V, scale 1.0, distinct K/V, FP32 LSE/accumulation,
separate dQ/dK/dV, local window/vision/document semantics, and global
lower-right causality.

Change only packed segment admission from `1 <= Sq <= Sk` to
`0 <= Sq <= Sk`, while retaining positive physical Q/K totals, positive exact
`max_seqlen_q`/`max_seqlen_k`, the locked K262144 ceiling, signed-INT32
cumulative totals, and every existing memory/sparse-work guard. This admits
paired empty segments (`Sq=Sk=0`) and query-empty/key-nonempty segments
(`Sq=0<Sk`) anywhere in a mixed batch. It does not admit `Sq>0,Sk=0`,
decreasing cumulative arrays, or an all-empty physical workload.

An empty-Q segment owns no packed O/LSE entries and must contribute exact-zero
dQ/dK/dV. Adjacent nonempty segments retain their existing semantics and must
not observe or update the empty segment's K/V storage.

## Hypothesis

Admitting cumulative-array plateaus for mixed packed H100 requests will produce
no work for empty-Q segments and preserve reference-equivalent O, FP32 LSE,
and separate dQ/dK/dV for neighboring nonempty segments, without changing any
compiled main-kernel object or scheduler/application cache class.

Falsification is any launch for an all-empty request, accepted decreasing
cumulative array, output/LSE entry owned by an empty query segment, nonzero
gradient in an empty segment's K/V slice, cross-segment leakage, numerical
policy failure in a nonempty neighbor, illegal access, barrier/race report,
new plateau-length compile key, or changed main-kernel bytes/resources.

## Single change

Permit nondecreasing packed cumulative arrays with zero-length segments across
the reference, project local/global validators, eager Transformers pack/unpack
logic, and the managed SM90 global-backward wrapper. Skip zero-query segments
in exact per-segment composers. Do not alter tiles, stages, warp-group roles,
mask predicates, numerical tolerances, cache keys, or performance settings.

## Correctness evidence

- [ ] CPU reference accepts mixed plateaus and rejects decreasing/all-empty input
- [ ] project local/global guards accept `0 <= Sq <= Sk` only with positive totals/maxima
- [ ] leading, middle, and trailing paired-empty segments pass
- [ ] query-empty/key-nonempty segments pass with exact-zero dK/dV slices
- [ ] local native text forward/backward passes O/LSE/dQ/dK/dV reference policy
- [ ] local multimodal dense and long sparse-metadata compositions pass
- [ ] global native forward/backward passes O/LSE/dQ/dK/dV reference policy
- [ ] forced global exact-composer budget fallback skips empty-Q segments exactly
- [ ] padded eager Transformers batches with a fully masked row use FA4 and restore
      zero O / `-inf` LSE sentinels for that row
- [ ] hostile neighbor mutation and gradient-source isolation remain exact
- [ ] all-empty totals, `Sq>Sk`, negative/decreasing arrays, wrong maxima, and
      malformed metadata fail before backend launch
- [ ] pinned Transformers model-layer probe covers a fully padded row beside a
      nonempty row

## Synchronization and generated code

- [ ] memcheck passes a mixed local/global empty-segment case
- [ ] synccheck passes a mixed local/global empty-segment case
- [ ] racecheck passes a mixed local/global empty-segment case
- [ ] leading/middle/trailing plateaus add no scheduler/application cache key
- [ ] main PTX/cubin/SASS bytes and register/stack/local/SMEM resources remain
      identical to EXP-0010 local and EXP-0014 global evidence

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: independent PyTorch packed reference and
  nonempty-batch replay with empty segments removed

No performance measurements are authorized by this experiment.

## Decision

Pending.

## Record

Pending schema-valid `EXP-0015` record after the exact accepted H100 tree passes
all gates.
