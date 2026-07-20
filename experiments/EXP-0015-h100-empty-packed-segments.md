# EXP-0015: H100 mixed empty packed segments

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256 and global-d512 packed-varlen forward/backward
- Architecture: sm_90
- Starting revision: `f4a68d5d6a54375e85776a5508cdbcbfb61e7fe6`
- Implementation revision: `cca09c8211b3c643b9b311f5fec0798f84a9ea0f`
- Result record: accepted `EXP-0015` entry in `experiments/results.jsonl`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `97dd1dd7c9c8efb5f2b2fd06f601a1bcc8abbe9ebcf43e9ea86365769c2e2749`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Accepted H100 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- Environment-policy hash:
  `807650a592d428fbfb2f7cacfd43622b008b5ffb68a815a02c37328c5b5a6935`
- Strict environment artifact:
  `agent_space/h100-check-exp0015.json`, SHA256
  `18c46284dd978362523f0d1d8b73adfc5fd45bdfd0ace0f7ec0c3aa86c5efdae`

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

- [x] CPU reference accepts mixed plateaus and rejects decreasing/all-empty input
- [x] project local/global guards accept `0 <= Sq <= Sk` only with positive totals/maxima
- [x] leading, middle, and trailing paired-empty segments pass
- [x] query-empty/key-nonempty segments pass with exact-zero dK/dV slices
- [x] local native text forward/backward passes O/LSE/dQ/dK/dV reference policy
- [x] local multimodal dense and long sparse-metadata compositions pass
- [x] global native forward/backward passes O/LSE/dQ/dK/dV reference policy
- [x] forced global exact-composer budget fallback skips empty-Q segments exactly
- [x] padded eager Transformers batches with a fully masked row use FA4 and restore
      zero O / `-inf` LSE sentinels for that row
- [x] hostile neighbor mutation and gradient-source isolation remain exact
- [x] all-empty totals, `Sq>Sk`, negative/decreasing arrays, wrong maxima, and
      malformed metadata fail before backend launch
- [x] pinned Transformers model-layer probe covers a fully padded row beside a
      nonempty row

The real H100 native local text case
`Q=[0,33,0,65,0], K=[0,64,1,65,0]` and the matching dense
vision/document case passed the independent upstream-relative O/LSE/dQ/dK/dV
policy. The long sparse case
`Q=[0,1,0,1,0], K=[0,1026,3,1026,0]` also passed, proving that zero-query
segments are omitted from sparse scheduling while their K/V slices remain
exact zero in backward. The global native probe passed its built-in mixed
case and an active-first hostile-isolation case
`Q=[33,0,65,0], K=[1025,1,2048,0]`; inactive gradients were exact zero and
the active segment was unchanged by hostile mutation.

The eager integration probe `global-packed-empty-row` selected
`fa4_global_varlen_native` in an actual pinned
`Gemma4TextAttention(layer_idx=5)`. It restored zero O / `-inf` LSE for the
fully padded row, preserved hostile-row isolation, produced exact-zero empty
row gradients, and left neighboring gradients nonzero. The H100 FakeTensor
matrix, including local/global plateau topologies, reported `16 passed`.

## Synchronization and generated code

- [x] memcheck passes a mixed local/global empty-segment case
- [x] synccheck passes a mixed local/global empty-segment case
- [x] racecheck passes a mixed local/global empty-segment case
- [x] leading/middle/trailing plateaus add no scheduler/application cache key
- [x] main PTX/cubin/SASS bytes and register/stack/local/SMEM resources remain
      identical to EXP-0010 local and EXP-0014 global evidence

Memcheck and synccheck each reported zero errors, and racecheck reported zero
hazards/errors/warnings, for both the global mixed-empty native backward and
the local long sparse vision/document case above. A fresh global cache audit
ended at 28 objects, 16 unique contents, 1,956,400 bytes, and 18 application
keys (nine fixed plus nine native). Empty replays in all SS/SM/MM classes
added no object or application key. Native global main-object hashes remain:

- dKV: `1df46be3f4071fa3fcf0a7d5a40f7ddbc4c589131ab83155b7f7089a51a23771`;
- dQ-low: `84ff8c1dddd65e9e1d5bb42c82b54aef0a68cb38543fc3b7f4764cbac208f775`;
- dQ-high: `b32f6711d185366c4effbda8c01aec99877f70da0ea11d73103c48aa1ac9f6ec`.

Fresh local text, dense metadata, and long sparse metadata cache probes ended
at four, four, and five objects respectively; every empty replay added zero
objects. Because the main objects are byte-identical to the retained
EXP-0010/0014 objects, their recorded PTX/cubin/SASS and resource inventories
are unchanged. Only the expected host-source cache namespace changed.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: independent PyTorch packed reference and
  nonempty-batch replay with empty segments removed

No performance measurements are authorized by this experiment.

## Decision

**ACCEPT.** Mixed packed workloads may contain per-segment plateaus satisfying
`0 <= Sq <= Sk <= 262144` on the declared H100 local/global paths. Overall
packed Q/K totals and exact maxima must remain positive, so an all-empty
physical workload is still rejected before backend dispatch. Fixed BSHD
behavior, model/mask semantics, tiles, synchronization, and application cache
classes are unchanged. This is a correctness and compatibility result, not a
performance or deterministic-gradient claim.

The exact accepted tree passed the strict H100 environment/patch check, the
pinned Transformers model oracle, the H100 suite (`313 passed, 16 skipped,
1 xfailed, 9 warnings`), and the dedicated FakeTensor matrix (`16 passed`).
The suite warnings are one retained PyTorch deprecation and documented
exact-fallback warnings; the xfail is the pinned generic FA4 mask adapter's
inability to encode Gemma's local vision overlay. The separate FakeTensor run
also reports retained upstream CuTe warpgroup deprecations.

`scripts/remote/bootstrap.sh h100` was also attempted. Package installation
completed, but its unconditional upstream FlashAttention `git fetch` returned
HTTP 403. That network failure is not reported as a successful bootstrap; the
already pinned checkout was instead checked directly, its new patch applied
exactly, and `scripts/remote/check.sh h100` passed with empty warnings/errors.

## Record

The schema-valid `EXP-0015` record names implementation revision
`cca09c8211b3c643b9b311f5fec0798f84a9ea0f`, the exact H100 environment
policy and strict artifact above, and makes no benchmark claim.
