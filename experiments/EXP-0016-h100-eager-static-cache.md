# EXP-0016: H100 eager StaticCache active-prefix semantics

- Date / author: 2026-07-19 / Codex
- Kernel family: Transformers integration over retained local-d256 and global-d512 paths
- Architecture: sm_90
- Starting revision: `77ad0013386f399e2c613eea1d33182942d3c94f`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `807650a592d428fbfb2f7cacfd43622b008b5ffb68a815a02c37328c5b5a6935`
- Exemplars:
  - `.upstream/transformers/src/transformers/cache_utils.py`, pinned
    `StaticLayer` and `StaticSlidingWindowLayer`;
  - `.upstream/transformers/src/transformers/masking_utils.py`, pinned cache
    mask-size and offset construction;
  - `.upstream/transformers/src/transformers/models/gemma4/modeling_gemma4.py`,
    pinned `Gemma4TextAttention` cache-update ordering.

## Invariant changed

No model, mask, numerical, ownership, tile, or synchronization invariant
changes. Retain exact BF16 prepared Q/K/V, scale 1.0, distinct K/V, FP32 LSE
and accumulation, local window semantics, and global causality.

Change only eager inference admission for the pinned `StaticCache`. Before a
static cache is full, Transformers returns the complete fixed-capacity K/V
backing while its mask and offsets identify a shorter contiguous active
prefix. For B1 text-only requests, snapshot the real scalar query offset when
the mask plan is built, before the cache update mutates its tensor counter.
At attention dispatch, derive

`active_k = q_offset_at_mask_creation + Sq - kv_offset`.

When `1 <= active_k <= physical_k`, the mask describes exactly one valid
prefix, and every unwritten physical-tail position is masked, expose only the
zero-copy K/V prefix `[:active_k]` to the retained FA4 path. Normalize the
mask plan and any compact metadata to the same logical interval. A full or
rolled local cache already has `active_k == physical_k` and remains unchanged.

This experiment admits only eager `torch.inference_mode()` execution, B1,
text-only, and one contiguous valid K interval. Left or holey padding,
all-masked input, mask/offset disagreement, gradients, B>1 cache padding,
vision/static-cache transport, arbitrary 4D masks, FakeTensor, and
`torch.compile` remain fail-closed or deferred. Arbitrary or nonmonotonic
`cache_position`, cache crop/rewrite/offload, export, CUDA graphs, and capacity
overflow are also outside this gate; pinned upstream failures must propagate.

## Hypothesis

Normalizing a pinned eager `StaticCache` from physical capacity to the exact
offset-proven active K interval will preserve reference-equivalent O and FP32
LSE for local/global prefill and decode, including the first local rollover,
while hostile unwritten-tail values remain unobservable and existing FA4
kernel objects and application cache classes remain unchanged.

Falsification is any stale/mutated query offset, admitted noncontiguous mask,
read from an unwritten cache slot, changed output after hostile tail mutation,
incorrect local rollover order, local/global reference-policy failure,
changed cache address, K/V aliasing, unexpected eager fallback, graph or
compile claim, new scheduler/application cache class, changed main-kernel
bytes/resources, or sanitizer finding.

## Single change

Add an eager-only active-prefix normalization at the Transformers adapter
boundary. Snapshot a real scalar tensor `q_offset` in the registered mask
adapter before cache mutation, validate the exact B1 contiguous-prefix
contract, and slice only the sequence extent of K/V and related mask metadata
before existing dispatch. Do not change a CuTe kernel, tile, stage, mask
predicate, backward path, cache key, dependency revision, or performance
setting.

## Correctness evidence

- [ ] scalar-tensor query offsets are snapshotted before cache mutation
- [ ] active K is derived exactly from Q length and q/kv offsets
- [ ] short right-unwritten masks normalize to a contiguous logical prefix
- [ ] left, holey, all-masked, inconsistent, and out-of-range intervals reject
- [ ] grad-enabled physical-capacity requests remain fail-closed
- [ ] existing dynamic-cache, fixed, padded, packed, and lower-right tests pass
- [ ] direct local prefill/decode O and LSE pass the frozen reference policy
- [ ] direct global composed and forward-only O/LSE pass the frozen policy
- [ ] actual pinned `Gemma4TextAttention` layers 0 and 5 match an exact eager
      oracle under `torch.inference_mode()`
- [ ] local prefill S1023 then q1/K1024 and first rolled q1/K1024 pass
- [ ] global prefill S32 then q1/K33 with capacity 65 passes
- [ ] global prefill S1024 then q1/K1025 with capacity 1026 passes
- [ ] K/V cache addresses remain stable and K/V storage stays distinct
- [ ] hostile finite and NaN unwritten-tail values do not affect O/LSE
- [ ] only the intended cache slots mutate at each step

The actual-layer oracle must use the same pinned weights, positions, scale,
dtype, and mask semantics with an independently initialized cache. Direct
prepared-Q/K/V cases retain the existing local/global output and LSE policies.
No absence of a crash is accepted as numerical evidence.

## Synchronization and generated code

- [ ] memcheck passes one local rollover case
- [ ] synccheck passes one local rollover case
- [ ] racecheck passes one local rollover case
- [ ] memcheck passes one global q1/K1025 static-prefix case
- [ ] synccheck passes one global q1/K1025 static-prefix case
- [ ] racecheck passes one global q1/K1025 static-prefix case
- [ ] active-length/capacity replays add no scheduler/application cache class
- [ ] retained main PTX/cubin/SASS bytes and resource signatures are unchanged

The host adapter source may create an expected source-bound cache namespace.
That does not authorize a new length-, offset-, capacity-, or payload-dependent
application key. Inspect fresh cache contents and compare retained main-object
hashes rather than inferring codegen stability from a successful launch.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: project reference plus pinned eager
  Transformers execution with an independently initialized cache

No performance measurements are authorized by this experiment.

## Decision

**PENDING.** Accept only if every declared CPU, H100 reference, actual-layer,
tail-isolation, cache/address, sanitizer, and generated-code gate passes. A
partial prefill/decode result does not establish static-cache compatibility.

Framework FakeTensor/fullgraph `torch.compile` is a separate next experiment;
compiled StaticCache decode follows only after both eager cache semantics and a
real compiler operator boundary are proven. No result here generalizes to
B300, multimodal cached generation, batched padding, training, full
checkpoint `generate`, maximum-capacity allocation, or performance.

## Record

No result record exists while the decision is pending.
