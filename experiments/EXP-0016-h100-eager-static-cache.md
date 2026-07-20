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
prefix. Expose only the zero-copy K/V prefix `[:active_k]` to the retained FA4
path and normalize the mask plan and compact metadata to that logical
interval. The logical prefix must be valid; the physical tail need not contain
zeros or be explicitly false in a 2D padding mask because the proven causal
predicate makes future-capacity slots unreachable. A full or rolled local
cache already has `active_k == physical_k` and remains unchanged.

This experiment admits only eager execution with no active autograd/backward,
B1, text-only, and one contiguous valid K interval. The actual layer probes
run under `torch.inference_mode()`. Left or holey padding,
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

- [x] scalar-tensor query offsets are snapshotted before cache mutation
- [x] active K is derived exactly from Q length and q/kv offsets
- [x] short right-unwritten masks normalize to a contiguous logical prefix
- [x] left, holey, all-masked, inconsistent, and out-of-range intervals reject
- [x] grad-enabled physical-capacity requests remain fail-closed
- [x] existing dynamic-cache, fixed, padded, packed, and lower-right tests pass
- [x] direct local prefill/decode O and LSE pass the frozen reference policy
- [x] direct global composed and forward-only O/LSE pass the frozen policy
- [x] actual pinned `Gemma4TextAttention` layers 0 and 5 execute through a
      captured prepared-operand eager oracle under `torch.inference_mode()`
- [x] local S32, local prefill S1023 then q1/K1024, and first rolled q1/K1024
      pass
- [x] global prefill S32 then q1/K33 with capacity 65 passes
- [x] global prefill S1024 then q1/K1025 with capacity 1026 passes
- [x] K/V cache addresses remain stable and K/V storage stays distinct
- [x] hostile finite and NaN unwritten-tail values do not affect O/LSE
- [x] only the intended cache slots mutate at each step

The five selectable actual-layer cases use the pinned layer implementation and
weight-identical clean, hostile, and dynamic caches. A custom probe backend
captures the pinned eager prepared Q/K/V/O directly. Candidate active Q/K/V
are bitwise equal to those captured operands, and replaying each captured or
candidate prepared O through the identical `o_proj` is bitwise exact. The FA4
prepared O and FP32 LSE are gated against the project FP32 reference under the
pre-existing local/global policies. Eager-versus-project and
candidate-versus-eager BF16 deltas are recorded observations, not pass
thresholds; no post-observation tolerance multiplier is used.

Pinned Transformers intentionally does not infer packing from `position_ids`
when a cache exists. After active-prefix trimming, reset position IDs therefore
remain prepared RoPE inputs and are not reinterpreted as packed-sequence
metadata. Tests also prove that explicit cumulative arrays, document metadata,
active vision metadata, B2, nonzero underfilled K offsets, FakeTensor, and
active backward fail closed without losing the original physical operands or
mask plan needed by a permitted fallback.

Representative accepted FA4-versus-project errors were at most 0.015625 in
prepared BF16 O and 0.00012970 in FP32 LSE across the recorded local/global
static-cache cases. These are correctness observations only, not performance
measurements.

## Synchronization and generated code

- [x] unfiltered memcheck passes one local rollover case
- [x] filtered project-kernel synccheck passes one local rollover case
- [x] filtered project-kernel racecheck passes one local rollover case
- [x] unfiltered memcheck passes one global q1/K1025 static-prefix case
- [x] filtered project-kernel synccheck passes one global q1/K1025 case
- [x] filtered project-kernel racecheck passes one global q1/K1025 case
- [x] active-length/capacity replays add no scheduler/application cache class
- [x] retained main PTX/cubin/SASS bytes and resource signatures are unchanged

The host adapter source may create an expected source-bound cache namespace.
That does not authorize a new length-, offset-, capacity-, or payload-dependent
application key. Inspect fresh cache contents and compare retained main-object
hashes rather than inferring codegen stability from a successful launch.

Unfiltered synccheck also inspected the actual-layer probe and reported
divergent barriers in NVIDIA's `libcublasLt.so.12` output-projection kernel,
not in the project FA4 launch. The retained CuTe object was inspected with
`nm -C`, and synccheck/racecheck were rerun with the verified
`kns=flash_attncuteflash_fwd_sm90` project-kernel substring; both local rollover
and global K1025 cases were clean. This vendor-library finding is retained
rather than silently discarded.

Fresh-cache capacity replays established both code paths:

- global Q1/K33 with physical capacities 65 BHSD and 129 BSHD-backed retained
  28 objects, 16 unique contents, 1,956,400 bytes, 18 backward application
  keys, and two forward application keys;
- local Q1/K33 with the same two physical layouts retained one object and one
  forward application key. Its object SHA256 is
  `366fe4840ce1f9f601e201e118dbe45ba4a86b27b813fa5ffb226c1c7a24b5ce`.

NaN physical tails were unobservable and both replays produced bitwise O/LSE
parity. No CuTe kernel source changed in this experiment, so retained generated
main-object bytes/resources are unchanged; only the expected host-source cache
namespace changed.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: project reference plus pinned eager
  Transformers execution with an independently initialized cache

No performance measurements are authorized by this experiment.

## Decision

**ACCEPT.** Implementation revision
`c5ee7bec833c9617ccf323955bcafc80b72cd932` passes every declared CPU, H100
reference, actual-layer, tail-isolation, cache/address, project-kernel
sanitizer, and generated-code gate. Final local verification reports
`298 passed, 83 skipped, 8 warnings`; the synced H100 tree reports
`369 passed, 16 skipped, 1 xfailed, 8 warnings`. The H100 FakeTensor
kernel-wrapper matrix separately reports `16 passed`; it remains lower-level
evidence and is not a framework compile claim.
The final immutable 18-case
`probe_h100_transformers_integration.py --case all` command also reports
`status: passed`, including all five StaticCache cases under the final oracle.

The strict H100 environment artifact is
`agent_space/h100-check-exp0016.json`, SHA256
`18c46284dd978362523f0d1d8b73adfc5fd45bdfd0ace0f7ec0c3aa86c5efdae`,
with an empty warning/error set and the pinned revisions above.

Framework FakeTensor/fullgraph `torch.compile` is a separate next experiment;
compiled StaticCache decode follows only after both eager cache semantics and a
real compiler operator boundary are proven. No result here generalizes to
B300, multimodal cached generation, batched padding, training, full
checkpoint `generate`, maximum-capacity allocation, or performance.

## Record

`experiments/results.jsonl` records the accepted implementation revision and
the strict EXP-0016 environment artifact. The acceptance record makes no
compile, B300, multimodal cached-generation, batched padded-cache, training,
maximum-capacity allocation, or performance claim.
