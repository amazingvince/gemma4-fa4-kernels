# Experimental plan

## North star

Model-weighted attention time for all 50 local and 10 global layers, forward
plus backward, at 32K and 128K BF16 on one H100 and one B300, compared with the
best semantically equivalent composite baseline.

A two-times speedup at 128K is a research target, not a promised outcome.

## M0 — establish the ruler

On each host:

1. capture `scripts/check_env.py --strict` output;
2. lock sustainable clocks or flag the run as unlocked;
3. record GEMM and HBM-copy denominators with `benchmarks/roofline.py`;
4. compile the current upstream FA4 matrix with persistent cache enabled;
5. capture separate fwd, bwd, and fwd+bwd timings;
6. profile first-call JIT separately from warm calls;
7. record exact unsupported cases rather than substituting different masks.

## Benchmark ladder

Smoke:

```text
local:  B=1, 32Q/16KV, d256, W1024, S={4K,32K}
global: B=1, 32Q/4KV,  d512, causal, S={8K,64K}
```

Run each in `fwd`, `bwd`, and `fwd_bwd`. Add TP-local GQA ratios 1/2/4/8 to the
full ladder. The full ladder spans 1K–256K, varlen document mixes, text and
vision masks, deterministic/nondeterministic modes, and BF16/FP16 coverage.

Memory preflight may skip infeasible entries but records estimates and free
memory. A skipped full shape is not silently replaced by a smaller head count.

## Semantic baseline rule

A result is comparable only if these match:

- scale 1.0;
- local/global mask including vision behavior;
- Q/K/V and GQA geometry;
- dtype and accumulation contract;
- fwd versus bwd versus combined timing region;
- LSE/output/gradient contract.

For example, plain full-causal SDPA is not a local-window baseline.

## Measurement hygiene

- fixed clocks, temperature, power, and a canary kernel;
- CUDA events for device time; CUDA graphs only as a separate stacked result;
- hot- and cold-L2 modes reported explicitly;
- direct GPU input generation and dynamically sized cache-thrash buffers;
- median, p25, p75, and IQR after warmup;
- cold compile and warm cache reported separately;
- append-only accepted and rejected experiment records;
- no chasing changes inside the measured noise floor.

Default acceptance requires at least 1.5% improvement beyond noise, no
correctness or sanitizer regression, no new spills, no cache-key churn, and no
more than 3% loss on another important regime.

## Milestones

### H100-M1: H100 correctness path

Status 2026-07-19: the pinned environment, fixed and packed local d256
forward/autograd backward, exact local multimodal masking, and composed global
d512 forward/backward paths passed their declared gates. EXP-0003's fixed
elementwise dQ/dK envelope remains rejected;
EXP-0004 diagnosed the BF16 oracle mismatch and accepted the unchanged local
backward under a predeclared upstream-relative policy. EXP-0005 preserves the
direct asymmetric GQA-8 backward rejection. EXP-0006 accepts a structural
split over the exact B1/S<=1024/32Q/4KV/GQA-8/d512/causal/scale-1.0 contract:
one dKV-only and two D256 dQ-only main launches per V256 slab, with FP32
cross-slab dQ/dK accumulation and separate dV slabs. Its 14-length numerical
matrix, S128/S129 memcheck/synccheck/racecheck, and generated-code resource
gates pass. Gradient repeats are non-bitwise because FP32 bulk/atomic
reduction order can vary, but every run passes the frozen numerical policy.
EXP-0007 accepts fixed B1 local multimodal masking. EXP-0008 accepts nonempty
packed local self-attention with `B>=1`, per-sequence
`1 <= Sq <= Sk <= 1025`, native lower-right text, and exact custom
vision/document semantics. EXP-0009 accepts native packed text through the
locked maximum `1 <= Sq <= Sk <= 262144`. EXP-0010 accepts exact
vision/document metadata through the same maximum inside the declared sparse
resource envelope. EXP-0011 accepts eager pinned-Transformers dispatch and
global no-grad forward through K262144. EXP-0012 validates the unchanged split
global backward scheduler through fixed S2048 and exactly composed
lower-right/packed K2048 under fail-closed HBM preflight. EXP-0013 accepts
native THD/cu-seqlens global backward for nonempty per-segment
`1 <= Sq <= Sk <= 2048` with exact BF16 32Q/4KV/GQA-8/d512/lower-right-causal/
scale-1.0/distinct-K/V geometry. Only the dedicated native HBM-budget exception
selects the exact EXP-0012 composer; validation, contract, assertion, and
runtime failures propagate. EXP-0014 extends only the native packed route to
nonempty per-segment `1 <= Sq <= Sk <= 262144`, subject to signed-INT32 and
guarded-HBM admission. Fixed BSHD and the exact composer remain capped at
S/K2048; a K>2048 budget rejection propagates before forward. EXP-0015 admits
mixed local/global packed segments with per-segment
`0 <= Sq <= Sk <= 262144`, positive aggregate Q/K totals, and positive exact
maxima. Paired empty and query-empty/key-nonempty segments pass reference,
isolation, sanitizer, cache, and unchanged-main-object gates; all-empty
physical workloads still reject before backend launch. EXP-0016 accepts eager
B1 text-only StaticCache active-prefix prefill/decode with no active backward.
EXP-0017 rejects the first no-cache framework FakeTensor/fullgraph custom-op
candidate: an empty `DynamicCache` was invisible until after mutation, the
registered mask wrapper did not prove exact callable origin, stock dynamic
compilation split S1 from S>1, and a partial Inductor run was not bitwise to
eager. EXP-0018 repairs cache/origin provenance and rejects every real cache
before mutation, but rejects the positive matrix when local Inductor S1023
exceeds its frozen full-layer numerical envelope. EXP-0019 localizes that
outer-layer drift and makes the whole-layer eager path bitwise, but rejects
the live-weight ABI; its ownership snapshot then creates two identical S1
backend captures. EXP-0020 removes the snapshot and admits exact live dormant
Parameters only under inference, but stock PyTorch 2.8 first raises
`TensorifyScalarRestartAnalysis` on the retained float-valued module/config
proof and then compiles the identical graph. The frozen one-attempt S1 gate
therefore rejects it. EXP-0021's explicit CPU-FP64 transport and EXP-0022's
public comptime static guards also reject: the latter removed every scalar
operand/node from FX, but the first identical Inductor graph still raised
`TensorifyScalarRestartAnalysis`. EXP-0023 then accepts the explicitly named
guarded no-cache facade for actual pinned layers 0/local and 5/global, B1 BF16
text inference through S1024. Live floats/module/weights validate outside the
compiled frame on every call; the tensor-only inner graph is bitwise to eager,
preserves the public S1/S>1 classes and bounded FA4 keys, and passes mutation,
input/API rejection, sanitizer, replay/stream, and unchanged-codegen gates. A
nondefault size-oblivious diagnostic also passes as one graph. Raw
`torch.compile(layer)` remains unsupported. EXP-0026 separately accepts only
global layer-5 one-token compiled StaticCache decode after eager prefill,
through sequential K34 and independent K1025. EXP-0027 rejects conservative
local-counter mutation, while EXP-0028 separately accepts only local layer-0
one-token compiled `StaticSlidingWindowLayer` decode through K33/K34
underfill, K1024 boundary fill, and repeated rollover through absolute
position 1025. Compiled prefill, cached multimodal decode, other layer
  indices, full-model compilation, and varlen facade inputs require separate
  predeclarations. EXP-0039 accepts opt-in deterministic global backward;
  deterministic local gradients remain deferred. See `docs/status.md` and the
  recorded experiments.

EXP-0042 completes the separately predeclared no-cache layer-index widening:
all 60 locked text layers pass eager and Inductor through the same guarded
facade, with exact index pinning and two family-only graph/FA4 classes. The
other-layer statement above continues to apply only to compiled cache facades.

- pinned FA4 CuTe SM90 build on CUDA 12.x;
- local d256 forward and a scoped local d256 backward configuration
  (**complete for the declared M1 text envelope**);
- global d512 single-launch forward with exact rollback, plus split backward
  (**complete through fixed/composed K2048 in EXP-0012 and native packed
  K262144 in EXP-0014**);
- exact scale, O/LSE, separate dQ/dK/dV, GQA, and text boundaries;
- local multimodal forward/backward (**complete for fixed B1**);
- packed local native/custom forward/backward (**complete through S1025**);
- native packed local text (**complete through S262144 in EXP-0009**);
- production-length vision/document metadata with an exact sparse schedule
  (**complete within the declared resource envelope in EXP-0010**);
- eager per-layer framework dispatch and context-offset integration
  (**complete for the declared eager envelope in EXP-0011 through EXP-0016**);
- mixed empty packed segments with positive aggregate totals/maxima
  (**complete in EXP-0015; all-empty physical workloads remain rejected**);
- eager B1 text-only StaticCache active-prefix prefill/decode with no active
  backward (**complete in EXP-0016**);
- guarded no-cache compiled facade (**complete through EXP-0042 for all 60
  locked layer indices, B1 BF16 text inference through S1024; raw
  `torch.compile(layer)` remains unsupported**) and scoped compiled-cache decode (**complete in EXP-0026 for
  pinned global layer 5 and EXP-0028 for pinned local layer 0**); compiled
  prefill, cached multimodal decode, other-layer cache/full-model, and
  varlen-facade integration plus deterministic local gradients remain deferred; EXP-0039
  accepts opt-in deterministic global backward;
- no performance tuning until every H100 correctness and sanitizer gate passes.

### B300-M1: B300 local correctness (deferred in the H100 session)

- prepared Q/K/V, BF16, fixed length, text mask;
- exact scale/window/O/LSE;
- one-CTA and two-CTA variants compile and pass sanitizer;
- add GQA-2 and boundary masks.

### M2: Global d=512 forward

- H100 D-split schedule;
- B300 one-CTA/two-CTA schedule comparison;
- GQA 8 plus TP-local ratios;
- O and LSE green through adversarial boundaries.

### M3: Global backward

H100 M1 began with a correctness-first six-main-launch composition with
temporary whole-tensor FP32 accumulation, validated through K2048. EXP-0035
keeps its two dKV slab launches but replaces four dQ slab launches with two
exact D256-streaming dQ launches. EXP-0037 replaces the two dKV slab launches
with one full-D512 dKV kernel that streams dO low/high/low-replay and reuses
dead Q shared storage for the epilogue, making three main launches the H100
route. EXP-0038 then executes both dQ D256 output halves sequentially in one
main launch while reusing the same registers, shared epilogue arena, and
barriers, making two main launches the H100 default: one dKV and one dQ. The
accepted rollback flags independently retain the EXP-0035 dKV slabs, the
EXP-0037 two-launch dQ route, and the earlier slab-dQ route.
EXP-0013 accepts the
native THD/cu-seqlens packed form for nonempty per-segment
`1 <= Sq <= Sk <= 2048`; this changes packed scheduling, not the split
ownership or temporary-accumulator design. EXP-0014 extends the unchanged
native split form through K262144 for nonempty resource-admissible segments;
fixed BSHD and the exact composer remain capped at S/K2048. EXP-0015 extends
the packed ABI to mixed `0 <= Sq <= Sk <= 262144` segments while retaining
positive aggregate totals/maxima, no work for empty queries, and unchanged
main-kernel objects. Target fused paths without whole-layer FP32 temporary
buffers still require:

- preprocess;
- owner-computes dQ;
- owner-computes dK and dV;
- no required whole-layer FP32 accumulation buffer in the target path;
- deterministic ownership mode.

### M4: Local multimodal, varlen, and integration

- fixed vision masking and packed document boundaries are correctness-complete
  for the EXP-0007/0008 envelopes;
- production-length vision/document tile classification is correctness-complete
  within EXP-0010's declared resource envelope;
- explicit verification that cross-layer KV reuse is disabled;
- per-layer HF dispatch and context-parallel offsets.

### M5: model-specific fusion and precision exploration

- shared-source K/V preparation fusion;
- exact preparation backward to dZ;
- FP8/FP4 only with numerical and training-convergence evidence.

## Profiler decision tree

1. Nsight Systems: launches, JIT, host gaps, graphs.
2. Nsight Compute: SOL, tensor pipeline, SMEM/L2/HBM, barriers, occupancy.
3. PTX/SASS and ptxas: target instructions, register counts, spills.
4. Compute Sanitizer: memcheck, synccheck, racecheck.
5. Error heatmaps: position, tile boundaries, vision/window edges, max drift.

Use the full CuTe skill's design brief, review checklist, and benchmark report
templates for every structural kernel change.
