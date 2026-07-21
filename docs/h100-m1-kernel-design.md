# H100 M1 CuTe DSL kernel design brief

This brief is complete before implementation. The locked model contract wins
over any upstream default or example.

## 1. Environment and version

- Date: 2026-07-19.
- `nvidia-cutlass-dsl`: installed and pinned by FA4 to `4.6.0.dev0`.
- `quack-kernels`: 0.5.3, pinned after resolving the successful H100 runtime.
- CUTLASS / FA source: FlashAttention
  `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`.
- CUDA toolkit / driver: CUDA 12.8 / 580.126.09; strict build gate passed.
- Python: 3.12.3 observed remotely.
- Framework: pinned PyTorch 2.8.0+cu128; target-specific policy passed.
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0.
- Official examples: pinned `flash_attn/cute/flash_fwd_sm90.py`,
  `flash_bwd_sm90.py`, `interface.py`, and `tests/cute/`.
- Verified APIs: SM90 forward/backward dispatch, explicit `softmax_scale`,
  `window_size`, `mask_mod`, returned FP32 LSE, and separate dq/dk/dv outputs.
- H100 patch stack: exact base revision above plus
  `patches/flash-attention/0004-sm90-gemma4-forward-d512-single-launch.patch`,
  SHA256
  `9d14635e23199200f0b25cbd9d33f464d1dd119a527838cc96098e9b8b3d91dd`.
  This patch includes EXP-0038's accepted single-launch dQ route. Together
  with EXP-0037's full-D dKV route, it makes two backward main launches the
  default; setting its experiment flag to `0` restores EXP-0037.
  The patch also contains EXP-0039's accepted explicit deterministic global
  backward route. It does not affect default dispatch and remains a
  correctness option rather than the long-context throughput route.
  EXP-0040 changes the fast default dKV ownership: one CTA accumulates all
  eight GQA Q heads for each KV tile and writes final BF16 dK/dV directly.
  The rollback flag value `0` restores EXP-0038 without changing deterministic
  dispatch.
  EXP-0041 changes the forward default to one M64 x N32 launch. Two consumer
  warpgroups own disjoint O256 halves while one produces QK, online softmax,
  BF16 P, and FP32 rescale factors for both. Setting
  `FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH=0` restores
  the exact two-V256-launch forward composition.

## 2. Operation contract

- Operation: Gemma 4 31B prepared-Q/K/V attention, BF16 forward and backward.
- Formula: `S[b,h,q,k] = sum_d Q[b,q,h,d] * K[b,k,h/g,d]`;
  `P = softmax(mask(S), dim=k)` because scale is exactly `1.0`;
  `O[b,q,h,v] = sum_k P[b,h,q,k] * V[b,k,h/g,v]`.
- Inputs: contiguous BSHD BF16 Q, K, and V for fixed calls, or contiguous THD
  operands plus explicit INT32 cumulative Q/K lengths for packed calls. K and
  V are always distinct operands even when they originate from one projection
  source.
- Outputs: BF16 O, FP32 LSE; BF16 dQ/dK/dV. Forward and backward WGMMA
  accumulators are FP32. In the pinned backward, P is rounded to BF16 before
  dV and dS is rounded to BF16 before dQ/dK; validation must model those
  intentional algorithmic boundaries rather than claim one final conversion.
- Local mask: `k > q - 1024 AND (k <= q OR same nonnegative vision block)`.
- Global mask: `k <= q`.
- NaN/Inf: match the reference; no silent sanitization. A nonempty query row
  must retain at least one valid key under the selected contract; an empty-Q
  packed segment owns no row at all.
- Empty inputs: EXP-0015 admits paired-empty and query-empty/key-nonempty
  segments inside a mixed packed workload. Aggregate Q/K totals and exact
  maxima remain positive; a fully all-empty physical workload rejects before
  backend launch.
- Aliasing: Q/K/V/O and dQ/dK/dV must not alias. No in-place operation.
- Determinism: correctness mode uses fixed seeds and repeated-run checks.
  Forward O/LSE repeats must be exact. The fast default's FP32 bulk/atomic
  reductions remain non-bitwise and every repeat must independently pass the
  frozen numerical policy. EXP-0039 accepts an explicit ordered global
  backward whose fixed and packed O/LSE/dQ/dK/dV repeats are bitwise exact;
  deterministic local backward remains deferred.

## 3. Shape and layout regime

- Rank: fixed BSHD rank 4 and packed THD plus rank-1 cumulative arrays have
  separate accepted adapters.
- Dynamic: B, packed totals, cumulative values, sequence lengths, vision IDs,
  and document IDs.
- Static compile configuration: architecture, dtype, QK/V head dimensions,
  GQA ratio, mask family, tile sizes, pipeline stages, and V-slab width.
- Local: 32 Q / 16 KV heads, d256, GQA 2, W1024.
- Global: 32 Q / 4 KV heads, d512, GQA 8.
- Strides: contiguous BSHD for fixed inputs and contiguous THD for packed
  inputs.
- Alignment: the adapter requires 16-byte-aligned BF16 base pointers; D
  extents satisfy the upstream 8-element alignment used by TMA descriptors.
- Accepted adapter envelope: fixed local B1 `1 <= S <= 1025`; packed local
  mixed `B>=1` text with per-sequence `0 <= Sq <= Sk <= 262144`, positive
  aggregate Q/K totals and positive exact maxima, with vision/document
  metadata subject to EXP-0010's sparse resource envelope; fixed global B1
  `1 <= S <= 2048`; native packed global mixed `B>=1` under the same length/
  aggregate/maxima contract, subject to signed-INT32 and guarded-HBM admission.
  No-grad global forward also extends through K262144. The exact fixed and
  composer backward paths remain capped at S/K2048 for active query segments.
- Eager cache envelope: EXP-0016 accepts only B1 text-only, no-active-backward
  underfilled pinned `StaticCache` requests whose offsets prove one contiguous
  active K prefix. EXP-0026 separately accepts compiled one-token decode only
  for pinned global layer 5 through K1025 after eager prefill. EXP-0028
  separately accepts compiled one-token decode only for pinned local layer 0
  through underfill, K1024 boundary fill, and repeated saturated rollover.
- Adversarial shapes: q positions 0, 1023, 1024, 1025; partial M/N tiles;
  unequal q/k lengths; GQA ratios 1,2,4,8; vision spans crossing tile and
  window boundaries; distinct random K and V.
- Rejected initially: non-BF16, noncontiguous D, dropout, d not in {256,512},
  cross-layer KV reuse, and normal-attention K/V aliasing.
- Cache key: FA revision, DSL version, SM90a, dtype, D, Dv, GQA ratio, mask
  family/callable hash, auxiliary layout metadata, varlen class, tile M/N,
  D-slab width/count, stage count, pack-GQA choice, and every
  constexpr/codegen flag. Runtime totals, cumulative values, lengths,
  metadata values, and pointers are excluded.

## 4. Target and kernel family

- Target: SM90a H100.
- Local family: pinned FA4 `FlashAttentionForwardSm90` through a
  model-contract adapter. Autograd uses the pinned M64 x N64
  `FlashAttentionBackwardSm90` path with Q/dO/PdS stages 1/1/1, two MMA
  warpgroups, unpacked GQA-2, and FP32 dK/dV workspaces. Upstream tests still
  exclude SM90 backward above d192; EXP-0004 supplies project-scoped
  correctness and sanitizer evidence for exact B1/32Q/16KV/d256/W1024 text
  attention through S1025.
- Global forward: EXP-0041's accepted cooperative SM90 `(512,512)` M64 x N32
  launch. WG0 owns QK/online-softmax and O-low, shares P/row scales through
  named barriers, and WG1 owns O-high. The project adapter enforces the exact
  global-causal contract. The historical two-`(512,256)` M128 x N32
  composition remains the environment rollback. No MLA or FA3 route is used.
- Global backward: EXP-0005 preserves the rejection of direct autograd through
  each unequal-dimension GQA-8 slab and the over-budget monolithic head-expanded
  diagnostic. EXP-0006 instead accepts three M64 x N32 ownership variants per
  V256 slab: one dKV-only main launch and two dQ-only main launches for D256
  offsets 0 and 256. Persistent FP32 accumulators sum dQ and dK across both slabs
  before one BF16 conversion; dV slabs remain separate and are concatenated.
  The six-main-launch path returns distinct dQ/dK/dV without changing model
  geometry. EXP-0013 threads THD tensors, INT32 `cu_seqlens_q/k`, and exact host
  maxima through the same three ownership variants. Packed FP32 workspaces
  retain per-segment coordinates without creating one fixed call per segment.
  The exact EXP-0012 composer remains available only when native HBM admission
  raises `GlobalBackwardBudgetExceeded` and every active-query K segment is at
  most 2048.
  EXP-0037 replaces the two dKV slabs with one full-D dKV kernel. EXP-0038
  replaces the two streamed dQ variants with one full-D score/dP/dS launch
  whose low/high D256 outputs run sequentially through one epilogue arena.
  The current default therefore has two main backward launches while retaining
  the earlier routes behind independent rollback flags.
  EXP-0040 removes the whole-sequence FP32 dK/dV workspace by assigning final
  dKV ownership to one CTA per KV tile; the remaining FP32 bulk workspace is dQ.
  EXP-0014 extends only native packed admission through K262144; a K>2048
  budget rejection propagates before forward. Validation, contract, assertion,
  and backend runtime failures also propagate. EXP-0015 admits mixed plateaus
  without changing this ownership: zero-Q segments schedule no main work, and
  their K/V slices receive exact-zero gradients.
- Fallback: repository PyTorch reference for correctness only, never reported
  as a kernel-performance equivalent.
- Forward compute atom: each launch consumes full d512 Q/K through repeated
  `HGMMA.64x32x16.F32.BF16` score instructions and produces one d256 output
  slab through `HGMMA.64x256x16.F32.BF16`. Both launches compute identical
  scores/softmax and must return bitwise-identical FP32 LSE.
- Backward compute: the dKV-only variant owns all eight query-head
  contributions to one KV head through explicit FP32 atomic reduction. Each
  dQ-only variant owns one D256 dQ output slice, using the matching K slice
  after full-d512 score recomputation, and omits dK/dV state.
- Guards: the pinned patch opens only the reviewed SM90 global specializations;
  the project adapter requires fixed B1/S<=2048 or packed mixed
  `B>=1`/per-segment `0<=Sq<=Sk<=262144` with positive aggregate Q/K totals and
  positive exact maxima, full d512, 32Q/4KV, scale 1.0, lower-right causal text
  inputs, legal aligned BF16 storage, distinct K/V, signed-INT32 cumulative and
  padded totals, and an HBM preflight before forward admission and backward
  scratch allocation. The composer independently requires every active-query K
  segment to be at most 2048; an empty-query segment schedules no composed
  call.

## 5. Tile and ownership hierarchy

- Modes: B, Hq/Hkv, M=query, N=key, D/Dv.
- Local baseline: upstream-selected SM90 tiles; record the realized tile and
  resources from compile output rather than restating an assumption.
- Global forward tile: M128 x N32, two stages, invoked once per V256 slab.
- Global backward tiles: M64 x N32, Q/dO/PdS stages 1/1/1. One compile-time
  variant owns dK/dV; two variants own dQ D256 offsets 0 and 256. Each variant
  uses two MMA warpgroups plus the producer group (384 threads).
- Cluster: one CTA, no multicast, no persistent scheduling initially.
- Warpgroup roles: unchanged pinned SM90 producer/consumer roles, with two MMA
  warpgroups per launch. The two V slabs are separate sequential launches,
  not concurrent owners sharing probability state inside one CTA.
- Instruction tiles: only SM90 WGMMA shapes already used by the d256 exemplar;
  exact emitted instructions must be verified in SASS.
- Grid: forward schedules one logical (M/query tile, Q head, batch) work item
  per CTA. Backward schedules (N/key tile, Q head, batch) work items and reduces
  dQ across N tiles.

The forward ownership is:

| Tensor | Logical modes | Shape/stride | Static/dynamic | Memspace | Owner | Consumer |
|---|---|---|---|---|---|---|
| Q | B,M,Hq,D | fixed BSHD or packed THD, D-contiguous | D static | GMEM -> SMEM | producer | QK consumers |
| K | B,N,Hkv,D | fixed BSHD or packed THD, D-contiguous | D static | GMEM -> SMEM | producer | QK consumers |
| V | B,N,Hkv,Dv | fixed BSHD or packed THD, Dv-contiguous | Dv static | GMEM -> SMEM | producer | PV owners |
| S/P | M,N | M128 x N32 per launch | tile static | RMEM | score/softmax | current PV launch |
| O | B,M,Hq,Dv | fixed BSHD or packed THD, Dv256 per launch | Dv static | RMEM -> GMEM | current launch | concatenating adapter |
| LSE | B,Hq,M | fixed BHM or packed H,T FP32 | M/T dynamic | RMEM -> GMEM | softmax owner | backward/caller |

## 6. Data movement

| Edge | Primitive | Layout | Alignment | Tail | Completion |
|---|---|---|---|---|---|
| GMEM -> SMEM | SM90 TMA | upstream swizzled row-major | descriptor-checked | zero-fill/predicate | producer mbarrier |
| SMEM -> WGMMA | descriptor operands | upstream d256 partition | atom-required | D slabs exact; M/N predicated | consumer wait |
| WGMMA -> accumulator | async WGMMA | FP32 S or O fragment | instruction-defined | none in D | WGMMA wait |
| accumulator -> epilogue | upstream conversion | Dv-slab ownership | vector-aligned | M predicate | CTA synchronization |
| epilogue -> GMEM | TMA store or upstream vector store | BSHD | descriptor/vector | M predicate | store wait before reuse/exit |

- SMEM layout/swizzle: reuse the pinned SM90 helpers; no invented swizzle.
- Storage estimate per asymmetric launch: Q128 x D512 is 128 KiB, two K
  stages at N32 x D512 total 64 KiB, and two V stages at N32 x D256 total
  32 KiB: 224 KiB core. `cuobjdump` reports an additional 1 KiB static shared
  section. The launch passed, but exact dynamic shared memory remains open
  because the pod denies Nsight Compute performance-counter access.
- Local backward evidence is separate from the global-forward estimate. The
  realized M64 x N64 Q1/dO1/PdS1 main backward models 208 KiB core dynamic
  storage; `cuobjdump` reports 1 KiB static shared memory, 168 registers, zero
  local memory, and zero stack. Its inspected SASS contains 44 BF16-to-FP32
  HGMMA instructions and 24 `UTMALDG.4D` instructions.
- EXP-0006's fixed realized dKV-only main launch allocates 222,208 bytes of
  dynamic shared memory; each dQ-only launch allocates 218,112 bytes. The fixed
  main objects use 168 registers and 1 KiB static shared memory with zero stack
  and zero local memory. EXP-0013's native THD objects retain the same generated
  shared-storage configuration and 168 registers: dKV reports zero stack/local,
  while dQ-low/high report a 16-byte stack, zero local memory, seven `LDL`, and
  four `STL` instructions.
- TMA assumptions: D-contiguous base pointers, legal shape/stride, aligned
  descriptors, and predicated M/N tails.
- TMEM: none on SM90.

## 7. Pipeline state machine

Each asymmetric forward launch uses the unchanged pinned two-stage SM90
pipeline, producer role, WGMMA consumers, named barriers, and epilogue. The
backward variants use the separately recorded Q1/dO1/PdS1 pipeline and add no
cross-variant in-kernel handoff.

The outer composition is deliberately simple:

1. materialize contiguous `V[..., :256]` and `V[..., 256:]` slabs;
2. launch the exact patched `(Dqk,Dv)=(512,256)` kernel for V0;
3. launch the same specialization for V1 on the current CUDA stream;
4. require bitwise-identical FP32 LSE from both launches;
5. concatenate O0 and O1 along Dv.

Because both calls receive the same Q/K, scale, causal mask, and tile config,
they compute the same `P`. Therefore `concat(PV0, PV1) = P concat(V0,V1)`.
This duplicates QK and softmax work but introduces no cross-CTA probability
sharing or new barrier ownership.

The accepted backward composition is also explicit:

1. preprocess the shared forward O/LSE state and each V256 dO slab;
2. for V0, launch dKV-only plus dQ offsets 0 and 256;
3. for V1, launch the same three main variants;
4. retain the cross-slab dQ and dK sums in common FP32 output accumulators;
5. postprocess/cast dQ and dK once, cast each dV slab, then concatenate dV.

For native packed execution, the same sequence is launched once over THD
totals. Each main variant receives the Q/K cumulative arrays and host maxima;
the pinned varlen coordinate tensors establish lower-right positions and
segment isolation. The packed postprocess layouts cast separate dQ/dK/dV only
after their FP32 ownership reductions. Runtime cumulative values and totals do
not enter the compile key.

Compile-time ownership flags remove disabled accumulator and epilogue state.
The dKV variant explicitly atomically reduces GQA-8 contributions. Sanitizer
evidence at S128 and the S129 partial tile found no memory, synchronization, or
race errors, but atomic ordering still permits numerically valid non-bitwise
gradient repeats.

## 8. Predication and neutral values

- Coordinates: explicit absolute q/k coordinates plus batch/head and optional
  vision-block IDs.
- M/N bounds: upstream coordinate tensors; D slabs are exact 256-wide.
- Invalid Q/K/V loads: zero; invalid score positions: negative infinity.
- Invalid stores: suppressed.
- Predicates: derive once per tile and reuse; no pointer-only bounds logic.
- Multimodal dispatch: encode the complete local predicate in `mask_mod`, set
  `causal=False`, pass no built-in window, and supply matching forward and
  transposed-backward block-sparse metadata. Upstream built-in local mode and
  `mask_mod` are not composed.
- Vector tails: unsupported in D initially; scalar/predicated M/N tail follows
  the exemplar.
- All-masked rows: rejected by the supported input contract.

## 9. Resource budget

- Threads: 384 per asymmetric launch: one producer group plus two upstream
  MMA warpgroups.
- CTAs/cluster: 1.
- SMEM: global forward models 224 KiB dynamic core; local backward models
  208 KiB. Global dKV-only backward allocates 222,208 bytes dynamically and
  each dQ-only variant allocates 218,112 bytes. Each backward cubin reports
  1 KiB static shared storage.
- Registers: `cuobjdump` reports 168 registers and zero local memory for the
  inspected forward, local-backward, and global backward mains. Fixed dKV/dQ
  and native dKV report zero stack; native dQ-low/high report a 16-byte stack
  with seven `LDL` and four `STL` instructions.
- TMEM: none.
- Residency: one CTA/SM is acceptable for the correctness prototype.
- Candidate filter: no spills, legal WGMMA/TMA, sanitizer-clean barriers, and
  complete O/LSE agreement before any deeper staging or larger N tile.

## 10. Correctness plan

- Reference: `src/gemma4_fa4/reference.py` plus the locked model spec.
- Policy: freeze each experiment's oracle before execution and never loosen it
  after a failure. EXP-0003's `atol=0.125, rtol=0.05` elementwise envelope
  remains rejected. EXP-0004 separately predeclares the pinned-upstream BF16
  policy for each gradient: `candidate_max <= 2 * independent_bf16_max +
  quantization_atol`, where `quantization_atol = 2 * max_abs((g_ref + 0.3 -
  0.3) - g_ref)`. Record candidate and independent-baseline max/mean errors
  for dQ, dK, and dV; neither policy may be rewritten retroactively.
- Follow-up forward hardening: one-hot/ramp coordinate tensors must expose
  head, D-slab, and tile swaps; zeros, repeated/large logits, and adversarial
  BF16 values remain required before integration. The initial M1 smoke
  acceptance used seeded random distinct K/V only and does not claim this
  broader matrix.
- Boundaries: window and tile edges listed in section 3, partial M/N, GQA
  ratios 1/2/4/8, and d512 specifically.
- Global backward matrix: exact B1/BF16/32Q/4KV/GQA-8/d512/causal/scale-1.0
  at S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`, including
  structured dO-slab superposition and isolated-Q-head ownership checks.
- Native global matrix: nonempty packed THD through per-segment K2048,
  including equal fixed/native parity, lower-right Q33/K1025, hostile mixed
  Q=[33,65]/K=[1025,2048], cross-segment and document isolation, dO-only,
  LSE-only, combined gradients, odd noncontiguous dO/dLSE, memory bounds,
  repeat/nondefault-stream, sanitizers, and bounded SS/SM/MM compile classes.
- EXP-0014 extends that native matrix through K262144 with K2049/K4097
  references and isolation, square S2049, nondefault-stream Q129/K4097,
  analytic finite-large and model-maximum cases, guarded long-square admission,
  focused sanitizers, and unchanged bounded scheduler/generated-object classes.
- EXP-0015 adds leading, middle, and trailing paired plateaus plus
  query-empty/key-nonempty segments beside active neighbors. Local native and
  exact metadata paths, global native/composer routing, eager fully padded-row
  restoration, exact-zero empty-slice dK/dV, focused sanitizers, and plateau
  cache replays pass without changing main-object bytes/resources.
- EXP-0016 adds eager B1 text-only StaticCache active-prefix prefill/decode with
  no active backward. Local boundary/rollover and global decode cases pass
  prepared-operand/reference, hostile-tail, stable-storage, sanitizer, and
  bounded-cache gates without changing a CuTe kernel.
- Concurrency: repeat on the default and a nondefault CUDA stream. Forward O
  and LSE must repeat exactly; bulk/atomic-reduced gradients must pass the frozen
  numerical rule on every run but are not required to be bitwise equal.
- Sanitizers: targeted memcheck, synccheck, then racecheck for every new
  protocol; exact pytest node IDs are added after the adapter exists.
- Expected failures: the unpatched upstream interface rejects `(512,256)`;
  strict environment validation rejects a missing, altered, or extra patch.
  Invalid dtype/layout/aliasing is rejected by the project adapter. Direct
  asymmetric GQA backward rejects unequal QK/V dimensions at the pinned SM90
  constructor (EXP-0005); the accepted EXP-0006 split does not revise that
  historical result.

## 11. Benchmark plan

- Baseline: only a semantically equivalent composite or the same verified
  kernel revision; plain causal SDPA is not a local multimodal baseline.
- Workloads: locked smoke ladder after all correctness gates pass.
- Warmup/timing: repaired benchmark API, CUDA events, separate fwd/bwd/
  fwd_bwd, hot/cold L2, and median/p25/p75/IQR.
- Synchronization: event pairs with one final host synchronization outside the
  measured region.
- Repetitions: repository defaults, recorded with clocks/power state.
- Search: none in M1 beyond correctness-preserving defaults.
- Key: complete compile key from section 3 plus runtime shape and environment
  in the result record.
- Acceptance: correctness and sanitizer gates first; performance acceptance is
  deferred to M2.

## 12. Assumptions and risks

- Verified: H100 capability 9.0; CUDA 12.8; pinned FA4 plus the one hash-locked
  patch; local d256 forward and scoped autograd backward; composed global d512
  forward and split backward through fixed S2048 and native packed K262144;
  fixed local multimodal and packed
  local native/custom paths through S1025; native packed local text through
  the locked S262144 maximum; fake compilation; numerical O/LSE and separate
  finite dQ/dK/dV; repeat/nondefault stream; memcheck, synccheck, and racecheck
  at the recorded specializations; SASS HGMMA/TMA paths; 168 registers and
  zero separately reported local memory. Packed custom forward has a 104-byte
  stack frame; its backward has zero stack. Global backward configures 222,208
  bytes of shared storage for dKV and 218,112 bytes for dQ; native dQ mains
  additionally report a 16-byte stack. EXP-0016 also verifies eager B1
  text-only StaticCache active prefixes with no active backward.
- Unverified: exact global-forward dynamic shared-memory launch metrics;
  deterministic local and long-context local dQ gradients; over-budget
  sparse schedules; raw/full-model `torch.compile`, compiled prefill, cached
  multimodal decode, other-layer/varlen-facade integration, and performance.
  EXP-0017 through EXP-0022 reject
  successive no-cache framework candidates while retaining cache/origin
  provenance, whole-layer opaque arithmetic, and snapshot-free inference-only
  weight transport. Runtime tensor transport and public comptime guards did
  not remove PyTorch 2.8's scalar-source restart. EXP-0023 accepts a separately
  named guarded facade for pinned layers 0/5, B1 BF16 no-cache text inference
  through S1024, with all live state validated outside Dynamo, exact per-call
  mutation rejection, frozen S1/S>1 graph bounds, bitwise eager equality,
  sanitizers, and unchanged codegen. Raw `torch.compile(layer)` remains
  unsupported. EXP-0026 separately accepts only pinned global layer-5
  compiled StaticCache one-token decode through K1025 after eager prefill;
  EXP-0027 rejects a conservative local counter ABI and EXP-0028 accepts only
  pinned local layer-0 compiled `StaticSlidingWindowLayer` one-token decode
  through underfill, boundary, and repeated rollover. All-empty physical
  packed workloads remain intentionally rejected rather than claimed as
  executable attention.
- EXP-0010 verifies exact production-length vision/document metadata within
  its declared padded-work, metadata, and free-HBM envelope using Q128/K80
  forward and independently generated/transposed Q64/K64 backward schedules.
- Version-sensitive helpers: TMA descriptor construction, SM90 WGMMA layout
  helpers, mbarriers, JIT cache keys, and mask-mod auxiliary tensors.
- Primary correctness risk: drift between the two otherwise-identical slab
  forward launches; the runtime exact-LSE check and locked patch/config guard
  it. Backward FP32 bulk/atomic reductions introduce non-bitwise order, so each run
  is checked against the frozen numerical policy.
- Primary performance risk: QK and softmax are executed twice, and V slabs are
  materialized contiguously. Backward executes six main launches and uses
  temporary FP32 accumulators. No speed claim is permitted for these paths.
- Future fused path: D-split PV ownership with shared probabilities requires a
  new role map, probability handoff/barriers, and slab-aware epilogue. Treat it
  as separate M2 work. Never route normal attention to MLA or model K and V as
  identical.
