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
  `patches/flash-attention/0001-sm90-d512-v256-forward.patch`, SHA256
  `8d3404ccc8bdb2b3fd8e6de09e1f100827d071de283885bf737932bcb41aca4f`.

## 2. Operation contract

- Operation: Gemma 4 31B prepared-Q/K/V attention, BF16 forward and backward.
- Formula: `S[b,h,q,k] = sum_d Q[b,q,h,d] * K[b,k,h/g,d]`;
  `P = softmax(mask(S), dim=k)` because scale is exactly `1.0`;
  `O[b,q,h,v] = sum_k P[b,h,q,k] * V[b,k,h/g,v]`.
- Inputs: contiguous BSHD BF16 Q, K, and V. K and V are always distinct
  operands even when they originate from one projection source.
- Outputs: BF16 O, FP32 LSE; BF16 dQ/dK/dV. Forward and backward WGMMA
  accumulators are FP32. In the pinned backward, P is rounded to BF16 before
  dV and dS is rounded to BF16 before dQ/dK; validation must model those
  intentional algorithmic boundaries rather than claim one final conversion.
- Local mask: `k > q - 1024 AND (k <= q OR same nonnegative vision block)`.
- Global mask: `k <= q`.
- NaN/Inf: match the reference; no silent sanitization. Fully masked rows are
  rejected by the supported contract/test generator.
- Empty inputs: reject initially; add only with an explicit reference test.
- Aliasing: Q/K/V/O and dQ/dK/dV must not alias. No in-place operation.
- Determinism: correctness mode uses fixed seeds and repeated-run checks;
  deterministic backward ownership is required before performance tuning.

## 3. Shape and layout regime

- Rank: fixed-length BSHD rank 4 first; packed varlen is deferred until fixed
  length passes.
- Dynamic: B, sequence lengths, and vision IDs.
- Static compile configuration: architecture, dtype, QK/V head dimensions,
  GQA ratio, mask family, tile sizes, pipeline stages, and V-slab width.
- Local: 32 Q / 16 KV heads, d256, GQA 2, W1024.
- Global: 32 Q / 4 KV heads, d512, GQA 8.
- Strides: contiguous last dimension and contiguous BSHD baseline only.
- Alignment: the adapter requires 16-byte-aligned BF16 base pointers; D
  extents satisfy the upstream 8-element alignment used by TMA descriptors.
- Accepted adapter envelope: B1 only; local `1 <= S <= 1025`, global
  `1 <= S <= 1024`. Longer production gates are deferred until global
  backward and multimodal correctness exist and are rejected by the M1
  adapter.
- Adversarial shapes: q positions 0, 1023, 1024, 1025; partial M/N tiles;
  unequal q/k lengths; GQA ratios 1,2,4,8; vision spans crossing tile and
  window boundaries; distinct random K and V.
- Rejected initially: non-BF16, noncontiguous D, dropout, d not in {256,512},
  cross-layer KV reuse, and normal-attention K/V aliasing.
- Cache key: FA revision, DSL version, SM90a, dtype, D, Dv, GQA ratio, mask
  family, tile M/N, D-slab width/count, stage count, pack-GQA choice, and every
  constexpr/codegen flag. Runtime lengths and vision IDs are excluded.

## 4. Target and kernel family

- Target: SM90a H100.
- Local family: pinned FA4 `FlashAttentionForwardSm90` through a
  model-contract adapter. Autograd uses the pinned M64 x N64
  `FlashAttentionBackwardSm90` path with Q/dO/PdS stages 1/1/1, two MMA
  warpgroups, unpacked GQA-2, and FP32 dK/dV workspaces. Upstream tests still
  exclude SM90 backward above d192; EXP-0004 supplies project-scoped
  correctness and sanitizer evidence for exact B1/32Q/16KV/d256/W1024 text
  attention through S1025.
- Global family: a correctness-first composition of two pinned SM90
  `(Dqk,Dv)=(512,256)` launches. A hash-locked interface patch enables that
  asymmetric dimension/tile specialization and selects M128 x N32. The patch
  does not by itself restrict attention mode; the project adapter enforces the
  exact global-causal contract. No MLA or FA3 route is used.
- Fallback: repository PyTorch reference for correctness only, never reported
  as a kernel-performance equivalent.
- Compute atom: each launch consumes full d512 Q/K through repeated
  `HGMMA.64x32x16.F32.BF16` score instructions and produces one d256 output
  slab through `HGMMA.64x256x16.F32.BF16`. Both launches compute identical
  scores/softmax and must return bitwise-identical FP32 LSE.
- Guards: the pinned patch opens only exact SM90 `(512,256)` dimensions; the
  project adapter requires full d512, 32Q/4KV, scale 1.0, causal text inputs,
  and distinct K/V storage.

## 5. Tile and ownership hierarchy

- Modes: B, Hq/Hkv, M=query, N=key, D/Dv.
- Local baseline: upstream-selected SM90 tiles; record the realized tile and
  resources from compile output rather than restating an assumption.
- Global accepted tile: M128 x N32, two stages, invoked once per V256 slab.
- Cluster: one CTA, no multicast, no persistent scheduling initially.
- Warpgroup roles: unchanged pinned SM90 producer/consumer roles, with two MMA
  warpgroups per launch. The two V slabs are separate sequential launches,
  not concurrent owners sharing probability state inside one CTA.
- Instruction tiles: only SM90 WGMMA shapes already used by the d256 exemplar;
  exact emitted instructions must be verified in SASS.
- Grid: one logical (M tile, Q head, batch) work item per CTA initially.

| Tensor | Logical modes | Shape/stride | Static/dynamic | Memspace | Owner | Consumer |
|---|---|---|---|---|---|---|
| Q | B,M,Hq,D | BSHD, D-contiguous | D static | GMEM -> SMEM | producer | QK consumers |
| K | B,N,Hkv,D | BSHD, D-contiguous | D static | GMEM -> SMEM | producer | QK consumers |
| V | B,N,Hkv,Dv | BSHD, Dv-contiguous | Dv static | GMEM -> SMEM | producer | PV owners |
| S/P | M,N | M128 x N32 per launch | tile static | RMEM | score/softmax | current PV launch |
| O | B,M,Hq,Dv | BSHD, Dv256 per launch | Dv static | RMEM -> GMEM | current launch | concatenating adapter |
| LSE | B,Hq,M | FP32 | M dynamic | RMEM -> GMEM | softmax owner | backward/caller |

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
- TMA assumptions: D-contiguous base pointers, legal shape/stride, aligned
  descriptors, and predicated M/N tails.
- TMEM: none on SM90.

## 7. Pipeline state machine

Each asymmetric launch uses the unchanged pinned two-stage SM90 pipeline,
producer role, WGMMA consumers, named barriers, and epilogue. The project adds
no in-kernel handoff or barrier protocol.

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
  208 KiB. Each inspected cubin reports 1 KiB static storage; exact launch
  dynamic-SMEM metrics remain unresolved.
- Registers: `cuobjdump` reports 168 registers, zero local memory, and zero
  stack for both inspected main kernels. No spill storage is present.
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
- Concurrency: repeat on the default and a nondefault CUDA stream.
- Sanitizers: targeted memcheck, synccheck, then racecheck for every new
  protocol; exact pytest node IDs are added after the adapter exists.
- Expected failures: the unpatched upstream interface rejects `(512,256)`;
  strict environment validation rejects a missing, altered, or extra patch.
  Invalid dtype/layout/aliasing is rejected by the project adapter.

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
  forward through S1024; fake compilation; numerical O/LSE and separate
  finite dQ/dK/dV; repeat/nondefault stream; memcheck, synccheck, and racecheck
  at the recorded specializations; SASS HGMMA/TMA paths; 168 registers and no
  local/stack spill storage.
- Unverified: exact dynamic shared-memory launch metrics; global d512
  backward; multimodal mask-mod forward/backward; long production lengths;
  performance.
- Version-sensitive helpers: TMA descriptor construction, SM90 WGMMA layout
  helpers, mbarriers, JIT cache keys, and mask-mod auxiliary tensors.
- Primary correctness risk: drift between the two otherwise-identical slab
  launches; the runtime exact-LSE check and locked patch/config guard it.
- Primary performance risk: QK and softmax are executed twice, and V slabs are
  materialized contiguously. No speed claim is permitted for this path.
- Future fused path: D-split PV ownership with shared probabilities requires a
  new role map, probability handoff/barriers, and slab-aware epilogue. Treat it
  as separate M2 work. Never route normal attention to MLA or model K and V as
  identical.
