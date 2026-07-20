# H100 global d512 dQ single-warpgroup design brief

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS exemplar: v4.6.0,
  `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- CUDA toolkit / driver: 12.8.93 / 580.126.09
- Python / framework: 3.12 / PyTorch 2.8.0+cu128
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs
- Official example: CUTLASS v4.6.0
  `examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py` for the
  Hopper TMA/WGMMA lifecycle; production starting point is pinned FA4
  `flash_attn/cute/flash_bwd_sm90.py` at `77aacb68...` plus the exact project
  patch.
- Verified APIs: the installed `FlashAttentionBackwardSm90` constructor,
  `cute.compile`, `PipelineTmaAsync.create`, `pipeline.NamedBarrier`,
  `setmaxregister_increase/decrease`, TMA copy atom construction, and kernel
  launch arguments were checked against the installed source. CUTLASS 4.6
  auto-SMEM and `cute.compile_to` additions are not adopted in this experiment.

## 2. Operation contract

- Operation: the dQ-only half of exact causal Gemma 4 global attention
  backward for one V=256 composition slab and one Q/K output-D=256 chunk.
- Formula: `P = softmax(Q K^T)` at scale 1.0;
  `dP_s = dO_s V_s^T`; `D_s = sum_j(O_s[j] dO_s[j])`;
  `dS_s = P * (dP_s - D_s)`; `dQ_chunk += dS_s K_chunk`.
  Summing the two slab results gives the exact full-V512 dQ.
- Inputs: BF16 Q `[B,Sq,32,512]`, K `[B,Sk,4,512]`, one BF16 V and dO
  slab with last dimension 256, FP32 LSE-log2 and slab dPsum. Packed THD is
  also supported through int32 cumulative lengths.
- Output: one FP32 dQ accumulator chunk of width 256, followed by the existing
  one-time BF16 postprocess conversion. dK and dV are not owned by this kernel.
- Accumulation: FP32 scores, statistics, and gradient accumulation.
- Mask: full attention is causal. No vision overlay or local window applies.
- NaN/Inf, empty/all-masked, aliasing, and determinism: unchanged from the
  accepted project adapter. Empty packed segments are allowed but total Q/K
  are nonzero; every valid query has at least its causal prefix; output buffers
  do not alias inputs; this first target does not promise bitwise determinism.

## 3. Shape and layout regime

- Ranks: fixed BSHD rank 4 for the short path; packed THD rank 3 for the long
  and batched path.
- Dynamic: batch, totals, cumulative values, and exact maxima within the three
  existing single/single, single/multi, and multi/multi scheduler classes.
- Static: SM90, BF16, 32 Q heads, 4 KV heads, Q/K d512, V slab d256, causal,
  scale 1.0, tile M64/N32, one Q/dO/PdS stage, 384 threads.
- Strides/alignment: contiguous project-adapter tensors; the existing
  `assume_tensor_aligned` and TMA descriptor checks remain authoritative.
- Representative confirmation: global S8K and S64K, B1. Correctness retains
  the accepted square, asymmetric, packed, and causal-boundary matrix.
- Rejected/fallback: non-SM90, non-BF16, wrong model geometry, fixed S>2048,
  invalid packed cumulative arrays, and guarded memory-budget violations.
- Specializations: the existing `dkv`, `dq_lo`, and `dq_hi` variants remain
  bounded. `dQ_single_wg` is an explicit codegen member of the compile key.

## 4. Target and kernel family

- Target: H100 SM90 only.
- Family: warp-specialized TMA + WGMMA FA4 backward; nonpersistent
  `SingleTileScheduler` / `SingleTileVarlenScheduler`.
- Fallback: the accepted two-warpgroup dQ specialization remains the rollback.
- Compute atom: the production FA4 SM90 WGMMA atoms are retained exactly.
  This experiment changes ownership only; it does not introduce a new atom,
  dtype, epilogue, or portable implementation.

## 5. Tile and ownership hierarchy

- Problem modes: query M, key N, head dimension K.
- CTA: M64 x N32, one KV head/batch/scheduler tile.
- Warp specialization: producer warp 0 performs TMA; producer warp 1 stores
  dQ accumulation; two 128-thread MMA warpgroups consume Q/K/V/dO.
- Candidate ownership: MMA WG0 alone issues the dQ WGMMA for the full M64 x
  D256 chunk; WG1 still computes QK, dP, and dS needed by its M/N partition.
- Grid/block: scheduler-derived grid; 384 threads; no cluster and no persistent
  work mapping.

| Tensor | Logical modes | Static/dynamic | Memory | Owner / consumer |
|---|---|---|---|---|
| Q | M,Dq,Hq,B | Dq512 static | GMEM/SMEM | TMA producer / both MMA WGs |
| K | N,Dq,Hkv,B | Dq512 static | GMEM/SMEM | TMA producer / both MMA WGs |
| V slab | N,Dv,Hkv,B | Dv256 static | GMEM/SMEM | TMA producer / both MMA WGs |
| dO slab | M,Dv,Hq,B | Dv256 static | GMEM/SMEM | TMA producer / both MMA WGs |
| LSE,dPsum | M,Hq,B | FP32 | GMEM/SMEM | TMA producer / both MMA WGs |
| dS | M,N | M64/N32 static | RMEM/SMEM | both WGs / candidate WG0 dQ |
| dQ accumulator | M,D256,Hq,B | D256 static | SMEM/GMEM FP32 | WG0 + store warp |

## 6. Data movement

No data-movement edge changes in EXP-0030.

| Edge | Primitive | Layout | Tail | Completion |
|---|---|---|---|---|
| GMEM to SMEM | existing TMA bulk tensor loads | accepted FA4 swizzles | existing predicates | `PipelineTmaAsync` transaction barrier |
| SMEM to descriptors/RMEM | existing partitioned copies/descriptors | WGMMA partitions | predicated | pipeline consumer wait |
| MMA to accumulator | SM90 WGMMA | existing tiled MMA | neutral masked scores | warpgroup wait |
| accumulator to epilogue | existing R2S dQ copy | M64 x D256 | predicated | existing named barrier/fence |
| epilogue to GMEM | existing bulk store/FP32 accumulation | accepted accumulator layout | predicated | existing store protocol |

SMEM layouts, spans, 1024-byte buffer alignment, TMA descriptor assumptions,
and the absence of TMEM are inherited unchanged from accepted generated code.

## 7. Pipeline state machine

There is one Q stage, one dO stage, and one PdS stage. Warp 0 produces TMA Q,
K, V, dO, LSE, and dPsum data. Both MMA warpgroups consume the same pipeline
states. The existing PdS named barrier has 256 MMA-thread participants. The
candidate uses the already-implemented `is_dQ_wg` branch: WG1 skips dQ issue
and dQ-accumulator ownership, but it does not skip any wait, fence, dS publish,
or end-of-iteration synchronization.

| Transition | Agent | State | Event | Match |
|---|---|---|---|---|
| acquire | producer warp | Q/dO `(index,phase)` | producer acquire | prior release |
| issue | producer warp | current stage | TMA load | transaction bytes |
| arrive | TMA engine | current stage | barrier completion | consumer wait |
| wait | both MMA WGs | current stage | consumer wait | TMA arrive |
| consume | both MMA WGs | current stage | QK, dOV, pointwise dS | producer issue |
| release | both MMA WGs | current stage | existing release/barrier | next acquire |

Prologue, steady state, drain, store completion, and scheduler reset are
unchanged. Therefore this experiment does not create a new pipeline/barrier
protocol or require a new participant-count proof.

## 8. Predication and neutral values

The accepted coordinate tensors and causal M/N predicates are unchanged.
Out-of-range loads retain their current fill and masked scores remain neutral
for softmax; invalid stores remain suppressed. Head dimensions are exact
multiples of the vector and WGMMA requirements, so only sequence tails need
predication. No new reduction or all-masked behavior is introduced.

## 9. Resource budget

- 384 threads, 12 warps, one CTA; no cluster.
- Baseline dQ generated code: about 218,112 bytes dynamic SMEM and 168
  registers/thread, one CTA/SM. Candidate SMEM is structurally unchanged.
- Candidate register allocation requests 256 registers for WG0, 224 for WG1,
  and 24 for producer warps, within the source's 504-register warpgroup budget.
- No TMEM. Candidate is rejected on main-kernel spills, SMEM growth, invalid
  occupancy, or any new codegen/sanitizer failure.

## 10. Correctness plan

- Reference: project PyTorch reference plus the accepted optional pinned HF
  oracle.
- Error policy: unchanged locked O/LSE/dQ/dK/dV tolerances; no tolerance edit.
- Tests: compile-key/routing CPU tests; H100 fixed and packed global backward
  cases including Sq/Sk tails, asymmetry, B>1, GQA8, and causal boundaries;
  repeated-run checks; full H100 verification bundle.
- Sanitizers: rerun memcheck, synccheck, and racecheck on the changed dQ
  specialization using the exact commands recorded by the H100 status flow.
- Generated code: retain compile metadata and inspect SASS/resource/spill
  differences for both dQ chunks.

## 11. Benchmark plan

- Baseline: EXP-0029 exact FA4 two-dQ-warpgroup specialization.
- Screening: global S8K bwd and fwd_bwd, hot L2, ten warmups and thirty CUDA
  event repetitions. Confirm a winner at global S64K and cold L2.
- Clocks: unlocked and labeled; record pre/post state. No clock mutation is
  authorized by this experiment.
- Metrics: median/IQR ms, exact FLOP/s, Nsight Systems dQ launch duration,
  registers, spills, SMEM, and sanitizer results.
- Search space: exactly `{dQ_single_wg=False, True}`.
- Cache key: explicit `dQ_single_wg` bool plus the existing bounded variant key.
- Accept only if global backward improves by at least 3%, timing IQRs do not
  overlap, correctness/sanitizers remain clean, and no main-kernel spill or
  resource regression appears. Otherwise restore `False` and record REJECT.

## 12. Assumptions and risks

- Verified: dQ is the measured dominant launch family; the knob and separate
  WG register budgets already exist in pinned FA4; no ABI/layout/mask change is
  required.
- Unverified: one warpgroup may or may not saturate dQ WGMMA issue throughput.
- Version-sensitive: CuTe pipeline, named-barrier, register-allocation, TMA,
  and compile APIs are pinned to the recorded environment.
- Correctness risk: a latent dependency on both WGs owning dQ could surface;
  sanitizers and full gradient comparison gate retention.
- Performance risk: halving dQ issuing warpgroups may simply double that GEMM's
  duration despite lower register pressure.
- Rollback: keep `dQ_single_wg=False`; then pursue the separately reviewed
  two-V-slab dQ fusion design suggested by EXP-0029.
