# H100 global d512 full-V dQ N16 design brief

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS exemplar: v4.6.0 at
  `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- CUDA toolkit / driver: 12.8.93 / 580.126.09
- Python / framework: 3.12 / PyTorch 2.8.0+cu128
- GPU: NVIDIA H100 80GB HBM3, SM90, 132 SMs
- Official example: CUTLASS v4.6.0 Hopper `fmha.py` for TMA/WGMMA
  construction and lifecycle. Production source remains pinned FA4
  `flash_bwd_sm90.py` at `77aacb68...` plus the exact project patch.
- Verified installed APIs: `FlashAttentionBackwardSm90`, `cute.compile`,
  `PipelineTmaAsync`, named barriers, SM90 WGMMA construction, TMA atoms, and
  launch configuration. No experimental CUTLASS 4.6 API is introduced.

## 2. Operation contract

- Operation: exact causal Gemma 4 global d512 attention dQ backward.
- Formula: `P = softmax(Q K^T)` with scale 1.0;
  `dP = dO_0 V_0^T + dO_1 V_1^T = dO V^T`;
  `D = sum_j(O_j dO_j) = D_0 + D_1`;
  `dS = P * (dP - D)`; `dQ = dS K`.
- Inputs: BF16 Q `[B,Sq,32,512]`, K/V `[B,Sk,4,512]`, dO
  `[B,Sq,32,512]`; packed THD equivalents; FP32 LSE-log2 and dPsum; int32
  cumulative lengths for packed calls.
- Outputs: two FP32 dQ accumulator chunks of width 256, each converted once to
  BF16 by the accepted postprocess. The accepted separate slab dKV kernels and
  separate final dK/dV remain unchanged.
- Accumulation: FP32 scores, statistics, and gradients.
- Mask: global causal only; no vision overlay and no local window.
- NaN/Inf, empty/all-masked, aliasing, and determinism policies are unchanged.
  Empty packed segments require positive aggregate totals; buffers do not
  alias inputs; nondeterministic FP32 reduction remains allowed and tested.

## 3. Shape and layout regime

- Fixed rank-4 BSHD through S2048 and native packed rank-3 THD through the
  guarded K262144 maximum.
- Dynamic batch/totals/cumulative values remain within the three existing
  scheduler classes. Static candidate: BF16, SM90, GQA8, M64, N16, Q/K/V512,
  one Q/dO/PdS stage, 384 threads, dQ output chunks D256.
- Contiguous tensors and the accepted 16-byte/TMA alignment assumptions remain.
- Representative performance: global S8K and S64K. Correctness includes fixed,
  packed, asymmetric Sq/Sk, tail, empty-segment, and causal-boundary cases.
- Rejection/fallback: any compile/resource failure, non-SM90 or non-BF16,
  wrong geometry, invalid cumulative lengths, or memory-budget rejection falls
  back to the accepted four-dQ-launch slab composition.
- Specialization count remains three main variants: one N32/Dv256 dKV plus two
  N16/Dv512 dQ output chunks. Tile N, V width, ownership flags, ABI class, and
  scheduler class are explicit in each compile-cache key.

## 4. Target and kernel family

- Target: H100 SM90 only.
- Family: nonpersistent, warp-specialized TMA + WGMMA FA4 backward.
- Portable fallback: the accepted N32 two-slab dQ composition.
- Compute atoms: pinned FA4 SM90 atoms. N16 divides across two MMA warpgroups as
  N8 per warpgroup, a supported WGMMA N granularity. Unsupported atoms/dtypes
  remain rejected by existing constructor and TMA alignment checks.

## 5. Tile and ownership hierarchy

- Problem modes: query M, key N, head D/K.
- dQ CTA: M64 x N16; full V/dO reduction width 512; output D256 per launch.
- Two 128-thread MMA warpgroups retain the accepted score/dP/dS/dQ ownership.
  Producer warp 0 performs TMA; producer warp 1 drains dQ accumulation.
- Scheduler grid doubles its N tiles relative to N32, while two dQ launches
  replace four. Total dQ CTA count stays constant, but QK and dQ arithmetic are
  each halved because the V slabs are combined before dS.
- No cluster or persistent scheduler.

| Tensor | Logical modes | Shape | Memory | Owner / consumer |
|---|---|---|---|---|
| Q | M,Dq,Hq,B | M64,D512 | GMEM/SMEM | TMA / both MMA WGs |
| K | N,Dq,Hkv,B | N16,D512 | GMEM/SMEM | TMA / both MMA WGs |
| V | N,Dv,Hkv,B | N16,D512 | GMEM/SMEM | TMA / both MMA WGs |
| dO | M,Dv,Hq,B | M64,D512 | GMEM/SMEM | TMA / both MMA WGs |
| LSE,dPsum | M,Hq,B | M64 FP32 | GMEM/SMEM | TMA / both MMA WGs |
| dS | M,N | M64,N16 BF16 | RMEM/SMEM | both WGs / dQ WGMMA |
| dQ accumulator | M,Dout,Hq,B | M64,D256 FP32 | SMEM/GMEM | MMA WGs / store warp |

## 6. Data movement

| Edge | Primitive | Layout | Tail | Completion |
|---|---|---|---|---|
| GMEM to SMEM | existing TMA bulk loads | FA4 SM90 swizzles | existing sequence predicates | existing transaction barrier |
| SMEM to WGMMA/RMEM | existing descriptors/partitioning | N8 per MMA WG | exact D, predicated M/N | existing consumer wait |
| WGMMA to accumulator | existing SM90 WGMMA | FP32 fragments | neutral masked scores | warpgroup wait |
| accumulator to SMEM | existing vector R2S | two D128 WG chunks | predicated stores | existing dQ barriers |
| SMEM to GMEM | existing bulk reduce-add | FP32 dQ accumulator | predicated | existing async completion |

No second shared slot or new copy edge is introduced. Full V and dO occupy the
existing single-stage buffers; N16 makes full V the same byte size as N32/V256.
SMEM swizzles, 1024-byte buffer alignment, descriptor assumptions, and the
absence of TMEM remain unchanged.

## 7. Pipeline state machine

The accepted one-stage Q, dO, and PdS pipelines are retained byte-for-byte.
Producer and consumer agents, named barriers, participant counts, `(index,
phase)` transitions, prologue, steady state, drain, dQ store completion, and
scheduler reset do not change. Only static tensor/tile extents passed through
the existing construction differ. Therefore EXP-0031 introduces no new
barrier, phase, handoff, reuse interval, or persistent-loop protocol.

| Transition | Agent | State | Event | Match |
|---|---|---|---|---|
| acquire | producer warp | existing Q/dO state | producer acquire | prior release |
| issue | producer warp | current stage | TMA Q/K/V/dO/stats | transaction bytes |
| arrive | TMA engine | current stage | transaction complete | consumer wait |
| wait | both MMA WGs | current stage | existing wait | TMA arrive |
| consume | both MMA WGs | current stage | QK, full dOV, dS, dQ | producer issue |
| release | both MMA WGs | current stage | existing release | next acquire |

## 8. Predication and neutral values

Existing causal coordinate tensors and M/N predicates remain. N16 changes only
the static tile extent. Invalid loads/stores, masked-score neutral values,
vector tails, and empty-segment behavior are unchanged. D512 and D256 are exact
multiples of every vector/WGMMA requirement.

## 9. Resource budget

- 384 threads / 12 warps / one CTA; no cluster.
- Measured baseline dQ: 218,112 bytes dynamic SMEM, 168 registers/thread,
  one CTA/SM.
- Predicted candidate delta from static extents: K N32->N16 saves 16,384
  bytes; dO D256->D512 adds 32,768 bytes; dS N32->N16 saves 2,048 bytes;
  V remains 16,384 bytes because N halves as D doubles. Predicted total is
  exactly 232,448 bytes, the H100 opt-in per-block ceiling. Compilation/launch
  must prove this; there is zero padding-growth margin.
- The dQ accumulator remains 65,536 bytes. No TMEM. Reject on SMEM overflow,
  spills, register regression that invalidates occupancy, or launch failure.

## 10. Correctness plan

- Reference: project PyTorch reference and accepted pinned HF oracle.
- Tolerances: locked O/LSE/dQ/dK/dV policy; no tolerance edit.
- First discriminator: fake compile plus real S128 fixed reference, three
  nondefault-stream repetitions. Then fixed S1024/1025/2048, packed mixed
  Sq/Sk/tails/empty rows, analytic long cases, isolation, and integration.
- Check separate dQ, dK, dV, O, and LSE; validate compile-key bounds and guarded
  memory accounting.
- Sanitizers: memcheck, synccheck, and racecheck on fixed S1025 plus packed
  mixed segments using the established commands.
- Generated code: retain SASS/resources/spills and confirm two N16 dQ main
  launches plus two unchanged N32 dKV launches.

## 11. Benchmark plan

- Baseline: EXP-0029 accepted two-slab/four-dQ-launch path.
- Screen global S8K bwd and fwd_bwd hot L2 with 10 warmups, 30 CUDA-event
  repetitions, median/p25/p75/IQR. Confirm a winner at S64K and cold L2.
- Unlocked clocks remain explicitly labeled and sampled; no clock mutation.
- Metrics: exact median/IQR/FLOP/s, Nsight Systems launch attribution,
  registers, spills, dynamic SMEM, sanitizer results.
- Search space: exactly `{N32/Dv256 slab dQ, N16/Dv512 full dQ}`.
- Cache key: explicit variant, N tile, V width, dQ width/offset, scheduler/ABI
  class, broadcast class, dtype, mask, stages, and architecture.
- Accept if global S8K backward improves by at least 10% with non-overlapping
  IQRs, S64K confirms the direction, and all correctness/resource/sanitizer
  gates pass. Otherwise restore EXP-0029.

## 12. Assumptions and risks

- Verified: dQ dominates measured backward; full dP and dPsum algebra is exact;
  N16 is the smallest supported WGMMA N granularity; the existing pipeline can
  express Dv512 when shared memory fits.
- Unverified: exact 232,448-byte generated SMEM fits the installed runtime and
  does not gain hidden padding; N8-per-WG WGMMA utilization is sufficient.
- Version-sensitive: FA4 layout construction, CuTe TMA/pipeline APIs, WGMMA N8,
  and auto launch SMEM.
- Correctness risk: FP32 association changes when slab dP/dPsum are combined
  before dS instead of accumulating slab dQ afterward.
- Performance risk: doubled N-grid scheduling and N8 WGMMA efficiency may
  offset the halved QK/dQ work.
- Rollback: restore the byte-locked EXP-0029 patch and its three accepted
  N32 variants.
